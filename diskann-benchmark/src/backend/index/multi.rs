/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use diskann::{
    graph::{self, glue, DiskANNIndex},
    provider, ANNResult,
};
use diskann_benchmark_core::{self as benchmark_core, build::ids::ToId};
use diskann_benchmark_runner::{
    self as runner,
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    registry::Benchmarks,
    utils::{percentiles, MicroSeconds},
    Any,
};
use diskann_providers::model::graph::provider::async_::{common as inmem_common, inmem};
use diskann_quantization::multi_vector::{Mat, MatRef, Standard};
use diskann_utils::future::AsyncFriendly;

use crate::{
    backend::index::{result::SearchResults, search::knn},
    inputs,
    utils::datafiles,
};

/// The distance metric used for multi-vector search (inner product for mean-pooled vectors).
#[cfg(feature = "spherical-quantization")]
const METRIC: diskann_vector::distance::Metric = diskann_vector::distance::Metric::InnerProduct;

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    benchmarks.register::<MultiVectorBuild<'static>>(
        "multi-vector-build",
        |job, _checkpoint, output| {
            job.run(output)?;
            Ok(serde_json::to_value(())?)
        },
    );

    spherical_rerank::register_benchmarks(benchmarks);
}

//////////////
// Dispatch //
//////////////

/// The dispatcher target for multi-vector build operations.
pub(super) struct MultiVectorBuild<'a> {
    input: &'a inputs::multi::BuildAndSearch,
}

impl<'a> MultiVectorBuild<'a> {
    fn new(input: &'a inputs::multi::BuildAndSearch) -> Self {
        Self { input }
    }
}

impl dispatcher::Map for MultiVectorBuild<'static> {
    type Type<'a> = MultiVectorBuild<'a>;
}

/// Dispatch from the concrete input type.
impl<'a> DispatchRule<&'a inputs::multi::BuildAndSearch> for MultiVectorBuild<'a> {
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a inputs::multi::BuildAndSearch) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn convert(from: &'a inputs::multi::BuildAndSearch) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        _from: Option<&&'a inputs::multi::BuildAndSearch>,
    ) -> std::fmt::Result {
        writeln!(f, "tag: \"{}\"", inputs::multi::BuildAndSearch::tag())
    }
}

/// Central dispatch mapping.
impl<'a> DispatchRule<&'a Any> for MultiVectorBuild<'a> {
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<inputs::multi::BuildAndSearch, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<inputs::multi::BuildAndSearch, Self>()
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<inputs::multi::BuildAndSearch, Self>(
            f,
            from,
            inputs::multi::BuildAndSearch::tag(),
        )
    }
}

impl<'a> MultiVectorBuild<'a> {
    fn run(self, mut output: &mut dyn runner::Output) -> anyhow::Result<()> {
        writeln!(output, "{}", self.input)?;
        run(&self.input, output)
    }
}

////////////////
// End-to-end //
////////////////

fn run(
    input: &inputs::multi::BuildAndSearch,
    mut output: &mut dyn runner::Output,
) -> anyhow::Result<()> {
    let index = build(&input.build, output)?;
    let queries: Arc<[_]> = datafiles::load_multi_vectors::<f32>(&input.search.queries)?.into();

    // TODO: Placeholder.
    let gt = datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth))?;

    let searcher = MultiKNN::new(
        index.clone(),
        queries,
        benchmark_core::search::graph::Strategy::broadcast(inmem::multi::Strategy::new()),
    )?;

    let steps = knn::SearchSteps::new(
        input.search.reps,
        &input.search.num_threads,
        &input.search.runs,
    );

    let summaries = knn::run(&searcher, &gt, steps)?;

    let results: Vec<_> = summaries
        .into_iter()
        .map(|r| SearchResults::from(r))
        .collect();
    writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;

    Ok(())
}

///////////
// Build //
///////////

fn build(
    input: &inputs::multi::Build,
    output: &mut dyn runner::Output,
) -> anyhow::Result<Arc<DiskANNIndex<inmem::multi::Provider<f32>>>> {
    let data = datafiles::load_multi_vectors::<f32>(&input.data)?;

    let dim = data.first().unwrap().vector_dim();

    let provider = inmem::DefaultProvider::<_, _, _, provider::DefaultContext>::new_empty(
        input.inmem_parameters(data.len(), dim),
        inmem::multi::Precursor::<f32>::new(dim),
        inmem_common::NoStore,
        inmem_common::NoDeletes,
    )
    .unwrap();

    let index = Arc::new(graph::DiskANNIndex::new(
        input.as_config().build()?,
        provider,
        None,
    ));

    let builder = Insert::new(
        index.clone(),
        data.into(),
        inmem::multi::Strategy::new(),
        benchmark_core::build::ids::Identity::new(),
    );

    let rt = benchmark_core::tokio::runtime(input.num_threads.get())?;
    let _ = benchmark_core::build::build_tracked(
        builder,
        benchmark_core::build::Parallelism::dynamic(diskann::utils::ONE, input.num_threads),
        &rt,
        Some(&super::build::ProgressMeter::new(output)),
    )?;

    Ok(index)
}

type MultiVec<T> = Mat<Standard<T>>;
type MultiVecRef<'a, T> = MatRef<'a, Standard<T>>;

#[derive(Debug)]
pub struct Insert<DP, T, S>
where
    DP: provider::DataProvider,
    T: Copy,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    data: Arc<[MultiVec<T>]>,
    strategy: S,
    to_id: Box<dyn ToId<DP::ExternalId>>,
}

impl<DP, T, S> Insert<DP, T, S>
where
    DP: provider::DataProvider,
    T: Copy,
{
    pub fn new<I>(
        index: Arc<graph::DiskANNIndex<DP>>,
        data: Arc<[MultiVec<T>]>,
        strategy: S,
        to_id: I,
    ) -> Arc<Self>
    where
        I: ToId<DP::ExternalId>,
    {
        Arc::new(Self {
            index,
            data,
            strategy,
            to_id: Box::new(to_id),
        })
    }
}

impl<DP, T, S> benchmark_core::build::Build for Insert<DP, T, S>
where
    DP: provider::DataProvider<Context: Default> + for<'a> provider::SetElement<MultiVecRef<'a, T>>,
    S: for<'a> glue::InsertStrategy<DP, MultiVecRef<'a, T>> + Clone + AsyncFriendly,
    T: AsyncFriendly + Copy,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.len()
    }

    async fn build(&self, range: std::ops::Range<usize>) -> ANNResult<Self::Output> {
        for i in range {
            let context = DP::Context::default();
            self.index
                .insert(
                    self.strategy.clone(),
                    &context,
                    &self.to_id.to_id(i)?,
                    self.data[i].as_view(),
                )
                .await?;
        }
        Ok(())
    }
}

////////////
// Search //
////////////

/// A built-in helper for benchmarking K-nearest neighbors search on multi-vector indices.
///
/// This is analogous to [`diskann_benchmark_core::search::graph::KNN`] but handles
/// multi-vector queries where each query is a `MultiVec<T>` rather than a slice `[T]`.
#[derive(Debug)]
pub struct MultiKNN<DP, T, S>
where
    DP: provider::DataProvider,
    T: Copy,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<[MultiVec<T>]>,
    strategy: benchmark_core::search::graph::Strategy<S>,
}

impl<DP, T, S> MultiKNN<DP, T, S>
where
    DP: provider::DataProvider,
    T: Copy,
{
    /// Construct a new [`MultiKNN`] searcher.
    ///
    /// If `strategy` is one of the container variants of [`Strategy`], its length
    /// must match the number of queries. If this is the case, then the strategies
    /// will have a querywise correspondence with the query collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of queries.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<[MultiVec<T>]>,
        strategy: benchmark_core::search::graph::Strategy<S>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.len())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
        }))
    }
}

/// Additional metrics collected during [`MultiKNN`] search.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SearchMetrics {
    /// The number of distance comparisons performed during search.
    pub comparisons: u32,
    /// The number of candidates expanded during search.
    pub hops: u32,
    /// The time spent in the post-processing rerank stage.
    pub rerank_latency: MicroSeconds,
}

#[derive(Debug, Clone)]
struct TimedPostProcessor<P> {
    inner: P,
    elapsed_micros: Arc<AtomicU64>,
}

impl<P> TimedPostProcessor<P> {
    fn new(inner: P, elapsed_micros: Arc<AtomicU64>) -> Self {
        Self {
            inner,
            elapsed_micros,
        }
    }
}

impl<A, T, O, P> glue::SearchPostProcess<A, T, O> for TimedPostProcessor<P>
where
    A: provider::BuildQueryComputer<T>,
    O: Send,
    P: glue::SearchPostProcess<A, T, O> + Send + Sync,
{
    type Error = P::Error;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: T,
        computer: &<A as provider::BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<A::Id>> + Send,
        B: graph::SearchOutputBuffer<O> + Send + ?Sized,
    {
        let elapsed_micros = Arc::clone(&self.elapsed_micros);
        let future = self
            .inner
            .post_process(accessor, query, computer, candidates, output);

        async move {
            let start = std::time::Instant::now();
            let result = future.await;
            elapsed_micros.store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            result
        }
    }
}

impl<DP, T, S> benchmark_core::search::Search for MultiKNN<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: benchmark_core::search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<DP, MultiVecRef<'a, T>, DP::ExternalId>
        + Clone
        + AsyncFriendly,
    T: AsyncFriendly + Copy,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::Knn;
    type Output = SearchMetrics;

    fn num_queries(&self) -> usize {
        self.queries.len()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> benchmark_core::search::IdCount {
        benchmark_core::search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        kind: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let rerank_micros = Arc::new(AtomicU64::new(0));
        let strategy = self.strategy.get(index)?;
        let processor = TimedPostProcessor::new(
            strategy.default_post_processor(),
            Arc::clone(&rerank_micros),
        );
        let stats = self
            .index
            .search_with(
                *kind,
                strategy,
                processor,
                &context,
                self.queries[index].as_view(),
                buffer,
            )
            .await?;

        Ok(SearchMetrics {
            comparisons: stats.cmps,
            hops: stats.hops,
            rerank_latency: MicroSeconds::new(rerank_micros.load(Ordering::Relaxed)),
        })
    }
}

/// An aggregated summary of multiple [`MultiKNN`] search runs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    /// The [`benchmark_core::search::Setup`] used for the batch of runs.
    pub setup: benchmark_core::search::Setup,

    /// The [`graph::search::KNN`] used for the batch of runs.
    pub parameters: graph::search::Knn,

    /// The end-to-end latency for each repetition in the batch.
    pub end_to_end_latencies: Vec<MicroSeconds>,

    /// The average latency for individual queries.
    pub mean_latencies: Vec<f64>,

    /// The average latency spent before post-processing rerank for individual queries.
    pub mean_first_stage_latencies: Vec<f64>,

    /// The average latency spent in post-processing rerank for individual queries.
    pub mean_rerank_latencies: Vec<f64>,

    /// The 90th percentile latency for individual queries.
    pub p90_latencies: Vec<MicroSeconds>,

    /// The 99th percentile latency for individual queries.
    pub p99_latencies: Vec<MicroSeconds>,

    /// The recall metrics for search.
    pub recall: benchmark_core::recall::RecallMetrics,

    /// The average number of distance comparisons per query.
    pub mean_cmps: f64,

    /// The average number of neighbor hops per query.
    pub mean_hops: f64,
}

/// A [`benchmark_core::search::Aggregate`] for collecting the results of multiple
/// [`MultiKNN`] search runs.
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn benchmark_core::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
}

impl<'a, I> Aggregator<'a, I> {
    /// Construct a new [`Aggregator`] using `groundtruth` for recall computation.
    pub fn new(
        groundtruth: &'a dyn benchmark_core::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
    ) -> Self {
        Self {
            groundtruth,
            recall_k,
            recall_n,
        }
    }
}

impl<I> benchmark_core::search::Aggregate<graph::search::Knn, I, SearchMetrics>
    for Aggregator<'_, I>
where
    I: benchmark_core::recall::RecallCompatible,
{
    type Output = Summary;

    fn aggregate(
        &mut self,
        run: benchmark_core::search::Run<graph::search::Knn>,
        mut results: Vec<benchmark_core::search::SearchResults<I, SearchMetrics>>,
    ) -> anyhow::Result<Summary> {
        let recall = match results.first() {
            Some(first) => benchmark_core::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                self.recall_n,
                true,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut mean_first_stage_latencies = Vec::with_capacity(results.len());
        let mut mean_rerank_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            let (total_rerank_micros, count) =
                r.output().iter().fold((0u64, 0usize), |(micros, n), o| {
                    (micros + o.rerank_latency.as_micros(), n + 1)
                });
            let mean_rerank_latency = if count > 0 {
                total_rerank_micros as f64 / count as f64
            } else {
                0.0
            };

            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    mean_first_stage_latencies.push((mean - mean_rerank_latency).max(0.0));
                    mean_rerank_latencies.push(mean_rerank_latency);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    mean_first_stage_latencies.push(0.0);
                    mean_rerank_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
        });

        // Compute average comparisons and hops
        let (total_cmps, total_hops, count) = results
            .iter()
            .flat_map(|r| r.output().iter())
            .fold((0u64, 0u64, 0usize), |(cmps, hops, n), o| {
                (cmps + o.comparisons as u64, hops + o.hops as u64, n + 1)
            });

        let mean_cmps = if count > 0 {
            total_cmps as f64 / count as f64
        } else {
            0.0
        };
        let mean_hops = if count > 0 {
            total_hops as f64 / count as f64
        } else {
            0.0
        };

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            recall,
            mean_latencies,
            mean_first_stage_latencies,
            mean_rerank_latencies,
            p90_latencies,
            p99_latencies,
            mean_cmps,
            mean_hops,
        })
    }
}

/////////////////
// Knn Adapter //
/////////////////

impl<DP, T, S> knn::Knn<DP::ExternalId> for Arc<MultiKNN<DP, T, S>>
where
    DP: provider::DataProvider<ExternalId: benchmark_core::recall::RecallCompatible>,
    MultiKNN<DP, T, S>: benchmark_core::search::Search<
        Id = DP::ExternalId,
        Parameters = graph::search::Knn,
        Output = SearchMetrics,
    >,
    T: Copy,
{
    fn search_all(
        &self,
        parameters: Vec<benchmark_core::search::Run<graph::search::Knn>>,
        groundtruth: &dyn benchmark_core::recall::Rows<DP::ExternalId>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = benchmark_core::search::search_all(
            self.clone(),
            parameters.into_iter(),
            Aggregator::new(groundtruth, recall_k, recall_n),
        )?;

        Ok(results.into_iter().map(SearchResults::from).collect())
    }
}

impl From<Summary> for SearchResults {
    fn from(summary: Summary) -> Self {
        let Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            mean_first_stage_latencies,
            mean_rerank_latencies,
            p90_latencies,
            p99_latencies,
            recall,
            mean_cmps,
            mean_hops,
            ..
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| recall.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            search_n: parameters.k_value().get(),
            search_l: parameters.l_value().get(),
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            mean_first_stage_latencies: Some(mean_first_stage_latencies),
            mean_rerank_latencies: Some(mean_rerank_latencies),
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
        }
    }
}

///////////////////////////////////
// Spherical Rerank Benchmark    //
///////////////////////////////////

// Feature-gated module for configurable multi-vector reranking with spherical quantization.
crate::utils::stub_impl!(
    "spherical-quantization",
    inputs::multi::SphericalRerankBuildAndSearch
);

mod spherical_rerank {
    use super::*;

    const NAME: &str = "multi-vector-spherical-rerank";

    pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
        #[cfg(feature = "spherical-quantization")]
        benchmarks.register::<SphericalRerankBuild<'static>>(NAME, |job, _checkpoint, output| {
            job.run(output)?;
            Ok(serde_json::to_value(())?)
        });

        #[cfg(not(feature = "spherical-quantization"))]
        super::imp::register(NAME, benchmarks);
    }

    #[cfg(feature = "spherical-quantization")]
    pub(super) struct SphericalRerankBuild<'a> {
        input: &'a inputs::multi::SphericalRerankBuildAndSearch,
    }

    #[cfg(feature = "spherical-quantization")]
    impl<'a> SphericalRerankBuild<'a> {
        fn new(input: &'a inputs::multi::SphericalRerankBuildAndSearch) -> Self {
            Self { input }
        }
    }

    #[cfg(feature = "spherical-quantization")]
    impl dispatcher::Map for SphericalRerankBuild<'static> {
        type Type<'a> = SphericalRerankBuild<'a>;
    }

    #[cfg(feature = "spherical-quantization")]
    impl<'a> DispatchRule<&'a inputs::multi::SphericalRerankBuildAndSearch>
        for SphericalRerankBuild<'a>
    {
        type Error = std::convert::Infallible;

        fn try_match(
            _from: &&'a inputs::multi::SphericalRerankBuildAndSearch,
        ) -> Result<MatchScore, FailureScore> {
            Ok(MatchScore(0))
        }

        fn convert(
            from: &'a inputs::multi::SphericalRerankBuildAndSearch,
        ) -> Result<Self, Self::Error> {
            Ok(Self::new(from))
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            _from: Option<&&'a inputs::multi::SphericalRerankBuildAndSearch>,
        ) -> std::fmt::Result {
            writeln!(
                f,
                "tag: \"{}\"",
                inputs::multi::SphericalRerankBuildAndSearch::tag()
            )
        }
    }

    #[cfg(feature = "spherical-quantization")]
    impl<'a> DispatchRule<&'a Any> for SphericalRerankBuild<'a> {
        type Error = anyhow::Error;

        fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
            from.try_match::<inputs::multi::SphericalRerankBuildAndSearch, Self>()
        }

        fn convert(from: &'a Any) -> Result<Self, Self::Error> {
            from.convert::<inputs::multi::SphericalRerankBuildAndSearch, Self>()
        }

        fn description(
            f: &mut std::fmt::Formatter<'_>,
            from: Option<&&'a Any>,
        ) -> std::fmt::Result {
            Any::description::<inputs::multi::SphericalRerankBuildAndSearch, Self>(
                f,
                from,
                inputs::multi::SphericalRerankBuildAndSearch::tag(),
            )
        }
    }

    #[cfg(feature = "spherical-quantization")]
    impl<'a> SphericalRerankBuild<'a> {
        fn run(self, mut output: &mut dyn runner::Output) -> anyhow::Result<()> {
            writeln!(output, "{}", self.input)?;
            run_configurable(self.input, output)
        }
    }
}

#[cfg(feature = "spherical-quantization")]
fn run_configurable(
    input: &inputs::multi::SphericalRerankBuildAndSearch,
    mut output: &mut dyn runner::Output,
) -> anyhow::Result<()> {
    use diskann_quantization::spherical;
    use inputs::multi::{DistanceMethod, InnerLoopMethod, RerankMethod};
    use rand::SeedableRng;

    let spherical_cfg = input.spherical.as_ref();

    // Helper: train a quantizer and build SphericalChamferData from multi-vector data.
    let build_chamfer_data = |data: &[MultiVec<f32>],
                              mut output: &mut dyn runner::Output|
     -> anyhow::Result<Arc<inmem::multi::SphericalChamferData>> {
        let cfg = spherical_cfg
            .ok_or_else(|| anyhow::anyhow!("spherical chamfer requires `spherical` config"))?;

        let dim = data.first().unwrap().vector_dim();
        let mut training_rows: Vec<f32> = Vec::new();
        for m in data {
            let view = m.as_view();
            for row in view.rows() {
                training_rows.extend_from_slice(row);
            }
        }
        let total_sub_vecs = training_rows.len() / dim;
        let training_matrix =
            diskann_utils::views::Matrix::try_from(training_rows.into(), total_sub_vecs, dim)?;

        let metric: diskann_vector::distance::Metric = METRIC;
        let pre_scale = match cfg.pre_scale {
            Some(v) => v.try_into()?,
            None => diskann_quantization::spherical::PreScale::None,
        };

        let start = std::time::Instant::now();
        let quantizer = diskann_quantization::spherical::SphericalQuantizer::train(
            training_matrix.as_view(),
            (&cfg.transform_kind).into(),
            metric.try_into()?,
            pre_scale,
            &mut rand::rngs::StdRng::seed_from_u64(cfg.seed),
            diskann_quantization::alloc::GlobalAllocator,
        )?;
        let training_time: MicroSeconds = start.elapsed().into();
        writeln!(
            output,
            "Spherical quantizer training: {}s",
            training_time.as_seconds()
        )?;

        macro_rules! make_data {
            ($N:literal) => {{
                let sq_impl = spherical::iface::Impl::<$N>::new(quantizer)?;
                Arc::new(inmem::multi::SphericalChamferData::new(sq_impl, data))
            }};
        }

        let result = match cfg.num_bits.get() {
            1 => make_data!(1),
            2 => make_data!(2),
            4 => make_data!(4),
            n => anyhow::bail!("unsupported num_bits: {}", n),
        };
        Ok(result)
    };

    // --- Build the index ---
    let index = match input.build.method {
        DistanceMethod::MeanPool => build(&input.build, output)?,
        DistanceMethod::SphericalChamfer => {
            let query_layout = input.build.query_layout.ok_or_else(|| {
                anyhow::anyhow!("spherical_chamfer build requires `query_layout` in `build`")
            })?;

            let build_data: Vec<MultiVec<f32>> =
                datafiles::load_multi_vectors::<f32>(&input.build.data)?;

            writeln!(output, "\nBuilding build-phase SphericalChamferData...")?;
            let chamfer_data = build_chamfer_data(&build_data, output)?;

            // Create provider and index (same as `build()` but with custom strategy).
            let dim = build_data.first().unwrap().vector_dim();

            let provider = inmem::DefaultProvider::<_, _, _, provider::DefaultContext>::new_empty(
                input.build.inmem_parameters(build_data.len(), dim),
                inmem::multi::Precursor::<f32>::new(dim),
                inmem_common::NoStore,
                inmem_common::NoDeletes,
            )
            .unwrap();

            let index = Arc::new(graph::DiskANNIndex::new(
                input.build.as_config().build()?,
                provider,
                None,
            ));

            let strategy =
                inmem::multi::SphericalChamferSearch::new(chamfer_data, query_layout.into());

            let builder = Insert::new(
                index.clone(),
                build_data.into(),
                strategy,
                benchmark_core::build::ids::Identity::new(),
            );

            let rt = benchmark_core::tokio::runtime(input.build.num_threads.get())?;
            let _ = benchmark_core::build::build_tracked(
                builder,
                benchmark_core::build::Parallelism::dynamic(
                    diskann::utils::ONE,
                    input.build.num_threads,
                ),
                &rt,
                Some(&super::build::ProgressMeter::new(output)),
            )?;

            index
        }
    };

    let queries: Arc<[_]> = datafiles::load_multi_vectors::<f32>(&input.search.queries)?.into();
    let gt = datafiles::load_groundtruth(datafiles::BinFile(&input.search.groundtruth))?;

    let steps = knn::SearchSteps::new(
        input.search.reps,
        &input.search.num_threads,
        &input.search.runs,
    );

    // If all three stages use defaults, use the legacy `MultiKNN` path.
    if matches!(input.build.method, DistanceMethod::MeanPool)
        && matches!(input.inner_loop, InnerLoopMethod::MeanPool)
        && matches!(input.rerank, RerankMethod::Chamfer)
    {
        let searcher = MultiKNN::new(
            index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(inmem::multi::Strategy::new()),
        )?;
        let summaries = knn::run(&searcher, &gt, steps)?;
        let results: Vec<_> = summaries.into_iter().map(SearchResults::from).collect();
        writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;
        return Ok(());
    }

    // --- Build inner-loop config ---
    let inner_loop = match &input.inner_loop {
        InnerLoopMethod::MeanPool => InnerLoop::MeanPool,
        InnerLoopMethod::SphericalChamfer {
            data: inner_data_path,
            query_data,
            query_layout,
        } => {
            let inner_data: Vec<MultiVec<f32>> = match inner_data_path {
                Some(path) => datafiles::load_multi_vectors::<f32>(path)?,
                None => datafiles::load_multi_vectors::<f32>(&input.build.data)?,
            };
            let inner_queries: Option<Arc<[_]>> = query_data
                .as_ref()
                .map(|qd| datafiles::load_multi_vectors::<f32>(qd))
                .transpose()?
                .map(|v| v.into());

            writeln!(output, "\nBuilding inner-loop SphericalChamferData...")?;
            let chamfer_data = build_chamfer_data(&inner_data, output)?;

            InnerLoop::SphericalChamfer {
                data: chamfer_data,
                inner_queries,
                layout: (*query_layout).into(),
            }
        }
    };

    // --- Build reranker config and run ---
    match &input.rerank {
        RerankMethod::Chamfer => {
            // Chamfer rerank with non-default inner loop (MeanPool+Chamfer handled above).
            // SphericalChamferAccessor doesn't have Store access for in-index Chamfer,
            // so we fall back to sideloading the build data for reranking.
            let build_data: Arc<[_]> =
                datafiles::load_multi_vectors::<f32>(&input.build.data)?.into();
            let reranker = MultiReranker::SideloadedChamfer {
                data: build_data,
                rerank_queries: None,
            };
            let searcher = ConfigurableMultiKNN::new(index.clone(), queries, inner_loop, reranker)?;
            let summaries = knn::run(&searcher, &gt, steps)?;
            let results: Vec<_> = summaries.into_iter().map(SearchResults::from).collect();
            writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;
        }

        RerankMethod::SideloadedChamfer { data, query_data } => {
            let rerank_data: Arc<[_]> = datafiles::load_multi_vectors::<f32>(data)?.into();
            let rerank_queries: Option<Arc<[_]>> = query_data
                .as_ref()
                .map(|qd| datafiles::load_multi_vectors::<f32>(qd))
                .transpose()?
                .map(|v| v.into());

            let reranker = MultiReranker::SideloadedChamfer {
                data: rerank_data,
                rerank_queries,
            };
            let searcher = ConfigurableMultiKNN::new(index.clone(), queries, inner_loop, reranker)?;
            let summaries = knn::run(&searcher, &gt, steps)?;
            let results: Vec<_> = summaries.into_iter().map(SearchResults::from).collect();
            writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;
        }

        RerankMethod::SphericalChamfer {
            data: rerank_data_path,
            query_data,
            query_layouts,
        } => {
            let rerank_data: Vec<MultiVec<f32>> = match rerank_data_path {
                Some(path) => datafiles::load_multi_vectors::<f32>(path)?,
                None => datafiles::load_multi_vectors::<f32>(&input.build.data)?,
            };
            let rerank_queries: Option<Arc<[_]>> = query_data
                .as_ref()
                .map(|qd| datafiles::load_multi_vectors::<f32>(qd))
                .transpose()?
                .map(|v| v.into());

            writeln!(output, "\nBuilding rerank SphericalChamferData...")?;
            let chamfer_data = build_chamfer_data(&rerank_data, output)?;

            for &layout in query_layouts {
                let reranker = MultiReranker::QuantizedChamfer {
                    data: chamfer_data.clone(),
                    rerank_queries: rerank_queries.clone(),
                    layout: layout.into(),
                };
                // Clone inner_loop data references for each layout iteration.
                let il = match &inner_loop {
                    InnerLoop::MeanPool => InnerLoop::MeanPool,
                    InnerLoop::SphericalChamfer {
                        data,
                        inner_queries,
                        layout,
                    } => InnerLoop::SphericalChamfer {
                        data: data.clone(),
                        inner_queries: inner_queries.clone(),
                        layout: *layout,
                    },
                };
                let searcher =
                    ConfigurableMultiKNN::new(index.clone(), queries.clone(), il, reranker)?;

                writeln!(output, "\nRerank query layout: {}", layout)?;
                let summaries = knn::run(&searcher, &gt, steps)?;
                let results: Vec<_> = summaries.into_iter().map(SearchResults::from).collect();
                writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;
            }
        }

        RerankMethod::None => {
            let reranker = MultiReranker::PassThrough;
            let searcher = ConfigurableMultiKNN::new(index.clone(), queries, inner_loop, reranker)?;
            let summaries = knn::run(&searcher, &gt, steps)?;
            let results: Vec<_> = summaries.into_iter().map(SearchResults::from).collect();
            writeln!(output, "{}", crate::utils::DisplayWrapper(&*results))?;
        }
    }

    Ok(())
}

///////////////////////////////
// Configurable Multi-Vector //
///////////////////////////////

/// Inner-loop strategy for [`ConfigurableMultiKNN`].
#[cfg(feature = "spherical-quantization")]
enum InnerLoop {
    /// Mean-pooled inner product (default). Uses the index's built-in mean-pooled vectors.
    MeanPool,
    /// Spherical-quantized Chamfer distance over compressed sub-vectors.
    SphericalChamfer {
        data: Arc<inmem::multi::SphericalChamferData>,
        inner_queries: Option<Arc<[MultiVec<f32>]>>,
        layout: diskann_quantization::spherical::iface::QueryLayout,
    },
}

/// Reranking strategy for [`ConfigurableMultiKNN`].
///
/// Each variant captures the data and configuration needed for a specific reranking method.
#[cfg(feature = "spherical-quantization")]
enum MultiReranker {
    /// Spherical-quantized Chamfer distance using pre-compressed sub-vectors.
    QuantizedChamfer {
        data: Arc<inmem::multi::SphericalChamferData>,
        rerank_queries: Option<Arc<[MultiVec<f32>]>>,
        layout: diskann_quantization::spherical::iface::QueryLayout,
    },
    /// Full-precision Chamfer distance using a separate (side-loaded) multi-vector set.
    SideloadedChamfer {
        data: Arc<[MultiVec<f32>]>,
        rerank_queries: Option<Arc<[MultiVec<f32>]>>,
    },
    /// No reranking — pass through first-stage candidate ordering.
    PassThrough,
}

/// A multi-vector KNN searcher with configurable reranking.
///
/// The first-stage search always uses mean-pooled vectors via [`inmem::multi::Strategy`].
/// The reranking stage is determined by the [`MultiReranker`] variant.
#[cfg(feature = "spherical-quantization")]
struct ConfigurableMultiKNN {
    index: Arc<DiskANNIndex<inmem::multi::Provider<f32>>>,
    queries: Arc<[MultiVec<f32>]>,
    inner_loop: InnerLoop,
    reranker: MultiReranker,
}

#[cfg(feature = "spherical-quantization")]
impl ConfigurableMultiKNN {
    fn new(
        index: Arc<DiskANNIndex<inmem::multi::Provider<f32>>>,
        queries: Arc<[MultiVec<f32>]>,
        inner_loop: InnerLoop,
        reranker: MultiReranker,
    ) -> anyhow::Result<Arc<Self>> {
        // Validate rerank query lengths if provided.
        match &reranker {
            MultiReranker::QuantizedChamfer {
                rerank_queries: Some(rq),
                ..
            }
            | MultiReranker::SideloadedChamfer {
                rerank_queries: Some(rq),
                ..
            } => {
                if rq.len() != queries.len() {
                    anyhow::bail!(
                        "rerank queries length ({}) must match search queries length ({})",
                        rq.len(),
                        queries.len()
                    );
                }
            }
            _ => {}
        }

        Ok(Arc::new(Self {
            index,
            queries,
            inner_loop,
            reranker,
        }))
    }
}

#[cfg(feature = "spherical-quantization")]
impl benchmark_core::search::Search for ConfigurableMultiKNN {
    type Id = u32;
    type Parameters = graph::search::Knn;
    type Output = SearchMetrics;

    fn num_queries(&self) -> usize {
        self.queries.len()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> benchmark_core::search::IdCount {
        benchmark_core::search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        kind: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<u32> + Send,
    {
        let context = provider::DefaultContext;
        let rerank_micros = Arc::new(AtomicU64::new(0));
        let query = self.queries[index].as_view();

        // Select the inner-loop query (may differ from the rerank/search query).
        let inner_query = match &self.inner_loop {
            InnerLoop::SphericalChamfer {
                inner_queries: Some(iq),
                ..
            } => iq[index].as_view(),
            _ => query,
        };

        // Helper macro: dispatch reranker with a given strategy, using `inner_query`
        // for first-stage and `query` for the reranker.
        macro_rules! dispatch_rerank {
            ($strategy:expr) => {
                match &self.reranker {
                    MultiReranker::QuantizedChamfer {
                        data,
                        rerank_queries,
                        layout,
                    } => {
                        let rq = rerank_queries
                            .as_ref()
                            .map(|q| q[index].as_view())
                            .unwrap_or(query);
                        let reranker =
                            inmem::multi::QuantizedChamferRerank::new(data.clone(), rq, *layout);
                        let processor = TimedPostProcessor::new(
                            glue::Pipeline::new(glue::FilterStartPoints, reranker),
                            Arc::clone(&rerank_micros),
                        );
                        self.index
                            .search_with(
                                *kind,
                                &$strategy,
                                processor,
                                &context,
                                inner_query,
                                buffer,
                            )
                            .await?
                    }
                    MultiReranker::SideloadedChamfer {
                        data,
                        rerank_queries,
                    } => {
                        let rq = rerank_queries
                            .as_ref()
                            .map(|q| q[index].clone())
                            .unwrap_or_else(|| self.queries[index].clone());
                        let reranker = inmem::multi::SideloadedChamferRerank::new(data.clone(), rq);
                        let processor = TimedPostProcessor::new(
                            glue::Pipeline::new(glue::FilterStartPoints, reranker),
                            Arc::clone(&rerank_micros),
                        );
                        self.index
                            .search_with(
                                *kind,
                                &$strategy,
                                processor,
                                &context,
                                inner_query,
                                buffer,
                            )
                            .await?
                    }
                    MultiReranker::PassThrough => {
                        let processor = TimedPostProcessor::new(
                            glue::Pipeline::new(glue::FilterStartPoints, glue::CopyIds),
                            Arc::clone(&rerank_micros),
                        );
                        self.index
                            .search_with(
                                *kind,
                                &$strategy,
                                processor,
                                &context,
                                inner_query,
                                buffer,
                            )
                            .await?
                    }
                }
            };
        }

        let stats = match &self.inner_loop {
            InnerLoop::MeanPool => {
                let strategy = inmem::multi::Strategy::new();
                dispatch_rerank!(strategy)
            }
            InnerLoop::SphericalChamfer { data, layout, .. } => {
                let strategy = inmem::multi::SphericalChamferSearch::new(data.clone(), *layout);
                dispatch_rerank!(strategy)
            }
        };

        Ok(SearchMetrics {
            comparisons: stats.cmps,
            hops: stats.hops,
            rerank_latency: MicroSeconds::new(rerank_micros.load(Ordering::Relaxed)),
        })
    }
}

#[cfg(feature = "spherical-quantization")]
impl knn::Knn<u32> for Arc<ConfigurableMultiKNN> {
    fn search_all(
        &self,
        parameters: Vec<benchmark_core::search::Run<graph::search::Knn>>,
        groundtruth: &dyn benchmark_core::recall::Rows<u32>,
        recall_k: usize,
        recall_n: usize,
    ) -> anyhow::Result<Vec<SearchResults>> {
        let results = benchmark_core::search::search_all(
            self.clone(),
            parameters.into_iter(),
            Aggregator::new(groundtruth, recall_k, recall_n),
        )?;

        Ok(results.into_iter().map(SearchResults::from).collect())
    }
}
