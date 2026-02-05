/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;
use std::sync::Arc;

use diskann::{ANNResult, graph::{self, glue, DiskANNIndex}, provider};
use diskann_benchmark_core::{self as benchmark_core, build::ids::ToId};
use diskann_benchmark_runner::{
    self as runner,
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    registry::Benchmarks,
    utils::{percentiles, MicroSeconds},
    Any,
};
use diskann_providers::model::graph::provider::async_::{common as inmem_common, inmem};
use diskann_quantization::multi_vector::{Mat, Standard};
use diskann_utils::future::AsyncFriendly;

use crate::{
    backend::index::{result::SearchResults, search::knn},
    inputs,
    utils::datafiles,
};

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
        Any::description::<inputs::multi::BuildAndSearch, Self>(f, from, inputs::multi::BuildAndSearch::tag())
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
    output: &mut dyn runner::Output,
) -> anyhow::Result<()> {
    let index = build(&input.build, output)?;
    let queries: Arc<[_]> = datafiles::load_multi_vectors::<f32>(&input.search.queries)?.into();

    // TODO: Placeholder.
    let gt = ZeroGroundtruth::new(queries.len(), 100);

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

    knn::run(
        &searcher,
        &gt,
        steps,
    )?;

    Ok(())
}

///////////
// Build //
///////////

fn build(
    input: &inputs::multi::Build,
    output: &mut dyn runner::Output,
) -> anyhow::Result<Arc<DiskANNIndex<inmem::multi::Provider<f32>>>>
{
    let data = datafiles::load_multi_vectors::<f32>(&input.data)?;

    let dim = data.first().unwrap().vector_dim();

    let provider = inmem::DefaultProvider::<_, _, _, provider::DefaultContext>::new_empty(
        input.inmem_parameters(data.len(), dim),
        inmem::multi::Precursor::<f32>::new(dim),
        inmem_common::NoStore,
        inmem_common::NoDeletes,
    ).unwrap();

    let index = Arc::new(graph::DiskANNIndex::new(
        input.as_config().build()?,
        provider,
        None
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
        benchmark_core::build::Parallelism::dynamic(
            diskann::utils::ONE,
            input.num_threads,
        ),
        &rt,
        Some(&super::build::ProgressMeter::new(output)),
    )?;

    Ok(index)
}

type MultiVec<T> = Mat<Standard<T>>;

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
    DP: provider::DataProvider<Context: Default> + provider::SetElement<MultiVec<T>>,
    S: glue::InsertStrategy<DP, MultiVec<T>> + Clone + AsyncFriendly,
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
                    &self.data[i],
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
}

impl<DP, T, S> benchmark_core::search::Search for MultiKNN<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: benchmark_core::search::Id>,
    S: glue::SearchStrategy<DP, MultiVec<T>, DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Copy,
{
    type Id = DP::ExternalId;
    type Parameters = graph::SearchParams;
    type Output = SearchMetrics;

    fn num_queries(&self) -> usize {
        self.queries.len()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> benchmark_core::search::IdCount {
        benchmark_core::search::IdCount::Fixed(
            std::num::NonZeroUsize::new(parameters.k_value).unwrap_or(diskann::utils::ONE),
        )
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let stats = self
            .index
            .search(
                self.strategy.get(index)?,
                &context,
                &self.queries[index],
                parameters,
                buffer,
            )
            .await?;

        Ok(SearchMetrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

/// An aggregated summary of multiple [`MultiKNN`] search runs.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    /// The [`benchmark_core::search::Setup`] used for the batch of runs.
    pub setup: benchmark_core::search::Setup,

    /// The [`graph::SearchParams`] used for the batch of runs.
    pub parameters: graph::SearchParams,

    /// The end-to-end latency for each repetition in the batch.
    pub end_to_end_latencies: Vec<MicroSeconds>,

    /// The average latency for individual queries.
    pub mean_latencies: Vec<f64>,

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

impl<I> benchmark_core::search::Aggregate<graph::SearchParams, I, SearchMetrics> for Aggregator<'_, I>
where
    I: benchmark_core::recall::RecallCompatible,
{
    type Output = Summary;

    fn aggregate(
        &mut self,
        run: benchmark_core::search::Run<graph::SearchParams>,
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
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
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

        let mean_cmps = if count > 0 { total_cmps as f64 / count as f64 } else { 0.0 };
        let mean_hops = if count > 0 { total_hops as f64 / count as f64 } else { 0.0 };

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            recall,
            mean_latencies,
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
        Parameters = graph::SearchParams,
        Output = SearchMetrics,
    >,
    T: Copy,
{
    fn search_all(
        &self,
        parameters: Vec<benchmark_core::search::Run<graph::SearchParams>>,
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
            search_n: parameters.k_value,
            search_l: parameters.l_value,
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
        }
    }
}

///////////////////
// Test Helpers  //
///////////////////

/// A stub groundtruth that returns zeros for all queries.
///
/// Useful for end-to-end testing when no groundtruth file is available.
/// Recall metrics will be meaningless but the search pipeline can still run.
pub struct ZeroGroundtruth {
    nrows: usize,
    row: Vec<u32>,
}

impl ZeroGroundtruth {
    /// Create a new stub groundtruth with `nrows` queries, each with `ncols` neighbors.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            row: vec![0u32; ncols],
        }
    }
}

impl benchmark_core::recall::Rows<u32> for ZeroGroundtruth {
    fn nrows(&self) -> usize {
        self.nrows
    }

    fn row(&self, _i: usize) -> &[u32] {
        &self.row
    }

    fn ncols(&self) -> Option<usize> {
        Some(self.row.len())
    }
}

