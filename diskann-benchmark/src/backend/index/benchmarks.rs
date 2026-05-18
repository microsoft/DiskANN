/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::SampleableForStart,
    graph::{glue, DiskANNIndex},
    provider::{self, DataProvider, DefaultContext},
    utils::VectorRepr,
};
use diskann_benchmark_core::{
    self as benchmark_core,
    streaming::{executors::bigann, Executor},
};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint, Registry,
};
use diskann_providers::{
    index::diskann_async,
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::{common, inmem},
    },
};
use diskann_utils::{
    future::AsyncFriendly,
    sampling::WithApproximateNorm,
    views::{Matrix, MatrixView},
};
use half::f16;

use super::{
    build::{self, load_index, save_index, single_or_multi_insert, BuildStats},
    product, scalar, search, spherical,
};
use crate::{
    backend::index::{
        post_processor,
        result::{AggregatedSearchResults, BuildResult},
        search::plugins,
        streaming::{self, managed, stats::StreamStats, FullPrecisionStream, Managed},
    },
    inputs::graph_index::{DynamicIndexRun, IndexBuild, IndexOperation, IndexSource, SearchPhase},
    utils::{
        self,
        datafiles::{self},
        filters::{generate_bitmaps, setup_filter_strategies},
    },
};

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    // Notes on registration:
    //
    // We register all supported search types for `f32`, but intentionally limit the number
    // of search types for the other data types mainly to help reduce compilation time.
    //
    // Feel free to add additional search plugins as needed during exploration and add them
    // permanently if demand is sufficient.
    //
    // Note that each plugin registration will trigger an new monomorphization, so use with
    // care.

    // Full Precision
    registry.register(
        "graph-index-full-precision-f32",
        FullPrecision::<f32>::new()
            .search(plugins::DeterminantDiversity)
            .search(plugins::Topk)
            .search(plugins::Range)
            .search(plugins::TopkBetaFilter)
            .search(plugins::TopkMultihopFilter),
    )?;

    registry.register(
        "graph-index-full-precision-f16",
        FullPrecision::<f16>::new().search(plugins::Topk),
    )?;
    registry.register(
        "graph-index-full-precision-u8",
        FullPrecision::<u8>::new().search(plugins::Topk),
    )?;
    registry.register(
        "graph-index-full-precision-i8",
        FullPrecision::<i8>::new().search(plugins::Topk),
    )?;

    // Dynamic Full Precision
    registry.register(
        "graph-index-dynamic-full-precision-f32",
        DynamicFullPrecision::<f32>::new(),
    )?;
    registry.register(
        "graph-index-dynamic-full-precision-f16",
        DynamicFullPrecision::<f16>::new(),
    )?;
    registry.register(
        "graph-index-dynamic-full-precision-u8",
        DynamicFullPrecision::<u8>::new(),
    )?;
    registry.register(
        "graph-index-dynamic-full-precision-i8",
        DynamicFullPrecision::<i8>::new(),
    )?;

    product::register_benchmarks(registry)?;
    scalar::register_benchmarks(registry)?;
    spherical::register_benchmarks(registry)?;
    Ok(())
}

type FullPrecisionProvider<T> = inmem::DefaultProvider<
    inmem::FullPrecisionStore<T>,
    common::NoStore,
    common::NoDeletes,
    DefaultContext,
>;

/// Associate a type (usually a [`diskann::provider::DataProvider`]) with a full-precision
/// element type. This is used in implementations of [`plugins::Plugin`] to derive the
/// correct query types to load.
pub(super) trait QueryType {
    type Element: VectorRepr;
}

impl<T> QueryType for FullPrecisionProvider<T>
where
    T: VectorRepr,
{
    type Element = T;
}

/// A [`Benchmark`] for full-precision searches containing a dynamic list of search types.
struct FullPrecision<T>
where
    T: VectorRepr,
{
    plugins:
        plugins::Plugins<FullPrecisionProvider<T>, SearchPhase, Strategy<common::FullPrecision>>,
}

impl<T> FullPrecision<T>
where
    T: VectorRepr,
{
    fn new() -> Self {
        Self {
            plugins: plugins::Plugins::new(),
        }
    }

    fn search<P>(mut self, plugin: P) -> Self
    where
        P: plugins::Plugin<FullPrecisionProvider<T>, SearchPhase, Strategy<common::FullPrecision>>
            + 'static,
    {
        self.plugins.register(plugin);
        self
    }
}

impl<T> Benchmark for FullPrecision<T>
where
    T: VectorRepr
        + diskann_utils::sampling::WithApproximateNorm
        + diskann::graph::SampleableForStart
        + AsDataType,
{
    type Input = IndexOperation;
    type Output = BuildResult;

    fn try_match(&self, input: &IndexOperation) -> Result<MatchScore, FailureScore> {
        let score = utils::match_data_type::<T>(*input.source.data_type());
        if self.plugins.is_match(&input.search_phase) {
            score
        } else {
            match score {
                Ok(_) => Err(FailureScore(0)),
                Err(score) => Err(score),
            }
        }
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&IndexOperation>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => {
                let desc = T::describe(*arg.source.data_type());
                if !desc.is_match() {
                    writeln!(f, "Data/Query Type: {}", desc)?;
                }

                if !self.plugins.is_match(&arg.search_phase) {
                    writeln!(
                        f,
                        "Unsupported search phase: \"{}\" - expected one of {}",
                        arg.search_phase.kind(),
                        self.plugins.format_kinds(),
                    )?;
                }
                Ok(())
            }
            None => {
                writeln!(f, "Data/Query Type: {}", T::DATA_TYPE)?;
                writeln!(f, "Search Kinds: {}", self.plugins.format_kinds())
            }
        }
    }

    fn run(
        &self,
        input: &IndexOperation,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<BuildResult> {
        writeln!(output, "{}", input)?;
        let (index, build_stats) = match &input.source {
            IndexSource::Build(build) => {
                let (index, build_stats) = run_build(
                    build,
                    common::FullPrecision,
                    None,
                    output,
                    |data| {
                        let index = diskann_async::new_index::<T, _>(
                            build.try_as_config()?.build()?,
                            build.inmem_parameters(data.nrows(), data.ncols()),
                            common::NoDeletes,
                        )?;
                        build::set_start_points(
                            index.provider(),
                            data.as_view(),
                            build.start_point_strategy,
                        )?;
                        Ok(index)
                    },
                    single_or_multi_insert,
                )?;

                // save the index if requested
                if let Some(save_path) = &build.save_path {
                    utils::tokio::block_on(save_index(index.clone(), save_path))?;
                }

                (index, Some(build_stats))
            }
            IndexSource::Load(load) => {
                let index_config: &IndexConfiguration = &load.to_config()?;

                let index =
                    { utils::tokio::block_on(load_index::<_>(&load.load_path, index_config))? };

                (Arc::new(index), None::<BuildStats>)
            }
        };

        // Save construction stats before running queries.
        checkpoint.checkpoint(&build_stats)?;

        let search_results = self.plugins.run(
            index,
            &input.search_phase,
            &Strategy::new(common::FullPrecision),
        )?;

        let result = BuildResult::new(build_stats, search_results);

        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}

// Graph Index Dynamic Run
pub(super) struct DynamicFullPrecision<T> {
    _type: std::marker::PhantomData<T>,
}

impl<T> DynamicFullPrecision<T> {
    fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for DynamicFullPrecision<T>
where
    T: VectorRepr
        + diskann_utils::sampling::WithApproximateNorm
        + diskann::graph::SampleableForStart
        + AsDataType,
{
    type Input = DynamicIndexRun;
    type Output = Vec<managed::Stats<StreamStats>>;

    fn try_match(&self, input: &DynamicIndexRun) -> Result<MatchScore, FailureScore> {
        utils::match_data_type::<T>(input.build.data_type)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&DynamicIndexRun>,
    ) -> std::fmt::Result {
        match input {
            Some(i) => write!(f, "{}", T::describe(i.build.data_type)),
            None => write!(f, "{}", T::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &DynamicIndexRun,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Vec<managed::Stats<StreamStats>>> {
        writeln!(output, "{}", input)?;

        let groundtruth_directory = input
            .runbook_params
            .resolved_gt_directory
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("Ground truth directory path was not resolved during validation")
            })?;

        let mut runbook = bigann::RunBook::load(
            &input.runbook_params.runbook_path,
            &input.runbook_params.dataset_name,
            &mut bigann::ScanDirectory::new(groundtruth_directory)?,
        )?;

        let mut streamer = full_precision_streaming::<T>(input, runbook.max_points())?;

        let mut results = Vec::new();
        let stages = runbook.len();
        let mut i = 1;

        runbook.run_with(
            &mut streamer,
            |o: managed::Stats<StreamStats>| -> anyhow::Result<()> {
                if o.inner().is_maintain() {
                    let message = format!("Ran maintenance before stage {}", i);
                    write!(output, "{}", crate::utils::SmallBanner(&message))?;
                } else {
                    let message =
                        format!("Finished stage {} of {}: {}", i, stages, o.inner().kind());
                    write!(output, "{}", crate::utils::SmallBanner(&message))?;
                    i += 1;
                }
                writeln!(output, "{}", o)?;
                results.push(o);
                Ok(())
            },
        )?;

        write!(
            output,
            "{}",
            crate::utils::SmallBanner("End of Run Summary")
        )?;

        writeln!(
            output,
            "{}",
            streaming::stats::Summary::new(results.iter().map(|r| r.inner()))
        )?;

        Ok(results)
    }
}

// Simplify reasoning about this rather hefty type.
type Index<DP> = Arc<DiskANNIndex<DP>>;

pub(super) fn run_build<T, BF, CF, B, DP>(
    input: &IndexBuild,
    build_strategy: B,
    data: Option<Arc<Matrix<T>>>,
    output: &mut dyn Output,
    create: CF,
    build: BF,
) -> anyhow::Result<(Index<DP>, BuildStats)>
where
    DP: DataProvider<Context = DefaultContext, InternalId = u32, ExternalId = u32>
        + for<'a> provider::SetElement<&'a [T]>,
    CF: FnOnce(MatrixView<T>) -> anyhow::Result<Arc<DiskANNIndex<DP>>>,
    T: diskann::graph::SampleableForStart + std::fmt::Debug + Copy + AsyncFriendly + bytemuck::Pod,
    B: for<'a> glue::SearchStrategy<DP, &'a [T]> + Clone + Send + Sync,
    BF: FnOnce(
        Index<DP>,
        B,
        Arc<Matrix<T>>,
        &IndexBuild,
        &mut dyn Output,
    ) -> anyhow::Result<BuildStats>,
{
    let data = match data {
        Some(data) => data,
        None => Arc::new(datafiles::load_dataset(datafiles::BinFile(&input.data))?),
    };

    let index = create(data.as_view())?;
    let build_stats = build(index.clone(), build_strategy.clone(), data, input, output)?;

    Ok((index, build_stats))
}

/// A new-type wrapper for [`glue::SearchStrategy`].
///
/// This exists so we can implement [`search::Plugin`] for a raw generic `DP` without
/// forming a blanket implementation for all `DP`/parameter `P` pairs.
#[derive(Debug, Clone, Copy)]
pub(super) struct Strategy<S>(S);

impl<S> Strategy<S> {
    pub(super) fn new(strategy: S) -> Self {
        Self(strategy)
    }

    pub(super) fn inner(&self) -> S
    where
        S: Clone,
    {
        self.0.clone()
    }
}

//------//
// Topk //
//------//

struct DeterminantDiversityKnn {
    index: Arc<DiskANNIndex<FullPrecisionProvider<f32>>>,
    queries: Arc<Matrix<f32>>,
    strategy: benchmark_core::search::graph::Strategy<common::FullPrecision>,
    post_processor: post_processor::DeterminantDiversity,
}

impl DeterminantDiversityKnn {
    fn new(
        index: Arc<DiskANNIndex<FullPrecisionProvider<f32>>>,
        queries: Arc<Matrix<f32>>,
        strategy: benchmark_core::search::graph::Strategy<common::FullPrecision>,
        post_processor: post_processor::DeterminantDiversity,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;
        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
            post_processor,
        }))
    }
}

impl benchmark_core::search::Search for DeterminantDiversityKnn
where
    common::FullPrecision: for<'a, 'b> glue::SearchStrategy<
        FullPrecisionProvider<f32>,
        &'a [f32],
        SearchAccessor<'b>: post_processor::determinant_diversity::FullPrecisionVectorAccessor,
    >,
{
    type Id = u32;
    type Parameters = diskann::graph::search::Knn;
    type Output = benchmark_core::search::graph::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> benchmark_core::search::IdCount {
        benchmark_core::search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> diskann::ANNResult<Self::Output>
    where
        O: diskann::graph::SearchOutputBuffer<Self::Id> + Send,
    {
        let context = DefaultContext;
        let stats = self
            .index
            .search_with(
                *parameters,
                self.strategy.get(index)?,
                self.post_processor,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(benchmark_core::search::graph::knn::Metrics::new(
            stats.cmps, stats.hops,
        ))
    }
}

impl search::Plugin<FullPrecisionProvider<f32>, SearchPhase, Strategy<common::FullPrecision>>
    for plugins::DeterminantDiversity
where
    common::FullPrecision: for<'a, 'b> glue::SearchStrategy<
        FullPrecisionProvider<f32>,
        &'a [f32],
        SearchAccessor<'b>: post_processor::determinant_diversity::FullPrecisionVectorAccessor,
    >,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::DeterminantDiversity::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::DeterminantDiversity::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<FullPrecisionProvider<f32>>>,
        phase: &SearchPhase,
        _strategy: &Strategy<common::FullPrecision>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let (topk, params) = plugins::DeterminantDiversity::get(phase)?;

        let queries = Arc::new(datafiles::load_dataset::<f32>(datafiles::BinFile(
            &topk.queries,
        ))?);
        let groundtruth = datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth))?;

        let knn = DeterminantDiversityKnn::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(common::FullPrecision),
            post_processor::DeterminantDiversity::new(params.power(), params.eta()),
        )?;

        let steps = search::knn::SearchSteps::new(topk.reps, &topk.num_threads, &topk.runs);
        let results = search::knn::run(&knn, &groundtruth, steps)?;

        Ok(AggregatedSearchResults::Topk(results))
    }
}

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::Topk
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [DP::Element]> + Clone + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::Topk::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::Topk::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let topk = phase.as_topk()?;

        let queries: Arc<Matrix<DP::Element>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&topk.queries))?);

        let groundtruth = datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth))?;

        let knn = benchmark_core::search::graph::KNN::new(
            index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
        )?;

        let steps = search::knn::SearchSteps::new(topk.reps, &topk.num_threads, &topk.runs);

        let results = search::knn::run(&knn, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(results))
    }
}

//-------//
// Range //
//-------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::Range
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [DP::Element]> + Clone + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::Range::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::Range::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let range = phase.as_range()?;
        let queries: Arc<Matrix<DP::Element>> =
            Arc::new(datafiles::load_dataset(datafiles::BinFile(&range.queries))?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&range.groundtruth))?;

        let steps =
            search::range::RangeSearchSteps::new(range.reps, &range.num_threads, &range.runs);

        let range = benchmark_core::search::graph::Range::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
        )?;

        let result = search::range::run(&range, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Range(result))
    }
}

//------------//
// BetaFilter //
//------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkBetaFilter
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [DP::Element]> + Clone + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::TopkBetaFilter::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::TopkBetaFilter::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let beta_filter = phase.as_topk_beta_filter()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&beta_filter.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&beta_filter.groundtruth))?;

        let bit_maps = generate_bitmaps(&beta_filter.query_predicates, &beta_filter.data_labels)?;

        let search_strategies = setup_filter_strategies(
            beta_filter.beta,
            bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider),
            strategy.inner(),
        );

        let knn = benchmark_core::search::graph::KNN::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::collection(search_strategies),
        )?;

        let steps = search::knn::SearchSteps::new(
            beta_filter.reps,
            &beta_filter.num_threads,
            &beta_filter.runs,
        );

        let result = search::knn::run(&knn, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//----------------//
// MultihopFilter //
//----------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopFilter
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [DP::Element]> + Clone + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::TopkMultihopFilter::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::TopkMultihopFilter::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_filter()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps =
            search::knn::SearchSteps::new(multihop.reps, &multihop.num_threads, &multihop.runs);

        let bit_maps = generate_bitmaps(&multihop.query_predicates, &multihop.data_labels)?;

        let multihop = benchmark_core::search::graph::MultiHop::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider)
                .collect(),
        )?;

        let result = search::knn::run(&multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

/// The stack looks like this:
///
/// - Bottom: [`FullPrecisionStream`]: The core streaming index implementation.
/// - Middle: [`Managed`]: Since the in-mem index currently does not split internal and external
///   IDs, the [`Managed`] layer is introduced as a temporary measure. This is responsible
///   for ID mapping.
/// - Top: [`bigann::WithData`]: The top layer maps raw index IDs to actual data points.
///
/// This function constructs the entire stack.
fn full_precision_streaming<T>(
    input: &DynamicIndexRun,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, StreamStats>>>
where
    T: bytemuck::Pod + VectorRepr + WithApproximateNorm + SampleableForStart,
{
    let topk = match &input.search_phase {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };
    let consolidate_threshold: f32 = input.runbook_params.consolidate_threshold;

    let data = datafiles::load_dataset::<T>(datafiles::BinFile(&input.build.data))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &topk.queries,
    ))?);

    // Create a little extra headroom to handle deferred maintenance.
    let max_points = ((max_points as f32) * (1.0 + 2.0 * consolidate_threshold)).ceil() as usize;

    let index = diskann_async::new_index::<T, _>(
        input.try_as_config(input.build.l_build)?.build()?,
        input.inmem_parameters(max_points, data.ncols()),
        common::TableBasedDeletes,
    )?;

    build::set_start_points(
        index.provider(),
        data.as_view(),
        input.build.start_point_strategy,
    )?;

    let num_threads_and_tasks = NonZeroUsize::new(input.build.num_threads).unwrap();
    let managed_stream = FullPrecisionStream {
        index,
        search: topk.clone(),
        runtime: benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
        ntasks: num_threads_and_tasks,
        inplace_delete_num_to_replace: input.runbook_params.ip_delete_num_to_replace,
        inplace_delete_method: input.runbook_params.ip_delete_method.into(),
    };

    let managed = Managed::new(max_points, consolidate_threshold, managed_stream);

    let layered = bigann::WithData::new(managed, data, queries, |path| {
        Ok(Box::new(datafiles::load_groundtruth(datafiles::BinFile(
            path,
        ))?))
    });

    Ok(layered)
}
