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
    recall::GroundTruthMode,
    streaming::{executors::bigann, Executor},
};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
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
    views::{Matrix, MatrixView},
};
use half::f16;

use super::{
    build::{self, load_index, save_index, single_or_multi_insert, BuildStats},
    inmem::{product, scalar, spherical},
    search,
};
use crate::{
    index::{
        result::{AggregatedSearchResults, BuildResult},
        search::plugins,
        streaming::{self, managed, stats::StreamStats, FullPrecisionStream, Managed},
    },
    inputs::graph_index::{
        DynamicIndexRun, IndexBuild, IndexOperation, IndexSource, MultihopFilterSearchPhase,
        SearchPhase,
    },
    utils::{
        self,
        datafiles::{self},
        filters::{generate_bitmaps, setup_filter_strategies},
    },
};

////////////////////////////
// Benchmark Registration //
////////////////////////////

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
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
            .search(plugins::Topk)
            .search(plugins::Range)
            .search(plugins::TopkBetaFilter)
            .search(plugins::TopkMultihopFilter)
            .search(plugins::TopkMultihopLiveFilter)
            .search(plugins::TopkMultihopLiveFilterCsr)
            .search(plugins::TopkMultihopLiveFilterBitmap)
            .search(plugins::TopkMultihopLiveFilterAuto)
            .search(plugins::TopkMultihopLiveFilterBitslice)
            .search(plugins::TopkMultihopLiveFilterBitsliceDnf)
            .search(plugins::TopkMultihopEncodedBitsliceDnf)
            .search(plugins::TopkMultihopEncodedBitsliceAst)
            .search(plugins::TopkMultihopEncodedBitmapAst)
            .search(plugins::TopkInlineLiveFilterBitsliceDnf)
            .search(plugins::TopkInlineFilter)
            .search(plugins::DeterminantDiversity),
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
pub(crate) trait QueryType {
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
    T: VectorRepr + diskann::graph::SampleableForStart + AsDataType,
{
    type Input = IndexOperation;
    type Output = BuildResult;

    fn try_match(&self, input: &IndexOperation, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        utils::match_data_type::<T>(&mut score, *input.source.data_type());
        if !self.plugins.is_match(&input.search_phase) {
            score.fail(
                1,
                &format_args!(
                    "Unsupported search phase: \"{}\" - expected one of {}",
                    input.search_phase.kind(),
                    self.plugins.format_kinds(),
                ),
            );
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Data/Query Type: {}", T::DATA_TYPE)?;
        writeln!(f, "Search Kinds: {}", self.plugins.format_kinds())
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
                            *build.start_point_strategy(),
                        )?;
                        Ok(index)
                    },
                    single_or_multi_insert,
                )?;

                // save the index if requested
                if let Some(save_path) = build.save_path() {
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
pub(crate) struct DynamicFullPrecision<T> {
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
    T: VectorRepr + diskann::graph::SampleableForStart + AsDataType,
{
    type Input = DynamicIndexRun;
    type Output = Vec<managed::Stats<StreamStats>>;

    fn try_match(&self, input: &DynamicIndexRun, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        utils::match_data_type::<T>(&mut score, input.build.data_type());
        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", T::DATA_TYPE)
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

pub(crate) fn run_build<T, BF, CF, B, DP>(
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
    B: for<'a> glue::SearchStrategy<'a, DP, &'a [T]> + Clone + Send + Sync,
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
        None => Arc::new(datafiles::load_dataset(datafiles::BinFile(input.data()))?),
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
pub(crate) struct Strategy<S>(S);

impl<S> Strategy<S> {
    pub(crate) fn new(strategy: S) -> Self {
        Self(strategy)
    }

    pub(crate) fn inner(&self) -> S
    where
        S: Clone,
    {
        self.0.clone()
    }
}

fn run_multihop_encoded<DP, S>(
    index: Arc<DiskANNIndex<DP>>,
    phase: &MultihopFilterSearchPhase,
    strategy: &Strategy<S>,
    mode: utils::filters::EncodedQueryMode,
    expected_format: diskann_label_index::LabelIndexFormat,
) -> anyhow::Result<AggregatedSearchResults>
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    let queries: Arc<Matrix<DP::Element>> =
        Arc::new(datafiles::load_dataset(datafiles::BinFile(&phase.queries))?);

    let groundtruth = datafiles::load_range_groundtruth(datafiles::BinFile(&phase.groundtruth))?;

    let steps = search::knn::SearchSteps::new(
        phase.reps,
        &phase.num_threads,
        &phase.runs,
        GroundTruthMode::Flexible,
    );

    // For encoded-label-index search phases, `data_labels` intentionally points at the persisted
    // encoded label-index file, not the raw labels JSONL. Loading that index and parsing/validating
    // the predicate JSONL (including ASTExpr -> LabelExpression conversion plus AST-JSON or DNF
    // source preparation) stay outside timing. Each timed repetition/search-L rebuilds fresh lazy
    // providers so the first `is_match` includes `EncodedLabelIndex::{query, query_ast_json}`,
    // label-id lookup, AST parsing/compilation, and bitmap AST dense materialization.
    let label_index = utils::filters::load_encoded_label_index(&phase.data_labels)?;
    if label_index.format() != expected_format {
        anyhow::bail!(
            "encoded search mode expected {:?} label storage, but {} contains {:?}",
            expected_format,
            phase.data_labels.display(),
            label_index.format()
        );
    }
    let query_sources = utils::filters::prepare_encoded_query_sources(
        label_index.as_ref(),
        &phase.query_predicates,
        mode,
    )?;
    let make_multihop = || {
        let providers =
            utils::filters::make_encoded_query_providers(label_index.clone(), &query_sources);
        benchmark_core::search::graph::MultiHop::new(
            index.clone(),
            queries.clone(),
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
        )
    };

    let result = search::knn::run_fresh_multihop(make_multihop, &groundtruth, steps)?;
    Ok(AggregatedSearchResults::Topk(result))
}

//------//
// Topk //
//------//

impl search::Plugin<FullPrecisionProvider<f32>, SearchPhase, Strategy<common::FullPrecision>>
    for plugins::DeterminantDiversity
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
        let (phase, params) = plugins::DeterminantDiversity::get(phase)?;

        let queries = Arc::new(datafiles::load_dataset::<f32>(datafiles::BinFile(
            &phase.queries,
        ))?);
        let groundtruth = datafiles::load_groundtruth(
            datafiles::BinFile(&phase.groundtruth),
            Some(phase.max_k()),
        )?;

        let knn = benchmark_core::search::graph::KNN::with_postprocessor(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(common::FullPrecision),
            inmem::DeterminantDiversity::new(params),
        )?;

        let steps = search::knn::SearchSteps::new(
            phase.reps,
            &phase.num_threads,
            &phase.runs,
            GroundTruthMode::Fixed,
        );
        let results = search::knn::run(&knn, &groundtruth, steps)?;

        Ok(AggregatedSearchResults::Topk(results))
    }
}

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::Topk
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
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

        // compute the maximum value of k used in any search
        let max_k = topk.max_k();

        let groundtruth =
            datafiles::load_groundtruth(datafiles::BinFile(&topk.groundtruth), Some(max_k))?;

        let knn = benchmark_core::search::graph::KNN::new(
            index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
        )?;

        let steps = search::knn::SearchSteps::new(
            topk.reps,
            &topk.num_threads,
            &topk.runs,
            GroundTruthMode::Fixed,
        );

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
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
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
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
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
            GroundTruthMode::Flexible,
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
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
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

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

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

//--------------------//
// MultihopLiveFilter //
//--------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopLiveFilter
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        // Build the in-memory attribute index once (a one-time index build), then construct a
        // live per-query provider that evaluates the predicate against each visited node's
        // attributes during search.
        let attribute_index = utils::filters::build_inline_attribute_index(&multihop.data_labels)?;
        let providers =
            utils::filters::make_live_providers(&attribute_index, &multihop.query_predicates)?;

        let multihop = benchmark_core::search::graph::MultiHop::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
        )?;

        let result = search::knn::run(&multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//--------------------------//
// MultihopLiveFilter (CSR) //
//--------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopLiveFilterCsr
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter_csr()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        // Build the in-memory CSR attribute index once, then construct a live per-query provider
        // that evaluates the predicate against each visited node's contiguous attribute row.
        let attribute_index =
            utils::filters::build_inline_attribute_index_csr(&multihop.data_labels)?;
        let providers =
            utils::filters::make_live_providers_csr(&attribute_index, &multihop.query_predicates)?;

        let multihop = benchmark_core::search::graph::MultiHop::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
        )?;

        let result = search::knn::run(&multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//-----------------------------//
// MultihopLiveFilter (Bitmap)  //
//-----------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopLiveFilterBitmap
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter_bitmap()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        // Build the posting-list index once. Fresh providers are constructed for every benchmark
        // repetition/search-L so lazy match-set materialization is charged to each query execution.
        let attribute_index =
            utils::filters::build_inline_attribute_index_posting(&multihop.data_labels)?;
        let make_multihop = || {
            let providers = utils::filters::make_live_providers_posting(
                &attribute_index,
                &multihop.query_predicates,
            )?;
            benchmark_core::search::graph::MultiHop::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
                providers.into(),
            )
        };

        let result = search::knn::run_fresh_multihop(make_multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//---------------------------//
// MultihopLiveFilter (Auto)  //
//---------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopLiveFilterAuto
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter_auto()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        let attribute_index =
            utils::filters::build_inline_attribute_index_auto(&multihop.data_labels)?;
        // Recreate providers for every repetition/search-L so the lazy strategy decision and any
        // dense materialization cannot be amortized across benchmark executions.
        let make_multihop = || {
            let providers = utils::filters::make_live_providers_auto(
                &attribute_index,
                &multihop.query_predicates,
            )?;
            benchmark_core::search::graph::MultiHop::new(
                index.clone(),
                queries.clone(),
                benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
                providers.into(),
            )
        };

        let result = search::knn::run_fresh_multihop(make_multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//--------------------------------//
// MultihopLiveFilter (Bitslice)   //
//--------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopLiveFilterBitslice
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter_bitslice()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        let attribute_index =
            utils::filters::build_inline_attribute_index_bitslice(&multihop.data_labels)?;
        let providers = utils::filters::make_live_providers_bitslice(
            &attribute_index,
            &multihop.query_predicates,
        )?;

        let multihop = benchmark_core::search::graph::MultiHop::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
        )?;

        let result = search::knn::run(&multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//------------------------------------//
// MultihopLiveFilter (Bitslice DNF)   //
//------------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>>
    for plugins::TopkMultihopLiveFilterBitsliceDnf
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let multihop = phase.as_topk_multihop_live_filter_bitslice_dnf()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&multihop.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&multihop.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            multihop.reps,
            &multihop.num_threads,
            &multihop.runs,
            GroundTruthMode::Flexible,
        );

        let attribute_index =
            utils::filters::build_inline_attribute_index_bitslice(&multihop.data_labels)?;
        let providers = utils::filters::make_live_providers_bitslice_dnf(
            &attribute_index,
            &multihop.query_predicates,
        )?;

        let multihop = benchmark_core::search::graph::MultiHop::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
        )?;

        let result = search::knn::run(&multihop, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//--------------------------------------//
// MultihopEncodedFilter (Bitslice DNF) //
//--------------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopEncodedBitsliceDnf
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        run_multihop_encoded(
            index,
            phase.as_topk_multihop_encoded_bitslice_dnf()?,
            strategy,
            utils::filters::EncodedQueryMode::Dnf,
            diskann_label_index::LabelIndexFormat::Bitslice,
        )
    }
}

//--------------------------------------//
// MultihopEncodedFilter (Bitslice AST) //
//--------------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopEncodedBitsliceAst
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        run_multihop_encoded(
            index,
            phase.as_topk_multihop_encoded_bitslice_ast()?,
            strategy,
            utils::filters::EncodedQueryMode::Ast,
            diskann_label_index::LabelIndexFormat::Bitslice,
        )
    }
}

//------------------------------------//
// MultihopEncodedFilter (Bitmap AST) //
//------------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkMultihopEncodedBitmapAst
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        run_multihop_encoded(
            index,
            phase.as_topk_multihop_encoded_bitmap_ast()?,
            strategy,
            utils::filters::EncodedQueryMode::Ast,
            diskann_label_index::LabelIndexFormat::Bitmap,
        )
    }
}

//---------------------------------//
// InlineFilter (Bitslice DNF)      //
//---------------------------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>>
    for plugins::TopkInlineLiveFilterBitsliceDnf
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let inline = phase.as_topk_inline_live_filter_bitslice_dnf()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&inline.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&inline.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            inline.reps,
            &inline.num_threads,
            &inline.runs,
            GroundTruthMode::Flexible,
        );

        let attribute_index =
            utils::filters::build_inline_attribute_index_bitslice(&inline.data_labels)?;
        let providers = utils::filters::make_live_providers_bitslice_dnf(
            &attribute_index,
            &inline.query_predicates,
        )?;

        let inline = benchmark_core::search::graph::InlineFilterSearch::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            providers.into(),
            inline.adaptive_l()?,
        )?;

        let result = search::knn::run(&inline, &groundtruth, steps)?;
        Ok(AggregatedSearchResults::Topk(result))
    }
}

//--------------//
// InlineFilter //
//--------------//

impl<DP, S> search::Plugin<DP, SearchPhase, Strategy<S>> for plugins::TopkInlineFilter
where
    DP: DataProvider<Context: Default, InternalId = u32, ExternalId = u32> + QueryType,
    S: for<'a> glue::DefaultSearchStrategy<
            'a,
            DP,
            &'a [DP::Element],
            SearchAccessor: glue::SearchAccessor,
        > + Clone
        + AsyncFriendly,
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        plugins::TopkInlineFilter::is_match(phase)
    }

    fn kind(&self) -> &'static str {
        plugins::TopkInlineFilter::as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        phase: &SearchPhase,
        strategy: &Strategy<S>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        let inline = phase.as_topk_inline_filter()?;

        let queries: Arc<Matrix<DP::Element>> = Arc::new(datafiles::load_dataset(
            datafiles::BinFile(&inline.queries),
        )?);

        let groundtruth =
            datafiles::load_range_groundtruth(datafiles::BinFile(&inline.groundtruth))?;

        let steps = search::knn::SearchSteps::new(
            inline.reps,
            &inline.num_threads,
            &inline.runs,
            GroundTruthMode::Flexible,
        );

        let bit_maps = generate_bitmaps(&inline.query_predicates, &inline.data_labels)?;

        let inline = benchmark_core::search::graph::InlineFilterSearch::new(
            index,
            queries,
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            bit_maps
                .into_iter()
                .map(utils::filters::as_query_label_provider)
                .collect(),
            inline.adaptive_l()?,
        )?;

        let result = search::knn::run(&inline, &groundtruth, steps)?;
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
    T: bytemuck::Pod + VectorRepr + SampleableForStart,
{
    let topk = input.search_phase.as_topk()?;

    let consolidate_threshold: f32 = input
        .runbook_params
        .consolidate_threshold
        .ok_or_else(|| anyhow::anyhow!("consolidate_threshold is required for inmem streaming"))?;

    let data = datafiles::load_dataset::<T>(datafiles::BinFile(input.build.data()))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &topk.queries,
    ))?);

    // Create a little extra headroom to handle deferred maintenance.
    let max_points = ((max_points as f32) * (1.0 + 2.0 * consolidate_threshold)).ceil() as usize;

    let index = diskann_async::new_index::<T, _>(
        input.try_as_config(input.build.l_build())?.build()?,
        input.inmem_parameters(max_points, data.ncols()),
        common::TableBasedDeletes,
    )?;

    build::set_start_points(
        index.provider(),
        data.as_view(),
        *input.build.start_point_strategy(),
    )?;

    let num_threads_and_tasks = NonZeroUsize::new(input.build.num_threads()).unwrap();
    let managed_stream = FullPrecisionStream {
        index,
        search: topk.clone(),
        runtime: benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
        ntasks: num_threads_and_tasks,
        inplace_delete_num_to_replace: input.runbook_params.ip_delete_num_to_replace,
        inplace_delete_method: input.runbook_params.ip_delete_method.into(),
    };

    let managed = Managed::new(
        max_points,
        managed::SlotReclaim::Deferred(consolidate_threshold),
        managed_stream,
    );

    // compute the maximum value of k used in any search
    let max_k = topk.max_k();

    let layered = bigann::WithData::new(managed, data, queries, move |path| {
        Ok(Box::new(datafiles::load_groundtruth(
            datafiles::BinFile(path),
            Some(max_k),
        )?))
    });

    Ok(layered)
}
