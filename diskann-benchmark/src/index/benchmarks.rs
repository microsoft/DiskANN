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
        DynamicIndexRun, HybridBloomSearchPhase, IndexBuild, IndexBuildOnly, IndexOperation,
        IndexSource, MultihopFilterSearchPhase, SearchPhase,
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
            .search(plugins::TopkMultihopEncodedBitsliceDnf)
            .search(plugins::TopkInlineFilter)
            .search(plugins::DeterminantDiversity),
    )?;

    registry.register(
        "graph-index-full-precision-f16",
        FullPrecision::<f16>::new().search(plugins::Topk),
    )?;
    registry.register(
        "graph-index-full-precision-u8",
        FullPrecision::<u8>::new()
            .search(plugins::Topk)
            .search(plugins::TopkMultihopEncodedBitsliceDnf)
            .search(plugins::TopkHybridEncodedBloom),
    )?;
    registry.register(
        "graph-index-build-only-u8",
        FullPrecisionBuildOnly::<u8>::new(),
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

fn build_full_precision<T>(
    input: &IndexBuild,
    output: &mut dyn Output,
) -> anyhow::Result<(Index<FullPrecisionProvider<T>>, BuildStats)>
where
    T: VectorRepr + SampleableForStart + AsDataType,
{
    run_build(
        input,
        common::FullPrecision,
        None,
        output,
        |data| {
            let index = diskann_async::new_index::<T, _>(
                input.try_as_config()?.build()?,
                input.inmem_parameters(data.nrows(), data.ncols()),
                common::NoDeletes,
            )?;
            build::set_start_points(
                index.provider(),
                data.as_view(),
                *input.start_point_strategy(),
            )?;
            Ok(index)
        },
        single_or_multi_insert,
    )
}

struct FullPrecisionBuildOnly<T> {
    _element: std::marker::PhantomData<T>,
}

impl<T> FullPrecisionBuildOnly<T> {
    fn new() -> Self {
        Self {
            _element: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for FullPrecisionBuildOnly<T>
where
    T: VectorRepr + SampleableForStart + AsDataType,
{
    type Input = IndexBuildOnly;
    type Output = BuildStats;

    fn try_match(&self, input: &IndexBuildOnly, context: &MatchContext) -> Score {
        let mut score = context.success(0);
        utils::match_data_type::<T>(&mut score, input.build.data_type());
        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Build and save a full-precision {} memory graph",
            T::DATA_TYPE
        )
    }

    fn run(
        &self,
        input: &IndexBuildOnly,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<BuildStats> {
        writeln!(output, "{}", input)?;
        let (index, stats) = build_full_precision::<T>(&input.build, output)?;
        let save_path = input.build.save_path().ok_or_else(|| {
            anyhow::anyhow!("validated graph-index-build-only input has no save_path")
        })?;
        utils::tokio::block_on(save_index(index, save_path))?;
        checkpoint.checkpoint(&stats)?;
        writeln!(output, "\n{}\nSaved graph at {}", stats, save_path)?;
        Ok(stats)
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
                let (index, build_stats) = build_full_precision::<T>(build, output)?;

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

fn run_with_query_results<S, F>(
    make_runner: F,
    groundtruth: &dyn benchmark_core::recall::Rows<u32>,
    steps: search::knn::SearchSteps<'_>,
    query_results_path: Option<&str>,
) -> anyhow::Result<Vec<crate::index::result::SearchResults>>
where
    S: benchmark_core::search::Search<
        Id = u32,
        Parameters = benchmark_core::search::graph::KnnParams,
        Output = benchmark_core::search::graph::knn::Metrics,
    >,
    F: FnMut() -> anyhow::Result<Arc<S>>,
{
    let mut query_results = query_results_path
        .map(|path| {
            let partial = std::path::Path::new(path).with_extension("jsonl.part");
            std::fs::File::create(&partial)
                .map(|file| (path, partial, std::io::BufWriter::new(file)))
        })
        .transpose()?;
    let result = search::knn::run_fresh_multihop(
        make_runner,
        groundtruth,
        steps,
        &mut |l, threads, results| {
            if let Some((_, _, writer)) = query_results.as_mut() {
                use std::io::Write;
                let ids = results.ids().as_rows();
                for query_id in 0..ids.nrows() {
                    let metrics = &results.output()[query_id];
                    let mut row = serde_json::json!({
                        "search_l": l,
                        "threads": threads,
                        "query_id": query_id,
                        "ids": ids.row(query_id),
                        "latency_us": results.latencies()[query_id].as_micros(),
                        "comparisons": metrics.comparisons,
                        "hops": metrics.hops,
                    });
                    if let Some(timings) = metrics.hybrid_timings {
                        let micros = |duration: std::time::Duration| {
                            u64::try_from(duration.as_micros())
                                .map_err(|_| anyhow::anyhow!("hybrid phase time exceeds u64"))
                        };
                        row["hybrid_phase_us"] = serde_json::json!({
                            "candidate_scan": micros(timings.candidate_scan)?,
                            "distance_evaluation": micros(timings.distance_evaluation)?,
                            "topk_selection": micros(timings.topk_selection)?,
                            "post_processing": micros(timings.post_processing)?,
                            "candidate_count": timings.candidate_count,
                        });
                    }
                    serde_json::to_writer(&mut *writer, &row)?;
                    writer.write_all(b"\n")?;
                }
            }
            Ok(())
        },
    )?;
    if let Some((path, partial, mut writer)) = query_results {
        use std::io::Write;
        writer.flush()?;
        drop(writer);
        std::fs::rename(partial, path)?;
    }
    Ok(result)
}

fn run_multihop_encoded<DP, S>(
    index: Arc<DiskANNIndex<DP>>,
    phase: &MultihopFilterSearchPhase,
    strategy: &Strategy<S>,
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

    // `data_labels` points at a persisted Bitslice or Bloom index, not raw labels JSONL.
    // Loading the index and parsing/validating DNF predicates stay outside timing. Each timed
    // repetition/search-L rebuilds fresh lazy providers so the first `is_match` includes label-ID
    // compilation.
    let label_index = utils::filters::load_encoded_label_index(&phase.data_labels)?;
    let query_sources = utils::filters::prepare_encoded_query_sources(
        label_index.as_ref(),
        &phase.query_predicates,
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

    let result = run_with_query_results(
        make_multihop,
        &groundtruth,
        steps,
        phase.query_results_path.as_deref(),
    )?;
    Ok(AggregatedSearchResults::Topk(result))
}

fn run_hybrid_encoded(
    index: Arc<DiskANNIndex<FullPrecisionProvider<u8>>>,
    phase: &HybridBloomSearchPhase,
    strategy: &Strategy<common::FullPrecision>,
) -> anyhow::Result<AggregatedSearchResults> {
    let phase_base = &phase.multihop;
    let queries = Arc::new(datafiles::load_dataset::<u8>(datafiles::BinFile(
        &phase_base.queries,
    ))?);
    let groundtruth =
        datafiles::load_range_groundtruth(datafiles::BinFile(&phase_base.groundtruth))?;
    let steps = search::knn::SearchSteps::new(
        phase_base.reps,
        &phase_base.num_threads,
        &phase_base.runs,
        GroundTruthMode::Flexible,
    );

    let label_index = utils::filters::load_encoded_label_index(&phase_base.data_labels)?;
    anyhow::ensure!(
        index.provider().capacity() == label_index.num_vectors() as usize,
        "Bloom vector count does not match the graph's base vectors"
    );
    anyhow::ensure!(
        queries.nrows() > 0,
        "hybrid search requires at least one query"
    );
    let accessor = inmem::FullAccessor::new(index.provider(), queries.row(0));
    anyhow::ensure!(
        inmem::GetFullPrecision::as_full_precision(&accessor).dim() == queries.ncols(),
        "hybrid queries do not match the graph's vector dimension"
    );
    let plans = utils::filters::prepare_flat_encoded_queries(
        label_index.as_ref(),
        &phase_base.query_predicates,
    )?;
    let make_hybrid = || {
        benchmark_core::search::graph::Hybrid::new(
            index.clone(),
            queries.clone(),
            benchmark_core::search::graph::Strategy::broadcast(strategy.inner()),
            plans.clone().into(),
            u64::from(phase.brute_force_threshold.get()),
        )
    };

    let result = run_with_query_results(
        make_hybrid,
        &groundtruth,
        steps,
        phase_base.query_results_path.as_deref(),
    )?;
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

        let results = if let Some(path) = topk.query_results_path.as_deref() {
            run_with_query_results(|| Ok(knn.clone()), &groundtruth, steps, Some(path))?
        } else {
            search::knn::run(&knn, &groundtruth, steps)?
        };
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
        )
    }
}

impl search::Plugin<FullPrecisionProvider<u8>, SearchPhase, Strategy<common::FullPrecision>>
    for plugins::TopkHybridEncodedBloom
{
    fn is_match(&self, phase: &SearchPhase) -> bool {
        Self::kind() == phase.kind()
    }

    fn kind(&self) -> &'static str {
        Self::kind().as_str()
    }

    fn run(
        &self,
        index: Arc<DiskANNIndex<FullPrecisionProvider<u8>>>,
        phase: &SearchPhase,
        strategy: &Strategy<common::FullPrecision>,
    ) -> anyhow::Result<AggregatedSearchResults> {
        run_hybrid_encoded(index, phase.as_topk_hybrid_encoded_bloom()?, strategy)
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
