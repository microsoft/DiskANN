/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use core::option::Option::None;
use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{glue, index::QueryLabelProvider, DiskANNIndex},
    graph::{SampleableForStart, StartPointStrategy},
    provider::{self, DataProvider, DefaultContext},
};
use diskann_benchmark_runner::{
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::datatype,
    Any, Checkpoint,
};
use diskann_providers::{
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::common,
    },
};
use diskann_inmem::{diskann_async, DefaultProvider};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
use half::f16;
use serde::Serialize;

use super::{
    build::{self, build_multi_insert, build_single_insert, load_index, save_index},
    multihop_filtered_search::run_multihop_search,
    product,
    range_search::{run_range_search, RangeSearchSteps},
    scalar,
    search::{self, run_search, SearchSteps},
    spherical,
    update::run_update,
};
use crate::{
    backend::index::result::{AggregatedSearchResults, BuildResult, BuildStats, DynamicRunResult},
    inputs::async_::{DynamicIndexRun, IndexBuild, IndexOperation, IndexSource, SearchPhase},
    utils::{
        self,
        datafiles::{self, UpdateOperationType},
        filters::{generate_bitmaps, setup_filter_strategies},
        streaming::{DynamicConfig, TagSlotManager},
    },
};

////////////////////////////
// Benchmark Registration //
////////////////////////////

macro_rules! register {
    ($disp:ident, $name:literal, $bench_type:ty) => {
        $disp.register::<$bench_type>($name, |object, checkpoint, output| {
            match <_ as $crate::backend::index::benchmarks::BuildAndSearch>::run(
                object, checkpoint, output,
            ) {
                Ok(v) => Ok(serde_json::to_value(v)?),
                Err(err) => Err(err),
            }
        });
    };
}
macro_rules! register_streaming {
    ($disp:ident, $name:literal, $bench_type:ty) => {
        $disp.register::<$bench_type>($name, |object, checkpoint, output| {
            match <_ as $crate::backend::index::benchmarks::BuildAndDynamicRun>::run(
                object, checkpoint, output,
            ) {
                Ok(v) => Ok(serde_json::to_value(v)?),
                Err(err) => Err(err),
            }
        });
    };
}

pub(super) use register;
// pub(super) use register_range;
pub(super) use register_streaming;

pub(super) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    // Full Precision
    register!(
        benchmarks,
        "async-full-precision-f32",
        FullPrecision<'static, f32>
    );
    register!(
        benchmarks,
        "async-full-precision-f16",
        FullPrecision<'static, f16>
    );
    register!(
        benchmarks,
        "async-full-precision-u8",
        FullPrecision<'static, u8>
    );
    register!(
        benchmarks,
        "async-full-precision-i8",
        FullPrecision<'static, i8>
    );

    // Dynamic Full Precision
    register_streaming!(
        benchmarks,
        "async-dynamic-full-precision-f32",
        DynamicFullPrecision<'static, f32>
    );
    register_streaming!(
        benchmarks,
        "async-dynamic-full-precision-f16",
        DynamicFullPrecision<'static, f16>
    );
    register_streaming!(
        benchmarks,
        "async-dynamic-full-precision-u8",
        DynamicFullPrecision<'static, u8>
    );
    register_streaming!(
        benchmarks,
        "async-dynamic-full-precision-i8",
        DynamicFullPrecision<'static, i8>
    );

    product::register_benchmarks(benchmarks);
    scalar::register_benchmarks(benchmarks);
    spherical::register_benchmarks(benchmarks);
}

//////////////
// Dispatch //
//////////////

pub(super) trait BuildAndSearch<'a> {
    /// The telemetry associated with the build and search.
    type Data: Serialize;

    /// Run the job, returning either the completed data or an error.
    fn run(
        self,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> Result<Self::Data, anyhow::Error>;
}

pub(super) trait BuildAndDynamicRun<'a> {
    /// The telemetry associated with the build and dynamic run.
    type Data: Serialize;

    /// Run the runbook, returning either the completed data or an error.
    fn run(
        self,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> Result<Self::Data, anyhow::Error>;
}

// Full Precision
pub(super) struct FullPrecision<'a, T> {
    input: &'a IndexOperation,
    _type: std::marker::PhantomData<T>,
}

impl<'a, T> FullPrecision<'a, T> {
    fn new(input: &'a IndexOperation) -> Self {
        Self {
            input,
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> diskann_benchmark_runner::dispatcher::Map for FullPrecision<'static, T>
where
    T: 'static,
{
    type Type<'a> = FullPrecision<'a, T>;
}

/// Dispatch to a full-precision only build.
impl<'a, T> DispatchRule<&'a IndexOperation> for FullPrecision<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = std::convert::Infallible;

    // Matching simply requires that we match the inner type.
    fn try_match(from: &&'a IndexOperation) -> Result<MatchScore, FailureScore> {
        match &from.source {
            IndexSource::Load(load) => datatype::Type::<T>::try_match(&load.data_type),
            IndexSource::Build(build) => datatype::Type::<T>::try_match(&build.data_type),
        }
    }

    fn convert(from: &'a IndexOperation) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a IndexOperation>,
    ) -> std::fmt::Result {
        // At this level, we only care about the data type, so return that description.
        match from {
            Some(arg) => match &arg.source {
                IndexSource::Load(load) => {
                    datatype::Type::<T>::description(f, Some(&load.data_type))
                }
                IndexSource::Build(build) => {
                    datatype::Type::<T>::description(f, Some(&build.data_type))
                }
            },
            None => datatype::Type::<T>::description(f, None::<&datatype::DataType>),
        }
    }
}

/// Central Dispatch
impl<'a, T> DispatchRule<&'a Any> for FullPrecision<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<IndexOperation, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<IndexOperation, Self>()
    }

    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<IndexOperation, Self>(f, from, IndexOperation::tag())
    }
}

// Async Dynamic Run
pub(super) struct DynamicFullPrecision<'a, T> {
    input: &'a DynamicIndexRun,
    _type: std::marker::PhantomData<T>,
}

impl<'a, T> DynamicFullPrecision<'a, T> {
    fn new(input: &'a DynamicIndexRun) -> Self {
        Self {
            input,
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> diskann_benchmark_runner::dispatcher::Map for DynamicFullPrecision<'static, T>
where
    T: 'static,
{
    type Type<'a> = DynamicFullPrecision<'a, T>;
}

/// Dispatch to a dynamic full-precision async index run.
impl<'a, T> DispatchRule<&'a DynamicIndexRun> for DynamicFullPrecision<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = std::convert::Infallible;
    // Matching simply requires that we match the inner type.
    fn try_match(from: &&'a DynamicIndexRun) -> Result<MatchScore, FailureScore> {
        datatype::Type::<T>::try_match(&from.build.data_type)
    }
    fn convert(from: &'a DynamicIndexRun) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }
    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a DynamicIndexRun>,
    ) -> std::fmt::Result {
        // At this level, we only care about the data type, so return that description.
        datatype::Type::<T>::description(f, from.map(|f| f.build.data_type).as_ref())
    }
}

/// Central Dispatch
impl<'a, T> DispatchRule<&'a Any> for DynamicFullPrecision<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = anyhow::Error;
    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<DynamicIndexRun, Self>()
    }
    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<DynamicIndexRun, Self>()
    }
    fn description(f: &mut std::fmt::Formatter<'_>, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<DynamicIndexRun, Self>(f, from, DynamicIndexRun::tag())
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
        + provider::SetElement<[T]>,
    CF: FnOnce(MatrixView<T>) -> anyhow::Result<Arc<DiskANNIndex<DP>>>,
    T: diskann::graph::SampleableForStart + std::fmt::Debug + Copy + AsyncFriendly + bytemuck::Pod,
    B: glue::SearchStrategy<DP, [T]> + Clone + Send + Sync,
    BF: AsyncFnOnce(
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

    let build_stats = {
        let rt = utils::tokio::runtime(input.num_threads)?;
        rt.block_on(build(
            index.clone(),
            build_strategy.clone(),
            data,
            input,
            output,
        ))?
    };

    Ok((index, build_stats))
}

pub(super) fn run_search_outer<T, S, DP>(
    input: &SearchPhase,
    search_strategy: S,
    index: Index<DP>,
    build_stats: Option<BuildStats>,
    checkpoint: Checkpoint<'_>,
) -> anyhow::Result<BuildResult>
where
    DP: DataProvider<Context = DefaultContext, InternalId = u32, ExternalId = u32>
        + provider::SetElement<[T]>,
    T: SampleableForStart + std::fmt::Debug + Copy + AsyncFriendly + bytemuck::Pod,
    S: glue::SearchStrategy<DP, [T]> + Clone + AsyncFriendly,
{
    match &input {
        SearchPhase::Topk(search_phase) => {
            // Handle Topk search phase

            let mut result = BuildResult::new_topk(build_stats);

            // Save construction stats before running queries.
            checkpoint.checkpoint(&result)?;

            let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &search_phase.queries,
            ))?);

            let groundtruth =
                datafiles::load_groundtruth(datafiles::BinFile(&search_phase.groundtruth))?;

            let steps = SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let search_strategies = Arc::new(vec![search_strategy.clone(); queries.nrows()]);

            let search_stats = run_search(
                index,
                search_strategies.clone(),
                queries,
                groundtruth.as_view(),
                steps,
            )?;

            result.append(AggregatedSearchResults::Topk(search_stats));
            Ok(result)
        }
        SearchPhase::Range(search_phase) => {
            // Handle Range search phase

            let mut result = BuildResult::new_range(build_stats);

            // Save construction stats before running queries.
            checkpoint.checkpoint(&result)?;

            let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &search_phase.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&search_phase.groundtruth))?;

            let steps = RangeSearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let search_strategies = Arc::new(vec![search_strategy.clone(); queries.nrows()]);

            let search_stats = run_range_search(
                index,
                search_strategies.clone(),
                queries,
                groundtruth,
                steps,
            )?;

            result.append(AggregatedSearchResults::Range(search_stats));
            Ok(result)
        }
        SearchPhase::TopkBetaFilter(search_phase) => {
            // Handle Beta Filtered Topk search phase
            let mut result = BuildResult::new_topk(build_stats);

            // Save construction stats before running queries.
            checkpoint.checkpoint(&result)?;

            let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &search_phase.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&search_phase.groundtruth))?;

            let steps = SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let bit_maps =
                generate_bitmaps(&search_phase.query_predicates, &search_phase.data_labels)?;

            let search_strategies =
                setup_filter_strategies(search_phase.beta, bit_maps, search_strategy.clone())?;

            let search_stats = run_search(
                index,
                search_strategies.clone(),
                queries,
                groundtruth,
                steps,
            )?;

            result.append(AggregatedSearchResults::Topk(search_stats));
            Ok(result)
        }
        SearchPhase::TopkMultihopFilter(search_phase) => {
            // Handle MultiHop Topk search phase
            let mut result = BuildResult::new_topk(build_stats);

            // Save construction stats before running queries.
            checkpoint.checkpoint(&result)?;

            let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &search_phase.queries,
            ))?);

            let groundtruth =
                datafiles::load_range_groundtruth(datafiles::BinFile(&search_phase.groundtruth))?;

            let steps = SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let bit_maps =
                generate_bitmaps(&search_phase.query_predicates, &search_phase.data_labels)?;

            // let filter: Arc<dyn QueryLabelProvider<IdType>> = Arc::new(BitMapFilter(bitmap));

            let bit_map_filters = Arc::new(
                bit_maps
                    .into_iter()
                    .map(|bit_map| {
                        let filter: Arc<dyn QueryLabelProvider<u32>> =
                            Arc::new(utils::filters::BitmapFilter(bit_map));
                        filter
                    })
                    .collect::<Vec<_>>(),
            );

            let search_strategies = Arc::new(vec![search_strategy.clone(); queries.nrows()]);

            let search_stats = run_multihop_search(
                index,
                search_strategies.clone(),
                queries,
                groundtruth,
                steps,
                bit_map_filters,
            )?;

            result.append(AggregatedSearchResults::Topk(search_stats));
            Ok(result)
        }
    }
}

/// Dynamic index run
/// insert_l is used for organizing results here and not setting insert_l in index.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_dynamic<T, S, SI, D, CF, F, Q, C>(
    config: DynamicConfig<'_, S, SI, D>,
    data: Option<Arc<Matrix<T>>>,
    _checkpoint: Checkpoint<'_>,
    mut output: &mut dyn Output,
    create: CF,
) -> anyhow::Result<DynamicRunResult>
where
    DefaultProvider<F, Q, C>: DataProvider<Context = DefaultContext, InternalId = u32, ExternalId = u32>
        + provider::SetElement<[T]>
        + provider::Delete,
    CF: FnOnce(usize, MatrixView<T>) -> anyhow::Result<Arc<DiskANNIndex<DefaultProvider<F, Q, C>>>>,
    T: diskann::graph::SampleableForStart + std::fmt::Debug + Copy + AsyncFriendly + bytemuck::Pod,
    S: glue::SearchStrategy<DefaultProvider<F, Q, C>, [T]> + Clone + AsyncFriendly,
    SI: glue::InsertStrategy<DefaultProvider<F, Q, C>, [T]> + Clone + Send + Sync,
    D: glue::InplaceDeleteStrategy<DefaultProvider<F, Q, C>> + Clone + Send + Sync,
{
    let data = match data {
        Some(data) => data,
        None => Arc::new(datafiles::load_dataset(datafiles::BinFile(
            &config.input.build.data,
        ))?),
    };

    // Use the resolved gt_directory path from validation
    let gt_directory_path = config
        .input
        .runbook_params
        .resolved_gt_directory
        .as_ref()
        .ok_or_else(|| {
            anyhow::anyhow!("Ground truth directory path was not resolved during validation")
        })?;

    let runbook = datafiles::DynamicRunbook::new_from_runbook_file(
        datafiles::RunbookFile(&config.input.runbook_params.runbook_path),
        config.input.runbook_params.dataset_name.clone(),
        Some(&gt_directory_path.to_string_lossy()),
    )?;

    // Allow slack for the starting point and unconsolidated nodes.
    let max_capacity = config.max_capacity(runbook.max_pts);
    let index = create(max_capacity, data.as_view())?;

    // `slot` is the internal id of DiskANN for vectors
    // `tag` is the ID associated for a vector by a runbook
    // `id` is the offset in binary file of the vector

    // The notation `tag` is used  in runbook only for replace operations.
    // However, insert and delete operations are specified in terms of tags as well.
    // In a runbook without replace operations, tag is always equal to (external) id.
    let mut bookkeeping = TagSlotManager::new(max_capacity);

    match &config.input.search_phase {
        SearchPhase::Topk(search_phase) => {
            // There is no build phase, start with the runbook directly
            let queries: Arc<Matrix<T>> = Arc::new(datafiles::load_dataset(datafiles::BinFile(
                &search_phase.queries,
            ))?);

            let mut result = DynamicRunResult::new(
                config.num_update_threads,
                config.insert_l,
                config.input.runbook_params.ip_delete_method,
                NonZeroUsize::new(config.input.runbook_params.ip_delete_num_to_replace).unwrap(),
                config.input.runbook_params.consolidate_threshold,
            );
            for stage in runbook.phases {
                if stage.operation == UpdateOperationType::Search {
                    // If it is search, run search and append stats
                    writeln!(output, "Running SEARCH stage {}", stage)?;
                    // let search_phase = &search_phase;

                    // Get the ground truth file path from the stage
                    let gt_file_path = stage.gt_filepath.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "Ground truth file path not found for search stage {}",
                            stage.stage_idx
                        )
                    })?;

                    let strategies =
                        Arc::new(vec![config.search_strategy.clone(); queries.nrows()]);
                    let groundtruth =
                        datafiles::load_groundtruth(datafiles::BinFile(gt_file_path))?;
                    let search_stats = search::run_search_queue_based(
                        index.clone(),
                        strategies,
                        queries.clone(),
                        groundtruth.as_view(),
                        SearchSteps::new(
                            search_phase.reps,
                            &search_phase.num_threads,
                            &search_phase.runs,
                        ),
                        &mut |ids| search::translate_ids(ids, &bookkeeping.slot_to_tag),
                    )?;
                    result.append_search_results(search_stats, stage.stage_idx);
                } else {
                    // If it is update, run update and append stats
                    writeln!(output, "Running UPDATE stage {}", stage)?;
                    let stage_idx = stage.stage_idx; // Extract stage_idx before moving stage

                    let update_stats = run_update(
                        index.clone(),
                        &config,
                        stage,
                        Some(data.clone()),
                        max_capacity,
                        &mut bookkeeping,
                        output,
                    )?;
                    result.append_update_results(update_stats, stage_idx);
                }
            }

            result.aggregate_metrics();
            Ok(result)
        }
        SearchPhase::Range(_) => Err(anyhow::anyhow!(
            "Range search phase is not supported in dynamic index run."
        )),
        SearchPhase::TopkBetaFilter(_) => Err(anyhow::anyhow!(
            "Top-k Beta filtered search phase is not supported in dynamic index run."
        )),
        SearchPhase::TopkMultihopFilter(_) => Err(anyhow::anyhow!(
            "Multi-hop search phase is not supported in dynamic index run."
        )),
    }
}

pub(super) async fn single_or_multi_insert<DP, S, T>(
    index: Index<DP>,
    strategy: S,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    output: &mut dyn Output,
) -> anyhow::Result<BuildStats>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + Send + Sync,
    S::PruneStrategy: Clone,
    for<'a> glue::aliases::InsertPruneAccessor<'a, S, DP, [T]>: glue::AsElement<&'a [T]>,
    T: Send + Sync + 'static + SampleableForStart + std::fmt::Debug + Clone,
{
    match &input.multi_insert {
        None => build_single_insert(index, strategy, data, input.num_threads, output)
            .await
            .map(|stats| stats.into()),
        Some(multi_insert) => build_multi_insert(
            index,
            strategy,
            data,
            multi_insert.batch_size.into(),
            output,
        )
        .await
        .map(|stats| stats.into()),
    }
}

#[cfg(feature = "scalar-quantization")]
pub(super) async fn only_single_insert<DP, S, T>(
    index: Index<DP>,
    strategy: S,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    output: &mut dyn Output,
) -> anyhow::Result<BuildStats>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    S: glue::InsertStrategy<DP, [T]> + Clone + Send + Sync,
    T: AsyncFriendly + SampleableForStart + std::fmt::Debug + Clone,
{
    match &input.multi_insert {
        None => build_single_insert(index, strategy, data, input.num_threads, output)
            .await
            .map(|stats| stats.into()),
        Some(_) => Err(anyhow::anyhow!(
            "please file a bug report, this quantization does not \
                 support multi-insert and this should have been rejected \
                 by the benchmark front-end"
        )),
    }
}

macro_rules! impl_build {
    ($T:ty) => {
        impl<'a> BuildAndSearch<'a> for FullPrecision<'a, $T> {
            type Data = BuildResult;
            fn run(
                self,
                checkpoint: Checkpoint<'_>,
                mut output: &mut dyn Output,
            ) -> Result<Self::Data, anyhow::Error> {
                writeln!(output, "{}", self.input)?;
                let (index, build_stats) = match &self.input.source {
                    IndexSource::Build(build) => {
                        let (index, build_stats) = run_build(
                            &build,
                            common::FullPrecision,
                            None,
                            output,
                            |data| {
                                let index = diskann_async::new_index::<$T, _>(
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
                            utils::tokio::block_on(save_index(index.clone(), &save_path))?;
                        }

                        (index, Some(build_stats))
                    }
                    IndexSource::Load(load) => {
                        let index_config: &IndexConfiguration = &load.to_config()?;

                        let index = {
                            utils::tokio::block_on(load_index::<_>(&load.load_path, index_config))?
                        };

                        (Arc::new(index), None::<BuildStats>)
                    }
                };

                let result = run_search_outer(
                    &self.input.search_phase,
                    common::FullPrecision,
                    index,
                    build_stats,
                    checkpoint,
                )?;

                writeln!(output, "\n\n{}", result)?;
                Ok(result)
            }
        }
    };
}

impl_build!(f32);
impl_build!(f16);
impl_build!(u8);
impl_build!(i8);

macro_rules! impl_dynamic_run {
    ($T:ty) => {
        impl<'a> BuildAndDynamicRun<'a> for DynamicFullPrecision<'a, $T> {
            type Data = DynamicRunResult;
            fn run(
                self,
                checkpoint: Checkpoint<'_>,
                mut output: &mut dyn Output,
            ) -> Result<Self::Data, anyhow::Error> {
                writeln!(output, "{}", self.input)?;

                let insert_l_val = self.input.build.l_build;

                let config = DynamicConfig::new(
                    self.input,
                    common::FullPrecision, // search_strategy
                    common::FullPrecision, // insert_strategy
                    common::FullPrecision, // delete_strategy
                    NonZeroUsize::new(insert_l_val).unwrap(),
                    NonZeroUsize::new(self.input.build.num_threads).unwrap(),
                );

                let result = run_dynamic(config, None, checkpoint, output, {
                    move |max_capacity: usize, data| {
                        let index = diskann_async::new_index::<$T, _>(
                            self.input.try_as_config(insert_l_val)?.build()?,
                            self.input.inmem_parameters(max_capacity, data.ncols()),
                            common::TableBasedDeletes,
                        )?;
                        build::set_start_points(
                            index.provider(),
                            data,
                            StartPointStrategy::Medoid,
                        )?;
                        Ok(index)
                    }
                })?;
                writeln!(output, "\n\n{}", result)?;
                Ok(result)
            }
        }
    };
}

impl_dynamic_run!(f32);
impl_dynamic_run!(f16);
impl_dynamic_run!(u8);
impl_dynamic_run!(i8);
