/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use core::option::Option::None;
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
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::datatype,
    Any, Checkpoint,
};
use diskann_inmem::diskann_async;
use diskann_providers::model::{configuration::IndexConfiguration, graph::provider::async_::common};
use diskann_utils::{
    future::AsyncFriendly,
    sampling::WithApproximateNorm,
    views::{Matrix, MatrixView},
};
use half::f16;
use serde::Serialize;

use super::{
    build::{self, load_index, save_index, single_or_multi_insert, BuildStats},
    product, scalar, search, spherical,
};
use crate::{
    backend::index::{
        result::{AggregatedSearchResults, BuildResult},
        streaming::{self, managed, stats::StreamStats, FullPrecisionStream, Managed},
    },
    inputs::async_::{DynamicIndexRun, IndexBuild, IndexOperation, IndexSource, SearchPhase},
    utils::{
        self,
        datafiles::{self},
        filters::{generate_bitmaps, setup_filter_strategies},
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

#[cfg(any(feature = "product-quantization", feature = "scalar-quantization"))]
pub(super) use register;

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

            let knn = benchmark_core::search::graph::KNN::new(
                index,
                queries,
                benchmark_core::search::graph::Strategy::broadcast(search_strategy),
            )?;

            let steps = search::knn::SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let search_results = search::knn::run(&knn, &groundtruth, steps)?;
            result.append(AggregatedSearchResults::Topk(search_results));
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

            let steps = search::range::RangeSearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let range = benchmark_core::search::graph::Range::new(
                index,
                queries,
                benchmark_core::search::graph::Strategy::broadcast(search_strategy),
            )?;

            let search_results = search::range::run(&range, &groundtruth, steps)?;
            result.append(AggregatedSearchResults::Range(search_results));
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

            let bit_maps =
                generate_bitmaps(&search_phase.query_predicates, &search_phase.data_labels)?;

            let search_strategies = setup_filter_strategies(
                search_phase.beta,
                bit_maps
                    .into_iter()
                    .map(utils::filters::as_query_label_provider),
                search_strategy.clone(),
            );

            let knn = benchmark_core::search::graph::KNN::new(
                index,
                queries,
                benchmark_core::search::graph::Strategy::collection(search_strategies),
            )?;

            let steps = search::knn::SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let search_results = search::knn::run(&knn, &groundtruth, steps)?;
            result.append(AggregatedSearchResults::Topk(search_results));
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

            let steps = search::knn::SearchSteps::new(
                search_phase.reps,
                &search_phase.num_threads,
                &search_phase.runs,
            );

            let bit_maps =
                generate_bitmaps(&search_phase.query_predicates, &search_phase.data_labels)?;

            let multihop = benchmark_core::search::graph::MultiHop::new(
                index,
                queries,
                benchmark_core::search::graph::Strategy::broadcast(search_strategy),
                bit_maps
                    .into_iter()
                    .map(utils::filters::as_query_label_provider)
                    .collect(),
            )?;

            let search_results = search::knn::run(&multihop, &groundtruth, steps)?;
            result.append(AggregatedSearchResults::Topk(search_results));
            Ok(result)
        }
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
            type Data = Vec<managed::Stats<StreamStats>>;
            fn run(
                self,
                _checkpoint: Checkpoint<'_>,
                mut output: &mut dyn Output,
            ) -> Result<Self::Data, anyhow::Error> {
                writeln!(output, "{}", self.input)?;

                let groundtruth_directory = self
                    .input
                    .runbook_params
                    .resolved_gt_directory
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Ground truth directory path was not resolved during validation"
                        )
                    })?;

                let mut runbook = bigann::RunBook::load(
                    &self.input.runbook_params.runbook_path,
                    &self.input.runbook_params.dataset_name,
                    &mut bigann::ScanDirectory::new(groundtruth_directory)?,
                )?;

                let mut streamer = full_precision_streaming(&self, runbook.max_points())?;

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
    };
}

impl_dynamic_run!(f32);
impl_dynamic_run!(f16);
impl_dynamic_run!(u8);
impl_dynamic_run!(i8);

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
    config: &DynamicFullPrecision<'_, T>,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, StreamStats>>>
where
    T: bytemuck::Pod + VectorRepr + WithApproximateNorm + SampleableForStart,
{
    let topk = match &config.input.search_phase {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };
    let consolidate_threshold: f32 = config.input.runbook_params.consolidate_threshold;

    let data = datafiles::load_dataset::<T>(datafiles::BinFile(&config.input.build.data))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &topk.queries,
    ))?);

    // Create a little extra headroom to handle deferred maintenance.
    let max_points = ((max_points as f32) * (1.0 + 2.0 * consolidate_threshold)).ceil() as usize;

    let index = diskann_async::new_index::<T, _>(
        config
            .input
            .try_as_config(config.input.build.l_build)?
            .build()?,
        config.input.inmem_parameters(max_points, data.ncols()),
        common::TableBasedDeletes,
    )?;

    build::set_start_points(
        index.provider(),
        data.as_view(),
        config.input.build.start_point_strategy,
    )?;

    let num_threads_and_tasks = NonZeroUsize::new(config.input.build.num_threads).unwrap();
    let managed_stream = FullPrecisionStream {
        index,
        search: topk.clone(),
        runtime: benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
        ntasks: num_threads_and_tasks,
        inplace_delete_num_to_replace: config.input.runbook_params.ip_delete_num_to_replace,
        inplace_delete_method: config.input.runbook_params.ip_delete_method.into(),
    };

    let managed = Managed::new(max_points, consolidate_threshold, managed_stream);

    let layered = bigann::WithData::new(managed, data, queries, |path| {
        Ok(Box::new(datafiles::load_groundtruth(datafiles::BinFile(
            path,
        ))?))
    });

    Ok(layered)
}
