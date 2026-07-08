/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{borrow::Cow, io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{DiskANNIndex, InplaceDeleteMethod, SampleableForStart},
    utils::{VectorRepr, ONE},
};
use diskann_benchmark_core::{self as benchmark_core, recall::Rows, streaming::executors::bigann};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{BfTreeProvider, NoStore};
use diskann_providers::{
    model::graph::provider::async_::common::FullPrecision,
    storage::{FileStorageProvider, SaveWith},
};
use diskann_utils::views::{Matrix, MatrixView};

use crate::{
    index::{
        build::{BuildKind, BuildStats},
        search::knn,
        streaming::{
            managed::{self, Managed},
            stats::{GenericStats, StreamStats},
            ManagedStream,
        },
    },
    inputs::{
        bftree::BfTreeDynamicRun,
        graph_index::{SearchPhase, TopkSearchPhase},
    },
    utils::{self, datafiles},
};

////////////////////////
// Streaming BfTree  //
////////////////////////

type BfTreeFPIndex<T> = Arc<DiskANNIndex<BfTreeProvider<T, NoStore>>>;

/// The bf_tree streaming index implementation.
///
/// Mirrors the in-memory `FullPrecisionStream` but targets `BfTreeProvider`.
struct BfTreeStream<T>
where
    T: VectorRepr,
{
    index: BfTreeFPIndex<T>,
    search: TopkSearchPhase,
    runtime: tokio::runtime::Runtime,
    ntasks: NonZeroUsize,
    inplace_delete_num_to_replace: usize,
    inplace_delete_method: InplaceDeleteMethod,
}

impl<T> BfTreeStream<T>
where
    T: VectorRepr,
{
    fn insert_(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<BuildStats> {
        let runner = benchmark_core::build::graph::SingleInsert::new(
            self.index.clone(),
            Arc::new(data.to_owned()),
            FullPrecision,
            benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = benchmark_core::build::build(
            runner,
            benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        BuildStats::new(BuildKind::SingleInsert, results)
    }
}

impl<T> ManagedStream<T> for BfTreeStream<T>
where
    T: VectorRepr,
{
    type Output = StreamStats;

    fn search(
        &self,
        queries: Arc<Matrix<T>>,
        groundtruth: &dyn Rows<u32>,
    ) -> anyhow::Result<Self::Output> {
        let knn = benchmark_core::search::graph::KNN::new(
            self.index.clone(),
            queries,
            benchmark_core::search::graph::Strategy::broadcast(FullPrecision),
        )?;

        let steps = knn::SearchSteps::new(
            self.search.reps,
            &self.search.num_threads,
            &self.search.runs,
        );
        let results = knn::run(&knn, groundtruth, steps)?;
        Ok(StreamStats::Search(results))
    }

    fn insert(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Insert(self.insert_(data, slots)?))
    }

    fn replace(&self, data: MatrixView<'_, T>, slots: &[u32]) -> anyhow::Result<Self::Output> {
        Ok(StreamStats::Replace(self.insert_(data, slots)?))
    }

    fn delete(&self, slots: &[u32]) -> anyhow::Result<Self::Output> {
        let runner = benchmark_core::streaming::graph::InplaceDelete::new(
            self.index.clone(),
            FullPrecision,
            self.inplace_delete_num_to_replace,
            self.inplace_delete_method,
            benchmark_core::build::ids::Slice::new(slots.into()),
        );

        let results = benchmark_core::build::build(
            runner,
            benchmark_core::build::Parallelism::fixed(Some(ONE), self.ntasks),
            &self.runtime,
        )?;

        Ok(StreamStats::Delete(GenericStats::new(
            Cow::Borrowed("Delete"),
            results,
        )?))
    }

    fn maintain(&self) -> anyhow::Result<Self::Output> {
        // bf-tree uses hard deletes — no deferred cleanup needed.
        Ok(StreamStats::Maintain(Vec::new()))
    }
}

/// The dynamic/streaming benchmark for bf_tree full precision.
pub(super) struct StreamingFullPrecision<T> {
    _type: std::marker::PhantomData<T>,
}

impl<T> StreamingFullPrecision<T> {
    pub(super) fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for StreamingFullPrecision<T>
where
    T: VectorRepr + SampleableForStart + AsDataType + bytemuck::Pod,
{
    type Input = BfTreeDynamicRun;
    type Output = Vec<managed::Stats<StreamStats>>;

    fn try_match(&self, input: &Self::Input) -> Result<MatchScore, FailureScore> {
        let mut failure_score: Option<u32> = None;

        if let Err(s) = utils::match_data_type::<T>(input.data_type()) {
            failure_score = Some(s.0);
        }

        if !matches!(input.search_phase(), SearchPhase::Topk(_)) {
            *failure_score.get_or_insert(0) += 1;
        }

        match failure_score {
            None => Ok(MatchScore(0)),
            Some(score) => Err(FailureScore(score)),
        }
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Self::Input>,
    ) -> std::fmt::Result {
        match input {
            Some(i) => write!(f, "{}", T::describe(i.build().data_type())),
            None => write!(f, "{}", T::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        let mut index_for_save: Option<BfTreeFPIndex<T>> = None;

        let results = super::streaming_utils::run_streaming::<T, _>(
            input.runbook_params(),
            |max_points| {
                let (streamer, index) = bftree_streaming::<T>(input, max_points)?;
                index_for_save = Some(index);
                Ok(streamer)
            },
            output,
        )?;

        // save the index if requested
        if let Some(save_path) = input.build().save_path() {
            let index = index_for_save.expect("index should have been set by make_streamer");
            crate::utils::tokio::block_on(
                index
                    .provider()
                    .save_with(&FileStorageProvider, &save_path.to_string()),
            )?;
        }

        Ok(results)
    }
}

fn bftree_streaming<T>(
    input: &BfTreeDynamicRun,
    max_points: usize,
) -> anyhow::Result<(
    bigann::WithData<T, u32, Managed<T, StreamStats>>,
    BfTreeFPIndex<T>,
)>
where
    T: bytemuck::Pod + VectorRepr + SampleableForStart,
{
    let topk = match &input.search_phase() {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };

    let data = datafiles::load_dataset::<T>(datafiles::BinFile(input.build().data()))?;
    let queries = Arc::new(datafiles::load_dataset::<T>(datafiles::BinFile(
        &topk.queries,
    ))?);

    let config = input.try_as_config()?.build()?;
    let params = input.bftree_parameters(max_points, data.ncols())?;
    let start_points = input
        .build()
        .start_point_strategy()
        .compute(data.as_view())?;
    let provider = BfTreeProvider::new(params, start_points.as_view(), NoStore)?;
    let index = Arc::new(DiskANNIndex::new(config, provider, None));
    let index_handle = index.clone();

    let num_threads_and_tasks = NonZeroUsize::new(input.build().num_threads()).unwrap();
    let managed_stream = BfTreeStream {
        index,
        search: topk.clone(),
        runtime: benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
        ntasks: num_threads_and_tasks,
        inplace_delete_num_to_replace: input.runbook_params().ip_delete_num_to_replace,
        inplace_delete_method: input.runbook_params().ip_delete_method.into(),
    };

    let num_start_points = input.build().start_point_strategy().count();
    let managed = Managed::new(
        max_points + num_start_points,
        managed::SlotReclaim::Immediate,
        managed_stream,
    );

    let max_k = topk.max_k();
    let layered = bigann::WithData::new(managed, data, queries, move |path| {
        Ok(Box::new(datafiles::load_groundtruth(
            datafiles::BinFile(path),
            Some(max_k),
        )?))
    });

    Ok((layered, index_handle))
}
