/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{
    graph::{DiskANNIndex, SampleableForStart},
    utils::VectorRepr,
};
use diskann_benchmark_core::{self as benchmark_core, streaming::executors::bigann};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{BfTreeProvider, NoStore};
use diskann_providers::model::graph::provider::async_::common::FullPrecision;
use diskann_utils::sampling::WithApproximateNorm;

use crate::{
    index::streaming::{
        managed::{self, Managed},
        runner::BfTreeMaintainer,
        stats::StreamStats,
        StreamRunner,
    },
    inputs::{
        bftree::BfTreeDynamicRun,
        graph_index::{
            InplaceDeleteMethod as InputDeleteMethod, SearchPhase,
        },
    },
    utils,
};

////////////////////////
// Streaming BfTree  //
////////////////////////

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
    T: VectorRepr + WithApproximateNorm + SampleableForStart + AsDataType + bytemuck::Pod,
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

        if matches!(
            input.runbook_params().ip_delete_method,
            InputDeleteMethod::VisitedAndTopK { .. }
        ) {
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
            Some(i) => {
                write!(f, "{}", T::describe(i.build().data_type()))?;
                if matches!(
                    i.runbook_params().ip_delete_method,
                    InputDeleteMethod::VisitedAndTopK { .. }
                ) {
                    write!(f, "\n- bf-tree does not support VisitedAndTopK delete method")?;
                }
                Ok(())
            }
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

        crate::index::streaming::run_streaming::<T, _>(
            input.runbook_params(),
            |max_points| bftree_streaming::<T>(input, max_points),
            output,
        )
    }
}

fn bftree_streaming<T>(
    input: &BfTreeDynamicRun,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<T, u32, Managed<T, StreamStats>>>
where
    T: bytemuck::Pod + VectorRepr + WithApproximateNorm + SampleableForStart,
{
    let topk = match &input.search_phase() {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };

    let consolidate_threshold: f32 = input.runbook_params().consolidate_threshold;
    let num_start_points = input.build().start_point_strategy().count();
    let capacity = max_points + num_start_points;

    crate::index::streaming::build_streamer(
        input.build().data(),
        topk,
        consolidate_threshold,
        capacity,
        |data, capacity| {
            let config = input.try_as_config()?.build()?;
            let params = input.bftree_parameters(capacity, data.ncols());
            let start_points = input
                .build()
                .start_point_strategy()
                .compute(data.as_view())?;
            let provider = BfTreeProvider::new(params, start_points.as_view(), NoStore)?;
            let index = Arc::new(DiskANNIndex::new(config, provider, None));

            let num_threads_and_tasks = NonZeroUsize::new(input.build().num_threads()).unwrap();
            Ok(StreamRunner::new(
                index,
                FullPrecision,
                topk.clone(),
                benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
                num_threads_and_tasks,
                input.runbook_params().ip_delete_num_to_replace,
                input.runbook_params().ip_delete_method.into(),
                BfTreeMaintainer,
            ))
        },
    )
}
