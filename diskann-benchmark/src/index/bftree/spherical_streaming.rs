/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
////////////////////////
// Streaming BfTree SQ //
////////////////////////

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::graph::DiskANNIndex;
use diskann_benchmark_core as benchmark_core;
use diskann_benchmark_core::streaming::executors::bigann;
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::BfTreeProvider;
use diskann_providers::model::graph::provider::async_::common::Quantized;

use crate::{
    index::streaming::{
        managed::{self, Managed},
        stats::StreamStats,
        BfTreeMaintainer, StreamRunner,
    },
    inputs::{
        bftree::{BfTreeStreamingRun, QuantConfig},
        graph_index::SearchPhase,
    },
    utils,
};

/// The streaming benchmark for bf_tree spherical quantization.
///
/// Dispatches `num_bits` at runtime to avoid const-generic monomorphization.
pub(super) struct StreamingSpherical;

impl StreamingSpherical {
    pub(super) fn new() -> Self {
        Self
    }
}

impl Benchmark for StreamingSpherical {
    type Input = BfTreeStreamingRun;
    type Output = Vec<managed::Stats<StreamStats>>;

    fn try_match(&self, input: &Self::Input) -> Result<MatchScore, FailureScore> {
        let mut failure_score: Option<u32> = None;

        if !matches!(input.quantization(), QuantConfig::Spherical { .. }) {
            *failure_score.get_or_insert(0) += 1;
        }
        if let Err(s) = utils::match_data_type::<f32>(input.data_type()) {
            *failure_score.get_or_insert(0) += s.0;
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
        input: Option<&BfTreeStreamingRun>,
    ) -> std::fmt::Result {
        match input {
            None => {
                writeln!(f, "- BfTree Streaming with spherical quantization")
            }
            Some(input) => {
                if !f32::is_match(input.data_type()) {
                    writeln!(f, "- Only `float32` supported, got {}", input.data_type())?;
                }
                Ok(())
            }
        }
    }

    fn run(
        &self,
        input: &BfTreeStreamingRun,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        crate::index::streaming::run_streaming::<f32, _>(
            input.runbook_params(),
            |max_points| bftree_sq_streaming_impl(input, max_points),
            output,
        )
    }
}

fn bftree_sq_streaming_impl(
    input: &BfTreeStreamingRun,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<f32, u32, Managed<f32, StreamStats>>> {
    let topk = match input.search_phase() {
        SearchPhase::Topk(topk) => topk,
        _ => anyhow::bail!("Only TopK is currently supported by the streaming index"),
    };

    let num_start_points = input.build().start_point_strategy().count();
    let capacity = max_points + num_start_points;

    crate::index::streaming::build_streamer(
        input.build().data(),
        topk,
        crate::index::streaming::managed::SlotReclaim::Immediate,
        capacity,
        |data, capacity| {
            let quantizer_poly = super::quantizer_util::build_quantizer(
                input.quantization(),
                data.as_view(),
                input.build().distance(),
            )?
            .expect("spherical quantization config guaranteed by try_match");

            let config = input.try_as_config()?.build()?;
            let params = input.bftree_parameters(capacity, data.ncols());
            let start_points = input
                .build()
                .start_point_strategy()
                .compute(data.as_view())?;
            let provider = BfTreeProvider::new(params, start_points.as_view(), quantizer_poly)?;
            let index = Arc::new(DiskANNIndex::new(config, provider, None));

            let num_threads_and_tasks = NonZeroUsize::new(input.build().num_threads()).unwrap();
            Ok(StreamRunner::new(
                index,
                Quantized,
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
