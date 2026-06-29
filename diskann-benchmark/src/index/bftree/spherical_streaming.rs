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
    benchmark::{MatchContext, Score},
    output::Output,
    Benchmark, Checkpoint,
};
use diskann_bftree::BfTreeProvider;
use diskann_providers::model::graph::provider::async_::common::Quantized;

use crate::{
    index::streaming::{stats::StreamStats, BfTreeMaintainer, StreamRunner},
    inputs::bftree::{BfTreeStreamingRun, QuantConfig},
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
    type Output = Vec<StreamStats>;

    fn try_match(&self, input: &Self::Input, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        if !matches!(input.quantization(), QuantConfig::Spherical { .. }) {
            score.fail(
                1,
                &"Spherical-quantized index requires a spherical quantization config",
            );
        }
        utils::match_data_type::<f32>(&mut score, input.data_type());

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "- BfTree Streaming with spherical quantization")
    }

    fn run(
        &self,
        input: &BfTreeStreamingRun,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        crate::index::streaming::run_streaming::<f32, BfTreeSphericalStream, _>(
            input.runbook_params(),
            |max_points| bftree_sq_streaming_impl(input, max_points),
            output,
        )
    }
}

type BfTreeSphericalStream = StreamRunner<
    BfTreeProvider<f32, diskann_bftree::quant::QuantVectorProvider>,
    f32,
    Quantized,
    BfTreeMaintainer,
>;

fn bftree_sq_streaming_impl(
    input: &BfTreeStreamingRun,
    max_points: usize,
) -> anyhow::Result<bigann::WithData<f32, u32, BfTreeSphericalStream>> {
    let search = input.search();

    let num_start_points = input.build().start_point_strategy().count();
    let capacity = max_points + num_start_points;

    crate::index::streaming::build_direct_streamer(
        input.build().data(),
        search,
        capacity,
        |data, capacity| {
            let quantizer_poly = super::quantizer_util::build_quantizer(
                input.quantization(),
                data.as_view(),
                input.build().distance(),
            )?
            .expect("spherical quantization config guaranteed by try_match");

            let config = input.try_as_config()?.build()?;
            let params = input.bftree_parameters(capacity, data.ncols())?;
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
                search.clone(),
                benchmark_core::tokio::runtime(num_threads_and_tasks.get())?,
                num_threads_and_tasks,
                input.runbook_params().ip_delete_num_to_replace,
                input.runbook_params().ip_delete_method.into(),
                BfTreeMaintainer,
            ))
        },
    )
}
