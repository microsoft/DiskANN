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
use diskann_providers::{
    model::graph::provider::async_::common::Quantized,
    storage::{FileStorageProvider, SaveWith},
};

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

        let mut save_index = None;
        let stats = crate::index::streaming::run_streaming::<f32, BfTreeSphericalStream, _>(
            input.runbook_params(),
            |_max_points| {
                let (streamer, index) = bftree_sq_streaming_impl(input)?;
                save_index = Some(index);
                Ok(streamer)
            },
            output,
        )?;

        if let Some(save_path) = input.build().save_path() {
            let index = save_index.expect("streamer is constructed during run_streaming");
            crate::utils::tokio::block_on(
                index
                    .provider()
                    .save_with(&FileStorageProvider, &save_path.to_string()),
            )?;
        }

        Ok(stats)
    }
}

type BfTreeSphericalStream = StreamRunner<
    BfTreeProvider<f32, diskann_bftree::quant::QuantVectorProvider>,
    f32,
    Quantized,
    BfTreeMaintainer,
>;

type BfTreeSQIndex =
    Arc<DiskANNIndex<BfTreeProvider<f32, diskann_bftree::quant::QuantVectorProvider>>>;

type BfTreeStreamingPayload = (
    bigann::WithData<f32, u32, BfTreeSphericalStream>,
    BfTreeSQIndex,
);

fn bftree_sq_streaming_impl(input: &BfTreeStreamingRun) -> anyhow::Result<BfTreeStreamingPayload> {
    let search = input.search();

    let num_start_points = input.build().start_point_strategy().count();

    let mut index_handle = None;
    let streamer =
        crate::index::streaming::build_direct_streamer(input.build().data(), search, |data| {
            // The direct (non-Managed) path uses absolute runbook tag IDs as slot IDs,
            // so the provider must span the full dataset ID space rather than the
            // runbook's max concurrent point count.
            let capacity = data.nrows() + num_start_points;
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
            index_handle = Some(index.clone());

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
        })?;

    let index = index_handle.expect("build_direct_streamer runs the builder eagerly");
    Ok((streamer, index))
}
