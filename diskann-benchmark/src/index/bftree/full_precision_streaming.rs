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
    benchmark::{MatchContext, Score},
    output::Output,
    utils::datatype::AsDataType,
    Benchmark, Checkpoint,
};
use diskann_bftree::{BfTreeProvider, NoStore};
use diskann_providers::{
    model::graph::provider::async_::common::FullPrecision,
    storage::{FileStorageProvider, SaveWith},
};
use diskann_utils::sampling::WithApproximateNorm;

use crate::{
    index::streaming::{runner::BfTreeMaintainer, stats::StreamStats, StreamRunner},
    inputs::bftree::{BfTreeStreamingRun, QuantConfig},
    utils,
};

////////////////////////
// Streaming BfTree  //
////////////////////////

/// The streaming benchmark for bf_tree full precision.
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
    type Input = BfTreeStreamingRun;
    type Output = Vec<StreamStats>;

    fn try_match(&self, input: &Self::Input, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        if !matches!(input.quantization(), QuantConfig::None) {
            score.fail(1, &"Full-precision index does not support quantization");
        }
        utils::match_data_type::<T>(&mut score, input.data_type());

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", T::DATA_TYPE)
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;

        let mut save_index = None;
        let stats = crate::index::streaming::run_streaming::<T, BfTreeFullPrecisionStream<T>, _>(
            input.runbook_params(),
            |_max_points| {
                let (streamer, index) = bftree_streaming::<T>(input)?;
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

type BfTreeFullPrecisionStream<T> =
    StreamRunner<BfTreeProvider<T, NoStore>, T, FullPrecision, BfTreeMaintainer>;

type BfTreeFPIndex<T> = Arc<DiskANNIndex<BfTreeProvider<T, NoStore>>>;

type BfTreeStreamingPayload<T> = (
    bigann::WithData<T, u32, BfTreeFullPrecisionStream<T>>,
    BfTreeFPIndex<T>,
);

fn bftree_streaming<T>(input: &BfTreeStreamingRun) -> anyhow::Result<BfTreeStreamingPayload<T>>
where
    T: bytemuck::Pod + VectorRepr + WithApproximateNorm + SampleableForStart,
{
    let search = input.search();

    let num_start_points = input.build().start_point_strategy().count();

    let mut index_handle = None;
    let streamer =
        crate::index::streaming::build_direct_streamer(input.build().data(), search, |data| {
            // The direct (non-Managed) path uses absolute runbook tag IDs as slot IDs,
            // so the provider must span the full dataset ID space rather than the
            // runbook's max concurrent point count.
            let capacity = data.nrows() + num_start_points;
            let config = input.try_as_config()?.build()?;
            let params = input.bftree_parameters(capacity, data.ncols())?;
            let start_points = input
                .build()
                .start_point_strategy()
                .compute(data.as_view())?;
            let provider = BfTreeProvider::new(params, start_points.as_view(), NoStore)?;
            let index = Arc::new(DiskANNIndex::new(config, provider, None));
            index_handle = Some(index.clone());

            let num_threads_and_tasks = NonZeroUsize::new(input.build().num_threads()).unwrap();
            Ok(StreamRunner::new(
                index,
                FullPrecision,
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
