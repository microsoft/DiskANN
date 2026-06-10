/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Bf_tree-backed benchmarks for DiskANN graph indices.
//!
//! Unlike the inmem provider, bf_tree uses a log-structured merge tree for storage,
//! allowing datasets larger than memory and supporting persistent save/load. This module
//! provides both full-precision and spherical-quantized variants, each with static (build
//! once, search) and streaming (insert/delete/search interleaved) modes.
//!
//! Registered tags:
//! - `graph-index-bftree-full-precision-f32` — static FP build + search
//! - `graph-index-bftree-stream-full-precision-f32` — streaming FP
//! - `graph-index-build-bftree-spherical-quantization` — static spherical (1/2/4-bit)
//! - `graph-index-stream-bftree-spherical-quantization` — streaming spherical (1/2/4-bit)

use crate::backend::index::search::plugins::Topk;
use diskann_benchmark_runner::Registry;

mod full_precision;
mod full_precision_streaming;
mod spherical;
mod spherical_streaming;
mod streaming_utils;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(
        "graph-index-bftree-full-precision-f32",
        full_precision::BfTreeFullPrecision::<f32>::new().search(Topk),
    )?;

    registry.register(
        "graph-index-bftree-stream-full-precision-f32",
        full_precision_streaming::StreamingFullPrecision::<f32>::new(),
    )?;

    registry.register(
        "graph-index-bftree-spherical-quantization",
        spherical::BfTreeSpherical::new().search(Topk),
    )?;

    registry.register(
        "graph-index-stream-bftree-spherical-quantization",
        spherical_streaming::StreamingSpherical::new(),
    )?;

    Ok(())
}
