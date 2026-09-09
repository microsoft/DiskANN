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

use diskann_benchmark_runner::Registry;

#[cfg(feature = "bftree")]
use crate::index::search::plugins::Topk;

// Avoid `cfg_if` as `rustfmt` doesn't reliably format through `cfg_if`.
#[cfg(feature = "bftree")]
mod full_precision;

#[cfg(feature = "bftree")]
mod full_precision_streaming;

#[cfg(feature = "bftree")]
mod spherical;

#[cfg(feature = "bftree")]
mod spherical_streaming;

#[cfg(feature = "bftree")]
mod streaming_utils;

#[cfg(feature = "bftree")]
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

#[cfg(not(feature = "bftree"))]
pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register_gated(
        "graph-index-build-bftree-full-precision",
        "graph-index-bftree-full-precision",
        diskann_benchmark_runner::Features::new("bftree"),
        "BFTree powered graph index build and search",
    )?;

    registry.register_gated(
        "graph-index-stream-bftree-full-precision",
        "graph-index-bftree-stream-full-precision",
        diskann_benchmark_runner::Features::new("bftree"),
        "BFTree powered graph index streaming",
    )?;

    registry.register_gated(
        "graph-index-build-bftree-spherical-quantization",
        "graph-index-bftree-spherical-quantization",
        diskann_benchmark_runner::Features::new("bftree"),
        "BFTree powered graph index build and search with spherical (RabitQ) quantization",
    )?;

    registry.register_gated(
        "graph-index-stream-bftree-spherical-quantization",
        "graph-index-stream-bftree-spherical-quantization",
        diskann_benchmark_runner::Features::new("bftree"),
        "BFTree powered graph index streaming with spherical (RabitQ) quantization",
    )?;

    Ok(())
}
