/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_providers::common::MinMax8;
use half::f16;

/// Associates a graph-IVF stored element type with the `data_type` tag used in job configs.
///
/// [`diskann_benchmark_runner::utils::datatype::AsDataType`] would be the natural home for
/// this mapping, but it cannot cover [`MinMax8`]: that trait is defined in
/// `diskann-benchmark-runner`, which deliberately depends on no `diskann-*` crate, while
/// `MinMax8` is defined in `diskann-providers`. Both are foreign to this crate, so the
/// orphan rule blocks the impl. Graph-IVF therefore carries its own mapping, which also
/// keeps the set of element types it accepts explicit.
pub(super) trait GraphIvfElement: VectorRepr {
    /// The `data_type` tag naming this element type in a job config.
    const DATA_TYPE: DataType;

    /// Whether a corpus of this type must reach graph-IVF still encoded as `T`.
    ///
    /// Native element types are encoded from `f32` element-wise while the inverted lists
    /// are written, so the corpus can be widened on load. Quantized composites cannot be:
    /// their rows carry per-vector metadata that element-wise encoding has no way to
    /// produce. Those corpora go through
    /// [`GraphIvfIndex::build_compressed_profiled`](diskann_graphivf::GraphIvfIndex::build_compressed_profiled)
    /// instead, which stores the original rows verbatim and decodes a separate `f32` copy
    /// for clustering.
    const STORED_VERBATIM: bool;
}

macro_rules! impl_graph_ivf_element {
    ($($ty:ty => $data_type:expr, $stored_verbatim:expr),* $(,)?) => {
        $(
            impl GraphIvfElement for $ty {
                const DATA_TYPE: DataType = $data_type;
                const STORED_VERBATIM: bool = $stored_verbatim;
            }
        )*
    };
}

impl_graph_ivf_element! {
    f32 => DataType::Float32, false,
    f16 => DataType::Float16, false,
    u8 => DataType::UInt8, false,
    i8 => DataType::Int8, false,
    MinMax8 => DataType::MinMax8, true,
}
