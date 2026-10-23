/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

pub(in crate::index::inmem2) mod bigann;
pub(in crate::index::inmem2) mod build;
pub(in crate::index::inmem2) mod knn;

/// Measurements shared by the static build-and-search benchmarks.
///
/// Build statistics cover insertion, excluding data loading and quantizer training.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(in crate::index::inmem2) struct BuildAndSearchStats {
    pub(in crate::index::inmem2) build: build::Stats,
    pub(in crate::index::inmem2) search: knn::Stats,
}

/// Run the common initialization sequence for quantized backends.
pub(in crate::index::inmem2) fn quantized(
    backend: di::repr::quantized::Backend,
    data: rowmajor::Ref<'_, f32>,
    max_degree: usize,
    distance: Metric,
) -> anyhow::Result<di::Provider<di::repr::Quantized, u32>> {
    let start = diskann::graph::StartPointStrategy::Medoid.compute(data)?;
    let config = di::repr::Quantized::config(
        backend,
        di::num::Capacity::new(data.nrows()),
        di::num::MaxDegree::new(max_degree),
        start,
        di::repr::quantized::Rerank::F16 {
            metric: distance.into(),
        },
    )?;

    Ok(di::Provider::new(config)?)
}
