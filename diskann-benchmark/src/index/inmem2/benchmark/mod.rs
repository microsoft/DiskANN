/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

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
