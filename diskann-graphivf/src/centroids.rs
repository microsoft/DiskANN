/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory full-precision DiskANN graph over the cluster centroids.
//!
//! Thin wrappers around the `diskann` / `diskann-providers` in-memory index that
//! build a graph over a set of centroids and run k-NN over it.

use std::sync::Arc;

use diskann::{
    graph::{
        config::{Builder, MaxDegree},
        search::Knn,
        search_output_buffer::{IdDistance, SearchOutputBuffer},
        strategy::FullPrecision,
    },
    provider::DefaultContext,
    utils::ONE,
    ANNError,
};
use diskann_providers::{
    index::diskann_async::{new_index, MemoryIndex},
    model::graph::provider::async_::{
        common::NoDeletes,
        inmem::{DefaultProviderParameters, SetStartPoints},
    },
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric as VectorMetric;
use tokio::runtime::Runtime;

use crate::{params::GraphParams, Result};

/// Build an in-memory full-precision graph over `centroids` (row `i` is centroid
/// `i`, returned as internal/external id `i`).
///
/// The data is assumed to already be in the internal representation expected by
/// the chosen metric (e.g. L2-normalized for cosine), so the graph is always
/// built with squared-L2.
pub(crate) fn build(
    centroids: Matrix<f32>,
    graph: &GraphParams,
    num_threads: usize,
) -> Result<MemoryIndex<f32>> {
    let num_clusters = centroids.nrows();
    let dim = centroids.ncols();

    let config = Builder::new_with(
        graph.degree,
        MaxDegree::slack(graph.slack),
        graph.l_build,
        VectorMetric::L2.into(),
        |b| {
            b.alpha(graph.alpha);
        },
    )
    .build()
    .map_err(ANNError::from)?;

    let params = DefaultProviderParameters {
        max_points: num_clusters,
        frozen_points: ONE,
        dim,
        metric: VectorMetric::L2,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: config.max_degree_u32().get(),
    };

    let index = new_index::<f32, _>(config, params, NoDeletes)?;

    // Use the mean of all centroids as the graph's start point.
    let mut mean = vec![0.0f32; dim];
    for row in centroids.row_iter() {
        for (m, v) in mean.iter_mut().zip(row.iter()) {
            *m += *v;
        }
    }
    for m in mean.iter_mut() {
        *m /= num_clusters as f32;
    }
    index
        .provider()
        .set_start_points(std::iter::once(mean.as_slice()))?;

    let ids: Arc<[u32]> = (0..num_clusters as u32).collect();
    let batch = Arc::new(centroids);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()
        .map_err(ANNError::from)?;

    runtime.block_on(index.multi_insert::<_, Matrix<f32>>(
        FullPrecision,
        &DefaultContext,
        batch,
        ids,
    ))?;

    Ok(index)
}

/// Run a k-NN search over the centroid graph, writing the resulting centroid ids
/// and distances into `ids_out` / `dist_out`. Returns the number of results
/// written (`<= ids_out.len()`).
///
/// `ids_out` and `dist_out` must have the same length, which is the requested
/// `k`. `l` is the centroid-graph search-list size and must be `>= k`.
pub(crate) fn search(
    index: &MemoryIndex<f32>,
    runtime: &Runtime,
    query: &[f32],
    l: usize,
    ids_out: &mut [u32],
    dist_out: &mut [f32],
) -> Result<usize> {
    debug_assert_eq!(ids_out.len(), dist_out.len());
    let k = ids_out.len();
    let knn = Knn::new(k, l, None).map_err(ANNError::from)?;

    let mut buffer = IdDistance::new(ids_out, dist_out);
    runtime.block_on(index.search(knn, &FullPrecision, &DefaultContext, query, &mut buffer))?;

    Ok(buffer.current_len())
}
