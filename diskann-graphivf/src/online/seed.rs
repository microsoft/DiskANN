/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Initial-centroid strategies for the online clusterer.

use diskann_providers::utils::create_thread_pool;
use diskann_utils::views::{Matrix, MatrixView};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    cluster,
    params::{EmptyClusterPolicy, OnlineParams},
    GraphIvfError, Result,
};

/// How the initial centroid set an online clusterer starts from is produced.
///
/// Experiments rarely start from an empty partition; the common case is
/// [`Warmup`](Self::Warmup), a light k-means over a prefix of the corpus.
/// [`Explicit`](Self::Explicit) passes an already-computed centroid matrix
/// through unchanged. New built-in strategies can be added without changing
/// the clusterer constructor.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SeedStrategy {
    /// Use a precomputed centroid matrix as-is (one row per centroid).
    Explicit(Matrix<f32>),
    /// Bootstrap by running a lightweight exact k-means over the first
    /// `warmup_points` points of the corpus, yielding `num_centroids`
    /// centroids refined for `iters` Lloyd iterations.
    ///
    /// `iters == 0` skips refinement and uses the sampled points directly.
    /// `warmup_points` is clamped to `[num_centroids, corpus_len]`.
    Warmup {
        /// Number of initial centroids to produce.
        num_centroids: usize,
        /// Number of leading corpus points to cluster.
        warmup_points: usize,
        /// Lloyd iterations for the warmup clustering.
        iters: usize,
    },
}

impl SeedStrategy {
    /// Resolve this strategy into a concrete centroid matrix over `points`.
    pub(super) fn resolve(
        self,
        points: MatrixView<'_, f32>,
        params: &OnlineParams,
    ) -> Result<Matrix<f32>> {
        match self {
            Self::Explicit(centroids) => Ok(centroids),
            Self::Warmup {
                num_centroids,
                warmup_points,
                iters,
            } => warmup_kmeans(points, num_centroids, warmup_points, iters, params),
        }
    }
}

/// Run a lightweight exact k-means over the first `warmup_points` corpus points
/// to bootstrap `num_centroids` initial centroids.
fn warmup_kmeans(
    points: MatrixView<'_, f32>,
    num_centroids: usize,
    warmup_points: usize,
    iters: usize,
    params: &OnlineParams,
) -> Result<Matrix<f32>> {
    let dim = points.ncols();
    let n = points.nrows();

    if num_centroids == 0 {
        return Err(GraphIvfError::invalid("num_centroids must be non-zero"));
    }
    if num_centroids > n {
        return Err(GraphIvfError::invalid(format!(
            "warmup num_centroids ({num_centroids}) exceeds corpus size ({n})"
        )));
    }
    let warmup_n = warmup_points.clamp(num_centroids, n);

    let mut window = vec![0.0f32; warmup_n * dim];
    for (dst, row) in window.chunks_mut(dim).zip(0..warmup_n) {
        dst.copy_from_slice(points.row(row));
    }
    let window = Matrix::try_from(window.into_boxed_slice(), warmup_n, dim)
        .map_err(|_| GraphIvfError::invalid("warmup window shape mismatch"))?;

    let mut rng = StdRng::seed_from_u64(params.seed);
    let sampled = rand::seq::index::sample(&mut rng, warmup_n, num_centroids).into_vec();
    let mut buffer = vec![0.0f32; num_centroids * dim];
    for (dst, &row) in buffer.chunks_mut(dim).zip(&sampled) {
        dst.copy_from_slice(window.row(row));
    }
    let mut centroids = Matrix::try_from(buffer.into_boxed_slice(), num_centroids, dim)
        .map_err(|_| GraphIvfError::invalid("warmup centroid shape mismatch"))?;

    if iters > 0 {
        let pool = create_thread_pool(params.num_threads)?;
        let mut assigner = cluster::ExactAssigner::default();
        cluster::lloyd(
            window.as_view(),
            &mut centroids,
            &mut assigner,
            iters,
            EmptyClusterPolicy::PreserveOld,
            params.normalize_centroids,
            &pool,
        )?;
    }
    Ok(centroids)
}
