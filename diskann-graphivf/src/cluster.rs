/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! k-means (Lloyd's) clustering driver with pluggable point-to-centroid
//! assignment.
//!
//! The assignment step dominates k-means cost (`O(num_points * num_clusters *
//! dim)` for a brute-force scan). At large `num_clusters` the scan becomes the
//! scaling bottleneck, so assignment is abstracted behind the [`Assigner`]
//! trait: [`ExactAssigner`] keeps the historical brute-force behavior, while
//! [`GraphAssigner`] searches an in-memory graph over the centroids to find the
//! nearest centroid approximately. The [`lloyd`] driver owns the update and
//! convergence logic and is independent of which assigner is used.

use diskann::ANNError;
use diskann_disk::utils::{compute_closest_centers, compute_vecs_l2sq};
use diskann_providers::{
    index::diskann_async::MemoryIndex,
    utils::{ParallelIteratorInPool, RayonThreadPool},
};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rayon::prelude::*;

use crate::{
    centroids,
    params::{EmptyClusterPolicy, GraphParams},
    GraphIvfError, Result,
};

/// Relative residual-improvement threshold below which Lloyd's is considered
/// converged. Matches the legacy brute-force k-means driver.
const CONVERGENCE_TOL: f32 = 0.00001;

/// Points are partitioned into chunks of this many when assignment is
/// parallelized across the thread pool.
const CHUNK: usize = 256;

/// Assigns every corpus point to its nearest centroid.
///
/// An assigner is stateful so it can amortize work across iterations (e.g.
/// caching point norms or reusing a centroid graph). [`assign`](Assigner::assign)
/// is called once per Lloyd's iteration with the current centroids.
pub(crate) trait Assigner {
    /// Write the nearest-centroid id of each row of `data` into `out`.
    ///
    /// `centroids` are the current centroids for iteration `iter` (0-based).
    /// `out.len()` equals `data.nrows()`.
    fn assign(
        &mut self,
        data: MatrixView<'_, f32>,
        centroids: MatrixView<'_, f32>,
        iter: usize,
        out: &mut [u32],
        pool: &RayonThreadPool,
    ) -> Result<()>;
}

/// Exact brute-force nearest-centroid assignment via the GEMM distance kernel.
///
/// Caches the squared norms of the corpus points (constant across iterations)
/// so they are computed only once.
#[derive(Default)]
pub(crate) struct ExactAssigner {
    docs_l2sq: Vec<f32>,
}

impl Assigner for ExactAssigner {
    fn assign(
        &mut self,
        data: MatrixView<'_, f32>,
        centroids: MatrixView<'_, f32>,
        _iter: usize,
        out: &mut [u32],
        pool: &RayonThreadPool,
    ) -> Result<()> {
        let num_points = data.nrows();
        let dim = data.ncols();
        let num_centers = centroids.nrows();

        if self.docs_l2sq.len() != num_points {
            self.docs_l2sq = vec![0.0f32; num_points];
            compute_vecs_l2sq(&mut self.docs_l2sq, data.as_slice(), dim, pool.as_ref())?;
        }

        compute_closest_centers(
            data.as_slice(),
            num_points,
            dim,
            centroids.as_slice(),
            num_centers,
            1,
            out,
            None,
            Some(&self.docs_l2sq),
            pool.as_ref(),
        )?;
        Ok(())
    }
}

/// Graph-accelerated approximate nearest-centroid assignment.
///
/// Builds an in-memory graph over the centroids and searches it for each point.
/// The graph is rebuilt every `rebuild_every` iterations (the centroids move
/// between iterations, so a stale graph degrades assignment quality). When
/// `rerank > 1`, the top candidates returned by the graph search are re-ranked
/// exactly against the current centroids, recovering most of the accuracy lost
/// to the approximate search.
pub(crate) struct GraphAssigner {
    graph: Option<MemoryIndex<f32>>,
    graph_params: GraphParams,
    assign_l: usize,
    rebuild_every: usize,
    rerank: usize,
    num_threads: usize,
}

impl GraphAssigner {
    pub(crate) fn new(
        graph_params: GraphParams,
        assign_l: usize,
        rebuild_every: usize,
        rerank: usize,
        num_threads: usize,
    ) -> Self {
        Self {
            graph: None,
            graph_params,
            assign_l,
            rebuild_every: rebuild_every.max(1),
            rerank: rerank.max(1),
            num_threads,
        }
    }
}

impl Assigner for GraphAssigner {
    fn assign(
        &mut self,
        data: MatrixView<'_, f32>,
        centroids: MatrixView<'_, f32>,
        iter: usize,
        out: &mut [u32],
        pool: &RayonThreadPool,
    ) -> Result<()> {
        let dim = centroids.ncols();
        let num_centers = centroids.nrows();

        // Rebuild the centroid graph when it is missing or stale.
        if self.graph.is_none() || iter.is_multiple_of(self.rebuild_every) {
            let owned = Matrix::try_from(
                centroids.as_slice().to_vec().into_boxed_slice(),
                num_centers,
                dim,
            )
            .map_err(|_| GraphIvfError::invalid("centroid matrix shape mismatch"))?;
            // Assignment is always squared-L2 (even for a hybrid `InnerProduct`
            // index, where only *search* uses inner product).
            self.graph = Some(centroids::build(
                owned,
                &self.graph_params,
                self.num_threads,
                VectorMetric::L2,
            )?);
        }
        let graph = self.graph.as_ref().expect("graph built above");

        let rerank = self.rerank;
        let assign_l = self.assign_l.max(rerank);

        out.par_chunks_mut(CHUNK).enumerate().try_for_each_in_pool(
            pool.as_ref(),
            |(ci, chunk)| -> Result<()> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .map_err(ANNError::from)?;
                let mut ids = vec![0u32; rerank];
                let mut dist = vec![0.0f32; rerank];
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let pid = ci * CHUNK + j;
                    let point = data.row(pid);
                    let n =
                        centroids::search(graph, &runtime, point, assign_l, &mut ids, &mut dist)?;
                    *slot = if rerank == 1 || n <= 1 {
                        ids[0]
                    } else {
                        // Re-rank the graph candidates against the exact centroids.
                        let mut best = ids[0];
                        let mut best_d = sq_l2(point, centroids.row(best as usize));
                        for &cid in &ids[1..n] {
                            let d = sq_l2(point, centroids.row(cid as usize));
                            if d < best_d {
                                best_d = d;
                                best = cid;
                            }
                        }
                        best
                    };
                }
                Ok(())
            },
        )?;
        Ok(())
    }
}

/// Result of a [`lloyd`] run.
pub(crate) struct LloydOutcome {
    /// Final assignment of each point to a centroid (from the last iteration).
    pub assignments: Vec<u32>,
    /// Number of iterations actually run (`<= max_iters`).
    pub iters_run: usize,
    /// Final clustering residual (sum of squared distances to assigned
    /// centroids).
    pub residual: f32,
}

/// Refine `centroids` in place with up to `max_iters` Lloyd's iterations over
/// `data`, using `assigner` for the point-to-centroid step.
///
/// Stops early once the relative residual improvement drops below
/// [`CONVERGENCE_TOL`] or the residual underflows. Empty clusters are handled
/// per `empty`.
pub(crate) fn lloyd(
    data: MatrixView<'_, f32>,
    centroids: &mut Matrix<f32>,
    assigner: &mut dyn Assigner,
    max_iters: usize,
    empty: EmptyClusterPolicy,
    normalize: bool,
    pool: &RayonThreadPool,
) -> Result<LloydOutcome> {
    let num_points = data.nrows();
    let num_centers = centroids.nrows();
    let mut assignments = vec![0u32; num_points];
    let mut dists = vec![0.0f32; num_points];
    let mut residual = 0.0f32;
    let mut iters_run = 0usize;

    for iter in 0..max_iters {
        let old_residual = residual;

        assigner.assign(data, centroids.as_view(), iter, &mut assignments, pool)?;
        let buckets = bucket(&assignments, num_centers);
        update_centroids(data, centroids, &buckets, empty, pool);
        if normalize {
            normalize_centroids(centroids, pool);
        }
        assigned_distances(data, centroids.as_view(), &assignments, &mut dists, pool);
        residual = dists.iter().sum();

        if matches!(empty, EmptyClusterPolicy::ReseedFarthest) {
            reseed_empty(data, centroids, &buckets, &dists);
        }

        iters_run = iter + 1;
        if (iter != 0 && (old_residual - residual) / residual < CONVERGENCE_TOL)
            || residual < f32::EPSILON
        {
            break;
        }
    }

    Ok(LloydOutcome {
        assignments,
        iters_run,
        residual,
    })
}

/// Partition point indices by their assigned centroid (in increasing point
/// order within each bucket).
fn bucket(assignments: &[u32], num_centers: usize) -> Vec<Vec<u32>> {
    let mut buckets = vec![Vec::new(); num_centers];
    for (p, &c) in assignments.iter().enumerate() {
        buckets[c as usize].push(p as u32);
    }
    buckets
}

/// Replace each centroid with the mean of its assigned points (f64 accumulation
/// for numerical stability). Empty clusters are handled per `empty`; for
/// [`EmptyClusterPolicy::ReseedFarthest`] they are left in place here and moved
/// later by [`reseed_empty`].
fn update_centroids(
    data: MatrixView<'_, f32>,
    centroids: &mut Matrix<f32>,
    buckets: &[Vec<u32>],
    empty: EmptyClusterPolicy,
    pool: &RayonThreadPool,
) {
    let dim = data.ncols();
    centroids
        .as_mut_slice()
        .par_chunks_mut(dim)
        .enumerate()
        .for_each_in_pool(pool.as_ref(), |(c, center)| {
            let bucket = &buckets[c];
            if bucket.is_empty() {
                if matches!(empty, EmptyClusterPolicy::Zero) {
                    center.fill(0.0);
                }
                return;
            }
            let mut sum = vec![0.0f64; dim];
            for &p in bucket {
                for (s, &v) in sum.iter_mut().zip(data.row(p as usize)) {
                    *s += v as f64;
                }
            }
            let inv = 1.0f64 / bucket.len() as f64;
            for (cv, &s) in center.iter_mut().zip(sum.iter()) {
                *cv = (s * inv) as f32;
            }
        });
}

/// Fill `dists[p]` with the squared distance from point `p` to its assigned
/// centroid.
fn assigned_distances(
    data: MatrixView<'_, f32>,
    centroids: MatrixView<'_, f32>,
    assignments: &[u32],
    dists: &mut [f32],
    pool: &RayonThreadPool,
) {
    dists
        .par_iter_mut()
        .enumerate()
        .for_each_in_pool(pool.as_ref(), |(p, d)| {
            let c = assignments[p] as usize;
            *d = sq_l2(data.row(p), centroids.row(c));
        });
}

/// Move every empty cluster's centroid onto the farthest-from-its-centroid
/// corpus point, splitting the most spread-out clusters. Distinct points are
/// chosen per empty cluster.
fn reseed_empty(
    data: MatrixView<'_, f32>,
    centroids: &mut Matrix<f32>,
    buckets: &[Vec<u32>],
    dists: &[f32],
) {
    let empties: Vec<usize> = buckets
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_empty())
        .map(|(c, _)| c)
        .collect();
    if empties.is_empty() {
        return;
    }
    let dim = data.ncols();
    let mut order: Vec<usize> = (0..data.nrows()).collect();
    order.sort_unstable_by(|&a, &b| dists[b].total_cmp(&dists[a]));
    for (&e, &p) in empties.iter().zip(order.iter()) {
        centroids.as_mut_slice()[e * dim..(e + 1) * dim].copy_from_slice(data.row(p));
    }
}

/// L2-normalize every centroid row onto the unit sphere (zero-norm rows are
/// left untouched).
fn normalize_centroids(centroids: &mut Matrix<f32>, pool: &RayonThreadPool) {
    let dim = centroids.ncols();
    centroids
        .as_mut_slice()
        .par_chunks_mut(dim)
        .for_each_in_pool(pool.as_ref(), |row| {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0 / norm;
                for x in row.iter_mut() {
                    *x *= inv;
                }
            }
        });
}

/// Squared Euclidean distance between two equal-length vectors.
pub(crate) fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_providers::utils::create_thread_pool;

    fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
        Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
    }

    /// Independent brute-force reference: argmin assignment over all centroids,
    /// then per-cluster mean (empty clusters left at zero).
    fn reference(
        data: &[f32],
        centroids: &[f32],
        num_points: usize,
        num_centers: usize,
        dim: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut assignments = vec![0u32; num_points];
        for (p, slot) in assignments.iter_mut().enumerate() {
            let point = &data[p * dim..(p + 1) * dim];
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..num_centers {
                let cen = &centroids[c * dim..(c + 1) * dim];
                let d = sq_l2(point, cen);
                if d < best_d {
                    best_d = d;
                    best = c as u32;
                }
            }
            *slot = best;
        }

        let mut means = vec![0.0f32; num_centers * dim];
        let mut counts = vec![0usize; num_centers];
        for (p, &c) in assignments.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            for d in 0..dim {
                means[c * dim + d] += data[p * dim + d];
            }
        }
        for c in 0..num_centers {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    means[c * dim + d] *= inv;
                }
            }
        }
        (assignments, means)
    }

    /// Deterministic pseudo-random corpus generator (no external rng crate).
    fn synthetic(num_points: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut out = vec![0.0f32; num_points * dim];
        for v in out.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *v = (state >> 40) as f32 / (1u64 << 24) as f32;
        }
        out
    }

    #[test]
    fn exact_one_iter_matches_bruteforce_reference() {
        let pool = create_thread_pool(2).unwrap();
        let (num_points, num_centers, dim) = (200usize, 6usize, 8usize);
        let data = synthetic(num_points, dim, 1);
        let seed = synthetic(num_centers, dim, 99);

        let (ref_assign, ref_centroids) = reference(&data, &seed, num_points, num_centers, dim);

        let data_mat = mat(data, num_points, dim);
        let mut centroids = mat(seed, num_centers, dim);
        let mut assigner = ExactAssigner::default();
        let outcome = lloyd(
            data_mat.as_view(),
            &mut centroids,
            &mut assigner,
            1,
            EmptyClusterPolicy::Zero,
            false,
            &pool,
        )
        .unwrap();

        assert_eq!(outcome.assignments, ref_assign);
        for (got, want) in centroids.as_slice().iter().zip(ref_centroids.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "centroid mismatch: {got} vs {want}"
            );
        }
    }

    #[test]
    fn residual_is_non_increasing_across_iterations() {
        let pool = create_thread_pool(2).unwrap();
        let (num_points, num_centers, dim) = (300usize, 8usize, 6usize);
        let data = synthetic(num_points, dim, 7);
        let seed = synthetic(num_centers, dim, 3);
        let data_mat = mat(data, num_points, dim);

        let mut c1 = mat(seed.clone(), num_centers, dim);
        let r1 = lloyd(
            data_mat.as_view(),
            &mut c1,
            &mut ExactAssigner::default(),
            1,
            EmptyClusterPolicy::PreserveOld,
            false,
            &pool,
        )
        .unwrap()
        .residual;

        let mut c5 = mat(seed, num_centers, dim);
        let r5 = lloyd(
            data_mat.as_view(),
            &mut c5,
            &mut ExactAssigner::default(),
            5,
            EmptyClusterPolicy::PreserveOld,
            false,
            &pool,
        )
        .unwrap()
        .residual;

        assert!(r5 <= r1 + 1e-3, "residual increased: {r1} -> {r5}");
    }

    #[test]
    fn graph_matches_exact_on_separated_clusters() {
        let pool = create_thread_pool(2).unwrap();
        let dim = 8usize;
        let num_centers = 12usize;
        // Well-separated blobs: 20 points around each of `num_centers` far-apart
        // anchors, so the true nearest centroid is unambiguous.
        let mut data = Vec::new();
        let jitter = synthetic(num_centers * 20, dim, 5);
        for c in 0..num_centers {
            for i in 0..20 {
                for d in 0..dim {
                    let anchor = (c * 100) as f32;
                    let j = jitter[(c * 20 + i) * dim + d];
                    data.push(anchor + j);
                }
            }
        }
        let num_points = num_centers * 20;
        let seed = synthetic(num_centers, dim, 2);
        let data_mat = mat(data, num_points, dim);

        let mut c_exact = mat(seed.clone(), num_centers, dim);
        let exact = lloyd(
            data_mat.as_view(),
            &mut c_exact,
            &mut ExactAssigner::default(),
            5,
            EmptyClusterPolicy::PreserveOld,
            false,
            &pool,
        )
        .unwrap();

        let mut c_graph = mat(seed, num_centers, dim);
        let graph = lloyd(
            data_mat.as_view(),
            &mut c_graph,
            &mut GraphAssigner::new(GraphParams::default(), 32, 1, 8, 2),
            5,
            EmptyClusterPolicy::PreserveOld,
            false,
            &pool,
        )
        .unwrap();

        assert_eq!(
            exact.assignments, graph.assignments,
            "graph assignment diverged from exact on separated data"
        );
    }

    #[test]
    fn empty_cluster_policies() {
        let pool = create_thread_pool(2).unwrap();
        let dim = 2usize;
        // Two tight blobs near origin plus one far-away centroid that will
        // receive no points.
        let data = vec![
            0.0, 0.0, 0.1, 0.0, 0.0, 0.1, // blob A
            10.0, 10.0, 10.1, 10.0, 10.0, 10.1, // blob B
        ];
        let num_points = 6;
        let num_centers = 3;
        // Centroid 2 starts far away and stays empty.
        let seed = vec![0.0, 0.0, 10.0, 10.0, 100.0, 100.0];
        let data_mat = mat(data, num_points, dim);

        // Zero: empty centroid becomes the zero vector.
        let mut c = mat(seed.clone(), num_centers, dim);
        lloyd(
            data_mat.as_view(),
            &mut c,
            &mut ExactAssigner::default(),
            1,
            EmptyClusterPolicy::Zero,
            false,
            &pool,
        )
        .unwrap();
        assert_eq!(&c.as_slice()[4..6], &[0.0, 0.0]);

        // PreserveOld: empty centroid keeps its seed position.
        let mut c = mat(seed.clone(), num_centers, dim);
        lloyd(
            data_mat.as_view(),
            &mut c,
            &mut ExactAssigner::default(),
            1,
            EmptyClusterPolicy::PreserveOld,
            false,
            &pool,
        )
        .unwrap();
        assert_eq!(&c.as_slice()[4..6], &[100.0, 100.0]);

        // ReseedFarthest: empty centroid jumps onto an existing corpus point.
        let mut c = mat(seed, num_centers, dim);
        lloyd(
            data_mat.as_view(),
            &mut c,
            &mut ExactAssigner::default(),
            1,
            EmptyClusterPolicy::ReseedFarthest,
            false,
            &pool,
        )
        .unwrap();
        let reseeded = &c.as_slice()[4..6];
        assert_ne!(reseeded, &[100.0, 100.0], "centroid was not reseeded");
        assert_ne!(reseeded, &[0.0, 0.0], "centroid should not be zeroed");
    }
}
