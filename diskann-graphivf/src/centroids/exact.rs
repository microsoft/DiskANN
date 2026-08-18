/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exact centroid lookup over a dense, live-only copy of the centroid vectors.
//!
//! The alternative to navigating the centroid graph. Where the graph trades
//! accuracy for a sub-linear walk, this scans every live centroid and is
//! therefore exact by construction: it cannot return fewer centroids than were
//! asked for while live ones remain, and it can never return a retired one.
//!
//! Scanning is expressed as a matrix multiply so that it is bounded by memory
//! bandwidth rather than by per-candidate function calls. For squared-L2 the
//! usual expansion applies,
//!
//! ```text
//! ||q - c||^2 = ||q||^2 - 2 q.c + ||c||^2
//! ```
//!
//! where `q.c` for every `(query, centroid)` pair is one `sgemm`, `||c||^2` is
//! cached on the dense copy, and `||q||^2` is constant across centroids and so
//! is added once at the end rather than inside the inner loop.

use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann_linalg::{sgemm, Transpose};
use diskann_providers::utils::{ParallelIteratorInPool, RayonThreadPool};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::Metric as VectorMetric;
use rayon::prelude::*;

use crate::{GraphIvfError, Result};

/// Marks an id that has no dense row, either because it was never allocated or
/// because it has been retired.
const NO_ROW: u32 = u32::MAX;

/// Centroids scored per `sgemm` call.
///
/// The score block is `queries x tile` floats, so this bounds the per-worker
/// scratch independently of how many centroids the index holds.
const CENTROID_TILE: usize = 8192;

/// Queries handled by one parallel work unit.
///
/// Large enough that the `sgemm` calls see a real matrix rather than a series
/// of matrix-vector products, small enough to keep the pool load-balanced when
/// a batch is only a few thousand queries.
const QUERY_CHUNK: usize = 32;

/// The distance an exact scan minimizes.
///
/// Both variants are "smaller is better", matching the convention of the graph
/// path, whose inner-product distance is likewise negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExactMetric {
    /// Squared L2. Cosine also uses this, relying on normalized data exactly as
    /// the graph path does.
    SqL2,
    /// Negated inner product, so that the maximum-IP centroid sorts first.
    NegInnerProduct,
}

impl ExactMetric {
    /// The exact equivalent of the metric the centroid graph would navigate
    /// with, so that switching between the two changes only how centroids are
    /// found and not which ones are considered closest.
    ///
    /// # Errors
    ///
    /// Returns an error for a metric the centroid path does not support.
    pub(crate) fn for_navigation(metric: VectorMetric) -> Result<Self> {
        match metric {
            VectorMetric::L2 | VectorMetric::Cosine => Ok(Self::SqL2),
            VectorMetric::InnerProduct => Ok(Self::NegInnerProduct),
            other => Err(GraphIvfError::invalid(format!(
                "exact centroid search does not support metric {other:?}"
            ))),
        }
    }

    /// The `sgemm` scaling that turns `q.c` into this distance, up to the
    /// query-constant term added by [`query_offset`](Self::query_offset).
    fn gemm_alpha(self) -> f32 {
        match self {
            Self::SqL2 => -2.0,
            Self::NegInnerProduct => -1.0,
        }
    }

    /// The term that depends only on the query, added after selection because
    /// it is identical for every candidate and so cannot affect the ranking.
    fn query_offset(self, query: &[f32]) -> f32 {
        match self {
            Self::SqL2 => query.iter().map(|&x| x * x).sum(),
            Self::NegInnerProduct => 0.0,
        }
    }
}

/// A contiguous, live-only copy of the centroid vectors, addressable by
/// centroid id.
///
/// Centroid ids are permanent and are never reused, so an id-indexed store is
/// necessarily sparse and grows without bound as clusters churn. Exact search
/// wants the opposite: a packed `live x dim` buffer it can hand to `sgemm`
/// without gathering. This holds both — a compact row array plus an id-to-row
/// map — and keeps them consistent across insertion and retirement in `O(dim)`,
/// so no caller ever has to re-pack.
///
/// Retirement moves the last row into the vacated slot, so dense row order is
/// unspecified and shifts over time. Anything that needs a stable order (the
/// flushed cluster numbering, for one) must iterate by id via
/// [`iter_by_id`](Self::iter_by_id).
pub(crate) struct DenseCentroids {
    dim: usize,
    /// Row-major `len() x dim`, live centroids only.
    vecs: Vec<f32>,
    /// `ids[row]` is the centroid id occupying that dense row.
    ids: Vec<u32>,
    /// `rows[id]` is the dense row holding `id`, or [`NO_ROW`].
    rows: Vec<u32>,
    /// `norms[row]` is the squared L2 norm of that dense row.
    norms: Vec<f32>,
}

impl DenseCentroids {
    /// An empty store able to address ids `0..id_capacity` without reallocating
    /// the id map.
    pub(crate) fn with_capacity(dim: usize, id_capacity: usize) -> Self {
        Self {
            dim,
            vecs: Vec::new(),
            ids: Vec::new(),
            rows: vec![NO_ROW; id_capacity],
            norms: Vec::new(),
        }
    }

    /// A store over `centroids`, where row `i` is centroid id `i`.
    ///
    /// This is the shape a flushed index loads: densification already happened
    /// when the index was written, so ids and rows coincide.
    pub(crate) fn from_matrix(centroids: &Matrix<f32>) -> Self {
        let mut dense = Self::with_capacity(centroids.ncols(), centroids.nrows());
        for row in 0..centroids.nrows() {
            dense.push(row as u32, centroids.row(row));
        }
        dense
    }

    /// Centroid dimensionality.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// Number of live centroids.
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no centroid is live.
    pub(crate) fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether `id` is live.
    pub(crate) fn contains(&self, id: u32) -> bool {
        self.row_of(id).is_some()
    }

    /// The vector of centroid `id`, or `None` if it is retired or was never
    /// allocated.
    pub(crate) fn get(&self, id: u32) -> Option<&[f32]> {
        let row = self.row_of(id)?;
        Some(&self.vecs[row * self.dim..(row + 1) * self.dim])
    }

    /// Live centroids as `(id, vector)` pairs in **ascending id order**.
    ///
    /// Slower than iterating rows — it walks the whole id space — but it is the
    /// only order that is stable across retirements.
    pub(crate) fn iter_by_id(&self) -> impl Iterator<Item = (u32, &[f32])> + '_ {
        (0..self.rows.len() as u32).filter_map(|id| self.get(id).map(|v| (id, v)))
    }

    /// Live centroid ids in **ascending order**.
    pub(crate) fn ids_by_id(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.rows.len() as u32).filter(|&id| self.contains(id))
    }

    /// Add centroid `id` with vector `vec`.
    ///
    /// # Panics
    ///
    /// Panics if `vec` has the wrong length or `id` is already live.
    pub(crate) fn push(&mut self, id: u32, vec: &[f32]) {
        assert_eq!(vec.len(), self.dim, "centroid dimension mismatch");
        let idx = id as usize;
        if idx >= self.rows.len() {
            self.rows.resize(idx + 1, NO_ROW);
        }
        assert_eq!(self.rows[idx], NO_ROW, "centroid {id} is already live");

        self.rows[idx] = self.ids.len() as u32;
        self.ids.push(id);
        self.norms.push(vec.iter().map(|&x| x * x).sum());
        self.vecs.extend_from_slice(vec);
    }

    /// Retire centroid `id`, returning whether it was live.
    ///
    /// The last dense row is moved into the vacated one, so this is `O(dim)`
    /// and leaves the buffer packed.
    pub(crate) fn remove(&mut self, id: u32) -> bool {
        let Some(row) = self.row_of(id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        if row != last {
            let (head, tail) = self.vecs.split_at_mut(last * self.dim);
            head[row * self.dim..(row + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            self.ids[row] = self.ids[last];
            self.norms[row] = self.norms[last];
            self.rows[self.ids[row] as usize] = row as u32;
        }
        self.vecs.truncate(last * self.dim);
        self.ids.truncate(last);
        self.norms.truncate(last);
        self.rows[id as usize] = NO_ROW;
        true
    }

    fn row_of(&self, id: u32) -> Option<usize> {
        match self.rows.get(id as usize) {
            Some(&NO_ROW) | None => None,
            Some(&row) => Some(row as usize),
        }
    }

    /// How many results a scan can produce for one query asking for `k`.
    fn results_per_query(&self, k: usize) -> usize {
        k.min(self.len())
    }

    /// Exact `k` nearest live centroids to a single query.
    ///
    /// Returns the number of results written, which is `min(k, len())`.
    ///
    /// `ids_out` and `dist_out` must have the same length, which is the
    /// requested `k`.
    pub(crate) fn search(
        &self,
        metric: ExactMetric,
        query: &[f32],
        ids_out: &mut [u32],
        dist_out: &mut [f32],
    ) -> Result<usize> {
        debug_assert_eq!(ids_out.len(), dist_out.len());
        let queries = MatrixView::try_from(query, 1, self.dim)
            .map_err(|_| GraphIvfError::invalid("query dimension mismatch"))?;
        let mut scratch = ScanScratch::default();
        self.scan_block(metric, queries, &mut scratch, ids_out, dist_out)?;
        Ok(self.results_per_query(ids_out.len()))
    }

    /// Exact `k` nearest live centroids for each row of `queries`.
    ///
    /// `ids_out` and `dist_out` are `queries.nrows() * k` long and are written
    /// as row-major `nrows x k` blocks in ascending distance. Returns the
    /// number of results written per query, which is `min(k, len())`.
    ///
    /// This is the entry point worth using whenever more than a handful of
    /// queries are available at once: the cost is dominated by streaming the
    /// centroids, which a batch pays for once instead of per query.
    pub(crate) fn search_batch(
        &self,
        metric: ExactMetric,
        queries: MatrixView<'_, f32>,
        k: usize,
        ids_out: &mut [u32],
        dist_out: &mut [f32],
        pool: &RayonThreadPool,
    ) -> Result<usize> {
        if k == 0 {
            return Err(GraphIvfError::invalid("k must be non-zero"));
        }
        if queries.ncols() != self.dim {
            return Err(GraphIvfError::invalid(format!(
                "query dim {} does not match centroid dim {}",
                queries.ncols(),
                self.dim
            )));
        }
        let expected = queries.nrows() * k;
        if ids_out.len() != expected || dist_out.len() != expected {
            return Err(GraphIvfError::invalid(format!(
                "exact centroid output must hold {expected} entries",
            )));
        }

        ids_out
            .par_chunks_mut(QUERY_CHUNK * k)
            .zip(dist_out.par_chunks_mut(QUERY_CHUNK * k))
            .enumerate()
            .try_for_each_init_in_pool(
                pool.as_ref(),
                ScanScratch::default,
                |scratch, (chunk, (ids_chunk, dist_chunk))| -> Result<()> {
                    let start = chunk * QUERY_CHUNK;
                    let rows = ids_chunk.len() / k;
                    let block = queries
                        .subview(start..start + rows)
                        .ok_or_else(|| GraphIvfError::invalid("query block out of range"))?;
                    self.scan_block(metric, block, scratch, ids_chunk, dist_chunk)
                        .map(|_| ())
                },
            )?;

        Ok(self.results_per_query(k))
    }

    /// Score every live centroid against every row of `queries` and write the
    /// best `k = ids_out.len() / queries.nrows()` of each.
    ///
    /// Each query receives [`results_per_query`](Self::results_per_query)
    /// results; trailing slots are left at id 0 and infinite distance.
    ///
    /// Centroids are streamed in tiles so that the score block stays bounded no
    /// matter how many clusters exist; a per-query bounded queue carries the
    /// running best across tiles.
    fn scan_block(
        &self,
        metric: ExactMetric,
        queries: MatrixView<'_, f32>,
        scratch: &mut ScanScratch,
        ids_out: &mut [u32],
        dist_out: &mut [f32],
    ) -> Result<()> {
        let nq = queries.nrows();
        let k = ids_out.len() / nq.max(1);
        ids_out.fill(0);
        dist_out.fill(f32::INFINITY);
        if nq == 0 || self.is_empty() {
            return Ok(());
        }

        scratch.reset(nq, k);
        for start in (0..self.len()).step_by(CENTROID_TILE) {
            let tile = CENTROID_TILE.min(self.len() - start);
            let scores = &mut scratch.scores;
            scores.clear();
            scores.resize(nq * tile, 0.0);

            sgemm(
                Transpose::None,
                Transpose::Ordinary,
                nq,
                tile,
                self.dim,
                metric.gemm_alpha(),
                queries.as_slice(),
                &self.vecs[start * self.dim..(start + tile) * self.dim],
                None,
                scores,
            )
            .map_err(|e| GraphIvfError::invalid(format!("centroid sgemm failed: {e}")))?;

            if metric == ExactMetric::SqL2 {
                let norms = &self.norms[start..start + tile];
                for row in scores.chunks_exact_mut(tile) {
                    for (score, norm) in row.iter_mut().zip(norms) {
                        *score += norm;
                    }
                }
            }

            let ids = &self.ids[start..start + tile];
            for (best, row) in scratch.best.iter_mut().zip(scores.chunks_exact(tile)) {
                for (&id, &distance) in ids.iter().zip(row) {
                    best.insert(Neighbor::new(id, distance));
                }
            }
        }

        for (q, best) in scratch.best.iter().enumerate() {
            // The expansion can land a hair below zero on a centroid that
            // coincides with the query; a negative squared distance would be
            // nonsense to every consumer, so clamp rather than propagate it.
            let offset = metric.query_offset(queries.row(q));
            for r in 0..best.size().min(k) {
                let neighbor = best.get(r);
                ids_out[q * k + r] = *neighbor.id();
                dist_out[q * k + r] = (neighbor.distance() + offset).max(0.0);
            }
        }
        Ok(())
    }
}

/// Reusable working set for one parallel unit of [`DenseCentroids::scan_block`].
#[derive(Default)]
struct ScanScratch {
    /// `queries x tile` score block for the current centroid tile.
    scores: Vec<f32>,
    /// Running best-`k` per query, carried across tiles.
    best: Vec<NeighborPriorityQueue<u32>>,
}

impl ScanScratch {
    fn reset(&mut self, nq: usize, k: usize) {
        // The queue's capacity is fixed at construction, so a change in `k`
        // requires new queues rather than a clear.
        if self.best.len() != nq || self.best.first().is_some_and(|b| b.capacity() != k) {
            self.best = (0..nq).map(|_| NeighborPriorityQueue::new(k)).collect();
        } else {
            for best in &mut self.best {
                best.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::sq_l2;
    use diskann_providers::utils::create_thread_pool;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Matrix<f32> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        Matrix::try_from(data.into_boxed_slice(), rows, cols).unwrap()
    }

    /// Reference ranking: every live centroid scored directly, sorted by
    /// distance with ties broken by id so the comparison is deterministic.
    fn reference(dense: &DenseCentroids, metric: ExactMetric, query: &[f32]) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = dense
            .iter_by_id()
            .map(|(id, v)| {
                let d = match metric {
                    ExactMetric::SqL2 => sq_l2(query, v),
                    ExactMetric::NegInnerProduct => {
                        -query.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()
                    }
                };
                (id, d)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored
    }

    /// Compare returned ids against the reference, tolerating a different
    /// tie-break: a returned centroid is correct if its distance is within
    /// rounding of the reference distance at the same rank.
    fn assert_matches_reference(
        dense: &DenseCentroids,
        metric: ExactMetric,
        query: &[f32],
        got: &[u32],
    ) {
        let want = reference(dense, metric, query);
        assert_eq!(got.len(), want.len().min(got.len()));
        for (rank, &id) in got.iter().enumerate() {
            let actual = want.iter().find(|(w, _)| *w == id).expect("live id").1;
            let expected = want[rank].1;
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "rank {rank}: got id {id} at {actual}, reference {expected}",
            );
        }
    }

    /// Push/remove keeps the id map, the packed rows, and the cached norms
    /// consistent, including when the removed centroid is the last row.
    #[test]
    fn push_remove_keeps_rows_consistent() {
        let mut rng = StdRng::seed_from_u64(7);
        let cents = matrix(32, 5, &mut rng);
        let mut dense = DenseCentroids::with_capacity(5, 32);
        for i in 0..32 {
            dense.push(i as u32, cents.row(i));
        }

        // Remove a middle row, the (new) last row, and the first row.
        for victim in [10u32, 31, 0] {
            assert!(dense.remove(victim));
            assert!(!dense.remove(victim), "removal is idempotent");
        }

        assert_eq!(dense.len(), 29);
        for id in 0..32u32 {
            let live = !matches!(id, 0 | 10 | 31);
            assert_eq!(dense.contains(id), live, "id {id}");
            match dense.get(id) {
                Some(v) => assert_eq!(v, cents.row(id as usize)),
                None => assert!(!live),
            }
        }
        // Every dense row still agrees with the id map and the cached norm.
        for (row, &id) in dense.ids.iter().enumerate() {
            assert_eq!(dense.rows[id as usize] as usize, row);
            let v = &dense.vecs[row * 5..(row + 1) * 5];
            let norm: f32 = v.iter().map(|x| x * x).sum();
            assert!((dense.norms[row] - norm).abs() < 1e-6);
        }
        assert_eq!(dense.ids_by_id().collect::<Vec<_>>().len(), 29);
    }

    /// A single-query scan reproduces a direct scored-and-sorted ranking, for
    /// both supported metrics.
    #[test]
    fn single_query_matches_reference() {
        let mut rng = StdRng::seed_from_u64(11);
        let cents = matrix(500, 16, &mut rng);
        let dense = DenseCentroids::from_matrix(&cents);
        let queries = matrix(8, 16, &mut rng);

        for metric in [ExactMetric::SqL2, ExactMetric::NegInnerProduct] {
            for q in 0..queries.nrows() {
                let mut ids = vec![0u32; 10];
                let mut dist = vec![0.0f32; 10];
                let found = dense
                    .search(metric, queries.row(q), &mut ids, &mut dist)
                    .unwrap();
                assert_eq!(found, 10);
                assert_matches_reference(&dense, metric, queries.row(q), &ids);
                assert!(dist.windows(2).all(|w| w[0] <= w[1]), "ascending");
            }
        }
    }

    /// The batch path agrees with the single-query path query for query, and
    /// crossing a centroid tile boundary changes nothing.
    #[test]
    fn batch_matches_single_across_tiles() {
        let mut rng = StdRng::seed_from_u64(13);
        // More centroids than one tile, so the running-best carry is exercised.
        let cents = matrix(CENTROID_TILE + 257, 8, &mut rng);
        let dense = DenseCentroids::from_matrix(&cents);
        let queries = matrix(70, 8, &mut rng);
        let pool = create_thread_pool(4).unwrap();
        let k = 12;

        let mut ids = vec![0u32; queries.nrows() * k];
        let mut dist = vec![0.0f32; queries.nrows() * k];
        let found = dense
            .search_batch(
                ExactMetric::SqL2,
                queries.as_view(),
                k,
                &mut ids,
                &mut dist,
                &pool,
            )
            .unwrap();
        assert_eq!(found, k);

        for q in 0..queries.nrows() {
            let mut one_ids = vec![0u32; k];
            let mut one_dist = vec![0.0f32; k];
            dense
                .search(
                    ExactMetric::SqL2,
                    queries.row(q),
                    &mut one_ids,
                    &mut one_dist,
                )
                .unwrap();
            assert_eq!(&ids[q * k..(q + 1) * k], one_ids.as_slice(), "query {q}");
        }
    }

    /// Asking for more centroids than exist returns every live one and nothing
    /// else, rather than erroring or padding with retired ids.
    #[test]
    fn k_larger_than_live_count() {
        let mut rng = StdRng::seed_from_u64(17);
        let cents = matrix(6, 4, &mut rng);
        let mut dense = DenseCentroids::from_matrix(&cents);
        dense.remove(2);
        dense.remove(4);

        let mut ids = vec![0u32; 10];
        let mut dist = vec![0.0f32; 10];
        let found = dense
            .search(ExactMetric::SqL2, cents.row(0), &mut ids, &mut dist)
            .unwrap();
        assert_eq!(found, 4);
        let mut got = ids[..found].to_vec();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 3, 5]);
    }
}
