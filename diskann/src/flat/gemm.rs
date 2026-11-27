/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tiled flat search with common filtering, selection, and metric semantics.
//!
//! Dense inputs use an explicit pool. Streaming sources can overlap asynchronous
//! reads with bounded CPU work and reuse the ordinary search post-processor.
//!
//! # Example
//! ```rust
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use diskann::flat::gemm::{self, AllCandidates, BatchOptions, Dataset, Selection};
//! use diskann_utils::views::rowmajor::{self, Matrix};
//! use diskann_vector::distance::Metric;
//! use std::num::NonZeroUsize;
//!
//! let data = rowmajor::Owned::from_element(8, 4, 1.0f32);
//! let queries = rowmajor::Owned::from_element(2, 4, 1.0f32);
//! let ids = [80u64, 10, 70, 20, 60, 30, 50, 40];
//! let database = Dataset::new(data.as_view(), &ids)?
//!     .select(Selection::Rows(&[3, 0, 7]))?;
//! let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
//! let results = gemm::knn_search(database, queries.as_view(),
//!     NonZeroUsize::new(2).unwrap(), Metric::Cosine, &AllCandidates,
//!     &BatchOptions::default(), &pool)?;
//! assert_eq!(*results[0].neighbors[0].id(), 40); // later candidate wins a tie
//! # Ok(())
//! # }
//! ```

use std::{cmp::Ordering, num::NonZeroUsize};

use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;
use rayon::ThreadPool;

use super::SearchOptions;
use crate::{ANNError, ANNResult, neighbor::Neighbor, utils::VectorId};

mod input;
mod kernel;
mod stream;

pub use input::{AllCandidates, CandidateFilter, Dataset, GemmElement, Selection};
pub use stream::{
    BatchSource, BlockResult, MemorySource, OwnedBlock, knn_search_batch,
    knn_search_batch_with_contexts,
};

/// Bounds score scratch space per worker independently of corpus size.
#[derive(Debug, Clone, Copy)]
pub struct GemmOptions {
    /// Per-request worker limit, capped by pool size. Defaults to one.
    pub max_workers: NonZeroUsize,
    pub query_block_size: NonZeroUsize,
    pub database_block_size: NonZeroUsize,
    pub collection: SearchOptions,
}

impl Default for GemmOptions {
    #[expect(clippy::unwrap_used, reason = "nonzero literals")]
    fn default() -> Self {
        Self {
            max_workers: NonZeroUsize::MIN,
            query_block_size: NonZeroUsize::new(32).unwrap(),
            database_block_size: NonZeroUsize::new(1024).unwrap(),
            collection: SearchOptions::default(),
        }
    }
}

impl GemmOptions {
    fn worker_count(&self, pool: &ThreadPool, rows: NonZeroUsize) -> usize {
        self.max_workers
            .get()
            .min(pool.current_num_threads())
            .min(rows.get())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Engine {
    /// Select per tile, including the cost of packing and the filter density.
    #[default]
    Auto,
    /// Use the original VectorRepr distance kernel on eligible pairs only.
    Direct,
    /// Use GEMM; exceptional or ill-conditioned scores may still be recomputed.
    Gemm,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchOptions {
    pub engine: Engine,
    pub gemm: GemmOptions,
    /// Maximum rows per async read. Independent of compute tiles to amortize
    /// scheduling; at most two such blocks are retained by the streaming driver.
    pub read_block_size: NonZeroUsize,
    /// Minimum query-count times dimension for Auto. Packing doubles this threshold.
    pub gemm_threshold: usize,
    /// Minimum eligible fraction of a tile for Auto, in [0, 1].
    pub min_gemm_density: f32,
}

impl Default for BatchOptions {
    #[expect(clippy::unwrap_used, reason = "nonzero literal")]
    fn default() -> Self {
        Self {
            engine: Engine::Auto,
            gemm: GemmOptions::default(),
            read_block_size: NonZeroUsize::new(4096).unwrap(),
            gemm_threshold: 4096,
            min_gemm_density: 0.25,
        }
    }
}

impl BatchOptions {
    #[track_caller]
    fn validate(&self) -> ANNResult<()> {
        if !(0.0..=1.0).contains(&self.min_gemm_density) {
            return Err(ANNError::message("minimum GEMM density must be in [0, 1]"));
        }
        Ok(())
    }

    fn use_gemm<T: GemmElement>(&self, nq: usize, nb: usize, dim: usize, eligible: usize) -> bool {
        if eligible == 0 {
            return false;
        }
        match self.engine {
            Engine::Direct => false,
            Engine::Gemm => true,
            Engine::Auto => {
                let threshold =
                    self.gemm_threshold
                        .saturating_mul(if T::NEEDS_PACKING { 2 } else { 1 });
                nb >= 32
                    && nq.saturating_mul(dim) >= threshold
                    && eligible as f64 >= self.min_gemm_density as f64 * nq as f64 * nb as f64
            }
        }
    }
}

/// Per-query counts. No narrowing to the legacy u32 SearchStats is performed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Evaluated vector pairs, including masked GEMM pairs and direct recomputations.
    pub cmps: u64,
    /// Candidates admitted by the filter, including ones whose score is NaN.
    pub eligible_candidates: u64,
    pub result_count: u64,
    pub gemm_tiles: u64,
    pub direct_tiles: u64,
    pub recomputations: u64,
}

impl BatchStats {
    #[track_caller]
    fn accumulate(&mut self, other: Self) -> ANNResult<()> {
        macro_rules! add {
            ($($field:ident),*) => { $(self.$field = self.$field.checked_add(other.$field)
                .ok_or_else(|| ANNError::message("flat search statistics overflow"))?;)* };
        }
        add!(
            cmps,
            eligible_candidates,
            gemm_tiles,
            direct_tiles,
            recomputations
        );
        Ok(())
    }
}

#[derive(Debug)]
pub struct SearchResult<I> {
    pub neighbors: Vec<Neighbor<I>>,
    pub stats: BatchStats,
}

/// Search selected rows using the same smaller-is-better metrics as the visitor.
/// Filtering precedes Top-k; equal scores prefer later selection positions.
///
/// Norms may be cached on Dataset for L2/cosine. GEMM changes accumulation order:
/// scores and rankings near a floating-point tie need not be bit-identical to Direct.
/// Direct preserves the original type's kernel, including integer cosine behavior.
///
/// # Errors
/// Returns an error for mismatched dimensions, invalid options, size/statistics
/// overflow, or a failed GEMM. All work executes in the supplied pool.
#[track_caller]
pub fn knn_search<T, I, F>(
    data: Dataset<'_, T, I>,
    queries: rowmajor::Ref<'_, T>,
    k: NonZeroUsize,
    metric: Metric,
    filter: &F,
    options: &BatchOptions,
    pool: &ThreadPool,
) -> ANNResult<Vec<SearchResult<I>>>
where
    T: GemmElement,
    I: VectorId,
    F: CandidateFilter<I> + ?Sized,
{
    kernel::search(data, queries, k, metric, filter, options, pool, 0)
        .map(|results| results.into_iter().map(Partial::finish).collect())
}

/// Compatibility entry point: dense FP32 negative inner product and u32 row IDs.
///
/// # Errors
/// Returns an error for mismatched dimensions, row IDs exceeding u32, invalid
/// scratch sizes, or a failed GEMM. The worker limit defaults to one.
#[track_caller]
pub fn knn_inner_product(
    data: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: NonZeroUsize,
    options: &GemmOptions,
    pool: &ThreadPool,
) -> ANNResult<Vec<Vec<Neighbor<u32>>>> {
    if data.nrows() > u32::MAX as usize {
        return Err(ANNError::message("database row IDs must fit in u32"));
    }
    knn_search(
        Dataset::sequential(data),
        queries,
        k,
        Metric::InnerProduct,
        &AllCandidates,
        &BatchOptions {
            engine: Engine::Gemm,
            gemm: *options,
            ..Default::default()
        },
        pool,
    )
    .map(|results| results.into_iter().map(|r| r.neighbors).collect())
}

#[derive(Debug)]
struct Partial<I> {
    // Ordinals survive partitioning and streaming. IDs are deliberately not tie keys.
    neighbors: Vec<Neighbor<(usize, I)>>,
    stats: BatchStats,
}

impl<I: Copy> Partial<I> {
    fn empty() -> Self {
        Self {
            neighbors: Vec::new(),
            stats: BatchStats::default(),
        }
    }

    fn merge(&mut self, other: Self, k: usize) -> ANNResult<()> {
        self.stats.accumulate(other.stats)?;
        self.neighbors.extend(other.neighbors);
        if self.neighbors.len() > k {
            self.neighbors.select_nth_unstable_by(k - 1, compare);
            self.neighbors.truncate(k);
        }
        Ok(())
    }

    fn finish(mut self) -> SearchResult<I> {
        self.neighbors.sort_unstable_by(compare);
        self.stats.result_count = self.neighbors.len() as u64;
        SearchResult {
            neighbors: self
                .neighbors
                .into_iter()
                .map(|n| Neighbor::new(n.id().1, *n.distance()))
                .collect(),
            stats: self.stats,
        }
    }
}

fn compare<I>(a: &Neighbor<(usize, I)>, b: &Neighbor<(usize, I)>) -> Ordering {
    a.distance()
        .partial_cmp(b.distance())
        .unwrap_or(Ordering::Equal)
        .then_with(|| b.id().0.cmp(&a.id().0))
}

fn partition_start(rows: usize, workers: usize, worker: usize) -> usize {
    (rows / workers) * worker + worker.min(rows % workers)
}

#[track_caller]
fn scratch_len(queries: usize, vectors: usize) -> ANNResult<usize> {
    let len = queries
        .checked_mul(vectors)
        .ok_or_else(|| ANNError::message("GEMM scratch dimensions overflow"))?;
    check_len::<f32>(len)?;
    Ok(len)
}

#[track_caller]
fn check_len<T>(len: usize) -> ANNResult<()> {
    if len > isize::MAX as usize / size_of::<T>().max(1) {
        return Err(ANNError::message("GEMM scratch dimensions overflow"));
    }
    Ok(())
}

#[cfg(test)]
mod batch_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        data: rowmajor::Ref<'_, f32>,
        queries: rowmajor::Ref<'_, f32>,
        k: usize,
    ) -> Vec<Vec<(u32, f32)>> {
        queries
            .rows()
            .map(|query| {
                let mut candidates: Vec<_> = data
                    .rows()
                    .enumerate()
                    .filter_map(|(id, vector)| {
                        let score = -query
                            .iter()
                            .zip(vector)
                            .map(|(&x, &y)| f64::from(x) * f64::from(y))
                            .sum::<f64>() as f32;
                        (!score.is_nan()).then_some((id as u32, score))
                    })
                    .collect();
                candidates.sort_unstable_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap().then_with(|| b.0.cmp(&a.0))
                });
                candidates.truncate(k);
                candidates
            })
            .collect()
    }

    #[test]
    fn tiles_partitions_and_collectors_match_exhaustive_search() {
        for (threads, max_workers) in [(1, 8), (3, 1), (3, 2), (3, 3), (3, 8)] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for dim in [0, 1, 7, 32, 63, 768] {
                // Dyadic inputs keep the reference scores exactly representable.
                let data = rowmajor::Owned::from_fn(137, dim, |rc| {
                    ((rc.row * 3 + rc.col * 7) % 17) as f32 / 4.0 - 2.0
                });
                let queries = rowmajor::Owned::from_fn(5, dim, |rc| {
                    ((rc.row * 5 + rc.col * 11) % 13) as f32 / 8.0 - 1.0
                });
                for k in [1, 2, 100, 200] {
                    for threshold in [0, usize::MAX] {
                        let options = GemmOptions {
                            max_workers: NonZeroUsize::new(max_workers).unwrap(),
                            query_block_size: NonZeroUsize::new(2).unwrap(),
                            database_block_size: NonZeroUsize::new(17).unwrap(),
                            collection: SearchOptions {
                                reservoir_min_k: threshold,
                            },
                        };
                        let actual = knn_inner_product(
                            data.as_view(),
                            queries.as_view(),
                            NonZeroUsize::new(k).unwrap(),
                            &options,
                            &pool,
                        )
                        .unwrap();
                        let actual: Vec<Vec<_>> = actual
                            .into_iter()
                            .map(|row| row.into_iter().map(Neighbor::as_tuple).collect())
                            .collect();
                        assert_eq!(
                            actual,
                            reference(data.as_view(), queries.as_view(), k),
                            "threads={threads}, max_workers={max_workers}, dim={dim}, k={k}, threshold={threshold}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn request_worker_limits_with_a_shared_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        assert_eq!(
            GemmOptions::default().worker_count(&pool, NonZeroUsize::MAX),
            1
        );
        for (limit, rows, expected) in [
            (1, 17, 1),
            (2, 17, 2),
            (4, 17, 4),
            (8, 17, 4),
            (8, 3, 3),
            (8, 1, 1),
        ] {
            let options = GemmOptions {
                max_workers: NonZeroUsize::new(limit).unwrap(),
                ..Default::default()
            };
            assert_eq!(
                options.worker_count(&pool, NonZeroUsize::new(rows).unwrap()),
                expected
            );
        }
        assert_eq!(pool.current_num_threads(), 4);
    }

    #[test]
    fn empty_inputs_and_dimension_errors() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for (nb, nq, dim) in [(0, 0, 0), (0, 3, 5), (3, 0, 5), (1, 3, 0)] {
            let data = rowmajor::Owned::from_element(nb, dim, 1.0);
            let queries = rowmajor::Owned::from_element(nq, dim, 1.0);
            let actual = knn_inner_product(
                data.as_view(),
                queries.as_view(),
                NonZeroUsize::MAX,
                &GemmOptions::default(),
                &pool,
            )
            .unwrap();
            assert_eq!(actual.len(), nq);
            assert!(actual.iter().all(|row| row.len() == nb));
        }
        let data = rowmajor::Owned::from_element(1, 3, 1.0);
        let queries = rowmajor::Owned::from_element(1, 2, 1.0);
        let error = knn_inner_product(
            data.as_view(),
            queries.as_view(),
            NonZeroUsize::MIN,
            &GemmOptions::default(),
            &pool,
        )
        .unwrap_err();
        assert!(error.to_string().contains("dimensions differ"));
    }

    #[test]
    fn non_finite_scores_and_ties_across_partitions() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let values = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            1.0,
            -0.0,
            0.0,
        ];
        let data = rowmajor::Owned::from_fn(values.len(), 1, |rc| values[rc.row]);
        let query_values = [1.0, -1.0, 0.0, f32::NAN];
        let queries = rowmajor::Owned::from_fn(query_values.len(), 1, |rc| query_values[rc.row]);
        for max_workers in [1, 3] {
            let options = GemmOptions {
                max_workers: NonZeroUsize::new(max_workers).unwrap(),
                ..Default::default()
            };
            for k in [1, 2, 10] {
                let actual = knn_inner_product(
                    data.as_view(),
                    queries.as_view(),
                    NonZeroUsize::new(k).unwrap(),
                    &options,
                    &pool,
                )
                .unwrap();
                let actual: Vec<Vec<_>> = actual
                    .into_iter()
                    .map(|row| row.into_iter().map(Neighbor::as_tuple).collect())
                    .collect();
                assert_eq!(actual, reference(data.as_view(), queries.as_view(), k));
            }
        }
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rejects_database_ids_that_do_not_fit() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let data = rowmajor::Owned::from_element(u32::MAX as usize + 1, 0, 0.0);
        let queries = rowmajor::Owned::from_element(1, 0, 0.0);
        let error = knn_inner_product(
            data.as_view(),
            queries.as_view(),
            NonZeroUsize::MIN,
            &GemmOptions::default(),
            &pool,
        )
        .unwrap_err();
        assert!(error.to_string().contains("fit in u32"));
    }

    #[test]
    fn scratch_dimensions_are_checked_before_allocation() {
        let data = rowmajor::Owned::from_element(2, 0, 0.0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let options = GemmOptions {
            query_block_size: NonZeroUsize::MAX,
            ..Default::default()
        };
        for nq in [usize::MAX, isize::MAX as usize] {
            let queries = rowmajor::Owned::from_element(nq, 0, 0.0);
            let error = knn_inner_product(
                data.as_view(),
                queries.as_view(),
                NonZeroUsize::MIN,
                &options,
                &pool,
            )
            .unwrap_err();
            assert!(error.to_string().contains("scratch dimensions overflow"));
        }
    }
}
