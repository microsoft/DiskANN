/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense FP32 inner-product search using blocked GEMM and an explicit thread pool.
//!
//! Database partitions are scanned in parallel, with one collector per query per
//! partition. Only a query tile and database tile's scores are materialized at a time.
//! The scalar visitor API remains appropriate for streaming or filtered providers.

use std::{cmp::Ordering, num::NonZeroUsize, ops::Range};

use diskann_linalg::{Transpose, sgemm};
use diskann_utils::views::rowmajor::{self, Matrix};
use rayon::ThreadPool;

use super::{SearchOptions, topk::BatchCollector};
use crate::{ANNError, ANNResult, neighbor::Neighbor};

/// Tile sizes bound score scratch space per worker to the product of the two
/// block sizes, in floats, independent of the corpus size.
#[derive(Debug, Clone, Copy)]
pub struct GemmOptions {
    /// Maximum workers used by one request, capped by the pool size and database
    /// row count. Defaults to one; batching still uses GEMM with this limit.
    pub max_workers: NonZeroUsize,
    pub query_block_size: NonZeroUsize,
    pub database_block_size: NonZeroUsize,
    pub collection: SearchOptions,
}

impl Default for GemmOptions {
    #[expect(
        clippy::unwrap_used,
        reason = "the default block sizes are nonzero literals"
    )]
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

/// Exhaustively search each query using negative inner product (smaller is better).
///
/// IDs are database row numbers. Results follow query order, with at most k
/// neighbors per query sorted by distance, then by descending row ID for ties.
/// NaN scores are ignored. Empty databases return an empty result for each query;
/// zero-dimensional vectors all have distance -0.0.
///
/// All work runs in the supplied pool, subject to the per-request worker limit.
/// Individual GEMMs are sequential to avoid nested thread pools.
/// Local top-k results are merged only after every
/// partition completes. Memory includes the final results, per-worker tile scores
/// and per-worker top-k collectors. No full query-by-database matrix is allocated.
///
/// GEMM may accumulate products in a different order than scalar distance kernels.
/// Distances, and rankings near a floating-point tie, can consequently differ.
///
/// # Errors
///
/// Returns an error for mismatched dimensions, more database rows than fit in a
/// u32, or overflowing scratch dimensions. GEMM errors are propagated.
#[track_caller]
pub fn knn_inner_product(
    data: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: NonZeroUsize,
    options: &GemmOptions,
    pool: &ThreadPool,
) -> ANNResult<Vec<Vec<Neighbor<u32>>>> {
    if data.ncols() != queries.ncols() {
        return Err(ANNError::message("database and query dimensions differ"));
    }
    if data.nrows() > u32::MAX as usize {
        return Err(ANNError::message("database row IDs must fit in u32"));
    }
    let Some(rows) = NonZeroUsize::new(data.nrows()) else {
        return Ok((0..queries.nrows()).map(|_| Vec::new()).collect());
    };
    if queries.nrows() == 0 {
        return Ok(Vec::new());
    }

    // Clamp before allocating so k larger than the database does not overallocate.
    let k = k.min(rows);
    let workers = options.worker_count(pool, rows);
    scratch_len(
        options.query_block_size.get().min(queries.nrows()),
        options
            .database_block_size
            .get()
            .min(data.nrows().div_ceil(workers)),
    )?;
    let mut output = Vec::with_capacity(queries.nrows());
    for q0 in (0..queries.nrows()).step_by(options.query_block_size.get()) {
        let nq = options.query_block_size.get().min(queries.nrows() - q0);
        let dim = data.ncols();
        let query_tile = &queries.as_slice()[q0 * dim..(q0 + nq) * dim];
        let mut partials: Vec<ANNResult<Vec<Vec<Neighbor<u32>>>>> =
            (0..workers).map(|_| Ok(Vec::new())).collect();
        pool.scope(|scope| {
            for (worker, result) in partials.iter_mut().enumerate() {
                let begin = partition_start(data.nrows(), workers, worker);
                let end = partition_start(data.nrows(), workers, worker + 1);
                scope.spawn(move |_| {
                    *result = scan_partition(data, query_tile, nq, begin..end, k, options);
                });
            }
        });

        let mut merged: Vec<Vec<Neighbor<u32>>> = (0..nq).map(|_| Vec::new()).collect();
        for result in partials {
            for (dst, candidates) in merged.iter_mut().zip(result?) {
                dst.extend(candidates);
            }
        }
        // Every global top-k candidate belongs to its partition's local top-k.
        // Global row IDs recover the scan-order tie rule regardless of task order.
        for neighbors in &mut merged {
            let compare = |a: &Neighbor<u32>, b: &Neighbor<u32>| {
                a.distance()
                    .partial_cmp(b.distance())
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| b.id().cmp(a.id()))
            };
            if neighbors.len() > k.get() {
                neighbors.select_nth_unstable_by(k.get() - 1, compare);
                neighbors.truncate(k.get());
            }
            neighbors.sort_unstable_by(compare);
        }
        output.extend(merged);
    }
    Ok(output)
}

fn partition_start(rows: usize, workers: usize, worker: usize) -> usize {
    (rows / workers) * worker + worker.min(rows % workers)
}

#[track_caller]
fn scratch_len(queries: usize, vectors: usize) -> ANNResult<usize> {
    queries
        .checked_mul(vectors)
        .filter(|&len| len <= isize::MAX as usize / size_of::<f32>())
        .ok_or_else(|| ANNError::message("GEMM scratch dimensions overflow"))
}

fn scan_partition(
    data: rowmajor::Ref<'_, f32>,
    queries: &[f32],
    nq: usize,
    range: Range<usize>,
    k: NonZeroUsize,
    options: &GemmOptions,
) -> ANNResult<Vec<Vec<Neighbor<u32>>>> {
    let Some(rows) = NonZeroUsize::new(range.len()) else {
        return Ok((0..nq).map(|_| Vec::new()).collect());
    };
    let block = options.database_block_size.get().min(range.len());
    let mut scores = vec![0.0; scratch_len(nq, block)?];
    let mut ids = vec![0; block];
    let local_k = k.min(rows);
    let mut collectors: Vec<_> = (0..nq)
        .map(|_| BatchCollector::new(local_k, options.collection.reservoir_min_k))
        .collect();
    let dim = data.ncols();
    for start in range.clone().step_by(block) {
        let len = block.min(range.end - start);
        let scores = &mut scores[..nq * len];
        if dim == 0 {
            scores.fill(-0.0);
        } else {
            sgemm(
                Transpose::None,
                Transpose::Ordinary,
                nq,
                len,
                dim,
                -1.0,
                queries,
                &data.as_slice()[start * dim..(start + len) * dim],
                None,
                scores,
            )
            .map_err(|error| ANNError::new(error).context("flat inner-product GEMM"))?;
        }
        for (offset, id) in ids[..len].iter_mut().enumerate() {
            *id = (start + offset) as u32;
        }
        for (collector, distances) in collectors.iter_mut().zip(scores.chunks_exact(len)) {
            collector.add_batch(&ids[..len], distances);
        }
    }
    Ok(collectors.into_iter().map(BatchCollector::finish).collect())
}

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
        let options = GemmOptions {
            max_workers: NonZeroUsize::new(3).unwrap(),
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
