/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Randomized Ball Carving (RBC) partitioning — iterative, parallel at every level.
//!
//! Recursively partitions the dataset into overlapping clusters using an iterative
//! work-queue approach. All oversized clusters at each level are processed in parallel.

use crate::rayon_util::ParIterInstalled;
use crate::{PiPNNError, PiPNNResult};
use diskann::utils::VectorRepr;
use rand::prelude::IndexedRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Maximum supported `fanout` value: hard upper bound on the size of the
/// stack-allocated top-k tracker [`assign_to_leaders`] uses on its hot path.
/// Enforced by [`crate::PiPNNConfig::validate`].
pub const MAX_FANOUT: usize = 16;

/// A leaf partition containing indices into the original dataset.
///
/// Uses `u32` instead of `usize` to halve memory on 64-bit platforms.
/// Sufficient for datasets up to 4 billion points.
#[derive(Debug, Clone)]
pub struct Leaf {
    pub indices: Vec<u32>,
}

/// Configuration for RBC partitioning.
///
/// Fields are private; construct via [`PartitionConfig::new`], which enforces
/// the partition-layer invariants on `fanout` and `leader_cap`.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    c_max: usize,
    c_min: usize,
    p_samp: f64,
    fanout: Vec<usize>,
    metric: diskann_vector::distance::Metric,
    leader_cap: usize,
}

impl PartitionConfig {
    /// Construct a validated [`PartitionConfig`]. Returns an error if any
    /// partition-layer invariant is violated (see [`Self::validate_params`]).
    pub fn new(
        c_max: usize,
        c_min: usize,
        p_samp: f64,
        fanout: Vec<usize>,
        metric: diskann_vector::distance::Metric,
        leader_cap: usize,
    ) -> PiPNNResult<Self> {
        Self::validate_params(c_max, c_min, p_samp, &fanout, leader_cap)?;
        Ok(Self {
            c_max,
            c_min,
            p_samp,
            fanout,
            metric,
            leader_cap,
        })
    }

    /// Validate raw partition parameters without constructing.
    ///
    /// Owns the full set of partition-layer rules so that upstream config
    /// validators (e.g. [`crate::PiPNNConfig::validate`]) can fail-fast on bad
    /// inputs by calling this directly, while [`Self::new`] enforces the same
    /// rules at construction time.
    pub(crate) fn validate_params(
        c_max: usize,
        c_min: usize,
        p_samp: f64,
        fanout: &[usize],
        leader_cap: usize,
    ) -> PiPNNResult<()> {
        if c_max == 0 {
            return Err(PiPNNError::Config("c_max must be > 0".into()));
        }
        if c_min == 0 {
            return Err(PiPNNError::Config("c_min must be > 0".into()));
        }
        if c_min > c_max {
            return Err(PiPNNError::Config(format!(
                "c_min ({}) must be <= c_max ({})",
                c_min, c_max
            )));
        }
        if !p_samp.is_finite() {
            return Err(PiPNNError::Config("p_samp must be finite".into()));
        }
        if p_samp <= 0.0 || p_samp > 1.0 {
            return Err(PiPNNError::Config(format!(
                "p_samp ({}) must be in (0.0, 1.0]",
                p_samp
            )));
        }
        if fanout.is_empty() {
            return Err(PiPNNError::Config("fanout must not be empty".into()));
        }
        if fanout.contains(&0) {
            return Err(PiPNNError::Config("all fanout values must be > 0".into()));
        }
        if let Some(&over) = fanout.iter().find(|&&f| f > MAX_FANOUT) {
            return Err(PiPNNError::Config(format!(
                "fanout value {} exceeds MAX_FANOUT ({})",
                over, MAX_FANOUT
            )));
        }
        if leader_cap < 2 {
            return Err(PiPNNError::Config(format!(
                "leader_cap ({}) must be >= 2",
                leader_cap
            )));
        }
        Ok(())
    }
}

/// Compute the number of leaders to sample, capped by leader_cap.
#[inline]
fn sample_num_leaders(n: usize, p_samp: f64, leader_cap: usize) -> usize {
    ((n as f64 * p_samp).ceil() as usize)
        .clamp(2, leader_cap)
        .min(n)
}

use diskann_vector::topk::topk_insert;

/// A cluster that needs further partitioning.
struct WorkItem {
    indices: Vec<u32>,
    level: usize,
    seed: u64,
}

/// Partition the dataset into overlapping leaves.
///
/// Uses an iterative work-queue: each round processes all oversized clusters in parallel,
/// producing new work items for the next round. Completed leaves are pushed to a shared
/// result vec. No recursion, parallel at every level.
pub fn partition<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    npoints: usize,
    config: &PartitionConfig,
    seed: u64,
) -> Vec<Leaf> {
    let initial_indices: Vec<u32> = (0..npoints as u32).collect();

    if npoints <= config.c_max {
        return vec![Leaf {
            indices: initial_indices,
        }];
    }

    let nl0 = sample_num_leaders(npoints, config.p_samp, config.leader_cap);
    tracing::info!(npoints, leaders = nl0, ndims, "Partition start");

    let mut leaves: Vec<Leaf> = Vec::new();
    let mut work = vec![WorkItem {
        indices: initial_indices,
        level: 0,
        seed,
    }];

    let mut iteration = 0;
    while !work.is_empty() {
        iteration += 1;
        // 50 iterations is a safety net against a partitioning loop divergence.
        // Leaves halve in size each level, so 50 covers > 2^50 points in
        // healthy runs. If we hit this the partition config is degenerate.
        assert!(
            iteration <= 50,
            "partition diverged at iteration {} (npoints={}, c_max={}, c_min={}, fanout={:?})",
            iteration,
            npoints,
            config.c_max,
            config.c_min,
            config.fanout
        );

        let results: Vec<(Vec<WorkItem>, Vec<Leaf>)> = work
            .into_par_iter()
            .map(|item| partition_one_level(data, ndims, config, item))
            .collect_installed();

        let total_work: usize = results.iter().map(|(w, _)| w.len()).sum();
        let total_leaves: usize = results.iter().map(|(_, l)| l.len()).sum();
        let mut next_work = Vec::with_capacity(total_work);
        leaves.reserve(total_leaves);
        for (wi, lv) in results {
            next_work.extend(wi);
            leaves.extend(lv);
        }
        work = next_work;
    }

    // Global merge of sub-c_min leaves across all work items / levels.
    // Eliminates the bug where per-call `merge_small` would leave one
    // < c_min remainder per partition_one_level call (10s of thousands of
    // tiny leaves at deep BFS levels in the prior implementation).
    global_merge_small(leaves, config.c_min, config.c_max)
}

/// Process one cluster: assign to leaders, emit oversized clusters as new work
/// items and the rest (including under-c_min) as leaves. Cross-work-item small
/// leaves are then combined by a single global `global_merge_small` pass at
/// the end of `partition()` — replaces the per-call `merge_small` which left
/// one < c_min remainder per work item (58K+ tiny leaves at deep levels).
fn partition_one_level<T: VectorRepr + Send + Sync>(
    data: &[T],
    ndims: usize,
    config: &PartitionConfig,
    item: WorkItem,
) -> (Vec<WorkItem>, Vec<Leaf>) {
    let n = item.indices.len();
    debug_assert!(n > config.c_max);

    let fanout = config.fanout.get(item.level).copied().unwrap_or(1).min(n);
    let num_leaders = sample_num_leaders(n, config.p_samp, config.leader_cap);

    // Deterministic seed derived from parent: no syscall, reproducible.
    let seed = item
        .seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(n as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let leaders: Vec<u32> = item
        .indices
        .choose_multiple(&mut rng, num_leaders)
        .copied()
        .collect();

    // Assign each point to its `fanout` nearest leaders → per-leader clusters.
    let clusters = assign_to_leaders(data, ndims, &item.indices, &leaders, fanout, config.metric);

    let mut next_work = Vec::new();
    let mut finished_leaves = Vec::new();
    for cluster in clusters {
        if cluster.is_empty() {
            continue;
        }
        if cluster.len() <= config.c_max {
            finished_leaves.push(Leaf { indices: cluster });
        } else {
            next_work.push(WorkItem {
                indices: cluster,
                level: item.level + 1,
                seed,
            });
        }
    }
    (next_work, finished_leaves)
}

// ─── Thread-local stripe buffers ─────────────────────────────────────────────

use std::cell::RefCell;

/// Reusable per-thread buffers for partition stripe processing.
/// Avoids per-stripe alloc + memset on hot path.
struct StripeBuffers {
    p_data: Vec<f32>,
    dots: Vec<f32>,
}

impl StripeBuffers {
    fn new() -> Self {
        Self {
            p_data: Vec::new(),
            dots: Vec::new(),
        }
    }
}

thread_local! {
    static STRIPE_BUFS: RefCell<StripeBuffers> = RefCell::new(StripeBuffers::new());
}

/// SIMD batch compute of ||p_i||² for `np` rows of length `ndims`. Returns a
/// `Vec<f32>` of length `np`. Replaces process_row's per-call
/// `p_row.iter().map(|v| v*v).sum()` which showed up at 10% of partition
/// cycles in the c4-48 PMU profile (perf attributed it as `Map::next`).
fn compute_p_norm_sq_batch(p_data: &[f32], np: usize, ndims: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; np];
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use std::arch::x86_64::*;
        let chunks = ndims / 16;
        let tail = ndims - chunks * 16;
        unsafe {
            for i in 0..np {
                let p = p_data.as_ptr().add(i * ndims);
                let mut acc = _mm512_setzero_ps();
                for c in 0..chunks {
                    let v = _mm512_loadu_ps(p.add(c * 16));
                    acc = _mm512_fmadd_ps(v, v, acc);
                }
                if tail > 0 {
                    let kmask: u16 = (1u16 << tail) - 1;
                    let v = _mm512_maskz_loadu_ps(kmask, p.add(chunks * 16));
                    acc = _mm512_fmadd_ps(v, v, acc);
                }
                *out.get_unchecked_mut(i) = _mm512_reduce_add_ps(acc);
            }
        }
        return out;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        for i in 0..np {
            let row = &p_data[i * ndims..(i + 1) * ndims];
            out[i] = row.iter().map(|v| v * v).sum();
        }
        out
    }
}

// ─── Assignment ──────────────────────────────────────────────────────────────

/// Assign each point to its `fanout` nearest leaders using native SIMD distance.
/// Point-by-point: no large temporary matrix, works directly on native type T.
/// All indices are u32 global IDs. Returns per-leader clusters as Vec<Vec<u32>>.
fn assign_to_leaders<T: VectorRepr + Send + Sync + 'static>(
    data: &[T],
    ndims: usize,
    points: &[u32],
    leaders: &[u32],
    fanout: usize,
    metric: diskann_vector::distance::Metric,
) -> Vec<Vec<u32>> {
    let np = points.len();
    let nl = leaders.len();
    let num_assign = fanout.min(nl);

    use diskann_vector::distance::Metric;

    // Extract leader data into contiguous f32 array.
    let mut l_data = vec![0.0f32; nl * ndims];
    for (i, &idx) in leaders.iter().enumerate() {
        let src = &data[idx as usize * ndims..(idx as usize + 1) * ndims];
        T::as_f32_into(src, &mut l_data[i * ndims..(i + 1) * ndims]).expect("f32 conversion");
    }

    // Precompute leader norms for L2/Cosine.
    let l_norms: Vec<f32> = match metric {
        Metric::L2 => l_data
            .chunks_exact(ndims)
            .map(|row| row.iter().map(|v| v * v).sum())
            .collect(),
        Metric::Cosine => l_data
            .chunks_exact(ndims)
            .map(|row| row.iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect(),
        Metric::CosineNormalized | Metric::InnerProduct => Vec::new(),
    };

    // Flat assignments.
    let mut assignments = vec![0u32; np * num_assign];

    // Single-layer chunking with runtime MB sized to the detected L2 cache.
    // Matches v2's `assign_to_leaders_v2` structure: par_chunks_mut at MB
    // granularity, no inner mini-batch loop. The empirical ablation showed
    // the old stripe+inner-MB structure is equivalent at the same closure
    // body size — keeping the simpler single-layer form to avoid the dead
    // codegen bloat that previously bound v1 to its specific chunk grain.
    let l2 = crate::partition_inner::l2_size_bytes();
    let mb = crate::partition_inner::compute_mb(nl, ndims, l2);

    // Skip-MB path: whole problem comfortably fits L2 → one sequential GEMM.
    if crate::partition_inner::should_skip_mb(np, nl, l2) {
        STRIPE_BUFS.with(|cell| {
            let mut bufs = cell.borrow_mut();
            if bufs.p_data.len() < np * ndims {
                bufs.p_data.resize(np * ndims, 0.0);
            }
            if bufs.dots.len() < np * nl {
                bufs.dots.resize(np * nl, 0.0);
            }
            let StripeBuffers { ref mut p_data, ref mut dots } = *bufs;
            let p_slice = &mut p_data[..np * ndims];
            let dots_slice = &mut dots[..np * nl];
            for (i, &idx) in points.iter().enumerate() {
                let src = &data[idx as usize * ndims..(idx as usize + 1) * ndims];
                T::as_f32_into(src, &mut p_slice[i * ndims..(i + 1) * ndims])
                    .expect("f32 conversion");
            }
            diskann_linalg::sgemm_abt(p_slice, np, ndims, &l_data, nl, dots_slice);
            // Batch-precompute ||p||² for all rows in one tight SIMD loop.
            // Previously process_row computed this per call via
            //   `p_row.iter().map(|v| v*v).sum()` which showed up as 10% of
            // partition cycles in the c4-48 PMU profile (Map::next).
            let p_norm_sq = compute_p_norm_sq_batch(p_slice, np, ndims);
            for i in 0..np {
                let dot_row = &dots_slice[i * nl..(i + 1) * nl];
                let out = &mut assignments[i * num_assign..(i + 1) * num_assign];
                crate::partition_inner::process_row(
                    dot_row, p_norm_sq[i], &l_norms, metric, num_assign, out,
                );
            }
        });
    } else {
        // Chunked path: par_chunks_mut at MB granularity, one GEMM per chunk,
        // top-k per row via process_row. STRIPE_BUFS provides per-thread
        // p_data / dots reuse across many chunks.
        let chunk_size = mb * num_assign;
        assignments
            .par_chunks_mut(chunk_size)
            .enumerate()
            .for_each_installed(|(idx, assign_chunk)| {
                STRIPE_BUFS.with(|cell| {
                    let mut bufs = cell.borrow_mut();
                    let row_start = idx * mb;
                    let chunk_rows = (row_start + mb).min(np) - row_start;
                    if bufs.p_data.len() < chunk_rows * ndims {
                        bufs.p_data.resize(mb * ndims, 0.0);
                    }
                    if bufs.dots.len() < chunk_rows * nl {
                        bufs.dots.resize(mb * nl, 0.0);
                    }
                    let StripeBuffers { ref mut p_data, ref mut dots } = *bufs;
                    let p_slice = &mut p_data[..chunk_rows * ndims];
                    let dots_slice = &mut dots[..chunk_rows * nl];
                    for (i, &gi) in points[row_start..row_start + chunk_rows].iter().enumerate() {
                        let src = &data[gi as usize * ndims..(gi as usize + 1) * ndims];
                        T::as_f32_into(src, &mut p_slice[i * ndims..(i + 1) * ndims])
                            .expect("f32 conversion");
                    }
                    diskann_linalg::sgemm_abt(p_slice, chunk_rows, ndims, &l_data, nl, dots_slice);
                    let p_norm_sq = compute_p_norm_sq_batch(p_slice, chunk_rows, ndims);
                    for i in 0..chunk_rows {
                        let dot_row = &dots_slice[i * nl..(i + 1) * nl];
                        let out = &mut assign_chunk[i * num_assign..(i + 1) * num_assign];
                        crate::partition_inner::process_row(
                            dot_row, p_norm_sq[i], &l_norms, metric, num_assign, out,
                        );
                    }
                });
            });
    }


    // Aggregate into per-leader clusters using global point IDs. For large
    // np, the serial scatter (np × num_assign pushes into per-leader Vecs)
    // is a real serial tail — parallelize via per-thread partial clusters
    // + per-leader merge.
    if np >= 100_000 {
        let n_threads = rayon::current_num_threads().max(1);
        let chunk_size = np.div_ceil(n_threads);

        let partials: Vec<Vec<Vec<u32>>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk_size;
                let end = ((t + 1) * chunk_size).min(np);
                let mut local: Vec<Vec<u32>> = (0..nl)
                    .map(|_| Vec::with_capacity(((end - start) * num_assign).div_ceil(nl)))
                    .collect();
                for i in start..end {
                    let pt = points[i];
                    let row = &assignments[i * num_assign..(i + 1) * num_assign];
                    for &leader_local in row {
                        local[leader_local as usize].push(pt);
                    }
                }
                local
            })
            .collect_installed();

        // Sum per-leader sizes once so each final Vec is allocated with
        // exact capacity (no realloc churn).
        let mut sizes = vec![0usize; nl];
        for partial in &partials {
            for (li, v) in partial.iter().enumerate() {
                sizes[li] += v.len();
            }
        }

        let clusters: Vec<Vec<u32>> = sizes
            .par_iter()
            .enumerate()
            .map(|(li, &sz)| {
                let mut out = Vec::with_capacity(sz);
                for partial in &partials {
                    out.extend_from_slice(&partial[li]);
                }
                out
            })
            .collect_installed();
        clusters
    } else {
        let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); nl];
        for (i, pt) in points.iter().enumerate() {
            let row = &assignments[i * num_assign..(i + 1) * num_assign];
            for &leader_local in row {
                clusters[leader_local as usize].push(*pt);
            }
        }
        clusters
    }
}

// ─── Global Merge of Small Leaves ────────────────────────────────────────────

/// Combine sub-c_min leaves into c_min/c_max-sized leaves via a HashSet buffer
/// that deduplicates overlapping point IDs. Big leaves (>= c_min) pass through
/// untouched. The final remainder, if still < c_min, is appended to the most
/// recent big leaf when it fits, otherwise emitted as a small tail leaf.
fn global_merge_small(leaves: Vec<Leaf>, c_min: usize, c_max: usize) -> Vec<Leaf> {
    let (mut good, small): (Vec<Leaf>, Vec<Leaf>) =
        leaves.into_iter().partition(|l| l.indices.len() >= c_min);

    if small.is_empty() {
        return good;
    }

    let mut buf = std::collections::HashSet::<u32>::with_capacity(c_max);
    for leaf in small {
        if buf.len() + leaf.indices.len() > c_max && buf.len() >= c_min {
            good.push(Leaf {
                indices: buf.drain().collect(),
            });
        }
        buf.extend(leaf.indices);
        if buf.len() >= c_min {
            good.push(Leaf {
                indices: buf.drain().collect(),
            });
        }
    }
    if !buf.is_empty() {
        let remainder: Vec<u32> = buf.into_iter().collect();
        if remainder.len() < c_min {
            // Try to attach to the last big leaf if it fits.
            if let Some(last) = good.last_mut() {
                if last.indices.len() + remainder.len() <= c_max {
                    last.indices.extend(remainder);
                    return good;
                }
            }
        }
        good.push(Leaf { indices: remainder });
    }

    good
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_basic() {
        let npoints = 1000;
        let ndims = 8;
        let data: Vec<f32> = {
            use rand::{Rng, SeedableRng};
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            (0..npoints * ndims).map(|_| rng.random::<f32>()).collect()
        };
        let config = PartitionConfig {
            c_max: 64,
            c_min: 16,
            p_samp: 0.1,
            fanout: vec![4, 2],
            metric: diskann_vector::distance::Metric::L2,
            leader_cap: 1000,
        };
        let leaves = partition(&data, ndims, npoints, &config, 123);

        // All points should appear at least once (overlapping partitions).
        let mut seen = vec![false; npoints];
        for leaf in &leaves {
            assert!(leaf.indices.len() <= config.c_max, "leaf too large");
            for &idx in &leaf.indices {
                seen[idx as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some points missing");
    }

    #[test]
    fn test_partition_small_dataset() {
        let npoints = 50;
        let ndims = 4;
        let data: Vec<f32> = vec![1.0; npoints * ndims];
        let config = PartitionConfig {
            c_max: 64,
            c_min: 8,
            p_samp: 0.1,
            fanout: vec![3],
            metric: diskann_vector::distance::Metric::L2,
            leader_cap: 1000,
        };
        let leaves = partition(&data, ndims, npoints, &config, 0);
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].indices.len(), npoints);
    }

    // Baseline partition rules (c_max > 0, c_min <= c_max, p_samp range,
    // fanout non-empty / non-zero) are covered via PiPNNConfig::validate
    // delegate path in `builder::tests::test_config_validate`. Only the
    // partition-specific rules introduced alongside `validate_params` are
    // tested here.

    #[test]
    fn validate_params_rejects_fanout_above_max() {
        let err = PartitionConfig::validate_params(1024, 256, 0.05, &[MAX_FANOUT + 1], 1000)
            .expect_err("fanout > MAX_FANOUT must reject");
        assert!(format!("{err}").contains("MAX_FANOUT"));
        // Boundary: exactly MAX_FANOUT is accepted.
        PartitionConfig::validate_params(1024, 256, 0.05, &[MAX_FANOUT], 1000)
            .expect("fanout == MAX_FANOUT must accept");
    }

    #[test]
    fn validate_params_rejects_leader_cap_below_two() {
        for bad in [0usize, 1] {
            let err = PartitionConfig::validate_params(1024, 256, 0.05, &[10, 3], bad)
                .expect_err(&format!("leader_cap={bad} must reject"));
            assert!(format!("{err}").contains("leader_cap"));
        }
        // Boundary: exactly 2 is accepted (matches sample_num_leaders clamp floor).
        PartitionConfig::validate_params(1024, 256, 0.05, &[10, 3], 2)
            .expect("leader_cap == 2 must accept");
    }

    #[test]
    fn new_propagates_validate_params_error() {
        // Contract test for the public constructor: external callers (e.g.
        // benches) hit `new` rather than going through PiPNNConfig::validate.
        let err = PartitionConfig::new(
            1024,
            256,
            0.05,
            vec![MAX_FANOUT + 1],
            diskann_vector::distance::Metric::L2,
            1000,
        )
        .expect_err("PartitionConfig::new must propagate validate_params errors");
        assert!(format!("{err}").contains("MAX_FANOUT"));
    }
}
