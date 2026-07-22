/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Main PiPNN builder: orchestrates partitioning, leaf building, and edge merging.
//!
//! Algorithm (from arXiv:2602.21247):
//! 1. G <- empty graph
//! 2. B <- Partition(X) via RBC
//! 3. For each leaf b_i in B (in parallel):
//!    edges <- Pick(b_i)  // GEMM + bi-directed k-NN
//!    Merge edges with HashPrune or an exact-deduplicated candidate accumulator
//! 4. Apply the shared Vamana RobustPrune kernel when configured
//! 5. return G

use std::time::Instant;

use diskann::{utils::VectorRepr, ANNError, ANNResult};
use rayon::prelude::*;

use crate::hash_prune::HashPrune;
use crate::leaf_build;
use crate::rayon_util::ParIterInstalled;
use crate::{direct_candidates::DirectCandidates, PiPNNBuildContext};

/// Build a PiPNN index from typed vector data.
///
/// Keeps data in its native type T and converts to f32 on-the-fly at each access point,
/// avoiding a full f32 copy of the dataset.
/// `data` is a flat slice of `T` in row-major order: npoints x ndims.
pub fn build_typed<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    ctx: &PiPNNBuildContext,
) -> ANNResult<Vec<Vec<u32>>> {
    if npoints == 0 || ndims == 0 {
        return Err(ANNError::log_dimension_mismatch_error(
            "PiPNN npoints and ndims must be > 0".into(),
        ));
    }
    if npoints > u32::MAX as usize {
        return Err(ANNError::log_index_config_error(
            "PiPNN npoints".into(),
            format!("npoints ({npoints}) exceeds the u32 graph ID limit"),
        ));
    }

    let expected_len = npoints.checked_mul(ndims).ok_or_else(|| {
        ANNError::log_dimension_mismatch_error(format!(
            "PiPNN dataset shape {npoints} x {ndims} overflows usize"
        ))
    })?;
    if data.len() != expected_len {
        return Err(ANNError::log_dimension_mismatch_error(format!(
            "PiPNN data length mismatch: expected {expected_len} elements ({npoints} x {ndims}), got {}",
            data.len()
        )));
    }

    let config = &ctx.config;
    tracing::info!(
        npoints = npoints,
        ndims = ndims,
        k = config.k,
        max_degree = ctx.max_degree.get(),
        c_max = config.c_max,
        replicas = config.replicas,
        "PiPNN build started (typed)"
    );

    // Always install a build-owned pool. Rayon treats zero as its automatic
    // thread count, so every parallel operation below has the same execution
    // contract regardless of whether the caller chose an explicit count.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.num_threads)
        .build()
        .map_err(|error| ANNError::log_thread_pool_error(error.to_string()))?;
    pool.install(|| build_in_pool(data, npoints, ndims, ctx))
}

/// Construct adjacency lists inside the pool installed by [`build_typed`].
fn build_in_pool<T: VectorRepr + Send + Sync>(
    data: &[T],
    npoints: usize,
    ndims: usize,
    ctx: &PiPNNBuildContext,
) -> ANNResult<Vec<Vec<u32>>> {
    let config = &ctx.config;
    let metric = ctx.metric;
    let max_degree = ctx.max_degree.get();
    let t_total = Instant::now();

    // Direct candidates bypass HashPrune and feed the shared final prune.
    let t0 = Instant::now();
    let hash_prune = (!config.skip_hash_prune)
        .then(|| {
            HashPrune::new(
                data,
                npoints,
                ndims,
                config.num_hash_planes,
                config.l_max,
                42,
            )
        })
        .transpose()?;
    let sketch_secs = t0.elapsed().as_secs_f64();
    if hash_prune.is_some() {
        tracing::info!(elapsed_secs = sketch_secs, "HashPrune init complete");
    }

    let mut partition_secs = 0.0f64;
    let mut leaf_build_secs = 0.0f64;

    // Replicas are a PARTITION concept (cf. gp-ann's dense ball carving,
    // `BuildApproximateNearestNeighborGraph`): each replica re-partitions the
    // full dataset with a different seed and its leaves are ACCUMULATED into one
    // shared pool. The leaf build (GEMM k-NN) and candidate merge then run ONCE
    // over the combined pool — not per replica — so a point's neighbors from
    // overlapping leaves across all replicas merge together before top-k.
    let mut leaves: Vec<crate::partition::Leaf> = Vec::new();
    for replica in 0..config.replicas {
        let seed = 1000 + replica as u64 * 7919;

        let t1 = Instant::now();
        let replica_leaves =
            crate::partition::partition(data, ndims, npoints, config, metric, seed)?;
        partition_secs += t1.elapsed().as_secs_f64();

        let total_pts: usize = replica_leaves.iter().map(|l| l.indices.len()).sum();
        tracing::info!(
            replica = replica,
            partition_secs = t1.elapsed().as_secs_f64(),
            num_leaves = replica_leaves.len(),
            avg_leaf_size = total_pts as f64 / replica_leaves.len().max(1) as f64,
            total_pts = total_pts,
            overlap = total_pts as f64 / npoints as f64,
            "Partition complete"
        );
        leaves.extend(replica_leaves);
    }
    let total_leaves = leaves.len();

    // Release partition scratch before leaf building so the two phases do not
    // contribute to peak RSS at the same time.
    let _ = rayon::broadcast(|_| crate::partition::release_thread_buffers());

    // Build all leaves (from every replica) in parallel, streaming edges to the
    // configured candidate accumulator. This is the one leaf-build + merge pass.
    let t2 = Instant::now();

    use std::sync::atomic::{AtomicUsize, Ordering};
    let total_edges = AtomicUsize::new(0);

    // Leaves processed in parallel via par_chunks. Each chunk shares one
    // thread-local buffer set, amortizing TLS + RefCell + Vec allocation
    // overhead across multiple leaves. Chunk size scales with leaf count
    // and rayon pool size so every thread gets ~4 work-stealing units.
    let nthreads = rayon::current_num_threads().max(1);
    let leaf_batch = (leaves.len() / (nthreads * 4)).clamp(1, 256);
    let direct_candidates = config
        .skip_hash_prune
        .then(|| DirectCandidates::new(npoints))
        .transpose()?;
    leaves
        .par_chunks(leaf_batch)
        .map(|chunk| -> ANNResult<()> {
            leaf_build::LEAF_BUFFERS.with(|cell| {
                let mut bufs = cell.borrow_mut();
                for leaf in chunk {
                    let group_edges = leaf_build::build_leaf_into(
                        data,
                        ndims,
                        &leaf.indices,
                        config.k,
                        metric,
                        &mut bufs,
                    )?;
                    total_edges.fetch_add(group_edges, Ordering::Relaxed);

                    let (edge_offsets, edges) = bufs.edges(group_edges);
                    if let Some(hash_prune) = &hash_prune {
                        hash_prune.add_leaf_edges(&leaf.indices, edge_offsets, edges);
                    } else if let Some(candidates) = &direct_candidates {
                        candidates.add_leaf_edges(&leaf.indices, edge_offsets, edges);
                    }
                }
                Ok(())
            })
        })
        .collect_installed::<ANNResult<()>>()?;
    // `leaves` is dropped right after the build to free the accumulated index
    // Vecs (~n·overlap u32) before extraction. (It is fully live *during* the
    // build — every leaf is an input — so incremental freeing can't lower the
    // leaf-build peak; the reduction target there is the reservoir slab, below.)
    drop(leaves);

    let total_edges_count = total_edges.load(Ordering::Relaxed);
    leaf_build_secs += t2.elapsed().as_secs_f64();

    tracing::info!(
        elapsed_secs = t2.elapsed().as_secs_f64(),
        total_leaves = total_leaves,
        total_edges = total_edges_count,
        "Leaf build and merge complete"
    );

    // Release thread-local leaf buffers so their arena pages can be reclaimed.
    let _ = rayon::broadcast(|_| leaf_build::release_thread_buffers());

    let finalize_started = Instant::now();
    let adjacency = match (direct_candidates, hash_prune, config.final_prune) {
        (Some(candidates), None, true) => {
            let candidates = candidates.into_rows()?;
            crate::graph_prune::prune_overfull_lists(
                data, ndims, candidates, max_degree, metric, ctx.alpha,
            )?
        }
        (None, Some(hash_prune), true) => {
            // HashPrune ends after exporting the candidates retained from leaf
            // builds. RobustPrune is a separate builder phase over that output.
            crate::graph_prune::prune_overfull_lists(
                data,
                ndims,
                hash_prune.into_candidate_lists(),
                max_degree,
                metric,
                ctx.alpha,
            )?
        }
        (None, Some(hash_prune), false) => hash_prune.into_nearest_lists(max_degree),
        (Some(_), Some(_), true)
        | (Some(_), Some(_), false)
        | (Some(_), None, false)
        | (None, None, true)
        | (None, None, false) => {
            return Err(crate::config_error(
                "invalid candidate-merge state after validated PiPNN build",
            ));
        }
    };
    tracing::info!(
        elapsed_secs = finalize_started.elapsed().as_secs_f64(),
        robust_prune = config.final_prune,
        "Graph finalization complete"
    );

    let total_secs = t_total.elapsed().as_secs_f64();

    let total_degree: usize = adjacency.iter().map(Vec::len).sum();
    let max_output_degree = adjacency.iter().map(Vec::len).max().unwrap_or(0);
    let isolated = adjacency
        .iter()
        .filter(|neighbors| neighbors.is_empty())
        .count();
    tracing::info!(
        total_secs,
        sketch_secs,
        partition_secs,
        leaf_build_secs,
        num_leaves = total_leaves,
        total_edges = total_edges_count,
        avg_degree = total_degree as f64 / adjacency.len() as f64,
        max_degree = max_output_degree,
        isolated,
        "PiPNN build complete"
    );

    Ok(adjacency)
}

#[cfg(test)]
mod tests;
