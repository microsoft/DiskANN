/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Ablation: how well does the in-memory centroid graph find the *true* nearest
//! centroids?
//!
//! This isolates centroid-graph quality from the rest of the graph-IVF pipeline.
//! It rebuilds the same Vamana graph the index builds over the centroids, then
//! for each query compares:
//!
//! * **graph top-`nlist`** — the centroids the graph search returns (what the
//!   real index probes), against
//! * **brute-force top-`nlist`** — the exact nearest centroids by squared-L2.
//!
//! `recall@nlist = mean over queries of |graph ∩ brute_force| / nlist`. A recall
//! below 1.0 means the graph is steering the IVF toward the wrong lists, which
//! caps end-to-end recall no matter how the lists are scored.
//!
//! It reads the centroids written by a build (`*.graphivf_centroids.fbin`, f32)
//! and an fp16 query file, and mirrors the 16384-centroid search config
//! (graph degree 32 / slack 1.2 / L_build 64 / alpha 1.2, centroid_search_l
//! 1024, nlist 64..1024).
//!
//! Run (defaults target the enron 16384 index):
//! ```text
//! cargo run --release --example centroid_graph_ablation -- [centroids.fbin] [queries_fp16.bin]
//! ```

use std::{fs::File, num::NonZeroUsize};

use diskann::{
    graph::{
        config::{Builder, MaxDegree},
        search::Knn,
        search_output_buffer::{IdDistance, SearchOutputBuffer},
        DiskANNIndex,
    },
    ANNError,
};
use diskann_inmem::{layers, provider as inmem, Context, Provider, Strategy};
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_utils::{io::read_bin, views::Matrix};
use diskann_vector::{distance::Metric as VectorMetric, Half};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use tokio::runtime::Runtime;

/// Mirrors `centroids::CentroidGraph`.
type CentroidGraph = DiskANNIndex<Provider<layers::Full<f32>, u32>>;

// --- Configuration (matches the 16384-centroid build/search json) ------------

const DEFAULT_CENTROIDS: &str =
    "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4/graphivf_index_16384.graphivf_centroids.fbin";
const DEFAULT_QUERIES: &str =
    "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4/query_vector_normalized_dim_384_fp16_top_1000.bin";

const NLISTS: [usize; 5] = [64, 128, 256, 512, 1024];
const CENTROID_SEARCH_L: usize = 1024;

const GRAPH_DEGREE: usize = 32;
const GRAPH_SLACK: f32 = 1.2;
const GRAPH_L_BUILD: usize = 64;
const GRAPH_ALPHA: f32 = 1.2;
const NUM_THREADS: usize = 8;
/// Guard slots to provision when `NUM_THREADS` does not demand more.
const MIN_GUARD_SLOTS: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let centroids_path = args.next().unwrap_or_else(|| DEFAULT_CENTROIDS.to_string());
    let queries_path = args.next().unwrap_or_else(|| DEFAULT_QUERIES.to_string());

    // --- Load centroids (f32) and queries (fp16 -> f32) ----------------------
    let centroids: Matrix<f32> = read_bin(&mut File::open(&centroids_path)?)?;
    let nc = centroids.nrows();
    let dim = centroids.ncols();
    println!("centroids: {nc} x {dim}  ({centroids_path})");

    let queries_u16: Matrix<u16> = read_bin(&mut File::open(&queries_path)?)?;
    let nq = queries_u16.nrows();
    if queries_u16.ncols() != dim {
        return Err(format!("query dim {} != centroid dim {dim}", queries_u16.ncols()).into());
    }
    // Widen fp16 query bits to f32 (the centroid graph is full-precision f32).
    let queries: Vec<f32> = queries_u16
        .as_slice()
        .iter()
        .map(|&bits| Half::from_bits(bits).to_f32())
        .collect();
    println!("queries:   {nq} x {dim}  ({queries_path})");

    // Keep a flat copy of the centroids for brute force before the matrix is
    // moved into the graph builder.
    let centroid_data: Vec<f32> = centroids.as_slice().to_vec();

    // --- Build the centroid graph (identical config to the index) ------------
    let build_start = std::time::Instant::now();
    let index = build_centroid_graph(centroids)?;
    println!(
        "built centroid graph in {:.2}s\n",
        build_start.elapsed().as_secs_f64()
    );

    // --- Per query: graph top-k and brute-force top-k centroids --------------
    // The ablation pins the beam rather than deriving it from `SearchParams`,
    // so that every nlist below is served by one search: a single graph search
    // at k = l = max_nlist yields, by prefix, what each smaller nlist would
    // return.
    let max_nlist = *NLISTS.iter().max().unwrap();
    let l = CENTROID_SEARCH_L.max(max_nlist);

    let pool = create_thread_pool(NUM_THREADS)?;
    let search_start = std::time::Instant::now();
    let results: Vec<(Vec<u32>, Vec<u32>)> = (0..nq)
        .into_par_iter()
        .map_init(
            || {
                tokio::runtime::Builder::new_current_thread()
                    .build()
                    .expect("failed to build per-thread runtime")
            },
            |runtime, qi| {
                let query = &queries[qi * dim..(qi + 1) * dim];
                let graph_ids = graph_search(&index, runtime, query, max_nlist, l);
                let bf_ids = brute_force_topk(&centroid_data, nc, dim, query, max_nlist);
                (graph_ids, bf_ids)
            },
        )
        .collect_in_pool(pool.as_ref());
    println!(
        "ran {nq} queries (graph + brute force) in {:.2}s\n",
        search_start.elapsed().as_secs_f64()
    );

    // --- Recall@nlist: overlap of graph vs brute-force nearest centroids -----
    println!("Centroid-graph recall (graph nearest-centroids vs exact):");
    println!("  NList   Recall@NList");
    println!("  -----   ------------");
    for &nlist in NLISTS.iter() {
        let mut total = 0.0f64;
        for (graph_ids, bf_ids) in &results {
            let truth: std::collections::HashSet<u32> =
                bf_ids.iter().take(nlist).copied().collect();
            let hits = graph_ids
                .iter()
                .take(nlist)
                .filter(|id| truth.contains(id))
                .count();
            total += hits as f64 / nlist as f64;
        }
        let recall = total / nq as f64;
        println!("  {nlist:>5}   {:>10.4}", recall);
    }

    Ok(())
}

/// Squared-L2 brute force: indices of the `k` nearest centroids, ascending.
fn brute_force_topk(centroids: &[f32], nc: usize, dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = (0..nc)
        .map(|c| {
            let row = &centroids[c * dim..(c + 1) * dim];
            let dist: f32 = row
                .iter()
                .zip(query.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            (dist, c as u32)
        })
        .collect();
    let k = k.min(scored.len());
    scored.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
    scored.truncate(k);
    scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scored.into_iter().map(|(_, id)| id).collect()
}

/// Run a k-NN over the centroid graph; returns the returned centroid ids.
fn graph_search(
    index: &CentroidGraph,
    runtime: &Runtime,
    query: &[f32],
    k: usize,
    l: usize,
) -> Vec<u32> {
    let mut ids = vec![0u32; k];
    let mut dist = vec![0.0f32; k];
    let knn = Knn::new(l, None).expect("invalid knn params");
    let mut buffer = IdDistance::new(&mut ids, &mut dist);
    runtime
        .block_on(index.search(knn, &Strategy, &Context, query, &mut buffer))
        .expect("centroid graph search failed");
    let n = buffer.current_len();
    ids.truncate(n);
    ids
}

/// Build the in-memory full-precision Vamana graph over the centroids, mirroring
/// the index's `centroids::build`.
fn build_centroid_graph(centroids: Matrix<f32>) -> Result<CentroidGraph, ANNError> {
    let nc = centroids.nrows();
    let dim = centroids.ncols();

    let config = Builder::new_with(
        GRAPH_DEGREE,
        MaxDegree::slack(GRAPH_SLACK),
        GRAPH_L_BUILD,
        VectorMetric::L2.into(),
        |b| {
            b.alpha(GRAPH_ALPHA);
        },
    )
    .build()
    .map_err(ANNError::from)?;

    let mut provider_config = inmem::Config::new(nc, config.max_degree_u32().get() as usize);
    provider_config.set_epoch_guard_slots(NonZeroUsize::new(
        NUM_THREADS.saturating_mul(4).max(MIN_GUARD_SLOTS),
    ));

    // Start point = a real centroid, picked deterministically (same as the
    // index). The mean is a poor start point under inner product: it has the
    // smallest norm, so nothing selects it as a neighbor and it dead-ends.
    let start = StdRng::seed_from_u64(nc as u64).random_range(0..nc);

    let provider = Provider::new(
        layers::Full::<f32>::new(dim, VectorMetric::L2),
        provider_config,
        std::iter::once(centroids.row(start)),
    )
    .map_err(ANNError::new)?;

    let index = DiskANNIndex::new(config, provider, None);

    // Single inserts in parallel, matching `centroids::insert_centroids`: row
    // `i` is centroid `i` whichever slot it lands in.
    const CHUNK: usize = 64;
    let pool = create_thread_pool(NUM_THREADS)?;
    (0..nc).into_par_iter().chunks(CHUNK).try_for_each_in_pool(
        pool.as_ref(),
        |rows| -> Result<(), ANNError> {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .map_err(ANNError::from)?;
            runtime.block_on(async {
                for i in rows {
                    index
                        .insert(&Strategy, &Context, &(i as u32), centroids.row(i))
                        .await?;
                }
                Ok::<_, ANNError>(())
            })
        },
    )?;

    Ok(index)
}
