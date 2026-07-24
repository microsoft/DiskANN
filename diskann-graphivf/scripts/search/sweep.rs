/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Parameter sweep over `nlist` for an existing graph-IVF index.
//!
//! Search-only (the index must already be built). For each `nlist` the search
//! returns the top-1000 neighbors once and reports **both** recall@50 and
//! recall@1000, together with the mean / p95 / p99 per-query latency and the
//! mean bytes read from disk per query.
//!
//! Works with any stored element type produced by `build_online` / `build_static`
//! (`minmax8`, `f16`, `f32`): the type is auto-detected from the index metadata,
//! so no extra flag is needed. The queries `.bin` must be in that same element
//! type.
//!
//! Run (all args optional, shown with defaults):
//! ```text
//! cargo run --release --example sweep -- \
//!     <index_prefix> <nlist_csv> <num_threads> <queries.bin> <groundtruth.bin>
//! ```

use std::{
    fs::File,
    io::Read,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use bytemuck::Pod;
use diskann_graphivf::{GraphIvfIndex, Half, SearchParams, VectorRepr};
use diskann_providers::{
    common::MinMaxElement,
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_utils::{io::read_bin, views::Matrix};
use rayon::prelude::*;

const DATA_DIR: &str = "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4";

/// Search-list size for the centroid graph search. `effective_l` raises this to
/// `nlist` automatically when `nlist` is larger.
const CENTROID_SEARCH_L: usize = 1024;
/// Top-k retrieved per query; we evaluate recall at both 50 and 1000.
const TOPK: usize = 1000;
const RECALL_KS: [usize; 2] = [50, 1000];

/// Parsed command line; shared by every stored element type.
struct Config {
    out_prefix: String,
    nlists: Vec<usize>,
    num_threads: usize,
    queries_path: String,
    gt_path: String,
    fmt: &'static str,
}

/// Read the stored element size (bytes) recorded in `<prefix>.graphivf_meta`.
///
/// The metadata header begins `[magic u32][version u32][metric u32][element_size u32]`;
/// only the element size is needed here to pick the generic parameter, and
/// `GraphIvfIndex::load` re-validates the whole header afterwards.
fn index_element_size(prefix: &str) -> std::io::Result<usize> {
    let mut f = File::open(format!("{prefix}.graphivf_meta"))?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr)?;
    Ok(u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out_prefix = args
        .next()
        .unwrap_or_else(|| format!("{DATA_DIR}/graphivf_index_16384_minmax8"));
    let nlist_csv = args
        .next()
        .unwrap_or_else(|| "164,410,656,901,1147,1392,1638".to_string());
    let num_threads: usize = args
        .next()
        .map(|s| s.parse().expect("num_threads must be an integer"))
        .unwrap_or(1);
    let queries_path = args
        .next()
        .unwrap_or_else(|| format!("{DATA_DIR}/queries_minmax8.bin"));
    let gt_path = args
        .next()
        .unwrap_or_else(|| format!("{DATA_DIR}/groundtruth_recall_1000_query_1000.bin"));

    let nlists: Vec<usize> = nlist_csv
        .split(',')
        .map(|s| s.trim().parse().expect("nlist values must be integers"))
        .collect();

    let cfg = Config {
        out_prefix,
        nlists,
        num_threads,
        queries_path,
        gt_path,
        fmt: "",
    };

    // Pick the stored element type from the index metadata: MinMax8 codes are
    // 1 byte, `Half` (f16) 2 bytes, `f32` 4 bytes.
    match index_element_size(&cfg.out_prefix)? {
        1 => run::<MinMaxElement<8>>(Config {
            fmt: "minmax8",
            ..cfg
        }),
        2 => run::<Half>(Config { fmt: "f16", ..cfg }),
        4 => run::<f32>(Config { fmt: "f32", ..cfg }),
        other => Err(format!("unsupported stored element size {other} bytes").into()),
    }
}

/// Sweep an index whose inverted lists store element type `T`.
fn run<T: VectorRepr + Pod>(cfg: Config) -> Result<(), Box<dyn std::error::Error>> {
    let Config {
        out_prefix,
        nlists,
        num_threads,
        queries_path,
        gt_path,
        fmt,
    } = cfg;

    // --- Load index, queries, and groundtruth --------------------------------
    let index = GraphIvfIndex::<T>::load(Path::new(&out_prefix), num_threads)?;
    let dim = index.dim();
    let num_clusters = index.num_clusters();

    let queries: Matrix<T> = read_bin(&mut File::open(&queries_path)?)?;
    let num_queries = queries.nrows();
    if queries.ncols() != dim {
        return Err(format!("query width {} != index dim {dim}", queries.ncols()).into());
    }

    let gt: Matrix<u32> = read_bin(&mut File::open(&gt_path)?)?;
    let gt_dim = gt.ncols();

    println!("index:   {out_prefix}  ({num_clusters} centroids, dim {dim}, {fmt})");
    println!("queries: {num_queries} x {dim}  {queries_path}");
    println!("gt:      {} x {gt_dim}  {gt_path}", gt.nrows());
    println!("threads: {num_threads}   topk: {TOPK}   centroid_l: {CENTROID_SEARCH_L}\n");

    let pool = create_thread_pool(num_threads)?;

    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12} {:>14} {:>10} {:>8} {:>10} {:>10} \
         {:>10} {:>14} {:>10} {:>12} {:>10} {:>8}",
        "nlist",
        "recall@50",
        "recall@1000",
        "mean_us",
        "p95_us",
        "p99_us",
        "bytes/q",
        "ios/q",
        "reqbytes",
        "qps",
        "preproc_us",
        "centroid_us",
        "planio_us",
        "diskread_us",
        "score_us",
        "topk_us"
    );
    for &nlist in nlists.iter() {
        let params = SearchParams {
            nlist,
            centroid_search_l: CENTROID_SEARCH_L,
        };
        let mut result_ids: Vec<u32> = vec![0; TOPK * num_queries];
        let mut latencies_us: Vec<u64> = vec![0; num_queries];
        let bytes_read = AtomicU64::new(0);
        let io_count = AtomicU64::new(0);
        // Per-stage time accumulators (nanoseconds summed across all queries).
        let preprocess_ns = AtomicU64::new(0);
        let centroid_ns = AtomicU64::new(0);
        let plan_io_ns = AtomicU64::new(0);
        let disk_read_ns = AtomicU64::new(0);
        let score_ns = AtomicU64::new(0);
        let topk_ns = AtomicU64::new(0);

        let zipped = queries
            .as_slice()
            .par_chunks(dim)
            .zip(result_ids.par_chunks_mut(TOPK))
            .zip(latencies_us.par_iter_mut());

        let start = Instant::now();
        zipped.for_each_init_in_pool(
            pool.as_ref(),
            || index.searcher().expect("searcher creation failed"),
            |searcher, ((query, id_chunk), latency)| {
                let q_start = Instant::now();
                let (results, profile) = searcher
                    .search_profiled(query, TOPK, &params)
                    .expect("search failed");
                for (slot, (id, _dist)) in id_chunk.iter_mut().zip(results.iter()) {
                    *slot = *id;
                }
                bytes_read.fetch_add(profile.bytes_read, Ordering::Relaxed);
                io_count.fetch_add(profile.io_count, Ordering::Relaxed);
                preprocess_ns.fetch_add(profile.preprocess.as_nanos() as u64, Ordering::Relaxed);
                centroid_ns.fetch_add(profile.centroid_search.as_nanos() as u64, Ordering::Relaxed);
                plan_io_ns.fetch_add(profile.plan_io.as_nanos() as u64, Ordering::Relaxed);
                disk_read_ns.fetch_add(profile.disk_read.as_nanos() as u64, Ordering::Relaxed);
                score_ns.fetch_add(profile.score.as_nanos() as u64, Ordering::Relaxed);
                topk_ns.fetch_add(profile.topk.as_nanos() as u64, Ordering::Relaxed);
                *latency = q_start.elapsed().as_micros() as u64;
            },
        );
        let elapsed = start.elapsed();

        let r50 = recall_at_k(&result_ids, TOPK, &gt, gt_dim, num_queries, RECALL_KS[0]);
        let r1000 = recall_at_k(&result_ids, TOPK, &gt, gt_dim, num_queries, RECALL_KS[1]);
        latencies_us.sort_unstable();
        let mean_us = latencies_us.iter().sum::<u64>() as f64 / num_queries as f64;
        let p95_us = percentile(&latencies_us, 0.95);
        let p99_us = percentile(&latencies_us, 0.99);
        let bytes_per_q = bytes_read.load(Ordering::Relaxed) / num_queries as u64;
        let ios_per_q = io_count.load(Ordering::Relaxed) as f64 / num_queries as f64;
        let req_bytes = if ios_per_q > 0.0 {
            bytes_per_q as f64 / ios_per_q
        } else {
            0.0
        };
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        // Mean per-query stage latency in microseconds.
        let nq = num_queries as f64;
        let mean_us_of = |acc: &AtomicU64| acc.load(Ordering::Relaxed) as f64 / 1000.0 / nq;
        let preproc_us = mean_us_of(&preprocess_ns);
        let centroid_us = mean_us_of(&centroid_ns);
        let planio_us = mean_us_of(&plan_io_ns);
        let diskread_us = mean_us_of(&disk_read_ns);
        let score_us = mean_us_of(&score_ns);
        let topk_us = mean_us_of(&topk_ns);

        println!(
            "{nlist:>6} {:>11.2}% {:>11.2}% {mean_us:>12.0} {p95_us:>12} {p99_us:>14} \
             {bytes_per_q:>10} {ios_per_q:>8.1} {req_bytes:>10.0} {qps:>10.1} {preproc_us:>10.1} {centroid_us:>14.1} \
             {planio_us:>10.1} {diskread_us:>12.1} {score_us:>10.1} {topk_us:>8.1}",
            r50 * 100.0,
            r1000 * 100.0
        );
    }
    Ok(())
}

/// Mean recall@`k`: fraction of each query's top-`k` predicted ids (taken from a
/// `topk`-wide result buffer) that appear in the groundtruth's first `k` ids.
fn recall_at_k(
    result_ids: &[u32],
    topk: usize,
    gt: &Matrix<u32>,
    gt_dim: usize,
    num_queries: usize,
    k: usize,
) -> f64 {
    let mut hit = 0usize;
    for q in 0..num_queries {
        let pred = &result_ids[q * topk..q * topk + k];
        let truth = &gt.as_slice()[q * gt_dim..q * gt_dim + k];
        for id in pred {
            if truth.contains(id) {
                hit += 1;
            }
        }
    }
    hit as f64 / (num_queries * k) as f64
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}
