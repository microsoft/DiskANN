/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Single-query latency-attribution harness ("performance layer cake") for a
//! MinMax8 graph-IVF index.
//!
//! Unlike `sweep` (which runs the full query set across many threads and
//! reports only aggregate means), this harness runs a *small* number of queries
//! at a *chosen* thread count and prints the full per-query
//! [`SearchProfile`](diskann_graphivf::SearchProfile) layer cake for the first
//! few queries, plus aggregate per-stage means and the effective disk
//! throughput. This isolates whether per-query disk-read latency is dominated by
//! single-query cost or by multi-thread contention on one physical device.
//!
//! The lists file is opened with `FILE_FLAG_NO_BUFFERING` (direct I/O), so the
//! OS page cache is bypassed: every read hits the device and there is no
//! cold/warm distinction.
//!
//! Run:
//! ```text
//! cargo run --release --example profile_layercake -- \
//!     <index_prefix> <nprobe> <num_queries> <threads> <queries_minmax8.bin>
//! ```

use std::{
    fs::File,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use diskann_graphivf::{GraphIvfIndex, SearchParams};
use diskann_providers::{
    common::MinMaxElement,
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_utils::{io::read_bin, views::Matrix};
use rayon::prelude::*;

type Elem = MinMaxElement<8>;

const CENTROID_SEARCH_L: usize = 1024;
const TOPK: usize = 1000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let prefix = args
        .next()
        .expect("usage: <index_prefix> <nprobe> <num_queries> <threads> <queries.bin>");
    let nprobe: usize = args.next().expect("nprobe").parse()?;
    let num_queries: usize = args.next().expect("num_queries").parse()?;
    let threads: usize = args.next().expect("threads").parse()?;
    let queries_path = args.next().expect("queries.bin");

    let index = GraphIvfIndex::<Elem>::load(Path::new(&prefix), threads)?;
    let dim = index.dim();
    let num_clusters = index.num_clusters();

    let queries: Matrix<Elem> = read_bin(&mut File::open(&queries_path)?)?;
    let total_q = queries.nrows();
    if queries.ncols() != dim {
        return Err(format!("query width {} != index dim {dim}", queries.ncols()).into());
    }
    let n = num_queries.min(total_q);

    println!("index:   {prefix}  ({num_clusters} centroids, dim {dim})");
    println!("queries: running {n} of {total_q}  ({queries_path})");
    println!(
        "nprobe:  {nprobe}   threads: {threads}   topk: {TOPK}   centroid_l: {CENTROID_SEARCH_L}"
    );
    println!("note:    lists file uses FILE_FLAG_NO_BUFFERING (direct I/O, page cache bypassed)\n");

    let params = SearchParams {
        nlist: nprobe,
        centroid_search_l: CENTROID_SEARCH_L,
    };
    let pool = create_thread_pool(threads)?;

    let preprocess_ns = AtomicU64::new(0);
    let centroid_ns = AtomicU64::new(0);
    let plan_io_ns = AtomicU64::new(0);
    let disk_read_ns = AtomicU64::new(0);
    let score_ns = AtomicU64::new(0);
    let topk_ns = AtomicU64::new(0);
    let bytes_read = AtomicU64::new(0);
    let io_count = AtomicU64::new(0);

    // Collect per-query wall latency + disk-read + bytes so we can print the
    // first few as individual layer cakes.
    let mut per_query: Vec<(u64, u64, u64, u64)> = vec![(0, 0, 0, 0); n]; // (wall_us, diskread_us, bytes, ios)

    let query_slice = &queries.as_slice()[..n * dim];
    let start = Instant::now();
    query_slice
        .par_chunks(dim)
        .zip(per_query.par_iter_mut())
        .for_each_init_in_pool(
            pool.as_ref(),
            || index.searcher().expect("searcher creation failed"),
            |searcher, (query, slot)| {
                let q_start = Instant::now();
                let (_results, profile) = searcher
                    .search_profiled(query, TOPK, &params)
                    .expect("search failed");
                preprocess_ns.fetch_add(profile.preprocess.as_nanos() as u64, Ordering::Relaxed);
                centroid_ns.fetch_add(profile.centroid_search.as_nanos() as u64, Ordering::Relaxed);
                plan_io_ns.fetch_add(profile.plan_io.as_nanos() as u64, Ordering::Relaxed);
                disk_read_ns.fetch_add(profile.disk_read.as_nanos() as u64, Ordering::Relaxed);
                score_ns.fetch_add(profile.score.as_nanos() as u64, Ordering::Relaxed);
                topk_ns.fetch_add(profile.topk.as_nanos() as u64, Ordering::Relaxed);
                bytes_read.fetch_add(profile.bytes_read, Ordering::Relaxed);
                io_count.fetch_add(profile.io_count, Ordering::Relaxed);
                *slot = (
                    q_start.elapsed().as_micros() as u64,
                    profile.disk_read.as_micros() as u64,
                    profile.bytes_read,
                    profile.io_count,
                );
            },
        );
    let elapsed = start.elapsed();

    // --- Per-query detail for the first few queries --------------------------
    let show = n.min(8);
    println!("per-query (first {show}):");
    println!(
        "  {:>4} {:>10} {:>12} {:>10} {:>6} {:>10}",
        "q", "wall_us", "diskread_us", "bytes", "ios", "MB/s"
    );
    for (i, &(wall, dr, bytes, ios)) in per_query.iter().take(show).enumerate() {
        let mbps = if dr > 0 {
            bytes as f64 / (dr as f64 / 1e6) / 1e6
        } else {
            0.0
        };
        println!("  {i:>4} {wall:>10} {dr:>12} {bytes:>10} {ios:>6} {mbps:>10.1}");
    }

    // --- Aggregate means -----------------------------------------------------
    let nq = n as f64;
    let mean_us = |acc: &AtomicU64| acc.load(Ordering::Relaxed) as f64 / 1000.0 / nq;
    let total_bytes = bytes_read.load(Ordering::Relaxed);
    let total_ios = io_count.load(Ordering::Relaxed);
    println!("\naggregate ({n} queries, {threads} threads):");
    println!(
        "  wall elapsed:        {:>10.3} ms",
        elapsed.as_secs_f64() * 1e3
    );
    println!(
        "  qps:                 {:>10.1}",
        nq / elapsed.as_secs_f64()
    );
    println!("  mean preprocess_us:  {:>10.1}", mean_us(&preprocess_ns));
    println!("  mean centroid_us:    {:>10.1}", mean_us(&centroid_ns));
    println!("  mean plan_io_us:     {:>10.1}", mean_us(&plan_io_ns));
    println!("  mean diskread_us:    {:>10.1}", mean_us(&disk_read_ns));
    println!("  mean score_us:       {:>10.1}", mean_us(&score_ns));
    println!("  mean topk_us:        {:>10.1}", mean_us(&topk_ns));
    println!("  mean bytes/q:        {:>10}", total_bytes / n as u64);
    println!("  mean ios/q:          {:>10.1}", total_ios as f64 / nq);
    println!(
        "  reqbytes (bytes/io): {:>10.0}",
        if total_ios > 0 {
            total_bytes as f64 / total_ios as f64
        } else {
            0.0
        }
    );
    // Effective device throughput = total bytes moved / wall time. This is the
    // real disk bandwidth; per-query diskread_us divided into it reveals how
    // many queries were contending.
    let agg_mbps = total_bytes as f64 / elapsed.as_secs_f64() / 1e6;
    println!(
        "  AGG device MB/s:     {:>10.1}  (total bytes / wall)",
        agg_mbps
    );
    Ok(())
}
