/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF (Inverted File) search phase.
//!
//! Loads only the centroids into RAM. Each query reads the `nprobe` closest cluster files
//! from disk, producing real I/O metrics comparable to `DiskSearchResult`.

use std::{
    fmt,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_tools::utils::{search_index_utils, KRecallAtN};
use diskann_utils::views::Matrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::ivf::{IvfLoad, IvfSearchPhase},
    utils::{datafiles, SimilarityMeasure},
};

use super::build::u32_to_metric;

// ─────────────────────────────────────────
// Lightweight handle — only centroids in RAM
// ─────────────────────────────────────────

struct IvfIndex {
    ndims: usize,
    nlist: usize,
    _metric: SimilarityMeasure,
    /// Row-major centroids kept in RAM: nlist × ndims
    centroids: Vec<f32>,
    /// Directory containing per-cluster files
    clusters_dir: PathBuf,
    /// Bytes per record: sizeof(u32) + ndims * sizeof(f32)
    record_bytes: usize,
}

impl IvfIndex {
    fn load(dir: &str) -> anyhow::Result<Self> {
        let dir = Path::new(dir);

        // 1) Meta
        let (ndims, nlist, _npoints, metric) = {
            let mut f = BufReader::new(File::open(dir.join("ivf_meta.bin"))?);
            let mut buf = [0u8; 16];
            f.read_exact(&mut buf)?;
            let ndims = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
            let nlist = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
            let npoints = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
            let metric_u32 = u32::from_le_bytes(buf[12..16].try_into().unwrap());
            (ndims, nlist, npoints, u32_to_metric(metric_u32)?)
        };

        // 2) Centroids — the only data kept in RAM
        let centroids = {
            let mut f = BufReader::new(File::open(dir.join("ivf_centroids.bin"))?);
            let mut buf = vec![0u8; nlist * ndims * 4];
            f.read_exact(&mut buf)?;
            bytemuck::cast_slice::<u8, f32>(&buf).to_vec()
        };

        let record_bytes = 4 + ndims * 4; // u32 id + ndims × f32

        Ok(IvfIndex {
            ndims,
            nlist,
            _metric: metric,
            centroids,
            clusters_dir: dir.join("clusters"),
            record_bytes,
        })
    }

    /// Read a single cluster file from disk. Returns (vector IDs, flat vector data).
    fn read_cluster(&self, cluster_idx: usize) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let path = self
            .clusters_dir
            .join(format!("cluster_{:04}.bin", cluster_idx));
        let mut f = BufReader::new(File::open(&path)?);

        // Read count
        let mut count_buf = [0u8; 4];
        f.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        // Read interleaved records: [id: u32][vec: ndims × f32] per record
        let mut record_buf = vec![0u8; count * self.record_bytes];
        f.read_exact(&mut record_buf)?;

        let mut ids = Vec::with_capacity(count);
        let mut vecs = Vec::with_capacity(count * self.ndims);

        for i in 0..count {
            let offset = i * self.record_bytes;
            let id = u32::from_le_bytes(record_buf[offset..offset + 4].try_into().unwrap());
            ids.push(id);

            let vec_bytes = &record_buf[offset + 4..offset + self.record_bytes];
            let vec_slice = bytemuck::cast_slice::<u8, f32>(vec_bytes);
            vecs.extend_from_slice(vec_slice);
        }

        Ok((ids, vecs))
    }
}

// ──────────────────────────────────
// Per-query statistics
// ──────────────────────────────────

struct QueryStats {
    latency_us: f64,
    io_count: f64,
    io_time_us: f64,
    cpu_time_us: f64,
    comparisons: u64,
}

// ──────────────────────────
// Public search result types
// ──────────────────────────

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct IvfSearchStats {
    pub(super) num_threads: usize,
    pub(super) recall_at: u32,
    pub(super) distance: SimilarityMeasure,
    pub(super) search_results_per_nprobe: Vec<IvfSearchResult>,
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct IvfSearchResult {
    pub(super) search_l: u32,
    pub(super) qps: f32,
    pub(super) mean_latency: f64,
    pub(super) p95_latency: MicroSeconds,
    pub(super) p999_latency: MicroSeconds,
    pub(super) mean_ios: f64,
    pub(super) mean_io_time: f64,
    pub(super) mean_cpu_time: f64,
    pub(super) mean_comparisons: f64,
    pub(super) recall: f32,
}

/// Squared L2 distance (lower = more similar).
fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Negative inner product (lower = more similar, matching L2 sort order).
fn neg_ip(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// Returns the distance function for the given metric.
/// All returned functions follow the convention: lower = more similar.
fn distance_fn(metric: SimilarityMeasure) -> fn(&[f32], &[f32]) -> f32 {
    match metric {
        SimilarityMeasure::SquaredL2 => sq_l2,
        SimilarityMeasure::InnerProduct
        | SimilarityMeasure::Cosine
        | SimilarityMeasure::CosineNormalized => neg_ip,
    }
}

/// Search a single query: rank centroids (RAM), then read `nprobe` cluster files from disk.
fn search_one(index: &IvfIndex, query: &[f32], nprobe: usize, k: usize) -> (Vec<u32>, QueryStats) {
    let start = Instant::now();
    let ndims = index.ndims;
    let dist_fn = distance_fn(index._metric);

    // 1) Rank centroids (in RAM)
    let mut centroid_dists: Vec<(usize, f32)> = (0..index.nlist)
        .map(|c| {
            let centroid = &index.centroids[c * ndims..(c + 1) * ndims];
            (c, dist_fn(query, centroid))
        })
        .collect();
    centroid_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let cpu_after_centroid = start.elapsed();

    let probe_count = nprobe.min(index.nlist);

    // 2) Read cluster files from disk and scan
    let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
    let mut total_comparisons: u64 = 0;
    let mut io_count: u64 = 0;

    let io_start = Instant::now();

    for &(c_idx, _) in centroid_dists.iter().take(probe_count) {
        // Real disk I/O: read the cluster file
        let (ids, vecs) = match index.read_cluster(c_idx) {
            Ok(data) => data,
            Err(_) => continue, // skip empty/missing clusters
        };
        io_count += 1;

        for (local_i, &vid) in ids.iter().enumerate() {
            let vec_start = local_i * ndims;
            let vec_slice = &vecs[vec_start..vec_start + ndims];
            let dist = dist_fn(query, vec_slice);
            total_comparisons += 1;

            if best.len() < k {
                best.push((dist, vid));
                if best.len() == k {
                    best.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            } else if dist < best[0].0 {
                best[0] = (dist, vid);
                best.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        }
    }

    let io_elapsed = io_start.elapsed();

    // Sort results by distance ascending for recall computation
    best.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let result_ids: Vec<u32> = best.iter().map(|(_, id)| *id).collect();

    let elapsed = start.elapsed();
    let cpu_time = cpu_after_centroid + (elapsed - io_start.elapsed().max(io_elapsed));

    let stats = QueryStats {
        latency_us: elapsed.as_secs_f64() * 1e6,
        io_count: io_count as f64,
        io_time_us: io_elapsed.as_secs_f64() * 1e6,
        cpu_time_us: cpu_time.as_secs_f64() * 1e6,
        comparisons: total_comparisons,
    };

    (result_ids, stats)
}

pub(super) fn search_ivf_index(
    index_load: &IvfLoad,
    search_params: &IvfSearchPhase,
) -> anyhow::Result<IvfSearchStats> {
    // Load centroids only — cluster data stays on disk
    let index = IvfIndex::load(&index_load.load_path)?;

    // Load queries
    let queries: Matrix<f32> = datafiles::load_dataset(datafiles::BinFile(&search_params.queries))?;
    let num_queries = queries.nrows();
    let recall_at = search_params.recall_at as usize;

    // Load ground truth
    let gt = datafiles::load_groundtruth(
        datafiles::BinFile(&search_params.groundtruth),
        Some(recall_at),
    )?;

    // Build thread pool
    let pool = create_thread_pool(search_params.num_threads)?;

    let mut search_results_per_nprobe = Vec::with_capacity(search_params.nprobe_list.len());

    for &nprobe in &search_params.nprobe_list {
        let start = Instant::now();

        // Run all queries in parallel — each thread does its own disk reads
        let results: Vec<(Vec<u32>, QueryStats)> = (0..num_queries)
            .into_par_iter()
            .map(|qi| {
                let query = queries.row(qi);
                search_one(&index, query, nprobe as usize, recall_at)
            })
            .collect_in_pool(pool.as_ref());

        let total_time = start.elapsed();

        // Gather result IDs into flat array for recall computation
        let mut result_ids = vec![0u32; num_queries * recall_at];
        for (qi, (ids, _)) in results.iter().enumerate() {
            let offset = qi * recall_at;
            let count = ids.len().min(recall_at);
            result_ids[offset..offset + count].copy_from_slice(&ids[..count]);
        }

        // Compute recall
        let recall = search_index_utils::calculate_recall(
            num_queries,
            gt.as_slice(),
            None,
            gt.ncols(),
            &result_ids,
            search_params.recall_at,
            KRecallAtN::new(search_params.recall_at, search_params.recall_at)?,
        )? as f32;

        // Compute statistics
        let stats_vec: Vec<&QueryStats> = results.iter().map(|(_, s)| s).collect();
        let n = num_queries as f64;

        let mean_latency: f64 = stats_vec.iter().map(|s| s.latency_us).sum::<f64>() / n;
        let mean_ios: f64 = stats_vec.iter().map(|s| s.io_count).sum::<f64>() / n;
        let mean_io_time: f64 = stats_vec.iter().map(|s| s.io_time_us).sum::<f64>() / n;
        let mean_cpu_time: f64 = stats_vec.iter().map(|s| s.cpu_time_us).sum::<f64>() / n;
        let mean_comparisons: f64 = stats_vec.iter().map(|s| s.comparisons as f64).sum::<f64>() / n;

        // P95 / P999 latency
        let mut latencies: Vec<f64> = stats_vec.iter().map(|s| s.latency_us).collect();
        latencies.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_idx = ((num_queries as f64) * 0.95).ceil() as usize;
        let p999_idx = ((num_queries as f64) * 0.999).ceil() as usize;
        let p95 = latencies[p95_idx.min(num_queries - 1)];
        let p999 = latencies[p999_idx.min(num_queries - 1)];

        let qps = if total_time.as_secs_f32() > 0.0 {
            num_queries as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };

        search_results_per_nprobe.push(IvfSearchResult {
            search_l: nprobe,
            qps,
            mean_latency,
            p95_latency: MicroSeconds::new(p95 as u64),
            p999_latency: MicroSeconds::new(p999 as u64),
            mean_ios,
            mean_io_time,
            mean_cpu_time,
            mean_comparisons,
            recall,
        });
    }

    Ok(IvfSearchStats {
        num_threads: search_params.num_threads,
        recall_at: search_params.recall_at,
        distance: search_params.distance,
        search_results_per_nprobe,
    })
}

// ─────────
// Display
// ─────────

impl fmt::Display for IvfSearchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_us = |v: f64| -> String { format!("{:.1}us", v) };

        let cols: [(&str, usize); 10] = [
            ("Nprobe", 7),
            ("KNN", 3),
            ("QPS", 8),
            ("Mean Latency", 13),
            ("95% Latency", 13),
            ("99.9 Latency", 13),
            ("IOs", 6),
            ("IO (us)", 10),
            ("Mean Comps", 11),
            ("Recall", 7),
        ];

        let mut header = String::new();
        for (i, (name, w)) in cols.iter().enumerate() {
            if i > 0 {
                header.push(' ');
            }
            header.push_str(&format!("{:>width$}", *name, width = *w));
        }
        let rule = "=".repeat(header.len());

        writeln!(f, "IVF Search Stats")?;
        writeln!(f, "Threads,          : {}", self.num_threads)?;
        writeln!(f, "Recall at,        : {}", self.recall_at)?;
        writeln!(f, "Distance,         : {}", self.distance)?;

        writeln!(f, "{rule}")?;
        writeln!(f, "{header}")?;
        writeln!(f, "{rule}")?;

        for r in &self.search_results_per_nprobe {
            let vals: [String; 10] = [
                format!("{}", r.search_l),
                format!("{}", self.recall_at),
                format!("{:.1}", r.qps),
                fmt_us(r.mean_latency),
                format!("{}", r.p95_latency),
                format!("{}", r.p999_latency),
                format!("{:.1}", r.mean_ios),
                fmt_us(r.mean_io_time),
                format!("{:.1}", r.mean_comparisons),
                format!("{:.3}", r.recall),
            ];

            let mut line = String::new();
            for ((_, w), v) in cols.iter().zip(vals.iter()) {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.push_str(&format!("{:>width$}", v, width = *w));
            }
            writeln!(f, "{line}")?;
        }

        Ok(())
    }
}
