/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF (Inverted File) search phase.
//!
//! Loads only the centroids into RAM. Each query reads the `nprobe` closest cluster files
//! from disk, producing real I/O metrics comparable to `DiskSearchResult`.
//!
//! Supports two modes:
//! - **Full precision**: cluster files contain f32 vectors, distances computed directly.
//! - **Quantized**: cluster files contain MinMax-compressed codes, distances computed using
//!   asymmetric `FullQuery × DataRef` MinMax distance functions. Optional reranking loads
//!   full-precision vectors from `vectors.bin` to recompute exact distances for top candidates.

use std::{
    fmt,
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};

use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_providers::utils::{create_thread_pool, ParallelIteratorInPool};
use diskann_quantization::{
    algorithms::transforms::{NullTransform, Transform},
    bits::{Representation, Unsigned},
    minmax::{
        self, Data, DataRef, MinMaxCosine, MinMaxCosineNormalized, MinMaxIP, MinMaxL2Squared,
        MinMaxQuantizer,
    },
    num::Positive,
    CompressInto,
};
use diskann_tools::utils::{search_index_utils, KRecallAtN};
use diskann_utils::{views::Matrix, Reborrow, ReborrowMut};
use diskann_vector::PureDistanceFunction;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::ivf::{IvfLoad, IvfSearchPhase},
    utils::{datafiles, SimilarityMeasure},
};

use super::build::u32_to_metric;

// ──────────────────────────────────
// Full-precision distance dispatch
// ──────────────────────────────────

/// Returns a SIMD-accelerated distance function for the given metric.
/// All returned functions follow the convention: lower = more similar.
fn distance_fn(metric: SimilarityMeasure) -> fn(&[f32], &[f32]) -> f32 {
    use diskann_vector::distance;

    match metric {
        SimilarityMeasure::SquaredL2 => |a, b| distance::SquaredL2::evaluate(a, b),
        SimilarityMeasure::InnerProduct
        | SimilarityMeasure::Cosine
        | SimilarityMeasure::CosineNormalized => |a, b| distance::InnerProduct::evaluate(a, b),
    }
}

// ──────────────────────────────────
// Index loading
// ──────────────────────────────────

/// Quantization metadata read from the on-disk meta file.
#[derive(Debug, Clone, Copy)]
struct QuantMeta {
    nbits: u8,
    grid_scale: f32,
}

struct IvfIndex {
    ndims: usize,
    nlist: usize,
    _npoints: usize,
    metric: SimilarityMeasure,
    /// Row-major centroids kept in RAM: nlist × ndims
    centroids: Vec<f32>,
    /// Directory containing per-cluster files
    clusters_dir: PathBuf,
    /// Bytes per record (includes u32 id prefix): depends on quantization
    record_bytes: usize,
    /// If the index was built with quantization
    quant: Option<QuantMeta>,
    /// Path to the index directory
    index_dir: PathBuf,
}

impl IvfIndex {
    fn load(dir: &str) -> anyhow::Result<Self> {
        let dir = Path::new(dir);

        // 1) Meta — extended format: [ndims:u32][nlist:u32][npoints:u32][metric:u32][quantized:u8]...
        let mut f = BufReader::new(File::open(dir.join("ivf_meta.bin"))?);
        let mut buf = [0u8; 16];
        f.read_exact(&mut buf)?;
        let ndims = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
        let nlist = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
        let npoints = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let metric_u32 = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let metric = u32_to_metric(metric_u32)?;

        // Try to read quantization flag (may not exist for old indices)
        let quant = {
            let mut flag_buf = [0u8; 1];
            match f.read_exact(&mut flag_buf) {
                Ok(()) => match flag_buf[0] {
                    0 => None, // explicitly unquantized
                    1 => {
                        let mut qbuf = [0u8; 5]; // 1 byte nbits + 4 bytes grid_scale
                        f.read_exact(&mut qbuf)?;
                        let nbits = qbuf[0];
                        let grid_scale = f32::from_le_bytes(qbuf[1..5].try_into().unwrap());
                        Some(QuantMeta { nbits, grid_scale })
                    }
                    other => anyhow::bail!(
                        "invalid quantization flag byte {} in ivf_meta.bin (expected 0 or 1)",
                        other
                    ),
                },
                Err(_) => None, // EOF — backward-compatible old format
            }
        };

        // 2) Centroids — the only data kept in RAM
        let centroids = {
            let mut f = BufReader::new(File::open(dir.join("ivf_centroids.bin"))?);
            let mut buf = vec![0u8; nlist * ndims * 4];
            f.read_exact(&mut buf)?;
            bytemuck::cast_slice::<u8, f32>(&buf).to_vec()
        };

        // Record bytes depend on quantization
        let record_bytes = match &quant {
            None => 4 + ndims * 4, // u32 id + ndims × f32
            Some(qm) => {
                let quant_bytes = match qm.nbits {
                    1 => Data::<1>::canonical_bytes(ndims),
                    4 => Data::<4>::canonical_bytes(ndims),
                    8 => Data::<8>::canonical_bytes(ndims),
                    other => anyhow::bail!("unsupported nbits: {}", other),
                };
                4 + quant_bytes // u32 id + quantized record
            }
        };

        Ok(IvfIndex {
            ndims,
            nlist,
            _npoints: npoints,
            metric,
            centroids,
            clusters_dir: dir.join("clusters"),
            record_bytes,
            quant,
            index_dir: dir.to_path_buf(),
        })
    }

    /// Read a single full-precision cluster file from disk.
    /// Returns (vector IDs, flat f32 vector data).
    fn read_cluster_f32(&self, cluster_idx: usize) -> anyhow::Result<(Vec<u32>, Vec<f32>)> {
        let path = self
            .clusters_dir
            .join(format!("cluster_{:04}.bin", cluster_idx));
        let mut f = BufReader::new(File::open(&path)?);

        let mut count_buf = [0u8; 4];
        f.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

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

    /// Read a single quantized cluster file from disk.
    /// Returns (vector IDs, raw quantized bytes for all records).
    fn read_cluster_quantized(
        &self,
        cluster_idx: usize,
    ) -> anyhow::Result<(Vec<u32>, Vec<u8>)> {
        let path = self
            .clusters_dir
            .join(format!("cluster_{:04}.bin", cluster_idx));
        let mut f = BufReader::new(File::open(&path)?);

        let mut count_buf = [0u8; 4];
        f.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;

        let mut record_buf = vec![0u8; count * self.record_bytes];
        f.read_exact(&mut record_buf)?;

        let quant_record_bytes = self.record_bytes - 4; // without ID prefix
        let mut ids = Vec::with_capacity(count);
        let mut quant_data = Vec::with_capacity(count * quant_record_bytes);

        for i in 0..count {
            let offset = i * self.record_bytes;
            let id = u32::from_le_bytes(record_buf[offset..offset + 4].try_into().unwrap());
            ids.push(id);
            quant_data.extend_from_slice(&record_buf[offset + 4..offset + self.record_bytes]);
        }

        Ok((ids, quant_data))
    }

    /// Read multiple full-precision vectors from `vectors.bin` by vector IDs.
    /// Opens the file once and seeks for each vector, avoiding repeated open/close overhead.
    fn read_vectors_from_blob(&self, vids: &[u32]) -> anyhow::Result<Vec<Vec<f32>>> {
        let path = self.index_dir.join("vectors.bin");
        let mut f = File::open(&path)?;
        let vec_bytes = self.ndims * 4;
        let mut buf = vec![0u8; vec_bytes];
        let mut results = Vec::with_capacity(vids.len());

        for &vid in vids {
            let byte_offset = vid as u64 * self.ndims as u64 * 4;
            f.seek(SeekFrom::Start(byte_offset))?;
            f.read_exact(&mut buf)?;
            results.push(bytemuck::cast_slice::<u8, f32>(&buf).to_vec());
        }

        Ok(results)
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
    /// Per-list read counts: `list_reads[i]` = number of queries that read cluster `i`.
    pub(super) list_reads: Vec<u32>,
}

// ──────────────────────────────────
// Full-precision search
// ──────────────────────────────────

/// Search a single query with full-precision vectors on disk.
fn search_one_f32(
    index: &IvfIndex,
    query: &[f32],
    nprobe: usize,
    k: usize,
) -> (Vec<u32>, QueryStats, Vec<usize>) {
    let start = Instant::now();
    let ndims = index.ndims;
    let dist_fn = distance_fn(index.metric);

    let (probe_indices, _) = rank_centroids(index, query);
    let probe_count = nprobe.min(index.nlist);

    let mut queue = NeighborPriorityQueue::new(k);
    let mut total_comparisons: u64 = 0;
    let mut io_count: u64 = 0;
    let mut io_time = std::time::Duration::ZERO;
    let mut probed_clusters: Vec<usize> = Vec::with_capacity(probe_count);

    for &(c_idx, _) in probe_indices.iter().take(probe_count) {
        probed_clusters.push(c_idx);
        let io_start = Instant::now();
        let (ids, vecs) = match index.read_cluster_f32(c_idx) {
            Ok(data) => data,
            Err(_) => continue,
        };
        io_time += io_start.elapsed();
        io_count += 1;

        for (local_i, &vid) in ids.iter().enumerate() {
            let vec_start = local_i * ndims;
            let vec_slice = &vecs[vec_start..vec_start + ndims];
            let dist = dist_fn(query, vec_slice);
            total_comparisons += 1;
            queue.insert(Neighbor::new(vid, dist));
        }
    }

    let result_ids: Vec<u32> = queue.iter().take(k).map(|n| n.id).collect();
    let elapsed = start.elapsed();
    let cpu_time = elapsed.saturating_sub(io_time);

    let stats = QueryStats {
        latency_us: elapsed.as_secs_f64() * 1e6,
        io_count: io_count as f64,
        io_time_us: io_time.as_secs_f64() * 1e6,
        cpu_time_us: cpu_time.as_secs_f64() * 1e6,
        comparisons: total_comparisons,
    };

    (result_ids, stats, probed_clusters)
}

// ──────────────────────────────────
// Quantized search
// ──────────────────────────────────

/// Search a single query using quantized vectors on disk, with optional reranking.
fn search_one_quantized<const NBITS: usize>(
    index: &IvfIndex,
    quantizer: &MinMaxQuantizer,
    query: &[f32],
    nprobe: usize,
    k: usize,
    rerank_search_l: Option<usize>,
) -> (Vec<u32>, QueryStats, Vec<usize>)
where
    Unsigned: Representation<NBITS>,
    MinMaxL2Squared: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxCosine: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxCosineNormalized: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
{
    let start = Instant::now();
    let ndims = index.ndims;
    let quant_record_bytes = Data::<NBITS>::canonical_bytes(ndims);

    // The number of candidates to keep from quantized search
    let candidate_k = rerank_search_l.unwrap_or(k);

    // Compress the query into a FullQuery (asymmetric distance: full-precision query × quantized data)
    let mut full_query = minmax::FullQuery::new_in(
        quantizer.output_dim(),
        diskann_quantization::alloc::GlobalAllocator,
    )
    .unwrap();
    quantizer
        .compress_into(query, full_query.reborrow_mut())
        .unwrap();

    let (probe_indices, _) = rank_centroids(index, query);
    let probe_count = nprobe.min(index.nlist);

    let mut queue = NeighborPriorityQueue::new(candidate_k);
    let mut total_comparisons: u64 = 0;
    let mut io_count: u64 = 0;
    let mut io_time = std::time::Duration::ZERO;
    let mut probed_clusters: Vec<usize> = Vec::with_capacity(probe_count);

    for &(c_idx, _) in probe_indices.iter().take(probe_count) {
        probed_clusters.push(c_idx);
        let io_start = Instant::now();
        let (ids, quant_data) = match index.read_cluster_quantized(c_idx) {
            Ok(data) => data,
            Err(_) => continue,
        };
        io_time += io_start.elapsed();
        io_count += 1;

        for (local_i, &vid) in ids.iter().enumerate() {
            let code_start = local_i * quant_record_bytes;
            let code_slice = &quant_data[code_start..code_start + quant_record_bytes];

            // Parse quantized vector from canonical bytes
            let data_ref =
                DataRef::<NBITS>::from_canonical_front(code_slice, ndims).unwrap();

            // Compute asymmetric distance: FullQuery × DataRef
            let dist = quantized_distance(index.metric, full_query.reborrow(), data_ref);
            total_comparisons += 1;
            queue.insert(Neighbor::new(vid, dist));
        }
    }

    // Reranking: if enabled, load full-precision vectors and recompute distances
    let result_ids = if rerank_search_l.is_some() {
        let dist_fn = distance_fn(index.metric);
        let candidates: Vec<Neighbor<u32>> = queue.iter().take(candidate_k).collect();
        let candidate_ids: Vec<u32> = candidates.iter().map(|c| c.id).collect();

        // Open vectors.bin once, read all candidate vectors
        let io_start = Instant::now();
        let vecs = index.read_vectors_from_blob(&candidate_ids).unwrap();
        io_time += io_start.elapsed();
        io_count += candidate_ids.len() as u64;

        let mut reranked = NeighborPriorityQueue::new(k);
        for (candidate, vec) in candidates.iter().zip(vecs.iter()) {
            let dist = dist_fn(query, vec);
            reranked.insert(Neighbor::new(candidate.id, dist));
        }

        reranked.iter().take(k).map(|n| n.id).collect()
    } else {
        queue.iter().take(k).map(|n| n.id).collect()
    };

    let elapsed = start.elapsed();
    let cpu_time = elapsed.saturating_sub(io_time);

    let stats = QueryStats {
        latency_us: elapsed.as_secs_f64() * 1e6,
        io_count: io_count as f64,
        io_time_us: io_time.as_secs_f64() * 1e6,
        cpu_time_us: cpu_time.as_secs_f64() * 1e6,
        comparisons: total_comparisons,
    };

    (result_ids, stats, probed_clusters)
}

/// Compute asymmetric quantized distance using the appropriate MinMax functor.
fn quantized_distance<const NBITS: usize>(
    metric: SimilarityMeasure,
    query: minmax::FullQueryRef<'_>,
    data: DataRef<'_, NBITS>,
) -> f32
where
    Unsigned: Representation<NBITS>,
    MinMaxL2Squared: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxIP: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxCosine: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
    MinMaxCosineNormalized: for<'a, 'b> PureDistanceFunction<
            minmax::FullQueryRef<'a>,
            DataRef<'b, NBITS>,
            diskann_quantization::distances::Result<f32>,
        >,
{
    match metric {
        SimilarityMeasure::SquaredL2 => MinMaxL2Squared::evaluate(query, data).unwrap(),
        SimilarityMeasure::InnerProduct => MinMaxIP::evaluate(query, data).unwrap(),
        SimilarityMeasure::Cosine => MinMaxCosine::evaluate(query, data).unwrap(),
        SimilarityMeasure::CosineNormalized => {
            MinMaxCosineNormalized::evaluate(query, data).unwrap()
        }
    }
}

// ──────────────────────────────────
// Shared helpers
// ──────────────────────────────────

/// Rank centroids by distance to the query. Returns sorted (index, distance) pairs.
fn rank_centroids(
    index: &IvfIndex,
    query: &[f32],
) -> (Vec<(usize, f32)>, std::time::Duration) {
    let start = Instant::now();
    let ndims = index.ndims;
    let dist_fn = distance_fn(index.metric);

    let mut centroid_dists: Vec<(usize, f32)> = (0..index.nlist)
        .map(|c| {
            let centroid = &index.centroids[c * ndims..(c + 1) * ndims];
            (c, dist_fn(query, centroid))
        })
        .collect();
    centroid_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    (centroid_dists, start.elapsed())
}

// ──────────────────────────────────
// Top-level search entry point
// ──────────────────────────────────

pub(super) fn search_ivf_index(
    index_load: &IvfLoad,
    search_params: &IvfSearchPhase,
) -> anyhow::Result<IvfSearchStats> {
    let index = IvfIndex::load(&index_load.load_path)?;

    // Validate: reranking requires a quantized index
    if search_params.rerank.is_some() && index.quant.is_none() {
        anyhow::bail!(
            "reranking is configured but the index was not built with quantization; \
             reranking requires quantized vectors and a vectors.bin blob"
        );
    }

    let queries: Matrix<f32> = datafiles::load_dataset(datafiles::BinFile(&search_params.queries))?;
    let num_queries = queries.nrows();
    let recall_at = search_params.recall_at.get() as usize;

    let gt = datafiles::load_groundtruth(
        datafiles::BinFile(&search_params.groundtruth),
        Some(recall_at),
    )?;

    let pool = create_thread_pool(search_params.num_threads.get())?;

    let rerank_search_l = search_params
        .rerank
        .map(|r| r.search_l.get() as usize);

    // Build quantizer if needed (shared across threads)
    let quantizer = index.quant.map(|qm| {
        MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(index.ndims).unwrap())),
            Positive::new(qm.grid_scale).unwrap(),
        )
    });

    let mut search_results_per_nprobe = Vec::with_capacity(search_params.nprobe_list.len());

    for &nprobe in &search_params.nprobe_list {
        let start = Instant::now();

        let results: Vec<(Vec<u32>, QueryStats, Vec<usize>)> = (0..num_queries)
            .into_par_iter()
            .map(|qi| {
                let query = queries.row(qi);
                match (&index.quant, &quantizer) {
                    (Some(qm), Some(q)) => match qm.nbits {
                        1 => search_one_quantized::<1>(
                            &index, q, query, nprobe.get() as usize, recall_at, rerank_search_l,
                        ),
                        4 => search_one_quantized::<4>(
                            &index, q, query, nprobe.get() as usize, recall_at, rerank_search_l,
                        ),
                        8 => search_one_quantized::<8>(
                            &index, q, query, nprobe.get() as usize, recall_at, rerank_search_l,
                        ),
                        _ => unreachable!("nbits validated during load"),
                    },
                    _ => search_one_f32(&index, query, nprobe.get() as usize, recall_at),
                }
            })
            .collect_in_pool(pool.as_ref());

        let total_time = start.elapsed();

        // Aggregate per-list read counts
        let mut list_reads = vec![0u32; index.nlist];
        for (_, _, probed) in &results {
            for &c_idx in probed {
                list_reads[c_idx] += 1;
            }
        }

        // Gather result IDs into flat array for recall computation
        let mut result_ids = vec![0u32; num_queries * recall_at];
        for (qi, (ids, _, _)) in results.iter().enumerate() {
            let offset = qi * recall_at;
            let count = ids.len().min(recall_at);
            result_ids[offset..offset + count].copy_from_slice(&ids[..count]);
        }

        // Compute recall
        let recall_at_u32 = search_params.recall_at.get();
        let recall = search_index_utils::calculate_recall(
            num_queries,
            gt.as_slice(),
            None,
            gt.ncols(),
            &result_ids,
            recall_at_u32,
            KRecallAtN::new(recall_at_u32, recall_at_u32)?,
        )? as f32;

        // Compute statistics
        let stats_vec: Vec<&QueryStats> = results.iter().map(|(_, s, _)| s).collect();
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
            search_l: nprobe.get(),
            qps,
            mean_latency,
            p95_latency: MicroSeconds::new(p95 as u64),
            p999_latency: MicroSeconds::new(p999 as u64),
            mean_ios,
            mean_io_time,
            mean_cpu_time,
            mean_comparisons,
            recall,
            list_reads,
        });
    }

    Ok(IvfSearchStats {
        num_threads: search_params.num_threads.get(),
        recall_at: search_params.recall_at.get(),
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
