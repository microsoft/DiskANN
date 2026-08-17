/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    sync::{atomic::AtomicU64, Mutex},
    time::Instant,
};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::utils::{fmt::Table, MicroSeconds};
use diskann_graphivf::{GraphIvfIndex, SearchParams};
use diskann_providers::{
    storage::FileStorageProvider,
    utils::{create_thread_pool, ParallelIteratorInPool},
};
use diskann_tools::utils::{search_index_utils, KRecallAtN, TruthSet};
use diskann_utils::views::Matrix;

use crate::{
    backend::graph_ivf::{
        build::{to_centroid_search, to_graphivf_metric},
        element::GraphIvfElement,
    },
    inputs::graph_ivf::{CentroidSearchConfig, GraphIvfLoad, GraphIvfSearchPhase, RecallAt},
    utils::{datafiles, SimilarityMeasure},
};

/// Recall at one `k`, as a percentage.
#[derive(Serialize, Deserialize, Debug)]
pub(super) struct RecallPoint {
    pub(super) at: u32,
    pub(super) recall: f32,
}

/// Score one result buffer at every configured `k`.
///
/// `result_ids` is `k_max` deep and distance-ordered, so its first `k` entries
/// are exactly what a search of depth `k` would have returned. That is what
/// lets a single sweep answer for every `k` instead of one sweep per `k`.
pub(super) fn recall_points(
    recall_at: &RecallAt,
    num_queries: usize,
    gt: &TruthSet,
    result_ids: &[u32],
    k_max: usize,
) -> anyhow::Result<Vec<RecallPoint>> {
    anyhow::ensure!(
        gt.index_num_points >= num_queries,
        "groundtruth has {} queries but search produced results for {num_queries}",
        gt.index_num_points,
    );
    // `calculate_recall` reads `k` ids per groundtruth row; a row shorter than
    // that would silently spill into the next query's row rather than fail.
    anyhow::ensure!(
        gt.index_dimension >= k_max,
        "groundtruth has {} neighbors per query but recall_at asks for {k_max}",
        gt.index_dimension,
    );
    anyhow::ensure!(
        result_ids.len() == num_queries.saturating_mul(k_max),
        "result buffer has {} ids but expected {} queries * {k_max} results",
        result_ids.len(),
        num_queries,
    );
    recall_at
        .iter()
        .map(|k| {
            let recall = search_index_utils::calculate_recall(
                num_queries,
                &gt.index_nodes,
                gt.distances.as_ref(),
                gt.index_dimension,
                result_ids,
                k_max as u32,
                KRecallAtN::new(k, k)?,
            )? as f32;
            Ok(RecallPoint { at: k, recall })
        })
        .collect()
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfSearchStats {
    pub(super) num_threads: usize,
    pub(super) recall_at: Vec<u32>,
    pub(super) distance: SimilarityMeasure,
    /// How clusters were selected. `centroid_search_alpha` has no effect under
    /// [`CentroidSearchConfig::Exact`].
    pub(super) centroid_search: CentroidSearchConfig,
    pub(super) centroid_search_alpha: f32,
    pub(super) search_results_per_nlist: Vec<GraphIvfSearchResult>,
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfSearchResult {
    /// Requested share of the index's clusters.
    pub(super) cluster_fraction: f64,
    /// Concrete number of clusters probed after rounding the fraction up.
    pub(super) nlist: usize,
    pub(super) qps: f32,
    pub(super) mean_latency: MicroSeconds,
    pub(super) p95_latency: MicroSeconds,
    pub(super) p999_latency: MicroSeconds,
    /// One entry per configured `recall_at`, all from the same sweep.
    pub(super) recalls: Vec<RecallPoint>,
    /// Mean per-query, per-stage latency breakdown (a "layer cake").
    pub(super) breakdown: GraphIvfLatencyBreakdown,
}

/// Mean per-query latency of each search stage, in nanoseconds.
#[derive(Serialize, Deserialize, Debug, Default)]
pub(super) struct GraphIvfLatencyBreakdown {
    pub(super) preprocess_ns: u64,
    pub(super) centroid_search_ns: u64,
    pub(super) plan_io_ns: u64,
    pub(super) disk_read_ns: u64,
    pub(super) score_ns: u64,
    pub(super) topk_ns: u64,
    pub(super) total_ns: u64,
    /// Mean disk read requests issued per query.
    pub(super) io_count: u64,
    /// Mean bytes fetched from disk per query.
    pub(super) bytes_read: u64,
}

/// Thread-shared accumulator summing each stage's wall-clock (in nanoseconds)
/// across all queries of one cluster-fraction sweep.
#[derive(Default)]
struct PhaseAccum {
    preprocess: AtomicU64,
    centroid_search: AtomicU64,
    plan_io: AtomicU64,
    disk_read: AtomicU64,
    score: AtomicU64,
    topk: AtomicU64,
    total: AtomicU64,
    io_count: AtomicU64,
    bytes_read: AtomicU64,
}

pub(super) fn record_first_error(first_error: &Mutex<Option<String>>, message: String) {
    let mut error = first_error
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if error.is_none() {
        *error = Some(message);
    }
}

pub(super) fn search_graph_ivf<T>(
    index_load: &GraphIvfLoad,
    search_params: &GraphIvfSearchPhase,
) -> anyhow::Result<GraphIvfSearchStats>
where
    T: GraphIvfElement,
{
    use std::sync::atomic::Ordering;

    // Map the distance measure and decide whether queries must be normalized.
    // Cosine normalizes the corpus at build time, so queries must be normalized
    // to match; L2 / already-normalized cosine leave queries untouched.
    let metric = to_graphivf_metric(search_params.distance)?;
    let normalize_queries = matches!(metric, diskann_graphivf::Metric::Cosine);
    // Normalizing a query means decoding it, scaling, then re-encoding to `T`. That is
    // element-wise only for native types; a quantized row's per-vector metadata would
    // have to be re-derived. Such corpora are expected to be normalized before they are
    // quantized, which is what `cosine_normalized` denotes.
    anyhow::ensure!(
        !(normalize_queries && T::STORED_VERBATIM),
        "{:?} queries cannot be normalized in place; pre-normalize the corpus and queries \
         before quantizing and use `cosine_normalized`",
        T::DATA_TYPE,
    );

    // Load the index from disk.
    let index = GraphIvfIndex::<T>::load(
        std::path::Path::new(&index_load.load_path),
        search_params.num_threads,
        to_centroid_search(index_load.centroid_search),
    )?;
    let dim = index.dim();
    let num_clusters = index.num_clusters();

    // Load the queries (stored as `T`) and optionally normalize them.
    let queries: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(&search_params.queries))?;
    let num_queries = queries.nrows();
    if queries.ncols() != dim {
        anyhow::bail!(
            "query dimension {} does not match index dimension {}",
            queries.ncols(),
            dim
        );
    }
    let prepared: Vec<T> = if normalize_queries {
        prepare_normalized_queries::<T>(&queries)?
    } else {
        queries.as_slice().to_vec()
    };

    // Load the groundtruth.
    let gt = search_index_utils::load_truthset(
        &FileStorageProvider,
        &search_params.groundtruth.to_string_lossy(),
    )?;

    // Confirm a searcher can be created before entering the parallel region; the
    // per-thread initializer below reuses the same fallible call.
    let _ = index.searcher()?;

    let recall_at = &search_params.recall_at;
    let k_max = recall_at.max() as usize;
    let pool = create_thread_pool(search_params.num_threads)?;
    anyhow::ensure!(
        num_clusters > 0,
        "cannot search a graph-IVF index with no clusters"
    );
    let mut search_results_per_nlist = Vec::with_capacity(search_params.cluster_fractions.len());

    for &cluster_fraction in &search_params.cluster_fractions {
        let nlist = cluster_fraction.nlist(num_clusters);
        let params = SearchParams {
            nlist,
            centroid_search_alpha: search_params.centroid_search_alpha,
        };

        let mut result_ids: Vec<u32> = vec![0; k_max * num_queries];
        let mut latencies_us: Vec<u64> = vec![0; num_queries];
        let first_error = Mutex::new(None::<String>);
        let accum = PhaseAccum::default();

        let zipped = prepared
            .par_chunks(dim)
            .zip(result_ids.par_chunks_mut(k_max))
            .zip(latencies_us.par_iter_mut());

        let start = Instant::now();
        zipped.for_each_init_in_pool(
            pool.as_ref(),
            // Each worker thread owns its own searcher (disk reader + runtime).
            || index.searcher().map_err(|error| format!("{error:#}")),
            |searcher, ((query, id_chunk), latency)| {
                let q_start = Instant::now();
                let searcher = match searcher {
                    Ok(searcher) => searcher,
                    Err(error) => {
                        id_chunk.fill(0);
                        record_first_error(
                            &first_error,
                            format!("failed to create a worker searcher: {error}"),
                        );
                        *latency = q_start.elapsed().as_micros() as u64;
                        return;
                    }
                };
                match searcher.search_profiled(query, k_max, &params) {
                    Ok((results, profile)) => {
                        for (slot, (id, _dist)) in id_chunk.iter_mut().zip(results.iter()) {
                            *slot = *id;
                        }
                        accum
                            .preprocess
                            .fetch_add(profile.preprocess.as_nanos() as u64, Ordering::Relaxed);
                        accum.centroid_search.fetch_add(
                            profile.centroid_search.as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                        accum
                            .plan_io
                            .fetch_add(profile.plan_io.as_nanos() as u64, Ordering::Relaxed);
                        accum
                            .disk_read
                            .fetch_add(profile.disk_read.as_nanos() as u64, Ordering::Relaxed);
                        accum
                            .score
                            .fetch_add(profile.score.as_nanos() as u64, Ordering::Relaxed);
                        accum
                            .topk
                            .fetch_add(profile.topk.as_nanos() as u64, Ordering::Relaxed);
                        accum
                            .total
                            .fetch_add(profile.total.as_nanos() as u64, Ordering::Relaxed);
                        accum
                            .io_count
                            .fetch_add(profile.io_count, Ordering::Relaxed);
                        accum
                            .bytes_read
                            .fetch_add(profile.bytes_read, Ordering::Relaxed);
                    }
                    Err(e) => {
                        id_chunk.fill(0);
                        record_first_error(&first_error, format!("{e:#}"));
                    }
                }
                *latency = q_start.elapsed().as_micros() as u64;
            },
        );
        let total_time = start.elapsed();

        if let Some(error) = first_error
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            anyhow::bail!("one or more graph-ivf searches failed: {error}");
        }

        let recalls = recall_points(recall_at, num_queries, &gt, &result_ids, k_max)?;

        latencies_us.sort_unstable();
        let percentile = |p: f64| -> u64 {
            if latencies_us.is_empty() {
                0
            } else {
                let idx = ((latencies_us.len() as f64 * p).ceil() as usize)
                    .saturating_sub(1)
                    .min(latencies_us.len() - 1);
                latencies_us[idx]
            }
        };
        let mean_us = if num_queries > 0 {
            latencies_us.iter().sum::<u64>() / num_queries as u64
        } else {
            0
        };
        let total_secs = total_time.as_secs_f32();

        let mean_ns = |a: &AtomicU64| -> u64 {
            if num_queries > 0 {
                a.load(Ordering::Relaxed) / num_queries as u64
            } else {
                0
            }
        };
        let breakdown = GraphIvfLatencyBreakdown {
            preprocess_ns: mean_ns(&accum.preprocess),
            centroid_search_ns: mean_ns(&accum.centroid_search),
            plan_io_ns: mean_ns(&accum.plan_io),
            disk_read_ns: mean_ns(&accum.disk_read),
            score_ns: mean_ns(&accum.score),
            topk_ns: mean_ns(&accum.topk),
            total_ns: mean_ns(&accum.total),
            io_count: mean_ns(&accum.io_count),
            bytes_read: mean_ns(&accum.bytes_read),
        };

        search_results_per_nlist.push(GraphIvfSearchResult {
            cluster_fraction: cluster_fraction.get(),
            nlist,
            qps: if total_secs > 0.0 {
                num_queries as f32 / total_secs
            } else {
                0.0
            },
            mean_latency: MicroSeconds::new(mean_us),
            p95_latency: MicroSeconds::new(percentile(0.95)),
            p999_latency: MicroSeconds::new(percentile(0.999)),
            recalls,
            breakdown,
        });
    }

    Ok(GraphIvfSearchStats {
        num_threads: search_params.num_threads,
        recall_at: recall_at.iter().collect(),
        distance: search_params.distance,
        centroid_search: index_load.centroid_search,
        centroid_search_alpha: search_params.centroid_search_alpha,
        search_results_per_nlist,
    })
}

/// L2-normalize every query row, returning a flat row-major buffer of `T`.
fn prepare_normalized_queries<T: VectorRepr>(queries: &Matrix<T>) -> anyhow::Result<Vec<T>> {
    let dim = queries.ncols();
    let mut out: Vec<T> = Vec::with_capacity(queries.nrows() * dim);
    let mut scratch = vec![0.0f32; dim];
    for row in queries.as_slice().chunks_exact(dim) {
        T::as_f32_into(row, &mut scratch)
            .map_err(|e| anyhow::anyhow!("failed to widen query to f32: {e}"))?;
        let norm = scratch.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in scratch.iter_mut() {
                *v /= norm;
            }
        }
        for &v in scratch.iter() {
            out.push(T::from_f32(v).ok_or_else(|| {
                anyhow::anyhow!("normalized query value not representable in target type")
            })?);
        }
    }
    Ok(out)
}

impl fmt::Display for GraphIvfSearchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // One recall column per configured `k`, between QPS and the latencies.
        let mut header: Vec<String> = vec!["NList".into(), "QPS".into()];
        header.extend(self.recall_at.iter().map(|k| format!("R@{k}")));
        header.extend([
            "MeanUs".into(),
            "P95Us".into(),
            "P999Us".into(),
            "Clusters%".into(),
        ]);
        let latency_col = 2 + self.recall_at.len();

        let mut table = Table::new(header, self.search_results_per_nlist.len());
        for (i, r) in self.search_results_per_nlist.iter().enumerate() {
            let mut row = table.row(i);
            row.insert(r.nlist.to_string(), 0);
            row.insert(format!("{:.1}", r.qps), 1);
            for (j, p) in r.recalls.iter().enumerate() {
                row.insert(format!("{:.2}", p.recall), 2 + j);
            }
            row.insert(format!("{}", r.mean_latency.as_micros()), latency_col);
            row.insert(format!("{}", r.p95_latency.as_micros()), latency_col + 1);
            row.insert(format!("{}", r.p999_latency.as_micros()), latency_col + 2);
            row.insert(
                format!("{:.4}", 100.0 * r.cluster_fraction),
                latency_col + 3,
            );
        }
        table.fmt(f)?;

        // Mean per-query latency layer cake (microseconds per stage).
        writeln!(f, "\nSearch latency breakdown (mean us/query):")?;
        let bd_header = [
            "NList",
            "Preproc",
            "Centroid",
            "PlanIO",
            "DiskRead",
            "Score",
            "TopK",
            "Total",
            "Clusters%",
        ];
        let mut bd = Table::new(bd_header, self.search_results_per_nlist.len());
        let us = |ns: u64| format!("{:.2}", ns as f64 / 1e3);
        for (i, r) in self.search_results_per_nlist.iter().enumerate() {
            let b = &r.breakdown;
            let mut row = bd.row(i);
            row.insert(r.nlist.to_string(), 0);
            row.insert(us(b.preprocess_ns), 1);
            row.insert(us(b.centroid_search_ns), 2);
            row.insert(us(b.plan_io_ns), 3);
            row.insert(us(b.disk_read_ns), 4);
            row.insert(us(b.score_ns), 5);
            row.insert(us(b.topk_ns), 6);
            row.insert(us(b.total_ns), 7);
            row.insert(format!("{:.4}", 100.0 * r.cluster_fraction), 8);
        }
        bd.fmt(f)?;

        // Per-query I/O volume (bytes moved and request counts).
        writeln!(f, "\nSearch I/O volume (mean per query):")?;
        let io_header = ["NList", "Reads", "DiskKiB", "Clusters%"];
        let mut io = Table::new(io_header, self.search_results_per_nlist.len());
        for (i, r) in self.search_results_per_nlist.iter().enumerate() {
            let b = &r.breakdown;
            let mut row = io.row(i);
            row.insert(r.nlist.to_string(), 0);
            row.insert(b.io_count.to_string(), 1);
            row.insert(format!("{:.1}", b.bytes_read as f64 / 1024.0), 2);
            row.insert(format!("{:.4}", 100.0 * r.cluster_fraction), 3);
        }
        io.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truthset(num_queries: usize, depth: usize) -> TruthSet {
        TruthSet {
            index_nodes: (0..num_queries * depth).map(|id| id as u32).collect(),
            distances: None,
            index_num_points: num_queries,
            index_dimension: depth,
        }
    }

    #[test]
    fn recall_rejects_too_few_groundtruth_queries() {
        let error = recall_points(
            &RecallAt::new(vec![2]),
            2,
            &truthset(1, 2),
            &[0, 1, 0, 1],
            2,
        )
        .unwrap_err();
        assert!(error.to_string().contains("groundtruth has 1 queries"));
    }

    #[test]
    fn recall_rejects_shallow_groundtruth() {
        let error =
            recall_points(&RecallAt::new(vec![2]), 1, &truthset(1, 1), &[0, 1], 2).unwrap_err();
        assert!(error.to_string().contains("groundtruth has 1 neighbors"));
    }

    #[test]
    fn recall_rejects_wrong_result_buffer_size() {
        let error =
            recall_points(&RecallAt::new(vec![2]), 1, &truthset(1, 2), &[0], 2).unwrap_err();
        assert!(error.to_string().contains("result buffer has 1 ids"));
    }
}
