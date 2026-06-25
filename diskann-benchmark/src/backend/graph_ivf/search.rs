/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    sync::atomic::{AtomicBool, AtomicU64},
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
use diskann_tools::utils::{search_index_utils, KRecallAtN};
use diskann_utils::views::Matrix;

use crate::{
    backend::graph_ivf::build::to_graphivf_metric,
    inputs::graph_ivf::{GraphIvfLoad, GraphIvfSearchPhase},
    utils::{datafiles, SimilarityMeasure},
};

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfSearchStats {
    pub(super) num_threads: usize,
    pub(super) recall_at: u32,
    pub(super) distance: SimilarityMeasure,
    pub(super) centroid_search_l: usize,
    pub(super) search_results_per_nlist: Vec<GraphIvfSearchResult>,
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfSearchResult {
    pub(super) nlist: usize,
    pub(super) qps: f32,
    pub(super) mean_latency: MicroSeconds,
    pub(super) p95_latency: MicroSeconds,
    pub(super) p999_latency: MicroSeconds,
    pub(super) recall: f32,
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
}

/// Thread-shared accumulator summing each stage's wall-clock (in nanoseconds)
/// across all queries of one nlist sweep.
#[derive(Default)]
struct PhaseAccum {
    preprocess: AtomicU64,
    centroid_search: AtomicU64,
    plan_io: AtomicU64,
    disk_read: AtomicU64,
    score: AtomicU64,
    topk: AtomicU64,
    total: AtomicU64,
}

pub(super) fn search_graph_ivf<T>(
    index_load: &GraphIvfLoad,
    search_params: &GraphIvfSearchPhase,
) -> anyhow::Result<GraphIvfSearchStats>
where
    T: VectorRepr,
{
    use std::sync::atomic::Ordering;

    // Map the distance measure and decide whether queries must be normalized.
    // Cosine normalizes the corpus at build time, so queries must be normalized
    // to match; L2 / already-normalized cosine leave queries untouched.
    let metric = to_graphivf_metric(search_params.distance)?;
    let normalize_queries = matches!(metric, diskann_graphivf::Metric::Cosine);

    // Load the index from disk.
    let index = GraphIvfIndex::<T>::load(
        std::path::Path::new(&index_load.load_path),
        search_params.num_threads,
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

    let recall_at = search_params.recall_at as usize;
    let pool = create_thread_pool(search_params.num_threads)?;
    let mut search_results_per_nlist = Vec::with_capacity(search_params.nlist.len());

    for &nlist in search_params.nlist.iter() {
        if nlist > num_clusters {
            anyhow::bail!(
                "nlist ({nlist}) cannot exceed the index's num_clusters ({num_clusters})"
            );
        }
        let params = SearchParams {
            nlist,
            centroid_search_l: search_params.centroid_search_l,
        };

        let mut result_ids: Vec<u32> = vec![0; recall_at * num_queries];
        let mut latencies_us: Vec<u64> = vec![0; num_queries];
        let failed = AtomicBool::new(false);
        let accum = PhaseAccum::default();

        let zipped = prepared
            .par_chunks(dim)
            .zip(result_ids.par_chunks_mut(recall_at))
            .zip(latencies_us.par_iter_mut());

        let start = Instant::now();
        zipped.for_each_init_in_pool(
            pool.as_ref(),
            // Each worker thread owns its own searcher (disk reader + runtime).
            // Pre-validated above, so this should not fail in practice.
            || {
                index
                    .searcher()
                    .expect("searcher creation failed after pre-validation")
            },
            |searcher, ((query, id_chunk), latency)| {
                let q_start = Instant::now();
                match searcher.search_profiled(query, recall_at, &params) {
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
                    }
                    Err(e) => {
                        eprintln!("graph-ivf search failed for a query: {e:?}");
                        id_chunk.fill(0);
                        failed.store(true, Ordering::Release);
                    }
                }
                *latency = q_start.elapsed().as_micros() as u64;
            },
        );
        let total_time = start.elapsed();

        if failed.load(Ordering::Acquire) {
            anyhow::bail!("one or more graph-ivf searches failed; see logs for details");
        }

        let recall = search_index_utils::calculate_recall(
            num_queries,
            &gt.index_nodes,
            gt.distances.as_ref(),
            gt.index_dimension,
            &result_ids,
            recall_at as u32,
            KRecallAtN::new(recall_at as u32, recall_at as u32)?,
        )? as f32;

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
        };

        search_results_per_nlist.push(GraphIvfSearchResult {
            nlist,
            qps: if total_secs > 0.0 {
                num_queries as f32 / total_secs
            } else {
                0.0
            },
            mean_latency: MicroSeconds::new(mean_us),
            p95_latency: MicroSeconds::new(percentile(0.95)),
            p999_latency: MicroSeconds::new(percentile(0.999)),
            recall,
            breakdown,
        });
    }

    Ok(GraphIvfSearchStats {
        num_threads: search_params.num_threads,
        recall_at: search_params.recall_at,
        distance: search_params.distance,
        centroid_search_l: search_params.centroid_search_l,
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
        let header = ["NList", "QPS", "Recall", "MeanUs", "P95Us", "P999Us"];
        let mut table = Table::new(header, self.search_results_per_nlist.len());
        for (i, r) in self.search_results_per_nlist.iter().enumerate() {
            let mut row = table.row(i);
            row.insert(r.nlist.to_string(), 0);
            row.insert(format!("{:.1}", r.qps), 1);
            row.insert(format!("{:.2}", r.recall), 2);
            row.insert(format!("{}", r.mean_latency.as_micros()), 3);
            row.insert(format!("{}", r.p95_latency.as_micros()), 4);
            row.insert(format!("{}", r.p999_latency.as_micros()), 5);
        }
        table.fmt(f)?;

        // Mean per-query latency layer cake (microseconds per stage).
        writeln!(f, "\nSearch latency breakdown (mean us/query):")?;
        let bd_header = [
            "NList", "Preproc", "Centroid", "PlanIO", "DiskRead", "Score", "TopK", "Total",
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
        }
        bd.fmt(f)
    }
}
