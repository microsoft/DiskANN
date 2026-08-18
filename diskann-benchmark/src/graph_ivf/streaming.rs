/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Runbook-driven online graph-IVF build.
//!
//! Replays a BigANN streaming runbook against a live [`OnlineClusterer`]: insert
//! stages feed corpus rows in, delete stages take them back out, and search
//! stages measure recall against the groundtruth the runbook names for that
//! point in the stream. The index is flushed once the runbook ends, so the job's
//! `search_phase` still measures the on-disk result.
//!
//! The clusterer addresses points by corpus row, and a BigANN insert stage uses
//! one range as both the dataset offsets *and* the external ids, so ids and rows
//! coincide and no translation layer is needed. That also means a `Replace`
//! stage — whose two ranges differ — has no meaning here and is rejected.
//!
//! Searches run against the in-memory `f32` corpus. That skips the quantization
//! error the flushed index would add, so recall read here is an upper bound for a
//! quantized element type and exact for `f32`.

use std::{
    fmt,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::Instant,
};

use diskann_benchmark_core::streaming::{executors::bigann, Executor as _, Stream};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_graphivf::SearchParams;
use diskann_providers::{
    storage::FileStorageProvider,
    utils::{create_thread_pool, ParallelIteratorInPool, RayonThreadPool},
};
use diskann_tools::utils::search_index_utils;
use diskann_utils::views::Matrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    graph_ivf::{
        build::decompress_to_f32,
        element::GraphIvfElement,
        online::{online_setup, OnlineSetup},
        search::{recall_points, record_first_error, RecallPoint},
    },
    inputs::graph_ivf::{CentroidSearchConfig, GraphIvfOnlineRunbook},
    utils::{datafiles, SimilarityMeasure},
};

////////////
// Output //
////////////

/// One measured sweep at one search stage.
#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfRunbookSearchResult {
    /// Requested share of the live clusters at this stage.
    pub(super) cluster_fraction: f64,
    /// Concrete number of clusters probed after rounding the fraction up.
    pub(super) nlist: usize,
    /// One entry per configured `recall_at`, all scored from the same sweep.
    /// Percentages, matching the search phase's convention.
    pub(super) recalls: Vec<RecallPoint>,
    /// Mean corpus vectors scored per query. A cluster fraction alone does not
    /// determine this — list sizes remain imbalanced as the partition churns —
    /// so it is the cost axis recall is meaningfully read against.
    pub(super) mean_points_scanned: u64,
    /// `mean_points_scanned` as a percentage of the live set at this stage.
    pub(super) pct_scanned: f32,
    /// Mean cost of one query, measured around each query individually so that
    /// it stays comparable no matter how many threads ran the sweep.
    pub(super) mean_latency: MicroSeconds,
    /// Mean fraction of the probed clusters that were genuinely the nearest
    /// ones, as a percentage. `None` unless `measure_centroid_recall` is set.
    ///
    /// Anything below 100 is recall this sweep lost before scanning a single
    /// point, and no amount of scanning gets it back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) centroid_recall: Option<f32>,
}

/// Size distribution of the live inverted lists after a stage.
///
/// The mean is fixed by the live point and cluster counts, so the spread is
/// what carries information: it is the shape of the partition, and therefore
/// what a fixed cluster fraction actually costs to probe.
#[derive(Serialize, Deserialize, Debug)]
pub(super) struct ClusterSizeStats {
    pub(super) min: usize,
    pub(super) p10: usize,
    pub(super) p50: usize,
    pub(super) p90: usize,
    pub(super) p99: usize,
    pub(super) max: usize,
    pub(super) mean: f32,
    /// Clusters below `merge_threshold`: underfull, and merged by the next
    /// delete that touches them.
    pub(super) under: usize,
    /// Clusters above `split_threshold`: overfull, and split by the next insert
    /// routed to them. A merge can leave one behind, since the partner is
    /// chosen without a size guard.
    pub(super) over: usize,
}

impl ClusterSizeStats {
    /// Summarize `sizes`, which is sorted in place. `under` and `over` count
    /// against the configured thresholds; a zero `merge_threshold` (merging
    /// disabled) leaves `under` at zero.
    fn new(sizes: &mut [usize], merge_threshold: usize, split_threshold: usize) -> Self {
        sizes.sort_unstable();
        let n = sizes.len();
        if n == 0 {
            return Self {
                min: 0,
                p10: 0,
                p50: 0,
                p90: 0,
                p99: 0,
                max: 0,
                mean: 0.0,
                under: 0,
                over: 0,
            };
        }
        // Nearest-rank percentile: `sizes` is ascending, so the number of
        // entries strictly below a threshold is its partition point.
        let pct = |q: usize| sizes[(q * n / 100).min(n - 1)];
        let total: usize = sizes.iter().sum();
        Self {
            min: sizes[0],
            p10: pct(10),
            p50: pct(50),
            p90: pct(90),
            p99: pct(99),
            max: sizes[n - 1],
            mean: total as f32 / n as f32,
            under: sizes.partition_point(|&s| s < merge_threshold),
            over: n - sizes.partition_point(|&s| s <= split_threshold),
        }
    }
}

/// What one runbook stage did, and what the index looked like afterwards.
#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfStageStats {
    pub(super) stage: usize,
    pub(super) kind: StageKind,
    /// Points inserted or deleted; zero for a search stage.
    pub(super) num_points: usize,
    pub(super) elapsed: MicroSeconds,
    /// Live clusters after the stage.
    pub(super) live_clusters: usize,
    /// Points held by the index after the stage.
    pub(super) live_points: usize,
    /// Size distribution of those clusters after the stage.
    pub(super) sizes: ClusterSizeStats,
    /// Out-edge health of the centroid graph after the stage. Measured on
    /// search stages only, where it lines up with centroid recall.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) centroid_adjacency: Option<CentroidAdjacencyStats>,
    /// One entry per cluster fraction; empty for insert and delete stages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) search: Vec<GraphIvfRunbookSearchResult>,
}

/// How much of the centroid graph's out-degree still points at live centroids.
///
/// Search traverses tombstones like any other node, so a low live fraction
/// means the candidate list is diluted with dead ends.
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub(super) struct CentroidAdjacencyStats {
    pub(super) out_edges: u64,
    pub(super) live_out_edges: u64,
    /// `live_out_edges / out_edges`, as a percentage.
    pub(super) live_pct: f64,
    pub(super) mean_live_degree: f64,
    /// Live centroids with no live out-edge at all.
    pub(super) starved: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum StageKind {
    Insert,
    Delete,
    Search,
}

impl fmt::Display for StageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Insert => write!(f, "insert"),
            Self::Delete => write!(f, "delete"),
            Self::Search => write!(f, "search"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfRunbookStats {
    /// How clusters were located throughout the replay.
    centroid_search: CentroidSearchConfig,
    corpus_load: MicroSeconds,
    decompress: MicroSeconds,
    seed: MicroSeconds,
    /// Time replaying the runbook, searches included.
    stream: MicroSeconds,
    flush: MicroSeconds,
    /// Cumulative time routing inserts through the centroid graph.
    routing: MicroSeconds,
    /// Cumulative time in split handling (2-means, graph mutation, reassignment).
    split: MicroSeconds,
    /// Cumulative time removing points from their inverted lists.
    delete: MicroSeconds,
    /// Cumulative time in merge handling (k-means, graph mutation, reassignment).
    merge: MicroSeconds,
    num_points: usize,
    dim: usize,
    centroid_capacity: usize,
    seeded_clusters: usize,
    final_clusters: usize,
    final_points: usize,
    total_inserts: u64,
    total_deletes: u64,
    total_splits: u64,
    total_merges: u64,
    /// Points that changed cluster because of a split.
    total_reassigned: u64,
    /// Points that changed cluster because of a merge.
    total_merge_reassigned: u64,
    stages: Vec<GraphIvfStageStats>,
}

impl fmt::Display for GraphIvfRunbookStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Online runbook: {:.3}s ({} stages over {} corpus points, dim {})",
            self.stream.as_seconds(),
            self.stages.len(),
            self.num_points,
            self.dim
        )?;
        writeln!(f, "  centroid search: {:?}", self.centroid_search)?;
        writeln!(f, "  corpus_load:    {:.3}s", self.corpus_load.as_seconds())?;
        writeln!(f, "  decompress:     {:.3}s", self.decompress.as_seconds())?;
        writeln!(f, "  seed:           {:.3}s", self.seed.as_seconds())?;
        writeln!(f, "  stream:         {:.3}s", self.stream.as_seconds())?;
        writeln!(f, "  flush:          {:.3}s", self.flush.as_seconds())?;
        writeln!(f, "  routing:        {:.3}s", self.routing.as_seconds())?;
        writeln!(f, "  split:          {:.3}s", self.split.as_seconds())?;
        writeln!(f, "  delete:         {:.3}s", self.delete.as_seconds())?;
        writeln!(f, "  merge:          {:.3}s", self.merge.as_seconds())?;
        writeln!(f, "  id budget:      {}", self.centroid_capacity)?;
        writeln!(
            f,
            "  clusters:       {} seeded -> {}",
            self.seeded_clusters, self.final_clusters
        )?;
        writeln!(
            f,
            "  live points:    {} of {}",
            self.final_points, self.num_points
        )?;
        writeln!(
            f,
            "  inserts/deletes:{} / {}",
            self.total_inserts, self.total_deletes
        )?;
        writeln!(
            f,
            "  splits/merges:  {} ({} moved) / {} ({} moved)",
            self.total_splits,
            self.total_reassigned,
            self.total_merges,
            self.total_merge_reassigned
        )?;
        for stage in &self.stages {
            write!(
                f,
                "  [{:>4}] {:<6} n={:<8} {:>8.3}s  clusters={:<6} points={}",
                stage.stage,
                stage.kind.to_string(),
                stage.num_points,
                stage.elapsed.as_seconds(),
                stage.live_clusters,
                stage.live_points
            )?;
            let d = &stage.sizes;
            write!(
                f,
                "\n         sizes: min={} p10={} p50={} p90={} p99={} max={} mean={:.1}  under={} over={}",
                d.min, d.p10, d.p50, d.p90, d.p99, d.max, d.mean, d.under, d.over
            )?;
            for r in &stage.search {
                write!(
                    f,
                    "\n         nlist={:<6} clusters={:>7.4}%",
                    r.nlist,
                    100.0 * r.cluster_fraction
                )?;
                for p in &r.recalls {
                    write!(f, " r@{}={:.2}", p.at, p.recall)?;
                }
                write!(
                    f,
                    "  scan={:<8} ({:.2}%)  mean={:.0}us",
                    r.mean_points_scanned,
                    r.pct_scanned,
                    r.mean_latency.as_seconds() * 1e6
                )?;
                if let Some(centroid) = r.centroid_recall {
                    write!(f, "  centroid={centroid:.2}%")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

////////////
// Driver //
////////////

pub(super) fn build_graph_ivf_runbook<T>(
    params: &GraphIvfOnlineRunbook,
) -> anyhow::Result<GraphIvfRunbookStats>
where
    T: GraphIvfElement,
{
    // Stored rows are written to the lists verbatim, so an online build cannot
    // normalize the corpus; queries must match it as stored.
    anyhow::ensure!(
        params.build.distance != SimilarityMeasure::Cosine,
        "online runbook builds store corpus rows verbatim and cannot normalize them; \
         pre-normalize the corpus and queries and use `cosine_normalized`"
    );

    let OnlineSetup {
        corpus,
        clusterer,
        dim,
        centroid_capacity,
        seeded_clusters,
        corpus_load,
        decompress,
        seed,
    } = online_setup::<T>(&params.build)?;
    let num_points = corpus.nrows();

    // Queries are stored as `T` like the corpus; the clusterer scores in `f32`,
    // so widen them once here rather than per query.
    let stored_queries: Matrix<T> =
        datafiles::load_dataset(datafiles::BinFile(&params.search.queries))?;
    let queries = decompress_to_f32(&stored_queries, dim)?;
    anyhow::ensure!(
        queries.ncols() == dim,
        "query dimension {} does not match corpus dimension {dim}",
        queries.ncols()
    );

    let gt_directory = params
        .runbook
        .resolved_gt_directory
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("groundtruth directory was not resolved by validation"))?;
    let mut runbook = bigann::RunBook::load(
        &params.runbook.runbook_path,
        &params.runbook.dataset_name,
        &mut bigann::ScanDirectory::new(gt_directory)?,
    )?;

    // Every centroid id ever allocated is permanent — a split retires the parent
    // and a merge retires the victim — so a long runbook can exhaust a budget
    // sized for a single pass over the corpus. Fail before the run rather than
    // partway through it.
    anyhow::ensure!(
        runbook.max_points() <= num_points,
        "runbook touches {} points but the corpus holds only {num_points}",
        runbook.max_points()
    );

    let mut stream = GraphIvfStream {
        clusterer,
        queries,
        dim,
        params,
        pool: create_thread_pool(params.search.num_threads)?,
        stage: 0,
        total_stages: runbook.len(),
        started: Instant::now(),
    };

    let stream_start = Instant::now();
    let mut stages = Vec::with_capacity(runbook.len());
    runbook.run_with(&mut stream, |s: GraphIvfStageStats| {
        stages.push(s);
        Ok(())
    })?;
    let stream_elapsed: MicroSeconds = stream_start.elapsed().into();

    let clusterer = stream.clusterer;
    let flush_start = Instant::now();
    clusterer.flush::<T>(Path::new(&params.build.save_path), corpus.as_view())?;
    let flush: MicroSeconds = flush_start.elapsed().into();

    let telemetry = clusterer.telemetry();
    if let Some(csv) = &params.build.telemetry_csv {
        let path = Path::new(csv);
        telemetry
            .write_csv(path)
            .map_err(|e| anyhow::anyhow!("failed to write split telemetry CSV {csv}: {e}"))?;
        let merges = merge_csv_path(path);
        telemetry.write_merges_csv(&merges).map_err(|e| {
            anyhow::anyhow!(
                "failed to write merge telemetry CSV {}: {e}",
                merges.display()
            )
        })?;
    }

    Ok(GraphIvfRunbookStats {
        centroid_search: params.build.routing.mode(),
        corpus_load,
        decompress,
        seed,
        stream: stream_elapsed,
        flush,
        routing: MicroSeconds::new(telemetry.routing_us),
        split: MicroSeconds::new(telemetry.split_us),
        delete: MicroSeconds::new(telemetry.delete_us),
        merge: MicroSeconds::new(telemetry.merge_us),
        num_points,
        dim,
        centroid_capacity,
        seeded_clusters,
        final_clusters: clusterer.num_clusters(),
        final_points: clusterer.cluster_sizes().iter().sum(),
        total_inserts: telemetry.total_inserts,
        total_deletes: telemetry.total_deletes,
        total_splits: telemetry.total_splits,
        total_merges: telemetry.total_merges,
        total_reassigned: telemetry.total_reassigned,
        total_merge_reassigned: telemetry.total_merge_reassigned,
        stages,
    })
}

/// Sibling path for the merge telemetry, next to the configured split CSV.
///
/// The two have different schemas, so they cannot share a file; deriving the
/// second name keeps the config to a single knob.
fn merge_csv_path(split_csv: &Path) -> std::path::PathBuf {
    let stem = split_csv
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "telemetry".to_string());
    let extension = split_csv
        .extension()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "csv".to_string());
    split_csv.with_file_name(format!("{stem}_merges.{extension}"))
}

////////////
// Stream //
////////////

struct GraphIvfStream<'a> {
    clusterer: diskann_graphivf::OnlineClusterer,
    /// Query rows widened to `f32`, matching the clusterer's corpus copy.
    queries: Matrix<f32>,
    dim: usize,
    params: &'a GraphIvfOnlineRunbook,
    /// Built once and reused by every sweep of every search stage.
    pool: RayonThreadPool,
    stage: usize,
    /// Stage count of the whole runbook, for the progress lines' denominator.
    total_stages: usize,
    /// Start of the replay, so each progress line carries cumulative elapsed.
    started: Instant,
}

impl GraphIvfStream<'_> {
    /// Wrap a completed stage in its stats record, snapshotting the index state.
    ///
    /// Also prints the stage's one-line progress record. Stage stats are only
    /// serialized after the whole runbook replays and the index flushes, so
    /// without this a multi-hour run is opaque until the moment it succeeds.
    fn finish(
        &mut self,
        kind: StageKind,
        num_points: usize,
        elapsed: std::time::Duration,
        search: Vec<GraphIvfRunbookSearchResult>,
    ) -> anyhow::Result<GraphIvfStageStats> {
        let mut sizes = self.clusterer.cluster_sizes();
        let centroid_adjacency = (kind == StageKind::Search)
            .then(|| self.clusterer.centroid_adjacency_census())
            .transpose()?
            .flatten()
            .map(|c| CentroidAdjacencyStats {
                out_edges: c.out_edges,
                live_out_edges: c.live_out_edges,
                live_pct: c.live_fraction() * 100.0,
                mean_live_degree: c.mean_live_degree(),
                starved: c.starved,
            });
        let stats = GraphIvfStageStats {
            stage: self.stage,
            kind,
            num_points,
            elapsed: elapsed.into(),
            live_clusters: self.clusterer.num_clusters(),
            live_points: sizes.iter().sum(),
            sizes: ClusterSizeStats::new(
                &mut sizes,
                self.params.build.merge_threshold,
                self.params.build.split_threshold,
            ),
            centroid_adjacency,
            search,
        };
        self.report(&stats);
        self.stage += 1;
        Ok(stats)
    }

    /// Print one stage's progress, plus a line per sweep for a search stage.
    ///
    /// `std::io::Stdout` is line-buffered, so each line reaches a redirected log
    /// as it is written rather than at process exit.
    fn report(&self, stats: &GraphIvfStageStats) {
        println!(
            "[stage {}/{}] {:<6} {:>9} pts  stage {:>8.2}s  run {:>8.1}s  clusters {:>9}  live {:>11}",
            stats.stage + 1,
            self.total_stages,
            stats.kind,
            stats.num_points,
            stats.elapsed.as_seconds(),
            self.started.elapsed().as_secs_f64(),
            stats.live_clusters,
            stats.live_points,
        );
        if let Some(a) = &stats.centroid_adjacency {
            println!(
                "    centroid graph: live edges {}/{} ({:.2}%)  mean live degree {:.2}  starved {}",
                a.live_out_edges, a.out_edges, a.live_pct, a.mean_live_degree, a.starved,
            );
        }
        for result in &stats.search {
            let recalls = result
                .recalls
                .iter()
                .map(|r| format!("r@{}={:.2}", r.at, r.recall))
                .collect::<Vec<_>>()
                .join(" ");
            println!(
                "    f={:<6.4} nlist={:<7} {}  scanned {:>9} ({:>5.2}%)  {:>8.2}ms/query{}",
                result.cluster_fraction,
                result.nlist,
                recalls,
                result.mean_points_scanned,
                result.pct_scanned,
                result.mean_latency.as_seconds() * 1000.0,
                result
                    .centroid_recall
                    .map(|r| format!("  centroid={r:.2}%"))
                    .unwrap_or_default(),
            );
        }
    }

    /// Run every configured cluster-fraction sweep against the current index.
    fn sweep(&self, groundtruth: &Path) -> anyhow::Result<Vec<GraphIvfRunbookSearchResult>> {
        let search = &self.params.search;
        let k_max = search.recall_at.max() as usize;
        let num_queries = self.queries.nrows();
        let num_clusters = self.clusterer.num_clusters();
        let live_points: usize = self.clusterer.cluster_sizes().iter().sum();

        let gt = search_index_utils::load_truthset(
            &FileStorageProvider,
            &groundtruth.to_string_lossy(),
        )?;

        anyhow::ensure!(
            num_clusters > 0,
            "cannot search an online graph-IVF index with no live clusters"
        );
        // Confirm a searcher can be created before entering the parallel region;
        // the per-thread initializer below reuses the same fallible call.
        let _ = self.clusterer.searcher()?;

        let mut results = Vec::with_capacity(search.cluster_fractions.len());
        for &cluster_fraction in &search.cluster_fractions {
            // Resolve at every stage because inserts, deletes, splits, and
            // dissolves all change the live cluster count over the runbook.
            let nlist = cluster_fraction.nlist(num_clusters);
            let params = SearchParams {
                nlist,
                centroid_search_alpha: search.centroid_search_alpha,
            };
            let mut result_ids = vec![0u32; k_max * num_queries];
            let first_error = Mutex::new(None::<String>);
            let scanned = AtomicU64::new(0);
            let latency_ns = AtomicU64::new(0);
            // Summed as a count rather than a mean so the accumulation stays
            // integral and order-independent across workers.
            let centroids_matched = AtomicU64::new(0);

            self.queries
                .as_slice()
                .par_chunks(self.dim)
                .zip(result_ids.par_chunks_mut(k_max))
                .for_each_init_in_pool(
                    self.pool.as_ref(),
                    // Each worker owns its own searcher; the handle is not shareable.
                    || {
                        self.clusterer
                            .searcher()
                            .map(|searcher| (searcher, Vec::with_capacity(k_max)))
                            .map_err(|error| format!("{error:#}"))
                    },
                    |state, (q, ids)| {
                        let start = Instant::now();
                        let (searcher, hits) = match state {
                            Ok(state) => state,
                            Err(error) => {
                                ids.fill(0);
                                record_first_error(
                                    &first_error,
                                    format!("failed to create a worker searcher: {error}"),
                                );
                                return;
                            }
                        };
                        match searcher.search_into(q, k_max, &params, hits) {
                            Ok(stats) => {
                                scanned.fetch_add(stats.points_scanned as u64, Ordering::Relaxed);
                                for (slot, (id, _)) in ids.iter_mut().zip(hits.iter()) {
                                    *slot = *id;
                                }
                            }
                            Err(error) => {
                                ids.fill(0);
                                record_first_error(&first_error, format!("{error:#}"));
                            }
                        }
                        latency_ns.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);

                        // Measured after the timed region: an exact centroid
                        // scan costs far more than the query it is scoring, and
                        // must not land in the reported latency.
                        if search.measure_centroid_recall {
                            match searcher.centroid_recall(q, &params) {
                                Ok(centroid) => {
                                    centroids_matched
                                        .fetch_add(centroid.matched as u64, Ordering::Relaxed);
                                }
                                Err(error) => {
                                    record_first_error(&first_error, format!("{error:#}"));
                                }
                            }
                        }
                    },
                );

            if let Some(error) = first_error
                .into_inner()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
            {
                anyhow::bail!("one or more graph-ivf runbook searches failed: {error}");
            }

            let queries = num_queries.max(1) as u64;
            let mean_points_scanned = scanned.load(Ordering::Relaxed) / queries;
            let recalls = recall_points(&search.recall_at, num_queries, &gt, &result_ids, k_max)?;

            results.push(GraphIvfRunbookSearchResult {
                cluster_fraction: cluster_fraction.get(),
                nlist,
                recalls,
                mean_points_scanned,
                pct_scanned: 100.0 * mean_points_scanned as f32 / live_points.max(1) as f32,
                mean_latency: MicroSeconds::new(
                    latency_ns.load(Ordering::Relaxed) / queries / 1000,
                ),
                centroid_recall: search.measure_centroid_recall.then(|| {
                    let requested = queries * nlist as u64;
                    100.0 * centroids_matched.load(Ordering::Relaxed) as f32
                        / requested.max(1) as f32
                }),
            });
        }
        Ok(results)
    }
}

impl Stream<bigann::Args> for GraphIvfStream<'_> {
    type Output = GraphIvfStageStats;

    fn search(&mut self, args: bigann::Search<'_>) -> anyhow::Result<Self::Output> {
        let start = Instant::now();
        let results = self.sweep(args.groundtruth)?;
        self.finish(StageKind::Search, 0, start.elapsed(), results)
    }

    fn insert(&mut self, args: bigann::Insert) -> anyhow::Result<Self::Output> {
        // A BigANN insert stage uses one range as both the dataset offsets and
        // the external ids, which is exactly the identity this backend relies on.
        anyhow::ensure!(
            args.offsets == args.ids,
            "graph-IVF addresses points by corpus row, so an insert stage's offsets \
             ({:?}) and ids ({:?}) must coincide",
            args.offsets,
            args.ids
        );
        let ids: Vec<u32> = args.ids.clone().map(|i| i as u32).collect();

        let start = Instant::now();
        for batch in ids.chunks(self.params.build.batch_size) {
            self.clusterer.insert_batch(batch)?;
        }
        self.finish(StageKind::Insert, ids.len(), start.elapsed(), Vec::new())
    }

    fn replace(&mut self, _args: bigann::Replace) -> anyhow::Result<Self::Output> {
        anyhow::bail!(
            "graph-IVF addresses points by corpus row, so a point cannot take on another \
             point's vector; use a runbook without replace stages"
        )
    }

    fn delete(&mut self, args: bigann::Delete) -> anyhow::Result<Self::Output> {
        let ids: Vec<u32> = args.ids.clone().map(|i| i as u32).collect();

        let start = Instant::now();
        for batch in ids.chunks(self.params.build.batch_size) {
            self.clusterer.delete_batch(batch)?;
        }
        self.finish(StageKind::Delete, ids.len(), start.elapsed(), Vec::new())
    }

    fn maintain(&mut self, _args: ()) -> anyhow::Result<Self::Output> {
        // Unreachable: `needs_maintenance` never asks for one.
        anyhow::bail!("graph-IVF has no maintenance operation")
    }

    fn needs_maintenance(&mut self) -> bool {
        // Deletes are eager — a deleted point leaves its inverted list at once —
        // so there is no deferred consolidation to run.
        false
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{fs, io::Write as _, path::PathBuf};

    use diskann_benchmark_runner::files::InputFile;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::inputs::graph_ivf::{
        ClusterFraction, GraphIvfOnlineBuild, GraphIvfRunbookConfig, GraphIvfRunbookSearch,
        OnlineRoutingConfig, RecallAt,
    };

    const DIM: usize = 8;
    const NUM_POINTS: usize = 600;
    const NUM_QUERIES: usize = 10;
    const RECALL_AT: usize = 10;
    const DATASET: &str = "unit-test";

    /// Write a `npts x ncols` `f32` matrix in `.bin` format.
    fn write_bin(path: &Path, data: &[f32], npts: usize, ncols: usize) {
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(npts as u32).to_le_bytes()).unwrap();
        file.write_all(&(ncols as u32).to_le_bytes()).unwrap();
        file.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    /// Write an ids-only truthset: the same header, then `npts * k` `u32` ids.
    fn write_truthset(path: &Path, ids: &[u32], npts: usize, k: usize) {
        let mut file = fs::File::create(path).unwrap();
        file.write_all(&(npts as u32).to_le_bytes()).unwrap();
        file.write_all(&(k as u32).to_le_bytes()).unwrap();
        file.write_all(bytemuck::cast_slice(ids)).unwrap();
    }

    /// Exact top-`k` by squared L2 over `live` only, so a stage's groundtruth
    /// reflects the points the index actually holds at that moment.
    fn brute_force(corpus: &[f32], queries: &[f32], live: &[u32], k: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity((queries.len() / DIM) * k);
        for query in queries.chunks(DIM) {
            let mut scored: Vec<(f32, u32)> = live
                .iter()
                .map(|&i| {
                    let point = &corpus[i as usize * DIM..][..DIM];
                    let d = point
                        .iter()
                        .zip(query)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (d, i)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ids.extend(scored.iter().take(k).map(|(_, i)| *i));
        }
        ids
    }

    struct Fixture {
        dir: PathBuf,
        corpus: Vec<f32>,
        queries: Vec<f32>,
        data: InputFile,
        query_file: InputFile,
        gt_directory: PathBuf,
        save_path: String,
    }

    fn fixture(dir: &Path) -> Fixture {
        let mut rng = StdRng::seed_from_u64(11);
        let corpus: Vec<f32> = (0..NUM_POINTS * DIM).map(|_| rng.random::<f32>()).collect();
        let queries: Vec<f32> = (0..NUM_QUERIES * DIM)
            .map(|_| rng.random::<f32>())
            .collect();

        let data = dir.join("corpus.bin");
        let query_path = dir.join("queries.bin");
        let gt_directory = dir.join("gt");
        write_bin(&data, &corpus, NUM_POINTS, DIM);
        write_bin(&query_path, &queries, NUM_QUERIES, DIM);
        fs::create_dir_all(&gt_directory).unwrap();

        Fixture {
            dir: dir.to_path_buf(),
            corpus,
            queries,
            data: InputFile::new(data),
            query_file: InputFile::new(query_path),
            gt_directory,
            save_path: dir.join("index").to_string_lossy().into_owned(),
        }
    }

    impl Fixture {
        /// Write the groundtruth a search stage will look for, over `live`.
        fn write_gt(&self, stage: usize, live: &[u32]) {
            let truth = brute_force(&self.corpus, &self.queries, live, RECALL_AT);
            write_truthset(
                &self.gt_directory.join(format!("step{stage}.gt100")),
                &truth,
                NUM_QUERIES,
                RECALL_AT,
            );
        }

        /// Write a runbook YAML whose body is `stages`, already indented.
        fn write_runbook(&self, name: &str, max_pts: usize, stages: &str) -> InputFile {
            let path = self.dir.join(name);
            fs::write(&path, format!("{DATASET}:\n  max_pts: {max_pts}\n{stages}")).unwrap();
            InputFile::new(path)
        }

        fn params(&self, runbook_path: InputFile, merge_threshold: usize) -> GraphIvfOnlineRunbook {
            GraphIvfOnlineRunbook {
                build: GraphIvfOnlineBuild {
                    data_type: <f32 as GraphIvfElement>::DATA_TYPE,
                    data: self.data.clone(),
                    distance: SimilarityMeasure::SquaredL2,
                    dim: DIM,
                    split_threshold: 64,
                    batch_size: 32,
                    max_clusters: None,
                    warmup_centroids: 4,
                    warmup_points: 100,
                    warmup_iters: 5,
                    two_means_iters: 8,
                    reassign_neighbors: 4,
                    merge_threshold,
                    min_clusters: 1,
                    capacity_mult: 3,
                    normalize: false,
                    routing: OnlineRoutingConfig::Graph {
                        assign_l: 32,
                        reassign_l: Some(32),
                        graph_degree: 16,
                        graph_slack: 1.2,
                        graph_l_build: 32,
                        graph_alpha: 1.2,
                    },
                    num_threads: 2,
                    seed: 0,
                    save_path: self.save_path.clone(),
                    telemetry_csv: None,
                },
                runbook: GraphIvfRunbookConfig {
                    runbook_path,
                    dataset_name: DATASET.to_string(),
                    gt_directory: self.gt_directory.to_string_lossy().into_owned(),
                    resolved_gt_directory: Some(self.gt_directory.clone()),
                },
                search: GraphIvfRunbookSearch {
                    queries: self.query_file.clone(),
                    // Probe every cluster, so every sweep is exhaustive and
                    // recall is exact however the cluster count changes. The
                    // beam scales with nlist, so it stays exhaustive too.
                    cluster_fractions: vec![ClusterFraction::new(1.0)],
                    centroid_search_alpha: 1.5,
                    recall_at: RecallAt::new(vec![5, RECALL_AT as u32]),
                    // Exhaustive probing makes centroid recall trivially 1.0,
                    // which is exactly what the test wants to pin down.
                    measure_centroid_recall: true,
                    num_threads: 2,
                },
            }
        }
    }

    /// Insert, search, delete, search, insert, search.
    const CHURN_STAGES: &str = "  0:\n    operation: \"insert\"\n    start: 0\n    end: 400\n  \
                                1:\n    operation: \"search\"\n  \
                                2:\n    operation: \"delete\"\n    start: 0\n    end: 200\n  \
                                3:\n    operation: \"search\"\n  \
                                4:\n    operation: \"insert\"\n    start: 400\n    end: 600\n  \
                                5:\n    operation: \"search\"\n";

    #[test]
    fn runbook_replays_inserts_deletes_and_searches() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        fixture.write_gt(1, &(0..400).collect::<Vec<u32>>());
        fixture.write_gt(3, &(200..400).collect::<Vec<u32>>());
        fixture.write_gt(5, &(200..600).collect::<Vec<u32>>());
        let runbook = fixture.write_runbook("runbook.yaml", NUM_POINTS, CHURN_STAGES);

        let stats = build_graph_ivf_runbook::<f32>(&fixture.params(runbook, 16)).unwrap();

        assert_eq!(stats.num_points, NUM_POINTS);
        assert_eq!(stats.total_inserts, 600);
        assert_eq!(stats.total_deletes, 200);

        let kinds: Vec<StageKind> = stats.stages.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                StageKind::Insert,
                StageKind::Search,
                StageKind::Delete,
                StageKind::Search,
                StageKind::Insert,
                StageKind::Search,
            ]
        );

        let live: Vec<usize> = stats.stages.iter().map(|s| s.live_points).collect();
        assert_eq!(
            live,
            vec![400, 400, 200, 200, 400, 400],
            "the index must hold exactly the points the runbook has put in and not taken out"
        );
        assert_eq!(stats.final_points, 400);

        for stage in stats.stages.iter().filter(|s| s.kind == StageKind::Search) {
            assert_eq!(
                stage.search.len(),
                1,
                "one sweep per configured cluster fraction"
            );
            let result = &stage.search[0];
            assert_eq!(result.cluster_fraction, 1.0);
            assert_eq!(
                result.nlist, stage.live_clusters,
                "the same fraction must be resolved from each stage's current cluster count"
            );
            assert_eq!(
                result.recalls.iter().map(|p| p.at).collect::<Vec<_>>(),
                vec![5, RECALL_AT as u32],
                "every configured depth is reported from the one sweep"
            );
            for p in &result.recalls {
                assert_eq!(
                    p.recall, 100.0,
                    "probing every cluster is exhaustive, so stage {} must be exact at k={}",
                    stage.stage, p.at
                );
            }
        }
        for stage in stats.stages.iter().filter(|s| s.kind != StageKind::Search) {
            assert!(stage.search.is_empty());
        }
    }

    #[test]
    fn cluster_size_stats_summarize_the_distribution() {
        // 1..=100, so the nearest-rank percentile of q is exactly q + 1.
        let mut sizes: Vec<usize> = (1..=100).collect();
        let d = ClusterSizeStats::new(&mut sizes, 10, 90);

        assert_eq!((d.min, d.max), (1, 100));
        assert_eq!((d.p10, d.p50, d.p90, d.p99), (11, 51, 91, 100));
        assert_eq!(d.mean, 50.5);
        assert_eq!(d.under, 9, "sizes 1..=9 are below merge_threshold 10");
        assert_eq!(d.over, 10, "sizes 91..=100 are above split_threshold 90");

        // Merging disabled: nothing is underfull, whatever the sizes.
        let mut sizes = vec![1, 1, 1];
        assert_eq!(ClusterSizeStats::new(&mut sizes, 0, 90).under, 0);

        let d = ClusterSizeStats::new(&mut [], 10, 90);
        assert_eq!((d.min, d.max, d.mean), (0, 0, 0.0));
    }

    #[test]
    fn deleted_points_do_not_reach_the_flushed_index() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        fixture.write_gt(1, &(0..400).collect::<Vec<u32>>());
        fixture.write_gt(3, &(200..400).collect::<Vec<u32>>());
        fixture.write_gt(5, &(200..600).collect::<Vec<u32>>());
        let runbook = fixture.write_runbook("runbook.yaml", NUM_POINTS, CHURN_STAGES);

        let stats = build_graph_ivf_runbook::<f32>(&fixture.params(runbook, 16)).unwrap();
        assert_eq!(stats.final_points, 400);

        // The flush writes only live rows and keeps their original corpus ids,
        // so an exhaustive search of the on-disk index can never surface one of
        // the deleted rows.
        let index = diskann_graphivf::GraphIvfIndex::<f32>::load(
            Path::new(&fixture.save_path),
            1,
            diskann_graphivf::CentroidSearch::Graph,
        )
        .expect("the runbook build must leave a loadable index");
        let mut searcher = index.searcher().unwrap();
        let params = SearchParams::new(index.num_clusters());
        let hits = searcher
            .search(&fixture.queries[..DIM], NUM_POINTS, &params)
            .unwrap();
        assert_eq!(
            hits.len(),
            400,
            "an exhaustive search must see exactly the live rows"
        );
        assert!(
            hits.iter().all(|&(id, _)| (200..600).contains(&id)),
            "a deleted row must not appear in the flushed lists"
        );
    }

    #[test]
    fn merges_pair_off_starved_clusters_during_a_runbook() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        fixture.write_gt(1, &(0..400).collect::<Vec<u32>>());
        fixture.write_gt(3, &(200..400).collect::<Vec<u32>>());
        fixture.write_gt(5, &(200..600).collect::<Vec<u32>>());
        let runbook = fixture.write_runbook("runbook.yaml", NUM_POINTS, CHURN_STAGES);

        // Half the split threshold is the largest merge threshold the hysteresis
        // rule allows, so this is the most merge-prone legal configuration.
        let with_merges = build_graph_ivf_runbook::<f32>(&fixture.params(runbook, 32)).unwrap();
        assert!(
            with_merges.total_merges > 0,
            "deleting half the corpus under an aggressive merge threshold must merge clusters"
        );
        assert_eq!(
            with_merges.final_points, 400,
            "a merge re-homes its points rather than dropping them"
        );
    }

    #[test]
    fn replace_stages_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let runbook = fixture.write_runbook(
            "replace.yaml",
            NUM_POINTS,
            "  0:\n    operation: \"insert\"\n    start: 0\n    end: 400\n  \
             1:\n    operation: \"replace\"\n    ids_start: 400\n    ids_end: 500\n    \
             tags_start: 0\n    tags_end: 100\n",
        );

        let error = build_graph_ivf_runbook::<f32>(&fixture.params(runbook, 0)).unwrap_err();
        assert!(
            format!("{error:#}").contains("cannot take on another"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn a_runbook_wider_than_the_corpus_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let runbook = fixture.write_runbook(
            "wide.yaml",
            NUM_POINTS + 1,
            "  0:\n    operation: \"insert\"\n    start: 0\n    end: 400\n",
        );

        let error = build_graph_ivf_runbook::<f32>(&fixture.params(runbook, 0)).unwrap_err();
        assert!(
            format!("{error:#}").contains("corpus holds only"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn merge_csv_sits_beside_the_split_csv() {
        assert_eq!(
            merge_csv_path(Path::new("/tmp/run/telemetry.csv")),
            PathBuf::from("/tmp/run/telemetry_merges.csv")
        );
        assert_eq!(
            merge_csv_path(Path::new("telemetry")),
            PathBuf::from("telemetry_merges.csv")
        );
    }
}
