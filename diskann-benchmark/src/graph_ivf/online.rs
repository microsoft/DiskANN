/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Online (incremental) graph-IVF build.
//!
//! Points are streamed into an [`OnlineClusterer`] in corpus order; clusters split
//! whenever they overflow `split_threshold`, so the final partition emerges from the
//! data rather than from a target cluster count. Clustering runs in `f32`, but the
//! inverted lists are written from the original corpus rows in their stored element
//! type, producing an on-disk index identical in format to a static build.

use std::{fmt, path::Path, time::Instant};

use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_graphivf::{
    GraphParams, OnlineCentroidRouting, OnlineClusterer, OnlineParams, SeedStrategy,
};
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

use crate::{
    graph_ivf::{
        build::{decompress_to_f32, load_stored_corpus, to_graphivf_metric},
        element::GraphIvfElement,
    },
    inputs::graph_ivf::{GraphIvfOnlineBuild, OnlineRoutingConfig},
};

/// Statistics for an online build.
#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfOnlineBuildStats {
    /// Time spent reading the corpus off disk.
    corpus_load: MicroSeconds,
    /// Time spent widening the stored rows to the `f32` clustering copy.
    decompress: MicroSeconds,
    /// Time spent seeding the initial centroids.
    seed: MicroSeconds,
    /// Time spent streaming every point through the clusterer.
    insert: MicroSeconds,
    /// Time spent writing the index to disk.
    flush: MicroSeconds,
    /// Cumulative time routing inserts through the centroid graph.
    routing: MicroSeconds,
    /// Cumulative time in split handling (2-means, graph mutation, reassignment).
    split: MicroSeconds,
    num_points: usize,
    /// Logical vector dimension (narrower than the stored row width for quantized types).
    dim: usize,
    /// Points inserted per batch (`1` streams one at a time).
    batch_size: usize,
    /// Centroid id slots pre-allocated, derived from the corpus size and `capacity_mult`.
    centroid_capacity: usize,
    /// Live centroids after the seed, before any insert.
    seeded_clusters: usize,
    /// Live centroids at the end of the build.
    final_clusters: usize,
    total_splits: u64,
    /// Points that changed cluster, summed over splits (a point moved twice counts twice).
    total_reassigned: u64,
    min_cluster_size: usize,
    mean_cluster_size: f64,
    max_cluster_size: usize,
    /// Sum of squared distances from each point to its centroid.
    residual: f64,
}

impl fmt::Display for GraphIvfOnlineBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Online build: {:.3}s ({} points, dim {})",
            self.insert.as_seconds() + self.flush.as_seconds() + self.seed.as_seconds(),
            self.num_points,
            self.dim
        )?;
        writeln!(f, "  corpus_load:    {:.3}s", self.corpus_load.as_seconds())?;
        writeln!(f, "  decompress:     {:.3}s", self.decompress.as_seconds())?;
        writeln!(f, "  seed:           {:.3}s", self.seed.as_seconds())?;
        writeln!(f, "  insert:         {:.3}s", self.insert.as_seconds())?;
        writeln!(f, "  flush:          {:.3}s", self.flush.as_seconds())?;
        writeln!(f, "  routing:        {:.3}s", self.routing.as_seconds())?;
        writeln!(f, "  split:          {:.3}s", self.split.as_seconds())?;
        writeln!(f, "  batch size:     {}", self.batch_size)?;
        writeln!(f, "  id budget:      {}", self.centroid_capacity)?;
        writeln!(
            f,
            "  clusters:       {} seeded -> {}",
            self.seeded_clusters, self.final_clusters
        )?;
        writeln!(f, "  splits:         {}", self.total_splits)?;
        writeln!(f, "  reassigned:     {}", self.total_reassigned)?;
        writeln!(
            f,
            "  cluster sizes:  min={} mean={:.1} max={}",
            self.min_cluster_size, self.mean_cluster_size, self.max_cluster_size
        )?;
        writeln!(f, "  residual:       {:.3e}", self.residual)
    }
}

/// Centroid id slots to pre-allocate.
///
/// The centroid graph is allocated eagerly and every split permanently retires the
/// parent id, so the budget must cover ids *consumed*, not clusters live. An uncapped
/// build settles near `2N/threshold` live clusters, having consumed roughly twice that;
/// `capacity_mult` adds headroom. A capped build never exceeds `max_clusters` live, so
/// `2 * max_clusters` ids suffice. Taking the max means the budget binds in neither case.
fn centroid_capacity(params: &GraphIvfOnlineBuild, num_points: usize) -> usize {
    ((params.capacity_mult * 2 * num_points) / params.split_threshold.max(1))
        .max(2 * params.warmup_centroids)
        .max(params.warmup_centroids + 1)
        .max(
            params
                .max_clusters
                .map_or(0, |m| 2 * m + params.warmup_centroids),
        )
}

/// Map the online routing config onto the library's routing parameters.
///
/// `reassign_neighbors` resolves the documented `reassign_l` default for a
/// config that was not run through validation.
fn to_online_routing(
    routing: OnlineRoutingConfig,
    reassign_neighbors: usize,
) -> OnlineCentroidRouting {
    match routing {
        OnlineRoutingConfig::Graph {
            assign_l,
            reassign_l,
            graph_degree,
            graph_slack,
            graph_l_build,
            graph_alpha,
        } => OnlineCentroidRouting::Graph {
            graph: GraphParams {
                degree: graph_degree,
                slack: graph_slack,
                l_build: graph_l_build,
                alpha: graph_alpha,
            },
            assign_l,
            reassign_l: reassign_l.unwrap_or_else(|| reassign_neighbors.max(assign_l)),
        },
        OnlineRoutingConfig::Exact => OnlineCentroidRouting::Exact,
    }
}

/// Everything an online run needs before its first insert: the stored corpus to
/// flush from, and a seeded clusterer over the `f32` clustering copy.
///
/// Shared by the corpus-order build and the runbook-driven one so both derive
/// the id budget, the metric, and the warmup seed the same way.
pub(super) struct OnlineSetup<T: GraphIvfElement> {
    pub(super) corpus: Matrix<T>,
    pub(super) clusterer: OnlineClusterer,
    pub(super) dim: usize,
    pub(super) centroid_capacity: usize,
    pub(super) seeded_clusters: usize,
    pub(super) corpus_load: MicroSeconds,
    pub(super) decompress: MicroSeconds,
    pub(super) seed: MicroSeconds,
}

pub(super) fn online_setup<T>(params: &GraphIvfOnlineBuild) -> anyhow::Result<OnlineSetup<T>>
where
    T: GraphIvfElement,
{
    // The stored rows are written to the inverted lists verbatim, so the corpus is kept
    // in its on-disk element type and only the clustering copy is widened.
    let corpus_load_start = Instant::now();
    let (corpus, dim) = load_stored_corpus::<T>(&params.data)?;
    let corpus_load: MicroSeconds = corpus_load_start.elapsed().into();
    let num_points = corpus.nrows();

    let decompress_start = Instant::now();
    let points = decompress_to_f32(&corpus, dim)?;
    let decompress: MicroSeconds = decompress_start.elapsed().into();

    let centroid_capacity = centroid_capacity(params, num_points);
    let online_params = OnlineParams {
        max_clusters: params.max_clusters,
        centroid_capacity,
        split_threshold: params.split_threshold,
        reassign_neighbors: params.reassign_neighbors,
        two_means_iters: params.two_means_iters,
        merge_threshold: params.merge_threshold,
        min_clusters: params.min_clusters,
        routing: to_online_routing(params.routing, params.reassign_neighbors),
        metric: to_graphivf_metric(params.distance)?,
        normalize_centroids: params.normalize,
        num_threads: params.num_threads,
        seed: params.seed,
    };

    let seed_strategy = SeedStrategy::Warmup {
        num_centroids: params.warmup_centroids,
        warmup_points: params.warmup_points,
        iters: params.warmup_iters,
    };

    let seed_start = Instant::now();
    let clusterer = OnlineClusterer::with_seed(points, seed_strategy, online_params)?;
    let seed: MicroSeconds = seed_start.elapsed().into();
    let seeded_clusters = clusterer.num_clusters();

    Ok(OnlineSetup {
        corpus,
        clusterer,
        dim,
        centroid_capacity,
        seeded_clusters,
        corpus_load,
        decompress,
        seed,
    })
}

pub(super) fn build_graph_ivf_online<T>(
    params: &GraphIvfOnlineBuild,
) -> anyhow::Result<GraphIvfOnlineBuildStats>
where
    T: GraphIvfElement,
{
    let OnlineSetup {
        corpus,
        mut clusterer,
        dim,
        centroid_capacity,
        seeded_clusters,
        corpus_load,
        decompress,
        seed: seed_elapsed,
    } = online_setup::<T>(params)?;
    let num_points = corpus.nrows();

    let insert_start = Instant::now();
    let ids: Vec<u32> = (0..num_points as u32).collect();
    for batch in ids.chunks(params.batch_size) {
        clusterer.insert_batch(batch)?;
    }
    let insert: MicroSeconds = insert_start.elapsed().into();

    let flush_start = Instant::now();
    clusterer.flush::<T>(Path::new(&params.save_path), corpus.as_view())?;
    let flush: MicroSeconds = flush_start.elapsed().into();

    let telemetry = clusterer.telemetry();
    if let Some(csv) = &params.telemetry_csv {
        telemetry
            .write_csv(Path::new(csv))
            .map_err(|e| anyhow::anyhow!("failed to write telemetry CSV {csv}: {e}"))?;
    }

    let sizes = clusterer.cluster_sizes();
    let (min_cluster_size, max_cluster_size) = sizes
        .iter()
        .fold((usize::MAX, 0usize), |(lo, hi), &s| (lo.min(s), hi.max(s)));

    Ok(GraphIvfOnlineBuildStats {
        corpus_load,
        decompress,
        seed: seed_elapsed,
        insert,
        flush,
        routing: MicroSeconds::new(telemetry.routing_us),
        split: MicroSeconds::new(telemetry.split_us),
        num_points,
        dim,
        batch_size: params.batch_size,
        centroid_capacity,
        seeded_clusters,
        final_clusters: clusterer.num_clusters(),
        total_splits: telemetry.total_splits,
        total_reassigned: telemetry.total_reassigned,
        min_cluster_size,
        mean_cluster_size: num_points as f64 / sizes.len().max(1) as f64,
        max_cluster_size,
        residual: clusterer.residual(),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{fs, io::Write as _};

    use diskann_benchmark_runner::files::InputFile;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::{
        graph_ivf::search::{search_graph_ivf, GraphIvfSearchResult},
        inputs::graph_ivf::{
            CentroidSearchConfig, ClusterFraction, GraphIvfLoad, GraphIvfSearchPhase, RecallAt,
        },
        utils::SimilarityMeasure,
    };

    const DIM: usize = 8;
    const NUM_POINTS: usize = 2_000;
    const NUM_QUERIES: usize = 20;
    const RECALL_AT: usize = 10;

    /// Write a `npts x ncols` `f32` matrix in `.bin` format: two `u32` header words
    /// followed by row-major data.
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

    /// Exact top-`k` by squared L2, used as groundtruth for the recall check.
    fn brute_force(corpus: &[f32], queries: &[f32], k: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity((queries.len() / DIM) * k);
        for query in queries.chunks(DIM) {
            let mut scored: Vec<(f32, u32)> = corpus
                .chunks(DIM)
                .enumerate()
                .map(|(i, point)| {
                    let d = point
                        .iter()
                        .zip(query)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>();
                    (d, i as u32)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ids.extend(scored.iter().take(k).map(|(_, i)| *i));
        }
        ids
    }

    /// A corpus, query set, and groundtruth written into `dir`.
    struct Fixture {
        data: InputFile,
        queries: InputFile,
        groundtruth: InputFile,
        save_path: String,
    }

    fn fixture(dir: &Path) -> Fixture {
        let mut rng = StdRng::seed_from_u64(7);
        let corpus: Vec<f32> = (0..NUM_POINTS * DIM).map(|_| rng.random::<f32>()).collect();
        let queries: Vec<f32> = (0..NUM_QUERIES * DIM)
            .map(|_| rng.random::<f32>())
            .collect();
        let truth = brute_force(&corpus, &queries, RECALL_AT);

        let data = dir.join("corpus.bin");
        let query_path = dir.join("queries.bin");
        let gt_path = dir.join("gt.bin");
        write_bin(&data, &corpus, NUM_POINTS, DIM);
        write_bin(&query_path, &queries, NUM_QUERIES, DIM);
        write_truthset(&gt_path, &truth, NUM_QUERIES, RECALL_AT);

        Fixture {
            data: InputFile::new(data),
            queries: InputFile::new(query_path),
            groundtruth: InputFile::new(gt_path),
            save_path: dir.join("index").to_string_lossy().into_owned(),
        }
    }

    fn online_params(fixture: &Fixture, split_threshold: usize) -> GraphIvfOnlineBuild {
        GraphIvfOnlineBuild {
            data_type: <f32 as GraphIvfElement>::DATA_TYPE,
            data: fixture.data.clone(),
            distance: SimilarityMeasure::SquaredL2,
            dim: DIM,
            split_threshold,
            batch_size: 1,
            max_clusters: None,
            warmup_centroids: 8,
            warmup_points: 200,
            warmup_iters: 5,
            two_means_iters: 8,
            reassign_neighbors: 4,
            merge_threshold: 0,
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
            save_path: fixture.save_path.clone(),
            telemetry_csv: None,
        }
    }

    #[test]
    fn online_build_splits_until_every_cluster_fits() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let split_threshold = 64;
        let stats =
            build_graph_ivf_online::<f32>(&online_params(&fixture, split_threshold)).unwrap();

        assert_eq!(stats.num_points, NUM_POINTS);
        assert_eq!(stats.dim, DIM);
        assert_eq!(
            stats.seeded_clusters, 8,
            "the warmup seeds `warmup_centroids`"
        );
        assert!(
            stats.total_splits > 0,
            "2000 points into 8 seed clusters at threshold 64 must split"
        );
        assert_eq!(
            stats.final_clusters,
            stats.seeded_clusters + stats.total_splits as usize,
            "each split retires one cluster and creates two, so live count grows by one"
        );
        assert!(
            stats.max_cluster_size <= split_threshold,
            "a cluster over the threshold should have been split: max={}",
            stats.max_cluster_size
        );
        assert!(
            stats.min_cluster_size > 0,
            "no cluster should be left empty"
        );
        assert!(stats.residual > 0.0);
    }

    #[test]
    fn batched_inserts_partition_the_whole_corpus() {
        // The batched path routes, splits, and reassigns on a different schedule
        // than the streaming one, so it lands on a different partition; what must
        // hold either way is that the cluster count grows by one per split and
        // every point ends up in a live cluster.
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let mut params = online_params(&fixture, 64);
        params.batch_size = 256;

        let stats = build_graph_ivf_online::<f32>(&params).unwrap();

        assert_eq!(stats.batch_size, 256);
        assert!(stats.total_splits > 0, "the batch must overflow clusters");
        assert_eq!(
            stats.final_clusters,
            stats.seeded_clusters + stats.total_splits as usize
        );
        assert!(stats.residual > 0.0);
    }

    #[test]
    fn centroid_capacity_covers_ids_consumed_not_clusters_live() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let mut params = online_params(&fixture, 64);

        // Uncapped: 3 * 2 * 2000 / 64 = 187, comfortably above the ~62 live clusters
        // this build settles at plus the ids its splits retire.
        assert_eq!(centroid_capacity(&params, NUM_POINTS), 187);

        // A cap reserves two ids per permitted cluster plus the warmup, which here
        // exceeds the uncapped estimate and so binds instead.
        params.max_clusters = Some(500);
        assert_eq!(centroid_capacity(&params, NUM_POINTS), 1008);

        // A tiny corpus must still leave room for at least one split past the warmup.
        params.max_clusters = None;
        assert!(centroid_capacity(&params, 1) > params.warmup_centroids);
    }

    #[test]
    fn max_clusters_caps_the_partition() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let mut params = online_params(&fixture, 64);
        params.max_clusters = Some(16);

        let stats = build_graph_ivf_online::<f32>(&params).unwrap();
        assert!(
            stats.final_clusters <= 16,
            "final_clusters={} exceeded max_clusters",
            stats.final_clusters
        );
        // With splitting disabled at the cap, clusters necessarily overflow.
        assert!(stats.max_cluster_size > 64);
    }

    #[test]
    fn telemetry_csv_is_written_only_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let csv = dir.path().join("splits.csv");

        let mut params = online_params(&fixture, 64);
        params.telemetry_csv = Some(csv.to_string_lossy().into_owned());
        let stats = build_graph_ivf_online::<f32>(&params).unwrap();

        let contents = fs::read_to_string(&csv).unwrap();
        let mut lines = contents.lines();
        assert!(
            lines.next().unwrap().starts_with("insert_index,cluster,"),
            "the CSV should lead with a header row"
        );
        assert_eq!(
            lines.count(),
            stats.total_splits as usize,
            "one row per split event"
        );

        fs::remove_file(&csv).unwrap();
        params.telemetry_csv = None;
        build_graph_ivf_online::<f32>(&params).unwrap();
        assert!(!csv.exists());
    }

    #[test]
    fn online_index_is_searchable_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let fixture = fixture(dir.path());
        let stats = build_graph_ivf_online::<f32>(&online_params(&fixture, 64)).unwrap();

        let load = GraphIvfLoad {
            data_type: <f32 as GraphIvfElement>::DATA_TYPE,
            load_path: fixture.save_path.clone(),
            centroid_search: CentroidSearchConfig::Graph,
        };
        let search = GraphIvfSearchPhase {
            queries: fixture.queries.clone(),
            groundtruth: fixture.groundtruth.clone(),
            num_threads: 2,
            // Scanning every cluster must recover the exact answer; a narrow probe
            // should be worse, which is what makes the recall number meaningful.
            cluster_fractions: vec![
                ClusterFraction::new(f64::MIN_POSITIVE),
                ClusterFraction::new(1.0),
            ],
            centroid_search_alpha: 1.5,
            // Two depths from one sweep: the shallower must be scored from the
            // prefix of the deeper search's results, not from a second search.
            recall_at: RecallAt::new(vec![5, RECALL_AT as u32]),
            distance: SimilarityMeasure::SquaredL2,
        };

        let results = search_graph_ivf::<f32>(&load, &search).unwrap();
        assert_eq!(results.search_results_per_nlist.len(), 2);
        assert_eq!(results.recall_at, vec![5, RECALL_AT as u32]);
        let narrow = &results.search_results_per_nlist[0];
        let exhaustive = &results.search_results_per_nlist[1];
        assert_eq!(narrow.nlist, 1);
        assert_eq!(exhaustive.nlist, stats.final_clusters);
        assert_eq!(exhaustive.cluster_fraction, 1.0);

        let at = |r: &GraphIvfSearchResult, k: u32| {
            r.recalls
                .iter()
                .find(|p| p.at == k)
                .unwrap_or_else(|| panic!("no recall reported at {k}"))
                .recall
        };

        for k in [5, RECALL_AT as u32] {
            assert!(
                (at(exhaustive, k) - 100.0).abs() < 1e-3,
                "probing all {} clusters should be exact at k={k}, got recall {}",
                stats.final_clusters,
                at(exhaustive, k)
            );
        }
        assert!(
            at(narrow, RECALL_AT as u32) < at(exhaustive, RECALL_AT as u32),
            "probing one cluster should lose recall: {} vs {}",
            at(narrow, RECALL_AT as u32),
            at(exhaustive, RECALL_AT as u32)
        );
    }
}
