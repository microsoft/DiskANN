use super::*;
use crate::{CentroidSearch, GraphIvfIndex, GraphParams, OnlineCentroidRouting, SearchParams};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn mat(data: Vec<f32>, nrows: usize, ncols: usize) -> Matrix<f32> {
    Matrix::try_from(data.into_boxed_slice(), nrows, ncols).unwrap()
}

fn params(target: usize, threshold: usize) -> OnlineParams {
    OnlineParams {
        max_clusters: Some(target),
        centroid_capacity: target.saturating_mul(2).max(1),
        split_threshold: threshold,
        reassign_neighbors: 8,
        routing: OnlineCentroidRouting::Graph {
            graph: GraphParams::default(),
            assign_l: 32,
            reassign_l: 32,
        },
        two_means_iters: 10,
        num_threads: 2,
        ..Default::default()
    }
}

/// [`params`] with merging enabled. `split` is raised to satisfy the
/// hysteresis requirement when the caller asks for a tight merge floor.
fn merge_params(target: usize, split: usize, merge: usize) -> OnlineParams {
    OnlineParams {
        split_threshold: split.max(2 * merge),
        merge_threshold: merge,
        ..params(target, split)
    }
}

/// Two well-separated Gaussian-ish blobs in 2D.
fn two_blobs(per: usize, seed: u64) -> (Matrix<f32>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v = Vec::new();
    for _ in 0..per {
        v.push(rng.random_range(-1.0..1.0));
        v.push(rng.random_range(-1.0..1.0));
    }
    for _ in 0..per {
        v.push(20.0 + rng.random_range(-1.0..1.0));
        v.push(20.0 + rng.random_range(-1.0..1.0));
    }
    let n = per * 2;
    (mat(v, n, 2), n)
}

/// Brute-force squared-L2.
fn sqd(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum()
}

/// Optimal residual for a fixed centroid set: every point to its globally
/// nearest centroid. The online (local) assignment can only be >= this.
fn optimal_residual(points: &Matrix<f32>, centroids: &[Box<[f32]>]) -> f64 {
    let mut sum = 0.0;
    for p in 0..points.nrows() {
        let row = points.row(p);
        let best = centroids
            .iter()
            .map(|c| sqd(row, c))
            .fold(f64::INFINITY, f64::min);
        sum += best;
    }
    sum
}

fn live_centroids(c: &OnlineClusterer) -> Vec<Box<[f32]>> {
    c.centroids
        .iter_live()
        .map(|(_, v)| v.to_vec().into_boxed_slice())
        .collect()
}

// ----- centroid-graph mutable ops -----

#[test]
fn mutable_graph_insert_delete_search() {
    // Four centroids at distinct corners; capacity leaves room to insert.
    let cents = mat(vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0], 4, 2);
    let graph =
        centroids::build_mutable(cents, &GraphParams::default(), 2, 8, VectorMetric::L2).unwrap();
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let mut ids = [0u32; 1];
    let mut dist = [0.0f32; 1];

    // Query near centroid 3 (10,10) -> returns 3.
    centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
    assert_eq!(ids[0], 3);

    // Delete centroid 3; the same query now returns a different live one.
    centroids::delete_centroid(&graph, &rt, 3).unwrap();
    centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
    assert_ne!(ids[0], 3);

    // Insert a new centroid (id 4) right at the query; it wins.
    centroids::insert_centroid(&graph, &rt, 4, &[9.5, 9.5]).unwrap();
    centroids::search_mut(&graph, &rt, &[9.5, 9.5], 8, &mut ids, &mut dist).unwrap();
    assert_eq!(ids[0], 4);
}

// ----- clusterer invariants -----

/// Every inserted point is accounted for exactly once in a live cluster.
fn assert_invariants(c: &OnlineClusterer, inserted: usize) {
    let live: Vec<u32> = (0..inserted as u32).collect();
    assert_live_invariants(c, &live);
}

/// Exactly the points in `live` are held, each once, by a live cluster, and
/// no other point is. This is the delete-aware form of
/// [`assert_invariants`], which is the special case `live == 0..inserted`.
fn assert_live_invariants(c: &OnlineClusterer, live: &[u32]) {
    // live_count matches the centroid table.
    let live_clusters = c.centroids.iter_live().count();
    assert_eq!(live_clusters, c.centroids.live_count());
    if let Some(k) = c.params.max_clusters {
        assert!(c.centroids.live_count() <= k);
    }
    assert!(c.centroids.live_count() <= c.centroids.capacity());

    // Sum of live list lengths == live count; retired ids hold nothing.
    let mut total = 0usize;
    for cid in 0..c.centroids.capacity() as u32 {
        if c.centroids.is_live(cid) {
            total += c.partition.list_len(cid);
        } else {
            assert!(
                c.partition.members(cid).is_empty(),
                "retired centroid has points"
            );
        }
    }
    assert_eq!(total, live.len());

    // Every live point sits on a live centroid, and every other point on
    // none.
    for pid in 0..c.points.nrows() as u32 {
        let a = c.partition.assignment(pid);
        if live.contains(&pid) {
            assert_ne!(a, UNASSIGNED, "live point {pid} is unassigned");
            assert!(
                c.centroids.is_live(a),
                "point {pid} sits on a retired centroid"
            );
        } else {
            assert_eq!(a, UNASSIGNED, "absent point {pid} is still assigned");
        }
    }
}

#[test]
fn no_split_matches_nearest_centroid() {
    // High threshold => no splits; pure online assignment with fixed
    // centroids. Residual must equal the optimal for those centroids.
    let (points, n) = two_blobs(40, 1);
    let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
    let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 10_000)).unwrap();
    for pid in 0..n as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    assert_invariants(&c, n);
    assert_eq!(c.num_clusters(), 2);

    let opt = optimal_residual(&points, &live_centroids(&c));
    // Graph routing is approximate, but for two far-apart blobs it is exact.
    assert!(
        (c.residual() - opt).abs() < 1e-3,
        "res={} opt={}",
        c.residual(),
        opt
    );
}

#[test]
fn split_creates_cluster_and_tightens() {
    // Start with ONE centroid; a low threshold forces a split of the single
    // overfull cluster into the two blobs. Points are streamed in shuffled
    // order so both blobs are represented by the time the split fires.
    let (points, n) = two_blobs(60, 2);
    let initial = mat(vec![10.0, 10.0], 1, 2);
    let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 30)).unwrap();

    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = StdRng::seed_from_u64(99);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.random_range(0..=i));
    }
    for &pid in &order {
        c.insert_batch(&[pid]).unwrap();
    }
    assert_invariants(&c, n);
    assert_eq!(
        c.num_clusters(),
        2,
        "the overfull cluster should have split"
    );

    // With two centroids at the blob centers the residual is far below the
    // single-centroid residual, and never below the optimal-for-2.
    let opt2 = optimal_residual(&points, &live_centroids(&c));
    assert!(c.residual() >= opt2 - 1e-3);
    // Sanity: two tight blobs => small residual per point.
    assert!(
        c.residual() / (n as f64) < 5.0,
        "residual too large: {}",
        c.residual()
    );
}

#[test]
fn many_splits_preserve_invariants_and_bound_residual() {
    // Random data, several initial centroids, many splits.
    let mut rng = StdRng::seed_from_u64(7);
    let (nn, dim) = (600usize, 8usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);

    // 4 initial centroids drawn from the data.
    let mut ib = vec![0.0f32; 4 * dim];
    for i in 0..4 {
        let src = rng.random_range(0..nn);
        ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(src));
    }
    let initial = mat(ib, 4, dim);

    let mut c = OnlineClusterer::new(points.clone(), initial, params(16, 40)).unwrap();
    for pid in 0..nn as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    assert_invariants(&c, nn);
    assert!(c.num_clusters() > 4, "expected some splits to occur");
    assert!(c.num_clusters() <= 16);

    // Online (local) residual is never below the optimal assignment for the
    // same centroid set.
    let opt = optimal_residual(&points, &live_centroids(&c));
    assert!(
        c.residual() >= opt - 1e-3,
        "res={} opt={}",
        c.residual(),
        opt
    );
}

#[test]
fn centroid_recall_scores_selection_against_an_exact_scan() {
    // Enough splits that the centroid graph carries retired parents and its
    // walk is genuinely approximate, which is the case worth measuring.
    let mut rng = StdRng::seed_from_u64(11);
    let (nn, dim) = (600usize, 8usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);
    let mut ib = vec![0.0f32; 4 * dim];
    for i in 0..4 {
        ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(rng.random_range(0..nn)));
    }

    let mut c = OnlineClusterer::new(points.clone(), mat(ib, 4, dim), params(64, 40)).unwrap();
    for pid in 0..nn as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    let num_clusters = c.num_clusters();
    assert!(num_clusters > 4, "expected some splits to occur");

    let query = points.row(0).to_vec();
    let mut searcher = c.searcher().unwrap();

    // Probing every cluster puts the whole graph inside the beam, so selection
    // is exact — which also pins that every live centroid stays reachable
    // despite the retired parents left behind by splitting.
    let all = searcher
        .centroid_recall(&query, &SearchParams::new(num_clusters))
        .unwrap();
    assert_eq!(all.requested, num_clusters);
    assert_eq!(all.retrieved, num_clusters);
    assert_eq!(all.matched, num_clusters);

    // A narrow probe is scored against the same exact reference, so it can only
    // lose ground, never invent it.
    let few = searcher
        .centroid_recall(&query, &SearchParams::new(4))
        .unwrap();
    assert_eq!(few.requested, 4);
    assert!(few.retrieved <= 4);
    assert!(few.matched <= few.retrieved);
    assert!((0.0..=1.0).contains(&few.recall()));
}

#[test]
fn exact_centroid_search_selects_perfectly_at_every_width() {
    // Same corpus as the graph-mode test above, but routed by exact scan. The
    // interesting part is that splits, merges and deletes all keep mutating the
    // centroid set underneath, so this pins that the dense mirror stays in step.
    let mut rng = StdRng::seed_from_u64(11);
    let (nn, dim) = (600usize, 8usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);
    let mut ib = vec![0.0f32; 4 * dim];
    for i in 0..4 {
        ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(rng.random_range(0..nn)));
    }

    let p = OnlineParams {
        routing: OnlineCentroidRouting::Exact,
        ..merge_params(64, 40, 8)
    };
    let mut c = OnlineClusterer::new(points.clone(), mat(ib, 4, dim), p).unwrap();
    for pid in 0..nn as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    // Thin the corpus out so merges retire centroids as well.
    c.delete_batch(&(0..nn as u32).filter(|p| p % 3 != 0).collect::<Vec<_>>())
        .unwrap();
    let num_clusters = c.num_clusters();
    assert!(num_clusters > 4, "expected some splits to occur");

    // No centroid graph exists in this mode, so there is nothing to census.
    assert!(c.centroid_adjacency_census().unwrap().is_none());

    let mut searcher = c.searcher().unwrap();
    for width in [1, 4, num_clusters / 2, num_clusters] {
        let r = searcher
            .centroid_recall(points.row(0), &SearchParams::new(width))
            .unwrap();
        assert_eq!(r.requested, width);
        assert_eq!(r.retrieved, width, "width={width}");
        assert_eq!(r.matched, width, "width={width}");
    }
}

#[test]
fn uncapped_splits_until_threshold_equilibrium() {
    // `max_clusters: None` removes the live-cluster ceiling: splitting is
    // driven purely by the threshold and continues for every point, so the
    // count grows well past any small fixed target and mean cluster size
    // settles near the split threshold.
    let mut rng = StdRng::seed_from_u64(11);
    let (nn, dim) = (800usize, 6usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);
    let initial = mat(points.row(0).to_vec(), 1, dim);

    let mut p = params(1, 20);
    p.max_clusters = None; // uncapped: threshold-driven only
    p.centroid_capacity = 4 * nn; // generous id budget, never binds

    let mut c = OnlineClusterer::new(points, initial, p).unwrap();
    for pid in 0..nn as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    assert_invariants(&c, nn);

    // Far more than the single seed centroid; roughly `~ 2 * nn / threshold`.
    assert!(c.num_clusters() > 10, "got {}", c.num_clusters());
    let mean = nn as f64 / c.num_clusters() as f64;
    assert!(mean <= 21.0, "mean cluster size {mean} exceeds threshold");
}

#[test]
fn batched_inserts_preserve_invariants_and_split() {
    // The batched path defers splitting to the end of a batch, so several
    // clusters overflow at once and are re-clustered together. The partition
    // it lands on differs from the streaming path's, but the structural
    // invariants and the +1-live-cluster-per-split accounting are the same.
    let mut rng = StdRng::seed_from_u64(13);
    let (nn, dim) = (900usize, 8usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);

    let mut ib = vec![0.0f32; 4 * dim];
    for i in 0..4 {
        ib[i * dim..(i + 1) * dim].copy_from_slice(points.row(rng.random_range(0..nn)));
    }
    let initial = mat(ib, 4, dim);

    let mut p = params(64, 40);
    p.centroid_capacity = 4 * nn;

    let mut c = OnlineClusterer::new(points.clone(), initial, p).unwrap();
    let ids: Vec<u32> = (0..nn as u32).collect();
    for batch in ids.chunks(128) {
        c.insert_batch(batch).unwrap();
    }

    assert_invariants(&c, nn);
    assert!(c.num_clusters() > 4, "batches should overflow clusters");
    assert_eq!(
        c.num_clusters(),
        4 + c.telemetry().total_splits as usize,
        "every split retires one parent and allocates two children"
    );
    assert_eq!(c.telemetry().total_inserts, nn as u64);

    // A batch's splits are re-clustered jointly, so several events share one
    // timestamp, but the timeline is still ordered.
    let mut prev = 0u64;
    for e in &c.telemetry().splits {
        assert!(e.insert_index >= prev);
        prev = e.insert_index;
    }

    // Local assignment can only be worse than the optimal one for the
    // centroid set it produced.
    let opt = optimal_residual(&points, &live_centroids(&c));
    assert!(c.residual() >= opt - 1e-3);
}

#[test]
fn batched_inserts_respect_max_clusters() {
    // Admission control has to hold even when a single batch overflows more
    // clusters than the cap has room for.
    let mut rng = StdRng::seed_from_u64(17);
    let (nn, dim) = (800usize, 4usize);
    let mut v = vec![0.0f32; nn * dim];
    for x in v.iter_mut() {
        *x = rng.random_range(-1.0..1.0);
    }
    let points = mat(v, nn, dim);
    let initial = mat(points.row(0).to_vec(), 1, dim);

    let mut p = params(6, 10);
    p.centroid_capacity = 4 * nn;

    let mut c = OnlineClusterer::new(points, initial, p).unwrap();
    let ids: Vec<u32> = (0..nn as u32).collect();
    for batch in ids.chunks(200) {
        c.insert_batch(batch).unwrap();
    }
    assert_invariants(&c, nn);
    assert!(c.num_clusters() <= 6, "got {}", c.num_clusters());
}

#[test]
fn telemetry_records_splits_and_reassignments() {
    // A split-heavy run records one telemetry event per split, with a
    // monotonic insert-index timeline and sane counters.
    let (points, n) = two_blobs(60, 21);
    let initial = mat(vec![10.0, 10.0], 1, 2);
    let mut c = OnlineClusterer::new(points, initial, params(8, 25)).unwrap();

    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = StdRng::seed_from_u64(5);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.random_range(0..=i));
    }
    for &pid in &order {
        c.insert_batch(&[pid]).unwrap();
    }

    let t = c.telemetry();
    assert_eq!(t.total_inserts, n as u64);
    assert!(t.total_splits >= 1, "expected at least one split");
    assert_eq!(t.splits.len() as u64, t.total_splits);

    // Per-split records are consistent and ordered in build time.
    let mut prev = 0u64;
    let mut reassigned_sum = 0u64;
    for e in &t.splits {
        assert!(
            e.insert_index >= prev,
            "insert_index must be non-decreasing"
        );
        assert!(e.insert_index >= 1 && e.insert_index <= n as u64);
        prev = e.insert_index;
        assert!(e.cluster_size >= 2);
        assert!(e.num_reassigned >= e.cluster_size); // all of C always moves
        reassigned_sum += e.num_reassigned as u64;
    }
    assert_eq!(reassigned_sum, t.total_reassigned);
    assert_eq!(
        t.splits.last().unwrap().live_after,
        c.num_clusters(),
        "last split's live_after should match the final cluster count"
    );

    // Cluster sizes cover every live cluster and sum to the corpus.
    let sizes = c.cluster_sizes();
    assert_eq!(sizes.len(), c.num_clusters());
    assert_eq!(sizes.iter().sum::<usize>(), n);

    // CSV export writes a header plus one row per split.
    let dir = std::env::temp_dir().join(format!("graphivf_tel_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = dir.join("splits.csv");
    t.write_csv(&csv).unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].starts_with("insert_index,cluster,cluster_size"));
    assert_eq!(lines.len(), 1 + t.splits.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn flush_roundtrips_through_load_and_search() {
    let (points, n) = two_blobs(50, 3);
    let initial = mat(vec![10.0, 10.0], 1, 2);
    let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 25)).unwrap();
    for pid in 0..n as u32 {
        c.insert_batch(&[pid]).unwrap();
    }

    let dir = std::env::temp_dir().join(format!("graphivf_online_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("idx");
    c.flush(&prefix, c.points.as_view()).unwrap();

    let index = GraphIvfIndex::<f32>::load(&prefix, 2, CentroidSearch::Graph).unwrap();
    assert_eq!(index.num_clusters(), 2);
    let mut searcher = index.searcher().unwrap();

    // A query in blob 0 should retrieve blob-0 points (small distances).
    let sp = SearchParams::new(2);
    let results = searcher.search(&[0.0f32, 0.0], 5, &sp).unwrap();
    assert!(!results.is_empty());
    // Nearest neighbor is within blob 0 (distance well under the blob gap).
    assert!(
        results[0].1 < 25.0,
        "nn distance {} too large",
        results[0].1
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rejects_bad_params() {
    let (points, _) = two_blobs(10, 4);
    // centroid_capacity < initial (params maps target -> capacity = 2*target,
    // so target 1 gives capacity 2 < the 3 initial centroids).
    let initial = mat(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], 3, 2);
    assert!(OnlineClusterer::new(points.clone(), initial, params(1, 10)).is_err());
    // threshold < 2
    let initial = mat(vec![0.0, 0.0], 1, 2);
    assert!(OnlineClusterer::new(points, initial, params(4, 1)).is_err());
}

// ----- deletes and merges -----

/// Four tight groups of `per` points at the corners of a 30x30 square,
/// returned with the corners themselves as the initial centroid set. Every
/// point is unambiguously nearest its own corner, so routing is exact and
/// the starting partition is known: group `i` occupies cluster `i`.
fn four_groups(per: usize) -> (Matrix<f32>, Matrix<f32>) {
    const CORNERS: [[f32; 2]; 4] = [[0.0, 0.0], [30.0, 0.0], [0.0, 30.0], [30.0, 30.0]];
    let mut rng = StdRng::seed_from_u64(4242);
    let mut v = Vec::with_capacity(4 * per * 2);
    for c in CORNERS {
        for _ in 0..per {
            v.push(c[0] + rng.random_range(-0.5..0.5));
            v.push(c[1] + rng.random_range(-0.5..0.5));
        }
    }
    let points = mat(v, 4 * per, 2);
    let initial = mat(CORNERS.iter().flatten().copied().collect(), 4, 2);
    (points, initial)
}

#[test]
fn delete_removes_points_without_merging() {
    // Merging disabled: deleting only shrinks lists, never the cluster set.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
    assert_eq!(c.cluster_sizes(), vec![5, 5, 5, 5]);

    // Delete across two clusters at once, out of order, to exercise the
    // group-by-cluster path.
    c.delete_batch(&[12, 0, 3, 11]).unwrap();

    let live: Vec<u32> = (0..20u32).filter(|p| ![0, 3, 11, 12].contains(p)).collect();
    assert_live_invariants(&c, &live);
    assert_eq!(c.num_clusters(), 4, "merging is off; no cluster dissolves");
    assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 16);
    assert_eq!(c.telemetry().total_deletes, 4);
    assert_eq!(c.telemetry().total_merges, 0);
}

#[test]
fn delete_batch_is_idempotent_within_a_batch() {
    // A pid repeated inside one batch is deduplicated rather than being
    // counted twice or corrupting the list.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    c.delete_batch(&[7, 7, 7]).unwrap();
    let live: Vec<u32> = (0..20u32).filter(|&p| p != 7).collect();
    assert_live_invariants(&c, &live);
    assert_eq!(c.telemetry().total_deletes, 1);
}

#[test]
fn delete_rejects_absent_and_out_of_range_points() {
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..10u32).collect::<Vec<_>>()).unwrap();

    // Never inserted.
    assert!(c.delete_batch(&[15]).is_err());
    // Past the end of the corpus.
    assert!(c.delete_batch(&[100]).is_err());
    // Already deleted.
    c.delete_batch(&[2]).unwrap();
    assert!(c.delete_batch(&[2]).is_err());

    // A rejected batch leaves the index exactly as it was.
    let live: Vec<u32> = (0..10u32).filter(|&p| p != 2).collect();
    assert_live_invariants(&c, &live);
}

#[test]
fn insert_rejects_points_already_present() {
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&[0, 1, 2]).unwrap();

    assert!(c.insert_batch(&[1]).is_err(), "re-insert must be rejected");
    assert!(
        c.insert_batch(&[5, 5]).is_err(),
        "a pid twice in one batch must be rejected"
    );
    assert_live_invariants(&c, &[0, 1, 2]);
}

#[test]
fn poisoned_clusterer_rejects_mutation_search_and_flush() {
    let (points, _) = two_blobs(5, 91);
    let stored = points.clone();
    let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
    let mut clusterer = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    clusterer.poisoned = Some("injected test failure".to_owned());

    assert!(clusterer.is_poisoned());
    assert!(matches!(
        clusterer.insert_batch(&[0]),
        Err(GraphIvfError::Poisoned(_))
    ));
    assert!(matches!(
        clusterer.delete_batch(&[0]),
        Err(GraphIvfError::Poisoned(_))
    ));
    assert!(matches!(
        clusterer.searcher(),
        Err(GraphIvfError::Poisoned(_))
    ));

    let prefix = std::env::temp_dir().join("graphivf_poisoned_must_not_flush");
    assert!(matches!(
        clusterer.flush(&prefix, stored.as_view()),
        Err(GraphIvfError::Poisoned(_))
    ));
}

#[test]
fn deleted_point_can_be_reinserted() {
    // Delete/insert of the same pid is the churn pattern a streaming
    // runbook produces; the point must come back on a live cluster.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    c.delete_batch(&[4, 9]).unwrap();
    c.insert_batch(&[9, 4]).unwrap();

    assert_live_invariants(&c, &(0..20u32).collect::<Vec<_>>());
    // Points 4 and 9 belong to groups 0 and 1, and route back there.
    assert_eq!(c.partition.assignment(4), c.partition.assignment(0));
    assert_eq!(c.partition.assignment(9), c.partition.assignment(5));
    assert_eq!(c.telemetry().total_inserts, 22);
    assert_eq!(c.telemetry().total_deletes, 2);
}

#[test]
fn underflow_retires_the_cluster_and_scatters_it_onto_survivors() {
    let (points, initial) = four_groups(5);
    let budget_before = {
        let c = OnlineClusterer::new(points.clone(), initial.clone(), merge_params(8, 10_000, 3))
            .unwrap();
        c.centroids.alloc_budget()
    };

    let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
    assert_eq!(c.num_clusters(), 4);

    // Group 0 drops to 2 members, below the merge threshold of 3.
    c.delete_batch(&[0, 1, 2]).unwrap();

    let live: Vec<u32> = (3..20u32).collect();
    assert_live_invariants(&c, &live);
    assert_eq!(
        c.num_clusters(),
        3,
        "the cell is gone and nothing replaces it: net -1"
    );
    assert_eq!(
        c.centroids.alloc_budget(),
        budget_before,
        "retiring a cluster fits nothing, so it consumes no id"
    );

    // The starved cluster's points land on a live cluster, not dropped.
    let a3 = c.partition.assignment(3);
    assert_ne!(a3, UNASSIGNED);
    assert!(c.centroids.is_live(a3));
    assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 17);

    let t = c.telemetry();
    assert_eq!(t.total_deletes, 3);
    assert_eq!(t.total_merges, 1);
    assert_eq!(t.merges.len(), 1);
    let e = t.merges[0];
    assert_eq!(e.victim_size, 2, "what was left of the starved cluster");
    assert_eq!(e.live_after, 3);
    assert_eq!(e.op_index, 23, "20 inserts followed by 3 deletes");
    assert_eq!(
        e.num_reassigned, 2,
        "only the victim's own members are re-placed"
    );
}

#[test]
fn retiring_adjacent_clusters_never_lands_a_point_on_a_retired_cell() {
    // Groups 0 and 1 are each other's nearest neighbors and both starve in
    // the same batch. Retiring the whole batch before placing any point is
    // what stops one victim's members landing on the other.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    c.delete_batch(&[0, 1, 2, 5, 6, 7]).unwrap();

    let live: Vec<u32> = (0..20u32)
        .filter(|p| ![0, 1, 2, 5, 6, 7].contains(p))
        .collect();
    assert_live_invariants(&c, &live);
    assert_eq!(
        c.telemetry().total_merges,
        2,
        "both victims go in one batch"
    );
    assert_eq!(c.num_clusters(), 2);
    assert_eq!(c.cluster_sizes().iter().sum::<usize>(), 14);
}

#[test]
fn merges_run_with_an_exhausted_id_budget() {
    // `centroid_capacity == initial` leaves no allocations at all. Only
    // splits draw on that budget, so an underfull cluster still retires.
    let (points, initial) = four_groups(5);
    let p = OnlineParams {
        centroid_capacity: 4,
        ..merge_params(8, 10_000, 3)
    };
    let mut c = OnlineClusterer::new(points, initial, p).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
    assert_eq!(c.centroids.alloc_budget(), 0);

    c.delete_batch(&[0, 1, 2]).unwrap();

    assert_live_invariants(&c, &(3..20u32).collect::<Vec<_>>());
    assert_eq!(c.telemetry().total_merges, 1);
    assert_eq!(c.num_clusters(), 3);
}

#[test]
fn merges_stop_at_the_min_clusters_floor() {
    // The same starvation, but the floor forbids giving up any cluster.
    let (points, initial) = four_groups(5);
    let p = OnlineParams {
        min_clusters: 4,
        ..merge_params(8, 10_000, 3)
    };
    let mut c = OnlineClusterer::new(points, initial, p).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    c.delete_batch(&[0, 1, 2]).unwrap();

    assert_live_invariants(&c, &(3..20u32).collect::<Vec<_>>());
    assert_eq!(c.num_clusters(), 4, "the floor holds the cluster open");
    assert_eq!(c.telemetry().total_merges, 0);
    assert_eq!(c.cluster_sizes(), vec![2, 5, 5, 5]);
}

#[test]
fn rejects_merge_threshold_without_hysteresis() {
    // A merge floor at more than half the split ceiling means a freshly
    // split cluster is immediately a merge candidate.
    let (points, initial) = four_groups(5);
    let mut p = params(8, 30);
    p.merge_threshold = 20;
    assert!(OnlineClusterer::new(points.clone(), initial.clone(), p).is_err());

    // Exactly half is the tightest setting that is accepted.
    p.merge_threshold = 15;
    assert!(OnlineClusterer::new(points, initial, p).is_ok());
}

#[test]
fn merge_telemetry_csv_has_one_row_per_merge() {
    // Groups 0 and 3 sit on opposite corners, so the two retirements are
    // independent and each scatters onto a different survivor.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, merge_params(8, 10_000, 3)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
    c.delete_batch(&[0, 1, 2, 15, 16, 17]).unwrap();

    let t = c.telemetry();
    assert_eq!(t.total_merges, 2);

    let dir = std::env::temp_dir().join(format!("graphivf_merge_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = dir.join("merges.csv");
    t.write_merges_csv(&csv).unwrap();
    let text = std::fs::read_to_string(&csv).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(
        lines[0],
        "op_index,victim,victim_size,num_neighbors,num_reassigned,\
         live_after,search_us,reassign_us,total_us"
    );
    assert_eq!(lines.len(), 1 + t.merges.len());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn flush_after_deletes_drops_them_and_keeps_original_ids() {
    // The flushed index holds exactly the live points, still labelled by
    // their corpus row, so groundtruth computed over the corpus lines up.
    let (points, n) = two_blobs(50, 3);
    let initial = mat(vec![10.0, 10.0], 1, 2);
    let mut c = OnlineClusterer::new(points.clone(), initial, params(2, 25)).unwrap();
    c.insert_batch(&(0..n as u32).collect::<Vec<_>>()).unwrap();

    let removed: Vec<u32> = (0..10u32).collect();
    c.delete_batch(&removed).unwrap();

    let dir = std::env::temp_dir().join(format!("graphivf_del_flush_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefix = dir.join("idx");
    c.flush(&prefix, c.points.as_view()).unwrap();

    let index = GraphIvfIndex::<f32>::load(&prefix, 2, CentroidSearch::Graph).unwrap();
    let mut searcher = index.searcher().unwrap();
    let sp = SearchParams::new(index.num_clusters());
    // Scanning every list returns the whole live corpus and nothing else.
    let results = searcher.search(&[0.0f32, 0.0], n, &sp).unwrap();
    assert_eq!(results.len(), n - removed.len());
    for (id, _) in &results {
        assert!(!removed.contains(id), "deleted point {id} was returned");
        assert!((*id as usize) < n);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ----- online search -----

/// Exact top-`k` over `live`, as ids in ascending-distance order.
fn brute_force(points: &Matrix<f32>, live: &[u32], q: &[f32], k: usize) -> Vec<u32> {
    let mut v: Vec<(u32, f64)> = live
        .iter()
        .map(|&p| (p, sqd(points.row(p as usize), q)))
        .collect();
    v.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(p, _)| p).collect()
}

#[test]
fn online_search_probing_everything_is_exact() {
    // Probing every list turns the search into a full scan, so it must
    // reproduce brute force exactly.
    let (points, n) = two_blobs(50, 31);
    let initial = mat(vec![10.0, 10.0], 1, 2);
    let mut c = OnlineClusterer::new(points.clone(), initial, params(4, 25)).unwrap();
    c.insert_batch(&(0..n as u32).collect::<Vec<_>>()).unwrap();

    let sp = SearchParams::new(c.num_clusters());
    let live: Vec<u32> = (0..n as u32).collect();
    let mut s = c.searcher().unwrap();
    for q in [[0.0f32, 0.0], [20.0, 20.0], [10.0, 10.0]] {
        let got: Vec<u32> = s
            .search(&q, 10, &sp)
            .unwrap()
            .into_iter()
            .map(|r| r.0)
            .collect();
        assert_eq!(got, brute_force(&points, &live, &q, 10));
    }
}

#[test]
fn online_search_after_deletes_and_merges_sees_only_live_points() {
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points.clone(), initial, merge_params(8, 10_000, 3)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();
    // Starves group 0 into a merge, so the query's own cluster is gone.
    c.delete_batch(&[0, 1, 2]).unwrap();

    let live: Vec<u32> = (3..20u32).collect();
    let sp = SearchParams::new(c.num_clusters());
    let mut s = c.searcher().unwrap();
    let got: Vec<u32> = s
        .search(&[0.0f32, 0.0], 20, &sp)
        .unwrap()
        .into_iter()
        .map(|r| r.0)
        .collect();

    assert_eq!(got.len(), live.len(), "deleted points must not be returned");
    assert_eq!(got, brute_force(&points, &live, &[0.0, 0.0], 20));
}

#[test]
fn points_scanned_counts_every_probed_list_member() {
    // The scan count is what recall should be read against, so it has to be
    // the exact list volume the probe touched, not an estimate from nlist.
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    let mut s = c.searcher().unwrap();
    assert_eq!(s.points_scanned(), 0);

    let sp = SearchParams::new(1);
    s.search(&[0.0f32, 0.0], 3, &sp).unwrap();
    assert_eq!(s.points_scanned(), 5, "one group's list, not the top-3");

    // Accumulates across queries, and grows with the probe width.
    let sp = SearchParams::new(4);
    s.search(&[0.0f32, 0.0], 3, &sp).unwrap();
    assert_eq!(
        s.points_scanned(),
        25,
        "5 from the first query, then all 20"
    );
}

#[test]
fn search_into_reuses_output_and_reports_per_query_scan() {
    let (points, _) = two_blobs(10, 31);
    let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
    let mut clusterer = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    clusterer
        .insert_batch(&(0..20u32).collect::<Vec<_>>())
        .unwrap();

    let search_params = SearchParams::new(2);
    let mut searcher = clusterer.searcher().unwrap();
    let mut output = Vec::with_capacity(20);
    let capacity = output.capacity();

    let stats = searcher
        .search_into(&[0.0, 0.0], 5, &search_params, &mut output)
        .unwrap();
    assert_eq!(output.len(), 5);
    assert_eq!(stats.points_scanned, 20);
    assert_eq!(searcher.points_scanned(), 20);
    assert_eq!(output.capacity(), capacity);

    let stats = searcher
        .search_into(&[20.0, 20.0], 5, &search_params, &mut output)
        .unwrap();
    assert_eq!(stats.points_scanned, 20);
    assert_eq!(searcher.points_scanned(), 40);
    assert_eq!(output.capacity(), capacity);
}

#[test]
fn online_search_returns_fewer_than_k_when_lists_are_short() {
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    let sp = SearchParams::new(1);
    let mut s = c.searcher().unwrap();
    let got = s.search(&[0.0f32, 0.0], 50, &sp).unwrap();
    assert_eq!(got.len(), 5, "one list holds one group");
    assert!(got.windows(2).all(|w| w[0].1 <= w[1].1));
}

#[test]
fn online_search_rejects_bad_queries() {
    let (points, initial) = four_groups(5);
    let mut c = OnlineClusterer::new(points, initial, params(8, 10_000)).unwrap();
    c.insert_batch(&(0..20u32).collect::<Vec<_>>()).unwrap();

    let sp = SearchParams::new(2);
    let mut s = c.searcher().unwrap();
    assert!(s.search(&[0.0, 0.0], 0, &sp).is_err(), "k must be non-zero");
    assert!(s.search(&[0.0, 0.0, 0.0], 5, &sp).is_err(), "wrong dim");
    let too_many = SearchParams::new(99);
    assert!(s.search(&[0.0, 0.0], 5, &too_many).is_err());
}

// ----- seeding -----

#[test]
fn warmup_seed_bootstraps_centroids() {
    // Warmup k-means over a prefix of two well-separated blobs recovers a
    // sensible starting partition, and streaming continues from it.
    let (points, n) = two_blobs(80, 11);
    let seed = SeedStrategy::Warmup {
        num_centroids: 4,
        warmup_points: 60,
        iters: 15,
    };
    let mut c = OnlineClusterer::with_seed(points.clone(), seed, params(8, 10_000)).unwrap();
    // The clusterer starts with exactly the requested centroids.
    assert_eq!(c.num_clusters(), 4);

    for pid in 0..n as u32 {
        c.insert_batch(&[pid]).unwrap();
    }
    assert_invariants(&c, n);

    // Warmed-up centroids sit inside the blobs, so the residual matches the
    // optimal assignment for that centroid set (no split here).
    let opt = optimal_residual(&points, &live_centroids(&c));
    assert!(
        c.residual() >= opt - 1e-3,
        "res={} opt={}",
        c.residual(),
        opt
    );
}

#[test]
fn warmup_zero_iters_uses_sampled_points() {
    // With iters == 0 the sampled prefix points are used verbatim as
    // centroids (no refinement), and every centroid is a real corpus point.
    let (points, _) = two_blobs(40, 12);
    let seed = SeedStrategy::Warmup {
        num_centroids: 3,
        warmup_points: 20,
        iters: 0,
    };
    let c = OnlineClusterer::with_seed(points.clone(), seed, params(8, 10_000)).unwrap();
    assert_eq!(c.num_clusters(), 3);
    for (_, cv) in c.centroids.iter_live() {
        let is_corpus_point = (0..points.nrows()).any(|r| points.row(r) == cv);
        assert!(is_corpus_point, "unrefined centroid must be a corpus point");
    }
}

#[test]
fn explicit_seed_matches_new() {
    // SeedStrategy::Explicit is a pass-through equivalent to `new`.
    let (points, _) = two_blobs(10, 13);
    let initial = mat(vec![0.0, 0.0, 20.0, 20.0], 2, 2);
    let c = OnlineClusterer::with_seed(points, SeedStrategy::Explicit(initial), params(4, 10_000))
        .unwrap();
    assert_eq!(c.num_clusters(), 2);
}

#[test]
fn warmup_rejects_bad_config() {
    let (points, _) = two_blobs(10, 14); // 20 points
                                         // more centroids than points
    let seed = SeedStrategy::Warmup {
        num_centroids: 100,
        warmup_points: 10,
        iters: 5,
    };
    assert!(OnlineClusterer::with_seed(points.clone(), seed, params(200, 10)).is_err());
    // zero centroids
    let seed = SeedStrategy::Warmup {
        num_centroids: 0,
        warmup_points: 10,
        iters: 5,
    };
    assert!(OnlineClusterer::with_seed(points, seed, params(8, 10)).is_err());
}
