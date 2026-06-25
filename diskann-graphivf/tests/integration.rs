/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! End-to-end tests: build a synthetic clustered corpus, query it, and check
//! recall against brute-force ground truth for both supported metrics.

use diskann_graphivf::{
    BuildParams, GraphIvfIndex, GraphParams, Half, Metric, SearchParams, VectorRepr,
};
use diskann_utils::views::Matrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIM: usize = 16;
const NUM_CLUSTERS_DATA: usize = 32;
const POINTS_PER_CLUSTER: usize = 64;
const NUM_POINTS: usize = NUM_CLUSTERS_DATA * POINTS_PER_CLUSTER;

/// Generate `NUM_POINTS` vectors grouped into `NUM_CLUSTERS_DATA` Gaussian-ish
/// blobs so that nearest-neighbor structure is non-trivial but learnable.
fn make_corpus(rng: &mut StdRng) -> Vec<f32> {
    let mut centers = vec![0.0f32; NUM_CLUSTERS_DATA * DIM];
    for c in centers.iter_mut() {
        *c = rng.random_range(-10.0..10.0);
    }
    let mut data = vec![0.0f32; NUM_POINTS * DIM];
    for p in 0..NUM_POINTS {
        let cluster = p / POINTS_PER_CLUSTER;
        for d in 0..DIM {
            let jitter: f32 = rng.random_range(-1.0..1.0);
            data[p * DIM + d] = centers[cluster * DIM + d] + jitter;
        }
    }
    data
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Brute-force top-`k` ids by squared-L2 over `data`, normalizing rows first
/// when `cosine` is set (matching the index's internal scoring).
fn brute_force(data: &[f32], query: &[f32], k: usize, cosine: bool) -> Vec<u32> {
    let mut q = query.to_vec();
    if cosine {
        normalize(&mut q);
    }
    let mut scored: Vec<(u32, f32)> = data
        .chunks_exact(DIM)
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.to_vec();
            if cosine {
                normalize(&mut r);
            }
            let dist: f32 = q.iter().zip(r.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            (i as u32, dist)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn recall(got: &[(u32, f32)], expected: &[u32]) -> f32 {
    let hits = got.iter().filter(|(id, _)| expected.contains(id)).count();
    hits as f32 / expected.len() as f32
}

fn build_params(metric: Metric) -> BuildParams {
    BuildParams {
        num_clusters: 48,
        metric,
        sample_size: NUM_POINTS,
        kmeans_iters: 15,
        assign_l: 48,
        graph: GraphParams {
            degree: 24,
            slack: 1.3,
            l_build: 64,
            alpha: 1.2,
        },
        num_threads: 2,
        seed: 42,
    }
}

/// Encode an `f32` query into the stored element type `T`.
fn encode<T: VectorRepr>(q: &[f32]) -> Vec<T> {
    q.iter()
        .map(|&v| T::from_f32(v).expect("query value representable in T"))
        .collect()
}

/// Build, search across a batch of corpus queries, and assert mean recall.
/// Generic over the stored inverted-list element type `T` (e.g. `f32` or f16).
fn run_metric<T: VectorRepr>(metric: Metric, min_mean_recall: f32) {
    let cosine = matches!(metric, Metric::Cosine);
    let mut rng = StdRng::seed_from_u64(7);
    let data = make_corpus(&mut rng);

    let matrix =
        Matrix::try_from(data.clone().into_boxed_slice(), NUM_POINTS, DIM).expect("matrix shape");

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("idx");

    let index =
        GraphIvfIndex::<T>::build(matrix.as_view(), &build_params(metric), &prefix).expect("build");
    assert_eq!(index.dim(), DIM);
    assert_eq!(index.num_clusters(), 48);

    let mut searcher = index.searcher().expect("searcher");
    let params = SearchParams {
        nlist: 12,
        centroid_search_l: 32,
    };

    let k = 10;
    let mut total = 0.0f32;
    let num_queries = 40;
    for qi in 0..num_queries {
        // Use existing corpus points as queries (perturbed slightly).
        let base = (qi * (NUM_POINTS / num_queries)) % NUM_POINTS;
        let mut query: Vec<f32> = data[base * DIM..(base + 1) * DIM].to_vec();
        for x in query.iter_mut() {
            *x += rng.random_range(-0.2..0.2);
        }

        // The searcher takes a `T` query and does not normalize for cosine, so
        // normalize here before encoding (the corpus was normalized at build).
        let mut qn = query.clone();
        if cosine {
            normalize(&mut qn);
        }
        let q_t = encode::<T>(&qn);

        let got = searcher.search(&q_t, k, &params).expect("search");
        assert!(got.len() <= k);
        // Results must be sorted ascending by distance.
        for w in got.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        let expected = brute_force(&data, &query, k, cosine);
        total += recall(&got, &expected);
    }

    let mean = total / num_queries as f32;
    assert!(
        mean >= min_mean_recall,
        "mean recall@{k} for {metric:?} was {mean:.3}, expected >= {min_mean_recall}"
    );
}

#[test]
fn recall_l2() {
    run_metric::<f32>(Metric::L2, 0.85);
}

#[test]
fn recall_cosine() {
    run_metric::<f32>(Metric::Cosine, 0.80);
}

#[test]
fn recall_l2_f16() {
    // f16 lists trade a little precision for half the on-disk vector size; recall
    // should stay close to the f32 baseline on this well-separated data.
    run_metric::<Half>(Metric::L2, 0.85);
}

#[test]
fn recall_cosine_f16() {
    run_metric::<Half>(Metric::Cosine, 0.80);
}

#[test]
fn load_round_trip() {
    let mut rng = StdRng::seed_from_u64(11);
    let data = make_corpus(&mut rng);
    let matrix =
        Matrix::try_from(data.clone().into_boxed_slice(), NUM_POINTS, DIM).expect("matrix shape");

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("idx");

    // Build and search.
    let built = GraphIvfIndex::<f32>::build(matrix.as_view(), &build_params(Metric::L2), &prefix)
        .expect("build");
    let params = SearchParams {
        nlist: 12,
        centroid_search_l: 32,
    };
    let query: Vec<f32> = data[0..DIM].to_vec();
    let from_built = {
        let mut s = built.searcher().expect("searcher");
        s.search(&query, 10, &params).expect("search")
    };
    drop(built);

    // Reload from disk and search again.
    let loaded = GraphIvfIndex::<f32>::load(&prefix, 2).expect("load");
    assert_eq!(loaded.dim(), DIM);
    assert_eq!(loaded.num_clusters(), 48);
    let from_loaded = {
        let mut s = loaded.searcher().expect("searcher");
        s.search(&query, 10, &params).expect("search")
    };

    // The reloaded index should return identical results.
    assert_eq!(from_built, from_loaded);
}

#[test]
fn load_rejects_format_mismatch() {
    let mut rng = StdRng::seed_from_u64(13);
    let data = make_corpus(&mut rng);
    let matrix = Matrix::try_from(data.into_boxed_slice(), NUM_POINTS, DIM).expect("matrix shape");

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("idx");

    // Written as f32 ...
    GraphIvfIndex::<f32>::build(matrix.as_view(), &build_params(Metric::L2), &prefix)
        .expect("build");
    // ... cannot be loaded as f16.
    assert!(GraphIvfIndex::<Half>::load(&prefix, 2).is_err());
    // ... but loads fine as f32.
    assert!(GraphIvfIndex::<f32>::load(&prefix, 2).is_ok());
}

#[test]
fn rejects_bad_params() {
    let mut rng = StdRng::seed_from_u64(3);
    let data = make_corpus(&mut rng);
    let matrix = Matrix::try_from(data.into_boxed_slice(), NUM_POINTS, DIM).expect("matrix shape");

    let dir = tempfile::tempdir().expect("tempdir");
    let prefix = dir.path().join("idx");

    // num_clusters greater than the corpus size is rejected.
    let mut bad = build_params(Metric::L2);
    bad.num_clusters = NUM_POINTS + 1;
    assert!(GraphIvfIndex::<f32>::build(matrix.as_view(), &bad, &prefix).is_err());

    // A valid build, then a search with nlist greater than the cluster count.
    let index = GraphIvfIndex::<f32>::build(matrix.as_view(), &build_params(Metric::L2), &prefix)
        .expect("build");
    let mut searcher = index.searcher().expect("searcher");
    let bad_search = SearchParams {
        nlist: index.num_clusters() + 1,
        centroid_search_l: 64,
    };
    assert!(searcher.search(&data_query(), 10, &bad_search).is_err());
}

fn data_query() -> Vec<f32> {
    vec![0.0f32; DIM]
}
