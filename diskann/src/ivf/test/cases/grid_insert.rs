/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Representative grid-insertion baselines followed by limited/full-probe search.
//!
//! Every run records list sizes and provider call counts. Full layouts are opt-in via
//! [`super::DUMP_IVF_STATE`]. A fresh Rayon build must match the Sequential baseline exactly.

use std::collections::BTreeSet;

use diskann_quantization::Parallelism;
use diskann_utils::views::rowmajor::Matrix;

use crate::{
    graph::test::synthetic::Grid,
    ivf::{
        Config, IVFIndex, InsertStats, ReassignCount,
        test::{
            harness::{
                assert_neighbors, assert_points, assert_protocol, assert_scans, brute_force,
                config, delta, insert, line, scanned_brute_force, search,
            },
            provider::{Metrics, Provider, Snapshot},
        },
    },
    test::{
        TestFormat, TestRoot,
        cmp::{assert_eq_verbose, verbose_eq},
        get_or_save_test_results_with_format,
    },
};

use super::DUMP_IVF_STATE;

const SIZE: usize = 100;

/////////////
// Results //
/////////////

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Parameters {
    grid_dims: usize,
    grid_size: usize,
    batch_size: usize,
    split_threshold: usize,
    split_k: usize,
    reassign_count: Option<usize>,
    split_iterations: usize,
}

verbose_eq!(Parameters {
    grid_dims,
    grid_size,
    batch_size,
    split_threshold,
    split_k,
    reassign_count,
    split_iterations,
});

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GridSearch {
    query: Vec<f32>,
    k: usize,
    /// Full-probe searches store the live-list count, not a sentinel.
    nprobe: usize,
    num_results: usize,
    results: Vec<(u32, f32)>,
    /// Work for this query only, excluding insertion and earlier searches.
    metrics: Metrics,
}

verbose_eq!(GridSearch {
    query,
    k,
    nprobe,
    num_results,
    results,
    metrics,
});

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ListState {
    id: u32,
    centroid: Vec<f32>,
    members: Vec<u32>,
}

verbose_eq!(ListState {
    id,
    centroid,
    members
});

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct GridInsertBaseline {
    description: String,
    parameters: Parameters,
    num_inserted: usize,
    splits_per_batch: Vec<usize>,
    insert_metrics: Metrics,
    list_sizes: Vec<(u32, usize)>,
    searches: Vec<GridSearch>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    lists: Option<Vec<ListState>>,
}

verbose_eq!(GridInsertBaseline {
    description,
    parameters,
    num_inserted,
    splits_per_batch,
    insert_metrics,
    list_sizes,
    searches,
    lists,
});

/////////////
// Helpers //
/////////////

/// Chunk input into separate builds; `batch_size` is independent of the split threshold.
async fn run_build(
    index: &mut IVFIndex,
    provider: &Provider,
    points: &[(u32, &[f32])],
    batch_size: usize,
    parallelism: Parallelism,
) -> (usize, Vec<usize>) {
    let mut num_inserted = 0;
    let mut splits_per_batch = Vec::new();
    for batch in points.chunks(batch_size) {
        let before = provider.snapshot();
        let metrics = provider.metrics();
        let stats = insert(index, provider, batch, parallelism).await.unwrap();
        let after = provider.snapshot();
        assert_protocol(
            &before,
            &after,
            *index.config(),
            &stats,
            delta(metrics, provider.metrics()),
        );
        let InsertStats { inserted, splits } = stats;
        assert_eq!(inserted, batch.len());
        num_inserted += inserted;
        splits_per_batch.push(splits);
    }
    (num_inserted, splits_per_batch)
}

async fn run_searches(
    index: &IVFIndex,
    provider: &Provider,
    snapshot: &Snapshot,
) -> Vec<GridSearch> {
    let k = 10;
    let all = snapshot.lists().len();
    let probes: BTreeSet<_> = [1, 2, all].map(|n| n.min(all)).into_iter().collect();
    let mut searches = Vec::new();
    for coordinate in [-1.0, SIZE as f32] {
        let query = vec![coordinate; snapshot.dim()];
        for &nprobe in &probes {
            let candidates = if nprobe >= all {
                brute_force(snapshot, &query)
            } else {
                scanned_brute_force(snapshot, &query, nprobe)
            };
            let before = provider.metrics();
            let found = search(index, provider, &query, k, nprobe).await.unwrap();
            assert_neighbors(&found, &candidates, k);
            let metrics = delta(before, provider.metrics());
            assert_scans(metrics, nprobe.min(all), candidates.len());
            searches.push(GridSearch {
                query: query.clone(),
                k,
                nprobe,
                num_results: found.len(),
                results: found
                    .into_iter()
                    .map(|neighbor| neighbor.as_tuple())
                    .collect(),
                metrics,
            });
        }
    }
    searches
}

async fn run_grid(
    batch_size: usize,
    config: Config,
    parallelism: Parallelism,
) -> GridInsertBaseline {
    let data = Grid::One.data(SIZE);
    let points: Vec<_> = data
        .rows()
        .enumerate()
        .map(|(id, vector)| (u32::try_from(id).unwrap(), vector))
        .collect();
    let provider = line(&[(0.0, &[])]).await;
    let mut index = IVFIndex::new(config).unwrap();
    let (num_inserted, splits_per_batch) =
        run_build(&mut index, &provider, &points, batch_size, parallelism).await;

    let snapshot = provider.snapshot();
    assert_points(&snapshot, points.iter().copied());
    assert_eq!(num_inserted, points.len());
    // Capture insertion work before search increments the same provider counters.
    let insert_metrics = provider.metrics();
    let searches = run_searches(&index, &provider, &snapshot).await;
    let list_sizes = snapshot
        .lists()
        .iter()
        .map(|(&id, list)| (id, list.members().len()))
        .collect();
    let lists = DUMP_IVF_STATE.then(|| {
        snapshot
            .lists()
            .iter()
            .map(|(&id, list)| ListState {
                id,
                centroid: list.centroid().to_vec(),
                members: list.members().to_vec(),
            })
            .collect()
    });

    GridInsertBaseline {
        description: format!(
            "Insert {} grid points in batches of {batch_size}.",
            points.len()
        ),
        parameters: Parameters {
            grid_dims: data.ncols(),
            grid_size: SIZE,
            batch_size,
            split_threshold: config.split_threshold.get(),
            split_k: config.split_k.get(),
            reassign_count: match config.reassign_count {
                ReassignCount::None => None,
                ReassignCount::Count(n) => Some(n.get()),
            },
            split_iterations: config.split_iterations.get(),
        },
        num_inserted,
        splits_per_batch,
        insert_metrics,
        list_sizes,
        searches,
        lists,
    }
}

async fn grid_insert_impl(batch_size: usize, configurations: &[(usize, usize)]) {
    for &(split_k, neighbors) in configurations {
        let config = config(8, split_k, neighbors);
        let baseline = run_grid(batch_size, config, Parallelism::Sequential).await;
        let reassign = if neighbors == 0 {
            "none".to_string()
        } else {
            neighbors.to_string()
        };
        let name = format!(
            "insert_{}_{}_batch_{batch_size}/k_{split_k}_reassign_{reassign}",
            baseline.parameters.grid_dims, baseline.parameters.grid_size,
        );
        let mut root = TestRoot::new("ivf/test/cases/grid_insert");
        let expected = get_or_save_test_results_with_format(
            &root.path().push(&name),
            &baseline,
            TestFormat::InlineNumericPairs,
        );
        assert_eq_verbose!(expected, baseline, "{name}");

        let rayon = run_grid(batch_size, config, Parallelism::Rayon).await;
        assert_eq_verbose!(baseline, rayon, "{name}");
    }
}

///////////
// Tests //
///////////

#[tokio::test]
async fn single_1d_100() {
    grid_insert_impl(1, &[(4, 0), (4, 2)]).await;
}

#[tokio::test]
async fn batch_10_1d_100() {
    grid_insert_impl(10, &[(2, 0), (2, 2), (4, 2)]).await;
}

#[tokio::test]
async fn batch_all_1d_100() {
    grid_insert_impl(SIZE, &[(4, 0)]).await;
}

#[tokio::test]
async fn grid_searches_use_distinct_available_probe_counts() {
    let lists: &[(f32, &[f32])] = &[(0.0, &[0.0]), (10.0, &[10.0]), (20.0, &[20.0])];
    let cases: [(usize, &[usize]); 3] = [(1, &[1]), (2, &[1, 2]), (3, &[1, 2, 3])];
    let index = IVFIndex::new(config(8, 2, 0)).unwrap();
    for (count, probes) in cases {
        let provider = line(&lists[..count]).await;
        let snapshot = provider.snapshot();
        let searches = run_searches(&index, &provider, &snapshot).await;
        let actual: Vec<_> = searches.into_iter().map(|s| (s.query, s.nprobe)).collect();
        let expected: Vec<_> = [-1.0, SIZE as f32]
            .into_iter()
            .flat_map(|x| probes.iter().map(move |&n| (vec![x], n)))
            .collect();
        assert_eq!(actual, expected, "{count} lists");
    }
}
