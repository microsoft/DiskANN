/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::Arc,
};

use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

use crate::{
    ivf::{
        Config, IVFIndex, InsertStats,
        region::Region as _,
        test::{
            harness::{
                PARALLELISMS, assert_abandoned, assert_fails, assert_neighbors, assert_points,
                assert_protocol, batch, brute_force, config, delta, insert, layout, line,
                nearest_lists, search, stage,
            },
            provider::{Call, Metrics, Provider, Snapshot},
        },
    },
    test::cmp::assert_eq_verbose,
};

fn calls(metrics: Metrics) -> [(Call, usize); 10] {
    [
        (Call::Route, metrics.route),
        (Call::Find, metrics.find),
        (Call::Fill, metrics.fill),
        (Call::AppendMembers, metrics.append_members),
        (Call::RemoveMembers, metrics.remove_members),
        (Call::Retire, metrics.retire),
        (Call::Create, metrics.create),
        (Call::RegionFinish, metrics.region_finish),
        (Call::Finish, metrics.finish),
        (Call::Scan, metrics.scan),
    ]
}

/// Hand-written insertion outcomes checked under both execution modes.
#[derive(Debug)]
struct InsertCase<'a> {
    lists: &'a [(f32, &'a [f32])],
    points: &'a [(u32, f32)],
    config: Config,
    splits: usize,
    expected: Vec<(u32, f32, Vec<f32>)>,
}

impl InsertCase<'_> {
    async fn run(&self) -> Vec<Metrics> {
        let mut metrics = Vec::new();
        for parallelism in PARALLELISMS {
            let provider = line(self.lists).await;
            let before = provider.snapshot();
            let mut index = IVFIndex::new(self.config).unwrap();
            let stats = insert(&mut index, &provider, &batch(self.points), parallelism)
                .await
                .unwrap();

            assert_eq!(
                stats,
                InsertStats {
                    inserted: self.points.len(),
                    splits: self.splits,
                },
                "{self:?}, {parallelism:?}"
            );
            let snapshot = provider.snapshot();
            assert_eq!(
                layout(&snapshot),
                self.expected,
                "{self:?}, {parallelism:?}"
            );
            let calls = provider.metrics();
            assert_protocol(&before, &snapshot, self.config, &stats, calls);
            metrics.push(calls);
        }
        metrics
    }
}

#[tokio::test]
async fn insert_without_split_only_routes_and_publishes() {
    for neighbors in [0, 2] {
        // The point at 5 breaks a routing tie; list 0 reaches the threshold exactly.
        InsertCase {
            lists: &[(0.0, &[]), (10.0, &[])],
            points: &[(0, 1.0), (1, 9.0), (2, 5.0)],
            config: config(2, 2, neighbors),
            splits: 0,
            expected: vec![(0, 0.0, vec![1.0, 5.0]), (1, 10.0, vec![9.0])],
        }
        .run()
        .await;
    }
}

#[tokio::test]
async fn insert_rejects_bad_batches_without_publishing() {
    let cases: [(&[(u32, f32)], &str); 3] = [
        (&[], "cannot insert an empty batch"),
        (&[(3, 1.0), (3, 2.0)], "point 3 already exists"),
        (&[(3, 1.0), (2, 2.0)], "point 2 already exists"),
    ];
    for (points, expected) in cases {
        for parallelism in PARALLELISMS {
            let provider = line(&[(0.0, &[0.0, 1.0]), (10.0, &[10.0])]).await;
            let before = provider.snapshot();
            let mut index = IVFIndex::new(config(2, 2, 2)).unwrap();
            assert_fails(
                insert(&mut index, &provider, &batch(points), parallelism).await,
                expected,
            );
            assert_abandoned(&provider, &before);
            assert_eq_verbose!(
                provider.metrics(),
                Metrics {
                    route: 1,
                    ..Metrics::default()
                }
            );
        }
    }
}

#[tokio::test]
async fn insert_requires_a_live_list() {
    for parallelism in PARALLELISMS {
        let provider = line(&[]).await;
        let before = provider.snapshot();
        let mut index = IVFIndex::new(config(2, 2, 0)).unwrap();
        assert_fails(
            insert(&mut index, &provider, &batch(&[(0, 1.0)]), parallelism).await,
            "there is no list to route point 0 to",
        );
        assert_abandoned(&provider, &before);
        assert_eq_verbose!(
            provider.metrics(),
            Metrics {
                route: 1,
                ..Metrics::default()
            }
        );
    }
}

#[tokio::test]
async fn insert_builds_on_published_children() {
    for parallelism in PARALLELISMS {
        let provider = line(&[(0.0, &[0.0, 1.0])]).await;
        let mut index = IVFIndex::new(config(2, 2, 0)).unwrap();
        let mut previous = provider.snapshot();
        let initial = Arc::clone(&previous);
        let steps = [
            (2, 2.0, vec![(1, 0.0, vec![0.0]), (2, 1.5, vec![1.0, 2.0])]),
            (
                3,
                3.0,
                vec![
                    (1, 0.0, vec![0.0]),
                    (3, 1.0, vec![1.0]),
                    (4, 2.5, vec![2.0, 3.0]),
                ],
            ),
        ];
        for (id, x, expected) in steps {
            let before = provider.metrics();
            let old_layout = layout(&previous);
            let stats = insert(&mut index, &provider, &batch(&[(id, x)]), parallelism)
                .await
                .unwrap();
            assert_eq!(
                stats,
                InsertStats {
                    inserted: 1,
                    splits: 1
                }
            );
            assert_eq!(
                layout(&previous),
                old_layout,
                "published snapshots must stay immutable"
            );
            let snapshot = provider.snapshot();
            assert_eq!(layout(&snapshot), expected);
            assert_protocol(
                &previous,
                &snapshot,
                *index.config(),
                &stats,
                delta(before, provider.metrics()),
            );
            previous = snapshot;
        }
        assert_eq!(layout(&initial), [(0, 0.0, vec![0.0, 1.0])]);
    }
}

#[tokio::test]
async fn insert_splits_into_the_configured_number_of_children() {
    let cases = [
        (
            2,
            vec![(1, 1.0, vec![0.0, 2.0]), (2, 10.0, vec![8.0, 10.0, 12.0])],
        ),
        (
            3,
            vec![
                (1, 0.0, vec![0.0]),
                (2, 2.0, vec![2.0]),
                (3, 10.0, vec![8.0, 10.0, 12.0]),
            ],
        ),
        (
            4,
            vec![
                (1, 0.0, vec![0.0]),
                (2, 2.0, vec![2.0]),
                (3, 8.0, vec![8.0]),
                (4, 11.0, vec![10.0, 12.0]),
            ],
        ),
    ];
    for (k, expected) in cases {
        InsertCase {
            lists: &[(0.0, &[0.0, 2.0, 8.0, 10.0])],
            points: &[(4, 12.0)],
            config: config(4, k, 0),
            splits: 1,
            expected,
        }
        .run()
        .await;
    }
}

#[tokio::test]
async fn insert_keeps_an_empty_child_for_equal_points() {
    for neighbors in [0, 2] {
        InsertCase {
            lists: &[(4.0, &[4.0, 4.0])],
            points: &[(2, 4.0)],
            config: config(2, 2, neighbors),
            splits: 1,
            expected: vec![(1, 4.0, vec![4.0, 4.0, 4.0]), (2, 0.0, vec![])],
        }
        .run()
        .await;
    }
}

#[tokio::test]
async fn insert_can_leave_children_oversized_after_one_batch() {
    let points: Vec<_> = (0..10).map(|id| (id, id as f32)).collect();
    InsertCase {
        lists: &[(0.0, &[])],
        points: &points,
        config: config(2, 2, 0),
        splits: 1,
        expected: vec![
            (1, 2.0, vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            (2, 7.0, vec![5.0, 6.0, 7.0, 8.0, 9.0]),
        ],
    }
    .run()
    .await;
}

#[tokio::test]
async fn insert_reassigns_points_with_neighbors() {
    for neighbors in [0, 1, 2] {
        let expected = if neighbors == 0 {
            vec![
                (1, 5.0, vec![5.5, 9.0, 5.2]),
                (2, 100.0, vec![100.0]),
                (3, 2.0, vec![0.0, 4.0, 2.0]),
                (4, 8.0, vec![10.0, 6.0]),
            ]
        } else {
            vec![
                (1, 5.0, vec![5.5, 5.2, 4.0, 6.0]),
                (2, 100.0, vec![100.0]),
                (3, 2.0, vec![0.0, 2.0]),
                (4, 8.0, vec![10.0, 9.0]),
            ]
        };
        InsertCase {
            lists: &[
                (3.0, &[0.0, 10.0, 4.0, 6.0]),
                (5.0, &[5.5, 9.0]),
                (100.0, &[100.0]),
            ],
            points: &[(7, 2.0), (8, 5.2)],
            config: config(3, 2, neighbors),
            splits: 1,
            expected,
        }
        .run()
        .await;
    }
}

#[tokio::test]
async fn insert_links_each_child_to_the_nearest_non_siblings() {
    for neighbors in [0, 1, 2, 3, 4, usize::MAX] {
        let case = InsertCase {
            lists: &[
                (1.0, &[0.0, 2.0]),
                (11.0, &[10.0, 12.0]),
                (100.0, &[100.0]),
                (200.0, &[]),
            ],
            points: &[(5, 0.5), (6, 10.5)],
            config: config(2, 2, neighbors),
            splits: 2,
            expected: vec![
                (2, 100.0, vec![100.0]),
                (3, 200.0, vec![]),
                (4, 0.25, vec![0.0, 0.5]),
                (5, 2.0, vec![2.0]),
                (6, 10.25, vec![10.0, 10.5]),
                (7, 12.0, vec![12.0]),
            ],
        };
        let linked = match neighbors {
            0 => 0,
            1 | 2 => 4,
            3 => 5,
            _ => 6,
        };
        for metrics in case.run().await {
            assert_eq!(metrics.remove_members, linked);
        }
    }
}

#[tokio::test]
async fn insert_reassigns_across_split_parents() {
    InsertCase {
        lists: &[(45.0, &[0.0, 40.0]), (81.0, &[62.0, 100.0])],
        points: &[(4, 60.0), (5, 64.0)],
        config: config(2, 2, 1),
        splits: 2,
        expected: vec![
            (2, 0.0, vec![0.0]),
            (3, 50.0, vec![40.0]),
            (4, 63.0, vec![62.0, 64.0, 60.0]),
            (5, 100.0, vec![100.0]),
        ],
    }
    .run()
    .await;
}

#[tokio::test]
async fn insert_abandons_routed_points_when_fitting_fails() {
    for parallelism in PARALLELISMS {
        let provider = line(&[(0.0, &[0.0, 1.0])]).await;
        let before = provider.snapshot();
        let mut index = IVFIndex::new(Config {
            dim: NonZeroUsize::new(2).unwrap(),
            ..config(2, 2, 0)
        })
        .unwrap();
        assert_fails(
            insert(&mut index, &provider, &batch(&[(2, 2.0)]), parallelism).await,
            "Expected dimension 2, got 1",
        );
        assert_abandoned(&provider, &before);
        assert_eq_verbose!(
            provider.metrics(),
            Metrics {
                route: 1,
                fill: 1,
                ..Metrics::default()
            }
        );

        let failed = provider.metrics();
        let mut index = IVFIndex::new(config(2, 2, 0)).unwrap();
        let stats = insert(&mut index, &provider, &batch(&[(2, 2.0)]), parallelism)
            .await
            .unwrap();
        assert_eq!(
            stats,
            InsertStats {
                inserted: 1,
                splits: 1
            }
        );
        let snapshot = provider.snapshot();
        assert_eq!(
            layout(&snapshot),
            [(1, 0.0, vec![0.0]), (2, 1.5, vec![1.0, 2.0])]
        );
        assert_protocol(
            &before,
            &snapshot,
            *index.config(),
            &stats,
            delta(failed, provider.metrics()),
        );
    }
}

#[tokio::test]
async fn failed_insert_at_every_provider_call_can_be_retried() {
    let lists: &[(f32, &[f32])] = &[
        (45.0, &[0.0, 40.0]),
        (81.0, &[62.0, 100.0]),
        (200.0, &[200.0]),
    ];
    let coordinates = [(5, 60.0), (6, 64.0)];
    let points = batch(&coordinates);
    let config = config(2, 2, usize::MAX);
    let expected = Metrics {
        route: 1,
        find: 4,
        fill: 2,
        append_members: 5,
        remove_members: 5,
        retire: 2,
        create: 4,
        region_finish: 2,
        finish: 1,
        ..Metrics::default()
    };

    for parallelism in PARALLELISMS {
        let golden = line(lists).await;
        let mut index = IVFIndex::new(config).unwrap();
        let stats = insert(&mut index, &golden, &points, parallelism)
            .await
            .unwrap();
        assert_eq!(
            stats,
            InsertStats {
                inserted: 2,
                splits: 2
            }
        );
        assert_eq_verbose!(golden.metrics(), expected);
        let published = golden.snapshot();
        assert_eq!(
            layout(&published),
            [
                (2, 200.0, vec![200.0]),
                (3, 0.0, vec![0.0]),
                (4, 50.0, vec![40.0]),
                (5, 63.0, vec![62.0, 64.0, 60.0]),
                (6, 100.0, vec![100.0]),
            ]
        );

        // Inject one failure at each call reached by the successful run; insertion never scans.
        for (call, count) in calls(expected) {
            for nth in 0..count {
                let provider = line(lists).await.with_fault(call, nth);
                let before = provider.snapshot();
                let mut index = IVFIndex::new(config).unwrap();
                assert_fails(
                    insert(&mut index, &provider, &points, parallelism).await,
                    &format!("injected {call:?} failure"),
                );
                assert_abandoned(&provider, &before);
                let failed = provider.metrics();
                assert_eq!(
                    calls(failed)
                        .into_iter()
                        .find(|&(kind, _)| kind == call)
                        .unwrap()
                        .1,
                    nth + 1,
                    "{call:?} {nth}, {parallelism:?}"
                );
                assert_eq!(failed.finish, usize::from(call == Call::Finish));

                let stats = insert(&mut index, &provider, &points, parallelism)
                    .await
                    .unwrap();
                assert_eq!(
                    stats,
                    InsertStats {
                        inserted: 2,
                        splits: 2
                    }
                );
                let snapshot = provider.snapshot();
                assert_eq!(layout(&snapshot), layout(&published));
                assert_eq!(snapshot.next_list(), published.next_list());
                assert_eq_verbose!(delta(failed, provider.metrics()), expected);
            }
        }
    }
}

/// Count only touched lists whose routed size exceeds the threshold, without running a split.
fn routing_model(snapshot: &Snapshot, points: &[(u32, &[f32])], threshold: usize) -> InsertStats {
    let mut sizes: BTreeMap<_, _> = snapshot
        .lists()
        .iter()
        .map(|(&id, list)| (id, list.members().len()))
        .collect();
    let mut touched = BTreeSet::new();
    for &(_, point) in points {
        let list = nearest_lists(snapshot, point, 1, |_| true)[0];
        *sizes.get_mut(&list).unwrap() += 1;
        touched.insert(list);
    }
    InsertStats {
        inserted: points.len(),
        splits: touched
            .into_iter()
            .filter(|id| sizes[id] > threshold)
            .count(),
    }
}

/// Compare incremental random builds across modes after each batch, using independent
/// routing and exhaustive search oracles.
async fn insert_fuzz_impl(dim: usize, ntrials: usize, rng: &mut StdRng) {
    let coordinate = Uniform::new_inclusive(-8i8, 8).unwrap();
    for trial in 0..ntrials {
        let threshold = rng.random_range(2..=8);
        let k = rng.random_range(2..=4.min(threshold + 1));
        let config = Config {
            dim: NonZeroUsize::new(dim).unwrap(),
            split_iterations: NonZeroUsize::new(rng.random_range(1..=4)).unwrap(),
            ..config(threshold, k, rng.random_range(0..=3))
        };
        let seeding = Provider::new(Snapshot::new(config.dim));
        let centroids: Vec<Vec<f32>> = (0..rng.random_range(1..=3))
            .map(|_| {
                (0..dim)
                    .map(|_| f32::from(coordinate.sample(rng)))
                    .collect()
            })
            .collect();
        stage(&seeding, &[], |region| {
            for centroid in &centroids {
                region.create(centroid, &[])?;
            }
            Ok(())
        })
        .await
        .unwrap();
        let initial = seeding.snapshot();
        let providers = PARALLELISMS.map(|_| Provider::new(Snapshot::clone(&initial)));
        let mut indexes = PARALLELISMS.map(|_| IVFIndex::new(config).unwrap());
        let mut vectors: Vec<Vec<f32>> = Vec::new();

        for round in 0..6 {
            let first = vectors.len();
            vectors.extend((0..rng.random_range(1..=12)).map(|_| {
                (0..dim)
                    .map(|_| f32::from(coordinate.sample(rng)))
                    .collect::<Vec<_>>()
            }));
            let points: Vec<_> = vectors
                .iter()
                .enumerate()
                .skip(first)
                .map(|(id, vector)| (u32::try_from(id).unwrap(), vector.as_slice()))
                .collect();
            let queries = [vec![0.0; dim], vectors[first].clone()];

            for ((parallelism, index), provider) in
                PARALLELISMS.into_iter().zip(&mut indexes).zip(&providers)
            {
                let before = provider.snapshot();
                let metrics = provider.metrics();
                let expected = routing_model(&before, &points, threshold);
                let stats = insert(index, provider, &points, parallelism).await.unwrap();
                assert_eq!(
                    stats, expected,
                    "dim {dim}, trial {trial}, round {round}, {parallelism:?}"
                );
                let snapshot = provider.snapshot();
                assert_points(
                    &snapshot,
                    vectors
                        .iter()
                        .enumerate()
                        .map(|(id, vector)| (u32::try_from(id).unwrap(), vector.as_slice())),
                );
                assert_protocol(
                    &before,
                    &snapshot,
                    config,
                    &stats,
                    delta(metrics, provider.metrics()),
                );
                for query in &queries {
                    let candidates = brute_force(&snapshot, query);
                    for k in [1, 5, vectors.len() + 1] {
                        let found = search(index, provider, query, k, usize::MAX).await.unwrap();
                        assert_neighbors(&found, &candidates, k);
                    }
                }
            }

            let sequential = providers[0].snapshot();
            let rayon = providers[1].snapshot();
            assert_eq!(sequential.lists(), rayon.lists());
            assert_eq!(sequential.next_list(), rayon.next_list());
            assert_eq_verbose!(providers[0].metrics(), providers[1].metrics());
        }
    }
}

#[tokio::test]
async fn insert_fuzz_preserves_invariants_and_matches_parallel_modes() {
    let mut rng = StdRng::seed_from_u64(0x6a1c9e2b47d03f85);
    for dim in [1, 2, 5] {
        insert_fuzz_impl(dim, 5, &mut rng).await;
    }
}
