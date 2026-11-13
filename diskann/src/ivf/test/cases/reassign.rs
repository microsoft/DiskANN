/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::BTreeMap;

use diskann_quantization::Parallelism;
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
    seq::SliceRandom,
};

use crate::{
    ANNResult,
    ivf::{
        split::{Linked, Links, Reassign as _, policies::NearestCentroid},
        test::{
            harness::{PARALLELISMS, assert_abandoned, assert_fails, layout, line, stage},
            provider::{Metrics, Provider, Snapshot},
        },
    },
    test::cmp::assert_eq_verbose,
};

/// Reassign `filled` through `links`, publishing only on success.
async fn reassign(
    provider: &Provider,
    filled: &[u32],
    links: &Links<u32>,
    parallelism: Parallelism,
) -> ANNResult<()> {
    stage(provider, filled, |region| {
        Linked::new(NearestCentroid).reassign(region, links, parallelism)
    })
    .await
}

#[tokio::test]
async fn linked_moves_only_along_links_and_keeps_empty_lists() {
    for parallelism in PARALLELISMS {
        let provider = line(&[
            (0.0, &[19.0, 10.0]),
            (10.0, &[20.0]),
            (20.0, &[30.0, 20.0, 9.0]),
            (30.0, &[-1.0]),
            (100.0, &[100.0]),
        ])
        .await;
        let mut links = Links::new([1, 0]);
        links.link(0, 1);
        links.link(1, 2);

        reassign(&provider, &[3, 2, 0, 1], &links, parallelism)
            .await
            .unwrap();

        // Point 19 cannot go straight from list 0 to list 2; point 30 cannot reach
        // the unlinked list 3. Incoming points follow source and member order.
        assert_eq!(
            layout(&provider.snapshot()),
            [
                (0, 0.0, vec![]),
                (1, 10.0, vec![19.0, 10.0, 9.0]),
                (2, 20.0, vec![30.0, 20.0, 20.0]),
                (3, 30.0, vec![-1.0]),
                (4, 100.0, vec![100.0]),
            ],
            "{parallelism:?}"
        );
        assert_eq_verbose!(
            provider.metrics(),
            Metrics {
                fill: 1,
                remove_members: 3,
                append_members: 3,
                region_finish: 1,
                finish: 1,
                ..Metrics::default()
            }
        );
    }
}

#[tokio::test]
async fn linked_breaks_ties_in_link_order_before_its_own_list() {
    for targets in [[2, 1], [1, 2]] {
        for parallelism in PARALLELISMS {
            // Point 5 is equally near both linked centroids and its own centroid.
            let provider = line(&[(0.0, &[5.0]), (10.0, &[]), (10.0, &[])]).await;
            let mut expected = layout(&provider.snapshot());
            expected[0].2.clear();
            expected[targets[0] as usize].2.push(5.0);

            let mut links = Links::new([0]);
            for target in targets {
                links.link(0, target);
            }
            reassign(&provider, &[0, 1, 2], &links, parallelism)
                .await
                .unwrap();

            assert_eq!(
                layout(&provider.snapshot()),
                expected,
                "targets {targets:?}, {parallelism:?}"
            );
        }
    }
}

#[tokio::test]
async fn linked_rejects_unfilled_sources_and_neighbors_before_writing() {
    let cases: [&[(u32, u32)]; 2] = [&[(0, 1), (1, 2)], &[(0, 1)]];
    for edges in cases {
        for parallelism in PARALLELISMS {
            let provider = line(&[(0.0, &[9.0]), (10.0, &[1.0]), (20.0, &[20.0])]).await;
            let before = provider.snapshot();
            let mut links = Links::new([0, 1, 2]);
            for &(from, to) in edges {
                links.link(from, to);
            }

            assert_fails(
                reassign(&provider, &[0, 1], &links, parallelism).await,
                "Split could not retrieve list 2 from the region",
            );
            assert_abandoned(&provider, &before);
            assert_eq_verbose!(
                provider.metrics(),
                Metrics {
                    fill: 1,
                    ..Metrics::default()
                }
            );
        }
    }
}

/// Independent 1D oracle: absolute distance gives the same ordering as squared L2.
///
/// Rank linked candidates in edge order before the source; defer arrivals until all
/// original memberships have been processed so incoming points cannot move twice.
fn model(snapshot: &Snapshot, sources: &[u32], edges: &[(u32, u32)]) -> BTreeMap<u32, Vec<u32>> {
    let mut expected: BTreeMap<_, _> = snapshot
        .lists()
        .iter()
        .map(|(&id, list)| (id, list.members().to_vec()))
        .collect();
    let mut moved = Vec::new();

    for &source in sources {
        let candidates: Vec<u32> = edges
            .iter()
            .filter_map(|&(from, to)| {
                if from == source {
                    Some(to)
                } else if to == source {
                    Some(from)
                } else {
                    None
                }
            })
            .chain(std::iter::once(source))
            .collect();

        expected.get_mut(&source).unwrap().retain(|&member| {
            let point = snapshot.vector(member).unwrap()[0];
            let distance = |id| (point - snapshot.lists()[&id].centroid()[0]).abs();
            let destination = candidates
                .iter()
                .copied()
                .min_by(|&a, &b| distance(a).total_cmp(&distance(b)))
                .unwrap();
            if destination == source {
                true
            } else {
                moved.push((destination, member));
                false
            }
        });
    }

    for (destination, member) in moved {
        expected.get_mut(&destination).unwrap().push(member);
    }
    expected
}

/// Compare randomized memberships with [`model`] in both execution modes.
async fn linked_fuzz_impl(nlists: u32, ntrials: usize, rng: &mut StdRng) {
    let coordinate = Uniform::new_inclusive(-16i8, 16).unwrap();
    let filled: Vec<u32> = (0..nlists).rev().collect();

    for trial in 0..ntrials {
        let lists: Vec<(f32, Vec<f32>)> = (0..nlists)
            .map(|_| {
                let centroid = f32::from(coordinate.sample(rng));
                let count = rng.random_range(0..=8);
                let points = (0..count)
                    .map(|_| f32::from(coordinate.sample(rng)))
                    .collect();
                (centroid, points)
            })
            .collect();
        let input: Vec<_> = lists
            .iter()
            .map(|(centroid, points)| (*centroid, points.as_slice()))
            .collect();

        let mut sources = filled.clone();
        sources.shuffle(rng);
        let count = rng.random_range(0..=sources.len());
        sources.truncate(count);
        let mut links = Links::new(sources.iter().copied());
        let mut edges = Vec::new();
        if sources.len() > 1 {
            for _ in 0..3 * sources.len() {
                let from = sources[rng.random_range(0..sources.len())];
                let to = sources[rng.random_range(0..sources.len())];
                if from != to {
                    links.link(from, to);
                    edges.push((from, to));
                }
            }
        }

        for parallelism in PARALLELISMS {
            let provider = line(&input).await;
            let before = provider.snapshot();
            let expected = model(&before, &sources, &edges);
            reassign(&provider, &filled, &links, parallelism)
                .await
                .unwrap();

            let snapshot = provider.snapshot();
            snapshot.is_consistent().unwrap();
            let actual: BTreeMap<_, _> = snapshot
                .lists()
                .iter()
                .map(|(&id, list)| (id, list.members().to_vec()))
                .collect();
            assert_eq!(
                actual, expected,
                "lists {nlists}, trial {trial}, {parallelism:?}"
            );
            for (&id, list) in before.lists() {
                assert_eq!(
                    snapshot.lists()[&id].centroid(),
                    list.centroid(),
                    "list {id}"
                );
            }
            assert_eq_verbose!(
                provider.metrics(),
                Metrics {
                    fill: 1,
                    remove_members: sources.len(),
                    append_members: sources.len(),
                    region_finish: 1,
                    finish: 1,
                    ..Metrics::default()
                }
            );
        }
    }
}

#[tokio::test]
async fn linked_fuzz() {
    let mut rng = StdRng::seed_from_u64(0x1f83d9abfb41bd6b);
    for nlists in [0, 1, 2, 5, 9] {
        linked_fuzz_impl(nlists, 10, &mut rng).await;
    }
}
