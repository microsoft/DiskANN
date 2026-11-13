/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_quantization::Parallelism;

use crate::{
    ANNResult,
    ivf::{
        split::{
            KMeansSplit, Split as _,
            policies::{FirstPoints, NearestCentroid},
        },
        test::{
            harness::{PARALLELISMS, assert_abandoned, assert_fails, layout, line, stage},
            provider::{Metrics, Provider},
        },
    },
    test::cmp::assert_eq_verbose,
};

/// Use first-point seeding so expected centroids and member order are deterministic.
fn kmeans(k: usize, iterations: usize, dim: usize) -> KMeansSplit {
    KMeansSplit::new(
        NonZeroUsize::new(k).unwrap(),
        NonZeroUsize::new(iterations).unwrap(),
        NonZeroUsize::new(dim).unwrap(),
        NearestCentroid,
        FirstPoints,
    )
}

/// Split `parents` among `filled`, publishing on success, and return the reported
/// `(parent, child)` pairs in emission order.
async fn split(
    provider: &Provider,
    filled: &[u32],
    parents: &[u32],
    mut policy: KMeansSplit,
    parallelism: Parallelism,
) -> ANNResult<Vec<(u32, u32)>> {
    stage(provider, filled, |region| {
        let mut children = Vec::new();
        policy.split(region, parents, parallelism, |parent, child| {
            children.push((parent, child))
        })?;
        Ok(children)
    })
    .await
}

#[tokio::test]
async fn kmeans_split_creates_parent_local_children_in_parent_order() {
    let cases = [
        (
            2,
            vec![(1, 4), (1, 5), (0, 6), (0, 7)],
            vec![
                (2, 50.0, vec![50.0]),
                (3, -50.0, vec![-50.0]),
                (4, 2.0, vec![1.0, 3.0]),
                (5, 8.0, vec![7.0, 9.0]),
                (6, 1.0, vec![0.0, 2.0]),
                (7, 9.0, vec![8.0, 10.0]),
            ],
        ),
        (
            3,
            vec![(1, 4), (1, 5), (1, 6), (0, 7), (0, 8), (0, 9)],
            vec![
                (2, 50.0, vec![50.0]),
                (3, -50.0, vec![-50.0]),
                (4, 1.0, vec![1.0]),
                (5, 3.0, vec![3.0]),
                (6, 8.0, vec![7.0, 9.0]),
                (7, 0.0, vec![0.0]),
                (8, 2.0, vec![2.0]),
                (9, 9.0, vec![8.0, 10.0]),
            ],
        ),
    ];

    for (k, children, expected) in cases {
        for parallelism in PARALLELISMS {
            // The parents overlap: a global child assignment would mix their points.
            // List 2 is filled but not split; list 3 is outside the region.
            let provider = line(&[
                (0.0, &[0.0, 2.0, 8.0, 10.0]),
                (10.0, &[1.0, 3.0, 7.0, 9.0]),
                (50.0, &[50.0]),
                (-50.0, &[-50.0]),
            ])
            .await;
            let emitted = split(
                &provider,
                &[2, 0, 1],
                &[1, 0],
                kmeans(k, 10, 1),
                parallelism,
            )
            .await
            .unwrap();

            assert_eq!(emitted, children, "k {k}, {parallelism:?}");
            assert_eq!(
                layout(&provider.snapshot()),
                expected,
                "k {k}, {parallelism:?}"
            );
            assert_eq_verbose!(
                provider.metrics(),
                Metrics {
                    fill: 1,
                    retire: 2,
                    create: 2 * k,
                    region_finish: 1,
                    finish: 1,
                    ..Metrics::default()
                }
            );
        }
    }
}

#[tokio::test]
async fn kmeans_split_assigns_against_the_fitted_centroids() {
    for parallelism in PARALLELISMS {
        let provider = line(&[(0.0, &[0.0, 2.0, 6.0, 10.0])]).await;
        let emitted = split(&provider, &[0], &[0], kmeans(2, 1, 1), parallelism)
            .await
            .unwrap();

        // One Lloyd iteration moves the second seed from 2 to 6. The point at 2 must
        // then join the centroid at 0, not remain in the second seed's bucket.
        assert_eq!(emitted, [(0, 1), (0, 2)], "{parallelism:?}");
        assert_eq!(
            layout(&provider.snapshot()),
            [(1, 0.0, vec![0.0, 2.0]), (2, 6.0, vec![6.0, 10.0])],
            "{parallelism:?}"
        );
    }
}

#[tokio::test]
async fn kmeans_split_keeps_an_empty_child_at_the_origin() {
    for parallelism in PARALLELISMS {
        let provider = line(&[(4.0, &[4.0, 4.0, 4.0, 4.0])]).await;
        let emitted = split(&provider, &[0], &[0], kmeans(2, 10, 1), parallelism)
            .await
            .unwrap();

        assert_eq!(emitted, [(0, 1), (0, 2)], "{parallelism:?}");
        assert_eq!(
            layout(&provider.snapshot()),
            [(1, 4.0, vec![4.0, 4.0, 4.0, 4.0]), (2, 0.0, vec![])],
            "{parallelism:?}"
        );
    }
}

#[tokio::test]
async fn kmeans_split_rejects_bad_inputs_before_writing() {
    let cases: [(&[u32], usize, &str); 5] = [
        (
            &[0, 3],
            1,
            "Split could not retrieve list 3 from the region",
        ),
        (&[], 1, "Parents is empty"),
        (&[0, 1], 1, "Expected at least 2 points to seed, got 1"),
        (&[0, 2], 1, "Expected at least 2 points to seed, got 0"),
        (&[0], 2, "Expected dimension 2, got 1"),
    ];

    for (parents, dim, expected) in cases {
        for parallelism in PARALLELISMS {
            let provider = line(&[
                (0.0, &[0.0, 2.0, 8.0, 10.0]),
                (10.0, &[10.0]),
                (20.0, &[]),
                (30.0, &[30.0, 31.0]),
            ])
            .await;
            let before = provider.snapshot();
            let mut policy = kmeans(2, 10, dim);
            let result = stage(&provider, &[0, 1, 2], |region| {
                policy.split(region, parents, parallelism, |_, _| {
                    panic!("an invalid split must not emit children")
                })
            })
            .await;

            assert_fails(result, expected);
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
