/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::BTreeMap;

use super::reassign::model;
use crate::ivf::{
    IVFIndex,
    split::{Linked, Links, Reassign as _},
    test::{
        harness::{
            PARALLELISMS, assert_abandoned, assert_fails, batch, config, insert, line, stage,
        },
        provider::{Call, Provider, Snapshot},
    },
};

fn memberships(snapshot: &Snapshot) -> BTreeMap<u32, Vec<u32>> {
    snapshot
        .lists()
        .iter()
        .map(|(&id, list)| (id, list.members().to_vec()))
        .collect()
}

#[tokio::test]
async fn linked_gemm_preserves_membership_order_empty_lists_and_candidate_boundaries() {
    let first: Vec<_> = (0..133).map(|i| [-2.0, 5.0, 7.0][i % 3]).collect();
    let second: Vec<_> = (0..129)
        .map(|i| if i % 2 == 0 { 5.0 } else { -1.0 })
        .collect();
    let mut lists: Vec<(f32, &[f32])> = vec![(5.0, &first), (-1.0, &second), (-1.0, &[])];
    lists.extend((3..=12).map(|id| (100.0 + id as f32, &[][..])));
    // This centroid is globally nearest for some points, but is not linked.
    lists.push((-2.0, &[-2.0]));
    let initial = line(&lists).await.snapshot();

    for preferred in [1u32, 2] {
        let edges: Vec<_> = [(0, preferred), (0, 3 - preferred), (1, 2)]
            .into_iter()
            .chain((3..=12).flat_map(|id| [(0, id), (1, id)]))
            .collect();
        let mut links = Links::new([1, 0]);
        for &(from, to) in &edges {
            links.link(from, to);
        }
        let expected = model(&initial, links.lists(), &edges);
        for parallelism in PARALLELISMS {
            let optimized = Provider::new(Snapshot::clone(&initial));
            stage(&optimized, links.lists(), |region| {
                Linked.reassign(region, &links, parallelism)
            })
            .await
            .unwrap();

            let actual = optimized.snapshot();
            assert_eq!(memberships(&actual), expected, "{parallelism:?}");
            actual.is_consistent().unwrap();
            assert_eq!(actual.lists()[&13].members(), [262]);
            assert!(actual.lists()[&preferred].members().contains(&0));
            assert_eq!(optimized.metrics().create, 0);
            assert_eq!(optimized.metrics().retire, 0);
        }
    }
}

#[tokio::test]
async fn linked_gemm_reports_missing_sources_before_updates() {
    for parallelism in PARALLELISMS {
        let provider = line(&[(0.0, &[0.0])]).await;
        let before = provider.snapshot();
        let mut links = Links::new([0]);
        links.link(0, 99);
        assert_fails(
            stage(&provider, &[0], |region| {
                Linked.reassign(region, &links, parallelism)
            })
            .await,
            "Split could not retrieve list 99 from the region",
        );
        assert_abandoned(&provider, &before);
        assert_eq!(provider.metrics().remove_members, 0);
        assert_eq!(provider.metrics().append_members, 0);
    }
}

#[tokio::test]
async fn reassignment_failure_does_not_publish_staged_changes() {
    for parallelism in PARALLELISMS {
        let seeded = line(&[(0.0, &[0.0, 1.0, 2.0, 3.0, 4.0])]).await;
        let provider =
            Provider::new(Snapshot::clone(&seeded.snapshot())).with_fault(Call::Finish, 0);
        let before = provider.snapshot();
        let mut index = IVFIndex::new(config(2, 2, 2)).unwrap();
        assert_fails(
            insert(&mut index, &provider, &batch(&[(5, 8.0)]), parallelism).await,
            "injected Finish failure",
        );
        assert_abandoned(&provider, &before);
        assert!(provider.metrics().remove_members > 0);
        assert_eq!(provider.metrics().finish, 1);
    }
}
