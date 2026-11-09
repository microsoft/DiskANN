/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;
use crate::ivf::{ReassignmentMode, split::policies::reassign::Gemm};

#[test]
fn linked_gemm_preserves_membership_order_empty_lists_and_candidate_boundaries() {
    let first: Vec<_> = (0..133).map(|i| [[-2.0], [5.0], [7.0]][i % 3]).collect();
    let second: Vec<_> = (0..129)
        .map(|i| if i % 2 == 0 { [5.0] } else { [-1.0] })
        .collect();
    let mut sources = BTreeMap::from([
        (
            1,
            Source {
                centroid: vec![5.0],
                members: (0..133).collect(),
                vectors: matrix(&first),
            },
        ),
        (
            2,
            Source {
                centroid: vec![-1.0],
                members: (1000..1129).collect(),
                vectors: matrix(&second),
            },
        ),
        (
            3,
            Source {
                centroid: vec![-1.0],
                members: Vec::new(),
                vectors: matrix::<1>(&[]),
            },
        ),
        // This is the global nearest centroid for some points, but is not linked.
        (
            99,
            Source {
                centroid: vec![-2.0],
                members: vec![99_000],
                vectors: matrix(&[[-2.0]]),
            },
        ),
    ]);
    for id in 4..=13 {
        sources.insert(
            id,
            Source {
                centroid: vec![100.0 + id as f32],
                members: Vec::new(),
                vectors: matrix::<1>(&[]),
            },
        );
    }
    for preferred in [2, 3] {
        let mut pairs = vec![(1, preferred), (1, 5 - preferred), (2, 3)];
        pairs.extend((4..=13).flat_map(|id| [(1, id), (2, id)]));
        let links = support::links(&pairs);
        let run = |gemm: bool| {
            let region = Region::new(&sources);
            let mut memberships = PendingLists::default();
            let mut created = Vec::new();
            {
                let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
                if gemm {
                    Gemm.reassign(&region, &mut update, &links).unwrap();
                } else {
                    Linked::new(NearestCentroid)
                        .reassign(&region, &mut update, &links)
                        .unwrap();
                }
            }
            assert!(created.is_empty());
            sources
                .iter()
                .map(|(&id, source)| {
                    (
                        id,
                        memberships
                            .members(id, &source.members)
                            .unwrap()
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };
        let expected = run(false);
        let actual = run(true);
        assert_eq!(actual, expected);
        assert_eq!(actual[&99], [99_000]);
        assert!(actual[&preferred].contains(&0));
        let all: Vec<_> = actual.values().flatten().copied().collect();
        assert_eq!(all.len(), 133 + 129 + 1);
        assert_eq!(
            all.iter().copied().collect::<BTreeSet<_>>().len(),
            all.len()
        );
    }
}

#[test]
fn linked_gemm_reports_missing_sources_before_updates() {
    let sources = BTreeMap::from([(
        1,
        Source {
            centroid: vec![0.0],
            members: vec![0],
            vectors: matrix(&[[0.0]]),
        },
    )]);
    let region = Region::new(&sources);
    let mut memberships = PendingLists::default();
    let mut created = Vec::new();
    let mut update = support::Update::new(&sources, 1, &mut memberships, &mut created);
    let error = Gemm
        .reassign(&region, &mut update, &support::links(&[(1, 99)]))
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<SplitError<u32>>(),
        Some(SplitError::UnableToRetrieve(99))
    ));
    assert!(created.is_empty());
    assert_eq!(memberships.into_changes().count(), 0);
}

#[tokio::test]
async fn gemm_build_failure_does_not_publish_staged_changes() {
    let (index, strategy) = index(seeded(), 2);
    let mut index = index.with_reassignment_mode(ReassignmentMode::Gemm);
    let snapshot = |store: &Store| {
        store
            .lists
            .iter()
            .map(|(&id, list)| {
                (
                    id,
                    list.centroid
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    list.members.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    let before = snapshot(&strategy.store());
    strategy.store().fail_finish = true;
    let points: [(u32, &[f32]); 1] = [(5, &[8.0, 0.0])];
    assert!(
        require_send(index.insert_batch(&strategy, &DefaultContext, &points[..]))
            .await
            .is_err()
    );
    let store = strategy.store();
    assert_eq!(snapshot(&store), before);
    assert_eq!(store.points.len(), 5);
    assert_eq!(store.finishes, 0);
}
