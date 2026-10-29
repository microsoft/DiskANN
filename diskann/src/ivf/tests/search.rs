/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{future::SendFuture, views::rowmajor::Matrix};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::{
    DefaultContext, ListView, PointRef, Store, StoredList, Strategy, WorkingSet, index, matrix,
    require_send,
};
use crate::{
    ANNError, ANNResult,
    ivf::{SearchProvider, SearchResult},
};

impl SearchProvider for Store {
    fn dim(&self) -> usize {
        self.dim
    }

    fn search_lists<'a>(
        &'a self,
        _: &'a DefaultContext,
        query: &'a [f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<WorkingSet<'a, u32, u32>>> {
        async move {
            if self.fail_fill {
                return Err(ANNError::message("injected search failure"));
            }
            if self.lists.is_empty() {
                return Err(ANNError::message("the IVF index is not initialized"));
            }
            let mut selected: Vec<_> = self
                .lists
                .iter()
                .map(|(&id, list)| {
                    let distance: f32 = SquaredL2::evaluate(query, list.centroid.as_slice());
                    (id, list, distance)
                })
                .collect();
            selected.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
            selected.truncate(nprobe);
            Ok(selected
                .into_iter()
                .map(|(id, list, _)| ListView {
                    id,
                    points: list
                        .members
                        .iter()
                        .map(|&id| PointRef {
                            id,
                            vector: &self.points[&id],
                        })
                        .collect(),
                })
                .collect())
        }
    }
}

fn ids(result: &SearchResult<u32>) -> Vec<u32> {
    result.neighbors.iter().map(|hit| *hit.id()).collect()
}

fn flat_top_k(store: &Store, query: &[f32], k: usize) -> Vec<u32> {
    let mut hits: Vec<_> = store
        .points
        .iter()
        .map(|(&id, point)| {
            let distance: f32 = SquaredL2::evaluate(query, point.as_slice());
            (id, distance)
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    hits.into_iter().take(k).map(|(id, _)| id).collect()
}

#[tokio::test]
async fn search_after_real_splits_matches_flat_without_mutating_the_provider() {
    let mut index = index(
        Store {
            dim: 2,
            next_list: 1,
            ..Store::default()
        },
        4,
    );
    index
        .initialize(&Strategy, &DefaultContext, matrix(&[[0.0, 0.0]]).as_view())
        .await
        .unwrap();
    let base: Vec<_> = (0..48).map(|i| [i as f32, (i % 7) as f32]).collect();
    let points: Vec<_> = base
        .iter()
        .enumerate()
        .map(|(i, row)| (i as u32, row.as_slice()))
        .collect();
    let mut splits = 0;
    for batch in points.chunks(8) {
        splits += index
            .insert_batch(&Strategy, &DefaultContext, batch)
            .await
            .unwrap()
            .splits;
    }
    assert!(splits > 0);
    let before = (index.provider().finishes, index.provider().fills);
    for query in [[0.4, 1.1], [20.0, -0.5], [49.0, 2.1]] {
        let truth = flat_top_k(index.provider(), &query, 3);
        let mut previous = (0, 0);
        let nlists = index.provider().lists.len();
        for nprobe in 1..=nlists {
            let result = require_send(index.search(&DefaultContext, &query, 3, nprobe))
                .await
                .unwrap();
            let found = ids(&result);
            let hits = found.iter().filter(|id| truth.contains(id)).count();
            assert!(result.candidates_scanned >= previous.0 && hits >= previous.1);
            previous = (result.candidates_scanned, hits);
            if nprobe == nlists {
                assert_eq!(found, truth);
                assert_eq!(result.candidates_scanned, base.len());
            }
        }
    }
    assert_eq!((index.provider().finishes, index.provider().fills), before);
}

#[tokio::test]
async fn ties_empty_lists_and_large_k_nprobe() {
    let store = Store {
        dim: 1,
        points: [
            (10_009, vec![1.0]),
            (10_003, vec![-1.0]),
            (10_005, vec![3.0]),
        ]
        .into(),
        lists: [
            (
                10,
                StoredList {
                    centroid: vec![0.0],
                    members: vec![10_009, 10_003],
                },
            ),
            (
                11,
                StoredList {
                    centroid: vec![0.0],
                    members: vec![],
                },
            ),
            (
                12,
                StoredList {
                    centroid: vec![3.0],
                    members: vec![10_005],
                },
            ),
        ]
        .into(),
        ..Store::default()
    };
    let index = index(store, 4);
    let result = index.search(&DefaultContext, &[0.0], 10, 1).await.unwrap();
    assert_eq!(ids(&result), [10_003, 10_009]);
    assert_eq!((result.lists_scanned, result.candidates_scanned), (1, 2));
    let result = index
        .search(&DefaultContext, &[0.0], 10, usize::MAX)
        .await
        .unwrap();
    assert_eq!(ids(&result), [10_003, 10_009, 10_005]);
    assert_eq!((result.lists_scanned, result.candidates_scanned), (3, 3));
}

#[tokio::test]
async fn invalid_queries_uninitialized_and_zero_k() {
    let mut index = index(
        Store {
            dim: 1,
            next_list: 1,
            ..Store::default()
        },
        2,
    );
    assert!(index.search(&DefaultContext, &[0.0], 1, 1).await.is_err());
    let empty = index.search(&DefaultContext, &[0.0], 0, 1).await.unwrap();
    assert!(empty.neighbors.is_empty());
    assert_eq!((empty.lists_scanned, empty.candidates_scanned), (0, 0));
    index
        .initialize(&Strategy, &DefaultContext, matrix(&[[0.0]]).as_view())
        .await
        .unwrap();
    let empty = index.search(&DefaultContext, &[0.0], 1, 1).await.unwrap();
    assert!(empty.neighbors.is_empty());
    assert_eq!((empty.lists_scanned, empty.candidates_scanned), (1, 0));
    for query in [&[][..], &[0.0, 0.0], &[f32::NAN], &[f32::INFINITY]] {
        assert!(index.search(&DefaultContext, query, 1, 1).await.is_err());
    }
    assert!(index.search(&DefaultContext, &[0.0], 1, 0).await.is_err());
}

#[tokio::test]
async fn provider_errors_and_malformed_point_dimensions_are_reported() {
    let mut store = super::seeded();
    store.fail_fill = true;
    let failing = index(store, 2);
    let error = failing
        .search(&DefaultContext, &[0.0, 0.0], 1, 1)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected search failure"));
    assert!(
        failing
            .search(&DefaultContext, &[0.0, 0.0], 0, 1)
            .await
            .is_ok()
    );
    let mut store = super::seeded();
    store.points.insert(10_000, vec![0.0]);
    let malformed = index(store, 2);
    let error = malformed
        .search(&DefaultContext, &[0.0, 0.0], 1, 1)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("invalid IVF stored point dimension")
    );
}
