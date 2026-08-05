/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::graph::{
    AdjacencyList,
    config::{self, MaxDegree},
};
use diskann_utils::views::MatrixView;

use super::*;

fn graph_config(degree: usize) -> Config {
    config::Builder::new_with(
        degree,
        MaxDegree::same(),
        degree,
        Metric::L2.into(),
        |builder| {
            builder.alpha(1.2);
        },
    )
    .build()
    .unwrap()
}

fn candidate_list(ids: impl IntoIterator<Item = u32>) -> AdjacencyList<u32> {
    AdjacencyList::from_iter_untrusted(ids)
}

#[test]
fn preserves_lists_within_the_degree_bound() {
    let data = [0.0_f32, 1.0, 2.0, 3.0];
    let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
    let candidates = vec![
        candidate_list([3, 1]),
        candidate_list([]),
        candidate_list([]),
        candidate_list([]),
    ];

    let actual = prune_overfull(data, candidates, &graph_config(2), Metric::L2).unwrap();

    assert_eq!(&*actual[0], &[1, 3]);
}

#[test]
fn prunes_an_overfull_list_with_the_vamana_kernel() {
    let data = [0.0_f32, 1.0, 2.0, -3.0];
    let data = MatrixView::try_from(&data[..], 4, 1).unwrap();
    let candidates = vec![
        candidate_list([3, 2, 1]),
        candidate_list([]),
        candidate_list([]),
        candidate_list([]),
    ];

    let actual = prune_overfull(data, candidates, &graph_config(2), Metric::L2).unwrap();

    assert_eq!(&*actual[0], &[1, 3]);
}

#[test]
fn rejects_invalid_candidate_ids_without_panicking() {
    let data = [0.0_f32, 1.0, 2.0];
    let data = MatrixView::try_from(&data[..], 3, 1).unwrap();
    let candidates = vec![
        candidate_list([1, 3]),
        candidate_list([]),
        candidate_list([]),
    ];

    let error = prune_overfull(data, candidates, &graph_config(1), Metric::L2).unwrap_err();

    assert!(matches!(
        error.downcast_ref::<FinalizationError>(),
        Some(FinalizationError::InvalidCandidateId {
            source_index: 0,
            candidate: 3,
            points: 3,
        })
    ));
}

#[test]
fn rejects_candidate_list_count_mismatch_without_panicking() {
    let data = [0.0_f32, 1.0, 2.0];
    let data = MatrixView::try_from(&data[..], 3, 1).unwrap();
    let candidates = vec![
        candidate_list([]),
        candidate_list([]),
        candidate_list([]),
        candidate_list([]),
    ];

    let error = prune_overfull(data, candidates, &graph_config(1), Metric::L2).unwrap_err();

    assert!(matches!(
        error.downcast_ref::<FinalizationError>(),
        Some(FinalizationError::CandidateListCountMismatch {
            lists: 4,
            points: 3
        })
    ));
}
