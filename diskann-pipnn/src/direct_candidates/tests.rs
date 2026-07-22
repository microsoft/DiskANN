/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::DirectCandidates;

#[test]
fn deduplicates_candidates_while_accumulating_leaves() {
    let candidates = DirectCandidates::new(3).unwrap();
    let point_ids = [0, 1, 2];
    let offsets = [0, 2, 4, 6];
    let edges = [(1, 1.0), (2, 2.0), (0, 1.0), (2, 1.5), (0, 2.0), (1, 1.5)];

    candidates.add_leaf_edges(&point_ids, &offsets, &edges);
    candidates.add_leaf_edges(&point_ids, &offsets, &edges);

    assert_eq!(
        candidates.into_rows().unwrap(),
        [vec![1, 2], vec![0, 2], vec![0, 1]]
    );
}
