/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::neighbor::{self, Neighbor};
use diskann_utils::views::rowmajor::{self, Matrix};

/// Compute the ground truth for a small dataset.
///
/// Counter intuitively, this function puts nearest neighbors **at the end** of the
/// vector rather than the beginning.
///
/// This allows filtering by `is_match` to be much more efficient because it decreases
/// the number of elements that have to be moved.
pub fn groundtruth<T, F>(data: rowmajor::Ref<T>, query: &[T], f: F) -> Vec<Neighbor<u32>>
where
    F: Fn(&[T], &[T]) -> f32,
{
    let mut results: Vec<_> = data
        .rows()
        .enumerate()
        .map(|(i, row)| Neighbor::new(i as u32, f(row, query)))
        .collect();

    results.sort_unstable_by(neighbor::ord::reverse(neighbor::ord::fast_distance));
    results
}

/// Decide if `neighbor` is the next nearest neighbor in the groundtruth slice, using
/// the `margin` factor to accommodate ties. If it *is* a match, return the position
/// of the matching entry in `groundtruth`.
///
/// # Panics
///
/// Panics if the `neighbor.id` cannot be found in `groundtruth`.
#[cfg(test)]
pub fn is_match(
    groundtruth: &[Neighbor<u32>],
    neighbor: Neighbor<u32>,
    margin: f32,
) -> Option<usize> {
    for i in (0..groundtruth.len()).rev() {
        // Check if the distance matches.
        let gt = groundtruth[i];
        if (gt.distance() - neighbor.distance()).abs() > margin {
            return None;
        }
        if gt.id() == neighbor.id() {
            return Some(i);
        }
    }
    panic!(
        "could not find neighbor {:?}. Remaining: {:?}",
        neighbor, groundtruth
    );
}

/// Asserts that the top-k results exactly match the ground truth.
///
/// The ground truth must be sorted with nearest neighbors at the end. Result distances
/// must match exactly in nearest-first order, but distinct IDs at equal distances may be
/// permuted, including ties at the top-k cutoff.
pub fn assert_top_k_exactly_match(
    query_id: usize,
    gt: &[Neighbor<u32>],
    ids: &[u32],
    distances: &[f32],
    top_k: usize,
) {
    assert!(
        gt.is_sorted_by(|a, b| a.distance() >= b.distance()),
        "ground truth must be sorted with nearest neighbors at the end"
    );

    for i in 0..top_k {
        let neighbor = gt[gt.len() - 1 - i];
        assert_eq!(
            *neighbor.distance(),
            distances[i],
            "failed on query {} for result {}",
            query_id,
            i
        );
        assert!(
            gt.iter()
                .any(|entry| { *entry.id() == ids[i] && *entry.distance() == distances[i] }),
            "failed on query {} for result {}: ID {} does not match a ground truth neighbor at distance {}",
            query_id,
            i,
            ids[i],
            distances[i]
        );
        assert!(
            !ids[..i].contains(&ids[i]),
            "failed on query {} for result {}: duplicate ID {}",
            query_id,
            i,
            ids[i]
        );
    }
}

/// Asserts that the range results exactly match the ground truth
///
/// For each of the range results, this function verifies that both the distance and ID
/// match exactly with what's expected in the ground truth.
#[cfg(test)]
pub fn assert_range_results_exactly_match(
    query_id: usize,
    gt: &[Neighbor<u32>],
    ids: &[u32],
    radius: f32,
    inner_radius: Option<f32>,
) {
    let gt_ids = if let Some(inner_radius) = inner_radius {
        gt.iter()
            .filter(|nbh| *nbh.distance() >= inner_radius && *nbh.distance() <= radius)
            .map(|nbh| *nbh.id())
            .collect::<Vec<_>>()
    } else {
        gt.iter()
            .filter(|nbh| *nbh.distance() <= radius)
            .map(|nbh| *nbh.id())
            .collect::<Vec<_>>()
    };
    if ids.iter().any(|id| !gt_ids.contains(id)) {
        panic!(
            "query {}: found ids {:?} in range search with radius {}, inner radius {}, but expected {:?}",
            query_id,
            ids,
            radius,
            inner_radius.unwrap_or(f32::MIN),
            gt_ids
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn tied_groundtruth() -> [Neighbor<u32>; 4] {
        [
            Neighbor::new(4, 3.0),
            Neighbor::new(3, 2.0),
            Neighbor::new(2, 2.0),
            Neighbor::new(1, 1.0),
        ]
    }

    #[rstest]
    #[case::ordered(&[1, 2, 3, 4], &[1.0, 2.0, 2.0, 3.0])]
    #[case::permuted_ties(&[1, 3, 2, 4], &[1.0, 2.0, 2.0, 3.0])]
    #[case::tie_at_cutoff(&[1, 3], &[1.0, 2.0])]
    fn top_k_accepts_ties(#[case] ids: &[u32], #[case] distances: &[f32]) {
        assert_top_k_exactly_match(0, &tied_groundtruth(), ids, distances, ids.len());
    }

    #[rstest]
    #[case::duplicate_id(&[1, 2, 2], &[1.0, 2.0, 2.0])]
    #[case::wrong_id_at_distance(&[1, 4], &[1.0, 2.0])]
    #[case::incorrect_distance(&[1], &[1.1])]
    #[case::unsorted_results(&[2, 1], &[2.0, 1.0])]
    #[should_panic(expected = "failed on query 0 for result")]
    fn top_k_rejects_incorrect_results(#[case] ids: &[u32], #[case] distances: &[f32]) {
        assert_top_k_exactly_match(0, &tied_groundtruth(), ids, distances, ids.len());
    }

    #[test]
    #[should_panic(expected = "ground truth must be sorted")]
    fn top_k_rejects_unsorted_groundtruth() {
        let gt = [
            Neighbor::new(4, 3.0),
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 2.0),
        ];
        assert_top_k_exactly_match(0, &gt, &[2], &[2.0], 1);
    }
}
