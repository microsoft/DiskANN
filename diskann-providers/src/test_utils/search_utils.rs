/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::neighbor::Neighbor;
use diskann_utils::views::MatrixView;

/// Compute the ground truth for a small dataset.
///
/// Counter intuitively, this function puts nearest neighbors **at the end** of the
/// vector rather than the beginning.
///
/// This allows filtering by `is_match` to be much more efficient because it decreases
/// the number of elements that have to be moved.
pub fn groundtruth<T, F>(data: MatrixView<T>, query: &[T], f: F) -> Vec<Neighbor<u32>>
where
    F: Fn(&[T], &[T]) -> f32,
{
    let mut results: Vec<_> = data
        .row_iter()
        .enumerate()
        .map(|(i, row)| Neighbor::new(i as u32, f(row, query)))
        .collect();

    results.sort_unstable_by(|a, b| a.cmp(b).reverse());
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
        if (gt.distance - neighbor.distance).abs() > margin {
            return None;
        }
        if gt.id == neighbor.id {
            return Some(i);
        }
    }
    panic!(
        "could not find neighbor {:?}. Remaining: {:?}",
        neighbor, groundtruth
    );
}

/// Asserts that the top-k results exactly match the ground truth
///
/// For each of the top-k results, this function verifies that both the distance and ID
/// match exactly with what's expected in the ground truth.
pub fn assert_top_k_exactly_match(
    query_id: usize,
    gt: &[Neighbor<u32>],
    ids: &[u32],
    distances: &[f32],
    top_k: usize,
) {
    for i in 0..top_k {
        let neighbor = gt[gt.len() - 1 - i];
        assert_eq!(
            neighbor.distance, distances[i],
            "failed on query {} for result {}",
            query_id, i
        );
        assert_eq!(
            neighbor.id, ids[i],
            "failed on query {} for result {}",
            query_id, i
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
            .filter(|nbh| nbh.distance >= inner_radius && nbh.distance <= radius)
            .map(|nbh| nbh.id)
            .collect::<Vec<_>>()
    } else {
        gt.iter()
            .filter(|nbh| nbh.distance <= radius)
            .map(|nbh| nbh.id)
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
