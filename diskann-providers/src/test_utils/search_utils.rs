/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Test assertion helpers for search result verification.
//!
//! `groundtruth` lives in `diskann::graph::test::search_utils` and is re-exported here.
//! The panicking assertion helpers are duplicated here because they cannot be exported
//! cross-crate from `diskann` (they are `#[cfg(test)]` there).

use diskann::neighbor::Neighbor;

pub use diskann::graph::test::search_utils::groundtruth;

/// Decide if `neighbor` is the next nearest neighbor in the groundtruth slice, using
/// the `margin` factor to accommodate ties. If it *is* a match, return the position
/// of the matching entry in `groundtruth`.
///
/// # Panics
///
/// Panics if `neighbor.id` cannot be found in `groundtruth`.
#[cfg(test)]
pub fn is_match(
    groundtruth: &[Neighbor<u32>],
    neighbor: Neighbor<u32>,
    margin: f32,
) -> Option<usize> {
    for i in (0..groundtruth.len()).rev() {
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

/// Asserts that the top-k results exactly match the ground truth.
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

/// Asserts that the range results exactly match the ground truth.
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
