/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Region-wide reassignment over all fitted child centroids.

use diskann_vector::distance::SquaredL2;
#[cfg(feature = "ivf-parallel")]
use rayon::prelude::*;

#[cfg(feature = "ivf-parallel")]
use super::use_parallel;
use crate::{
    ANNResult,
    ivf::{metrics::ComputeStats, workingset::ListView},
};

#[cfg(feature = "ivf-experimental-compute")]
mod gemm;

/// Region assignment policy. GEMM can change partitions through rounding.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ReassignmentMode {
    #[default]
    Baseline,
    Gemm,
}

pub(super) fn assign<I: Sync, L>(
    lists: &[ListView<'_, I, L>],
    centers: &[f32],
    dim: usize,
    mode: ReassignmentMode,
    assignments: &mut Vec<usize>,
    sizes: &mut [usize],
) -> ANNResult<ComputeStats> {
    #[cfg(not(feature = "ivf-experimental-compute"))]
    let _ = mode;
    #[cfg(feature = "ivf-experimental-compute")]
    if mode == ReassignmentMode::Gemm && sizes.len() >= gemm::GEMM_MIN_CENTERS {
        let mut stats = ComputeStats::default();
        let count = lists.iter().map(|list| list.points.len()).sum();
        gemm::assign_gemm(
            lists
                .iter()
                .flat_map(|list| list.points.iter().map(|point| point.vector)),
            centers,
            dim,
            count,
            assignments,
            &mut stats,
        )?;
        for &to in assignments.iter() {
            sizes[to] += 1;
        }
        return Ok(stats);
    }
    assign_direct(lists, centers, dim, assignments, sizes);
    Ok(ComputeStats::default())
}

fn assign_direct<I: Sync, L>(
    lists: &[ListView<'_, I, L>],
    centers: &[f32],
    dim: usize,
    assignments: &mut Vec<usize>,
    sizes: &mut [usize],
) {
    #[cfg(feature = "ivf-parallel")]
    {
        let count: usize = lists.iter().map(|list| list.points.len()).sum();
        if use_parallel(count.saturating_mul(dim).saturating_mul(sizes.len())) {
            assignments.resize(count, 0);
            assign_parallel(lists, centers, dim, assignments);
            for &to in assignments.iter() {
                sizes[to] += 1;
            }
            return;
        }
    }
    for point in lists.iter().flat_map(|list| &list.points) {
        let to = nearest(point.vector, centers, dim);
        assignments.push(to);
        sizes[to] += 1;
    }
}

#[cfg(feature = "ivf-parallel")]
fn assign_parallel<I: Sync, L>(
    lists: &[ListView<'_, I, L>],
    centers: &[f32],
    dim: usize,
    assignments: &mut [usize],
) {
    // Borrow point rows and disjoint output chunks; even one parent can use the pool.
    let mut remaining = assignments;
    let mut tasks = Vec::new();
    for list in lists {
        let (labels, rest) = remaining.split_at_mut(list.points.len());
        remaining = rest;
        tasks.extend(list.points.chunks(256).zip(labels.chunks_mut(256)));
    }
    tasks.into_par_iter().for_each(|(points, labels)| {
        for (point, label) in points.iter().zip(labels) {
            *label = nearest(point.vector, centers, dim);
        }
    });
}

/// Prefer the first centroid on ties; each split parent supplies two centers.
#[inline(always)]
pub(super) fn nearest(point: &[f32], centers: &[f32], dim: usize) -> usize {
    debug_assert!(dim > 0 && centers.len() >= 2 * dim);
    debug_assert_eq!(centers.len() % (2 * dim), 0);
    let (first, remaining) = centers.split_at(2 * dim);
    let distances = SquaredL2::evaluate_batch(point, [&first[..dim], &first[dim..]]);
    let mut best = 0;
    let mut best_distance = distances[0];
    if distances[1].total_cmp(&best_distance).is_lt() {
        best = 1;
        best_distance = distances[1];
    }
    for (pair, centers) in remaining.chunks_exact(2 * dim).enumerate() {
        // Keep candidate order unchanged, including ties and infinite distances.
        let distances = SquaredL2::evaluate_batch(point, [&centers[..dim], &centers[dim..]]);
        for (offset, distance) in distances.into_iter().enumerate() {
            if distance.total_cmp(&best_distance).is_lt() {
                best = 2 * (pair + 1) + offset;
                best_distance = distance;
            }
        }
    }
    best
}
