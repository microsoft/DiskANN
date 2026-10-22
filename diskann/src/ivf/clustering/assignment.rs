/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exhaustive top-1 reassignment across all fitted children.

use diskann_vector::{PureDistanceFunction, distance::SquaredL2};
use diskann_wide::{ARCH, Architecture, SIMDMask, SIMDPartialOrd, SIMDVector};

use super::super::workingset::ListView;

const LANES: usize = 8;

/// Retain each small point tile while streaming through the child centroids.
/// This bounds the point working set when the complete centroid matrix no longer
/// fits beside one point in L1. No point coordinates are copied.
pub(super) fn assign<I, L>(
    lists: &[ListView<'_, I, L>],
    centers: &[f32],
    dim: usize,
) -> Vec<usize> {
    let mut assignments = Vec::with_capacity(lists.iter().map(|list| list.points.len()).sum());
    let count = centers.len() / dim;
    // Small centroid matrices already fit in cache, so avoid the extra point
    // bookkeeping unless the matrix is at least 32 KiB.
    if count < 16 || centers.len() < 8192 {
        let mut scratch = vec![0.0; count];
        for point in lists.iter().flat_map(|list| &list.points) {
            assignments.push(nearest(point.vector, centers, dim, &mut scratch));
        }
        return assignments;
    }
    for list in lists {
        for points in list.points.chunks(4) {
            let mut best = [Top1::new(), Top1::new(), Top1::new(), Top1::new()];
            for (index, center) in centers.chunks_exact(dim).enumerate() {
                let (pairs, tail) = points.as_chunks::<2>();
                for (pair, points) in pairs.iter().enumerate() {
                    let mut distances =
                        SquaredL2::evaluate_pair(center, points[0].vector, points[1].vector);
                    // Finite squared differences are symmetric, including
                    // overflow to infinity. Preserve the original operand order
                    // for NaNs, whose sign/payload can affect total ordering.
                    for (distance, point) in distances.iter_mut().zip(points) {
                        if distance.is_nan() {
                            *distance = SquaredL2::evaluate(point.vector, center);
                        }
                    }
                    best[2 * pair].offer_each(&distances[..1], index);
                    best[2 * pair + 1].offer_each(&distances[1..], index);
                }
                if let Some(point) = tail.first() {
                    let distance: f32 = SquaredL2::evaluate(point.vector, center);
                    best[points.len() - 1].offer_each(&[distance], index);
                }
            }
            assignments.extend(best[..points.len()].iter().map(|best| best.index));
        }
    }
    assignments
}

/// Prefer the first centroid on ties, including infinite distances. The caller
/// supplies nonempty, complete rows of positive dimension, matching `point`.
/// The distance scratch has one element per centroid and is reused across points.
pub(super) fn nearest(point: &[f32], centers: &[f32], dim: usize, distances: &mut [f32]) -> usize {
    debug_assert_eq!(distances.len(), centers.len() / dim);
    // The common one-parent split only needs the paired kernel, with no distance
    // row stores or SIMD ranking setup. The same holds for other short rows.
    if distances.len() < LANES {
        let mut best = Top1::new();
        let mut pairs = centers.chunks_exact(2 * dim);
        for (i, centers) in pairs.by_ref().enumerate() {
            best.offer_each(
                &SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]),
                2 * i,
            );
        }
        if !pairs.remainder().is_empty() {
            best.offer_each(
                &[SquaredL2::evaluate(point, pairs.remainder())],
                centers.len() / dim - 1,
            );
        }
        return best.index;
    }
    let mut pairs = centers.chunks_exact(2 * dim);
    for (pair, centers) in distances
        .as_chunks_mut::<2>()
        .0
        .iter_mut()
        .zip(pairs.by_ref())
    {
        *pair = SquaredL2::evaluate_pair(point, &centers[..dim], &centers[dim..]);
    }
    // Splits currently produce an even number of children, but keep the kernel
    // correct for any nonempty centroid set without padding or dropping a row.
    if !pairs.remainder().is_empty() {
        distances[centers.len() / dim - 1] = SquaredL2::evaluate(point, pairs.remainder());
    }
    let mut best = Top1::new();
    let (groups, tail) = distances.as_chunks::<LANES>();
    for (group, distances) in groups.iter().enumerate() {
        best.offer_group(ARCH, distances, group * LANES);
    }
    best.offer_each(tail, groups.len() * LANES);
    best.index
}

struct Top1 {
    index: usize,
    key: i32,
}

/// Signed integer ordering of these keys is exactly `f32::total_cmp`, including
/// signed zeros, infinities, and NaNs. Float SIMD `<` would change that ordering.
#[inline(always)]
fn key(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    bits ^ ((bits >> 31) & i32::MAX)
}

impl Top1 {
    fn new() -> Self {
        // The largest total-order key is a positive NaN, not +infinity. If every
        // input has this key, index zero still implements the first-on-ties rule.
        Self {
            index: 0,
            key: i32::MAX,
        }
    }

    #[inline(always)]
    fn offer_each(&mut self, distances: &[f32], first: usize) {
        for (lane, &distance) in distances.iter().enumerate() {
            let key = key(distance);
            if key < self.key {
                self.index = first + lane;
                self.key = key;
            }
        }
    }

    #[inline(always)]
    fn offer_group<A: Architecture>(&mut self, arch: A, distances: &[f32; LANES], first: usize) {
        if A::i32x8::EMULATED {
            self.offer_each(distances, first);
            return;
        }
        let bits = A::i32x8::from_array(arch, distances.map(|x| x.to_bits() as i32));
        let keys = bits ^ ((bits >> 31) & A::i32x8::splat(arch, i32::MAX));
        let eligible = keys.lt_simd(A::i32x8::splat(arch, self.key));
        if let Some(lane) = eligible.first() {
            // Earlier lanes cannot improve the incumbent. Scan the rest in
            // ordinal order, checking against the progressively smaller key.
            self.offer_each(&distances[lane..], first + lane);
        }
    }
}

#[cfg(test)]
mod tests;
