/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Distance and clustering kernels for split planning.
//!
//! Clustering and placement use squared Euclidean distance regardless of the search
//! metric.

use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

/// Squared Euclidean distance between two vectors of equal length.
pub(super) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    SquaredL2::evaluate(a, b)
}

/// The candidate nearest to `point`, preferring the earliest candidate on ties.
pub(super) fn nearest<L: Copy>(point: &[f32], candidates: &[(L, &[f32])]) -> Option<L> {
    candidates
        .iter()
        .map(|&(list, centroid)| (list, squared_l2(point, centroid)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(list, _)| list)
}

/// Scratch buffers for [`lloyd`], reusable across calls.
#[derive(Debug, Default)]
pub(in crate::ivf) struct LloydScratch {
    sums: Vec<f32>,
    counts: Vec<usize>,
}

/// Refine `centers`, stored row-major with `dim` columns, with `iterations` Lloyd
/// steps over the points yielded by `points`.
///
/// Each point joins its nearest center, preferring the earliest on ties. A center that
/// attracts no points keeps its position.
pub(super) fn lloyd<'p, P, I>(
    points: P,
    centers: &mut [f32],
    dim: usize,
    iterations: usize,
    scratch: &mut LloydScratch,
) where
    P: Fn() -> I,
    I: Iterator<Item = &'p [f32]>,
{
    if dim == 0 || centers.is_empty() {
        return;
    }
    let LloydScratch { sums, counts } = scratch;
    for _ in 0..iterations {
        sums.clear();
        sums.resize(centers.len(), 0.0);
        counts.clear();
        counts.resize(centers.len() / dim, 0);

        for point in points() {
            let Some(center) = centers
                .chunks_exact(dim)
                .map(|center| squared_l2(point, center))
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(center, _)| center)
            else {
                continue;
            };
            counts[center] += 1;
            for (sum, x) in sums[center * dim..(center + 1) * dim].iter_mut().zip(point) {
                *sum += x;
            }
        }

        for ((center, sum), &count) in centers
            .chunks_exact_mut(dim)
            .zip(sums.chunks_exact(dim))
            .zip(counts.iter())
        {
            if count > 0 {
                let scale = 1.0 / count as f32;
                for (x, s) in center.iter_mut().zip(sum) {
                    *x = s * scale;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_prefers_the_earliest_tie() {
        let (a, b, c) = ([0.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        let candidates = [(7u32, &a[..]), (8, &b[..]), (9, &c[..])];
        assert_eq!(nearest(&[1.8, 0.1], &candidates), Some(8));
        // Equidistant from 8 and 9.
        assert_eq!(nearest(&[1.0, 1.0], &candidates[1..]), Some(8));
        assert_eq!(nearest::<u32>(&[1.0, 1.0], &[]), None);
    }

    #[test]
    fn lloyd_moves_centers_to_cluster_means() {
        let points: Vec<[f32; 2]> = vec![[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]];
        let mut centers = [1.0, 1.0, 9.0, 1.0];
        let mut scratch = LloydScratch::default();
        lloyd(
            || points.iter().map(|p| &p[..]),
            &mut centers,
            2,
            3,
            &mut scratch,
        );
        assert_eq!(centers, [0.0, 1.0, 10.0, 1.0]);
    }

    #[test]
    fn lloyd_keeps_centers_without_points() {
        let points: Vec<[f32; 1]> = vec![[0.0], [1.0]];
        let mut centers = [0.5, 100.0];
        lloyd(
            || points.iter().map(|p| &p[..]),
            &mut centers,
            1,
            2,
            &mut LloydScratch::default(),
        );
        assert_eq!(centers, [0.5, 100.0]);
    }
}
