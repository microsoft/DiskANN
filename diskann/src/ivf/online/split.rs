/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Fitting the two centroids that replace a split list.

use diskann_utils::views::{Matrix, MatrixView};
use rand::{Rng, rngs::StdRng};

use super::{
    index_error,
    kernels::{LloydScratch, lloyd, nearest},
};
use crate::ANNResult;

/// A list's two replacement centroids and the child each of its points joins.
#[derive(Debug)]
pub(in crate::ivf) struct TwoMeans {
    /// Row `c` is the centroid of child `c`.
    pub(in crate::ivf) centroids: Matrix<f32>,
    /// `children[i]` is the child row `i` of the clustered points joins.
    pub(in crate::ivf) children: Vec<usize>,
}

/// Cluster `points` into two groups, seeding Lloyd's algorithm with two distinct
/// points.
///
/// # Errors
///
/// Fails if `points` holds fewer than two rows.
pub(in crate::ivf) fn two_means(
    points: MatrixView<'_, f32>,
    iterations: usize,
    rng: &mut StdRng,
) -> ANNResult<TwoMeans> {
    let (count, dim) = (points.nrows(), points.ncols());
    if count < 2 {
        return Err(index_error(format!(
            "a split needs at least two points, got {count}"
        )));
    }

    let first = rng.random_range(0..count);
    let second = (first + rng.random_range(1..count)) % count;
    let mut centroids = Matrix::new(0.0f32, 2, dim);
    centroids.row_mut(0).copy_from_slice(points.row(first));
    centroids.row_mut(1).copy_from_slice(points.row(second));

    lloyd(
        || points.row_iter(),
        centroids.as_mut_slice(),
        dim,
        iterations.max(1),
        &mut LloydScratch::default(),
    );

    let children = {
        let candidates = [(0, centroids.row(0)), (1, centroids.row(1))];
        points
            .row_iter()
            // There are always two candidates, so every point joins one of them.
            .map(|point| nearest(point, &candidates).unwrap_or(0))
            .collect()
    };

    Ok(TwoMeans {
        centroids,
        children,
    })
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn matrix<const D: usize>(rows: &[[f32; D]]) -> Matrix<f32> {
        let data: Box<[f32]> = rows.iter().flatten().copied().collect();
        Matrix::try_from(data, rows.len(), D).unwrap()
    }

    #[test]
    fn two_means_separates_distant_clusters() {
        let points = matrix(&[[0.0, 0.0], [10.0, 0.0], [0.0, 1.0], [10.0, 1.0], [1.0, 0.0]]);
        let fit = two_means(points.as_view(), 10, &mut StdRng::seed_from_u64(7)).unwrap();

        assert_eq!(fit.centroids.nrows(), 2);
        assert_eq!(fit.children.len(), 5);
        // Rows 0, 2 and 4 are near the origin; rows 1 and 3 are near x = 10.
        let low = fit.children[0];
        assert_eq!(fit.children, vec![low, 1 - low, low, 1 - low, low]);
        assert_eq!(fit.centroids.row(low), &[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(fit.centroids.row(1 - low), &[10.0, 0.5]);
    }

    #[test]
    fn two_means_needs_two_points() {
        let points = matrix(&[[1.0]]);
        let err = two_means(points.as_view(), 1, &mut StdRng::seed_from_u64(7)).unwrap_err();
        assert!(err.to_string().contains("at least two points"), "{err}");
    }
}
