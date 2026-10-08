/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical fitting and assignment over borrowed matrices.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::{Rng, rngs::StdRng};

use super::{
    index_error,
    kernels::{LloydScratch, lloyd, nearest},
};
use crate::ANNResult;

/// Fit two children directly into `centroids`, using distinct random seeds.
///
/// `centroids` must have two rows and the same dimension as `points`.
/// Zero iterations still performs one Lloyd step.
///
/// # Errors
///
/// Returns an error when fewer than two training points are supplied.
pub(in crate::ivf) fn fit_two_means(
    points: rowmajor::Ref<'_, f32>,
    mut centroids: rowmajor::Mut<'_, f32>,
    iterations: usize,
    rng: &mut StdRng,
    scratch: &mut LloydScratch,
) -> ANNResult<()> {
    let count = points.nrows();
    if count < 2 {
        return Err(index_error(format!(
            "a split needs at least two points, got {count}"
        )));
    }

    let first = rng.random_range(0..count);
    let second = (first + rng.random_range(1..count)) % count;
    centroids.row_mut(0).copy_from_slice(points.row(first));
    centroids.row_mut(1).copy_from_slice(points.row(second));
    lloyd(
        || points.rows(),
        centroids.as_mut_slice(),
        points.ncols(),
        iterations.max(1),
        scratch,
    );
    Ok(())
}

/// Assign by squared Euclidean distance, preferring the earliest centroid on ties.
///
/// # Errors
///
/// Returns an error when there are no centroids or the dimensions differ.
pub(in crate::ivf) fn assign_nearest(
    points: rowmajor::Ref<'_, f32>,
    centroids: rowmajor::Ref<'_, f32>,
) -> ANNResult<Box<[usize]>> {
    if centroids.nrows() == 0 {
        return Err(index_error("assignment needs at least one centroid"));
    }
    if points.ncols() != centroids.ncols() {
        return Err(index_error(format!(
            "assignment points have dimension {}, but centroids have dimension {}",
            points.ncols(),
            centroids.ncols()
        )));
    }

    let candidates: Vec<_> = centroids.rows().enumerate().collect();
    let mut assigned = Vec::with_capacity(points.nrows());
    for point in points.rows() {
        assigned.push(
            nearest(point, &candidates)
                .ok_or_else(|| index_error("assignment needs at least one centroid"))?,
        );
    }
    Ok(assigned.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    fn matrix<const D: usize>(rows: &[[f32; D]]) -> rowmajor::Owned<f32> {
        let data: Box<[f32]> = rows.iter().flatten().copied().collect();
        rowmajor::Owned::try_from_data(data, rows.len(), D).unwrap()
    }

    fn fit(
        points: rowmajor::Ref<'_, f32>,
        iterations: usize,
        rng: &mut StdRng,
    ) -> ANNResult<rowmajor::Owned<f32>> {
        let mut centroids = rowmajor::Owned::from_element(2, points.ncols(), 0.0);
        fit_two_means(
            points,
            centroids.as_view_mut(),
            iterations,
            rng,
            &mut LloydScratch::default(),
        )?;
        Ok(centroids)
    }

    #[test]
    fn two_means_separates_distant_clusters() {
        let points = matrix(&[[0.0, 0.0], [10.0, 0.0], [0.0, 1.0], [10.0, 1.0], [1.0, 0.0]]);
        let centroids = fit(points.as_view(), 10, &mut StdRng::seed_from_u64(7)).unwrap();
        let children = assign_nearest(points.as_view(), centroids.as_view()).unwrap();

        let low = children[0];
        assert_eq!(children.as_ref(), &[low, 1 - low, low, 1 - low, low]);
        assert_eq!(centroids.row(low), &[1.0 / 3.0, 1.0 / 3.0]);
        assert_eq!(centroids.row(1 - low), &[10.0, 0.5]);
    }

    #[test]
    fn two_means_needs_two_points() {
        for count in [0, 1] {
            let points = rowmajor::Owned::from_element(count, 1, 1.0f32);
            let err = fit(points.as_view(), 1, &mut StdRng::seed_from_u64(7)).unwrap_err();
            assert!(err.to_string().contains("at least two points"), "{err}");
        }
    }

    #[test]
    fn two_means_preserves_seed_order_and_rng_consumption() {
        let points = matrix(&[[0.0], [10.0]]);
        let mut rng = StdRng::seed_from_u64(7);
        let mut expected_rng = rng.clone();
        let first = expected_rng.random_range(0..points.nrows());
        let second = (first + expected_rng.random_range(1..points.nrows())) % points.nrows();
        let centroids = fit(points.as_view(), 10, &mut rng).unwrap();

        assert_eq!(centroids.row(0), points.row(first));
        assert_eq!(centroids.row(1), points.row(second));
        assert_eq!(rng.random::<u64>(), expected_rng.random::<u64>());
    }

    #[test]
    fn two_means_zero_iterations_still_refines_centroids() {
        let points = matrix(&[[0.0], [2.0], [10.0], [12.0]]);
        let zero = fit(points.as_view(), 0, &mut StdRng::seed_from_u64(7)).unwrap();
        let one = fit(points.as_view(), 1, &mut StdRng::seed_from_u64(7)).unwrap();
        assert_eq!(zero.as_slice(), one.as_slice());
    }

    #[test]
    fn two_means_writes_only_the_output_subview_and_reuses_scratch() {
        let points = matrix(&[[0.0], [10.0]]);
        let mut output = rowmajor::Owned::from_element(6, 1, -1.0);
        let mut scratch = LloydScratch::default();
        let mut rng = StdRng::seed_from_u64(7);
        for rows in [1..3, 3..5] {
            let out = rowmajor::Mut::try_from_data(&mut output.as_mut_slice()[rows], 2, 1).unwrap();
            fit_two_means(points.as_view(), out, 10, &mut rng, &mut scratch).unwrap();
        }
        assert_eq!(output.row(0), &[-1.0]);
        assert_eq!(output.row(5), &[-1.0]);
        for rows in [1..3, 3..5] {
            let assigned = assign_nearest(points.as_view(), output.subview(rows).unwrap()).unwrap();
            assert_ne!(assigned[0], assigned[1]);
        }
    }

    #[test]
    fn nearest_centroid_assigns_points_to_an_independent_candidate_set() {
        let points = matrix(&[[-1.0], [12.0], [19.0]]);
        let centroids = matrix(&[[0.0], [10.0], [20.0]]);
        let children = assign_nearest(points.as_view(), centroids.as_view()).unwrap();
        assert_eq!(children.as_ref(), &[0, 1, 2]);
    }

    #[test]
    fn nearest_centroid_prefers_the_earliest_row_on_ties() {
        let points = matrix(&[[1.8, 0.1], [1.0, 1.0]]);
        let centroids = matrix(&[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
        let children = assign_nearest(points.as_view(), centroids.as_view()).unwrap();
        assert_eq!(children.as_ref(), &[1, 0]);
    }

    #[test]
    fn nearest_centroid_supports_one_candidate() {
        let points = matrix(&[[-1.0], [12.0], [19.0]]);
        let centroids = matrix(&[[10.0]]);
        let children = assign_nearest(points.as_view(), centroids.as_view()).unwrap();
        assert_eq!(children.as_ref(), &[0, 0, 0]);
    }

    #[test]
    fn nearest_centroid_supports_an_empty_point_batch() {
        let points = matrix::<1>(&[]);
        let centroids = matrix(&[[10.0]]);
        assert!(
            assign_nearest(points.as_view(), centroids.as_view())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn nearest_centroid_needs_a_candidate() {
        let points = matrix(&[[1.0]]);
        let centroids = matrix::<1>(&[]);
        let err = assign_nearest(points.as_view(), centroids.as_view()).unwrap_err();
        assert!(err.to_string().contains("at least one centroid"), "{err}");
    }

    #[test]
    fn nearest_centroid_requires_matching_dimensions() {
        let points = matrix(&[[1.0]]);
        let centroids = matrix(&[[0.0, 0.0]]);
        let err = assign_nearest(points.as_view(), centroids.as_view()).unwrap_err();
        assert!(
            err.to_string()
                .contains("points have dimension 1, but centroids have dimension 2"),
            "{err}"
        );
    }
}
