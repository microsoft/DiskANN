/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Nearest-center scoring with bounded, sequential SGEMM tiles.
//! Uncertain scores are rechecked with direct L2; finite rounding can still change labels.

use std::num::NonZeroUsize;

use diskann_linalg::{MatrixName, SgemmError, Transpose, sgemm};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

const TILE_ROWS: NonZeroUsize = NonZeroUsize::new(128).unwrap();

/// Squared center norm for uncentered scoring. Reuse only while the center is unchanged.
pub fn center_norm(center: &[f32]) -> f64 {
    center.iter().map(|&x| f64::from(x) * f64::from(x)).sum()
}

/// Choose the first nearest center using 128-row SGEMM tiles and cached center norms.
/// Norms must be computed with [center_norm] in the same order as the center rows.
/// Points are borrowed without translation; no worker pool is created.
///
/// # Errors
/// Returns an error if a score tile's dimensions overflow.
///
/// # Panics
/// Panics for mismatched dimensions, label/norm counts, no centers, or more than
/// u32::MAX centers.
pub fn assign(
    points: rowmajor::Ref<'_, f32>,
    centers: rowmajor::Ref<'_, f32>,
    center_norms: &[f64],
    labels: &mut [u32],
) -> Result<(), SgemmError> {
    assert_eq!(points.ncols(), centers.ncols());
    assert_eq!(points.nrows(), labels.len());
    assert_eq!(centers.nrows(), center_norms.len());
    assert!(centers.nrows() > 0 && u32::try_from(centers.nrows()).is_ok());
    if centers.nrows() == 1 {
        labels.fill(0);
        return Ok(());
    }

    let point_norms: Vec<_> = points.rows().map(super::square_norm).collect();
    let dim = points.ncols();
    let nu = (dim as f64 + 8.0) * (f64::from(f32::EPSILON) * 0.5);
    let gamma = nu / (1.0 - nu);
    if !center_norms.iter().all(|n| n.is_finite()) || !(0.0..0.5).contains(&gamma) {
        for (point, label) in points.rows().zip(labels) {
            *label = nearest(point, centers);
        }
        return Ok(());
    }

    let max_norm = center_norms.iter().copied().fold(0.0f64, f64::max);
    let k = centers.nrows();
    let mut scores = Vec::new();
    for ((tile, output), norms) in points
        .window_iter(TILE_ROWS)
        .zip(labels.chunks_mut(TILE_ROWS.get()))
        .zip(point_norms.chunks(TILE_ROWS.get()))
    {
        let m = tile.nrows();
        let score_len = m.checked_mul(k).ok_or(SgemmError::DimensionOverflow {
            matrix_name: MatrixName::C,
            rows: m,
            cols: k,
        })?;
        scores.resize(score_len, 0.0);
        sgemm(
            Transpose::None,
            Transpose::Ordinary,
            m,
            k,
            dim,
            1.0,
            tile.as_slice(),
            centers.as_slice(),
            None,
            &mut scores,
        )?;
        for (r, ((point, label), &point_norm)) in tile.rows().zip(output).zip(norms).enumerate() {
            let mut best = 0;
            let mut minimum = f64::INFINITY;
            let mut second = f64::INFINITY;
            let mut finite = point_norm.is_finite();
            for (j, (&dot, &norm)) in scores[r * k..][..k].iter().zip(center_norms).enumerate() {
                let score = norm - 2.0 * f64::from(dot);
                finite &= score.is_finite();
                if score < minimum {
                    second = minimum;
                    minimum = score;
                    best = j as u32;
                } else if score < second {
                    second = score;
                }
            }
            // Bound dot-product error using gamma_D and Cauchy-Schwarz. Four accounts
            // for the two factor-2 dot terms in a score difference. Inflate the f32
            // point norm and include an absolute underflow allowance. This protects
            // large offsets and ties without promising bitwise L2 equivalence.
            let guard = 4.0 * gamma * (f64::from(point_norm) * max_norm / (1.0 - gamma)).sqrt()
                + 2.0 * f64::EPSILON * (dim as f64 + 1.0) * max_norm
                + dim as f64 * f64::from(f32::MIN_POSITIVE);
            *label = if finite && second - minimum > guard {
                best
            } else {
                nearest(point, centers)
            };
        }
    }
    Ok(())
}

fn nearest(point: &[f32], centers: rowmajor::Ref<'_, f32>) -> u32 {
    let mut best = 0;
    let mut minimum: f32 = SquaredL2::evaluate(point, centers.row(0));
    for (j, center) in centers.rows().enumerate().skip(1) {
        let distance: f32 = SquaredL2::evaluate(point, center);
        if distance < minimum {
            minimum = distance;
            best = j as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoring_covers_tiles_tails_and_large_offsets() {
        for dim in [1, 7, 33, 896] {
            for rows in [0, 1, 127, 128, 129, 257] {
                for offset in [0.0, 10_000.0] {
                    let centers = rowmajor::Owned::try_from_data(
                        (0..12)
                            .flat_map(|j| std::iter::repeat_n(offset + 4.0 * j as f32, dim))
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        12,
                        dim,
                    )
                    .unwrap();
                    let points = rowmajor::Owned::try_from_data(
                        (0..rows)
                            .flat_map(|i| {
                                std::iter::repeat_n(offset + 4.0 * (i % 12) as f32 + 0.25, dim)
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                        rows,
                        dim,
                    )
                    .unwrap();
                    let norms: Vec<_> = centers.rows().map(center_norm).collect();
                    let mut labels = vec![u32::MAX; rows];
                    assign(points.as_view(), centers.as_view(), &norms, &mut labels).unwrap();
                    assert_eq!(
                        labels,
                        (0..rows).map(|i| (i % 12) as u32).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn uncertain_and_nonfinite_scores_preserve_direct_l2_choices() {
        for (values, candidates) in [
            (vec![0.0, 0.5, 1.0, f32::NAN], vec![0.0, 1.0]),
            (vec![5.0, 6.0], vec![2.0, 8.0, 8.0]),
            (vec![0.0], vec![-f32::MAX, f32::MAX]),
            (vec![-f32::MAX, f32::MAX], vec![f32::MAX, f32::MAX / 2.0]),
            (vec![1.0e20], vec![0.0, 1.0e20]),
            (vec![0.0, 1.0], vec![f32::NAN, 1.0]),
        ] {
            let n = values.len();
            let k = candidates.len();
            let points = rowmajor::Owned::try_from_data(values.into_boxed_slice(), n, 1).unwrap();
            let centers =
                rowmajor::Owned::try_from_data(candidates.into_boxed_slice(), k, 1).unwrap();
            let norms: Vec<_> = centers.rows().map(center_norm).collect();
            let expected: Vec<_> = points
                .rows()
                .map(|p| nearest(p, centers.as_view()))
                .collect();
            let mut labels = vec![u32::MAX; n];
            assign(points.as_view(), centers.as_view(), &norms, &mut labels).unwrap();
            assert_eq!(labels, expected);
        }
    }

    #[test]
    fn one_center_needs_no_scoring() {
        let points =
            rowmajor::Owned::try_from_data(vec![f32::NAN, 3.0].into_boxed_slice(), 2, 1).unwrap();
        let centers = rowmajor::Owned::try_from_data(vec![1.0].into_boxed_slice(), 1, 1).unwrap();
        let mut labels = [99; 2];
        assign(points.as_view(), centers.as_view(), &[1.0], &mut labels).unwrap();
        assert_eq!(labels, [0, 0]);
    }
}
