/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Regional split policies and the assignment they share.

use diskann_utils::views::rowmajor::Matrix;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use crate::ivf::{region::AsMatRef, split::Dst};

pub(crate) mod kmeans;
pub(crate) use kmeans::{FirstPoints, KMeansSplit};

////////////
// Assign //
////////////

/// Chooses a destination for every point of one list, reading points as `V`.
pub(crate) trait Assign<V> {
    /// Assign points to candidates and call `emit` exactly once per member.
    fn assign<I>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(Dst, &[f32])],
        emit: impl FnMut(I, Dst),
    ) where
        I: Copy;
}

/// Assigns each point to its nearest candidate by L2.
///
/// The earliest candidate wins ties, including infinite distances.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NearestCentroid;

impl<'a, V> Assign<V> for NearestCentroid
where
    V: AsMatRef<'a, f32>,
{
    fn assign<I: Copy>(
        &self,
        members: &[I],
        points: V,
        candidates: &[(Dst, &[f32])],
        mut emit: impl FnMut(I, Dst),
    ) {
        match candidates.split_first() {
            // Return if there are no candidates.
            None => return,
            Some((def, rest)) => {
                // Naive loop over the members
                for (&id, point) in members.iter().zip(points.as_view().rows()) {
                    emit(id, nearest(point, *def, rest));
                }
            }
        }
    }
}

/// # Panics
///
/// Panics if `candidates` is empty.
#[inline(always)]
fn nearest<T>(point: &[T], default: (Dst, &[T]), candidates: &[(Dst, &[T])]) -> Dst
where
    SquaredL2: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T]>,
{
    let mut best = default.0;
    let mut min: f32 = SquaredL2::evaluate(point, default.1);

    for &(dst, centroid) in candidates.iter() {
        let distance: f32 = SquaredL2::evaluate(point, centroid);
        if distance < min {
            min = distance;
            best = dst;
        }
    }

    best
}
