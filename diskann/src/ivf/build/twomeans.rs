/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means splitting and nearest-centroid regional assignment.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::Split;
use crate::{
    ANNError, ANNResult,
    ivf::region::{AsMatRef, Update, View},
    utils::{TWO, VectorId},
};

////////////
// Errors //
////////////

#[derive(Debug, thiserror::Error)]
pub(crate) enum TwoMeansError<L> {
    #[error("IVF two-means could not retrieve list {0:?} from the region")]
    UnableToRetrieve(L),
    #[error("IVF two-means needs at least two points in list {list:?}, got {actual}")]
    NotEnoughPoints { list: L, actual: usize },
    #[error("Centroid count overflows for {0} parents")]
    Overflow(usize),
    #[error("Parents is empty")]
    EmptyParents,
}

impl<L: Debug + Send + Sync + 'static> From<TwoMeansError<L>> for ANNError {
    #[track_caller]
    fn from(error: TwoMeansError<L>) -> Self {
        Self::new(error)
    }
}

/// Two-means per parent followed by nearest-centroid assignment over the region.
///
/// Seeds are the first two rows of each parent. Lloyd iterations reset empty
/// centers to zero. One centroid matrix is allocated up front for all parents.
/// Retained neighbors precede new children when resolving equal-distance assignments.
/// All computation completes before any updates are recorded.
pub struct TwoMeansSplit {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeansSplit {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }

    /// Fit and output the centroids given by the region and the parents.
    fn fit<I, L, R>(&self, region: &R, parents: &[L]) -> ANNResult<rowmajor::Owned<f32>>
    where
        L: VectorId,
        R: View<I, L>,
        for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
    {
        let rows = parents
            .len()
            .checked_mul(TWO.get())
            .ok_or(TwoMeansError::<L>::Overflow(parents.len()))?;

        let mut centers = rowmajor::Owned::try_from_element(rows, self.dim.get(), 0.0)?;

        for (&list, mut dst) in parents.iter().zip(centers.window_iter_mut(TWO)) {
            let vectors = region
                .get_vectors(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            let points = vectors.reborrow().as_view();

            dst.as_mut_slice().copy_from_slice(
                points
                    .subview(0..TWO.get())
                    .ok_or(TwoMeansError::NotEnoughPoints {
                        list,
                        actual: points.nrows(),
                    })?
                    .as_slice(),
            );

            let _ = lloyds(points, dst, self.iterations.get());
        }

        Ok(centers)
    }
}

/////////////
// Targets //
/////////////

#[derive(Clone, Copy)]
/// Defines the category of the target centroid.
enum TargetKind<L> {
    // Existing centroid, from a neighbor for example
    Exists(L),
    // New centroid as a result of a split.
    New,
}

#[derive(Clone)]
/// Target object holding the target kind and reference to centroid vector.
struct Target<'a, L> {
    target: TargetKind<L>,
    centroid: &'a [f32],
}

///////////
// Split //
///////////

impl<I, L, R> Split<I, L, R> for TwoMeansSplit
where
    I: VectorId,
    L: VectorId,
    R: View<I, L>,
    for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
{
    type Error = ANNError;

    fn split<U: Update<I, L>>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        neighbors: &[L],
    ) -> ANNResult<()> {
        if parents.is_empty() {
            return Err(TwoMeansError::<L>::EmptyParents.into());
        }

        // Fit children

        let children = self.fit(region, parents)?;

        let mut targets = Vec::new();

        for &neighbor in neighbors {
            let view = region
                .get_list(neighbor)
                .ok_or(TwoMeansError::UnableToRetrieve(neighbor))?;

            targets.push(Target {
                target: TargetKind::Exists(neighbor),
                centroid: view.centroid,
            });
        }

        targets.extend(children.rows().map(|centroid| Target {
            target: TargetKind::New,
            centroid,
        }));

        // Assign

        let mut memberships = vec![Vec::new(); targets.len()];

        for &list in parents.iter().chain(neighbors.iter()) {
            let view = region
                .get_list(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            let vectors = region
                .get_vectors(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            let points = vectors.reborrow().as_view();

            for (&id, point) in view.members.iter().zip(points.rows()) {
                memberships[nearest(point, &targets)].push(id);
            }
        }

        // Update

        for (target, members) in targets.iter().zip(&memberships) {
            match target.target {
                TargetKind::New => update.create(target.centroid, members)?,
                TargetKind::Exists(list) => update.set_members(list, members)?,
            };
        }

        for &parent in parents {
            update.retire(parent)?;
        }
        Ok(())
    }
}

/// Destinations are nonempty; the earliest wins ties, including infinite distances.
#[inline(always)]
fn nearest<L>(point: &[f32], targets: &[Target<'_, L>]) -> usize {
    let mut best = 0;
    let mut distance: f32 = SquaredL2::evaluate(point, targets[0].centroid);
    for (index, target) in targets.iter().enumerate().skip(1) {
        let candidate = SquaredL2::evaluate(point, target.centroid);
        if candidate < distance {
            best = index;
            distance = candidate;
        }
    }
    best
}
