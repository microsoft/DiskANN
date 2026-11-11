/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! k-means splitting: each parent is replaced by `k` fitted children.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_quantization::{Parallelism, algorithms::kmeans::lloyds::lloyds};
use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix, MatrixMut},
};

use super::{Assign, Buckets, Groups, NearestCentroid, SplitError, TryMap};
use crate::{
    ANNError, ANNResult,
    error::{ErrorContext, IntoANNResult},
    ivf::{
        region::{AsMatRef, Region},
        split::Split,
    },
};

////////////
// Errors //
////////////

/// A parent's points cannot be fitted to its children.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FitError {
    #[error("Expected at least {expected} points to seed, got {got}")]
    NotEnoughPoints { expected: usize, got: usize },
    #[error("Expected dimension {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

impl From<FitError> for ANNError {
    #[track_caller]
    fn from(error: FitError) -> Self {
        Self::new(error)
    }
}

//////////
// Seed //
//////////

/// Chooses initial centroids for one parent's points.
///
/// Parents may be seeded in parallel, so seeding must depend only on its arguments.
pub(crate) trait Seed: Sync {
    /// The error returned by a failed seeding.
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Fill every row of `centers` with an initial centroid for `points`.
    fn seed(
        &self,
        points: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error>;
}

/// Seeds with the first rows of `points`.
///
/// Fails with [`FitError::NotEnoughPoints`] if `points` has fewer rows than `centers`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FirstPoints;

impl Seed for FirstPoints {
    type Error = FitError;

    fn seed(
        &self,
        points: rowmajor::Ref<'_, f32>,
        mut centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error> {
        debug_assert!(points.ncols() == centers.ncols());

        let n = points.nrows();
        let k = centers.nrows();

        if n < k {
            return Err(FitError::NotEnoughPoints {
                expected: k,
                got: n,
            });
        }

        let first = &points.as_slice()[..centers.as_slice().len()];
        centers.as_mut_slice().copy_from_slice(first);

        Ok(())
    }
}

////////////
// KMeans //
////////////

/// The children fitted to one parent.
struct Children<I> {
    // Centroids of the children
    centroids: rowmajor::Owned<f32>,
    // Groups mapping members to centroid index; indexed by `centroid`
    members: Groups<I>,
}

/// Splits each parent into `k` children, fitted by k-means over the parent's points.
///
/// # Algorithm
///
/// 1. Fit `k` children per parent: seed them with `S`, then run Lloyd's iterations over
///    the parent's points.
///
/// 2. For each parent, assign its points among its own children with `A`, then retire it.
///
/// 3. Create the children in parent order, each holding the points assigned to it.
pub struct KMeansSplit<A = NearestCentroid, S = FirstPoints> {
    /// Children per parent.
    k: NonZeroUsize,
    /// Lloyd iterations per parent.
    iterations: NonZeroUsize,
    /// Dimension of the points and children.
    dim: NonZeroUsize,
    assign: A,
    seed: S,
}

impl<A, S: Seed> KMeansSplit<A, S> {
    /// Construct a policy that splits each parent into `k` children of dimension `dim`,
    /// fitted with `iterations` Lloyd iterations.
    pub fn new(
        k: NonZeroUsize,
        iterations: NonZeroUsize,
        dim: NonZeroUsize,
        assign: A,
        seed: S,
    ) -> Self {
        Self {
            k,
            iterations,
            dim,
            assign,
            seed,
        }
    }

    /// Fit `k` children to the points of `parent`, and assign each point to one of them.
    ///
    /// This reads only `region`, so parents can be fitted in parallel.
    fn fit<I, L, R>(&self, region: &R, parent: L) -> ANNResult<Children<I>>
    where
        I: Copy,
        L: Copy + Debug + Send + Sync + 'static,
        R: Region<I, L>,
        A: for<'a> Assign<R::VectorsRef<'a>, f32>,
        for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
    {
        let view = region
            .get_list(parent)
            .ok_or(SplitError::UnableToRetrieve(parent))?;

        let vectors = region
            .get_vectors(parent)
            .ok_or(SplitError::UnableToRetrieve(parent))?;

        let points = vectors.reborrow().as_view();

        let mut centroids = rowmajor::Owned::try_from_element(self.k.get(), self.dim.get(), 0.0)?;

        if points.ncols() != centroids.ncols() {
            return Err(FitError::DimensionMismatch {
                expected: centroids.ncols(),
                got: points.ncols(),
            }
            .into());
        }

        self.seed
            .seed(points, centroids.as_view_mut())
            .into_ann_result()
            .with_context(|| format!("while seeding list {parent:?}"))?;

        let _ = lloyds(points, centroids.as_view_mut(), self.iterations.get());

        // Each point moves to one of the parent's own children, chosen by `A`.
        let candidates: Vec<_> = centroids.rows().enumerate().collect();

        let mut members = Buckets::new(self.k.get());

        self.assign
            .assign(view.members, vectors.reborrow(), &candidates, |id, c| {
                members.push(id, c)
            });

        Ok(Children {
            members: members.group(),
            centroids,
        })
    }
}

///////////
// Split //
///////////

impl<I, L, R, A, S> Split<I, L, R> for KMeansSplit<A, S>
where
    I: Copy + Send,
    L: Copy + Debug + Send + Sync + 'static,
    R: Region<I, L>,
    A: for<'a> Assign<R::VectorsRef<'a>, f32>,
    S: Seed,
    for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
{
    type Error = ANNError;

    fn split<F>(
        &mut self,
        region: &mut R,
        parents: &[L],
        parallelism: Parallelism,
        mut emit: F,
    ) -> ANNResult<()>
    where
        F: FnMut(L, L),
    {
        if parents.is_empty() {
            return Err(SplitError::<L>::EmptyParents.into());
        }

        let families =
            parallelism.try_map(parents.len(), |p: usize| self.fit(region, parents[p]))?;

        // Write in parent order.
        for (&parent, children) in parents.iter().zip(&families) {
            region.retire(parent)?;

            for (c, centroid) in children.centroids.rows().enumerate() {
                emit(parent, region.create(centroid, children.members.get(c))?);
            }
        }

        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn points(nrows: usize) -> rowmajor::Owned<f32> {
        let data: Vec<f32> = (0..nrows).flat_map(|i| [i as f32, -(i as f32)]).collect();
        rowmajor::Owned::try_from_data(data.into(), nrows, 2).unwrap()
    }

    #[test]
    fn first_points_copies_the_leading_rows() {
        let points = points(3);
        for k in [2, 3] {
            let mut centers = rowmajor::Owned::try_from_element(k, 2, -1.0).unwrap();
            FirstPoints
                .seed(points.as_view(), centers.as_view_mut())
                .unwrap();
            assert_eq!(centers.as_slice(), &points.as_slice()[..2 * k]);
        }
    }

    #[test]
    fn first_points_needs_as_many_points_as_centroids() {
        let mut centers = rowmajor::Owned::try_from_element(3, 2, -1.0).unwrap();
        let err = FirstPoints
            .seed(points(2).as_view(), centers.as_view_mut())
            .unwrap_err();
        assert_eq!(err.to_string(), "Expected at least 3 points to seed, got 2");
        assert!(
            centers.as_slice().iter().all(|&x| x == -1.0),
            "the centroids should be untouched"
        );
    }
}
