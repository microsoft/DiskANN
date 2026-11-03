/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! k-means splitting: each parent is replaced by `k` fitted children.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::{
    Reborrow, ReborrowMut,
    views::rowmajor::{self, Matrix, MatrixMut},
};

use super::{Assign, Buckets, NearestCentroid, SplitError};
use crate::{
    ANNError, ANNResult,
    error::{ErrorContext, IntoANNResult},
    ivf::{
        region::{AsMatRef, Update, View},
        split::Split,
    },
    utils::VectorId,
};

////////////
// Errors //
////////////

/// A parent has fewer points than the children it must seed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum FitError {
    #[error("Expected at least {0} points to seed, got {1}")]
    NotEnoughPoints(usize, usize),
    #[error("Expected dimension {0}, got {1}")]
    DimensionMismatch(usize, usize),
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
pub(crate) trait Seed {
    /// The error returned by a failed seeding.
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Fill every row of `centers` with an initial centroid for `points`.
    fn seed(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error>;
}

/// Seeds with the first rows of `points`.
///
/// Fails with [`NotEnoughPoints`] if `points` has fewer rows than `centers`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FirstPoints;

impl Seed for FirstPoints {
    type Error = FitError;

    fn seed(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        mut centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error> {
        debug_assert!(points.ncols() == centers.ncols());

        let n = points.nrows();
        let k = centers.nrows();

        if n < k {
            return Err(FitError::NotEnoughPoints(n, k));
        }

        let first = &points.as_slice()[..centers.as_slice().len()];
        centers.as_mut_slice().copy_from_slice(first);

        Ok(())
    }
}

////////////
// KMeans //
////////////

/// Splits each parent into `k` children, fitted by k-means over the parent's points.
///
/// # Algorithm
///
/// 1. Fit `k` children per parent: seed them with `S`, then run Lloyd's iterations over
///    the parent's points. Fitting finishes for every parent before anything is written,
///    so its failures leave the update untouched.
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

    /// Fit `k` children per parent. The children of the parent at index `p` are rows
    /// `k * p..k * (p + 1)` of the returned matrix.
    fn fit<I, L, R>(&mut self, region: &R, parents: &[L]) -> ANNResult<rowmajor::Owned<f32>>
    where
        L: VectorId,
        R: View<I, L>,
        for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
    {
        let mut children =
            rowmajor::Owned::try_from_element(parents.len() * self.k.get(), self.dim.get(), 0.0)?;

        for (&list, mut dst) in parents.iter().zip(children.window_iter_mut(self.k)) {
            let vectors = region
                .get_vectors(list)
                .ok_or(SplitError::UnableToRetrieve(list))?;

            let points = vectors.reborrow().as_view();

            if points.ncols() != dst.ncols() {
                return Err(FitError::DimensionMismatch(points.ncols(), dst.ncols()).into());
            }

            self.seed
                .seed(points, dst.reborrow_mut())
                .into_ann_result()
                .with_context(|| format!("while seeding list {list:?}"))?;

            let _ = lloyds(points, dst, self.iterations.get());
        }

        Ok(children)
    }
}

///////////
// Split //
///////////

impl<I, L, R, A, S> Split<I, L, R> for KMeansSplit<A, S>
where
    I: Copy,
    L: VectorId,
    R: View<I, L>,
    A: for<'a> Assign<R::VectorsRef<'a>>,
    S: Seed,
    for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
{
    type Error = ANNError;

    fn split<U, F>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        mut emit: F,
    ) -> ANNResult<()>
    where
        U: Update<I, L>,
        F: FnMut(L, L),
    {
        if parents.is_empty() {
            return Err(SplitError::<L>::EmptyParents.into());
        }

        let children = self.fit(region, parents)?;
        let k = self.k.get();

        let mut moved = Buckets::new(children.nrows());
        let mut candidates = Vec::new();

        // Each parent's points move to one of its own children, chosen by `A`.
        for (p, &parent) in parents.iter().enumerate() {
            candidates.clear();
            candidates.extend((k * p..k * (p + 1)).map(|c| (c, children.row(c))));

            let view = region
                .get_list(parent)
                .ok_or(SplitError::UnableToRetrieve(parent))?;
            let vectors = region
                .get_vectors(parent)
                .ok_or(SplitError::UnableToRetrieve(parent))?;

            self.assign
                .assign(view.members, vectors.reborrow(), &candidates, |id, c| {
                    moved.push(id, c)
                });

            update.retire(parent)?;
        }

        let incoming = moved.group();

        for (c, child) in children.rows().enumerate() {
            // Every parent has exactly `k` children, so child `c` belongs to parent `c / k`.
            let p = c / k;
            emit(parents[p], update.create(child, incoming.get(c))?);
        }

        Ok(())
    }
}
