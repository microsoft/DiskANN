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

#[derive(Debug, thiserror::Error)]
#[error("Expected at least {required} points to seed, got {actual}")]
pub(crate) struct NotEnoughPoints {
    pub(crate) actual: usize,
    pub(crate) required: usize,
}

impl From<NotEnoughPoints> for ANNError {
    #[track_caller]
    fn from(error: NotEnoughPoints) -> Self {
        Self::new(error)
    }
}

//////////
// Seed //
//////////

/// Chooses initial centroids for one parent's points.
pub(crate) trait Seed {
    /// Error type for seeding algorithm
    type Error: Debug + Send + Sync + Into<ANNError>;

    /// Fill every row of `centers` with an initial centroid for `points`.
    fn seed(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error>;
}

/// Seeds with the first rows of `points`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FirstPoints;

impl Seed for FirstPoints {
    type Error = NotEnoughPoints;

    fn seed(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        mut centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error> {
        let n = points.nrows();
        let k = centers.nrows();

        if n < k {
            return Err(NotEnoughPoints {
                actual: n,
                required: k,
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

/// k-means per parent, seeded by `S`, with each parent's points assigned to its own
/// children by `A`.
pub struct KMeansSplit<A = NearestCentroid, S = FirstPoints> {
    /// Children per parent.
    k: NonZeroUsize,
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
    assign: A,
    seed: S,
}

impl<A, S: Seed> KMeansSplit<A, S> {
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

    /// Fit `k` children per parent, in parent order.
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

        // Each parent's points move to the nearest of its own children.
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
            // We cluster every parent into exactly `k` clusters, so integer division gives us the parent
            let p = c / k; 
            emit(parents[p], update.create(child, incoming.get(c))?);
        }


        Ok(())
    }
}
