/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means splitting with assignment restricted to plan neighborhoods.

use std::{fmt::Debug, num::NonZeroUsize};

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::{
    Reborrow, ReborrowMut,
    views::rowmajor::{self, Matrix, MatrixMut},
};

use crate::{
    ANNError, ANNResult,
    error::{IntoANNResult, ann_error::ErrorContext},
    ivf::{
        region::{AsMatRef, Update, View},
        split::{
            Plan, Split,
            plan::{Buckets, Destinations, Dst, Src},
            policies::{Assign, NearestCentroid},
        },
    },
    utils::{TWO, VectorId},
};


////////////
// Errors //
////////////

#[derive(Debug, thiserror::Error)]
pub(crate) enum SplitError<L> {
    #[error("Split could not retrieve list {0:?} from the region")]
    UnableToRetrieve(L),
    #[error("Parents is empty")]
    EmptyParents,
}

impl<L: Debug + Send + Sync + 'static> From<SplitError<L>> for ANNError {
    #[track_caller]
    fn from(error: SplitError<L>) -> Self {
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
    ///
    /// `points` must have at least as many rows as `centers`.
    ///
    /// # Errors
    ///
    /// - Fails if no initial centroids can be chosen.
    fn seed(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Mut<'_, f32>,
    ) -> Result<(), Self::Error>;
}

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

//////////////
// TwoMeans //
//////////////

/// Two-means per parent, seeded by `S`, then assignment within each neighborhood by `A`.
pub struct TwoMeansSplit<A = NearestCentroid, S = FirstPoints> {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
    assign: A,
    seed: S,
}

impl<A, S: Seed> TwoMeansSplit<A, S> {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize, assign: A, seed: S) -> Self {
        Self {
            iterations,
            dim,
            assign,
            seed,
        }
    }

    /// Fit two children per parent, in parent order.
    fn fit<I, L, R>(&mut self, region: &R, parents: &[L]) -> ANNResult<rowmajor::Owned<f32>>
    where
        L: VectorId,
        R: View<I, L>,
        for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
    {
        let mut children =
            rowmajor::Owned::try_from_element(parents.len() * TWO.get(), self.dim.get(), 0.0)?;

        for (&list, mut dst) in parents.iter().zip(children.window_iter_mut(TWO)) {
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

/////////////
// Targets //
/////////////

/// Destination centroids of one split: neighbor centroids and children.
///
/// A simple struct to to zip a centroid id or a neighbor id with
/// it's appropriate centroid vector in order to use for assignment.
struct Targets<'a> {
    neighbors: Vec<&'a [f32]>,
    children: &'a rowmajor::Owned<f32>,
    /// Children per parent.
    k: NonZeroUsize,
}

impl<'a> Targets<'a> {
    fn new<I, L, R>(
        region: &'a R,
        plan: &Plan<L>,
        children: &'a rowmajor::Owned<f32>,
        k: NonZeroUsize,
    ) -> ANNResult<Self>
    where
        L: VectorId,
        R: View<I, L>,
        I: 'a,
    {
        let neighbors = plan
            .neighbors()
            .iter()
            .map(|&list| {
                region
                    .get_list(list)
                    .map(|view| view.centroid)
                    .ok_or(SplitError::UnableToRetrieve(list))
            })
            .collect::<Result<_, _>>()?;

        Ok(Self {
            neighbors,
            children,
            k,
        })
    }

    /// Append the destinations owned by `src`: a parent's children or a neighbor's centroid.
    fn extend(&self, src: Src, out: &mut Vec<(Dst, &'a [f32])>) {
        match src {
            Src::Parent(p) => {
                let range = self.k.get() * p..self.k.get() * (p + 1);
                out.extend(range.map(|c| (Dst::Child(c), self.children.row(c))));
            }
            Src::Neighbor(n) => out.push((Dst::Neighbor(n), self.neighbors[n])),
        }
    }
}

///////////
// Split //
///////////

impl<I, L, R, A, S> Split<I, L, R> for TwoMeansSplit<A, S>
where
    I: Copy,
    L: VectorId,
    R: View<I, L>,
    A: for<'a> Assign<R::VectorsRef<'a>>,
    S: Seed,
    for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
{
    type Error = ANNError;

    fn split<U: Update<I, L>>(
        &mut self,
        region: &R,
        update: &mut U,
        plan: &Plan<L>,
    ) -> ANNResult<()> {
        let neighbors = plan.neighbors();

        if plan.parents().is_empty() {
            return Err(SplitError::<L>::EmptyParents.into());
        }

        let children = self.fit(region, plan.parents())?;
        let targets = Targets::new(region, plan, &children, TWO)?;

        let mut moved = Buckets::new(Destinations {
            neighbors: neighbors.len(),
            children: children.nrows(),
        });

        let mut candidates = Vec::new();

        for (src, list) in plan.sources() {
            candidates.clear();

            // Add the src to be expanded, and then expand the neighborhood
            targets.extend(src, &mut candidates);

            for s in plan.neighborhood(src) {
                targets.extend(s, &mut candidates);
            }

            let view = region
                .get_list(list)
                .ok_or(SplitError::UnableToRetrieve(list))?;
            let vectors = region
                .get_vectors(list)
                .ok_or(SplitError::UnableToRetrieve(list))?;

            // A neighbor's points may stay put; a parent's points always move.
            let home = match src {
                Src::Parent(_) => None,
                Src::Neighbor(n) => Some(Dst::Neighbor(n)),
            };

            self.assign
                .assign(view.members, vectors.reborrow(), &candidates, |id, dst| {
                    if Some(dst) != home {
                        moved.push(id, dst);
                    }
                });

            let left = moved.seal();

            match src {
                // Retire parent
                Src::Parent(_) => update.retire(list)?,
                // Remove members from neighbor
                Src::Neighbor(_) => update.remove_members(list, left)?,
            }
        }

        let incoming = moved.group();

        for (n, &list) in neighbors.iter().enumerate() {
            update.append_members(list, incoming.get(Dst::Neighbor(n)))?;
        }
        for (c, child) in children.rows().enumerate() {
            update.create(child, incoming.get(Dst::Child(c)))?;
        }

        Ok(())
    }
}
