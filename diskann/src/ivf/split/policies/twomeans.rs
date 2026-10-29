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

use crate::{
    ANNError, ANNResult,
    ivf::{
        region::{AsMatRef, Update, View},
        split::{
            Plan, Split,
            plan::{Buckets, Destinations, Dst, Src},
        },
    },
    utils::{TWO, VectorId},
};

#[derive(Debug, thiserror::Error)]
pub(crate) enum SplitError<L> {
    #[error("Split could not retrieve list {0:?} from the region")]
    UnableToRetrieve(L),
    #[error("Split in list {list:?} does not have enough points, got {actual}")]
    NotEnoughPoints { list: L, actual: usize },
    #[error("Parents is empty")]
    EmptyParents,
}

impl<L: Debug + Send + Sync + 'static> From<SplitError<L>> for ANNError {
    #[track_caller]
    fn from(error: SplitError<L>) -> Self {
        Self::new(error)
    }
}

/// Two-means per parent followed by nearest-centroid assignment within each neighborhood.
pub struct TwoMeansSplit {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeansSplit {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }

    /// Fit two children per parent, in parent order.
    fn fit<I, L, R>(&self, region: &R, parents: &[L]) -> ANNResult<rowmajor::Owned<f32>>
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

            dst.as_mut_slice().copy_from_slice(
                points
                    .subview(0..TWO.get())
                    .ok_or(SplitError::NotEnoughPoints {
                        list,
                        actual: points.nrows(),
                    })?
                    .as_slice(),
            );

            let _ = lloyds(points, dst, self.iterations.get());
        }

        Ok(children)
    }
}

///////////////
// Centroids //
///////////////

/// Destination centroids of one split: borrowed neighbor centroids and fitted children.
struct Centroids<'a> {
    neighbors: Vec<&'a [f32]>,
    children: &'a rowmajor::Owned<f32>,
    /// Children per parent.
    k: NonZeroUsize,
}

impl<'a> Centroids<'a> {
    fn new<I: 'a, L, R>(
        region: &'a R,
        neighbors: &[L],
        children: &'a rowmajor::Owned<f32>,
        k: NonZeroUsize,
    ) -> ANNResult<Self>
    where
        L: VectorId,
        R: View<I, L>,
    {
        let neighbors = neighbors
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
                let (k, children) = (self.k.get(), self.children);
                out.extend((k * p..k * (p + 1)).map(|c| (Dst::Child(c), children.row(c))));
            }
            Src::Neighbor(n) => out.push((Dst::Neighbor(n), self.neighbors[n])),
        }
    }
}

///////////
// Split //
///////////

impl<I, L, R> Split<I, L, R> for TwoMeansSplit
where
    I: Copy,
    L: VectorId,
    R: View<I, L>,
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
        let centroids = Centroids::new(region, neighbors, &children, TWO)?;

        let mut moved = Buckets::new(Destinations {
            neighbors: neighbors.len(),
            children: children.nrows(),
        });

        let mut candidates = Vec::new();

        for (src, list) in plan.sources() {
            candidates.clear();

            // Add the src to be expanded, and then expand the neighborhood
            centroids.extend(src, &mut candidates);

            for s in plan.neighborhood(src) {
                centroids.extend(s, &mut candidates);
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

            for (&id, point) in view.members.iter().zip(vectors.reborrow().as_view().rows()) {
                let dst = nearest(point, &candidates);

                if Some(dst) != home {
                    moved.push(id, dst);
                }
            }

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

/// The nearest candidate; the earliest wins ties, including infinite distances.
///
/// # Panics
///
/// Panics if `candidates` is empty.
#[inline(always)]
fn nearest(point: &[f32], candidates: &[(Dst, &[f32])]) -> Dst {
    let (mut best, first) = candidates[0];
    let mut min: f32 = SquaredL2::evaluate(point, first);
    for &(dst, centroid) in &candidates[1..] {
        let distance: f32 = SquaredL2::evaluate(point, centroid);
        if distance < min {
            min = distance;
            best = dst;
        }
    }

    best
}
