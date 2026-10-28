/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-means splitting and nearest-centroid regional assignment.

use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData, num::NonZeroUsize, option::Iter};

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
            plan::{Buckets, Destinations, Dst},
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
    #[error("Child centroid count overflows for {0} parents")]
    Overflow(usize),
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

    fn assign<'a, I, L, R, F>(
        &self,
        list: L,
        region: &R,
        itr: &[(Dst, &'a [f32])],
        default: (Dst, &'a [f32]), // default to map to.
        mut cb: F,
    ) -> ANNResult<()>
    where
        F: FnMut(I, Dst),
        R: View<I, L>,
        L: VectorId,
        I: Copy,
        for<'b> R::VectorsRef<'b>: AsMatRef<'b, f32>,
    {
        let view = region
            .get_list(list)
            .ok_or(SplitError::UnableToRetrieve(list))?;

        let vectors = region
            .get_vectors(list)
            .ok_or(SplitError::UnableToRetrieve(list))?;

        for (&id, point) in view.members.iter().zip(vectors.reborrow().as_view().rows()) {
            let dst = nearest(point, default, itr);
            cb(id, dst);
        }

        Ok(())
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
        let parents = plan.parents();
        let neighbors = plan.neighbors();

        if parents.is_empty() {
            return Err(SplitError::<L>::EmptyParents.into());
        }

        let children = self.fit(region, parents)?;

        // Collect the centroids for the neighbors once
        let centroids_n = plan
            .neighbors()
            .iter()
            .map(|n| {
                region
                    .get_list(*n)
                    .map(|list| list.centroid)
                    .ok_or(SplitError::UnableToRetrieve(*n))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let k = TWO;

        let child =
            |p: usize| (k.get() * p..k.get() * (p + 1)).map(|c| (Dst::Child(c), children.row(c)));

        let neighbor = |n: usize| (Dst::Neighbor(n), centroids_n[n]);

        let layout = Destinations {
            neighbors: plan.neighbors().len(),
            children: children.nrows(),
        };
        let mut moved = Buckets::new(layout);

        // Assign the parents
        for (source, &list) in parents.iter().enumerate() {
            let mut itr = plan
                .parent_links(source)
                .iter()
                .map(|&n| neighbor(n))
                .chain(child(source))
                .collect::<Vec<_>>();

            let first = itr.pop().ok_or(SplitError::<L>::EmptyParents)?;

            self.assign(list, region, itr.as_slice(), first, |id, dst| {
                moved.push(id, dst);
            })?;

            moved.seal();

            update.retire(list)?; // Retire parent
        }

        // Assign the neighbors
        for (source, &list) in neighbors.iter().enumerate() {
            let itr = plan
                .neighbor_links(source)
                .iter()
                .flat_map(|&p| child(p))
                .collect::<Vec<_>>();

            let mut remove = Vec::new();

            self.assign(list, region, itr.as_slice(), neighbor(source), |id, dst| {
                if dst != Dst::Neighbor(source) {
                    moved.push(id, dst);
                    remove.push(id);
                }
            })?;

            moved.seal();

            update.remove_members(list, &remove)?; // remove the members moving out.
        }

        let incoming = moved.group();

        for (row, &list) in neighbors.iter().enumerate() {
            update.append_members(list, incoming.get(Dst::Neighbor(row)))?;
        }

        for (row, child) in children.rows().enumerate() {
            update.create(child, incoming.get(Dst::Child(row)))?;
        }

        Ok(())
    }
}

/// Return the nearest candidate; the first wins ties, including infinite distances.
/// `candidates` must be nonempty.
#[inline(always)]
fn nearest<'a>(point: &[f32], default: (Dst, &'a [f32]), itr: &[(Dst, &'a [f32])]) -> Dst {
    let mut dst = default.0;
    let mut min = SquaredL2::evaluate(point, default.1);
    for row in itr {
        let dist: f32 = SquaredL2::evaluate(point, row.1);
        if dist < min {
            min = dist;
            dst = row.0;
        }
    }
    dst
}
