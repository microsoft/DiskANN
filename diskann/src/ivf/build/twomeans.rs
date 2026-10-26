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

use super::{Split, utils::Buckets};
use crate::{
    ANNError, ANNResult,
    ivf::region::{AsMatRef, Update, View},
    utils::TWO,
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
    #[error("Centroid count overflows for {parents} parents and {neighbors} neighbors")]
    Overflow { parents: usize, neighbors: usize },
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
pub struct TwoMeansSplit {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeansSplit {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }

    /// Fit two children per parent into consecutive row pairs of `children`.
    fn fit<I, L, R>(
        &self,
        region: &R,
        parents: &[L],
        mut children: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()>
    where
        L: Copy + Debug + Send + Sync + 'static,
        R: View<I, L>,
        for<'a> R::VectorsRef<'a>: AsMatRef<'a, f32>,
    {
        debug_assert!(parents.len() > 0);

        for (&list, mut dst) in parents.iter().zip(children.window_iter_mut(TWO)) {
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

        Ok(())
    }
}

///////////
// Split //
///////////

impl<I, L, R> Split<I, L, R> for TwoMeansSplit
where
    I: Copy,
    L: Copy + Debug + Send + Sync + 'static,
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

        let (p, n) = (parents.len(), neighbors.len());

        // Centroids: retained neighbors, then two children per parent.
        let rows = p
            .checked_mul(TWO.get())
            .and_then(|children| children.checked_add(n))
            .ok_or(TwoMeansError::<L>::Overflow {
                parents: p,
                neighbors: n,
            })?;

        let mut centroids = rowmajor::Owned::try_from_element(rows, self.dim.get(), 0.0)?;

        for (&list, row) in neighbors.iter().zip(centroids.rows_mut()) {
            let view = region
                .get_list(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            row.copy_from_slice(view.centroid);
        }

        #[expect(clippy::expect_used, reason = "`n <= rows`")]
        let children = centroids
            .subview_mut(n..rows)
            .expect("children follow neighbors");

        self.fit(region, parents, children)?;

        // Assign: moved points are bucketed by source, keyed by their destination rows.
        let mut moved = Buckets::new();

        for (s, &list) in parents.iter().chain(neighbors).enumerate() {
            let view = region
                .get_list(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            let vectors = region
                .get_vectors(list)
                .ok_or(TwoMeansError::UnableToRetrieve(list))?;

            // A neighbor's own row is its index among neighbors; parents have none.
            let home = s.checked_sub(p);

            for (&id, point) in view.members.iter().zip(vectors.reborrow().as_view().rows()) {
                let destination = nearest(point, centroids.as_view());

                if Some(destination) != home {
                    moved.push(id, destination);
                }
            }

            moved.seal();
        }

        // Update
        let incoming = moved.regroup(rows);

        for (i, &list) in neighbors.iter().enumerate() {
            // SAFETY: `moved` sealed one bucket per parent and neighbor, and `incoming`
            // has `rows = n + 2p` buckets; `i < n`.
            unsafe {
                update.remove_members(list, moved.get_unchecked(p + i))?;
                update.append_members(list, incoming.get_unchecked(i))?;
            }
        }
        for (i, centroid) in centroids.as_view().rows().enumerate().skip(n) {
            // SAFETY: `i < rows`, the number of buckets in `incoming`.
            update.create(centroid, unsafe { incoming.get_unchecked(i) })?;
        }
        for &parent in parents {
            update.retire(parent)?;
        }
        Ok(())
    }
}

/// Index of the nearest centroid; the earliest row wins ties, including infinite distances.
#[inline(always)]
fn nearest(point: &[f32], centroids: rowmajor::Ref<'_, f32>) -> usize {
    let mut best = 0;
    let mut distance: f32 = SquaredL2::evaluate(point, centroids.row(0));
    for (index, centroid) in centroids.rows().enumerate().skip(1) {
        let candidate = SquaredL2::evaluate(point, centroid);
        if candidate < distance {
            best = index;
            distance = candidate;
        }
    }
    best
}
