/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Replaceable child fitting and regional assignment over borrowed point rows.
//!
//! Fitting writes new centroids into reusable scratch. Assignment borrows the
//! destinations, groups source members, and records regional writes directly.

use std::{collections::BTreeSet, num::NonZeroUsize};

use diskann_quantization::algorithms::kmeans::lloyds::lloyds;
use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::region::{RegionUpdate, RegionView};
use crate::{ANNError, ANNResult, error::IntoANNResult, ivf::region::AsMatRef};

/// Fit a parent's replacement centroids without assigning regional memberships.
pub trait Clusterer<V, T = f32> {
    /// Fill `dst` with finite centroids of the canonical dimension.
    ///
    /// Implementations own initialization, convergence, empty-cluster handling,
    /// and reusable fitting scratch. `dst` has `num_centers()` rows and `dim()`
    /// columns. The number of centers must be at least two.
    fn fit(&mut self, points: V, dst: rowmajor::Mut<'_, T>) -> ANNResult<()>;

    fn num_centers(&self) -> NonZeroUsize;

    fn dim(&self) -> NonZeroUsize;
}

/// Assign the whole region, allowing policies that coordinate multiple sources.
pub trait Assigner<I, L, R: RegionView<I, L>> {
    /// Assign each source member to a destination and record complete memberships.
    ///
    /// Create new destinations, replace retained memberships, and retire sources
    /// not retained as destinations. Validate and compute assignments before
    /// recording writes. A write error may leave partial private updates.
    fn assign<U>(
        &mut self,
        region: &R,
        update: &mut U,
        sources: &[L],
        destinations: &[Target<'_, L>],
    ) -> ANNResult<()>
    where
        U: RegionUpdate<I, L>;
}

pub enum Type<L> {
    /// Existing centroid.
    Exists(L),
    /// New centroid as a result of the split.
    New,
}

/// A borrowed centroid; only retained destinations have an existing list ID.
pub struct Target<'a, L> {
    pub list: Type<L>,
    pub centroid: &'a [f32],
}

pub struct SplitAndAssign<C, A> {
    pub clusterer: C,
    pub assigner: A,
    scratch: Option<rowmajor::Owned<f32>>, // scratch for the centroids.
}

impl<C, A> SplitAndAssign<C, A> {
    pub fn new(clusterer: C, assigner: A) -> Self {
        Self {
            clusterer,
            assigner,
            scratch: None,
        }
    }

    /// Initialize the scratch for a particular number of output cluster centroids.
    fn initialize<V>(&mut self, num_jobs: NonZeroUsize) -> ANNResult<NonZeroUsize>
    where
        C: Clusterer<V>,
    {
        let c = self.clusterer.num_centers();

        let rows = num_jobs
            .checked_mul(c)
            .ok_or_else(|| ANNError::message("IVF centroid count overflow"))?
            .get();

        let dim = self.clusterer.dim().get();

        self.scratch = Some(rowmajor::Owned::try_from_element(rows, dim, f32::NAN)?);

        Ok(c)
    }

    /// Compute the complete result before registering any provider writes.
    ///
    /// Concrete-reference clusterers may need an explicit `R` type argument because
    /// of a compiler limitation when inferring higher-ranked associated types.
    pub fn run<I: Copy, L: Copy, R, U>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &BTreeSet<L>,
        neighbors: &BTreeSet<L>,
    ) -> ANNResult<()>
    where
        A: Assigner<I, L, R>,
        R: RegionView<I, L>,
        C: for<'a> Clusterer<R::VectorsRef<'a>>,
        U: RegionUpdate<I, L>,
    {
        let n_centroids = self
            .initialize::<R::VectorsRef<'_>>(
                NonZeroUsize::new(parents.len())
                    .ok_or_else(|| ANNError::message("IVF splitting needs at least one parent"))?,
            )?
            .get();
        let mut s = 0;

        let mut clusters = self
            .scratch
            .as_mut()
            .ok_or_else(|| ANNError::message("IVF centroid scratch was not initialized"))?
            .as_view_mut();

        for &p in parents {
            let vecs = region.get_vectors(p).ok_or_else(|| {
                ANNError::message("IVF split source vectors are missing from the region")
            })?;

            let end = s + n_centroids;

            self.clusterer.fit(
                vecs.reborrow(),
                clusters.subview_mut(s..end).ok_or_else(|| {
                    ANNError::message("IVF centroid scratch range is out of bounds")
                })?,
            )?;

            s = end;
        }

        let mut destinations: Vec<_> = clusters
            .rows()
            .map(|centroid| Target {
                list: Type::New,
                centroid,
            })
            .collect();

        for &n in neighbors {
            let view = region
                .get_view(n)
                .ok_or_else(|| ANNError::message("IVF neighbor is missing from the region"))?;

            destinations.push(Target {
                list: Type::Exists(n),
                centroid: view.centroid,
            });
        }

        let sources = parents.iter().chain(neighbors).copied().collect::<Vec<_>>();

        self.assigner
            .assign(region, update, &sources, &destinations)
    }
}

/// Random input seeds followed by Lloyd iterations, which reset empty centers to zero.
pub struct TwoMeans {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeans {
    pub fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }
}

impl<B: AsMatRef<f32>> Clusterer<B> for TwoMeans {
    fn fit(&mut self, points: B, mut dst: rowmajor::Mut<'_, f32>) -> ANNResult<()> {
        let dim = self.dim.get();
        let num_centers = <Self as Clusterer<B>>::num_centers(self).get();

        let vectors = points.as_view().into_ann_result()?;

        if vectors.nrows() < num_centers {
            return Err(ANNError::message("an IVF split needs at least two points"));
        }
        if vectors.ncols() != dim || dst.ncols() != dim || dst.nrows() != num_centers {
            return Err(ANNError::message(
                "two-means needs canonical input rows and a two-row destination of the canonical dimension",
            ));
        }

        dst.as_mut_slice().copy_from_slice(
            vectors
                .subview(0..num_centers)
                .ok_or(ANNError::message("oh no!"))?
                .as_slice(),
        );

        let _ = lloyds(vectors, dst, self.iterations.get());

        Ok(())
    }

    fn num_centers(&self) -> NonZeroUsize {
        const { crate::utils::TWO }
    }

    fn dim(&self) -> NonZeroUsize {
        self.dim
    }
}

pub struct RegionalNearest;

impl<I: Copy, L: Copy + Ord, R> Assigner<I, L, R> for RegionalNearest
where
    R: RegionView<I, L>,
    for<'a> R::VectorsRef<'a>: AsMatRef<f32>,
{
    fn assign<U>(
        &mut self,
        region: &R,
        update: &mut U,
        sources: &[L],
        destinations: &[Target<'_, L>],
    ) -> ANNResult<()>
    where
        U: RegionUpdate<I, L>,
    {
        let mut memberships: Vec<Vec<I>> = (0..destinations.len()).map(|_| Vec::new()).collect();

        for &src in sources {
            let view = region.get_view(src).ok_or_else(|| {
                ANNError::message("IVF assignment source is missing from the region")
            })?;
            let vectors = region.get_vectors(src).ok_or_else(|| {
                ANNError::message("IVF assignment source vectors are missing from the region")
            })?;
            let vectors = vectors.reborrow();
            let vectors = vectors.as_view().into_ann_result()?;

            for (&id, point) in view.members.iter().zip(vectors.rows()) {
                let dst = nearest(point, destinations.iter().map(|d| d.centroid));
                memberships[dst].push(id);
            }
        }

        let mut retained = BTreeSet::new();

        for (destination, members) in destinations.iter().zip(memberships) {
            match &destination.list {
                Type::Exists(id) => {
                    update.set_members(*id, &members)?;
                    retained.insert(*id);
                }
                Type::New => update.create(destination.centroid, &members)?,
            }
        }

        for &source in sources {
            if !retained.contains(&source) {
                update.retire(source)?;
            }
        }

        Ok(())
    }
}

/// Prefer the first centroid on ties.
fn nearest<'a>(point: &[f32], targets: impl Iterator<Item = &'a [f32]>) -> usize {
    let mut min = f32::MAX;
    let mut best = usize::MAX;
    for (i, center) in targets.enumerate() {
        let dist = SquaredL2::evaluate(point, center);
        if min > dist {
            best = i;
            min = dist;
        }
    }

    best
}
