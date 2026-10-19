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
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::region::{RegionUpdate, RegionView};
use crate::{ANNError, ANNResult};

/// Fit a parent's replacement centroids without assigning regional memberships.
pub(super) trait Clusterer<T = f32> {
    /// Fill `dst` with finite centroids of the canonical dimension.
    ///
    /// Implementations own initialization, convergence, empty-cluster handling,
    /// and reusable fitting scratch. `dst` has `num_centers()` rows and `dim()`
    /// columns. The number of centers must be at least two.
    fn fit(&mut self, points: rowmajor::Ref<'_, T>, dst: rowmajor::Mut<'_, T>) -> ANNResult<()>;

    fn num_centers(&self) -> NonZeroUsize;

    fn dim(&self) -> NonZeroUsize;
}

/// Assign the whole region, allowing policies that coordinate multiple sources.
pub(super) trait Assigner<I, L> {
    /// Assign each source member to a destination and record complete memberships.
    ///
    /// Create new destinations, replace retained memberships, and retire sources
    /// not retained as destinations. Validate and compute assignments before
    /// recording writes. A write error may leave partial private updates.
    fn assign<R, U>(
        &mut self,
        region: &R,
        update: &mut U,
        sources: &[L],
        destinations: &[Destination<'_, L>],
    ) -> ANNResult<()>
    where
        R: RegionView<I, L>,
        U: RegionUpdate<I, L>;
}

pub(super) enum Type<L> {
    /// Existing centroid.
    Exists(L),
    /// New centroid as a result of the split.
    New,
}

/// A borrowed centroid; only retained destinations have an existing list ID.
pub(super) struct Destination<'a, L> {
    pub list: Type<L>,
    pub centroid: &'a [f32],
}

pub(super) struct SplitAndAssign<C, A> {
    pub clusterer: C,
    pub assigner: A,
    scratch: Option<rowmajor::Owned<f32>>, // scratch for the centroids.
}

impl<C: Clusterer, A> SplitAndAssign<C, A> {
    pub(super) fn new(clusterer: C, assigner: A) -> Self {
        Self {
            clusterer,
            assigner,
            scratch: None,
        }
    }

    /// Initialize the scratch for a particular number of output cluster centroids.
    fn initialize(&mut self, num_jobs: NonZeroUsize) -> ANNResult<()> {
        let c = self.clusterer.num_centers();

        let rows = num_jobs
            .checked_mul(c)
            .ok_or_else(|| ANNError::message("IVF centroid count overflow"))?
            .get();

        let dim = self.clusterer.dim().get();

        self.scratch = Some(rowmajor::Owned::try_from_element(rows, dim, f32::NAN)?);

        Ok(())
    }
}

impl<C: Clusterer, A> SplitAndAssign<C, A> {
    /// Compute the complete result before registering any provider writes.
    pub(super) fn run<I: Copy, L: Copy + Ord, R, U>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &BTreeSet<L>,
        neighbors: &BTreeSet<L>,
    ) -> ANNResult<()>
    where
        A: Assigner<I, L>,
        R: RegionView<I, L>,
        U: RegionUpdate<I, L>,
    {
        self.initialize(NonZeroUsize::new(parents.len()).ok_or(ANNError::message("err"))?)?;

        let n_centroids = self.clusterer.num_centers().get();
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
                vecs,
                clusters.subview_mut(s..end).ok_or_else(|| {
                    ANNError::message("IVF centroid scratch range is out of bounds")
                })?,
            )?;

            s = end;
        }

        let mut destinations: Vec<_> = clusters
            .rows()
            .map(|centroid| Destination {
                list: Type::New,
                centroid,
            })
            .collect();

        for &n in neighbors {
            let view = region
                .get_view(n)
                .ok_or_else(|| ANNError::message("IVF neighbor is missing from the region"))?;

            destinations.push(Destination {
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
pub(super) struct TwoMeans {
    iterations: NonZeroUsize,
    dim: NonZeroUsize,
}

impl TwoMeans {
    pub(super) fn new(iterations: NonZeroUsize, dim: NonZeroUsize) -> Self {
        Self { iterations, dim }
    }
}

impl Clusterer for TwoMeans {
    fn fit(
        &mut self,
        points: rowmajor::Ref<'_, f32>,
        mut dst: rowmajor::Mut<'_, f32>,
    ) -> ANNResult<()> {
        let dim = self.dim().get();
        let num_centers = self.num_centers().get();

        if points.nrows() < num_centers {
            return Err(ANNError::message("an IVF split needs at least two points"));
        }
        if points.ncols() != dim || dst.ncols() != dim || dst.nrows() != num_centers {
            return Err(ANNError::message(
                "two-means needs canonical input rows and a two-row destination of the canonical dimension",
            ));
        }

        dst.as_mut_slice().copy_from_slice(
            points
                .subview(0..num_centers)
                .ok_or(ANNError::message("oh no!"))?
                .as_slice(),
        );

        let _ = lloyds(points, dst, self.iterations.get());

        Ok(())
    }

    fn num_centers(&self) -> NonZeroUsize {
        const { NonZeroUsize::new(2).unwrap() }
    }

    fn dim(&self) -> NonZeroUsize {
        self.dim
    }
}

pub(super) struct RegionalNearest;

impl<I: Copy, L: Copy + Ord> Assigner<I, L> for RegionalNearest {
    fn assign<R, U>(
        &mut self,
        region: &R,
        update: &mut U,
        sources: &[L],
        destinations: &[Destination<'_, L>],
    ) -> ANNResult<()>
    where
        R: RegionView<I, L>,
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
