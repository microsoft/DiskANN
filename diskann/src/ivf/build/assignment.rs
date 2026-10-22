/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Assign regional memberships and record the corresponding updates.

use std::collections::BTreeSet;

use diskann_utils::{Reborrow, views::rowmajor::Matrix};
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use crate::{
    ANNError, ANNResult,
    error::IntoANNResult,
    ivf::region::{AsMatRef, RegionUpdate, RegionView},
};

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
