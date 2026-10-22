/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Coordinate per-parent fitting and whole-region assignment.

use std::{collections::BTreeSet, num::NonZeroUsize};

use diskann_utils::{
    Reborrow,
    views::rowmajor::{self, Matrix, MatrixMut},
};

use super::{
    assignment::{Assigner, Target, Type},
    clustering::Clusterer,
};
use crate::{
    ANNError, ANNResult,
    ivf::region::{RegionUpdate, RegionView},
};

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
