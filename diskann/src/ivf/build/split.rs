/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Coordinate per-parent fitting and whole-region assignment.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};

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
}

impl<C, A> SplitAndAssign<C, A> {
    pub fn new(clusterer: C, assigner: A) -> Self {
        Self {
            clusterer,
            assigner,
        }
    }

    /// Compute the complete result before registering any provider writes.
    ///
    /// Concrete-reference clusterers may need an explicit `R` type argument because
    /// of a compiler limitation when inferring higher-ranked associated types.
    pub fn run<I: Copy, L: Copy, R, U>(
        &mut self,
        region: &R,
        update: &mut U,
        parents: &[L],
        neighbors: &[L],
    ) -> ANNResult<()>
    where
        A: Assigner<I, L, R>,
        C: Clusterer<I, L, R>,
        R: RegionView<I, L>,
        U: RegionUpdate<I, L>,
    {
        if parents.is_empty() {
            return Err(ANNError::message("IVF splitting needs at least one parent"));
        }

        let rows = self
            .clusterer
            .num_centers(parents.len())
            .ok_or_else(|| ANNError::message("IVF centroid count overflow"))?;

        let mut clusters =
            rowmajor::Owned::try_from_element(rows, self.clusterer.dim().get(), f32::NAN)?;

        self.clusterer
            .fit(parents, region, clusters.as_view_mut())?;

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
