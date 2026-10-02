/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A staged and routed insert batch.

use diskann_utils::views::Matrix;

use super::{check_rows, first_repeat, index_error};
use crate::{
    ANNResult,
    error::ErrorExt,
    ivf::{
        dynamic::{CentroidIndex, Provider, StageElements},
        grouped::Grouped,
        update::{InsertionUpdate, PointDelta},
    },
    utils::VectorId,
};

/// Staged points, their canonical vectors, and their routes.
///
/// Batch position `i` refers to `ids[i]`, row `i` of `vectors`, and `routes[i]`.
pub(in crate::ivf) struct StagedBatch<Id, L> {
    pub(super) ids: Vec<Id>,
    pub(super) vectors: Matrix<f32>,
    /// The nearest live list of each point.
    pub(super) routes: Vec<L>,
    /// Batch positions grouped by route.
    pub(super) by_route: Grouped<L, usize>,
}

impl<Id: VectorId, L: VectorId> StagedBatch<Id, L> {
    /// Stage every point with the accessor and route it to its nearest live list.
    ///
    /// # Errors
    ///
    /// Fails if any accessor call fails or returns inconsistent data.
    pub(in crate::ivf) async fn stage<P, A, T>(
        accessor: &mut A,
        points: &[(P::ExternalId, T)],
    ) -> ANNResult<Self>
    where
        P: Provider<InternalId = Id, ListId = L>,
        A: StageElements<P, T>,
        T: Sync,
    {
        let dim = accessor.dim();
        if dim == 0 {
            return Err(index_error(
                "canonical vectors must have a non-zero dimension",
            ));
        }

        let mut vectors = Matrix::new(f32::NAN, points.len(), dim);
        let ids = accessor
            .stage(points, vectors.as_mut_view())
            .await
            .escalate("insert must stage every point")?;
        if ids.len() != points.len() {
            return Err(index_error(format!(
                "staging returned {} ids for {} points",
                ids.len(),
                points.len()
            )));
        }
        if let Some(id) = first_repeat(ids.iter().copied()) {
            return Err(index_error(format!(
                "staging returned point {id} more than once"
            )));
        }
        check_rows(&ids, &vectors)?;

        let centroids = accessor.centroids();
        let mut routes = Vec::with_capacity(ids.len());
        for vector in vectors.row_iter() {
            let list = centroids
                .select(vector, 1)
                .await
                .escalate("insert must route every point")?
                .first()
                .map(|selected| selected.id)
                .ok_or_else(|| index_error("centroid selection returned no list"))?;
            routes.push(list);
        }

        let by_route = Grouped::from_pairs(routes.iter().copied().zip(0..).collect());
        Ok(Self {
            ids,
            vectors,
            routes,
            by_route,
        })
    }

    /// Append every point to its route.
    pub(in crate::ivf) fn into_update(self) -> InsertionUpdate<Id, L> {
        let appends = self
            .ids
            .iter()
            .zip(&self.routes)
            .map(|(&id, &to)| PointDelta::Append { id, to })
            .collect();
        InsertionUpdate::new(Vec::new(), appends)
    }

    /// Lists that received points, ascending, with the number of points routed to
    /// each.
    pub(in crate::ivf) fn routed(&self) -> impl ExactSizeIterator<Item = (L, usize)> {
        self.by_route
            .iter()
            .map(|(list, positions)| (list, positions.len()))
    }
}
