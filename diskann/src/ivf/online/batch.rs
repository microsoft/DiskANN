/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A staged and routed insert batch.

use diskann_utils::views::Matrix;

use super::{first_repeat, gather::read_rows, index_error};
use crate::{
    ANNResult,
    error::ErrorExt,
    ivf::{
        dynamic::{CentroidIndex, MaintenanceAccessor},
        grouped::Grouped,
        update::Appends,
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
    /// Stage every point, read its canonical vector, and route it to its nearest live
    /// list.
    ///
    /// # Errors
    ///
    /// Fails if any accessor call fails or returns inconsistent data.
    pub(in crate::ivf) async fn stage<A, T>(
        accessor: &mut A,
        points: &[(A::ExternalId, T)],
    ) -> ANNResult<Self>
    where
        A: MaintenanceAccessor<T, Id = Id, ListId = L>,
        T: Copy + Send + Sync,
    {
        let dim = accessor.dim();
        if dim == 0 {
            return Err(index_error(
                "canonical vectors must have a non-zero dimension",
            ));
        }

        let mut ids = Vec::with_capacity(points.len());
        for (external, element) in points {
            ids.push(
                accessor
                    .stage_insert(external, *element)
                    .await
                    .escalate("insert must stage every point")?,
            );
        }
        if let Some(id) = first_repeat(ids.iter().copied()) {
            return Err(index_error(format!(
                "staging returned point {id} more than once"
            )));
        }
        let vectors = read_rows(accessor, &ids, dim).await?;

        let centroids = accessor.centroids();
        let mut routes = Vec::with_capacity(ids.len());
        for vector in vectors.row_iter() {
            let plan = centroids
                .select(vector, 1)
                .await
                .escalate("insert must route every point")?;
            let list = plan
                .selected()
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

    /// Place every point on its route.
    pub(in crate::ivf) fn into_appends(self) -> Appends<Id, L> {
        Appends::new(&self.ids, &self.routes)
    }

    /// Lists that received points, ascending, with the number of points routed to
    /// each.
    pub(in crate::ivf) fn routed(&self) -> impl ExactSizeIterator<Item = (L, usize)> {
        self.by_route
            .iter()
            .map(|(list, positions)| (list, positions.len()))
    }
}
