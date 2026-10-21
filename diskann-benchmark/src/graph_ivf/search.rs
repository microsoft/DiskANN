/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Static search adapter until `diskann::ivf` exposes its own search API.
//! DiskANN routes to centroid IDs; flat search scans only their posting lists.

use std::num::NonZeroUsize;

use diskann::{
    flat::{knn_search, DistancesUnordered},
    graph::{glue::CopyIds, search::Knn, search_output_buffer::IdDistance, SearchOutputBuffer},
    provider::{DefaultContext, HasId},
    ANNResult,
};
use diskann_providers::{
    index::diskann_async::{self, MemoryIndex},
    model::graph::provider::async_::{
        common,
        inmem::{DefaultProviderParameters, SetStartPoints},
    },
};
use diskann_vector::{
    distance::{Metric, SquaredL2},
    PureDistanceFunction,
};

use super::provider::Store;
use crate::inputs::graph_ivf::Graph;

pub(super) struct CentroidGraph {
    index: MemoryIndex<f32>,
    list_ids: Vec<u32>,
}

impl CentroidGraph {
    pub async fn build(store: &Store, config: &Graph) -> anyhow::Result<Self> {
        let list_ids: Vec<_> = store.lists.keys().copied().collect();
        anyhow::ensure!(!list_ids.is_empty(), "no centroids to index");
        anyhow::ensure!(list_ids.len() < u32::MAX as usize, "too many centroids");
        let first = &store.lists[&list_ids[0]].centroid;
        let graph_config = config.config()?;
        let parameters = DefaultProviderParameters::simple(
            list_ids.len(),
            first.len(),
            Metric::L2,
            graph_config.max_degree_u32().get(),
        );
        let index =
            diskann_async::new_index::<f32, _>(graph_config, parameters, common::NoDeletes)?;
        index
            .provider()
            .set_start_points(std::iter::once(first.as_slice()))?;
        for (row, &list_id) in list_ids.iter().enumerate() {
            index
                .insert(
                    &common::FullPrecision,
                    &DefaultContext,
                    &(row as u32),
                    store.lists[&list_id].centroid.as_slice(),
                )
                .await?;
        }
        Ok(Self { index, list_ids })
    }

    pub async fn search(
        &self,
        store: &Store,
        query: &[f32],
        k: NonZeroUsize,
        nprobe: usize,
        l_search: usize,
    ) -> anyhow::Result<QueryResult> {
        let probes = nprobe.min(self.list_ids.len());
        let mut centroid_ids = vec![0; probes];
        let mut centroid_distances = vec![0.0; probes];
        let mut buffer = IdDistance::new(&mut centroid_ids, &mut centroid_distances);
        let graph_stats = self
            .index
            .search(
                Knn::new(l_search, None)?,
                &common::FullPrecision,
                &DefaultContext,
                query,
                &mut buffer,
            )
            .await?;
        let selected_count = buffer.current_len();
        let lists: Vec<_> = centroid_ids[..selected_count]
            .iter()
            .map(|&row| self.list_ids[row as usize])
            .collect();
        let mut ids = vec![0; k.get()];
        let mut distances = vec![0.0; k.get()];
        let mut visitor = PostingScan {
            store,
            lists: &lists,
            query,
        };
        let stats = knn_search(
            &mut visitor,
            k,
            CopyIds,
            query,
            &mut IdDistance::new(&mut ids, &mut distances),
        )
        .await?;
        ids.truncate(stats.result_count as usize);
        Ok(QueryResult {
            ids,
            scanned: stats.cmps as usize,
            lists: selected_count,
            centroid_comparisons: graph_stats.cmps as usize,
        })
    }
}

pub(super) struct QueryResult {
    pub ids: Vec<u32>,
    pub scanned: usize,
    pub lists: usize,
    pub centroid_comparisons: usize,
}

struct PostingScan<'a> {
    store: &'a Store,
    lists: &'a [u32],
    query: &'a [f32],
}

impl HasId for PostingScan<'_> {
    type Id = u32;
}

impl DistancesUnordered for PostingScan<'_> {
    type Error = diskann::ANNError;

    async fn distances_unordered<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: Send + FnMut(u32, f32),
    {
        for list in self.lists {
            for &id in &self.store.lists[list].members {
                f(id, SquaredL2::evaluate(self.query, self.store.vector(id)?));
            }
        }
        Ok(())
    }
}
