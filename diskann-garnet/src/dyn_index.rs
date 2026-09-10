/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    SearchResults,
    garnet::{Context, GarnetId},
    provider::{DynamicQuantization, GarnetProvider},
};
use diskann::{
    ANNResult,
    graph::{InplaceDeleteMethod, index::SearchStats, search},
    neighbor::Neighbor,
    provider::DataProvider,
    utils::VectorRepr,
};
use diskann_providers::index::wrapped_async::DiskANNIndex;

/// Type-erased version of `DiskANNIndex<GarnetProvider>`.
/// All vector data is passed as untyped byte slices.
pub(crate) trait DynIndex: Send + Sync {
    /// Inserts a vector with id into the index
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    /// Sets the attributes for a vector
    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    /// Deletes the attributes for a vector
    fn delete_attributes(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    /// Searches for the nearest neighbors of a vector
    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: search::Knn,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    /// Searches for the nearest neighbors of an existing vector in the index
    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: search::Knn,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    /// Filtered search for a vector
    fn filtered_search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: search::InlineFilterSearch,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    /// Filtered search for an existing vector in the index
    fn filtered_search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: search::InlineFilterSearch,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    /// Delete a vector from the index
    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    /// Return an approximate count of vectors in the index
    fn approximate_count(&self) -> u64;

    /// Return the maximum degree of the index graph
    fn max_degree(&self) -> usize;

    /// Set a start point if one doesn't already exist.
    /// If there is already a start point, this is a no-op.
    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()>;

    /// Check if a vector exists by its internal id.
    /// Returns true if the vector exists and false otherwise.
    fn internal_id_exists(&self, context: &Context, id: u32) -> bool;

    /// Check if a vector exists by its external id.
    /// Returns true if the vector exists false otherwise.
    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool;

    /// Train the quantizer.
    /// Returns true if training was successful and false otherwise.
    fn train_quantizer(&self, context: &Context) -> bool;

    /// Quantize a group of previously inserted vectors.
    /// This function will be called `task_count` times with `task_idx` as a zero-based
    /// identifier of the group. This will attempt to quantize `total_vectors / task_count`
    /// vectors and returns true if it was successful and false otherwise.
    fn backfill_quant_vectors(&self, context: &Context, task_idx: usize, task_count: usize)
    -> bool;

    /// Return `count` random vectors from the index.
    /// Returns true on success and false otherwise.
    fn random_members(&self, context: &Context, count: u32, output: &mut SearchResults<'_>)
    -> bool;

    /// Returns the neighbors of and distances from the target vector
    fn neighbors(&self, context: &Context, id: &GarnetId) -> ANNResult<Vec<Neighbor<GarnetId>>>;

    /// Log a message to Garnet. The context term can be used to scope the log
    /// message to an area (e.g. `Term::Quantized` for quantization related
    /// messages).
    fn log(&self, context: &Context, msg: &str);
}

impl<T: VectorRepr> DynIndex for DiskANNIndex<GarnetProvider<T>> {
    /// Inserts a type erased vector into the index.
    ///
    /// The data slice here must be aligned to `T` or this will panic.
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.insert(
            &DynamicQuantization,
            context,
            id,
            bytemuck::cast_slice::<u8, T>(data),
        )
    }

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .set_attributes(context, id, data)
            .map_err(|e| e.into())
    }

    fn delete_attributes(&self, context: &Context, id: &GarnetId) -> ANNResult<()> {
        self.inner
            .provider()
            .delete_attributes(context, id)
            .map_err(|e| e.into())
    }

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: search::Knn,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        self.search(params, &DynamicQuantization, context, query, output)
    }

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: search::Knn,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        // Look up internal ID
        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = self.inner.provider().get_full_vector(context, iid)?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector(context, data_bytes, params, output)
    }

    fn filtered_search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: search::InlineFilterSearch,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        self.search(params, &DynamicQuantization, context, query, output)
    }

    fn filtered_search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: search::InlineFilterSearch,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        // Look up internal ID
        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = self.inner.provider().get_full_vector(context, iid)?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.filtered_search_vector(context, data_bytes, params, output)
    }

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()> {
        self.inplace_delete(
            DynamicQuantization,
            context,
            id,
            3,
            InplaceDeleteMethod::TwoHopAndOneHop,
        )
    }

    fn approximate_count(&self) -> u64 {
        self.inner.provider().max_internal_id() as u64
    }

    fn max_degree(&self) -> usize {
        self.inner.provider().max_degree()
    }

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .maybe_set_start_point(context, bytemuck::cast_slice::<u8, T>(data))
            .map_err(|e| e.into())
    }

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool {
        self.inner.provider().vector_iid_exists(context, id)
    }

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool {
        self.inner.provider().vector_id_exists(context, id)
    }

    fn train_quantizer(&self, context: &Context) -> bool {
        self.inner.provider().train_quantizer(context)
    }

    fn backfill_quant_vectors(
        &self,
        context: &Context,
        task_idx: usize,
        task_count: usize,
    ) -> bool {
        self.inner
            .provider()
            .backfill_quant_vectors(context, task_idx, task_count)
    }

    fn random_members(
        &self,
        context: &Context,
        count: u32,
        output: &mut SearchResults<'_>,
    ) -> bool {
        self.inner.provider().random_members(context, count, output)
    }

    fn neighbors(&self, context: &Context, id: &GarnetId) -> ANNResult<Vec<Neighbor<GarnetId>>> {
        self.inner.provider().neighbors(context, id)
    }

    fn log(&self, context: &Context, msg: &str) {
        self.inner.provider().log(context, msg);
    }
}
