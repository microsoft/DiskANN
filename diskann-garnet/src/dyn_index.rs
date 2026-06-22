/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetQueryLabelProvider,
    provider::{DynamicQuantization, GarnetProvider},
};
use diskann::{
    ANNResult,
    graph::{InplaceDeleteMethod, index::SearchStats, search},
    provider::DataProvider,
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex, model::graph::provider::layers::BetaFilter,
};
use std::sync::Arc;

/// Type-erased version of `DiskANNIndex<GarnetProvider>`.
/// All vector data is passed as untyped byte slices.
pub(crate) trait DynIndex: Send + Sync {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    fn approximate_count(&self) -> u64;

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()>;

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool;

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool;

    fn train_quantizer(&self, context: &Context) -> bool;

    fn backfill_quant_vectors(&self, context: &Context, task_idx: usize, task_count: usize);
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

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        if let Some((labels, beta)) = filter {
            let beta_filter = BetaFilter::new(DynamicQuantization, Arc::new(labels.clone()), beta);
            self.search(*params, &beta_filter, context, query, output)
        } else {
            self.search(*params, &DynamicQuantization, context, query, output)
        }
    }

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        // Look up internal ID
        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = self.inner.provider().get_full_vector(context, iid)?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector(context, data_bytes, params, filter, output)
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

    fn backfill_quant_vectors(&self, context: &Context, task_idx: usize, task_count: usize) {
        self.inner
            .provider()
            .backfill_quant_vectors(context, task_idx, task_count);
    }
}
