/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetFilter,
    provider::{self, GarnetProvider},
};
use diskann::{
    ANNError, ANNResult,
    graph::{InplaceDeleteMethod, config, glue::SearchStrategy, index::SearchStats, search},
    provider::{Accessor, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
use std::sync::Arc;

/// Type-erased version of `DiskANNIndex<GarnetProvider>`.
/// All vector data is passed as untyped byte slices.
pub trait DynIndex: Send + Sync {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<&GarnetFilter>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<&GarnetFilter>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    fn approximate_count(&self) -> u64;

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()>;

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool;

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool;
}

impl<T: VectorRepr> DynIndex for DiskANNIndex<GarnetProvider<T>> {
    /// Inserts a type erased vector into the index.
    ///
    /// The data slice here must be aligned to `T` or this will panic.
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.insert(
            FullPrecision,
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
        filter: Option<&GarnetFilter>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);

        match filter {
            Some(GarnetFilter::Callback(provider, max_effort)) => {
                let ef = params.l_value().get();
                let effort_cap = std::cmp::max(ef, *max_effort);
                let two_queue = search::TwoQueueSearch::new(
                    *params,
                    provider,
                    effort_cap,
                    config::defaults::RESULT_SIZE_FACTOR,
                );
                let result = self.search(two_queue, &FullPrecision, context, query, output)?;
                Ok(result.stats)
            }
            Some(GarnetFilter::Bitmap(labels, beta)) => {
                let beta_filter = BetaFilter::new(FullPrecision, Arc::new(labels.clone()), *beta);
                self.search(*params, &beta_filter, context, query, output)
            }
            None => self.search(*params, &FullPrecision, context, query, output),
        }
    }

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<&GarnetFilter>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = rt.block_on(accessor.get_element(iid))?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector(context, data_bytes, params, filter, output)
    }

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()> {
        self.inplace_delete(
            FullPrecision,
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
}
