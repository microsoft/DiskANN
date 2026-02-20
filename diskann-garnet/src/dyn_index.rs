use std::sync::Arc;

use diskann::{
    ANNError, ANNResult,
    graph::{
        InplaceDeleteMethod, SearchOutputBuffer, glue::SearchStrategy, index::SearchStats, search,
    },
    provider::{Accessor, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};

use crate::{
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetQueryLabelProvider,
    provider::{self, GarnetProvider},
};

/// Wraps `SearchResults` (FFI output buffer) with a provider and context so that
/// `SearchOutputBuffer<u32>` can be implemented: each internal ID (u32) is converted
/// to a `GarnetId` via `to_external_id` and written into the underlying buffer.
/// Start point (internal ID 0) is skipped since it has no external ID mapping.
///
/// TODO: Once diskann-providers >= 0.45.0 exports BetaAccessor/BetaComputer/Unwrap,
/// implement SearchStrategy<..., GarnetId> for BetaFilter directly and remove this wrapper.
struct FilteredSearchResults<'a, 'b, T: VectorRepr> {
    inner: &'a mut SearchResults<'b>,
    provider: &'a GarnetProvider<T>,
    context: &'a Context,
}

impl<T: VectorRepr> SearchOutputBuffer<u32> for FilteredSearchResults<'_, '_, T> {
    fn size_hint(&self) -> Option<usize> {
        self.inner.size_hint()
    }

    fn push(&mut self, id: u32, distance: f32) -> diskann::graph::BufferState {
        // Skip start point (internal ID 0) which has no external ID mapping
        if id == 0 {
            if let Some(sz) = self.inner.size_hint()
                && sz == 0
            {
                return diskann::graph::BufferState::Full;
            } else {
                return diskann::graph::BufferState::Available;
            }
        }

        let eid = self.provider.to_external_id(self.context, id).unwrap();

        self.inner.push(eid, distance)
    }

    fn current_len(&self) -> usize {
        self.inner.current_len()
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (u32, f32)>,
    {
        let initial = self.inner.current_len();

        for (id, dist) in itr {
            if self.push(id, dist).is_full() {
                break;
            }
        }

        self.inner.current_len() - initial
    }
}

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
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        if let Some((labels, beta)) = filter {
            let beta_filter = BetaFilter::new(FullPrecision, Arc::new(labels.clone()), beta);
            let provider = self.inner.provider();
            let mut filtered = FilteredSearchResults {
                inner: output,
                provider,
                context,
            };

            let stats = self.search(&beta_filter, context, query, params, &mut filtered)?;

            Ok(SearchStats {
                result_count: filtered.current_len() as u32,
                ..stats
            })
        } else {
            self.search(&FullPrecision, context, query, params, output)
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
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _, GarnetId>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        // Look up internal ID
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
