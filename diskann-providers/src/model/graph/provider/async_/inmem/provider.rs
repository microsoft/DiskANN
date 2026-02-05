/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, future::Future, num::NonZeroUsize};

use crate::storage::{StorageReadProvider, StorageWriteProvider};
#[cfg(test)]
use diskann::neighbor::Neighbor;
use diskann::{
    ANNError, ANNResult,
    graph::AdjacencyList,
    provider::{
        DataProvider, DefaultAccessor, DefaultContext, Delete, ElementStatus, ExecutionContext,
        NeighborAccessor, NeighborAccessorMut, NoopGuard, SetElement,
    },
    utils::{IntoUsize, ONE, VectorRepr},
};
use diskann_utils::future::AsyncFriendly;
use diskann_vector::distance::Metric;

use crate::{
    model::graph::provider::async_::{
        SimpleNeighborProviderAsync, StartPoints, TableDeleteProviderAsync,
        common::{
            CreateDeleteProvider, CreateVectorStore, NoDeletes, NoStore, PrefetchCacheLineLevel,
            SetElementHelper, VectorStore,
        },
    },
    storage::{AsyncIndexMetadata, AsyncQuantLoadContext, DiskGraphOnly, LoadWith, SaveWith},
};

/////////////////////
// DefaultProvider //
/////////////////////

/// An in-memory implementation of a [`DataProvider`] built around the idea of having up to
/// two layers of vector stores: a base store and an auxiliary store.
///
/// This provider must be pre-configured with the number of elements it is going to contain
/// and uses the identity mapping between external and internal vector IDs.
///
/// In addition to a pre-configured number of points, this struct is also parameterized by
/// the concept of "frozen" point, which serve as the entry points for graph search.
/// Internally, these are stored consecutively just after the "max_points" slot.
///
/// # Type Parameters:
///
/// * `U`: The primary vector store that holds the main representation of vectors.
///   Typical use cases:
///   - Full precision vectors (e.g., [`FullPrecisionStore`])
///   - Quantized vectors when no higher fidelity representation is required
///   - May be `NoStore` if no base representation is required
///
/// * `V`: The auxiliary vector store that complements `base_vectors`.
///   Typical use cases:
///   - Quantized vectors when `base_vectors` holds full precision
///   - Alternative compressed formats (e.g., 1-bit scalar with 8-bit scalar)
///   - May be `NoStore` if no auxiliary representation is required
///
/// * `D`: The type of the deleted vector store. Like the quantized store, this is also
///   not constrained by a trait and rather relies on implementation for concrete types.
///   These are:
///
///   - [`NoDeletes`]: Do not support deletion at all (this disables implementation of
///     the [`Delete`] trait.
///   - [`TableDeleteProviderAsync`]: A bitmap storing deletion information.
///
/// * `Ctx`: A parameter controlling the [`ExecutionContext`] to be associated with this
///   provider. For the majority of cases, this is [`DefaultContext`], but is left as
///   a parameter to allow extension.
///
/// # Indexing Strategies
///
/// * [`FullPrecision`]: The strategies implemented by [`FullPrecision`] only retrieve data
///   from the full-precision portion of the index. No quantized vectors are used.
///
///   During search, start points are filtered from the final results.
///
/// * [`Quantized`]: The strategies implemented by [`Quantized`] can use a mix of quantized
///   and full-precision vectors.
///
///   - Search: During search, quantized vectors are used with reranking applied to the
///     results before returning.
///
///   - Insertion: Quantized vectors are used during the search phase. During the pruning
///     phase, a hybrid of quantized and full-precision vectors are used.
///
/// # Examples
///
/// The following code demonstrates how to instantiate and use the `DefaultProvider` in
/// a number of different scenarios.
///
/// ## Full-Precision Only - No Deletes
///
/// This example demonstrates how to create a `DefaultProvider` that only supports
/// full-precision vectors.
/// ```
/// use std::num::NonZeroUsize;
///
/// use diskann::provider::DefaultContext;
/// use diskann_providers::model::graph::provider::async_::{
///     inmem::{
///         DefaultProvider, DefaultProviderParameters,
///         CreateFullPrecision,
///     },
///     common::{NoStore, NoDeletes},
/// };
/// use diskann_vector::distance::Metric;
///
/// let dim = 4;
/// let prefetch_cache_line_level = None;
/// let parameters = DefaultProviderParameters {
///     max_points: 5,
///     frozen_points: NonZeroUsize::new(1).unwrap(),
///     dim,
///     metric: Metric::L2,
///     prefetch_lookahead: None,
///     max_degree: 40,
///     prefetch_cache_line_level,
/// };
///
/// // Create a table that supports 5 points and 1 "frozen" point.
/// let provider = DefaultProvider::<_, _, _, DefaultContext>::new_empty(
///     parameters,
///     CreateFullPrecision::<f32>::new(dim, prefetch_cache_line_level),
///     NoStore,
///     NoDeletes,
/// );
/// ```
///
/// ## Full-Precision and PQ - No Deletes
///
/// To create a two-level provider with a PQ-based quant vector store, a
/// [`FixedChunkPQTable`] can be supplied for the `quant_precursor` argument, as this
/// implements the [`CreateQuantProvider`] trait.
/// ```
/// use std::num::NonZeroUsize;
///
/// use diskann::provider::DefaultContext;
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///         inmem::{
///             DefaultProvider, DefaultProviderParameters,
///             CreateFullPrecision,
///         },
///         common::NoDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
///
/// // An example PQ table.
/// let dim = 4;
/// let table = FixedChunkPQTable::new(
///     dim,
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0, dim]),
///     None,
/// ).unwrap();
///
/// let prefetch_cache_line_level = None;
/// let parameters = DefaultProviderParameters {
///     max_points: 5,
///     frozen_points: NonZeroUsize::new(1).unwrap(),
///     dim,
///     metric: Metric::L2,
///     prefetch_lookahead: None,
///     max_degree: 40,
///     prefetch_cache_line_level,
/// };
///
/// // Create a table that supports 5 points and 1 "frozen" point.
/// let provider = DefaultProvider::<_, _, _, DefaultContext>::new_empty(
///     parameters,
///     CreateFullPrecision::<f32>::new(dim, prefetch_cache_line_level),
///     table,
///     NoDeletes,
/// );
/// ```
///
/// ## Full-Precision and PQ - With Deletes.
///
/// If deletes are desired, than the type [`TableBasedDeletes`] can be passed to the
/// constructor.
/// ```
/// use std::num::NonZeroUsize;
///
/// use diskann::provider::DefaultContext;
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///         inmem::{
///             DefaultProvider, DefaultProviderParameters,
///             CreateFullPrecision,
///         },
///         common::TableBasedDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
///
/// // An example PQ table.
/// let dim = 4;
/// let table = FixedChunkPQTable::new(
///     dim,
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0.0, 0.0, 0.0, 0.0]),
///     Box::new([0, dim]),
///     None,
/// ).unwrap();
///
/// let prefetch_cache_line_level = None;
/// let parameters = DefaultProviderParameters {
///     max_points: 5,
///     frozen_points: NonZeroUsize::new(1).unwrap(),
///     dim,
///     metric: Metric::L2,
///     prefetch_lookahead: None,
///     max_degree: 40,
///     prefetch_cache_line_level,
/// };
///
/// // Create a table that supports 5 points and 1 "frozen" point.
/// let provider = DefaultProvider::<_, _, _, DefaultContext>::new_empty(
///     parameters,
///     CreateFullPrecision::<f32>::new(dim, prefetch_cache_line_level),
///     table,
///     TableBasedDeletes,
/// );
/// ```
pub struct DefaultProvider<U, V = NoStore, D = NoDeletes, Ctx = DefaultContext> {
    /// The primary vector store that holds the main representation of vectors.
    pub base_vectors: U,

    /// The auxiliary vector store that complements `base_vectors`.
    pub aux_vectors: V,

    // Provider that holds the graph structure as neighbors of vectors.
    pub(crate) neighbor_provider: SimpleNeighborProviderAsync<u32>,

    /// The delete provider. If `D == NoDeletes`, then delete related operations are disabled.
    ///
    /// The size of this store must be kept in-sync with `quant_vectors` and `full-vectors`.
    pub(super) deleted: D,

    /// The metric to use for distances.
    pub(super) metric: Metric,

    pub(super) start_points: StartPoints,

    context: std::marker::PhantomData<Ctx>,
}

#[derive(Debug, Clone)]
pub struct DefaultProviderParameters {
    /// The maximum number of valid points that provider can hold.
    pub max_points: usize,

    /// The number of frozen-points (start points) to store. The two level provider
    /// stores these points starting at the linear index just after `max_points`.
    pub frozen_points: NonZeroUsize,

    /// The logical dimension of the full-precision data.
    pub dim: usize,

    /// The metric to use for distance computations.
    pub metric: Metric,

    /// The prefetch amount to use when performing bulk retrievals.
    ///
    /// Careful selection of this parameter can have a dramatic improvement on search
    /// performance.
    pub prefetch_lookahead: Option<usize>,

    pub prefetch_cache_line_level: Option<PrefetchCacheLineLevel>,

    /// The **actual** maximum number of neighbors to store for each vector.
    pub max_degree: u32,
}

impl DefaultProviderParameters {
    pub fn simple(max_points: usize, dim: usize, metric: Metric, max_degree: u32) -> Self {
        Self {
            max_points,
            frozen_points: ONE,
            metric,
            dim,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            max_degree,
        }
    }
}

impl<U, V, D, Ctx> DefaultProvider<U, V, D, Ctx> {
    /// Construct a new, unpopulated data provider.
    ///
    /// # Arguments
    /// * `params`: An instance of [`DefaultProviderParameters`] collecting shared
    ///   configuration information.
    /// * `base_precursor`: A precursor type for the base layer.
    /// * `aux_precursor`: A precursor type for the auxiliary layer.
    /// * `delete_precursor`: A precursor type for the delete layer.
    /// * `neighbor_precursor`: A precursor type for the neighbor layer.
    pub fn new_empty<CU, CV, CD>(
        params: DefaultProviderParameters,
        base_precursor: CU,
        aux_precursor: CV,
        delete_precursor: CD,
    ) -> ANNResult<Self>
    where
        CU: CreateVectorStore<Target = U>,
        CV: CreateVectorStore<Target = V>,
        CD: CreateDeleteProvider<Target = D>,
    {
        let npts = params.max_points + params.frozen_points.get();
        Ok(Self {
            base_vectors: base_precursor.create(npts, params.metric, params.prefetch_lookahead),
            aux_vectors: aux_precursor.create(npts, params.metric, params.prefetch_lookahead),
            neighbor_provider: SimpleNeighborProviderAsync::new(npts, 1, params.max_degree, 1.0),
            deleted: delete_precursor.create(npts),
            metric: params.metric,
            start_points: StartPoints::new(params.max_points as u32, params.frozen_points)?,
            context: std::marker::PhantomData,
        })
    }

    /// Return a predicate that can be applied to `Iter::filter` to remove start points
    /// from an iterator of neighbors.
    #[cfg(test)]
    pub(crate) fn is_not_start_point(&self) -> impl Fn(&Neighbor<u32>) -> bool {
        let range = self.start_points.range();
        move |neighbor| !range.contains(&neighbor.id)
    }

    /// Return a vector of starting points.
    pub fn starting_points(&self) -> ANNResult<Vec<u32>> {
        Ok(self.start_points.range().collect())
    }

    /// An iterator over all ids including start points (even if they are deleted).
    pub fn iter(&self) -> std::ops::Range<u32> {
        0..self.start_points.end()
    }

    /// Return a reference to the neighbor provider.
    pub fn neighbors(&self) -> &SimpleNeighborProviderAsync<u32> {
        &self.neighbor_provider
    }

    pub fn num_start_points(&self) -> usize {
        self.start_points.len()
    }

    /// Return the total capacity of the provider, **excluding** start points.
    pub fn capacity(&self) -> usize {
        self.start_points.start().into_usize()
    }

    /// Return the total capacity of the provider, **including** start points.
    pub fn total_points(&self) -> usize {
        self.start_points.end().into_usize()
    }
}

/// Allow `&DefaultProvider` to implement `IntoIter`.
impl<U, V, D, Ctx> IntoIterator for &DefaultProvider<U, V, D, Ctx> {
    type Item = u32;
    type IntoIter = std::ops::Range<u32>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<U, V, Ctx> DefaultProvider<U, V, TableDeleteProviderAsync, Ctx> {
    /// A temporary method while development of deletion is in progress.
    pub fn clear_delete_set(&self) {
        self.deleted.clear();
    }
}

impl<U, V, D, Ctx> DefaultProvider<U, V, D, Ctx>
where
    U: VectorStore,
    V: VectorStore,
{
    /// Return the number of vector reads for base vector and aux vector stores respectively.
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (
            self.base_vectors.count_for_get_vector(),
            self.aux_vectors.count_for_get_vector(),
        )
    }
}

pub trait SetStartPoints<T>
where
    T: ?Sized + 'static,
{
    fn set_start_points<'a, Itr>(&self, itr: Itr) -> ANNResult<()>
    where
        Itr: ExactSizeIterator<Item = &'a T> + 'a;
}

impl<T, U, V, D> SetStartPoints<[T]> for DefaultProvider<U, V, D>
where
    U: SetElementHelper<T>,
    V: SetElementHelper<T>,
    T: std::fmt::Debug + 'static,
{
    fn set_start_points<'a, Itr>(&self, itr: Itr) -> ANNResult<()>
    where
        Itr: ExactSizeIterator<Item = &'a [T]> + 'a,
    {
        let start_points = self.start_points.range();
        if itr.len() != start_points.len() {
            return Err(ANNError::log_async_index_error(format!(
                "expected `itr` to contain `{}` items, instead it has {}",
                start_points.len(),
                itr.len(),
            )));
        }

        for (i, v) in std::iter::zip(start_points, itr) {
            self.aux_vectors.set_element(&i, v)?;
            self.base_vectors.set_element(&i, v)?;
        }

        Ok(())
    }
}

////////////
// Saving //
////////////

impl<U, V, D, Ctx> SaveWith<(u32, AsyncIndexMetadata)> for DefaultProvider<U, V, D, Ctx>
where
    U: AsyncFriendly + SaveWith<AsyncIndexMetadata>,
    V: AsyncFriendly + SaveWith<AsyncIndexMetadata>,
    D: AsyncFriendly,
    ANNError: From<U::Error> + From<V::Error>,
    Ctx: ExecutionContext,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        provider: &P,
        auxiliary: &(u32, AsyncIndexMetadata),
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        self.base_vectors.save_with(provider, &auxiliary.1).await?;
        self.aux_vectors.save_with(provider, &auxiliary.1).await?;
        self.neighbor_provider
            .save_with(provider, auxiliary)
            .await?;
        Ok(())
    }
}

impl<U, V, D, Ctx> SaveWith<(u32, u32, DiskGraphOnly)> for DefaultProvider<U, V, D, Ctx>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Ok = ();
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        provider: &P,
        auxiliary: &(u32, u32, DiskGraphOnly),
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        self.neighbor_provider
            .save_with(provider, auxiliary)
            .await?;
        Ok(())
    }
}

/////////////
// Loading //
/////////////

impl<U, V, D, Ctx> LoadWith<AsyncQuantLoadContext> for DefaultProvider<U, V, D, Ctx>
where
    U: VectorStore + LoadWith<AsyncQuantLoadContext>,
    V: VectorStore + AsyncFriendly + LoadWith<AsyncQuantLoadContext>,
    D: AsyncFriendly + LoadWith<usize>,
    ANNError: From<U::Error> + From<V::Error> + From<D::Error>,
    Ctx: ExecutionContext,
{
    type Error = ANNError;

    async fn load_with<P>(provider: &P, ctx: &AsyncQuantLoadContext) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let base_vectors = U::load_with(provider, ctx).await?;
        let aux_vectors = V::load_with(provider, ctx).await?;
        let deleted = D::load_with(provider, &base_vectors.total()).await?;

        // Take the maximum of the two totals so that if either store is `NoStore`,
        // we still compute the correct overall number of points.
        let npts = std::cmp::max(base_vectors.total(), aux_vectors.total());

        let valid_points = npts
            .checked_sub(ctx.num_frozen_points.get())
            .ok_or_else(|| {
                ANNError::log_index_error(format_args!(
                    "Expected {} start points but the stored index only has {} total points",
                    ctx.num_frozen_points.get(),
                    base_vectors.total(),
                ))
            })?;
        let start_points = StartPoints::new(valid_points as u32, ctx.num_frozen_points)?;
        Ok(Self {
            base_vectors,
            aux_vectors,
            neighbor_provider: SimpleNeighborProviderAsync::load_with(provider, ctx).await?,
            deleted,
            metric: ctx.metric,
            start_points,
            context: std::marker::PhantomData,
        })
    }
}

impl LoadWith<usize> for NoDeletes {
    type Error = ANNError;

    async fn load_with<P>(_: &P, _num_points: &usize) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Ok(NoDeletes)
    }
}

impl LoadWith<usize> for TableDeleteProviderAsync {
    type Error = ANNError;

    async fn load_with<P>(_: &P, num_points: &usize) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        Ok(TableDeleteProviderAsync::new(*num_points))
    }
}

///////////////////
// Data Provider //
///////////////////

impl<U, V, D, Ctx> DataProvider for DefaultProvider<U, V, D, Ctx>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Context = Ctx;
    /// The `DefaultProvider` uses the identity map for IDs.
    type InternalId = u32;
    /// The `DefaultProvider` uses the identity map for IDs.
    type ExternalId = u32;
    /// Use a general error type for now.
    type Error = ANNError;

    /// Translate an external id to its corresponding internal id.
    fn to_internal_id(
        &self,
        _context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    /// Translate an internal id its corresponding external id.
    fn to_external_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        Ok(id)
    }
}

/// Support deletes when we have a valid delete provider.
impl<U, V, Ctx> Delete for DefaultProvider<U, V, TableDeleteProviderAsync, Ctx>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    Ctx: ExecutionContext,
{
    fn release(
        &self,
        _context: &Ctx,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.deleted.undelete(id.into_usize());
        let res = self
            .neighbor_provider
            .set_neighbors_sync(id.into_usize(), &[])
            .map_err(|err| err.context(format!("resetting neighbors for undeleted id {}", id)));
        std::future::ready(res)
    }

    /// Delete an item by external ID.
    #[inline]
    fn delete(
        &self,
        _context: &Ctx,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.deleted.delete(gid.into_usize());
        std::future::ready(Ok(()))
    }

    /// Check the status via external ID.
    #[inline]
    fn status_by_external_id(
        &self,
        context: &Ctx,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        // NOTE: ID translation is the identity, so we can refer to `status_by_internal_id`.
        self.status_by_internal_id(context, *gid)
    }

    /// Check the status via internal ID.
    #[inline]
    fn status_by_internal_id(
        &self,
        _context: &Ctx,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        let status = if self.deleted.is_deleted(id.into_usize()) {
            ElementStatus::Deleted
        } else {
            ElementStatus::Valid
        };
        std::future::ready(Ok(status))
    }
}

impl NeighborAccessor for &SimpleNeighborProviderAsync<u32> {
    async fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        self.get_neighbors_sync(id.into_usize(), neighbors)?;
        Ok(self)
    }
}

impl NeighborAccessorMut for &SimpleNeighborProviderAsync<u32> {
    async fn set_neighbors(self, id: u32, neighbors: &[u32]) -> ANNResult<Self> {
        self.set_neighbors_sync(id.into_usize(), neighbors)?;
        Ok(self)
    }

    async fn append_vector(self, id: u32, new_neighbor_ids: &[u32]) -> ANNResult<Self> {
        self.append_vector_sync(id.into_usize(), new_neighbor_ids)?;
        Ok(self)
    }
}

impl<U, V, D, Ctx> DefaultAccessor for DefaultProvider<U, V, D, Ctx>
where
    U: AsyncFriendly,
    V: AsyncFriendly,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type Accessor<'a> = &'a SimpleNeighborProviderAsync<u32>;
    fn default_accessor(&self) -> Self::Accessor<'_> {
        self.neighbors()
    }
}

////////////////
// SetElement //
////////////////

// Assign to both the base and aux vector stores.
impl<U, V, D, Ctx, T> SetElement<[T]> for DefaultProvider<U, V, D, Ctx>
where
    T: VectorRepr,
    U: AsyncFriendly + SetElementHelper<T>,
    V: AsyncFriendly + SetElementHelper<T>,
    D: AsyncFriendly,
    Ctx: ExecutionContext,
{
    type SetError = ANNError;
    type Guard = NoopGuard<u32>;

    /// Store the provided element in just the full-precision vector stores.
    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: &[T],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        // First try adding to the aux vector store
        if let Err(err) = self.aux_vectors.set_element(id, element) {
            return std::future::ready(Err(err));
        }

        // Next, add to the base vector store.
        if let Err(err) = self.base_vectors.set_element(id, element) {
            return std::future::ready(Err(err));
        }

        // Success.
        std::future::ready(Ok(NoopGuard::new(*id)))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::provider::async_::{
        common::{NoStore, TableBasedDeletes},
        inmem::CreateFullPrecision,
    };

    #[tokio::test]
    async fn test_data_provider_and_delete_interface() {
        let ctx = &DefaultContext;
        let provider = DefaultProvider::new_empty(
            DefaultProviderParameters {
                max_points: 10,
                frozen_points: NonZeroUsize::new(2).unwrap(),
                dim: 5,
                metric: Metric::L2,
                prefetch_lookahead: None,
                max_degree: (64.0 * 1.2) as u32,
                prefetch_cache_line_level: None,
            },
            CreateFullPrecision::<f32>::new(5, None),
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        // Iterator
        assert_eq!((&provider).into_iter(), 0..(10 + 2));

        let iter = provider.iter();
        for i in iter.clone() {
            assert_eq!(provider.to_external_id(ctx, i).unwrap(), i);
            assert_eq!(provider.to_internal_id(ctx, &i).unwrap(), i);
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );

            // Delete by external ID.
            provider.delete(ctx, &i).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Deleted
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Deleted
            );
        }

        // Call `release` to "undelete" it ID.
        for i in iter.clone() {
            // set adjacency list to non-empty before release
            provider
                .neighbor_provider
                .set_neighbors(i, &[1, 2])
                .await
                .unwrap();
            provider.release(ctx, i).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );
            // check that adjacency list was reset after release
            let mut neighbors = AdjacencyList::new();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut neighbors)
                .await
                .unwrap();
            assert!(neighbors.to_vec().is_empty());

            // Put it back to "deleted" to test `clear`.
            provider.delete(ctx, &i).await.unwrap();
        }

        provider.clear_delete_set();
        for i in iter.clone() {
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Valid
            );
            assert_eq!(
                provider.status_by_external_id(ctx, &i).await.unwrap(),
                ElementStatus::Valid
            );
        }

        // out-of-bound set-element fails.
        assert!(
            provider
                .set_element(ctx, &100, &[1.0, 2.0, 3.0, 4.0])
                .await
                .is_err()
        );
    }
}
