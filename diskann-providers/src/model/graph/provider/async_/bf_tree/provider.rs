/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fmt::Debug,
    future::Future,
    io::{Read, Write},
    num::NonZeroUsize,
    str::FromStr,
    sync::Arc,
};

use serde::{Deserialize, Serialize};

use bf_tree::{BfTree, Config};
use diskann::{
    ANNError, ANNResult,
    graph::{
        AdjacencyList, DiskANNIndex, SearchOutputBuffer,
        glue::{
            self, ExpandBeam, FillSet, InplaceDeleteStrategy, InsertStrategy, PruneStrategy,
            SearchExt, SearchStrategy,
        },
    },
    neighbor::Neighbor,
    provider::{
        Accessor, BuildDistanceComputer, BuildQueryComputer, DataProvider, DefaultContext,
        DelegateNeighbor, Delete, ElementStatus, HasId, NeighborAccessor, NeighborAccessorMut,
        NoopGuard, SetElement,
    },
    utils::{IntoUsize, VectorRepr},
};
use diskann_utils::{future::AsyncFriendly, views::MatrixView};
use diskann_vector::{DistanceFunction, distance::Metric};

use crate::model::{
    graph::provider::async_::{
        TableDeleteProviderAsync,
        bf_tree::{
            neighbor_provider::NeighborProvider, quant_vector_provider::QuantVectorProvider,
            vector_provider::VectorProvider,
        },
        common::{
            CreateDeleteProvider, FullPrecision, Hybrid, Internal, NoDeletes, NoStore, Panics,
        },
        distances,
        postprocess::{AsDeletionCheck, DeletionCheck, RemoveDeletedIdsAndCopy},
    },
    pq::{self, FixedChunkPQTable, NUM_PQ_CENTROIDS},
};

use crate::storage::{LoadWith, PQStorage, SaveWith};

use crate::storage::{StorageReadProvider, StorageWriteProvider};

/////////////////////
// BfTreeProvider //
/////////////////////

/// An Bf-tree based implementation of a [`DataProvider`] built around the idea of having up to
/// two layers of vector stores: a full-precision store and an optional quantized store.
/// This provider uses the identity mapping between external and internal vector IDs.
///
/// # Type Parameters:
///
/// * `T`: The primitive element type of the full-precision vector. This is typically some
///   type like `f32` or `half::f16`.
///
/// * `Q`: The full type of the quant vector store. This is not constrained by a trait and
///   rather relies on implementation for several concrete types, including:
///
///   - [`BfTreeQuantVectorProviderAsync`]: A Bf-Tree based PQ-based quantized vector store.
///   - [`NoStore`]: Disable quantization altogether. Note that this disables all
///     methods reached through quantization based [`Accessor`]s at compile-time.
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
/// * [`Hybrid`]: The strategies implemented by [`Hybrid`] can use a mix of quantized
///   and full-precision vectors.
///
///   - Search: During search, quantized vectors are used with reranking applied to the
///     results before returning.
///
///   - Insertion: Quantized vectors are used during the search phase. During the pruning
///     phase, a hybrid of quantized and full-precision vectors are used.
///
///     The ratio of full-precision and quantized vectors is controlled by the
///     `max_fp_vecs_per_prune` parameter, which adjusts the implementation of [`FillSet`].
///
/// # Examples
///
/// The following code demonstrates how to instantiate and use the `BfTreeProvider` in
/// a number of different scenarios.
///
/// ## Full-Precision Only - No Deletes
///
/// This example demonstrates how to create a `BfTreeProvider` that only supports
/// full-precision vectors.
/// ```
/// use diskann_providers::model::graph::provider::async_::{
///     bf_tree::{
///         BfTreeProvider, BfTreeProviderParameters
///     },
///     common::{NoStore, NoDeletes},
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
///
/// let parameters = BfTreeProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = BfTreeProvider::<f32, _>::new_empty(
///     parameters,
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
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///         bf_tree::{
///             BfTreeProvider, BfTreeProviderParameters
///     },
///     common::NoDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
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
/// let parameters = BfTreeProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = BfTreeProvider::<f32>::new_empty(
///     parameters,
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
/// use diskann_providers::model::{
///     pq::FixedChunkPQTable,
///     graph::provider::async_::{
///     bf_tree::{
///         BfTreeProvider, BfTreeProviderParameters
///     },
///     common::TableBasedDeletes,
///     },
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
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
/// let parameters = BfTreeProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_fp_vecs_per_fill: None,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let provider = BfTreeProvider::<f32, _, _>::new_empty(
///     parameters,
///     table,
///     TableBasedDeletes,
/// );
/// ```
pub struct BfTreeProvider<T, Q = QuantVectorProvider, D = NoDeletes>
where
    T: VectorRepr,
{
    // The quant vector store. If `Q == NoStore`, the quantized operations are disabled.
    //
    pub(super) quant_vectors: Q,

    // The full vector store.
    //
    pub(super) full_vectors: VectorProvider<T>,

    // Provider that holds the graph structure as neighbors of vectors.
    //
    pub(crate) neighbor_provider: NeighborProvider<u32>,

    // The delete provider. If `D == NoDeletes`, then delete related operations are disabled.
    //
    pub(super) deleted: D,

    // A parameter controlling hybrid pruning, where some set of full-precision vectors are
    // fetched and the rest are quantized vectors
    //
    pub(super) max_fp_vecs_per_fill: usize,

    // The metric to use for distances
    //
    pub(super) metric: Metric,
}

#[derive(Debug, Clone)]
pub struct BfTreeProviderParameters {
    // The maximum number of valid points that provider can hold.
    pub max_points: usize,

    // The number of start points (frozen points) for graph search entry.
    pub num_start_points: NonZeroUsize,

    // The dimension of the full-precision vectors.
    pub dim: usize,

    // The metric to use for distance computations
    pub metric: Metric,

    // If quantization is used, this parameter controls how many full-precision
    // vectors are retrieved for each [`FillSet`] operation
    pub max_fp_vecs_per_fill: Option<usize>,

    // The maximum number of neighbors to store for each vector
    pub max_degree: u32,

    // bf-tree config for vector provider
    pub vector_provider_config: Config,

    // bf-tree config for quant vector provider
    pub quant_vector_provider_config: Config,

    // bf-tree config for neighbor list provider
    pub neighbor_list_provider_config: Config,
}

pub type Index<T, D = NoDeletes> = Arc<DiskANNIndex<BfTreeProvider<T, NoStore, D>>>;
pub type QuantIndex<T, Q, D = NoDeletes> = Arc<DiskANNIndex<BfTreeProvider<T, Q, D>>>;

impl<T, Q, D> BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
{
    /// Construct a new, unpopulated data provider.
    ///
    /// # Arguments
    /// * `params`: An instance of [`BfTreeProviderParameters`] collecting shared
    ///   configuration information.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    /// * `delete_precursor`: A precursor type for the delete layer.
    /// * `neighbor_precursor`: A precursor type for the neighbor layer.
    ///   or the neighbor layer
    pub fn new_empty<TQ, TD>(
        params: BfTreeProviderParameters,
        quant_precursor: TQ,
        delete_precursor: TD,
    ) -> ANNResult<Self>
    where
        TQ: CreateQuantProvider<Target = Q>,
        TD: CreateDeleteProvider<Target = D>,
    {
        let num_start_points = params.num_start_points.get();

        Ok(Self {
            quant_vectors: quant_precursor.create(
                params.max_points,
                num_start_points,
                params.metric,
                params.quant_vector_provider_config,
            )?,
            full_vectors: VectorProvider::new_with_config(
                params.max_points,
                params.dim,
                num_start_points,
                params.vector_provider_config,
            )?,
            neighbor_provider: NeighborProvider::new_with_config(
                params.max_degree,
                params.neighbor_list_provider_config,
            )?,
            deleted: delete_precursor.create(params.max_points + num_start_points),
            max_fp_vecs_per_fill: params.max_fp_vecs_per_fill.unwrap_or(usize::MAX),
            metric: params.metric,
        })
    }

    /// Construct a new data provider with start points initialized.
    ///
    /// This is the primary constructor for `BfTreeProvider`. It creates the provider
    /// and sets the start points in one operation.
    ///
    /// # Arguments
    /// * `params`: An instance of [`BfTreeProviderParameters`] collecting shared
    ///   configuration information.
    /// * `start_points`: A matrix view containing the start point vectors. The number
    ///   of rows must match `params.num_start_points.get()`.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    /// * `delete_precursor`: A precursor type for the delete layer.
    ///
    /// # Type Constraints
    /// * `Self: StartPoint<T>` - The provider must implement the `StartPoint` trait.
    pub fn new<TQ, TD>(
        params: BfTreeProviderParameters,
        start_points: MatrixView<'_, T>,
        quant_precursor: TQ,
        delete_precursor: TD,
    ) -> ANNResult<Self>
    where
        Self: StartPoint<T>,
        TQ: CreateQuantProvider<Target = Q>,
        TD: CreateDeleteProvider<Target = D>,
    {
        // Early validation before allocating resources
        if start_points.nrows() != params.num_start_points.get() {
            return Err(ANNError::log_async_index_error(format!(
                "start_points matrix has {} rows, but params.num_start_points is {}",
                start_points.nrows(),
                params.num_start_points.get(),
            )));
        }

        let provider = Self::new_empty(params.clone(), quant_precursor, delete_precursor)?;
        provider.set_start_points(Hidden(()), start_points)?;
        {
            // Initialize all neighborhoods to be empty lists.
            // This is a temporary solution to the problem of trying to access
            // an uninitialized neighbor list in functions `consolidate_deletes` and
            // `consolidate_simple` and getting an error. This is a stop-gap solution
            // until BF-tree API is improved to handle `exists` queries.
            for i in 0..params.max_points {
                let vector_id = i as u32;
                provider.neighbor_provider.set_neighbors(vector_id, &[])?;
            }
        }
        Ok(provider)
    }

    // /// Return a predicate that can be applied to `Iter::filter` to remove start points
    // /// from an iterator of neighbors.
    // ///
    // /// This is used during post-processing
    // ///
    // pub(crate) fn is_not_start_point(&self) -> impl Fn(&Neighbor<u32>) -> bool {
    //     let range = self.full_vectors.start_point_range();
    //     move |neighbor| !range.contains(&neighbor.id.into_usize())
    // }

    /// Return a vector of starting points.
    pub fn starting_points(&self) -> ANNResult<Vec<u32>> {
        Ok(self.full_vectors.starting_points()?)
    }

    /// An iterator over all ids including start points (even if they are deleted).
    pub fn iter(&self) -> std::ops::Range<u32> {
        0..(self.full_vectors.total() as u32)
    }

    pub fn num_start_points(&self) -> usize {
        self.full_vectors.num_start_points
    }

    /// Return the maximum number of points (excluding frozen/start points)
    pub fn max_points(&self) -> usize {
        self.full_vectors.max_vectors
    }

    /// Return the vector dimension
    pub fn dim(&self) -> usize {
        self.full_vectors.dim()
    }

    /// Return the distance metric
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Return the maximum degree from the neighbor provider
    pub fn max_degree(&self) -> u32 {
        self.neighbor_provider.max_degree()
    }
}

impl<T, Q> BfTreeProvider<T, Q, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    /// A temporary method while development of deletion is in progress
    ///
    pub fn clear_delete_set(&self) {
        self.deleted.clear();
    }
}

impl<T, D> BfTreeProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
{
    /// Return the number of vector reads for full-precision and quant-vectors respectively
    ///
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (
            self.full_vectors.num_get_calls.get(),
            self.quant_vectors.num_get_calls.get(),
        )
    }
}

impl<T, D> BfTreeProvider<T, NoStore, D>
where
    T: VectorRepr,
{
    /// Return the number of vector reads for full-precision and quant-vectors respectively
    ///
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (self.full_vectors.num_get_calls.get(), 0)
    }
}

/// Allow `&BfTreeProvider` to implement `IntoIter`
///
impl<T, Q, D> IntoIterator for &BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
{
    type Item = u32;
    type IntoIter = std::ops::Range<u32>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A helper trait to select the quant vector store.
///
/// This is also implemented for [`NoStore`], which explicitly disables deletion
/// related functionality
///
pub trait CreateQuantProvider {
    // The type of the created quant provider.
    //
    type Target;

    // Create a quant provider capable of tracking `max_points` with and additional
    // `frozen_points` at the end.
    //
    fn create(
        self,
        max_points: usize,
        frozen_points: usize,
        metric: Metric,
        bf_tree_config: Config,
    ) -> ANNResult<Self::Target>;
}

impl CreateQuantProvider for NoStore {
    type Target = NoStore;
    fn create(
        self,
        _max_points: usize,
        _frozen_points: usize,
        _metric: Metric,
        _bf_tree_config: Config,
    ) -> ANNResult<Self::Target> {
        Ok(self)
    }
}

/// Allow a `FixedChunkPQTable` to be promoted to full quant vector store.
///
impl CreateQuantProvider for FixedChunkPQTable {
    type Target = QuantVectorProvider;
    fn create(
        self,
        max_points: usize,
        frozen_points: usize,
        metric: Metric,
        bf_tree_config: Config,
    ) -> ANNResult<Self::Target> {
        QuantVectorProvider::new_with_config(
            metric,
            max_points,
            frozen_points,
            self,
            bf_tree_config,
        )
    }
}

impl<T, Q, D> BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    pub fn neighbors(&self) -> &NeighborProvider<u32> {
        &self.neighbor_provider
    }
}

///////////////////
// Data Provider //
///////////////////

impl<T, Q, D> DataProvider for BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Context = DefaultContext;

    // The `BfTreeProvider` uses the identity map for IDs.
    //
    type InternalId = u32;

    // The `BfTreeProvider` uses the identity map for IDs.
    //
    type ExternalId = u32;

    // Use a general error type for now.
    //
    type Error = ANNError;

    // Translate an external id to its corresponding internal id.
    //
    fn to_internal_id(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        Ok(*gid)
    }

    // Translate an internal id its corresponding external id.
    //
    fn to_external_id(
        &self,
        _context: &DefaultContext,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        Ok(id)
    }
}

impl<T, Q, D> HasId for BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<'a, T, Q, D> DelegateNeighbor<'a> for BfTreeProvider<T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.neighbors()
    }
}

/// Support deletes when we have a valid delete provider.
///
impl<T, Q> Delete for BfTreeProvider<T, Q, TableDeleteProviderAsync>
where
    Q: AsyncFriendly,
    T: VectorRepr,
{
    fn release(
        &self,
        _: &DefaultContext,
        id: Self::InternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        // delete the vector from bf-tree
        if let Err(e) = self.neighbor_provider.delete_vector(id) {
            return std::future::ready(Err(e));
        }
        self.deleted.undelete(id.into_usize());
        // set its neighbors to an empty list in the neighbor provider
        // self.neighbor_provider.set_neighbors(id, &[]);
        let res = self
            .neighbor_provider
            .set_neighbors(id, &[])
            .map_err(|err| err.context(format!("resetting neighbors for undeleted id {}", id)));
        std::future::ready(res)
    }

    /// Delete an item by external ID
    ///
    #[inline]
    fn delete(
        &self,
        _context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send {
        self.deleted.delete(gid.into_usize());
        std::future::ready(Ok(()))
    }

    /// Check the status via external ID
    ///
    #[inline]
    fn status_by_external_id(
        &self,
        context: &DefaultContext,
        gid: &Self::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, Self::Error>> + Send {
        // NOTE: ID translation is the identity, so we can refer to `status_by_internal_id`.
        self.status_by_internal_id(context, *gid)
    }

    /// Check the status via internal ID
    ///
    #[inline]
    fn status_by_internal_id(
        &self,
        _context: &DefaultContext,
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

impl NeighborAccessor for &NeighborProvider<u32> {
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(self.get_neighbors(id, neighbors).map(|_| self))
    }
}

impl NeighborAccessorMut for &NeighborProvider<u32> {
    fn set_neighbors(
        self,
        vector_id: u32,
        neighbors: &[u32],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(self.set_neighbors(vector_id, neighbors).map(|_| self))
    }

    fn append_vector(
        self,
        vector_id: u32,
        new_neighbor_ids: &[u32],
    ) -> impl Future<Output = ANNResult<Self>> + Send {
        std::future::ready(
            self.append_vector(vector_id, new_neighbor_ids)
                .map(|_| self),
        )
    }
}

////////////////
// SetElement //
////////////////

/// Assign to both the full-precision and quant vector stores
///
impl<T, D> SetElement<[T]> for BfTreeProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type SetError = ANNError;
    type Guard = NoopGuard<u32>;

    /// Store the provided element in both the full-precision and quant vector stores.
    ///
    /// The process of storing the element in the quant store will compress the vector
    ///
    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: &[T],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        // First, try adding to the quant provider.
        //
        if let Err(err) = self.quant_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Next, add to the full precision provider.
        //
        if let Err(err) = self.full_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Success
        //
        std::future::ready(Ok(NoopGuard::new(*id)))
    }
}

/// Assign to just the full-precision store
///
impl<T, D> SetElement<[T]> for BfTreeProvider<T, NoStore, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type SetError = ANNError;
    type Guard = NoopGuard<u32>;

    /// Store the provided element in just the full-precision vector stores
    ///
    fn set_element(
        &self,
        _context: &Self::Context,
        id: &u32,
        element: &[T],
    ) -> impl Future<Output = Result<Self::Guard, Self::SetError>> + Send {
        // Add to the full precision provider
        //
        if let Err(err) = self.full_vectors.set_vector_sync(id.into_usize(), element) {
            return std::future::ready(Err(err));
        }

        // Success
        //
        std::future::ready(Ok(NoopGuard::new(*id)))
    }
}

//////////////////////
// StartPoint Trait //
//////////////////////

/// A struct with a private member that cannot be constructed outside of this module.
///
/// This is used to prevent users from calling internal methods directly.
pub struct Hidden(());

/// A trait for setting the start points of a BfTreeProvider.
///
/// This trait is implemented by `BfTreeProvider` variants that support setting start points.
/// The `Hidden` parameter ensures that users cannot call `set_start_points` directly;
/// they must go through the `BfTreeProvider::new` constructor which handles this internally.
pub trait StartPoint<T> {
    /// Set the start points of the provider.
    ///
    /// # Safety
    /// This method is internal and should not be called directly by users.
    /// Use `BfTreeProvider::new` instead.
    #[doc(hidden)]
    fn set_start_points(&self, hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()>;
}

////////////////////
// SetStartPoints //
////////////////////

/// Set start points for the BfTreeProvider with quantization.
///
/// This implementation sets both the full-precision and quantized vectors for each
/// start point, as well as initializing empty neighbor lists.
impl<T, D> StartPoint<T> for BfTreeProvider<T, QuantVectorProvider, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn set_start_points(&self, _hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()> {
        let start_point_ids = self.full_vectors.starting_points()?;
        if start_points.nrows() != start_point_ids.len() {
            return Err(ANNError::log_async_index_error(format!(
                "expected start_points to contain `{}` rows, instead it has {}",
                start_point_ids.len(),
                start_points.nrows(),
            )));
        }

        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            // Set the quantized vector
            self.quant_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            self.neighbor_provider.set_neighbors(id, &[])?;
        }

        Ok(())
    }
}

/// Set start points for the BfTreeProvider without quantization.
///
/// This implementation sets the full-precision vectors for each start point
/// and initializes empty neighbor lists.
impl<T, D> StartPoint<T> for BfTreeProvider<T, NoStore, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn set_start_points(&self, _hidden: Hidden, start_points: MatrixView<'_, T>) -> ANNResult<()> {
        let start_point_ids = self.full_vectors.starting_points()?;
        if start_points.nrows() != start_point_ids.len() {
            return Err(ANNError::log_async_index_error(format!(
                "expected start_points to contain `{}` rows, instead it has {}",
                start_point_ids.len(),
                start_points.nrows(),
            )));
        }

        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            self.neighbor_provider.set_neighbors(id, &[])?;
        }

        Ok(())
    }
}

//////////////////
// FullAccessor //
//////////////////

/// An accessor for retrieving full-precision vectors from the `BfTreeProvider`.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`BfTreeProvider`].
/// * [`ComputerAccessor`] for comparing full-precision distances.
/// * [`BuildQueryComputer`].
///
pub struct FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    /// The host provider.
    provider: &'a BfTreeProvider<T, Q, D>,
    /// A buffer to store retrieved elements.
    element: Box<[T]>,
}

impl<T, Q, D> HasId for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<T, Q, D> SearchExt for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, T, Q, D> FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    pub(crate) fn new(provider: &'a BfTreeProvider<T, Q, D>) -> Self {
        Self {
            provider,
            element: (0..provider.full_vectors.dim())
                .map(|_| T::default())
                .collect(),
        }
    }
}

impl<'a, T, Q, D> DelegateNeighbor<'a> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;

    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, Q, D> Accessor for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    /// The lifetime extended version requires an allocation.
    type Extended = Box<[T]>;

    /// This accessor returns a reference to a local copy of the vector.
    type Element<'a>
        = &'a [T]
    where
        Self: 'a;

    /// The reference version of `Element` is the same as `Element`.
    type ElementRef<'a> = &'a [T];

    // Choose to panic on an out-of-bounds access rather than propagate an error.
    //
    type GetError = Panics;

    /// Return the full-precision vector stored at index `i`.
    ///
    /// This function always completes synchronously
    ///
    #[inline(always)]
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB (undefined behavior) that can result from
        // potentially mixing unsynchronized reads and writes on the underlying memory
        //
        #[allow(clippy::expect_used)]
        self.provider
            .full_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
            .expect("Full vector provider failed to retrieve element");

        std::future::ready(Ok(&*self.element))
    }
}

impl<T, Q, D> BuildDistanceComputer for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type DistanceComputerError = Panics;
    type DistanceComputer = T::Distance;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        Ok(T::distance(
            self.provider.metric,
            Some(self.provider.full_vectors.dim()),
        ))
    }
}

impl<T, Q, D> BuildQueryComputer<[T]> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type QueryComputerError = Panics;
    type QueryComputer = T::QueryDistance;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(T::query_distance(from, self.provider.metric))
    }
}
impl<T, Q, D> ExpandBeam<[T]> for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
}

impl<T, Q, D> FillSet for FullAccessor<'_, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        for i in itr {
            set.entry(i).or_insert_with(|| {
                #[allow(clippy::expect_used)]
                self.provider
                    .full_vectors
                    .get_vector_sync(i.into_usize())
                    .expect("Full vector provider failed to retrieve element")
                    .into()
            });
        }
        Ok(())
    }
}

impl<'a, T, Q, D> AsDeletionCheck for FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

///////////////////
// QuantAccessor //
///////////////////

/// An accessor that retrieves the quantized portion of the [`BfTreeProvider`].
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the `BfTreeProvider`.
/// * [`BuildQueryComputer`].
///
pub struct QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
    element: Box<[u8]>,
}

impl<T, D> HasId for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<T, D> SearchExt for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }
}

impl<'a, T, D> QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    pub(crate) fn new(provider: &'a BfTreeProvider<T, QuantVectorProvider, D>) -> Self {
        Self {
            provider,
            element: (0..provider.quant_vectors.pq_chunks())
                .map(|_| u8::default())
                .collect(),
        }
    }
}

impl<'a, T, D> DelegateNeighbor<'a> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, D> Accessor for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// The extended element type requires an allocation.
    type Extended = Box<[u8]>;

    /// This accessor returns a reference to a local copy of the element.
    type Element<'a>
        = &'a [u8]
    where
        Self: 'a;

    /// The reference version of `Element` is simply `Element`.
    type ElementRef<'a> = &'a [u8];

    // ANNError on access failures in bf-tree
    //
    type GetError = ANNError;

    /// Return the quantized vector stored at index `i`.
    ///
    /// This function always completes synchronously.
    ///
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        let v = self
            .provider
            .quant_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
            .map(|_: ()| &*self.element);

        std::future::ready(v)
    }

    /// Perform a bulk operation
    ///
    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + FnMut(Self::ElementRef<'_>, Self::Id),
    {
        for i in itr {
            match self
                .provider
                .quant_vectors
                .get_vector_into(i.into_usize(), &mut self.element)
            {
                Ok(()) => f(&self.element, i),
                Err(e) => {
                    return std::future::ready(Err(e));
                }
            }
        }
        std::future::ready(Ok(()))
    }
}

impl<T, D> BuildQueryComputer<[T]> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type QueryComputerError = ANNError;
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.provider.quant_vectors.query_computer(from)
    }
}

impl<T, D> ExpandBeam<[T]> for QuantAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
}

impl<'a, T, D> AsDeletionCheck for QuantAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Checker = D;
    fn as_deletion_check(&self) -> &D {
        &self.provider.deleted
    }
}

/////////////////////
// Hybrid Accessor //
/////////////////////

/// A hybrid accessor that fetches a mixture of full-precision and quantized vectors during
/// pruning. This allows the application to trade full-precision fetches for accuracy.
///
/// This type implements the following traits:
///
/// * [`Accessor`] for the [`BfTreeProvider`].
/// * [`BuildDistanceComputer`] for computing distances among [`distances::pq::Hybrid`]
///   element types.
/// * [`FillSet`] for populating a mixture of full-precision and quant vectors.
///
pub struct HybridAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
}

impl<'a, T, D> HybridAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    fn new(provider: &'a BfTreeProvider<T, QuantVectorProvider, D>) -> Self {
        Self { provider }
    }
}

impl<T, D> HasId for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Id = u32;
}

impl<'a, T, D> DelegateNeighbor<'a> for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Delegate = &'a NeighborProvider<u32>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.provider.neighbors()
    }
}

impl<T, D> Accessor for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// Extending requires an allocation.
    type Extended = distances::pq::Hybrid<Vec<T>, Vec<u8>>;

    /// The [`distances::pq::Hybrid`] is an enum consisting of either a full-precision
    /// vector or a quantized vector.
    ///
    /// This accessor can return either.
    type Element<'a>
        = distances::pq::Hybrid<Vec<T>, Vec<u8>>
    where
        Self: 'a;

    /// The generalized reference form of `Element`.
    type ElementRef<'a> = distances::pq::Hybrid<&'a [T], &'a [u8]>;

    // Choose to panic on an out-of-bounds access rather than propagate an error.
    type GetError = Panics;

    /// The default behavior of `get_element` returns a full-precision vector. The
    /// implementation of [`FillSet`] is how the `max_fp_vecs_per_fill` is used
    ///
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        // SAFETY: We've decided to live with UB that can result from potentially mixing
        // unsynchronized reads and writes on the underlying memory.
        #[allow(clippy::expect_used)]
        std::future::ready(Ok(distances::pq::Hybrid::Full(
            self.provider
                .full_vectors
                .get_vector_sync(id.into_usize())
                .expect("Full vector provider failed to retrieve element"),
        )))
    }
}

impl<T, D> BuildDistanceComputer for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputerError = ANNError;
    type DistanceComputer = distances::pq::HybridComputer<T>;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        let metric = self.provider.quant_vectors.metric();
        Ok(distances::pq::HybridComputer::new(
            self.provider.quant_vectors.distance_computer()?,
            T::distance(metric, Some(self.provider.full_vectors.dim())),
        ))
    }
}

impl<T, D> FillSet for HybridAccessor<'_, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    /// Fill up to `max_fp_per_prune` as full precision. The rest are quantized.
    ///
    /// if a full-precision vector already exists regardless of whether a full-precision
    /// vector or quant vector is needed, it is left as-is
    ///
    async fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> Result<(), Self::GetError>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        let threshold = self.provider.max_fp_vecs_per_fill;
        itr.enumerate().for_each(|(i, id)| {
            let e = set.entry(id);

            // Below the threshold, we try to fetch full-precision vectors.
            //
            if i < threshold {
                // If the item already exists but is not full precision, make it full
                // precision.
                e.and_modify(|v| {
                    if !v.is_full() {
                        // SAFETY: We've decided to live with UB (undefined behavior) that
                        // can result from potentially mixing unsynchronized reads and
                        // writes on the underlying memory.
                        //
                        #[allow(clippy::expect_used)]
                        let vec = self
                            .provider
                            .full_vectors
                            .get_vector_sync(id.into_usize())
                            .expect("Full vector provider failed to retrieve element");
                        *v = distances::pq::Hybrid::Full(vec);
                    }
                })
                .or_insert_with(|| {
                    // Only invoke this method if the entry is not occupied.
                    //
                    // SAFETY: We've decided to live with UB (undefined behavior) that
                    // can result from potentially mixing unsynchronized reads and
                    // writes on the underlying memory.
                    //
                    #[allow(clippy::expect_used)]
                    let vec = self
                        .provider
                        .full_vectors
                        .get_vector_sync(id.into_usize())
                        .expect("Full vector provider failed to retrieve element");
                    distances::pq::Hybrid::Full(vec)
                });
            } else {
                // Otherwise, only insert into the cache if the entry is not occupied.
                //
                e.or_insert_with(|| {
                    // SAFETY: We've decided to live with UB (undefined behavior) that
                    // can result from potentially mixing unsynchronized reads and
                    // writes on the underlying memory.
                    //
                    #[allow(clippy::expect_used)]
                    distances::pq::Hybrid::Quant(
                        self.provider
                            .quant_vectors
                            .get_vector_sync(id.into_usize())
                            .expect("Fail to retrieve a quant vector during fillset"),
                    )
                });
            }
        });
        Ok(())
    }
}

////////////////
// Strategies //
////////////////

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<T, Q, D> SearchStrategy<BfTreeProvider<T, Q, D>, [T]> for Internal<FullPrecision>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type SearchAccessorError = Panics;
    type PostProcessor = RemoveDeletedIdsAndCopy;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<T, Q, D> SearchStrategy<BfTreeProvider<T, Q, D>, [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = T::QueryDistance;
    type SearchAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type SearchAccessorError = Panics;
    type PostProcessor = glue::Pipeline<glue::FilterStartPoints, RemoveDeletedIdsAndCopy>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// An [`glue::SearchPostProcess`] implementation that reranks PQ vectors.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rerank;

impl<'a, T, D> glue::SearchPostProcess<QuantAccessor<'a, T, D>, [T]> for Rerank
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type Error = Panics;

    fn post_process<I, B>(
        &self,
        accessor: &mut QuantAccessor<'a, T, D>,
        query: &[T],
        _computer: &pq::distance::QueryComputer<Arc<FixedChunkPQTable>>,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>>,
        B: SearchOutputBuffer<u32> + ?Sized,
    {
        let provider = &accessor.provider;
        let checker = accessor.as_deletion_check();
        let f = T::distance(provider.metric, Some(provider.full_vectors.dim()));

        // Filter before computing the full precision distances.
        let mut reranked: Vec<(u32, f32)> = candidates
            .filter_map(|n| {
                if checker.deletion_check(n.id) {
                    None
                } else {
                    #[allow(clippy::expect_used)]
                    let vec = provider
                        .full_vectors
                        .get_vector_sync(n.id.into_usize())
                        .expect("Full vector provider failed to retrieve element");
                    Some((n.id, f.evaluate_similarity(query, &vec)))
                }
            })
            .collect();

        // Sort the full precision distances.
        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        // Store the reranked results.
        std::future::ready(Ok(output.extend(reranked)))
    }
}

/// Perform a search entirely in the quantized space.
///
/// Starting points are not filtered out of the final results but results are reranked using
/// the full-precision data.
impl<T, D> SearchStrategy<BfTreeProvider<T, QuantVectorProvider, D>, [T]> for Internal<Hybrid>
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, T, D>;
    type SearchAccessorError = Panics;
    type PostProcessor = Rerank;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// Perform a search entirely in the quantized space.
///
/// Starting points are are filtered out of the final results and results are reranked using
/// the full-precision data.
impl<T, D> SearchStrategy<BfTreeProvider<T, QuantVectorProvider, D>, [T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type QueryComputer = pq::distance::QueryComputer<Arc<FixedChunkPQTable>>;
    type SearchAccessor<'a> = QuantAccessor<'a, T, D>;
    type SearchAccessorError = Panics;
    type PostProcessor = glue::Pipeline<glue::FilterStartPoints, Rerank>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        Ok(QuantAccessor::new(provider))
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

// Pruning
impl<T, Q, D> PruneStrategy<BfTreeProvider<T, Q, D>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type DistanceComputer = T::Distance;
    type PruneAccessor<'a> = FullAccessor<'a, T, Q, D>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(FullAccessor::new(provider))
    }
}

/// Implementing this trait allows `FullPrecision` to be used for multi-insert
///
impl<'a, T, Q, D> glue::AsElement<&'a [T]> for FullAccessor<'a, T, Q, D>
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly,
{
    type Error = diskann::error::Infallible;
    fn as_element(
        &mut self,
        vector: &'a [T],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(vector))
    }
}

impl<T, D> PruneStrategy<BfTreeProvider<T, QuantVectorProvider, D>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type DistanceComputer = distances::pq::HybridComputer<T>;
    type PruneAccessor<'a> = HybridAccessor<'a, T, D>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        Ok(HybridAccessor::new(provider))
    }
}

/// Implementing this trait allows `Hybrid` to be used for multi-insert.
///
impl<'a, T, D> glue::AsElement<&'a [T]> for HybridAccessor<'a, T, D>
where
    T: VectorRepr,
    D: AsyncFriendly,
{
    type Error = diskann::error::Infallible;
    fn as_element(
        &mut self,
        vector: &'a [T],
        _id: Self::Id,
    ) -> impl Future<Output = Result<Self::Element<'_>, Self::Error>> + Send {
        std::future::ready(Ok(distances::pq::Hybrid::Full(vector.to_vec())))
    }
}

impl<T, Q, D> InsertStrategy<BfTreeProvider<T, Q, D>, [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, D> InsertStrategy<BfTreeProvider<T, QuantVectorProvider, D>, [T]> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

/// Inplace Delete
///
impl<T, Q, D> InplaceDeleteStrategy<BfTreeProvider<T, Q, D>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    D: AsyncFriendly + DeletionCheck,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(Self)
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q, D>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        #[allow(clippy::expect_used)]
        let elt = provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .expect("Failed to get delete element")
            .into();
        Ok(elt)
    }
}

impl<T, D> InplaceDeleteStrategy<BfTreeProvider<T, QuantVectorProvider, D>> for Hybrid
where
    T: VectorRepr,
    D: AsyncFriendly + DeletionCheck,
{
    type DeleteElementError = Panics;
    type DeleteElement<'a> = [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type SearchStrategy = Internal<Self>;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Internal(*self)
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider, D>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        #[allow(clippy::expect_used)]
        let elt = provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .expect("Failed to get delete element")
            .into();
        Ok(elt)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BfTreeParams {
    pub bytes: usize,
    pub max_record_size: usize,
    pub leaf_page_size: usize,
}

impl BfTreeParams {
    /// Apply the saved BfTree parameters to a Config.
    pub fn apply(&self, config: &mut Config) {
        config.cb_max_record_size(self.max_record_size);
        config.leaf_page_size(self.leaf_page_size);
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuantParams {
    pub num_pq_bytes: usize,
    pub max_fp_vecs_per_fill: usize,
    pub params_quant: BfTreeParams,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SavedParams {
    pub max_points: usize,
    pub frozen_points: NonZeroUsize,
    pub dim: usize,
    pub metric: String,
    pub max_degree: u32,
    pub prefix: String,
    pub params_vector: BfTreeParams,
    pub params_neighbor: BfTreeParams,
    pub quant_params: Option<QuantParams>,
}

/// Helper struct for generating consistent file paths for BfTreeProvider persistence.
/// Centralizes all path patterns to avoid hardcoded strings throughout the codebase.
pub struct BfTreePaths;

impl BfTreePaths {
    /// Returns the path for the parameters JSON file
    pub fn params_json(prefix: &str) -> String {
        format!("{}_params.json", prefix)
    }

    /// Returns the path for the vectors BfTree file
    pub fn vectors_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_vectors.bftree", prefix))
    }

    /// Returns the path for the neighbors BfTree file
    pub fn neighbors_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_neighbors.bftree", prefix))
    }

    /// Returns the path for the quantized vectors BfTree file
    pub fn quant_bftree(prefix: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(format!("{}_quant.bftree", prefix))
    }

    /// Returns the path for the delete bitmap file
    pub fn delete_bin(prefix: &str) -> String {
        format!("{}_delete.bin", prefix)
    }

    /// Returns the path for the PQ pivots file
    pub fn pq_pivots_bin(prefix: &str) -> String {
        format!("{}_pq_pivots.bin", prefix)
    }
}

// SaveWith/LoadWith for BfTreeProvider with TableDeleteProviderAsync

impl<T> SaveWith<SavedParams> for BfTreeProvider<T, NoStore, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        storage: &P,
        saved_params: &SavedParams,
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        // Save only essential parameters as JSON
        {
            let params_filename = BfTreePaths::params_json(&saved_params.prefix);
            let params_json = serde_json::to_string(saved_params).map_err(|e| {
                ANNError::log_index_error(format!("Failed to serialize params: {}", e))
            })?;
            let mut params_writer = storage.create_for_write(&params_filename)?;
            params_writer.write_all(params_json.as_bytes())?;
        }

        // Save vectors and neighbors
        self.full_vectors.snapshot();
        self.neighbor_provider.snapshot();

        // Save delete bitmap
        {
            let filename = BfTreePaths::delete_bin(&saved_params.prefix);
            let bitmap_bytes = self.deleted.to_bytes();
            let mut writer = storage.create_for_write(&filename)?;
            writer.write_all(&bitmap_bytes)?;
        }

        Ok(0)
    }
}

impl<T> LoadWith<String> for BfTreeProvider<T, NoStore, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    type Error = ANNError;

    async fn load_with<P>(storage: &P, prefix: &String) -> Result<Self, Self::Error>
    where
        P: StorageReadProvider,
    {
        // Read SavedParams from JSON file
        let saved_params: SavedParams = {
            let params_filename = BfTreePaths::params_json(prefix);
            let mut params_reader = storage.open_reader(&params_filename)?;
            let mut params_json = String::new();
            params_reader.read_to_string(&mut params_json)?;
            serde_json::from_str(&params_json).map_err(|e| {
                ANNError::log_index_error(format!("Failed to deserialize params: {}", e))
            })?
        }; // params_reader is dropped here

        // Convert metric string back to Metric enum
        let metric = Metric::from_str(&saved_params.metric)
            .map_err(|e| ANNError::log_index_error(format!("Failed to parse metric: {}", e)))?;

        let vector_path = BfTreePaths::vectors_bftree(&saved_params.prefix);
        let mut vector_config = Config::new(&vector_path, saved_params.params_vector.bytes);
        saved_params.params_vector.apply(&mut vector_config);
        vector_config.storage_backend(bf_tree::StorageBackend::Std);

        let neighbor_path = BfTreePaths::neighbors_bftree(&saved_params.prefix);
        let mut neighbor_config = Config::new(&neighbor_path, saved_params.params_neighbor.bytes);
        saved_params.params_neighbor.apply(&mut neighbor_config);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);

        let vector_index =
            BfTree::new_from_snapshot(vector_config.clone(), None).map_err(super::ConfigError)?;
        let full_vectors = VectorProvider::<T>::new_from_bftree(
            saved_params.max_points,
            saved_params.dim,
            saved_params.frozen_points.get(),
            vector_index,
        );

        let adjacency_list_index =
            BfTree::new_from_snapshot(neighbor_config.clone(), None).map_err(super::ConfigError)?;
        let neighbor_provider =
            NeighborProvider::<u32>::new_from_bftree(saved_params.max_degree, adjacency_list_index);

        // Load delete bitmap
        let total_points = saved_params.max_points + saved_params.frozen_points.get();
        let filename = BfTreePaths::delete_bin(&saved_params.prefix);

        let deleted = if storage.exists(&filename) {
            let mut reader = storage.open_reader(&filename)?;
            let mut bitmap_bytes = Vec::new();
            reader.read_to_end(&mut bitmap_bytes)?;
            TableDeleteProviderAsync::from_bytes(&bitmap_bytes, total_points)
                .map_err(|e| ANNError::log_index_error(e))?
        } else {
            // If file doesn't exist, create a new empty delete provider
            TableDeleteProviderAsync::new(total_points)
        };

        Ok(Self {
            quant_vectors: NoStore,
            full_vectors,
            neighbor_provider,
            deleted,
            max_fp_vecs_per_fill: 0,
            metric,
        })
    }
}

impl<T> SaveWith<SavedParams> for BfTreeProvider<T, QuantVectorProvider, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(
        &self,
        storage: &P,
        saved_params: &SavedParams,
    ) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        // Save only essential parameters as JSON
        {
            let params_filename = BfTreePaths::params_json(&saved_params.prefix);
            let params_json = serde_json::to_string(saved_params).map_err(|e| {
                ANNError::log_index_error(format!("Failed to serialize params: {}", e))
            })?;
            let mut params_writer = storage.create_for_write(&params_filename)?;
            params_writer.write_all(params_json.as_bytes())?;
        }

        // Save vectors, neighbors, and quant vectors
        self.full_vectors.snapshot();
        self.neighbor_provider.snapshot();
        self.quant_vectors.snapshot();

        // Save PQ table metadata and data using PQStorage format
        let filename = BfTreePaths::pq_pivots_bin(&saved_params.prefix);
        let pq_storage = PQStorage::new(&filename, "", None);
        let pq_table = &self.quant_vectors.pq_chunk_table;
        pq_storage.write_pivot_data(
            pq_table.get_pq_table(),
            pq_table.get_centroids(),
            pq_table.get_chunk_offsets(),
            NUM_PQ_CENTROIDS,
            pq_table.get_dim(),
            storage,
        )?;

        // Save delete bitmap
        {
            let filename = BfTreePaths::delete_bin(&saved_params.prefix);
            let bitmap_bytes = self.deleted.to_bytes();
            let mut writer = storage.create_for_write(&filename)?;
            writer.write_all(&bitmap_bytes)?;
        }

        Ok(0)
    }
}

impl<T> LoadWith<String> for BfTreeProvider<T, QuantVectorProvider, TableDeleteProviderAsync>
where
    T: VectorRepr,
{
    type Error = ANNError;

    async fn load_with<P>(storage: &P, prefix: &String) -> Result<Self, Self::Error>
    where
        P: StorageReadProvider,
    {
        // Read SavedParams from JSON file
        let saved_params: SavedParams = {
            let params_filename = BfTreePaths::params_json(prefix);
            let mut params_reader = storage.open_reader(&params_filename)?;
            let mut params_json = String::new();
            params_reader.read_to_string(&mut params_json)?;
            serde_json::from_str(&params_json).map_err(|e| {
                ANNError::log_index_error(format!("Failed to deserialize params: {}", e))
            })?
        }; // params_reader is dropped here

        // Extract quant_params - required for quantized provider
        let quant_params = saved_params.quant_params.ok_or_else(|| {
            ANNError::log_index_error("Missing quant_params in saved params for quantized provider")
        })?;

        // Convert metric string back to Metric enum
        let metric = Metric::from_str(&saved_params.metric)
            .map_err(|e| ANNError::log_index_error(format!("Failed to parse metric: {}", e)))?;

        let vector_path = BfTreePaths::vectors_bftree(&saved_params.prefix);
        let mut vector_config = Config::new(&vector_path, saved_params.params_vector.bytes);
        saved_params.params_vector.apply(&mut vector_config);
        vector_config.storage_backend(bf_tree::StorageBackend::Std);

        let neighbor_path = BfTreePaths::neighbors_bftree(&saved_params.prefix);
        let mut neighbor_config = Config::new(&neighbor_path, saved_params.params_neighbor.bytes);
        saved_params.params_neighbor.apply(&mut neighbor_config);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);

        let quant_path = BfTreePaths::quant_bftree(&saved_params.prefix);
        let mut quant_config = Config::new(&quant_path, quant_params.params_quant.bytes);
        quant_params.params_quant.apply(&mut quant_config);
        quant_config.storage_backend(bf_tree::StorageBackend::Std);

        let vector_index =
            BfTree::new_from_snapshot(vector_config.clone(), None).map_err(super::ConfigError)?;
        let full_vectors = VectorProvider::<T>::new_from_bftree(
            saved_params.max_points,
            saved_params.dim,
            saved_params.frozen_points.get(),
            vector_index,
        );

        let adjacency_list_index =
            BfTree::new_from_snapshot(neighbor_config.clone(), None).map_err(super::ConfigError)?;
        let neighbor_provider =
            NeighborProvider::<u32>::new_from_bftree(saved_params.max_degree, adjacency_list_index);

        // Read PQ table from file using PQStorage format
        let filename = BfTreePaths::pq_pivots_bin(&saved_params.prefix);
        let pq_storage = PQStorage::new(&filename, "", None);
        let pq_table =
            pq_storage.load_pq_pivots_bin(&filename, quant_params.num_pq_bytes, storage)?;

        let quant_vector_index =
            BfTree::new_from_snapshot(quant_config.clone(), None).map_err(super::ConfigError)?;
        let quant_vectors = QuantVectorProvider::new_from_bftree(
            metric,
            saved_params.max_points,
            saved_params.frozen_points.get(),
            pq_table.clone(),
            quant_vector_index,
        );

        // Load delete bitmap
        let total_points = saved_params.max_points + saved_params.frozen_points.get();
        let filename = BfTreePaths::delete_bin(&saved_params.prefix);

        let deleted = if storage.exists(&filename) {
            let mut reader = storage.open_reader(&filename)?;
            let mut bitmap_bytes = Vec::new();
            reader.read_to_end(&mut bitmap_bytes)?;
            TableDeleteProviderAsync::from_bytes(&bitmap_bytes, total_points)
                .map_err(|e| ANNError::log_index_error(e))?
        } else {
            // If file doesn't exist, create a new empty delete provider
            TableDeleteProviderAsync::new(total_points)
        };

        Ok(Self {
            quant_vectors,
            full_vectors,
            neighbor_provider,
            deleted,
            max_fp_vecs_per_fill: quant_params.max_fp_vecs_per_fill,
            metric,
        })
    }
}

///////////
// Tests //
///////////

/// These unit tests target the correctness and functionality of Bf-Tree data provider
/// For functionality tests, Bf-Tree data provider runs against edge cases and is verified with hard-coded ground truth
/// For correctness tests, Bf-Tree data provider runs common workflows and is verified with the inmem data provider
///
/// Note that the test scenarios here should focus on assumptions made at the data provider level such as 'new vector should
/// have no neighbors'. Tests at individual index provider level should be added in the correponding provider mod instead while
/// tests regarding DiskANN async algorithms should be added to index::diskann_async
///
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::graph::provider::async_::common::TableBasedDeletes;
    use crate::storage::file_storage_provider::FileStorageProvider;

    #[tokio::test]
    async fn test_data_provider_and_delete_interface() {
        let ctx = &DefaultContext;
        let provider = BfTreeProvider::new_empty(
            BfTreeProviderParameters {
                max_points: 10,
                num_start_points: NonZeroUsize::new(2).unwrap(),
                dim: 5,
                metric: Metric::L2,
                max_fp_vecs_per_fill: None,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
            },
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        // Iterator
        //
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
            //
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
        //
        for i in iter.clone() {
            // set adjacency list to non-empty before release
            provider
                .neighbor_provider
                .set_neighbors(i, &[1, 2])
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
                .unwrap();
            assert!(neighbors.to_vec().is_empty());

            // Put it back to "deleted" to test `clear`.
            //
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
        //
        assert!(
            provider
                .set_element(ctx, &100, &[1.0, 2.0, 3.0, 4.0])
                .await
                .is_err()
        );
    }

    /// This functionality test targets scenarios of empty neighbor lists and ensures:
    /// 1. A new vector's neighbor list is empty
    /// 2. A vector's neighbor list could be set to empty
    /// 3. A non-existant vector's neighbor list is empty
    ///
    #[tokio::test]
    async fn test_empty_neighbor_list() {
        let num_points = 100u32;
        let ctx = &DefaultContext;
        let provider = BfTreeProvider::<f32, _, _>::new_empty(
            BfTreeProviderParameters {
                max_points: num_points as usize,
                num_start_points: NonZeroUsize::new(2).unwrap(),
                dim: 3,
                metric: Metric::L2,
                max_fp_vecs_per_fill: None,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
            },
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        let neighbor_accessor = &mut provider.neighbors();

        // Insert new vectors without neighbors and empty neighbor list is
        // expected for each newly inserted vector
        //
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            provider.set_element(ctx, &i, &vector).await.unwrap();

            // First attempt should fail as NotFound
            let mut out = AdjacencyList::new();
            assert!(neighbor_accessor.get_neighbors(i, &mut out).await.is_err());

            // After we set the empty neighbor list, our attempt should succeed
            neighbor_accessor.set_neighbors(i, &[]).await.unwrap();
            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert!(out.is_empty());
        }

        // Add a non-empty neighbor list for a vector and then set it to empty
        // In the end, an empty neighbor list is expected for the vector
        //
        for i in 0..num_points {
            let mut out = AdjacencyList::new();
            let neighbors = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
            neighbor_accessor
                .set_neighbors(i, &neighbors)
                .await
                .unwrap();

            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert_eq!(&*out, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

            neighbor_accessor.set_neighbors(i, &[]).await.unwrap();
            neighbor_accessor.get_neighbors(i, &mut out).await.unwrap();

            assert!(out.is_empty());
        }

        // Non-existant vectors have empty neighbor lists
        //
        let mut out = AdjacencyList::from_iter_untrusted([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

        // Attempt to access non-existant vector's neighbor list should fail as NotFound
        assert!(
            neighbor_accessor
                .get_neighbors(200, &mut out)
                .await
                .is_err()
        );
        assert!(out.is_empty());
    }

    ///////////////////////////////////////////////
    // SaveWith/LoadWith Tests for BfTree-based //
    ///////////////////////////////////////////////

    use tempfile::tempdir;

    /// Test saving and loading of BfTreeProvider without quantization, including deleted vertices
    #[tokio::test]
    async fn test_bf_tree_provider_save_load_no_quant() {
        let num_points = 50usize;
        let dim = 4usize;
        let max_degree = 32u32;
        let num_start_points = NonZeroUsize::new(2).unwrap();
        let ctx = &DefaultContext;

        // Create a temporary directory for test files
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        let prefix = temp_path
            .join("test_bf_tree_provider")
            .to_string_lossy()
            .to_string();
        // Create configs with storage backends
        let vector_path = BfTreePaths::vectors_bftree(&prefix);
        let neighbor_path = BfTreePaths::neighbors_bftree(&prefix);

        let bytes_vector = 1024 * 1024;
        let mut vector_config = Config::new(&vector_path, bytes_vector);
        vector_config.leaf_page_size(8192);
        vector_config.cb_max_record_size(1024);
        vector_config.storage_backend(bf_tree::StorageBackend::Std);

        let bytes_neighbor = 1024 * 1024;
        let mut neighbor_config = Config::new(&neighbor_path, bytes_neighbor);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);

        // Create provider parameters
        let params = BfTreeProviderParameters {
            max_points: num_points,
            num_start_points,
            dim,
            metric: Metric::L2,
            max_fp_vecs_per_fill: None,
            max_degree,
            vector_provider_config: vector_config.clone(),
            quant_vector_provider_config: Config::default(),
            neighbor_list_provider_config: neighbor_config.clone(),
        };

        // Create provider
        let provider = BfTreeProvider::<f32, NoStore, TableDeleteProviderAsync>::new_empty(
            params.clone(),
            NoStore,
            TableBasedDeletes,
        )
        .unwrap();

        // Populate provider with vectors
        for i in 0..num_points {
            let vector: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect();
            provider
                .set_element(ctx, &(i as u32), &vector)
                .await
                .unwrap();
        }

        // Populate provider with neighbor lists
        let neighbor_accessor = &mut provider.neighbors();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            neighbor_accessor
                .set_neighbors(i, &neighbors)
                .await
                .unwrap();
        }

        // Delete some vectors to test deletion persistence
        let deleted_ids = vec![5u32, 10u32, 15u32, 20u32, 25u32];
        for id in &deleted_ids {
            provider.delete(ctx, id).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, *id).await.unwrap(),
                ElementStatus::Deleted
            );
        }

        assert_eq!(vector_config.get_leaf_page_size(), 8192);
        assert_eq!(vector_config.get_cb_max_record_size(), 1024);

        let storage = FileStorageProvider;

        let metric_str = params.metric.as_str();
        let saved_params = SavedParams {
            max_points: params.max_points,
            frozen_points: params.num_start_points,
            dim: params.dim,
            metric: metric_str.to_string(),
            max_degree: params.max_degree,
            prefix: prefix.clone(),
            params_vector: BfTreeParams {
                bytes: bytes_vector,
                leaf_page_size: vector_config.get_leaf_page_size(),
                max_record_size: vector_config.get_cb_max_record_size(),
            },
            params_neighbor: BfTreeParams {
                bytes: bytes_neighbor,
                leaf_page_size: neighbor_config.get_leaf_page_size(),
                max_record_size: neighbor_config.get_cb_max_record_size(),
            },
            quant_params: None,
        };

        provider.save_with(&storage, &saved_params).await.unwrap();

        // Load using trait method (includes delete bitmap)
        let loaded_provider = BfTreeProvider::<f32, NoStore, TableDeleteProviderAsync>::load_with(
            &storage,
            &prefix.clone(),
        )
        .await
        .unwrap();

        // Verify vectors
        for i in 0..num_points as u32 {
            let original = provider.full_vectors.get_vector_sync(i as usize).unwrap();
            let loaded = loaded_provider
                .full_vectors
                .get_vector_sync(i as usize)
                .unwrap();
            assert_eq!(original, loaded, "Vector mismatch at index {}", i);
        }

        // Verify neighbor lists
        for i in 0..num_points as u32 {
            let mut original_list = AdjacencyList::new();
            let mut loaded_list = AdjacencyList::new();

            provider
                .neighbor_provider
                .get_neighbors(i, &mut original_list)
                .unwrap();
            loaded_provider
                .neighbor_provider
                .get_neighbors(i, &mut loaded_list)
                .unwrap();

            assert_eq!(
                &*original_list, &*loaded_list,
                "Neighbor list mismatch at index {}",
                i
            );
        }

        // Verify deleted status persists across save/load
        for id in &deleted_ids {
            assert_eq!(
                loaded_provider
                    .status_by_internal_id(ctx, *id)
                    .await
                    .unwrap(),
                ElementStatus::Deleted,
                "Deletion status not preserved for id {}",
                id
            );
        }

        // Verify non-deleted vectors remain valid
        for i in 0..num_points as u32 {
            if !deleted_ids.contains(&i) {
                assert_eq!(
                    loaded_provider.status_by_internal_id(ctx, i).await.unwrap(),
                    ElementStatus::Valid,
                    "Non-deleted vector {} incorrectly marked as deleted",
                    i
                );
            }
        }

        // Cleanup is automatic when temp_dir goes out of scope
    }

    /// Test saving and loading of BfTreeProvider with quantization, including deleted vertices
    #[tokio::test]
    async fn test_bf_tree_provider_save_load_quant() {
        let num_points = 50usize;
        let dim = 8usize;
        let max_degree = 32u32;
        let num_start_points = NonZeroUsize::new(2).unwrap();
        let ctx = &DefaultContext;

        // Create a temporary directory for test files
        let temp_dir = tempdir().unwrap();
        let temp_path = temp_dir.path();

        let prefix = temp_path
            .join("test_bf_tree_provider_quant")
            .to_string_lossy()
            .to_string();
        // Create configs with storage backends
        let vector_path = BfTreePaths::vectors_bftree(&prefix);
        let neighbor_path = BfTreePaths::neighbors_bftree(&prefix);
        let quant_path = BfTreePaths::quant_bftree(&prefix);

        let bytes_vector = 1024 * 1024;
        let mut vector_config = Config::new(&vector_path, bytes_vector);
        vector_config.storage_backend(bf_tree::StorageBackend::Std);

        let bytes_neighbor = 1024 * 1024;
        let mut neighbor_config = Config::new(&neighbor_path, bytes_neighbor);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);

        let bytes_quant = 1024 * 1024;
        let mut quant_config = Config::new(&quant_path, bytes_quant);
        quant_config.storage_backend(bf_tree::StorageBackend::Std);

        // Create PQ table
        let pq_table = FixedChunkPQTable::new(
            dim,
            vec![0.0; dim * 256].into_boxed_slice(),
            vec![0.0; dim].into_boxed_slice(),
            Box::new([0, 4, dim]),
            None,
        )
        .unwrap();

        // Create provider parameters
        let params = BfTreeProviderParameters {
            max_points: num_points,
            num_start_points,
            dim,
            metric: Metric::L2,
            max_fp_vecs_per_fill: Some(10),
            max_degree,
            vector_provider_config: vector_config.clone(),
            quant_vector_provider_config: quant_config.clone(),
            neighbor_list_provider_config: neighbor_config.clone(),
        };

        // Create provider with quantization
        let provider =
            BfTreeProvider::<f32, QuantVectorProvider, TableDeleteProviderAsync>::new_empty(
                params.clone(),
                pq_table.clone(),
                TableBasedDeletes,
            )
            .unwrap();

        // Populate provider with vectors
        for i in 0..num_points {
            let vector: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect();
            provider
                .set_element(ctx, &(i as u32), &vector)
                .await
                .unwrap();
        }

        // Populate provider with neighbor lists
        let neighbor_accessor = &mut provider.neighbors();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            neighbor_accessor
                .set_neighbors(i, &neighbors)
                .await
                .unwrap();
        }

        // Delete some vectors to test deletion persistence
        let deleted_ids = vec![3u32, 8u32, 15u32, 22u32, 30u32];
        for id in &deleted_ids {
            provider.delete(ctx, id).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, *id).await.unwrap(),
                ElementStatus::Deleted
            );
        }

        let storage = FileStorageProvider;

        // Create SavedParamsQuant outside of save_with
        let metric_str = params.metric.as_str();
        let num_pq_bytes = pq_table.get_num_chunks();
        let saved_params = SavedParams {
            max_points: params.max_points,
            frozen_points: params.num_start_points,
            dim: params.dim,
            metric: metric_str.to_string(),
            max_degree: params.max_degree,
            prefix: prefix.clone(),
            params_vector: BfTreeParams {
                bytes: bytes_vector,
                leaf_page_size: vector_config.get_leaf_page_size(),
                max_record_size: vector_config.get_cb_max_record_size(),
            },
            params_neighbor: BfTreeParams {
                bytes: bytes_neighbor,
                leaf_page_size: neighbor_config.get_leaf_page_size(),
                max_record_size: neighbor_config.get_cb_max_record_size(),
            },
            quant_params: Some(QuantParams {
                num_pq_bytes,
                max_fp_vecs_per_fill: params.max_fp_vecs_per_fill.unwrap_or(0),
                params_quant: BfTreeParams {
                    bytes: bytes_quant,
                    leaf_page_size: quant_config.get_leaf_page_size(),
                    max_record_size: quant_config.get_cb_max_record_size(),
                },
            }),
        };

        provider.save_with(&storage, &saved_params).await.unwrap();

        // Load using trait method (includes delete bitmap and quantization)
        let loaded_provider =
            BfTreeProvider::<f32, QuantVectorProvider, TableDeleteProviderAsync>::load_with(
                &storage,
                &prefix.clone(),
            )
            .await
            .unwrap();

        // Verify PQ table
        let original_pq = &provider.quant_vectors.pq_chunk_table;
        let loaded_pq = &loaded_provider.quant_vectors.pq_chunk_table;
        assert_eq!(
            original_pq.get_dim(),
            loaded_pq.get_dim(),
            "PQ table dim mismatch"
        );
        assert_eq!(
            original_pq.get_num_chunks(),
            loaded_pq.get_num_chunks(),
            "PQ table num_chunks mismatch"
        );
        assert_eq!(
            original_pq.get_num_centers(),
            loaded_pq.get_num_centers(),
            "PQ table num_centers mismatch"
        );
        assert_eq!(
            original_pq.get_pq_table(),
            loaded_pq.get_pq_table(),
            "PQ table data mismatch"
        );
        assert_eq!(
            original_pq.get_centroids(),
            loaded_pq.get_centroids(),
            "PQ table centroids mismatch"
        );
        assert_eq!(
            original_pq.get_chunk_offsets(),
            loaded_pq.get_chunk_offsets(),
            "PQ table chunk_offsets mismatch"
        );

        // Verify vectors
        for i in 0..num_points as u32 {
            let original = provider.full_vectors.get_vector_sync(i as usize).unwrap();
            let loaded = loaded_provider
                .full_vectors
                .get_vector_sync(i as usize)
                .unwrap();
            assert_eq!(original, loaded, "Vector mismatch at index {}", i);
        }

        // Verify quantized vectors
        for i in 0..num_points as u32 {
            let original = provider.quant_vectors.get_vector_sync(i as usize).unwrap();
            let loaded = loaded_provider
                .quant_vectors
                .get_vector_sync(i as usize)
                .unwrap();
            assert_eq!(original, loaded, "Quant vector mismatch at index {}", i);
        }

        // Verify neighbor lists
        for i in 0..num_points as u32 {
            let mut original_list = AdjacencyList::new();
            let mut loaded_list = AdjacencyList::new();

            provider
                .neighbor_provider
                .get_neighbors(i, &mut original_list)
                .unwrap();
            loaded_provider
                .neighbor_provider
                .get_neighbors(i, &mut loaded_list)
                .unwrap();

            assert_eq!(
                &*original_list, &*loaded_list,
                "Neighbor list mismatch at index {}",
                i
            );
        }

        // Verify deleted status persists across save/load
        for id in &deleted_ids {
            assert_eq!(
                loaded_provider
                    .status_by_internal_id(ctx, *id)
                    .await
                    .unwrap(),
                ElementStatus::Deleted,
                "Deletion status not preserved for id {}",
                id
            );
        }

        // Verify non-deleted vectors remain valid
        for i in 0..num_points as u32 {
            if !deleted_ids.contains(&i) {
                assert_eq!(
                    loaded_provider.status_by_internal_id(ctx, i).await.unwrap(),
                    ElementStatus::Valid,
                    "Non-deleted vector {} incorrectly marked as deleted",
                    i
                );
            }
        }

        // Cleanup is automatic when temp_dir goes out of scope
    }
}
