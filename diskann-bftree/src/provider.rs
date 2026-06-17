/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::Debug,
    future::Future,
    io::{Read, Write},
    num::NonZeroUsize,
    str::FromStr,
};

use diskann_quantization::{
    alloc::{GlobalAllocator, Poly},
    spherical::iface::{self as spherical_iface, try_deserialize, Opaque, Quantizer},
};
use serde::{Deserialize, Serialize};

use bf_tree::{BfTree, Config};
use diskann::{
    default_post_processor,
    error::{ErrorExt, Infallible, RankedError},
    graph::{
        glue::{
            self, Batch, CopyIds, DefaultPostProcessor, InplaceDeleteStrategy, InsertStrategy,
            MultiInsertStrategy, PruneStrategy, SearchStrategy,
        },
        strategy::{FullPrecision, Quantized},
        workingset::map,
        AdjacencyList, SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{DataProvider, DefaultContext, Delete, ElementStatus, HasId, NoopGuard, SetElement},
    utils::{IntoUsize, VectorRepr},
    ANNError, ANNResult,
};
use diskann_utils::{
    future::{AsyncFriendly, SendFuture},
    views::MatrixView,
};
use diskann_vector::{distance::Metric, DistanceFunction, PreprocessedDistanceFunction};

use super::{
    neighbors::{NeighborAccessor, NeighborProvider},
    quant::QuantVectorProvider,
    vectors::VectorProvider,
    AccessError, NoStore,
};
use diskann_providers::model::graph::provider::async_::distances::UnwrapErr;
use diskann_providers::storage::{LoadWith, SaveWith, StorageReadProvider, StorageWriteProvider};

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
///   - [`QuantVectorProvider`]: A Bf-Tree based spherical quantized vector store.
///   - [`NoStore`]: Disable quantization altogether. Note that this disables all
///     methods reached through quantization based [`Accessor`]s at compile-time.
///
/// # Indexing Strategies
///
/// * [`FullPrecision`]: Only retrieves data from the full-precision portion of the index.
///   No quantized vectors are used. During search, start points are filtered from the
///   final results.
///
/// * [`Quantized`]: Performs all operations (search, pruning, insert) entirely in the
///   quantized space using spherical distance functions. Post-processing copies candidate
///   IDs forward without reranking. Fastest option — full-precision vectors are not
///   touched at query time.
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
/// use diskann_bftree::provider::{
///     BfTreeProvider, BfTreeProviderParameters
/// };
/// use diskann_bftree::NoStore;
/// use diskann_vector::distance::Metric;
/// use diskann_utils::views::{Init, Matrix};
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
///
/// let parameters = BfTreeProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
///     graph_params: None,
///     use_snapshot: false,
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let start_points = Matrix::new(Init(|| 0.0f32), 1, 4);
/// let provider = BfTreeProvider::<f32, _>::new(
///     parameters,
///     start_points.as_view(),
///     NoStore,
/// );
/// ```
///
/// ## Full-Precision and Spherical Quantization
///
/// To create a two-level provider with a spherical quantization-based quant vector store,
/// a `Poly<dyn Quantizer>` can be supplied for the `quant_precursor` argument, as this
/// implements the [`CreateQuantProvider`] trait.
/// ```
/// use diskann_quantization::{
///     alloc::{GlobalAllocator, Poly, poly},
///     algorithms::TransformKind,
///     spherical::{iface, SphericalQuantizer, SupportedMetric, PreScale},
/// };
/// use diskann_utils::views::{Init, Matrix};
/// use diskann_bftree::provider::{
///     BfTreeProvider, BfTreeProviderParameters
/// };
/// use diskann_vector::distance::Metric;
/// use bf_tree::Config;
/// use std::num::NonZeroUsize;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let dim = 4;
/// let data = Matrix::new(Init(|| 1.0f32), 4, dim);
/// let mut rng = StdRng::seed_from_u64(42);
/// let sq = SphericalQuantizer::train(
///     data.as_view(), TransformKind::Null,
///     SupportedMetric::SquaredL2, PreScale::None,
///     &mut rng, GlobalAllocator,
/// ).unwrap();
/// let imp = iface::Impl::<1>::new(sq).unwrap();
/// let poly = Poly::new(imp, GlobalAllocator).unwrap();
/// let quantizer: Poly<dyn iface::Quantizer> = poly!(iface::Quantizer, poly);
///
/// let parameters = BfTreeProviderParameters {
///     max_points: 5,
///     num_start_points: NonZeroUsize::new(1).unwrap(),
///     dim: 4,
///     metric: Metric::L2,
///     max_degree: 32,
///     vector_provider_config: Config::default(),
///     quant_vector_provider_config: Config::default(),
///     neighbor_list_provider_config: Config::default(),
///     graph_params: None,
///     use_snapshot: false,
/// };
///
/// // Create a table that supports 5 points and 1 start point.
/// let start_points = Matrix::new(Init(|| 0.0f32), 1, 4);
/// let provider = BfTreeProvider::<f32, _>::new(
///     parameters,
///     start_points.as_view(),
///     quantizer,
/// );
/// ```
pub struct BfTreeProvider<T, Q = QuantVectorProvider>
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

    // The metric to use for distances
    //
    pub(super) metric: Metric,

    // Graph configuration parameters for persistence
    //
    pub(crate) graph_params: Option<GraphParams>,

    // Whether CPR snapshot support is enabled for this provider's trees.
    pub(crate) use_snapshot: bool,
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

    // The physical maximum degree (maximum neighbor list capacity per vertex).
    // Callers are responsible for applying any slack factor externally before
    // passing this value.
    pub max_degree: u32,

    // bf-tree config for vector provider.
    // bf-tree requires a minimum circular buffer size relative to leaf_page_size:
    //   - Cache-only mode: cb_size_byte >= 4 * leaf_page_size
    //   - Non cache-only mode: cb_size_byte >= 2 * leaf_page_size
    // The default leaf_page_size is 4096 and default cb_size_byte is 32MB.
    pub vector_provider_config: Config,

    // bf-tree config for quant vector provider.
    // Same minimum circular buffer constraint as above.
    pub quant_vector_provider_config: Config,

    // bf-tree config for neighbor list provider.
    // Same minimum circular buffer constraint as above.
    pub neighbor_list_provider_config: Config,

    // Optional graph configuration parameters for persistence
    pub graph_params: Option<GraphParams>,

    // Whether to enable CPR snapshot support on the underlying bf-trees.
    pub use_snapshot: bool,
}

impl<T, Q> BfTreeProvider<T, Q>
where
    T: VectorRepr,
{
    /// Construct a new data provider from empty. Callers of this are required to manually set start
    /// points before performing search tasks.
    ///
    /// This constructor for `BfTreeProvider` should be used when building and constructing from
    /// scratch.
    ///
    /// # Arguments
    /// * `params`: An instance of [`BfTreeProviderParameters`] collecting shared
    ///   configuration information.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    ///
    /// # Type Constraints
    /// * `Self: StartPoint<T>` - The provider must implement the `StartPoint` trait.
    fn new_empty<TQ>(params: BfTreeProviderParameters, quant_precursor: TQ) -> ANNResult<Self>
    where
        Self: StartPoint<T>,
        TQ: CreateQuantProvider<Target = Q>,
    {
        Ok(Self {
            quant_vectors: quant_precursor.create(params.quant_vector_provider_config)?,
            full_vectors: VectorProvider::new_with_config(
                params.max_points,
                params.dim,
                params.num_start_points.get(),
                params.vector_provider_config,
            )?,
            neighbor_provider: NeighborProvider::new_with_config(
                params.max_degree,
                params.neighbor_list_provider_config,
            )?,
            metric: params.metric,
            graph_params: params.graph_params,
            use_snapshot: params.use_snapshot,
        })
    }

    /// Construct a new data provider with start points initialized.
    ///
    /// This is the primary constructor for `BfTreeProvider` where start points are known from the
    /// beginning. It creates the provider and sets the start points in one operation.
    ///
    /// # Arguments
    /// * `params`: An instance of [`BfTreeProviderParameters`] collecting shared
    ///   configuration information.
    /// * `start_points`: A matrix view containing the start point vectors. The number
    ///   of rows must match `params.num_start_points.get()`.
    /// * `quant_precursor`: A precursor type for the quantizer layer.
    ///
    /// # Type Constraints
    /// * `Self: StartPoint<T>` - The provider must implement the `StartPoint` trait.
    pub fn new<TQ>(
        params: BfTreeProviderParameters,
        start_points: MatrixView<'_, T>,
        quant_precursor: TQ,
    ) -> ANNResult<Self>
    where
        Self: StartPoint<T>,
        TQ: CreateQuantProvider<Target = Q>,
    {
        // Early validation before allocating resources
        if start_points.nrows() != params.num_start_points.get() {
            return Err(ANNError::log_async_index_error(format!(
                "start_points matrix has {} rows, but params.num_start_points is {}",
                start_points.nrows(),
                params.num_start_points.get(),
            )));
        }

        let provider = Self::new_empty(params.clone(), quant_precursor)?;
        provider.set_start_points(Hidden(()), start_points)?;
        {
            // Initialize all neighborhoods to be empty lists.
            // This is a temporary solution to the problem of trying to access
            // an uninitialized neighbor list in functions `consolidate_deletes` and
            // `consolidate_simple` and getting an error. This is a stop-gap solution
            // until BF-tree API is improved to handle `exists` queries.
            let mut scratch = provider.neighbor_provider.scratch();
            for i in 0..params.max_points {
                let vector_id = i as u32;
                scratch.write_neighbors(vector_id, &[])?;
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
        self.full_vectors.starting_points()
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

impl<T> BfTreeProvider<T, QuantVectorProvider>
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

impl<T> BfTreeProvider<T, NoStore>
where
    T: VectorRepr,
{
    /// Return the number of vector reads for full-precision and quant-vectors respectively
    ///
    pub fn counts_for_get_vector(&self) -> (usize, usize) {
        (self.full_vectors.num_get_calls.get(), 0)
    }
}

/// Hard-delete implementation for `BfTreeProvider`.
///
/// This provider performs **hard deletes**: vector data is immediately and irrecoverably erased
/// from the underlying bf-tree storage when [`Delete::delete`] is called. This has an important
/// consequence for inplace delete: the [`InplaceDeleteMethod::VisitedAndTopK`] strategy is
/// **incompatible** with this provider because it requires reading the deleted vector's data
/// (via [`InplaceDeleteStrategy::get_delete_element`]) *after* the delete has already been
/// committed. Use [`InplaceDeleteMethod::OneHop`] or [`InplaceDeleteMethod::TwoHopAndOneHop`]
/// instead, as these strategies only require neighbor topology (which remains accessible).
impl<T, Q> Delete for BfTreeProvider<T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    fn release(
        &self,
        _context: &Self::Context,
        _id: Self::InternalId,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        std::future::ready(Ok(()))
    }

    fn delete(
        &self,
        _context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let id = *gid;
        self.full_vectors.delete_vector(id as usize);

        std::future::ready(Ok(()))
    }

    fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl std::future::Future<Output = Result<diskann::provider::ElementStatus, Self::Error>> + Send
    {
        self.status_by_internal_id(context, *gid)
    }

    fn status_by_internal_id(
        &self,
        _context: &Self::Context,
        id: Self::InternalId,
    ) -> impl std::future::Future<Output = Result<diskann::provider::ElementStatus, Self::Error>> + Send
    {
        let status = match self.full_vectors.get_vector_sync(id.into_usize()) {
            Ok(_) => Ok(ElementStatus::Valid),
            Err(RankedError::Transient(_)) => Ok(ElementStatus::Deleted),
            Err(RankedError::Error(e)) => Err(e),
        };
        std::future::ready(status)
    }
}

/// Allow `&BfTreeProvider` to implement `IntoIter`
///
impl<T, Q> IntoIterator for &BfTreeProvider<T, Q>
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
    fn create(self, bf_tree_config: Config) -> ANNResult<Self::Target>;
}

impl CreateQuantProvider for NoStore {
    type Target = NoStore;
    fn create(self, _bf_tree_config: Config) -> ANNResult<Self::Target> {
        Ok(self)
    }
}

/// Allow a `FixedChunkPQTable` to be promoted to full quant vector store.
///
impl CreateQuantProvider for Poly<dyn Quantizer> {
    type Target = QuantVectorProvider;
    fn create(self, bf_tree_config: Config) -> ANNResult<Self::Target> {
        QuantVectorProvider::new_with_config(self, bf_tree_config)
    }
}

impl<T, Q> BfTreeProvider<T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    pub fn neighbors(&self) -> &NeighborProvider<u32> {
        &self.neighbor_provider
    }
}

///////////////////
// Data Provider //
///////////////////

impl<T, Q> DataProvider for BfTreeProvider<T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
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

    // No insert-ID recovery.
    type Guard = NoopGuard<u32>;

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

impl<T, Q> HasId for BfTreeProvider<T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type Id = u32;
}

////////////////
// SetElement //
////////////////

/// Assign to both the full-precision and quant vector stores
///
impl<T> SetElement<&[T]> for BfTreeProvider<T, QuantVectorProvider>
where
    T: VectorRepr,
{
    type SetError = ANNError;

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
impl<T> SetElement<&[T]> for BfTreeProvider<T, NoStore>
where
    T: VectorRepr,
{
    type SetError = ANNError;

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
impl<T> StartPoint<T> for BfTreeProvider<T, QuantVectorProvider>
where
    T: VectorRepr,
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

        let mut scratch = self.neighbor_provider.scratch();
        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            self.quant_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            scratch.write_neighbors(id, &[])?;
        }

        Ok(())
    }
}

/// Set start points for the BfTreeProvider without quantization.
///
/// This implementation sets the full-precision vectors for each start point
/// and initializes empty neighbor lists.
impl<T> StartPoint<T> for BfTreeProvider<T, NoStore>
where
    T: VectorRepr,
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

        let mut scratch = self.neighbor_provider.scratch();
        for (id, v) in std::iter::zip(start_point_ids, start_points.row_iter()) {
            // Set the full-precision vector
            self.full_vectors.set_vector_sync(id.into_usize(), v)?;
            // Initialize empty neighbor list
            scratch.write_neighbors(id, &[])?;
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
/// * [`BuildQueryComputer`].
///
pub struct FullAccessor<'a, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    /// The host provider.
    provider: &'a BfTreeProvider<T, Q>,
    /// The fused query-distance computer.
    computer: T::QueryDistance,
    /// A buffer to store retrieved elements.
    element: Box<[T]>,
}

impl<'a, T, Q> FullAccessor<'a, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    pub(crate) fn new(provider: &'a BfTreeProvider<T, Q>, query: &[T]) -> Self {
        Self {
            provider,
            computer: T::query_distance(query, provider.metric),
            element: (0..provider.full_vectors.dim())
                .map(|_| T::default())
                .collect(),
        }
    }

    fn get_distance(&mut self, id: u32) -> Result<f32, AccessError> {
        self.provider
            .full_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
            .map(|_: ()| self.computer.evaluate_similarity(&self.element))
    }
}

impl<T, Q> HasId for FullAccessor<'_, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type Id = u32;
}

impl<T, Q> glue::SearchAccessor for FullAccessor<'_, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        for i in self.provider.starting_points()? {
            f(
                i,
                self.get_distance(i)
                    .escalate("starting point retrieval must succeed")?,
            )
        }
        Ok(())
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let mut neighbors = AdjacencyList::new();
        for n in ids {
            self.provider.neighbors().get_neighbors(n, &mut neighbors)?;
            for &id in neighbors.iter().filter(|i| pred.eval_mut(i)) {
                if let Some(distance) = self
                    .get_distance(id)
                    .allow_transient("skipping deleted vectors")?
                {
                    on_neighbors(id, distance)
                }
            }
        }
        Ok(())
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
///
pub struct QuantAccessor<'a, T>
where
    T: VectorRepr,
{
    provider: &'a BfTreeProvider<T, QuantVectorProvider>,
    /// The fused query-distance computer.
    computer: super::quant::QuantQueryComputer,
    /// A buffer to store retrieved elements.
    element: Box<[u8]>,
}

impl<'a, T> QuantAccessor<'a, T>
where
    T: VectorRepr,
{
    pub(crate) fn new(
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        query: &[T],
    ) -> ANNResult<Self> {
        let computer = provider.quant_vectors.query_computer(query)?;
        Ok(Self {
            provider,
            computer,
            element: (0..provider.quant_vectors.quantizer.bytes())
                .map(|_| u8::default())
                .collect(),
        })
    }

    fn get_distance(&mut self, id: u32) -> Result<f32, AccessError> {
        match self
            .provider
            .quant_vectors
            .get_vector_into(id.into_usize(), &mut self.element)
        {
            Ok(()) => self
                .computer
                .evaluate(&self.element)
                .map_err(RankedError::Error),
            Err(err) => Err(err),
        }
    }
}

impl<T> HasId for QuantAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<T> glue::SearchAccessor for QuantAccessor<'_, T>
where
    T: VectorRepr,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<u32>>> {
        std::future::ready(self.provider.starting_points())
    }

    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        for i in self.provider.starting_points()? {
            f(
                i,
                self.get_distance(i)
                    .escalate("starting point retrieval must succeed")?,
            )
        }
        Ok(())
    }

    async fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        mut pred: P,
        mut on_neighbors: F,
    ) -> ANNResult<()>
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(Self::Id, f32) + Send,
    {
        let mut neighbors = AdjacencyList::new();
        for n in ids {
            self.provider.neighbors().get_neighbors(n, &mut neighbors)?;
            for &id in neighbors.iter().filter(|i| pred.eval_mut(i)) {
                if let Some(distance) = self
                    .get_distance(id)
                    .allow_transient("skipping deleted vectors")?
                {
                    on_neighbors(id, distance)
                }
            }
        }
        Ok(())
    }
}

///////////////////////
// FullPruneAccessor //
///////////////////////

/// A [`glue::PruneAccessor`] for full-precision vectors in the `BfTreeProvider`.
pub struct FullPruneAccessor<'a, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    provider: &'a BfTreeProvider<T, Q>,
    neighbors: NeighborAccessor<'a, u32>,
    set: map::Map<u32, Box<[T]>, map::Ref<[T]>>,
    distance: T::Distance,
}

impl<'a, T, Q> FullPruneAccessor<'a, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    fn new(
        provider: &'a BfTreeProvider<T, Q>,
        set: map::Map<u32, Box<[T]>, map::Ref<[T]>>,
    ) -> Self {
        Self {
            provider,
            neighbors: provider.neighbor_provider.scratch(),
            set,
            distance: T::distance(provider.metric, Some(provider.full_vectors.dim())),
        }
    }
}

impl<T, Q> HasId for FullPruneAccessor<'_, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type Id = u32;
}

impl<'q, T, Q> glue::PruneAccessor for FullPruneAccessor<'q, T, Q>
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type ElementRef<'a> = &'a [T];

    type View<'a>
        = map::View<'a, u32, Box<[T]>, map::Ref<[T]>>
    where
        Self: 'a;

    type Distance<'a>
        = &'a T::Distance
    where
        Self: 'a;

    type Neighbors<'a>
        = diskann::provider::Neighbors<'a, NeighborAccessor<'q, u32>>
    where
        Self: 'a;

    fn fill<Itr>(
        &mut self,
        itr: Itr,
    ) -> impl SendFuture<ANNResult<(Self::View<'_>, Self::Distance<'_>)>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        let mut buf: Option<Box<[T]>> = None;

        let view = self.set.fill(itr, |i: u32| -> ANNResult<_> {
            let mut b = match buf.take() {
                Some(b) => b,
                None => std::iter::repeat_n(T::default(), self.provider.dim()).collect(),
            };

            match self
                .provider
                .full_vectors
                .get_vector_into(i.into_usize(), &mut b)
                .allow_transient("transient errors allowed during fill")?
            {
                Some(()) => Ok(Some(b)),
                None => {
                    buf = Some(b);
                    Ok(None)
                }
            }
        });

        let result = view.map(|v| (v, &self.distance));
        std::future::ready(result)
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        diskann::provider::Neighbors(&mut self.neighbors)
    }
}

////////////////////////
// QuantPruneAccessor //
////////////////////////

/// A [`glue::PruneAccessor`] for quantized vectors in the `BfTreeProvider`.
pub struct QuantPruneAccessor<'a, T>
where
    T: VectorRepr,
{
    provider: &'a BfTreeProvider<T, QuantVectorProvider>,
    neighbors: NeighborAccessor<'a, u32>,
    set: map::Map<u32, Owned>,
    distance: UnwrapErr<spherical_iface::DistanceComputer, spherical_iface::DistanceError>,
}

impl<'a, T> QuantPruneAccessor<'a, T>
where
    T: VectorRepr,
{
    fn new(
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        capacity: usize,
    ) -> ANNResult<Self> {
        let distance = provider
            .quant_vectors
            .distance_computer()
            .map(UnwrapErr::new)?;
        let set = map::Builder::new(map::Capacity::Default).build(capacity);
        Ok(Self {
            provider,
            neighbors: provider.neighbor_provider.scratch(),
            set,
            distance,
        })
    }
}

impl<T> HasId for QuantPruneAccessor<'_, T>
where
    T: VectorRepr,
{
    type Id = u32;
}

impl<'q, T> glue::PruneAccessor for QuantPruneAccessor<'q, T>
where
    T: VectorRepr,
{
    type ElementRef<'a> = Opaque<'a>;

    type View<'a>
        = map::View<'a, u32, Owned>
    where
        Self: 'a;

    type Distance<'a>
        = &'a UnwrapErr<spherical_iface::DistanceComputer, spherical_iface::DistanceError>
    where
        Self: 'a;

    type Neighbors<'a>
        = diskann::provider::Neighbors<'a, NeighborAccessor<'q, u32>>
    where
        Self: 'a;

    fn fill<Itr>(
        &mut self,
        itr: Itr,
    ) -> impl SendFuture<ANNResult<(Self::View<'_>, Self::Distance<'_>)>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
    {
        let mut buf: Option<Box<[u8]>> = None;
        let bytes = self.provider.quant_vectors.quantizer.bytes();

        let view = self.set.fill(itr, |i: u32| -> ANNResult<_> {
            let mut b = match buf.take() {
                Some(b) => b,
                None => std::iter::repeat_n(0, bytes).collect(),
            };

            match self
                .provider
                .quant_vectors
                .get_vector_into(i.into_usize(), &mut b)
                .allow_transient("transient errors allowed during fill")?
            {
                Some(()) => Ok(Some(Owned(b))),
                None => {
                    buf = Some(b);
                    Ok(None)
                }
            }
        });

        let result = view.map(|v| (v, &self.distance));
        std::future::ready(result)
    }

    fn neighbors(&mut self) -> Self::Neighbors<'_> {
        diskann::provider::Neighbors(&mut self.neighbors)
    }
}

/// An owned quantized vector that reborrows to [`Opaque`].
///
/// Unlike inmem providers (which hand back zero-copy references into a contiguous backing
/// array), bf_tree copies vector data out of the tree on every access. The
/// [`workingset::View`] trait requires `get` to return something that implements
/// `Reborrow<'short, Target = Opaque<'short>>`, so we need an owned type that bridges
/// bf_tree's copy-out model with the working set's reborrow expectation.
pub struct Owned(Box<[u8]>);

impl<'short> diskann_utils::Reborrow<'short> for Owned {
    type Target = Opaque<'short>;
    fn reborrow(&'short self) -> Self::Target {
        Opaque::new(&self.0)
    }
}

////////////////
// Strategies //
////////////////

/// Perform a search entirely in the full-precision space.
///
/// Starting points are not filtered out of the final results.
impl<'a, T, Q> SearchStrategy<'a, BfTreeProvider<T, Q>, &'a [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type SearchAccessor = FullAccessor<'a, T, Q>;
    type SearchAccessorError = Infallible;

    fn search_accessor(
        &'a self,
        provider: &'a BfTreeProvider<T, Q>,
        _context: &'a DefaultContext,
        query: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        Ok(FullAccessor::new(provider, query))
    }
}

impl<'a, T, Q> DefaultPostProcessor<'a, BfTreeProvider<T, Q>, &'a [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, CopyIds>);
}

// Pruning
impl<T, Q> PruneStrategy<BfTreeProvider<T, Q>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type PruneAccessor<'a> = FullPruneAccessor<'a, T, Q>;
    type PruneAccessorError = diskann::error::Infallible;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q>,
        _context: &'a DefaultContext,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let set = map::Builder::new(map::Capacity::Default).build(capacity);
        Ok(FullPruneAccessor::new(provider, set))
    }
}

impl<'a, T, Q> InsertStrategy<'a, BfTreeProvider<T, Q>, &'a [T]> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, Q, B> MultiInsertStrategy<BfTreeProvider<T, Q>, B> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
    B: for<'a> Batch<Element<'a> = &'a [T]> + Debug,
{
    type Seed = map::Builder<u32, map::Ref<[T]>>;
    type FinishError = diskann::error::Infallible;
    type PruneStrategy = Self;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &BfTreeProvider<T, Q>,
        _ctx: &DefaultContext,
        batch: &std::sync::Arc<B>,
        ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        let overlay = map::Overlay::from_batch(batch.clone(), ids);
        let builder = map::Builder::new(map::Capacity::Default).with_overlay(overlay);
        std::future::ready(Ok(builder))
    }

    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q>,
        _context: &'a DefaultContext,
        seed: &'a Self::Seed,
        capacity: usize,
    ) -> ANNResult<FullPruneAccessor<'a, T, Q>> {
        let set = seed.clone().build(capacity);
        Ok(FullPruneAccessor::new(provider, set))
    }
}

/// Inplace delete strategy using full-precision vectors.
///
/// # Compatibility
///
/// This strategy is used with [`InplaceDeleteMethod::OneHop`] and
/// [`InplaceDeleteMethod::TwoHopAndOneHop`]. It is **not compatible** with
/// [`InplaceDeleteMethod::VisitedAndTopK`] because `BfTreeProvider` performs hard deletes —
/// the vector data is erased before `get_delete_element` is called, causing it to fail.
impl<T, Q> InplaceDeleteStrategy<BfTreeProvider<T, Q>> for FullPrecision
where
    T: VectorRepr,
    Q: AsyncFriendly,
{
    type DeleteElementError = ANNError;
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = FullAccessor<'a, T, Q>;
    type SearchPostProcessor = CopyIds;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        Self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        CopyIds
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, Q>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        use diskann::error::ErrorExt;
        let elt = provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .escalate("get_delete_element: failed to read vector for inplace delete")?
            .into();
        Ok(elt)
    }
}

/// Perform a search entirely in the quantized space.
///
/// Starting points are not filtered out of the final results.
impl<'a, T> SearchStrategy<'a, BfTreeProvider<T, QuantVectorProvider>, &'a [T]> for Quantized
where
    T: VectorRepr,
{
    type SearchAccessor = QuantAccessor<'a, T>;
    type SearchAccessorError = ANNError;

    fn search_accessor(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        _context: &'a DefaultContext,
        query: &'a [T],
    ) -> Result<Self::SearchAccessor, Self::SearchAccessorError> {
        QuantAccessor::new(provider, query)
    }
}

impl<'a, T> DefaultPostProcessor<'a, BfTreeProvider<T, QuantVectorProvider>, &'a [T]> for Quantized
where
    T: VectorRepr,
{
    default_post_processor!(glue::Pipeline<glue::FilterStartPoints, Rerank>);
}

impl<'a, T> InsertStrategy<'a, BfTreeProvider<T, QuantVectorProvider>, &'a [T]> for Quantized
where
    T: VectorRepr,
{
    type PruneStrategy = Self;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }
}

impl<T, B> MultiInsertStrategy<BfTreeProvider<T, QuantVectorProvider>, B> for Quantized
where
    T: VectorRepr,
    B: glue::Batch,
    B: for<'a> Batch<Element<'a> = &'a [T]> + Debug,
{
    type Seed = ();
    type FinishError = diskann::error::Infallible;
    type PruneStrategy = Self;
    type InsertStrategy = Self;

    fn insert_strategy(&self) -> Self::InsertStrategy {
        *self
    }

    fn finish<Itr>(
        &self,
        _provider: &BfTreeProvider<T, QuantVectorProvider>,
        _ctx: &DefaultContext,
        _batch: &std::sync::Arc<B>,
        _ids: Itr,
    ) -> impl std::future::Future<Output = Result<Self::Seed, Self::FinishError>> + Send
    where
        Itr: ExactSizeIterator<Item = u32> + Send,
    {
        std::future::ready(Ok(()))
    }

    fn seeded_prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        _context: &'a DefaultContext,
        _seed: &'a (),
        capacity: usize,
    ) -> ANNResult<QuantPruneAccessor<'a, T>> {
        QuantPruneAccessor::new(provider, capacity)
    }
}

/// Inplace delete strategy using quantized vectors.
///
/// # Compatibility
///
/// Same constraint as [`FullPrecision`]'s impl: not compatible with
/// [`InplaceDeleteMethod::VisitedAndTopK`] due to hard deletes.
impl<T> InplaceDeleteStrategy<BfTreeProvider<T, QuantVectorProvider>> for Quantized
where
    T: VectorRepr,
{
    type DeleteElementError = ANNError;
    type DeleteElement<'a> = &'a [T];
    type DeleteElementGuard = Box<[T]>;
    type PruneStrategy = Self;
    type DeleteSearchAccessor<'a> = QuantAccessor<'a, T>;
    type SearchPostProcessor = Rerank;
    type SearchStrategy = Self;
    fn search_strategy(&self) -> Self::SearchStrategy {
        *self
    }

    fn prune_strategy(&self) -> Self::PruneStrategy {
        *self
    }

    fn search_post_processor(&self) -> Self::SearchPostProcessor {
        Rerank
    }

    async fn get_delete_element<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        _context: &'a DefaultContext,
        id: u32,
    ) -> Result<Self::DeleteElementGuard, Self::DeleteElementError> {
        use diskann::error::ErrorExt;
        provider
            .full_vectors
            .get_vector_sync(id.into_usize())
            .escalate("get_delete_element: failed to read vector for inplace delete")
            .map(Into::into)
    }
}

// Pruning
impl<T> PruneStrategy<BfTreeProvider<T, QuantVectorProvider>> for Quantized
where
    T: VectorRepr,
{
    type PruneAccessor<'a> = QuantPruneAccessor<'a, T>;
    type PruneAccessorError = ANNError;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a BfTreeProvider<T, QuantVectorProvider>,
        _context: &'a DefaultContext,
        capacity: usize,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        QuantPruneAccessor::new(provider, capacity)
    }
}

/// Post-processor that reranks quantized search results using full-precision distances.
#[derive(Debug, Default, Clone, Copy)]
pub struct Rerank;

impl<'a, T> glue::SearchPostProcess<QuantAccessor<'a, T>, &[T]> for Rerank
where
    T: VectorRepr,
{
    type Error = ANNError;

    fn post_process<I, B>(
        &self,
        accessor: &mut QuantAccessor<'a, T>,
        query: &[T],
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        use diskann::error::ErrorExt;
        let provider = accessor.provider;
        let f = T::distance(provider.metric, Some(provider.full_vectors.dim()));

        let mut reranked: Vec<(u32, f32)> = Vec::new();
        for n in candidates {
            match provider
                .full_vectors
                .get_vector_sync(n.id.into_usize())
                .allow_transient("stale candidate during rerank")
            {
                Ok(Some(vec)) => {
                    reranked.push((n.id, f.evaluate_similarity(query, &vec)));
                }
                Ok(None) => {
                    // Transient (deleted/missing) — skip this candidate.
                }
                Err(e) => return std::future::ready(Err(e)),
            }
        }

        reranked
            .sort_unstable_by(|a, b| (a.1).partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        std::future::ready(Ok(output.extend(reranked)))
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct BfTreeParams {
    pub bytes: usize,
    pub max_record_size: usize,
    pub leaf_page_size: usize,
}

impl BfTreeParams {
    /// Build a BfTree Config from the saved parameters and a file path.
    /// When `is_memory` is true, the config uses an in-memory storage backend,
    /// ensuring the circular buffer is at least as large as the bf-tree default.
    pub fn to_config(&self, path: &std::path::Path, is_memory: bool) -> Config {
        let mut config = Config::new(path, self.bytes);
        config.cb_max_record_size(self.max_record_size);
        config.leaf_page_size(self.leaf_page_size);
        if is_memory {
            config.storage_backend(bf_tree::StorageBackend::Memory);
        } else {
            config.storage_backend(bf_tree::StorageBackend::Std);
        }
        config
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuantParams {
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
    pub graph_params: Option<GraphParams>,
    /// Whether the original model was in-memory (`true`) or on-disk (`false`).
    pub is_memory: bool,
    /// Whether CPR snapshot support was enabled.
    #[serde(default)]
    pub use_snapshot: bool,
}

/// The element type of the full-precision vectors stored in the index.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VectorDtype {
    F32,
    F16,
    U8,
    I8,
}

/// A trait for mapping concrete vector element types to their [`VectorDtype`]
/// discriminant at compile time.
pub trait AsVectorDtype {
    const DATA_TYPE: VectorDtype;
}

impl AsVectorDtype for f32 {
    const DATA_TYPE: VectorDtype = VectorDtype::F32;
}

impl AsVectorDtype for half::f16 {
    const DATA_TYPE: VectorDtype = VectorDtype::F16;
}

impl AsVectorDtype for i8 {
    const DATA_TYPE: VectorDtype = VectorDtype::I8;
}

impl AsVectorDtype for u8 {
    const DATA_TYPE: VectorDtype = VectorDtype::U8;
}

/// Graph configuration parameters persisted alongside the index.
/// These are needed to reconstruct the `DiskANNIndex` config on load.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphParams {
    /// l_build is the search list size used during index construction.
    /// When inserting a new vector into the DiskANN graph, the algorithm
    /// performs a greedy search to find the best neighbors to connect to.
    /// l_build controls how many candidate nodes are tracked during that search.
    pub l_build: usize,
    /// alpha is the pruning aggressiveness parameter used during graph
    /// construction. During pruning, when deciding whether to keep a candidate
    /// neighbor k for node i, the algorithm checks if there's already a
    /// closer neighbor j that "occludes" k. The occlusion test is is governed by alpha.
    pub alpha: f32,
    /// backedge_ratio controls how many reverse (back) edges are added after
    /// pruning during graph construction.
    pub backedge_ratio: f32,
    /// vector_dtype indicates the data type of the vectors stored in the index, which is necessary for correctly interpreting the raw bytes of the vectors when loading the index from disk.
    pub vector_dtype: VectorDtype,
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

    /// Returns the path for the spherical quantizer data file
    pub fn quant_data_bin(prefix: &str) -> String {
        format!("{}_quant_data.bin", prefix)
    }
}

/// Save a BfTree to disk via CPR snapshot.
///
/// # Errors
///
/// Returns an error if `use_snapshot` is false, indicating the tree was not
/// configured with snapshot support.
fn save_bftree(
    tree: &BfTree,
    target_path: std::path::PathBuf,
    use_snapshot: bool,
) -> ANNResult<()> {
    if !use_snapshot {
        return Err(ANNError::log_index_error(
            "cannot snapshot a BfTree that was not configured with use_snapshot(true)",
        ));
    }
    tree.cpr_snapshot(&target_path);
    Ok(())
}

/// Load a BfTree from a CPR snapshot file.
///
/// The 0.5.x loader reconstructs config from the snapshot file header, so
/// no external `BfTreeParams` are needed.
fn load_bftree(snapshot_path: std::path::PathBuf, use_snapshot: bool) -> Result<BfTree, ANNError> {
    BfTree::new_from_cpr_snapshot(snapshot_path, use_snapshot, None, None, None)
        .map_err(|e| ANNError::from(super::ConfigError(e)))
}

//////////////////////
// Serialization    //
//////////////////////

impl<T> SaveWith<String> for BfTreeProvider<T, NoStore>
where
    T: VectorRepr,
{
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(&self, storage: &P, prefix: &String) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        let saved_params = SavedParams {
            max_points: self.max_points(),
            frozen_points: NonZeroUsize::new(self.num_start_points())
                .ok_or_else(|| ANNError::log_index_error("num_start_points is zero"))?,
            dim: self.dim(),
            metric: self.metric().as_str().to_string(),
            max_degree: self.max_degree(),
            prefix: prefix.clone(),
            params_vector: BfTreeParams {
                bytes: self.full_vectors.config().get_cb_size_byte(),
                max_record_size: self.full_vectors.config().get_cb_max_record_size(),
                leaf_page_size: self.full_vectors.config().get_leaf_page_size(),
            },
            params_neighbor: BfTreeParams {
                bytes: self.neighbor_provider.config().get_cb_size_byte(),
                max_record_size: self.neighbor_provider.config().get_cb_max_record_size(),
                leaf_page_size: self.neighbor_provider.config().get_leaf_page_size(),
            },
            quant_params: None,
            graph_params: self.graph_params.clone(),
            is_memory: self.full_vectors.config().is_memory_backend(),
            use_snapshot: self.use_snapshot,
        };

        debug_assert_eq!(
            self.full_vectors.config().is_memory_backend(),
            self.neighbor_provider.config().is_memory_backend(),
            "Vector and neighbor stores have mismatched storage backends"
        );

        {
            let params_filename = BfTreePaths::params_json(&saved_params.prefix);
            let params_json = serde_json::to_string(&saved_params).map_err(|e| {
                ANNError::log_index_error(format!("Failed to serialize params: {}", e))
            })?;
            let mut params_writer = storage.create_for_write(&params_filename)?;
            params_writer.write_all(params_json.as_bytes())?;
        }

        save_bftree(
            self.full_vectors.bftree(),
            BfTreePaths::vectors_bftree(&saved_params.prefix),
            self.use_snapshot,
        )?;
        save_bftree(
            self.neighbor_provider.bftree(),
            BfTreePaths::neighbors_bftree(&saved_params.prefix),
            self.use_snapshot,
        )?;

        Ok(0)
    }
}

impl<T> LoadWith<String> for BfTreeProvider<T, NoStore>
where
    T: VectorRepr,
{
    type Error = ANNError;

    async fn load_with<P>(storage: &P, prefix: &String) -> Result<Self, Self::Error>
    where
        P: StorageReadProvider,
    {
        let saved_params: SavedParams = {
            let params_filename = BfTreePaths::params_json(prefix);
            let mut params_reader = storage.open_reader(&params_filename)?;
            let mut params_json = String::new();
            params_reader.read_to_string(&mut params_json)?;
            serde_json::from_str(&params_json).map_err(|e| {
                ANNError::log_index_error(format!("Failed to deserialize params: {}", e))
            })?
        };

        let metric = Metric::from_str(&saved_params.metric)
            .map_err(|e| ANNError::log_index_error(format!("Failed to parse metric: {}", e)))?;

        let vector_index = load_bftree(
            BfTreePaths::vectors_bftree(&saved_params.prefix),
            saved_params.use_snapshot,
        )?;
        let full_vectors = VectorProvider::<T>::new_from_bftree(
            saved_params.max_points,
            saved_params.dim,
            saved_params.frozen_points.get(),
            vector_index,
        );

        let adjacency_list_index = load_bftree(
            BfTreePaths::neighbors_bftree(&saved_params.prefix),
            saved_params.use_snapshot,
        )?;
        let neighbor_provider = NeighborProvider::<u32>::new_from_bftree(
            saved_params.max_degree,
            adjacency_list_index,
        )?;

        Ok(Self {
            quant_vectors: NoStore,
            full_vectors,
            neighbor_provider,
            metric,
            graph_params: saved_params.graph_params,
            use_snapshot: saved_params.use_snapshot,
        })
    }
}

impl<T> SaveWith<String> for BfTreeProvider<T, QuantVectorProvider>
where
    T: VectorRepr,
{
    type Ok = usize;
    type Error = ANNError;

    async fn save_with<P>(&self, storage: &P, prefix: &String) -> Result<Self::Ok, Self::Error>
    where
        P: StorageWriteProvider,
    {
        let saved_params = SavedParams {
            max_points: self.max_points(),
            frozen_points: NonZeroUsize::new(self.num_start_points())
                .ok_or_else(|| ANNError::log_index_error("num_start_points is zero"))?,
            dim: self.dim(),
            metric: self.metric().as_str().to_string(),
            max_degree: self.max_degree(),
            prefix: prefix.clone(),
            params_vector: BfTreeParams {
                bytes: self.full_vectors.config().get_cb_size_byte(),
                max_record_size: self.full_vectors.config().get_cb_max_record_size(),
                leaf_page_size: self.full_vectors.config().get_leaf_page_size(),
            },
            params_neighbor: BfTreeParams {
                bytes: self.neighbor_provider.config().get_cb_size_byte(),
                max_record_size: self.neighbor_provider.config().get_cb_max_record_size(),
                leaf_page_size: self.neighbor_provider.config().get_leaf_page_size(),
            },
            quant_params: Some(QuantParams {
                params_quant: BfTreeParams {
                    bytes: self.quant_vectors.config().get_cb_size_byte(),
                    max_record_size: self.quant_vectors.config().get_cb_max_record_size(),
                    leaf_page_size: self.quant_vectors.config().get_leaf_page_size(),
                },
            }),
            graph_params: self.graph_params.clone(),
            is_memory: self.full_vectors.config().is_memory_backend(),
            use_snapshot: self.use_snapshot,
        };

        debug_assert_eq!(
            self.full_vectors.config().is_memory_backend(),
            self.neighbor_provider.config().is_memory_backend(),
            "Vector and neighbor stores have mismatched storage backends"
        );
        debug_assert_eq!(
            self.full_vectors.config().is_memory_backend(),
            self.quant_vectors.config().is_memory_backend(),
            "Vector and quant stores have mismatched storage backends"
        );

        {
            let params_filename = BfTreePaths::params_json(&saved_params.prefix);
            let params_json = serde_json::to_string(&saved_params).map_err(|e| {
                ANNError::log_index_error(format!("Failed to serialize params: {}", e))
            })?;
            let mut params_writer = storage.create_for_write(&params_filename)?;
            params_writer.write_all(params_json.as_bytes())?;
        }

        save_bftree(
            self.full_vectors.bftree(),
            BfTreePaths::vectors_bftree(&saved_params.prefix),
            self.use_snapshot,
        )?;
        save_bftree(
            self.neighbor_provider.bftree(),
            BfTreePaths::neighbors_bftree(&saved_params.prefix),
            self.use_snapshot,
        )?;
        save_bftree(
            self.quant_vectors.bftree(),
            BfTreePaths::quant_bftree(&saved_params.prefix),
            self.use_snapshot,
        )?;

        let filename = BfTreePaths::quant_data_bin(&saved_params.prefix);
        let serialized = self
            .quant_vectors
            .quantizer
            .serialize(GlobalAllocator)
            .map_err(|e| ANNError::log_index_error(format!("{e}")))?;
        let mut writer = storage.create_for_write(&filename)?;
        writer.write_all(&serialized)?;

        Ok(0)
    }
}

impl<T> LoadWith<String> for BfTreeProvider<T, QuantVectorProvider>
where
    T: VectorRepr,
{
    type Error = ANNError;

    async fn load_with<P>(storage: &P, prefix: &String) -> Result<Self, Self::Error>
    where
        P: StorageReadProvider,
    {
        let saved_params: SavedParams = {
            let params_filename = BfTreePaths::params_json(prefix);
            let mut params_reader = storage.open_reader(&params_filename)?;
            let mut params_json = String::new();
            params_reader.read_to_string(&mut params_json)?;
            serde_json::from_str(&params_json).map_err(|e| {
                ANNError::log_index_error(format!("Failed to deserialize params: {}", e))
            })?
        };

        let _quant_params = saved_params.quant_params.ok_or_else(|| {
            ANNError::log_index_error("Missing quant_params in saved params for quantized provider")
        })?;

        let metric = Metric::from_str(&saved_params.metric)
            .map_err(|e| ANNError::log_index_error(format!("Failed to parse metric: {}", e)))?;

        let vector_index = load_bftree(
            BfTreePaths::vectors_bftree(&saved_params.prefix),
            saved_params.use_snapshot,
        )?;
        let full_vectors = VectorProvider::<T>::new_from_bftree(
            saved_params.max_points,
            saved_params.dim,
            saved_params.frozen_points.get(),
            vector_index,
        );

        let adjacency_list_index = load_bftree(
            BfTreePaths::neighbors_bftree(&saved_params.prefix),
            saved_params.use_snapshot,
        )?;
        let neighbor_provider = NeighborProvider::<u32>::new_from_bftree(
            saved_params.max_degree,
            adjacency_list_index,
        )?;

        let filename = BfTreePaths::quant_data_bin(&saved_params.prefix);
        let mut reader = storage.open_reader(&filename)?;
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let quantizer: Poly<dyn Quantizer> = try_deserialize(&bytes, GlobalAllocator)
            .map_err(|e| ANNError::log_index_error(format!("{e}")))?;

        let quant_vector_index = load_bftree(
            BfTreePaths::quant_bftree(&saved_params.prefix),
            saved_params.use_snapshot,
        )?;
        let quant_vectors = QuantVectorProvider::new_from_bftree(quantizer, quant_vector_index);

        Ok(Self {
            quant_vectors,
            full_vectors,
            neighbor_provider,
            metric,
            graph_params: saved_params.graph_params,
            use_snapshot: saved_params.use_snapshot,
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
    use std::sync::Arc;

    use super::*;
    use crate::neighbors::NeighborProvider;
    use crate::quant::create_test_quantizer;
    use crate::vectors::VectorProvider;
    use diskann::{
        graph::DiskANNIndex,
        graph::{self, search::Knn},
        neighbor::BackInserter,
    };
    use diskann_providers::storage::FileStorageProvider;
    use diskann_utils::views::{Init, Matrix};

    fn create_quant_index() -> Arc<DiskANNIndex<BfTreeProvider<f32, QuantVectorProvider>>> {
        let start_point = Matrix::new(Init(|| 0.0f32), 1, 5);
        let dim = 5;
        let logical_max_degree = 6;
        let physical_max_degree = (logical_max_degree as f32 * 1.3) as u32;
        let metric = Metric::L2;

        let provider = BfTreeProvider::new(
            BfTreeProviderParameters {
                max_points: 20,
                num_start_points: NonZeroUsize::new(1).unwrap(),
                dim,
                metric,
                max_degree: physical_max_degree,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
                use_snapshot: false,
            },
            start_point.as_view(),
            create_test_quantizer(5),
        )
        .unwrap();

        let index_config = graph::config::Builder::new_with(
            logical_max_degree as usize,
            graph::config::MaxDegree::new(physical_max_degree as usize),
            10,
            metric.into(),
            |_| {},
        )
        .build()
        .unwrap();

        Arc::new(DiskANNIndex::new(index_config, provider, None))
    }

    #[tokio::test]
    async fn test_quantized_index_search() {
        let index = create_quant_index();
        let ctx = &DefaultContext;

        for i in 0..15 {
            let point = vec![i as f32; 5];
            index
                .insert(&Quantized, ctx, &i, point.as_slice())
                .await
                .unwrap();
        }

        let query = vec![3.0; 5];
        let params = Knn::new(5, 10, None).unwrap();

        let mut neighbors = vec![Neighbor::<u32>::default(); 5];
        let res = index
            .search(
                params,
                &Quantized,
                &DefaultContext,
                query.as_slice(),
                &mut BackInserter::new(neighbors.as_mut_slice()),
            )
            .await
            .unwrap();

        assert_eq!(
            res.result_count, 5,
            "there are 15 points and we're asking for 5, we expect 5"
        );
        assert_eq!(neighbors[0].id, 3);
    }

    #[tokio::test]
    async fn test_quantized_index_multi_insert_search() {
        let index = create_quant_index();
        let ctx = &DefaultContext;

        let data = Matrix::new(
            Init({
                let mut row = 0usize;
                let mut col = 0usize;
                move || {
                    let val = row as f32;
                    col += 1;
                    if col == 5 {
                        col = 0;
                        row += 1;
                    }
                    val
                }
            }),
            15,
            5,
        );
        let ids: Arc<[u32]> = (0u32..15).collect::<Vec<_>>().into();
        let batch: Arc<Matrix<f32>> = Arc::new(data);
        index
            .multi_insert::<Quantized, Matrix<f32>>(Quantized, ctx, batch, ids)
            .await
            .unwrap();

        let query = vec![3.0; 5];
        let params = Knn::new(5, 10, None).unwrap();

        let mut neighbors = vec![Neighbor::<u32>::default(); 5];
        let res = index
            .search(
                params,
                &Quantized,
                &DefaultContext,
                query.as_slice(),
                &mut BackInserter::new(neighbors.as_mut_slice()),
            )
            .await
            .unwrap();

        assert_eq!(
            res.result_count, 5,
            "there are 15 points and we're asking for 5, we expect 5"
        );
        let neighbor_ids: Vec<u32> = neighbors.iter().map(|n| n.id).collect();
        for expected in 1u32..=5 {
            assert!(
                neighbor_ids.contains(&expected),
                "expected id {expected} in results, got {neighbor_ids:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_quantized_delete_and_search() {
        let index = create_quant_index();
        let ctx = &DefaultContext;

        for i in 0..15 {
            let point = vec![i as f32; 5];
            index
                .insert(&Quantized, ctx, &i, point.as_slice())
                .await
                .unwrap();
        }

        index
            .inplace_delete(Quantized, ctx, &2u32, 2, graph::InplaceDeleteMethod::OneHop)
            .await
            .unwrap();
        index
            .inplace_delete(Quantized, ctx, &4u32, 2, graph::InplaceDeleteMethod::OneHop)
            .await
            .unwrap();

        let query = vec![3.0; 5];
        let params = Knn::new(5, 10, None).unwrap();

        let mut neighbors = vec![Neighbor::<u32>::default(); 5];
        let res = index
            .search(
                params,
                &Quantized,
                &DefaultContext,
                query.as_slice(),
                &mut BackInserter::new(neighbors.as_mut_slice()),
            )
            .await
            .unwrap();

        assert_eq!(res.result_count, 5);
        let neighbor_ids: Vec<u32> = neighbors.iter().map(|n| n.id).collect();
        assert!(!neighbor_ids.contains(&2u32));
        assert!(!neighbor_ids.contains(&4u32));
    }

    fn create_full_precision_index() -> Arc<DiskANNIndex<BfTreeProvider<f32, NoStore>>> {
        let start_point = Matrix::new(Init(|| 0.0f32), 1, 5);
        let logical_max_degree = 6;
        let physical_max_degree = (logical_max_degree as f32 * 1.3) as u32;
        let metric = Metric::L2;

        let provider = BfTreeProvider::new(
            BfTreeProviderParameters {
                max_points: 20,
                num_start_points: NonZeroUsize::new(1).unwrap(),
                dim: 5,
                metric,
                max_degree: physical_max_degree,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
                use_snapshot: false,
            },
            start_point.as_view(),
            NoStore,
        )
        .unwrap();

        let index_config = graph::config::Builder::new_with(
            logical_max_degree as usize,
            graph::config::MaxDegree::new(physical_max_degree as usize),
            10,
            metric.into(),
            |_| {},
        )
        .build()
        .unwrap();

        Arc::new(DiskANNIndex::new(index_config, provider, None))
    }

    #[tokio::test]
    async fn test_full_precision_index_search() {
        let index = create_full_precision_index();
        let ctx = &DefaultContext;

        for i in 0u32..15 {
            let point = vec![i as f32; 5];
            index
                .insert(&FullPrecision, ctx, &i, point.as_slice())
                .await
                .unwrap();
        }

        let query = vec![3.0; 5];
        let params = Knn::new(5, 10, None).unwrap();

        let mut neighbors = vec![Neighbor::<u32>::default(); 5];
        let res = index
            .search(
                params,
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &mut BackInserter::new(neighbors.as_mut_slice()),
            )
            .await
            .unwrap();

        assert_eq!(
            res.result_count, 5,
            "there are 15 points and we're asking for 5, we expect 5"
        );
        assert_eq!(neighbors[0].id, 3);
    }

    #[tokio::test]
    async fn test_full_precision_delete_and_search() {
        let index = create_full_precision_index();
        let ctx = &DefaultContext;

        for i in 0u32..15 {
            let point = vec![i as f32; 5];
            index
                .insert(&FullPrecision, ctx, &i, point.as_slice())
                .await
                .unwrap();
        }

        index
            .inplace_delete(
                FullPrecision,
                ctx,
                &2u32,
                2,
                graph::InplaceDeleteMethod::OneHop,
            )
            .await
            .unwrap();
        index
            .inplace_delete(
                FullPrecision,
                ctx,
                &4u32,
                2,
                graph::InplaceDeleteMethod::OneHop,
            )
            .await
            .unwrap();

        let query = vec![3.0; 5];
        let params = Knn::new(5, 10, None).unwrap();

        let mut neighbors = vec![Neighbor::<u32>::default(); 5];
        let res = index
            .search(
                params,
                &FullPrecision,
                &DefaultContext,
                query.as_slice(),
                &mut BackInserter::new(neighbors.as_mut_slice()),
            )
            .await
            .unwrap();

        assert_eq!(res.result_count, 5);
        let neighbor_ids: Vec<u32> = neighbors.iter().map(|n| n.id).collect();
        assert!(!neighbor_ids.contains(&2u32));
        assert!(!neighbor_ids.contains(&4u32));
    }

    #[tokio::test]
    async fn test_data_provider_and_delete_interface() {
        let ctx = &DefaultContext;
        let num_start_points = 2;
        let dim = 5;
        let start_points = Matrix::try_from(
            vec![0.0f32; dim]
                .into_iter()
                .chain(vec![0.5f32; dim])
                .collect::<Box<[_]>>(),
            num_start_points,
            dim,
        )
        .unwrap();

        let provider = BfTreeProvider::new(
            BfTreeProviderParameters {
                max_points: 10,
                num_start_points: NonZeroUsize::new(num_start_points).unwrap(),
                dim,
                metric: Metric::L2,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
                use_snapshot: false,
            },
            start_points.as_view(),
            NoStore,
        )
        .unwrap();

        // Iterator
        //
        assert_eq!((&provider).into_iter(), 0..(10 + 2));

        let iter = provider.iter();

        // Insert vectors so they exist in the bf_tree (hard-delete checks presence)
        for i in iter.clone() {
            let vector: Vec<f32> = (0..5).map(|j| (i * 5 + j) as f32).collect();
            provider.set_element(ctx, &i, &vector).await.unwrap();
        }

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

        // With hard deletes, `release` is a no-op (data is permanently removed).
        // Verify that released IDs remain deleted.
        for i in iter.clone() {
            provider.release(ctx, i).await.unwrap();
            assert_eq!(
                provider.status_by_internal_id(ctx, i).await.unwrap(),
                ElementStatus::Deleted
            );
        }

        // out-of-bound set-element fails.
        //
        assert!(provider
            .set_element(ctx, &100, &[1.0, 2.0, 3.0, 4.0])
            .await
            .is_err());
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

        let num_start_points = 2;
        let dim = 3;
        let start_points = Matrix::new(Init(|| 0.0f32), num_start_points, dim);

        let provider = BfTreeProvider::<f32, _>::new(
            BfTreeProviderParameters {
                max_points: num_points as usize,
                num_start_points: NonZeroUsize::new(num_start_points).unwrap(),
                dim,
                metric: Metric::L2,
                max_degree: 64,
                vector_provider_config: Config::default(),
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: Config::default(),
                graph_params: None,
                use_snapshot: false,
            },
            start_points.as_view(),
            NoStore,
        )
        .unwrap();

        let mut scratch = provider.neighbor_provider.scratch();

        // Insert new vectors without neighbors and empty neighbor list is
        // expected for each newly inserted vector
        //
        for i in 0..num_points {
            let vector = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
            provider.set_element(ctx, &i, &vector).await.unwrap();

            // First attempt should return empty
            let mut out = AdjacencyList::new();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut out)
                .unwrap();
            assert!(out.is_empty());

            // After we set the empty neighbor list, our attempt should succeed
            scratch.write_neighbors(i, &[]).unwrap();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut out)
                .unwrap();

            assert!(out.is_empty());
        }

        // Add a non-empty neighbor list for a vector and then set it to empty
        // In the end, an empty neighbor list is expected for the vector
        //
        for i in 0..num_points {
            let mut out = AdjacencyList::new();
            let neighbors = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
            scratch.write_neighbors(i, &neighbors).unwrap();

            provider
                .neighbor_provider
                .get_neighbors(i, &mut out)
                .unwrap();

            assert_eq!(&*out, &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

            scratch.write_neighbors(i, &[]).unwrap();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut out)
                .unwrap();

            assert!(out.is_empty());
        }

        // Non-existant vectors have empty neighbor lists
        //
        let mut out = AdjacencyList::from_iter_untrusted([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]); // len = 10

        // Attempt to access non-existant vector's neighbor list should fail as NotFound
        assert!(provider
            .neighbor_provider
            .get_neighbors(200, &mut out)
            .is_err());
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
        vector_config.use_snapshot(true);

        let bytes_neighbor = 1024 * 1024;
        let mut neighbor_config = Config::new(&neighbor_path, bytes_neighbor);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);
        neighbor_config.use_snapshot(true);

        // Create provider parameters
        let params = BfTreeProviderParameters {
            max_points: num_points,
            num_start_points,
            dim,
            metric: Metric::L2,
            max_degree,
            vector_provider_config: vector_config.clone(),
            quant_vector_provider_config: Config::default(),
            neighbor_list_provider_config: neighbor_config.clone(),
            graph_params: None,
            use_snapshot: true,
        };

        let start_points = Matrix::new(Init(|| 0.0f32), num_start_points.into(), dim);

        // Create provider
        let provider =
            BfTreeProvider::<f32, NoStore>::new(params.clone(), start_points.as_view(), NoStore)
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
        let mut scratch = provider.neighbor_provider.scratch();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            scratch.write_neighbors(i, &neighbors).unwrap();
        }

        assert_eq!(vector_config.get_leaf_page_size(), 8192);
        assert_eq!(vector_config.get_cb_max_record_size(), 1024);

        let storage = FileStorageProvider;

        // Save to a different prefix to exercise the snapshot copy logic
        let save_dir = tempdir().unwrap();
        let save_prefix = save_dir
            .path()
            .join("saved_bf_tree_provider")
            .to_string_lossy()
            .to_string();
        provider.save_with(&storage, &save_prefix).await.unwrap();

        // Load using trait method (includes delete bitmap)
        let loaded_provider = BfTreeProvider::<f32, NoStore>::load_with(&storage, &save_prefix)
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
        vector_config.use_snapshot(true);

        let bytes_neighbor = 1024 * 1024;
        let mut neighbor_config = Config::new(&neighbor_path, bytes_neighbor);
        neighbor_config.storage_backend(bf_tree::StorageBackend::Std);
        neighbor_config.use_snapshot(true);

        let bytes_quant = 1024 * 1024;
        let mut quant_config = Config::new(&quant_path, bytes_quant);
        quant_config.storage_backend(bf_tree::StorageBackend::Std);
        quant_config.use_snapshot(true);

        // Create spherical quantizer
        let quantizer = create_test_quantizer(dim);

        // Create provider parameters
        let params = BfTreeProviderParameters {
            max_points: num_points,
            num_start_points,
            dim,
            metric: Metric::L2,
            max_degree,
            vector_provider_config: vector_config.clone(),
            quant_vector_provider_config: quant_config.clone(),
            neighbor_list_provider_config: neighbor_config.clone(),
            graph_params: None,
            use_snapshot: true,
        };

        let start_points = Matrix::new(Init(|| 0.0f32), num_start_points.into(), dim);
        // Create provider with quantization
        let provider = BfTreeProvider::<f32, QuantVectorProvider>::new(
            params.clone(),
            start_points.as_view(),
            quantizer,
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
        let mut scratch = provider.neighbor_provider.scratch();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            scratch.write_neighbors(i, &neighbors).unwrap();
        }

        let storage = FileStorageProvider;

        // Save to a different prefix to exercise the snapshot copy logic
        let save_dir = tempdir().unwrap();
        let save_prefix = save_dir
            .path()
            .join("saved_bf_tree_provider_quant")
            .to_string_lossy()
            .to_string();
        provider.save_with(&storage, &save_prefix).await.unwrap();

        // Load using trait method (includes delete bitmap and quantization)
        let loaded_provider =
            BfTreeProvider::<f32, QuantVectorProvider>::load_with(&storage, &save_prefix)
                .await
                .unwrap();

        // Verify quantizer properties match after round-trip
        assert_eq!(
            provider.quant_vectors.quantizer.full_dim(),
            loaded_provider.quant_vectors.quantizer.full_dim(),
            "Quantizer full_dim mismatch"
        );
        assert_eq!(
            provider.quant_vectors.quantizer.bytes(),
            loaded_provider.quant_vectors.quantizer.bytes(),
            "Quantizer bytes mismatch"
        );
        assert_eq!(
            provider.quant_vectors.quantizer.nbits(),
            loaded_provider.quant_vectors.quantizer.nbits(),
            "Quantizer nbits mismatch"
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

        // Cleanup is automatic when temp_dir goes out of scope
    }

    /// Test saving an in-memory (no disk) BfTreeProvider without quantization and loading it back.
    #[tokio::test]
    async fn test_bf_tree_provider_memory_save_load_no_quant() {
        let num_points = 20usize;
        let dim = 4usize;
        let max_degree = 16u32;
        let num_start_points = NonZeroUsize::new(1).unwrap();
        let ctx = &DefaultContext;

        let mut vector_config = Config::default();
        vector_config.use_snapshot(true);
        let mut neighbor_config = Config::default();
        neighbor_config.use_snapshot(true);

        let start_points = Matrix::new(Init(|| 0.0f32), num_start_points.into(), dim);
        // In-memory config (no file path needed)
        let provider = BfTreeProvider::<f32, NoStore>::new(
            BfTreeProviderParameters {
                max_points: num_points,
                num_start_points,
                dim,
                metric: Metric::L2,
                max_degree,
                vector_provider_config: vector_config,
                quant_vector_provider_config: Config::default(),
                neighbor_list_provider_config: neighbor_config,
                graph_params: None,
                use_snapshot: true,
            },
            start_points.as_view(),
            NoStore,
        )
        .unwrap();

        // Populate vectors and neighbors
        for i in 0..num_points {
            let vector: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect();
            provider
                .set_element(ctx, &(i as u32), &vector)
                .await
                .unwrap();
        }
        let mut scratch = provider.neighbor_provider.scratch();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            scratch.write_neighbors(i, &neighbors).unwrap();
        }

        // Delete a couple of vectors
        provider.delete(ctx, &3u32).await.unwrap();
        provider.delete(ctx, &7u32).await.unwrap();

        // Save to disk from in-memory
        let save_dir = tempdir().unwrap();
        let save_prefix = save_dir
            .path()
            .join("mem_no_quant")
            .to_string_lossy()
            .to_string();
        let storage = FileStorageProvider;
        provider.save_with(&storage, &save_prefix).await.unwrap();

        // Load back
        let loaded = BfTreeProvider::<f32, NoStore>::load_with(&storage, &save_prefix)
            .await
            .unwrap();

        // Verify vectors
        for i in 0..num_points as u32 {
            if i == 3 || i == 7 {
                continue;
            }
            assert_eq!(
                provider.full_vectors.get_vector_sync(i as usize).unwrap(),
                loaded.full_vectors.get_vector_sync(i as usize).unwrap(),
                "Vector mismatch at {}",
                i
            );
        }

        // Verify neighbors
        for i in 0..num_points as u32 {
            let mut orig = AdjacencyList::new();
            let mut load = AdjacencyList::new();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut orig)
                .unwrap();
            loaded
                .neighbor_provider
                .get_neighbors(i, &mut load)
                .unwrap();
            assert_eq!(&*orig, &*load, "Neighbor mismatch at {}", i);
        }

        // Verify deletes
        assert_eq!(
            loaded.status_by_internal_id(ctx, 3).await.unwrap(),
            ElementStatus::Deleted
        );
        assert_eq!(
            loaded.status_by_internal_id(ctx, 7).await.unwrap(),
            ElementStatus::Deleted
        );
        assert_eq!(
            loaded.status_by_internal_id(ctx, 0).await.unwrap(),
            ElementStatus::Valid
        );
    }

    /// Test saving an in-memory BfTreeProvider with PQ quantization and loading it back.
    #[tokio::test]
    async fn test_bf_tree_provider_memory_save_load_quant() {
        let num_points = 20usize;
        let dim = 8usize;
        let max_degree = 16u32;
        let num_start_points = NonZeroUsize::new(1).unwrap();
        let ctx = &DefaultContext;

        let quantizer = create_test_quantizer(dim);
        let mut vector_config = Config::default();
        vector_config.use_snapshot(true);
        let mut neighbor_config = Config::default();
        neighbor_config.use_snapshot(true);
        let mut quant_config = Config::default();
        quant_config.use_snapshot(true);

        let start_points = Matrix::new(Init(|| 0.0f32), num_start_points.into(), dim);
        let provider = BfTreeProvider::<f32, QuantVectorProvider>::new(
            BfTreeProviderParameters {
                max_points: num_points,
                num_start_points,
                dim,
                metric: Metric::L2,
                max_degree,
                vector_provider_config: vector_config,
                quant_vector_provider_config: quant_config,
                neighbor_list_provider_config: neighbor_config,
                graph_params: None,
                use_snapshot: true,
            },
            start_points.as_view(),
            quantizer,
        )
        .unwrap();

        // Populate vectors and neighbors
        for i in 0..num_points {
            let vector: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect();
            provider
                .set_element(ctx, &(i as u32), &vector)
                .await
                .unwrap();
        }
        let mut scratch = provider.neighbor_provider.scratch();
        for i in 0..num_points as u32 {
            let neighbors: Vec<u32> = (0..std::cmp::min(i, max_degree))
                .map(|j| (i + j) % num_points as u32)
                .collect();
            scratch.write_neighbors(i, &neighbors).unwrap();
        }

        provider.delete(ctx, &2u32).await.unwrap();

        // Save to disk from in-memory
        let save_dir = tempdir().unwrap();
        let save_prefix = save_dir
            .path()
            .join("mem_quant")
            .to_string_lossy()
            .to_string();
        let storage = FileStorageProvider;
        provider.save_with(&storage, &save_prefix).await.unwrap();

        // Load back
        let loaded = BfTreeProvider::<f32, QuantVectorProvider>::load_with(&storage, &save_prefix)
            .await
            .unwrap();

        // Verify full vectors (skip deleted id 2)
        for i in 0..num_points as u32 {
            if i == 2 {
                continue;
            }
            assert_eq!(
                provider.full_vectors.get_vector_sync(i as usize).unwrap(),
                loaded.full_vectors.get_vector_sync(i as usize).unwrap(),
                "Vector mismatch at {}",
                i
            );
        }

        // Verify quant vectors (skip deleted id 2)
        for i in 0..num_points as u32 {
            if i == 2 {
                continue;
            }
            assert_eq!(
                provider.quant_vectors.get_vector_sync(i as usize).unwrap(),
                loaded.quant_vectors.get_vector_sync(i as usize).unwrap(),
                "Quant vector mismatch at {}",
                i
            );
        }

        // Verify neighbors (skip deleted id 2)
        for i in 0..num_points as u32 {
            if i == 2 {
                continue;
            }
            let mut orig = AdjacencyList::new();
            let mut load = AdjacencyList::new();
            provider
                .neighbor_provider
                .get_neighbors(i, &mut orig)
                .unwrap();
            loaded
                .neighbor_provider
                .get_neighbors(i, &mut load)
                .unwrap();
            assert_eq!(&*orig, &*load, "Neighbor mismatch at {}", i);
        }

        // Verify delete
        assert_eq!(
            loaded.status_by_internal_id(ctx, 2).await.unwrap(),
            ElementStatus::Deleted
        );
        assert_eq!(
            loaded.status_by_internal_id(ctx, 0).await.unwrap(),
            ElementStatus::Valid
        );
    }

    #[test]
    fn test_validate_rejects_undersized_vector_config() {
        // 1536 * 4 = 6144 bytes + 8-byte key = 6152 bytes needed
        let result = VectorProvider::<f32>::new_with_config(
            100,
            1536,
            1,
            Config::default(), // cb_max_record_size = 1952
        );
        let err = result.err().expect("should fail").to_string();
        assert!(
            err.contains("vector_provider"),
            "should name the failing config; got: {err}"
        );
        assert!(
            err.contains("6152"),
            "should state the required size; got: {err}"
        );
    }

    #[test]
    fn test_validate_rejects_undersized_neighbor_config() {
        // max_degree=500 → value = (500+1)*4 = 2004 bytes + 4-byte key = 2008 bytes
        let mut neighbor_config = Config::default();
        neighbor_config.cb_max_record_size(1952);

        let result = NeighborProvider::<u32>::new_with_config(500, neighbor_config);
        let err = result.err().expect("should fail").to_string();
        assert!(
            err.contains("neighbor_provider"),
            "should name the failing config; got: {err}"
        );
    }

    #[test]
    fn test_validate_accepts_valid_config() {
        // dim=128, f32 → key=8 + value=512 = 520 bytes, fits default 1952
        if let Err(e) = VectorProvider::<f32>::new_with_config(100, 128, 1, Config::default()) {
            panic!("VectorProvider should succeed: {e}");
        }
        // max_degree=64 → key=4 + value=(64+1)*4 = 264 bytes, fits default 1952
        if let Err(e) = NeighborProvider::<u32>::new_with_config(64, Config::default()) {
            panic!("NeighborProvider should succeed: {e}");
        }
    }
}
