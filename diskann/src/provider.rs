/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The [`DataProvider`] trait encompasses the following concepts:
//!
//! * Storage for arbitrary data that is contextually accessed through the [`Accessor`] trait.
//!
//! * Mapping between an "external id" (a unique external identifier for an entry in data
//!   store) and an "internal id", which is a simple typed used as a handle to vector data
//!   internally.
//!
//! * Support for deletion via the [`Delete`] sub-trait.
//!
//! # Important Related Traits
//!
//! The [`DataProvider`] trait is really an ensemble of multiple related traits all working
//! together to solve a problem.
//!
//! Important associated traits are described here.
//!
//! * [`SetElement`]: This is the method that allows assigning of data into the
//!   [`DataProvider`] via external ID. This trait is parameterized by the type of the
//!   "vector" being assigned, providing a mechanism for inclusion of arbitrary associated
//!   data along with the raw vector.
//!
//!   The responsibility of the `set_element` method is several fold:
//!
//!   * It must assign an internal ID for the provided external ID. It can do this by using
//!     a naive identity map.
//!
//!   * After insertion, the implementer may return a guard that will notify the provider
//!     of an unsuccessful operation, allowing automatic cleanup and consistency control
//!     over the external ID to internal ID mapping.
//!
//! * [`Delete`]: A sub-trait of [`DataProvider`] indicating that the provider supports the
//!   notion of deleted items.
//!
//!   We differentiate between two different kinds of deletion:
//!
//!   * Soft Deletion: Where a external ID is marked as deleted, but the internal vector id
//!     may still be reachable in an index.
//!
//!   * Hard Deletion (aka "release): This deletes an item by internal ID, removing it
//!     completely from the store.
//!
//!   Note that the exact semantics of deletion are defined by the implementer. Some
//!   implementations may allow internal IDs to be reused immediately following the deletion
//!   of the external ID while others may wait for a "release".
//!
//! * [`HasId`]: Traits such as [`Accessor`], [`NeighborAccessor`] and [`DelegateNeighbors`]
//!   all need to interact with the underlying [`DataProvider`] using an internal ID type.
//!
//!   The [`HasId`] trait provides a common base-trait for these related concepts, which
//!   ensures implementers only need to define and constrain it once.
//!
//! * [`Accessor`]: A contextual proxy object for retrieving data from the [`DataProvider`].
//!   The key idea behind an accessor is that data to be retrieved from a data store
//!   is contextual. In other words, in some contexts, we may retrieve one kind of data
//!   (for example, full precision vectors), while in other contexts we may want to
//!   retrieve another kind (such as quantized vectors).
//!
//!   Accessors can be implemented as simple handles to a provider, or can have local
//!   scratch to assist in bulk operations.
//!
//!   Finally, one handy feature of accessors is that they can carry a lifetime, which
//!   surprisingly can make writing algorithms involving borrows easier than trying to
//!   manage a lifetime strictly through associated types with lifetimes.
//!
//! * [`BuildDistanceComputer`]: A sub-trait of [`Accessor`] that allows for random-access
//!   distance computations on the retrieved elements.
//!
//! * [`BuildQueryComputer`]: A sub-trait of [`Accessor`] that allows for specialized query
//!   based computations. This allows a query to be pre-processed in a way that allows
//!   faster computations.
//!
//! # Neighbor Delegation
//!
//! Index search requires that accessor types implement both the data-centric [`Accessor`]
//! trait and the graph retrieval [`NeighborAccessor`]/[`NeighborAccessorMut`] traits.
//!
//! While having multiple implementations of [`Accessor`] is common to support different
//! kinds of searches in the quantized space, neighbor retrieval commonly need not vary.
//! Instead of requiring all implementations of [`Accessor`] to manually forward the methods
//! (both required and provided) in the [`NeighborAccesosr`] traits, we use a delegation
//! technique supplied by the [`DelegateNeighbor`] trait.
//!
//! [`Accessor`] types should implement [`DelegateNeighbor`] to return a [`NeighborAccessor`].
//! Implementation of [`DelegateNeighbors`] will automatically implement [`AsNeighbor`] and
//! [`AsNeighborMut`] (the latter is only applicable if the returned type implements
//! [`NeighborAccessorMut`].
//!
//! Similarly, algorithms requiring graph access should accept `&mut T` where
//! `T: AsNeighbor` or `T: AsNeighborMut`. This provides access to blanket implementations
//! of [`NeighborAccessor`] and [`NeighborAccessorMut`] for `&mut T`.

use std::ops::Deref;

use diskann_utils::{Reborrow, WithLifetime};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};
use sealed::{BoundTo, Sealed};

use crate::{ANNError, ANNResult, error::ToRanked, graph::AdjacencyList, utils::VectorId};

//////////////////////
// ExecutionContext //
//////////////////////

/// An execution context given to various providers, threaded through insertions, searches
/// etc. for information forwarding.
pub trait ExecutionContext: Send + Sync + Clone + 'static {
    //////////////////////
    // Provided Methods //
    //////////////////////

    /// Provide a customization point for tasks spawned while under this context.
    ///
    /// The future `f` is a Future that DiskANN intends to spawn as a task using an spawn
    /// method such as `tokio::spawn`. `DiskANN will pass this Future through this function
    /// before creating the task.
    ///
    /// This allows the `ExecutionContext` to nest that Future inside another Future if desired.
    /// An example of such a nesting would be to nest `f` inside of a profiling future
    /// to record the CPU cycles spent executing `f`.
    ///
    /// The default implementation of this method is the identity, simply passing through
    /// the future unmodified.
    fn wrap_spawn<F, T>(&self, f: F) -> impl std::future::Future<Output = T> + Send + 'static
    where
        F: std::future::Future<Output = T> + Send + 'static,
    {
        f
    }
}

//////////////////
// DataProvider //
//////////////////

/// A base trait for the struct providing data into the index.
///
/// The requirements on this trait are quite sparse. Instead, additional behavior is accessed
/// through derived and related traits, namely
///
/// * [`SetElement`]: An overloadable version of `set_vector`.
/// * [`Accessor`]: An overloadable, contextual class for retrieving data elements from
///   the data provider.
///
/// Indexing algorithms explose overloadable "strategies" that allow data provider to
/// select the accessor.
///
/// Example strategies include:
///
/// * [`crate::graph::glue::SearchStrategy`]
/// * [`crate::graph::glue::PruneStrategy`]
/// * [`crate::graph::glue::InsertStrategy`]
///
/// # Type Constraints:
///
/// * `Sized`: Data provider types are used in contexts where `Self` is used to instantiate
///   a generic. This can only be done if `Self: Sized`.
///
/// * `Send` and `Sync`: Mostly to help async code compile and be `Send`.
///
/// * `'static`: Helpful for avoiding excess lifetime constraints.
pub trait DataProvider: Sized + Send + Sync + 'static {
    type Context: ExecutionContext;
    type InternalId: VectorId;
    type ExternalId: PartialEq + Send + Sync + 'static;

    type Error: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// Translate an external id to its corresponding internal id.
    ///
    /// The vector referenced by `gid` must already have been added to the provider via
    /// [`SetElement`]. The mapping is undefined until then.
    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error>;

    /// Translate an internal id to its corresponding external id.
    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error>;
}

////////////
// Delete //
////////////

pub trait Delete: DataProvider {
    /// Delete an item by external ID.
    ///
    /// Note that internal vector IDs may still be reachable. In the context of a graph
    /// index, this is equivalent to a "soft" delete where the deleted ID should no longer
    /// be returned as the result of search methods, but may still be accessed by its
    /// private ID during graph node expansion.
    fn delete(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    /// Release a node by an internal ID.
    ///
    /// This is called by the index only when there are no longer any incoming edges to
    /// a particular data point.
    ///
    /// In particular, the index makes the guarantee that when it invokes `release` on an
    /// internal ID, it will not try to retrive an element via the same internal ID via
    /// an accessor derived from `self` until `SetElement` yields the internal ID.
    fn release(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send;

    /// Check the status via internal ID.
    fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl std::future::Future<Output = Result<ElementStatus, Self::Error>> + Send;

    /// Check the status via external ID.
    fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> impl std::future::Future<Output = Result<ElementStatus, Self::Error>> + Send;

    /// A potentially optimized bulk version of `status_by_internal_id`.
    fn statuses_unordered<Itr, F>(
        &self,
        context: &Self::Context,
        itr: Itr,
        mut f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send
    where
        Itr: Iterator<Item = Self::InternalId> + Send,
        F: FnMut(Result<ElementStatus, Self::Error>, Self::InternalId) + Send,
    {
        async move {
            for i in itr {
                f(self.status_by_internal_id(context, i).await, i);
            }
            Ok(())
        }
    }
}

/// Describe the status of accessing a vector by internal or external id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ElementStatus {
    /// The ID is valid.
    Valid,
    /// The ID used to be valid but points to a deleted element. Some values behind the
    /// deleted may still be accessible until it is removed entirely from the graph.
    Deleted,
}

impl ElementStatus {
    /// Return `true` if `ElementStatus::Valid`.
    pub fn is_valid(self) -> bool {
        self == Self::Valid
    }

    /// Return `true` if `ElementStatus::Deleted`.
    pub fn is_deleted(self) -> bool {
        self == Self::Deleted
    }
}

///////////
// HasId //
///////////

/// Indicate an association with an Id type.
pub trait HasId {
    type Id: VectorId;
}

impl<T> HasId for &T
where
    T: HasId,
{
    type Id = T::Id;
}

impl<T> HasId for &mut T
where
    T: HasId,
{
    type Id = T::Id;
}

////////////////
// SetElement //
////////////////

/// An overloadable `DataProvider` sub-trait allowing element assignment.
pub trait SetElement<T>: DataProvider
where
    T: ?Sized,
{
    /// The kind of error yielded by `set_element`.
    type SetError: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// The operation guard returned by `set_element` to notify `self` if an operation fails.
    ///
    /// This is required to be '`static` for multi-insert compatibility (where it is required
    /// to cross spawn boundaries).
    type Guard: Guard<Id = Self::InternalId> + 'static;

    /// Internally store the value of `element` and associate it with `id`.
    ///
    /// The storing does not necessarily need to be lossless if, for example, the parent
    /// data provider solely uses a quantized representation.
    ///
    /// Note that it is suggests that a well-behaved implementation rolls-back internal
    /// state in the event that an error is returned.
    ///
    /// Furthermore, a guard is returned. The caller of `set_element` will `complete` the
    /// guard once operation completes successfully. Unsuccessful execution may roll back
    /// external to internal mappings, but may not reclaim the local slot.
    fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &T,
    ) -> impl std::future::Future<Output = Result<Self::Guard, Self::SetError>> + Send;
}

/// A guard object that will be completed when an insert operation is successful.
///
/// This is used as the return type of [`SetElement`] (for example, at the beginning of
/// an insert operation), and has three main jobs:
///
/// 1. It provides a means for the data provider to associate a private ID with a public ID
///    and communicate that association to algorithms.
///
/// 2. The guard's lifetime is associated with the duration of an operation (e.g. insert),
///    and calling the `complete` method indicates a successful completion of that operation.
///
/// 3. Dropping the guard before calling `complete` can notifies the data provider of an
///    unsuccessful completion of the operation, allowing the provider to clean up internal
///    state.
pub trait Guard: Send + Sync + 'static {
    /// The Id type associated with the Guard.
    type Id;

    /// Successfully complete the guarded operation.
    fn complete(self) -> impl std::future::Future<Output = ()> + Send;

    /// Retrieve the inernal ID the data provider assigns to the external ID.
    fn id(&self) -> Self::Id;
}

/// A simple `Guard` implementation where completion and dropping is a no-op.
#[derive(Debug, Default)]
pub struct NoopGuard<I>(I);

impl<I> NoopGuard<I> {
    /// Construct a new guard that yields `id` on `retrieve`.
    pub fn new(id: I) -> Self {
        Self(id)
    }
}

impl<I> Guard for NoopGuard<I>
where
    I: Send + Sync + Copy + 'static,
{
    type Id = I;
    async fn complete(self) {}
    fn id(&self) -> Self::Id {
        self.0
    }
}

//////////////
// Accessor //
//////////////

/// A lens through which [`DataProvider`]s contextually viewed.
///
/// Accessors are **not** required to be `'static` and almost always contain a scoped
/// reference to their parent provider.
///
/// # Element Relationship
///
/// Accessors are expected to define three associated element types:
///
/// * `Element<'_>`: The type returned by `get_element`. This is scoped to the borrow
///   of the accessor at the `get_element` call site. As a consequence, there may only
///   be one sucn `Element` active at a time.
///
/// * `ElementRef<'_>`: A generalized borrowed form of `Element` obtainable via
///   `Reborrow`. This is the type on which distance computations are defined and is the
///   element type provided to the `on_element_unordered` bulk operation.
///
/// * `Extended`: An extended version of `Element` whose lifetime is only limited to the
///   lifetime of the `Self` type (rather than scoped to a particular borrow).
///
///   This is needed to handle situations where two or more elements are needed
///   simultaneously from the same `Accessor` and is an escape hatch for the lifetime in
///   `Element`.
///
///   It is expected to [`Reborrow`] to `ElementRef`.
///
/// The below diagram summarizes the relationship.
///
/// ```text
///           Convert (may allocate)
///      +----- escapes the borrow -------> Extended -----------+
///      |         of Element                                   |
///      |                                                   Reborrow
///      |                                                      |
/// Element<'_> ------ Reborrow ----> ElementRef<'_> <----------+
///        ~~~~                                 ~~~~
///         ^                                    ^
///         |                                    |
///   Lifetime tied                       Arbitrarily short
///  to the Accessor                     lifetime decoupled
///                                       from the Accessor
/// ```
///
/// ## Technical Details
///
/// The need for `ElementRef` arises to allow HRTB bounds to distance computers without
/// inducing a `'static` bound on `Self`. In traits like [`BuildQueryComputer`], attempting
/// to use `Element` directly will result in such a requirement on the implementing Accessor.
///
/// The associated `Extended` type is really only needed for index construction.
pub trait Accessor: HasId + Send + Sync {
    /// A generalized reference type used for distance computations.
    ///
    /// Note that the lifetime of `ElementRef` is unconstrained and thus using it in a
    /// [HRTB](https://doc.rust-lang.org/nomicon/hrtb.html) will not induce a `'static`
    /// requirement on `Self`.
    type ElementRef<'a>;

    /// The concrete type of the data element associated with this accessor.
    ///
    /// For distance computations, this should be cheaply convertible via [`Reborrow`] to
    /// `Self::ElementRef`.
    ///
    /// To persist for a longer lifetime, it must be convertible via [`Into`] to
    /// `Self::Extended`.
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
        + Into<Self::Extended>
        + Send
        + Sync
    where
        Self: 'a;

    /// A version of `Self::Element` whose lifetime is not restricted to a borrow of `Self`.
    ///
    /// This is expected to still [`Reborrow`] to `ElementRef` for use in distance
    /// computations.
    type Extended: for<'a> Reborrow<'a, Target = Self::ElementRef<'a>> + Send + Sync;

    /// The error (if any) returned by [`Self::get_element`].
    type GetError: ToRanked + std::fmt::Debug + Send + Sync + 'static;

    /// Return the value associated with the key `id`.
    ///
    /// It is expected that index algorithms will only invoke `get_element` on valid IDs,
    /// that can be derived from [`SetElement::set_element`] or by some other means.
    ///
    /// Implementations are suggested to return an error if this invariant is broken, but
    /// may also panic if that is an acceptable error mode.
    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send;

    /// A bulk interface for invoking [`Self::get_element`] on each item in an iterator and
    /// invoking the closure with the reborrowed element.
    ///
    /// Algorithms are encouraged to use this interface if appropriate as accessor
    /// implementations may specialize the implementation for better performance.
    fn on_elements_unordered<Itr, F>(
        &mut self,
        itr: Itr,
        mut f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::GetError>> + Send
    where
        Self: Sync,
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + for<'a> FnMut(Self::ElementRef<'a>, Self::Id),
    {
        async move {
            for i in itr {
                f(self.get_element(i).await?.reborrow(), i);
            }
            Ok(())
        }
    }
}

/// A utility trait for adding a caching layer to Accessors.
///
/// Note that the implementation of `as_cached` is such that this trait is only implementable
/// if `Self::Map::Of` can be safely transmuted from `Self::Element`. The easiest way to do
/// this safely is to have `Self::Map::Of` be the same type as `Self::Element`.
///
/// This dance is mainly needed to allow down stream implementations to properly constrain
/// the elements returned from a cache are "compatible" with `Self::Element` in a way that
/// does not introduce a '`static` requirement on the `Accessor`.
pub trait CacheableAccessor: Accessor {
    /// A `'static` helper for describing a type with an arbitrary lifetime. The lifetime
    /// annotated type is the value passed between `Self` and the cache.
    type Map: WithLifetime;

    /// Take a cached item with an arbitrary lifetime (limited to that of `Self`) and
    /// cheaply construct `Self::Element` with the same lifetime.
    fn from_cached<'a>(element: <Self::Map as WithLifetime>::Of<'a>) -> Self::Element<'a>
    where
        Self: 'a;

    /// View `Self::Element` as the adapted cached type.
    fn as_cached<'a, 'b>(element: &'a Self::Element<'b>) -> &'a <Self::Map as WithLifetime>::Of<'b>
    where
        Self: 'a + 'b;
}

/// A specialized [`Accessor`] that provides random-access distance computations.
pub trait BuildDistanceComputer: Accessor {
    /// The error type (if any) associated with distance computer construction.
    ///
    /// Implementations are encouraged to make distance computer construction infallible.
    type DistanceComputerError: std::error::Error + Into<ANNError> + Send + Sync + 'static;

    /// The concrete type of the distance computer, which must be applicable to all pairs
    /// of elements yielded by the [`Accessor`].
    type DistanceComputer: for<'a, 'b> DistanceFunction<Self::ElementRef<'a>, Self::ElementRef<'b>>
        + Send
        + Sync
        + 'static;

    /// Build the random-access distance computer for this accessor.
    ///
    /// This method is expected to be relatively cheap to invoke and implementations are
    /// encouraged to make this method infallible.
    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError>;
}

/// A specialized [`Accessor`] that provides query computations for a query type `T`.
///
/// Query computers are allowed to preprocess the query to enable more efficient distance
/// computations.
pub trait BuildQueryComputer<T>: Accessor
where
    T: ?Sized,
{
    /// The error type (if any) associated with distance computer construction.
    type QueryComputerError: std::error::Error + Into<ANNError> + Send + Sync + 'static;

    /// The concrete type of the distance computer, which must be applicable for all
    /// elements yielded by the [`Accessor`].
    type QueryComputer: for<'a> PreprocessedDistanceFunction<Self::ElementRef<'a>, f32>
        + Send
        + Sync
        + 'static;

    /// Build the query computer for this accessor.
    ///
    /// This method is encouraged to be as fast as possible, but will generally only be
    /// invoked once per search or graph insert.
    fn build_query_computer(
        &self,
        from: &T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError>;

    /// Compute the distances for the elements in the iterator `itr` using the
    /// `computer` and apply the closure `f` to each distance and ID. The default
    /// implementation uses on_elements_unordered to iterate over the elements
    /// and compute the distances using `computer` parameter.
    fn distances_unordered<Itr, F>(
        &mut self,
        vec_id_itr: Itr,
        computer: &Self::QueryComputer,
        mut f: F,
    ) -> impl std::future::Future<Output = Result<(), Self::GetError>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        F: Send + FnMut(f32, Self::Id),
    {
        self.on_elements_unordered(vec_id_itr, move |element, i| {
            // Default is to use the computer to evaluate the similarity.
            let distance = computer.evaluate_similarity(element);
            f(distance, i);
        })
    }
}

/////////////////////////
// Neighbor Delegation //
/////////////////////////

/// An accessor that provides random-access neighbor retrieval from a graph.
///
/// Generally, neighbor access and data access are logically decoupled, being served from
/// different stores. However, there are situations where data and neighbors are
/// interleaved in the underlying storage medium.
///
/// As such, [`Accessors`] used in congunction with graph operations need to additionally
/// provide an implementation of this trait.
///
/// To avoid repeating implementations for every [`Accessor`] flavor, the trait
/// [`DelegateNeighbor`] should be used instead to route the implementation of
/// [`NeighborAccessor`] to a single type if applicable.
///
/// # Note
///
/// The `NeighborAccessor` method receive by value. Implementations are strongly encouraged
/// to be cheap to construct, copy, or clone. This can generally be achieved by implementing
/// `NeighborAccessor` for `&T`, `&mut T`, or a thing wrapper around such references.
pub trait NeighborAccessor: HasId + Sized + Send + Sync {
    /// Get the neighbors for the node associated with `id`.
    ///
    /// Populate the neighbors into the `neighbors` out parameter.
    ///
    /// Implementations are expected to clear `neighbors` prior to populating.
    fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send;
}

/// A mutable extension of [`NeighborAccessor`] that enables the underlying graph to be
/// mutated.
///
/// Generally, [`Accessors`] should implement [`DelegateNeighbor`] instead of extending this
/// trait if graph and data access are naturally decoupled.
pub trait NeighborAccessorMut: NeighborAccessor {
    /// Overwrite the neighbor list for the node associated with `id`.
    fn set_neighbors(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send;

    /// Append all entries in `neighbors` tothe neighbor list currently associated with `id`.
    ///
    /// The behavior when the resulting list exceeds some pre-configured capacity or
    /// contains duplicates is implementation defined.
    fn append_vector(
        self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send;

    /// A potentially optimized bulk implementation of [`Self::set_neighbors`].
    ///
    /// For each `id`/`neighbors` pair in `iter`, set the adjacency list for the node associated
    /// with `id` to `neighbors`.
    ///
    /// Implementations are allowed to commit entries out of order.
    fn set_neighbors_bulk<I, T>(
        mut self,
        iter: I,
    ) -> impl std::future::Future<Output = ANNResult<Self>> + Send
    where
        I: Iterator<Item = (Self::Id, T)> + Send,
        T: Deref<Target = [Self::Id]> + Send,
    {
        async move {
            for (vector_id, neighbors) in iter {
                self = self.set_neighbors(vector_id, neighbors.deref()).await?;
            }
            Ok(self)
        }
    }
}

/// This implementation allows `&mut T` to be used as a [`NeighborAccessor`] without
/// requiring manual invocation of [`DelegateNeighbor`].
impl<T> NeighborAccessor for &mut T
where
    T: AsNeighbor,
{
    async fn get_neighbors(
        self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        self.delegate_neighbor()
            .get_neighbors(id, neighbors)
            .await?;
        Ok(self)
    }
}

/// This implementation allows `&mut T` to be used as a [`NeighborAccessorMut`] without
/// requiring manual invocation of [`DelegateNeighbor`].
impl<T> NeighborAccessorMut for &mut T
where
    T: AsNeighborMut,
{
    async fn set_neighbors(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        self.delegate_neighbor()
            .set_neighbors(id, neighbors)
            .await?;
        Ok(self)
    }
    async fn append_vector(self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        self.delegate_neighbor()
            .append_vector(id, neighbors)
            .await?;
        Ok(self)
    }
    async fn set_neighbors_bulk<I, U>(self, iter: I) -> ANNResult<Self>
    where
        I: Iterator<Item = (Self::Id, U)> + Send,
        U: Deref<Target = [Self::Id]> + Send,
    {
        self.delegate_neighbor().set_neighbors_bulk(iter).await?;
        Ok(self)
    }
}

/// Accessor may delegate the responsibility of being a [`NeighborAccessor`] to an auxiliary
/// type by implementing this trait.
///
/// If the implementation of a [`NeighborAccessor`]/[`NeighborAccessorMut`] are coupled with
/// the [`Accessor`] itself (meaning that no delegation can take place), then implementations
/// may do the following. Assume the accessor has type `T`. Then:
///
/// 1. Implement [`NeighborAccessor`]/[`NeighborAccessorMut`] for a thin wrapper around
///    `&[mut] T`.
///
/// 2. Implement [`DelegateNeighbor`] to return this wrapper as its delegate.
///
/// Implementation code can use this trait through the [`AsNeighbor`] and [`AsNeighborMut`]
/// convenience traits to ensure proper application of the HRTB requirements.
///
/// Additionally, the `&mut T` blanket implementation of [`NeighborAccessor`] and
/// [`NeighborAccessorMut`] can be used to avoid manually invoking `delegate_neighbor`.
pub trait DelegateNeighbor<'this, Lifetime: Sealed = BoundTo<&'this Self>>:
    HasId + Send + Sync
{
    /// The type of the delegated [`NeighborAccessor`].
    type Delegate: NeighborAccessor<Id = Self::Id>;

    /// Construct the delegate.
    fn delegate_neighbor(&'this mut self) -> Self::Delegate;
}

impl<'this, T> DelegateNeighbor<'this> for T
where
    T: Copy + NeighborAccessor,
{
    type Delegate = Self;
    fn delegate_neighbor(&'this mut self) -> Self::Delegate {
        *self
    }
}

/// A convenience HRTB wrapper for [`DelegateNeighbor`]. Accessors should implement
/// [`DelegateNeighbor`].
///
/// # Note
///
/// This trait should never be implemented manually. Instead, this trait is automatically
/// implemented for a type `T` when it implements [`DelegateNeighbor`].
pub trait AsNeighbor: for<'a> DelegateNeighbor<'a> {}

/// A convenience HRTB wrapper for [`DelegateNeighbor`]. Accessors should implement
/// [`DelegateNeighbor`].
///
/// # Note
///
/// This trait should never be implemented manually. Instead, this trait is automatically
/// implemented for a type `T` when it implements [`DelegateNeighbor`].
pub trait AsNeighborMut: for<'a> DelegateNeighbor<'a, Delegate: NeighborAccessorMut> {}

impl<T> AsNeighbor for T where T: for<'a> DelegateNeighbor<'a> {}
impl<T> AsNeighborMut for T where T: for<'a> DelegateNeighbor<'a, Delegate: NeighborAccessorMut> {}

/// Get a default accessor from a provider.
///
/// Some providers are able to produce accessors directly from the provides, and will implement
/// this to make getting an accessor convenient.
pub trait DefaultAccessor: DataProvider {
    type Accessor<'a>: HasId<Id = Self::InternalId>
    where
        Self: 'a;
    fn default_accessor(&self) -> Self::Accessor<'_>;
}

////////////////////
// DefaultContext //
////////////////////

/// A light-weight struct implementing [`ExecutionContext`].
///
/// Used for situations where a more refined execution context is not needed.
#[derive(Default, Clone)]
pub struct DefaultContext;

impl std::fmt::Display for DefaultContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "default context")
    }
}

impl ExecutionContext for DefaultContext {}

//////////
// Misc //
//////////

// Constraint for HRBT-style associated types.
mod sealed {
    pub trait Sealed: Sized {}
    pub struct BoundTo<T>(T);
    impl<T> Sealed for BoundTo<T> {}
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        future::Future,
        pin::Pin,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
        task,
    };

    use pin_project::{pin_project, pinned_drop};

    use super::*;
    use crate::{always_escalate, error::Infallible};

    ////////////////////
    // DefaultContext //
    ////////////////////

    #[test]
    fn test_default_context() {
        let ctx = DefaultContext;

        // Check that the implementation of `Display` is correct.
        assert_eq!(ctx.to_string(), "default context");

        assert_eq!(
            std::mem::size_of::<DefaultContext>(),
            0,
            "expected DefaultContext to be an empty class"
        );
    }

    //////////////////
    // Test Context //
    //////////////////

    #[derive(Debug)]
    struct TestContextInner {
        /// The number of tasks spawned.
        spawned: AtomicUsize,
        /// THe number of tasks dropped.
        dropped: AtomicUsize,
    }

    /// The goal of this test is to exercise the functionality of `wrap_spawn`.
    /// We want to make sure that we can correctly hook into tasks spawning.
    #[derive(Debug, Clone)]
    struct TestContext {
        inner: Arc<TestContextInner>,
    }

    impl Default for TestContext {
        fn default() -> Self {
            Self {
                inner: Arc::new(TestContextInner {
                    spawned: AtomicUsize::new(0),
                    dropped: AtomicUsize::new(0),
                }),
            }
        }
    }

    impl std::fmt::Display for TestContext {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "test context")
        }
    }

    /// A Future wrapper that when dropped, increments the `dropped` field of its parent
    /// `TestContext`.
    #[pin_project(PinnedDrop)]
    pub struct SpawnCounter<F> {
        #[pin]
        inner: F,
        parent: TestContext,
    }

    #[pinned_drop]
    impl<F> PinnedDrop for SpawnCounter<F> {
        fn drop(self: Pin<&mut Self>) {
            self.parent.inner.dropped.fetch_add(1, Ordering::AcqRel);
        }
    }

    impl<F> Future for SpawnCounter<F>
    where
        F: Future,
    {
        type Output = F::Output;
        fn poll(self: Pin<&mut Self>, cx: &mut task::Context<'_>) -> task::Poll<Self::Output> {
            self.project().inner.poll(cx)
        }
    }

    impl ExecutionContext for TestContext {
        /// Override task spawning to record the number of tasks spawned and the number
        /// of tasks dropped.
        fn wrap_spawn<F, T>(&self, f: F) -> impl Future<Output = T> + Send + 'static
        where
            F: Future<Output = T> + Send + 'static,
        {
            // Increment spawn count.
            self.inner.spawned.fetch_add(1, Ordering::AcqRel);

            // Create a future that will increment drop count when dropped.
            SpawnCounter {
                inner: f,
                parent: self.clone(),
            }
        }
    }

    ///////////////////
    // Spawning Test //
    ///////////////////

    /// This is a recursive function. At each level, it spawns `width` new instances of
    /// itself with `depth` decreased by 1.
    ///
    /// Each spawned instance uses `context.wrap_spawn`.
    ///
    /// This needs to be manually `async` so we can aply the `'static` bound. Since it's
    /// recursive, Rust struggles to properly deduce the hidden type for the opaque return
    /// type.
    #[allow(clippy::manual_async_fn)]
    fn test_spawning<Context>(
        context: Context,
        width: usize,
        depth: usize,
    ) -> impl Future<Output = ()> + Send + 'static
    where
        Context: ExecutionContext + 'static + std::fmt::Debug,
    {
        async move {
            if depth == 0 {
                return;
            }

            let handles: Box<[_]> = (0..width)
                .map(|_| {
                    let clone = context.clone();
                    tokio::spawn(context.wrap_spawn(test_spawning(clone, width, depth - 1)))
                })
                .collect();

            for h in handles {
                h.await.unwrap();
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_task_spawning() {
        let context = TestContext::default();
        assert_eq!(context.inner.spawned.load(Ordering::Acquire), 0);
        assert_eq!(context.inner.dropped.load(Ordering::Acquire), 0);

        // How to we compute the number of spawned tasks:
        // 1. The first invocation of `test_spawning` spawned `width` tasks.
        // 2. Each tasks spawns `width` tasks, meaning the second level creates `width ^ 2`
        //    tasks.
        // 3. In general, depth `d` spawnS `width ^ d` tasks.
        //
        // The total number of tasks is then:
        // ```
        // S = width + width^2 + width^3 ... width^depth
        // ```
        // This forms a finite geometric series with a closed for solution of
        // ```
        // S = (width ^ (depth + 1) - 1) / (width - 1) - 1
        // ```
        let width = 10;
        let depth = 3;
        test_spawning(context.clone(), width, depth).await;

        let expected = (width.pow((depth + 1).try_into().unwrap()) - 1) / (width - 1) - 1;
        assert_eq!(context.inner.spawned.load(Ordering::Acquire), expected);
        assert_eq!(context.inner.dropped.load(Ordering::Acquire), expected);
    }

    ///////////////////
    // Data Provider //
    ///////////////////

    #[tokio::test]
    async fn test_noop_guard() {
        // A guard that completes successfully.
        {
            let guard = NoopGuard::<usize>::new(10);
            assert_eq!(guard.id(), 10);
            guard.complete().await;
        }

        // A guard that completes unsuccessfully.
        // The Noop guard specifically does not complain if `complete` is not invoked.
        {
            let guard = NoopGuard::<usize>::new(5);
            assert_eq!(guard.id(), 5);
            // Destructor runs here.
        }
    }

    #[test]
    fn simple_status_test() {
        let valid = ElementStatus::Valid;
        assert!(valid.is_valid());
        assert!(!valid.is_deleted());

        let deleted = ElementStatus::Deleted;
        assert!(!deleted.is_valid());
        assert!(deleted.is_deleted());
    }

    /// A simple data provider that contains values consisting of floats and strings.
    ///
    /// The start point for this provider is as `u32::MAX`.
    struct SimpleProvider {
        data: Mutex<HashMap<u32, (f32, String)>>,
    }

    impl SimpleProvider {
        fn new(v: f32, st: String) -> Self {
            let mut data = HashMap::new();
            data.insert(u32::MAX, (v, st));
            Self {
                data: Mutex::new(data),
            }
        }
    }

    impl DataProvider for SimpleProvider {
        type Context = DefaultContext;
        // Use the identity mapping for IDs.
        type InternalId = u32;
        type ExternalId = u32;
        type Error = ANNError;

        fn to_internal_id(&self, _context: &DefaultContext, gid: &u32) -> Result<u32, ANNError> {
            Ok(*gid)
        }

        fn to_external_id(&self, _context: &DefaultContext, id: u32) -> Result<u32, ANNError> {
            Ok(id)
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Missing;

    impl std::fmt::Display for Missing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "key is missing")
        }
    }

    impl std::error::Error for Missing {}
    impl From<Missing> for ANNError {
        #[cold]
        fn from(missing: Missing) -> ANNError {
            ANNError::log_async_error(missing)
        }
    }

    always_escalate!(Missing);

    // An accessor for the `f32` portion of the data stored in the SimpleProvider.
    struct FloatAccessor<'a>(&'a SimpleProvider);
    impl HasId for FloatAccessor<'_> {
        type Id = u32;
    }
    impl Accessor for FloatAccessor<'_> {
        type Extended = f32;
        type Element<'a>
            = f32
        where
            Self: 'a;
        type ElementRef<'a> = f32;

        type GetError = Missing;

        fn get_element(
            &mut self,
            id: u32,
        ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
            let guard = self.0.data.lock().unwrap();
            let v = match guard.get(&id) {
                None => Err(Missing),
                Some(v) => Ok(v.0),
            };
            std::future::ready(v)
        }

        // Implement `on_elements_unordered` by only acquiring the lock once.
        //
        // Real implementations will need to take care to avoid deadlocks.
        async fn on_elements_unordered<Itr, F>(
            &mut self,
            itr: Itr,
            mut f: F,
        ) -> Result<(), Self::GetError>
        where
            Self: Sync,
            Itr: Iterator<Item = u32>,
            F: Send + FnMut(f32, u32),
        {
            let guard = self.0.data.lock().unwrap();
            for i in itr {
                match guard.get(&i) {
                    None => return Err(Missing),
                    Some(v) => f(v.0, i),
                }
            }
            Ok(())
        }
    }

    // An accessor for the `String` portion of the data stored in the SimpleProvider.
    //
    // We keep a local buffer `buf` into which the contents of the string are copied.
    // This allows us to elide allocating on `get_element` calls.
    struct StringAccessor<'a> {
        provider: &'a SimpleProvider,
        buf: String,
    }

    impl<'a> StringAccessor<'a> {
        fn new(provider: &'a SimpleProvider) -> Self {
            Self {
                provider,
                buf: String::new(),
            }
        }
    }

    impl HasId for StringAccessor<'_> {
        type Id = u32;
    }
    impl Accessor for StringAccessor<'_> {
        type Extended = String;
        type Element<'a>
            = &'a str
        where
            Self: 'a;
        type ElementRef<'a> = &'a str;

        type GetError = Missing;

        fn get_element(
            &mut self,
            id: u32,
        ) -> impl Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
            let guard = self.provider.data.lock().unwrap();
            let v = match guard.get(&id) {
                None => Err(Missing),
                Some(v) => {
                    self.buf.clone_from(&v.1);
                    Ok(&*self.buf)
                }
            };
            std::future::ready(v)
        }
    }

    #[tokio::test]
    async fn test_default_implementations() {
        let provider = SimpleProvider::new(-1.0, "hello".to_string());
        {
            let mut data = provider.data.lock().unwrap();
            data.insert(0, (0.0, "world".to_string()));
            data.insert(1, (1.0, "foo".to_string()));
            data.insert(2, (2.0, "bar".to_string()));
        }

        // Float accessor
        {
            let mut accessor = FloatAccessor(&provider);
            assert_eq!(accessor.get_element(0).await.unwrap(), 0.0);
            assert_eq!(accessor.get_element(1).await.unwrap(), 1.0);
            assert_eq!(accessor.get_element(u32::MAX).await.unwrap(), -1.0);

            let mut v = Vec::new();
            accessor
                .on_elements_unordered([2, 1, 0].into_iter(), |element, id| v.push((element, id)))
                .await
                .unwrap();

            assert_eq!(&v, &[(2.0, 2), (1.0, 1), (0.0, 0)]);

            // Test error propagation.
            // Trying to access element 3 will result in an error, which should be propagated
            // up.
            let err = accessor
                .on_elements_unordered([2, 1, 0, 3].into_iter(), |element, id| {
                    v.push((element, id))
                })
                .await
                .unwrap_err();
            assert_eq!(err, Missing);
        }

        // String accessor
        {
            let mut accessor = StringAccessor::new(&provider);
            assert_eq!(accessor.get_element(0).await.unwrap(), "world");
            assert_eq!(accessor.get_element(1).await.unwrap(), "foo");
            assert_eq!(accessor.get_element(u32::MAX).await.unwrap(), "hello");

            // This method tests the provided implementation of `on_elements_unordered`.
            let expected = [("bar", 2), ("foo", 1), ("world", 0)];

            let mut expected_iter = expected.into_iter();
            accessor
                .on_elements_unordered([2, 1, 0].into_iter(), |element, id| {
                    assert_eq!((element, id), expected_iter.next().unwrap());
                })
                .await
                .unwrap();
            assert!(expected_iter.next().is_none());

            // Test error propagation.
            // Trying to access element 3 will result in an error, which should be propagated
            // up.
            let mut expected_iter = expected.into_iter();
            let err = accessor
                .on_elements_unordered([2, 1, 0, 3].into_iter(), |element, id| {
                    assert_eq!((element, id), expected_iter.next().unwrap());
                })
                .await
                .unwrap_err();
            assert_eq!(err, Missing);
            assert!(expected_iter.next().is_none());
        }
    }

    /////////////////////////////////
    // Supported Accessor Patterns //
    /////////////////////////////////

    // This suite of tests ensure that patterns we want out of the `Accessor` associated
    // trait hierarchy are all supported.
    //
    // These include:
    //
    // * Accessors that always allocate.
    // * Accessors that simply reference the underlying store directly.
    // * Accessors that use a local buffer.

    #[derive(Debug)]
    struct Store {
        data: Box<[u8]>,
    }

    impl Store {
        fn new() -> Self {
            Self {
                data: Box::from([1, 2, 3, 4]),
            }
        }

        fn dim(&self) -> usize {
            self.data.len()
        }
    }

    macro_rules! common_test_accessor {
        ($T:ty) => {
            impl HasId for $T {
                type Id = u32;
            }

            impl BuildDistanceComputer for $T {
                type DistanceComputerError = Infallible;
                type DistanceComputer = <u8 as crate::utils::VectorRepr>::Distance;

                fn build_distance_computer(&self) -> Result<Self::DistanceComputer, Infallible> {
                    Ok(<u8 as crate::utils::VectorRepr>::distance(
                        diskann_vector::distance::Metric::L2,
                        None,
                    ))
                }
            }
        };
    }

    // An accessor that always allocates.
    struct Allocating<'a> {
        store: &'a Store,
    }

    impl<'a> Allocating<'a> {
        fn new(store: &'a Store) -> Self {
            Self { store }
        }
    }

    common_test_accessor!(Allocating<'_>);

    impl Accessor for Allocating<'_> {
        type Extended = Box<[u8]>;
        type Element<'a>
            = Box<[u8]>
        where
            Self: 'a;
        type ElementRef<'a> = &'a [u8];
        type GetError = Infallible;

        async fn get_element(&mut self, _: u32) -> Result<Box<[u8]>, Infallible> {
            Ok(self.store.data.clone())
        }
    }

    // An accessor that forwards - returning references directly into the underlying
    // store without reallocation or copying.
    struct Forwarding<'a> {
        store: &'a Store,
    }

    impl<'a> Forwarding<'a> {
        fn new(store: &'a Store) -> Self {
            Self { store }
        }
    }

    common_test_accessor!(Forwarding<'_>);

    impl<'provider> Accessor for Forwarding<'provider> {
        type Extended = &'provider [u8];
        // NOTE: The lifetime of `Element` is `'provider` - not `'a`. This is what makes
        // it a forwarding accessor.
        type Element<'a>
            = &'provider [u8]
        where
            Self: 'a;
        type ElementRef<'a> = &'a [u8];
        type GetError = Infallible;

        async fn get_element(&mut self, _: u32) -> Result<&'provider [u8], Infallible> {
            Ok(&*self.store.data)
        }
    }

    // An accessor that returns a non-reference type with a lifetime.
    struct Wrapping<'a> {
        store: &'a Store,
    }

    impl<'a> Wrapping<'a> {
        fn new(store: &'a Store) -> Self {
            Self { store }
        }
    }

    #[derive(Debug)]
    struct Wrapped<'a>(&'a [u8]);

    impl<'a> Reborrow<'a> for Wrapped<'_> {
        type Target = &'a [u8];
        fn reborrow(&'a self) -> Self::Target {
            self.0
        }
    }

    impl From<Wrapped<'_>> for Box<[u8]> {
        fn from(wrapped: Wrapped<'_>) -> Self {
            wrapped.0.into()
        }
    }

    common_test_accessor!(Wrapping<'_>);

    impl Accessor for Wrapping<'_> {
        type Extended = Box<[u8]>;
        type Element<'a>
            = Wrapped<'a>
        where
            Self: 'a;
        type ElementRef<'a> = &'a [u8];
        type GetError = Infallible;

        async fn get_element(&mut self, _: u32) -> Result<Wrapped<'_>, Infallible> {
            Ok(Wrapped(&self.store.data))
        }
    }

    // An accessor that shares local state.
    #[derive(Debug)]
    struct Sharing<'a> {
        store: &'a Store,
        local: Box<[u8]>,
    }

    impl<'a> Sharing<'a> {
        fn new(store: &'a Store) -> Self {
            Self {
                store,
                local: (0..store.dim()).map(|_| 0).collect(),
            }
        }
    }

    common_test_accessor!(Sharing<'_>);

    impl Accessor for Sharing<'_> {
        type Extended = Box<[u8]>;
        type Element<'a>
            = &'a [u8]
        where
            Self: 'a;
        type ElementRef<'a> = &'a [u8];
        type GetError = Infallible;

        async fn get_element(&mut self, _: u32) -> Result<&[u8], Infallible> {
            self.local.copy_from_slice(&self.store.data);
            Ok(&self.local)
        }
    }

    #[tokio::test]
    async fn test_accessor_patterns() {
        let store = Store::new();

        // A slice against which we compute distances.
        let base: &[u8] = &[2, 3, 4, 5];

        {
            let mut accessor = Allocating::new(&store);
            let computer = accessor.build_distance_computer().unwrap();

            let element = accessor.get_element(0).await.unwrap();
            assert_eq!(computer.evaluate_similarity(base, element.reborrow()), 4.0);
        }

        {
            let mut accessor = Forwarding::new(&store);
            let computer = accessor.build_distance_computer().unwrap();

            let element = accessor.get_element(0).await.unwrap();
            assert_eq!(computer.evaluate_similarity(base, element.reborrow()), 4.0);
        }

        {
            let mut accessor = Wrapping::new(&store);
            let computer = accessor.build_distance_computer().unwrap();

            let element = accessor.get_element(0).await.unwrap();
            assert_eq!(computer.evaluate_similarity(base, element.reborrow()), 4.0);
        }

        {
            let mut accessor = Sharing::new(&store);
            let computer = accessor.build_distance_computer().unwrap();

            let element = accessor.get_element(0).await.unwrap();
            assert_eq!(computer.evaluate_similarity(base, element.reborrow()), 4.0);
        }
    }
}
