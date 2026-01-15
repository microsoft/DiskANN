/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The caching provider wraps another `DataProvider` implementation and its associated accessors
//! and strategies, in order to provide a memory-backed cache for index terms. It is useful for
//! accelerating disk-backed providers where accessor operations are slow at the cost of the extra
//! memory for the cache.
//!
//! The real `DataProvider` is known as the **inner** provider.
//!
//! Reads from a caching provider first check the cache, then if the term is present, returns the
//! term; if not present, the underlying provider is accessed. If the underlying provider returns
//! the term, it is placed in the cache and then returned.
//!
//! Writes to a caching provider first write to the underlying provider and then evict the
//! generated internal ID from the cache.
//!
//! Errors are reported using [`CachingError`], which differentiates between an error yielded
//! by the inner provider and an error generated while performing cache operations.
//!
//! Several traits are needed to tie the inner [`DataProvider`] to the cache:
//!
//! * [`AsCacheAccessorFor`]: Create a cache accessor for a [`DataProvider`]/element type
//!   combination.
//!
//! * [`CachedFillSet`]: The caching version of [`FillSet`]. A provided implementation can
//!   be used if desired, or the behavior can be customized per [`Accessor`]/cache accessor
//!   pair.
//!
//! * [`CachedAsElement`]: The caching version of [`AsElement`].
//!
//! To make use of a caching provider, after creating the inner [`DataProvider`] (`DP`),
//! create a cache `C` that implements the above traits for `DP` and the accessors whose
//! operations you want to cache.
//!
//! With the `DP`/`C` pair, construct a [`CachingProvider`] and use the [`Cached`] strategy
//! to wrap strategies for the inner provider.
//!
//! # Implementing Caches
//!
//! Caches are expectged to implement the above traits as well as [`Evict`]. Access to the
//! cache is done through proxy objects called "cache accessors". These accessors should
//! implement the following traits:
//!
//! * [`ElementCache`]: Get and retrieve items from the cache.
//!
//! * [`NeighborCache`]: Get and retrieve adjacency list terms from the cache.
//!
//! The utilities like [`super::bf_cache::Cache`] and [`super::utils::Graph`] can be helpful
//! for writing custom caches.
//!
//! # Naming Conventions
//!
//! * `[X}Cache`: A trait implemented by a cache accessor.
//! * `Cached[X}`: The caching version of a [`DataProvider`] trait `X`. This is implemented
//!   by the inner [`Accessor`] and is customized for a cache accessor.
//! * The traits [`Evict`] and [`AsCacheAccessorFor`] are implemented by the **cache**.

use std::{
    collections::{HashMap, hash_map::Entry},
    fmt::Debug,
};

use diskann::{
    ANNResult,
    error::{self as core_error, IntoANNResult, StandardError, ToRanked},
    graph::{
        AdjacencyList, SearchOutputBuffer,
        glue::{
            self, AsElement, ExpandBeam, FillSet, InplaceDeleteStrategy, InsertStrategy, Pipeline,
            PruneStrategy, SearchExt, SearchPostProcessStep, SearchStrategy,
        },
    },
    neighbor::Neighbor,
    provider::{
        Accessor, AsNeighbor, BuildDistanceComputer, BuildQueryComputer, CacheableAccessor,
        DataProvider, DelegateNeighbor, Delete, ElementStatus, HasId, NeighborAccessor,
        NeighborAccessorMut, SetElement,
    },
};
use diskann_utils::{
    WithLifetime,
    future::{AssertSend, AsyncFriendly, SendFuture},
};
use thiserror::Error;

////////////
// Traits //
////////////

/// A missing cache entry. A value can be set in the cache using [`Self::set`].
#[derive(Debug)]
pub struct Missing<'a, C, I> {
    cache: &'a mut C,
    key: I,
}

impl<'a, C, I> Missing<'a, C, I>
where
    I: Clone,
{
    /// Set the current entry to `element` - consuming `self`.
    fn set<E>(self, element: &E::Of<'_>) -> Result<(), C::Error>
    where
        C: ElementCache<I, E>,
        E: WithLifetime,
    {
        self.cache.set_cached(self.key, element)
    }
}

/// The result of invoking [`ElementCache::entry`].
///
/// If an item is missing, the `Missing` variant can be used to set it with the true value.
#[derive(Debug)]
pub enum MaybeCached<'a, C, I, E>
where
    E: WithLifetime,
    C: ElementCache<I, E>,
    I: Clone,
{
    Present(E::Of<'a>),
    Missing(Missing<'a, C, I>),
}

/// Indicate that the implementor is a cache for elements of type `E` with keys of type `I`.
///
/// This is a **cache accessor** trait.
pub trait ElementCache<I, E>: Send + Sync + Sized
where
    E: WithLifetime,
    I: Clone,
{
    type Error: StandardError;

    /// Attempt to retrieve a cached element for the key. Return `Ok(Some)` is the value
    /// is in the cache and is well formed. Return `Ok(None)` if the value is not in the
    /// cache.
    ///
    /// Returns any critical error.
    fn get_cached(&mut self, key: I) -> Result<Option<E::Of<'_>>, Self::Error>;

    /// Attempt to store a value in the cache - returning any critical error.
    ///
    /// This will overwrite any existing value in the cache at the same key.
    ///
    /// # Note
    ///
    /// Because the canonical [`bf_tree::BfTree`] used as the cache does not indicate
    /// whether an existing item was inserted or over-written, that information is not
    /// communicated in the return type of this trait.
    fn set_cached(&mut self, key: I, v: &E::Of<'_>) -> Result<(), Self::Error>;

    /// Attempt to get an entry from the cache. If the requested key is present, returns
    /// [`MaybeCached::Present`] with the retrieved value. Otherwise, returns a
    /// [`MaybeCached::Missing`] that can be used to insert into the cache.
    ///
    /// This exists to work around the
    /// [`get_or_insert`](https://nikomatsakis.github.io/rust-belt-rust-2019/#72) pattern
    /// not working in Rust without the Polonius borrow checker. As such, it is difficult
    /// to implement manually and should be left as a provided implementation.
    ///
    /// ## Details
    ///
    /// The provided implementation uses the humorous
    /// [polonius-the-crab](https://docs.rs/polonius-the-crab/latest/polonius_the_crab/)
    /// crate to encapsulate the (sound) unsafe pattern required to implement `get_or_insert`
    /// without the next generation borrow checker.
    ///
    /// When (if) polonius lands, this function can be rewritten or discarded entirely.
    fn try_get(&mut self, key: I) -> Result<MaybeCached<'_, Self, I, E>, Self::Error> {
        use polonius_the_crab as ptc;
        type Output<E> = ptc::ForLt!(<E as WithLifetime>::Of<'_>);

        // This method returns either:
        //
        // 1. The result of a successful `get_cached` call with the returned element. The
        //    borrowed element will be scoped to the lifetime of `&mut self`.
        //
        // 2. A reborrow of `self`, which can be given to `MaybeCached::Missing`.
        let result_or_cache =
            ptc::polonius::<_, Result<(), Self::Error>, Output<E>>(self, |cache| {
                match cache.get_cached(key.clone()) {
                    Ok(Some(element)) => ptc::PoloniusResult::Borrowing(element),
                    Ok(None) => ptc::PoloniusResult::Owned(Ok(())),
                    Err(err) => ptc::PoloniusResult::Owned(Err(err)),
                }
            });

        match result_or_cache {
            ptc::PoloniusResult::Borrowing(v) => Ok(MaybeCached::Present(v)),
            ptc::PoloniusResult::Owned {
                value,
                input_borrow: cache, // This is a reborrow of `self`.
            } => {
                // A result indicating a cache miss if `Ok` or critical error if not.
                value?;

                // Return the reborrow of `self` in a `MaybeCached::Missing`.
                Ok(MaybeCached::Missing(Missing { cache, key }))
            }
        }
    }
}

/// Attempt to retrieve the element associated with `id` from the `cache`. If present, return
/// the cached item.
///
/// Otherwise, attempt to retrieve the element from `accessor`. If this operation is
/// successful, store the value in `cache`.
pub fn get_or_insert<'a, A, C>(
    accessor: &'a mut A,
    cache: &'a mut C,
    id: A::Id,
) -> impl SendFuture<Result<A::Element<'a>, CachingError<A::GetError, C::Error>>>
where
    A: CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    async move {
        match cache.try_get(id).map_err(CachingError::Cache)? {
            MaybeCached::Present(element) => Ok(A::from_cached(element)),
            MaybeCached::Missing(missing) => {
                let element = accessor
                    .get_element(id)
                    .await
                    .map_err(CachingError::Inner)?;
                missing
                    .set(A::as_cached(&element))
                    .map_err(CachingError::Cache)?;
                Ok(element)
            }
        }
    }
}

/// Reporting status for the results of a graph retrieval.
///
/// The value of `NeighborStatus` influences how the caching accessor will respond.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "NeighborStatus must be observed and acted on"]
pub enum NeighborStatus {
    /// The requested item was found in the cache.
    Hit,
    /// The requested item was not found in the cache and the accessor should attempt to
    /// fill the cache entry for this item.
    Miss,
    /// The requested item was not found in the cache and the accessor should not attempt
    /// to cache the value.
    Uncacheable,
}

/// A targeted variation of [`ElementCache`] specifically targeting adjacency list storage and
/// retrieval. The interface models [`diskann::glue::NeighborAccessor`] and
/// [`diskann::glue::NeighborAccessorMut`].
///
/// This is a **cache accessor** trait.
pub trait NeighborCache<I>: Send + Sync {
    /// A single unified error type between `get` and `set` to keep things simple.
    type Error: StandardError;

    /// Attempt to retrieve a cached adjacency lists, storing the retireved values into
    /// `neighbors`.
    ///
    /// On success, returns `Ok(true)`. If the value is not in the cache, returns `Ok(false)`
    /// and should leave `neighbors` unmodified.
    ///
    /// Critical errors are reported.
    fn try_get_neighbors(
        &mut self,
        id: I,
        neighbors: &mut AdjacencyList<I>,
    ) -> Result<NeighborStatus, Self::Error>;

    /// Attempt to store a value in the cache - returning any critical error.
    ///
    /// This will overwrite any existing value in the cache at the same key.
    fn set_neighbors(&mut self, id: I, neighbors: &[I]) -> Result<(), Self::Error>;

    /// Invalidate any cached adjacency list for `id`.
    ///
    /// # Note
    ///
    /// The canonical [`bf_tree::BfTree`] used as the implementation does not provide
    /// indication on whether keys are successfully removed, so that information cannot be
    /// reported through this interface.
    fn invalidate_neighbors(&mut self, id: I);
}

/// Invalidate *all* cached items associated with `id`.
pub trait Evict<I> {
    fn evict(&self, id: I);
}

/// Customization point for creating a cache accessor from a graph, tailored for a data
/// provider and specific element type.
///
/// * `DP`: The type of the underlying data provider.
/// * `E`: The [`diskann::provider::Accessor`] element type to retrieve from the cache.
///
/// Customization includes the creation of cache accessors tailored for the requested
/// element type. For example, difference cache accessors may be needed for full precision
/// versus quantized vectors.
pub trait AsCacheAccessorFor<'a, A>
where
    A: CacheableAccessor,
{
    /// The type of the returned accessor. This accessor is meant to interface directly
    /// with the underlying cache, providing caching services for the element type `E`.
    ///
    /// This should **not** be a [`CachingAccessor`] since the [`CachingAccessor`] will call
    /// this method internally when created.
    type Accessor: ElementCache<A::Id, A::Map>;

    /// Errors that can occur while creating the cache accessor.
    ///
    /// Implementations are encouraged to make this construction infallible if at all
    /// possible.
    type Error: StandardError;

    /// Return a cache accessor for the underlying `provider`.
    fn as_cache_accessor_for(
        &'a self,
        accessor: A,
    ) -> Result<CachingAccessor<A, Self::Accessor>, Self::Error>;
}

/// The caching equivalent of [`diskann::glue::FillSet`], implemented by the
/// **inner** [`Accessor`] for a cache accessor `C`. This allows the [`Accessor`] to
///
/// customize the interaction with the cache.
///
/// # Provided
///
/// The provided implementation iterates through `itr` and only attempts to mutate ids that
/// are not already present in `set`. The cache will first be checked and if an item is
/// not present, it will be retrieved via [`Self::get_element`] and inserted into the cache.
///
/// **ALL** errors are propagated eagerly by this method.
pub trait CachedFillSet<C>: CacheableAccessor
where
    C: ElementCache<Self::Id, Self::Map>,
{
    fn cached_fill_set<Itr>(
        &mut self,
        cache: &mut C,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> impl SendFuture<Result<(), CachingError<Self::GetError, C::Error>>>
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        async move {
            for i in itr {
                if let Entry::Vacant(e) = set.entry(i) {
                    match cache.try_get(i).map_err(CachingError::Cache)? {
                        // Conversion chain:
                        //
                        // * `C::Element -> A::Element<'_>` via `CacheableAccessor`.
                        // * `A::Element<'_> -> A::Extended` via `Into`.
                        MaybeCached::Present(element) => {
                            e.insert(Self::from_cached(element).into());
                        }
                        MaybeCached::Missing(missing) => {
                            let element = self.get_element(i).await.map_err(CachingError::Inner)?;
                            missing
                                .set(Self::as_cached(&element))
                                .map_err(CachingError::Cache)?;
                            e.insert(element.into());
                        }
                    }
                }
            }
            Ok(())
        }
    }
}

/// The caching equivalent of [`diskann::glue::AsElement`], implemented by the
/// **underlying** [`Accessor`]. This allows the [`Accessor`] to customize the interaction
/// with the cache.
pub trait CachedAsElement<T, C>: CacheableAccessor
where
    T: Send,
{
    type Error: ToRanked + std::fmt::Debug + Send + Sync;

    /// Efficiently convert `vector` with corresponding internal `id` to `Self::Element`.
    fn cached_as_element<'a>(
        &'a mut self,
        cache: &'a mut C,
        vector: T,
        id: Self::Id,
    ) -> impl SendFuture<Result<Self::Element<'a>, Self::Error>>;
}

///////////////
// New Types //
///////////////

/// A [`diskann::provider::DataProvider`] that provides a caching service for the
/// underlying provider of type `T` using a cache of type `C`.
///
/// Search and insert strategies must be wrapped inside the thin [`Cached`] strategy wrapper
/// to avoid Rust's orphan rule for implementations.
///
/// Provider access will be done using the [`CachingAccessor`] type, which wraps an
/// [`Accessor`] to the inner provider and a caching interface layer.
///
/// Some amount of work is required to properly interface the underlying provider with the
/// cache layer.
///
/// * [`AsCacheAccessorFor`]: Create an accessor for the underlying cache targeting the
///   underlying provider with a specific element type.
///
/// * [`CachedFillSet`]: [`diskann::glue::FillSet`] specialization for the cache.
///
/// * [`CachedAsElement`]: [`diskann::glue::AsElement`] specialization for the cache.
pub struct CachingProvider<T, C> {
    provider: T,
    cache: C,
}

impl<T, C> CachingProvider<T, C> {
    /// Construct a new [`CachingProvider`] tying together the underlying `provider` and `cache`.
    pub fn new(provider: T, cache: C) -> Self {
        Self { provider, cache }
    }

    /// Return a reference to the underlying provider.
    pub fn inner(&self) -> &T {
        &self.provider
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &C {
        &self.cache
    }
}

/// A generic [`Accessor`] that ties together an [`Accessor`] of type `A` for an underlying
/// [`diskann::provider::DataProvider`] and cache accesssor `C`.
///
/// To be useful, `C` should implment [`ElementCache<A::Id, A::Element>`] and
/// [`NeighborCache<A::Id>`].
#[derive(Debug)]
pub struct CachingAccessor<A, C> {
    inner: A,
    cache: C,
}

impl<A, C> CachingAccessor<A, C> {
    /// Construct a new [`CachingAccessor`] directly over the inner and cache accessors.
    pub fn new(inner: A, cache: C) -> Self {
        Self { inner, cache }
    }

    /// Return a reference to the inner accessor for the underlying provider.
    pub fn inner(&self) -> &A {
        &self.inner
    }

    /// Return a reference to the cache accessor.
    pub fn cache(&self) -> &C {
        &self.cache
    }
}

/// A new-type wrapper for inner strategies to interface with [`CachingProvider`].
///
/// The implementations of [`SearchStrategy`] and related items will use the associated
/// strategies for the underlying type `S`, but propagate [`Cached`] to those strategies.
#[derive(Debug, Clone, Copy)]
pub struct Cached<S> {
    strategy: S,
}

impl<S> Cached<S> {
    /// Construct a new [`Cached`] around the inner `strategy`.
    pub fn new(strategy: S) -> Self {
        Self { strategy }
    }
}

//----------------//
// Error Handling //
//----------------//

/// Error type associated with cache access related operations.
///
/// The goal of this type is to propagate critical cache related errors in addition to
/// any errors yielded by the underlying accessor or provider.
#[derive(Debug, Error)]
pub enum CachingError<E, C> {
    #[error("encountered error from backing provider")]
    Inner(#[source] E),
    #[error("encountered error while accessing cache")]
    Cache(#[source] C),
}

#[cfg(test)]
impl<E, C> CachingError<E, C>
where
    E: Debug,
    C: Debug,
{
    fn expect_inner(self) -> E {
        match self {
            Self::Inner(e) => e,
            Self::Cache(c) => panic!("expected an `Inner` error but got a `Cache`: {:?}", c),
        }
    }
}

/// A local new-type for working with [`ToRanked`] [`CachingError`]s.
#[derive(Debug, Error)]
#[error(transparent)]
pub struct Transient<T>(T);

/// Support conversion to `ANNError` only when the inner error type `E` is convertible to
/// `ANNError`. Even though this is not strictly necessary, it keeps us from implementiong
/// `IntoANNError` when `E: ToRanked` since `ToRanked` does not imply `IntoANNError`.
impl<E, C> From<CachingError<E, C>> for diskann::ANNError
where
    E: Into<diskann::ANNError>,
    C: StandardError,
{
    #[track_caller]
    fn from(err: CachingError<E, C>) -> Self {
        match err {
            CachingError::Inner(inner) => inner.into(),
            CachingError::Cache(err) => err.into(),
        }
    }
}

/// A transparent wrapper for `T as core_error::TranesientError<E>`.
impl<E, C, T> core_error::TransientError<CachingError<E, C>> for Transient<T>
where
    T: core_error::TransientError<E>,
{
    #[track_caller]
    fn acknowledge<D>(self, why: D)
    where
        D: std::fmt::Display,
    {
        self.0.acknowledge(why)
    }

    #[track_caller]
    fn escalate<D>(self, why: D) -> CachingError<E, C>
    where
        D: std::fmt::Display,
    {
        CachingError::Inner(self.0.escalate(why))
    }

    #[track_caller]
    fn acknowledge_with<F, D>(self, why: F)
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        self.0.acknowledge_with(why)
    }

    #[track_caller]
    fn escalate_with<F, D>(self, why: F) -> CachingError<E, C>
    where
        F: FnOnce() -> D,
        D: std::fmt::Display,
    {
        CachingError::Inner(self.0.escalate_with(why))
    }
}

impl<E, C> core_error::ToRanked for CachingError<E, C>
where
    E: core_error::ToRanked,
    C: StandardError,
{
    /// Cache errors are always escaslated.
    type Error = CachingError<E::Error, C>;
    type Transient = Transient<E::Transient>;

    fn to_ranked(self) -> core_error::RankedError<Self::Transient, Self::Error> {
        use core_error::RankedError;
        match self {
            Self::Inner(err) => match err.to_ranked() {
                RankedError::Transient(v) => core_error::RankedError::Transient(Transient(v)),
                RankedError::Error(v) => core_error::RankedError::Error(CachingError::Inner(v)),
            },
            Self::Cache(err) => core_error::RankedError::Error(CachingError::Cache(err)),
        }
    }

    fn from_transient(transient: Self::Transient) -> Self {
        Self::Inner(E::from_transient(transient.0))
    }

    fn from_error(error: Self::Error) -> Self {
        match error {
            CachingError::Inner(err) => Self::Inner(E::from_error(err)),
            CachingError::Cache(err) => Self::Cache(err),
        }
    }
}

///////////////////
// Data Provider //
///////////////////

impl<T, C> DataProvider for CachingProvider<T, C>
where
    T: DataProvider,
    C: AsyncFriendly,
{
    type Context = T::Context;
    type Error = T::Error;
    type ExternalId = T::ExternalId;
    type InternalId = T::InternalId;

    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        self.provider.to_external_id(context, id)
    }

    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        self.provider.to_internal_id(context, gid)
    }
}

impl<DP, C> Delete for CachingProvider<DP, C>
where
    DP: DataProvider + Delete,
    C: Evict<DP::InternalId> + AsyncFriendly,
{
    fn delete(
        &self,
        context: &DP::Context,
        gid: &DP::ExternalId,
    ) -> impl Future<Output = Result<(), DP::Error>> + Send {
        self.provider.delete(context, gid)
    }

    fn release(
        &self,
        context: &DP::Context,
        id: DP::InternalId,
    ) -> impl Future<Output = Result<(), DP::Error>> + Send {
        // The very first thing we do is evict from the cache.
        //
        // This will always be correct, even if `release` somehow fails.
        self.cache.evict(id);
        self.provider.release(context, id)
    }

    fn status_by_internal_id(
        &self,
        context: &DP::Context,
        id: DP::InternalId,
    ) -> impl Future<Output = Result<ElementStatus, DP::Error>> + Send {
        self.provider.status_by_internal_id(context, id)
    }

    fn status_by_external_id(
        &self,
        context: &DP::Context,
        gid: &DP::ExternalId,
    ) -> impl Future<Output = Result<ElementStatus, DP::Error>> + Send {
        self.provider.status_by_external_id(context, gid)
    }
}

impl<DP, C, T> SetElement<T> for CachingProvider<DP, C>
where
    DP: SetElement<T>,
    T: Send + Sync + ?Sized,
    C: AsyncFriendly + Evict<DP::InternalId>,
{
    type SetError = DP::SetError;
    type Guard = DP::Guard;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &T,
    ) -> Result<Self::Guard, Self::SetError> {
        use diskann::provider::Guard;

        let guard = self.provider.set_element(context, id, element).await?;
        // Invalidate to ensure we don't have a stale local copy.
        self.cache.evict(guard.id());
        Ok(guard)
    }
}

//////////////
// Accessor //
//////////////

impl<A, C> HasId for CachingAccessor<A, C>
where
    A: HasId,
{
    type Id = A::Id;
}

impl<A, C> NeighborAccessor for CachingAccessor<A, &mut C>
where
    A: NeighborAccessor,
    C: NeighborCache<A::Id>,
{
    async fn get_neighbors(
        mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> ANNResult<Self> {
        // 1. If `status == NeighborStatus::Hit` - we're done.
        // 2. If `status == NeighborStatus::Miss` - retrieve from the inner accessor and
        //    fill the cache.
        // 3. If `status == NeighborStatus::Uncacheable` - retrieve from the inner accessor
        //    but do not fill the cache.
        let status = self
            .cache
            .try_get_neighbors(id, neighbors)
            .into_ann_result()?;
        if status != NeighborStatus::Hit {
            self.inner = self.inner.get_neighbors(id, neighbors).await?;
            if status != NeighborStatus::Uncacheable {
                self.cache.set_neighbors(id, neighbors).into_ann_result()?;
            }
        }

        Ok(self)
    }
}

impl<A, C> NeighborAccessorMut for CachingAccessor<A, &mut C>
where
    A: NeighborAccessorMut,
    C: NeighborCache<A::Id>,
{
    async fn set_neighbors(mut self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        self.inner = self.inner.set_neighbors(id, neighbors).await?;
        self.cache.invalidate_neighbors(id);

        Ok(self)
    }

    async fn append_vector(mut self, id: Self::Id, neighbors: &[Self::Id]) -> ANNResult<Self> {
        self.inner = self.inner.append_vector(id, neighbors).await?;
        self.cache.invalidate_neighbors(id);
        Ok(self)
    }
}

impl<'a, A, C> DelegateNeighbor<'a> for CachingAccessor<A, C>
where
    A: DelegateNeighbor<'a>,
    C: NeighborCache<Self::Id>,
{
    type Delegate = CachingAccessor<A::Delegate, &'a mut C>;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        CachingAccessor::new(self.inner.delegate_neighbor(), &mut self.cache)
    }
}

impl<A, C> Accessor for CachingAccessor<A, C>
where
    A: CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    type Extended = A::Extended;
    type Element<'a>
        = A::Element<'a>
    where
        Self: 'a;
    type ElementRef<'a> = A::ElementRef<'a>;

    type GetError = CachingError<A::GetError, C::Error>;

    async fn get_element(&mut self, id: Self::Id) -> Result<A::Element<'_>, Self::GetError> {
        get_or_insert(&mut self.inner, &mut self.cache, id)
            .send()
            .await
    }
}

impl<A, C> BuildDistanceComputer for CachingAccessor<A, C>
where
    A: BuildDistanceComputer + CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    type DistanceComputerError = A::DistanceComputerError;
    type DistanceComputer = A::DistanceComputer;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        self.inner.build_distance_computer()
    }
}

impl<T, A, C> BuildQueryComputer<T> for CachingAccessor<A, C>
where
    T: ?Sized,
    A: BuildQueryComputer<T> + CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    type QueryComputerError = A::QueryComputerError;
    type QueryComputer = A::QueryComputer;

    fn build_query_computer(
        &self,
        from: &T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner.build_query_computer(from)
    }
}

impl<T, A, C> AsElement<T> for CachingAccessor<A, C>
where
    A: CachedAsElement<T, C>,
    C: ElementCache<A::Id, A::Map>,
    T: Send,
{
    type Error = <A as CachedAsElement<T, C>>::Error;
    async fn as_element(
        &mut self,
        vector: T,
        id: Self::Id,
    ) -> Result<Self::Element<'_>, Self::Error> {
        self.inner
            .cached_as_element(&mut self.cache, vector, id)
            .await
    }
}

impl<A, C> FillSet for CachingAccessor<A, C>
where
    A: CacheableAccessor + CachedFillSet<C>,
    C: ElementCache<A::Id, A::Map>,
{
    fn fill_set<Itr>(
        &mut self,
        set: &mut HashMap<Self::Id, Self::Extended>,
        itr: Itr,
    ) -> impl Future<Output = Result<(), Self::GetError>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send + Sync,
    {
        self.inner.cached_fill_set(&mut self.cache, set, itr)
    }
}

impl<A, C, T> ExpandBeam<T> for CachingAccessor<A, C>
where
    T: ?Sized,
    A: BuildQueryComputer<T> + CacheableAccessor + AsNeighbor,
    C: ElementCache<A::Id, A::Map> + NeighborCache<A::Id>,
{
}

/// Post Process
#[derive(Debug, Default, Clone, Copy)]
pub struct Unwrap;

impl<A, C, T> SearchPostProcessStep<CachingAccessor<A, C>, T> for Unwrap
where
    T: ?Sized,
    A: BuildQueryComputer<T> + CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    type Error<NextError>
        = NextError
    where
        NextError: StandardError;

    type NextAccessor = A;

    fn post_process_step<I, B, Next>(
        &self,
        next: &Next,
        accessor: &mut CachingAccessor<A, C>,
        query: &T,
        computer: &<A as BuildQueryComputer<T>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl Future<Output = Result<usize, Self::Error<Next::Error>>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<A::Id> + Send + ?Sized,
        Next: glue::SearchPostProcess<Self::NextAccessor, T, A::Id> + Sync,
    {
        next.post_process(&mut accessor.inner, query, computer, candidates, output)
    }
}

impl<A, C> SearchExt for CachingAccessor<A, C>
where
    A: SearchExt + CacheableAccessor,
    C: ElementCache<A::Id, A::Map>,
{
    fn starting_points(&self) -> impl Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        self.inner.starting_points()
    }

    fn terminate_early(&mut self) -> bool {
        self.inner.terminate_early()
    }

    fn is_not_start_point(
        &self,
    ) -> impl Future<Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>> + Send
    {
        self.inner.is_not_start_point()
    }
}

//////////////
// Strategy //
//////////////

type SearchAccessor<'a, S, DP, T> = <S as SearchStrategy<DP, T>>::SearchAccessor<'a>;
type PruneAccessor<'a, S, DP> = <S as PruneStrategy<DP>>::PruneAccessor<'a>;

/// A description of what is happening with the trait requirements:
///
/// The strategy `S` needs to be a search strategy for the underlying provider. That
/// strategy has a `SearchAccessor` with an associated element type `E`.
///
/// We are requiring that the underlying cache `C` is convertible via `AsCacheAccessorFor`
/// to an implementation of `ElementCache` that is compatible with the element type `E` and
/// that the relevant accessor can also access the underlying graph.
impl<DP, C, T, S, E> SearchStrategy<CachingProvider<DP, C>, T> for Cached<S>
where
    T: ?Sized,
    DP: DataProvider,
    S: for<'a> SearchStrategy<DP, T, SearchAccessor<'a>: CacheableAccessor>,
    C: for<'a> AsCacheAccessorFor<
            'a,
            SearchAccessor<'a, S, DP, T>,
            Accessor: NeighborCache<DP::InternalId>,
            Error = E,
        > + AsyncFriendly,
    E: StandardError,
{
    type QueryComputer = S::QueryComputer;
    type SearchAccessor<'a> = CachingAccessor<
        SearchAccessor<'a, S, DP, T>,
        <C as AsCacheAccessorFor<'a, SearchAccessor<'a, S, DP, T>>>::Accessor,
    >;
    type SearchAccessorError = CachingError<S::SearchAccessorError, E>;
    type PostProcessor = Pipeline<Unwrap, S::PostProcessor>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a CachingProvider<DP, C>,
        context: &'a DP::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let inner = self
            .strategy
            .search_accessor(&provider.provider, context)
            .map_err(CachingError::Inner)?;

        provider
            .cache
            .as_cache_accessor_for(inner)
            .map_err(CachingError::Cache)
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Pipeline::new(Unwrap, self.strategy.post_processor())
    }
}

/// We need `S` to be a [`PruneStrategy`] for the underlying provider.
///
/// This strategy has an associated [`PruneElement`] type `E`
///
/// We are requiring the cache `C` to be convertible via [`AsCacheAccessorFor`] to an
/// implementation of `ElementCache` that is compatible with `E` and allows mutation of the
/// cached graph.
///
/// Finally, the underlying [`PruneAccessor`] needs to implement [`CachedFillSet`] for the
/// corresponding cached accessor.
impl<DP, C, S, E> PruneStrategy<CachingProvider<DP, C>> for Cached<S>
where
    DP: DataProvider,
    S: for<'a> PruneStrategy<DP, PruneAccessor<'a>: CacheableAccessor>,
    C: for<'a> AsCacheAccessorFor<
            'a,
            PruneAccessor<'a, S, DP>,
            Accessor: NeighborCache<DP::InternalId>,
            Error = E,
        > + AsyncFriendly,
    for<'a> S::PruneAccessor<'a>:
        CachedFillSet<<C as AsCacheAccessorFor<'a, PruneAccessor<'a, S, DP>>>::Accessor>,
    E: StandardError,
{
    type DistanceComputer = S::DistanceComputer;
    type PruneAccessor<'a> = CachingAccessor<
        PruneAccessor<'a, S, DP>,
        <C as AsCacheAccessorFor<'a, PruneAccessor<'a, S, DP>>>::Accessor,
    >;
    type PruneAccessorError = CachingError<S::PruneAccessorError, E>;

    fn prune_accessor<'a>(
        &'a self,
        provider: &'a CachingProvider<DP, C>,
        context: &'a DP::Context,
    ) -> Result<Self::PruneAccessor<'a>, Self::PruneAccessorError> {
        let inner = self
            .strategy
            .prune_accessor(&provider.provider, context)
            .map_err(CachingError::Inner)?;

        provider
            .cache
            .as_cache_accessor_for(inner)
            .map_err(CachingError::Cache)
    }
}

/// Surprisingly - the `where` clause for this, while not pretty, is not too bad.
impl<DP, C, T, S> InsertStrategy<CachingProvider<DP, C>, T> for Cached<S>
where
    DP: DataProvider,
    S: InsertStrategy<DP, T>,
    T: ?Sized,
    Cached<S>: SearchStrategy<CachingProvider<DP, C>, T>,
    Cached<S::PruneStrategy>: PruneStrategy<CachingProvider<DP, C>>,
    C: AsyncFriendly,
{
    type PruneStrategy = Cached<S::PruneStrategy>;
    fn prune_strategy(&self) -> Self::PruneStrategy {
        Cached {
            strategy: self.strategy.prune_strategy(),
        }
    }
}

/// More surprisingly - the `where` clause for this implementation is **also** straightforward.
impl<DP, C, S> InplaceDeleteStrategy<CachingProvider<DP, C>> for Cached<S>
where
    DP: DataProvider,
    S: InplaceDeleteStrategy<DP>,
    Cached<S::PruneStrategy>: PruneStrategy<CachingProvider<DP, C>>,
    Cached<S::SearchStrategy>: for<'a> SearchStrategy<CachingProvider<DP, C>, S::DeleteElement<'a>>,
    C: AsyncFriendly,
{
    type DeleteElement<'a> = S::DeleteElement<'a>;
    type DeleteElementGuard = S::DeleteElementGuard;
    type DeleteElementError = S::DeleteElementError;

    type PruneStrategy = Cached<S::PruneStrategy>;
    type SearchStrategy = Cached<S::SearchStrategy>;

    fn prune_strategy(&self) -> Self::PruneStrategy {
        Cached {
            strategy: self.strategy.prune_strategy(),
        }
    }

    fn search_strategy(&self) -> Self::SearchStrategy {
        Cached {
            strategy: self.strategy.search_strategy(),
        }
    }

    fn get_delete_element<'a>(
        &'a self,
        provider: &'a CachingProvider<DP, C>,
        context: &'a DP::Context,
        id: DP::InternalId,
    ) -> impl Future<Output = Result<Self::DeleteElementGuard, Self::DeleteElementError>> + Send
    {
        self.strategy
            .get_delete_element(&provider.provider, context, id)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{
        fmt::Display,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use diskann::{
        ANNError,
        error::{RankedError, ToRanked, TransientError},
    };

    #[derive(Debug, Default)]
    struct Counters {
        acknowledge: AtomicUsize,
        acknowledge_with: AtomicUsize,
        escalate: AtomicUsize,
        escalate_with: AtomicUsize,
    }

    #[derive(Debug)]
    struct TransientErr {
        counters: Arc<Counters>,
        token: usize,
    }

    impl TransientErr {
        fn new(counters: &Arc<Counters>, token: usize) -> Self {
            Self {
                counters: counters.clone(),
                token,
            }
        }
    }

    #[derive(Debug, Error)]
    #[error("super critical error: {0}")]
    struct Critical(usize);

    impl From<Critical> for ANNError {
        fn from(err: Critical) -> Self {
            ANNError::opaque(err)
        }
    }

    #[derive(Debug)]
    enum Generic {
        Transient(TransientErr),
        Critical(Critical),
    }

    impl TransientError<Critical> for TransientErr {
        fn acknowledge<D>(self, _why: D)
        where
            D: Display,
        {
            self.counters.acknowledge.fetch_add(1, Ordering::Relaxed);
        }

        fn acknowledge_with<F, D>(self, _why: F)
        where
            F: FnOnce() -> D,
            D: Display,
        {
            self.counters
                .acknowledge_with
                .fetch_add(1, Ordering::Relaxed);
        }

        fn escalate<D>(self, _why: D) -> Critical
        where
            D: Display,
        {
            self.counters.escalate.fetch_add(1, Ordering::Relaxed);
            Critical(self.token)
        }

        fn escalate_with<F, D>(self, _why: F) -> Critical
        where
            F: FnOnce() -> D,
            D: Display,
        {
            self.counters.escalate_with.fetch_add(1, Ordering::Relaxed);
            Critical(self.token)
        }
    }

    impl ToRanked for Generic {
        type Transient = TransientErr;
        type Error = Critical;

        fn to_ranked(self) -> RankedError<TransientErr, Critical> {
            match self {
                Self::Transient(e) => RankedError::Transient(e),
                Self::Critical(e) => RankedError::Error(e),
            }
        }

        fn from_transient(transient: TransientErr) -> Self {
            Self::Transient(transient)
        }

        fn from_error(error: Critical) -> Self {
            Self::Critical(error)
        }
    }

    #[derive(Debug, Error)]
    #[error("always a critical error")]
    struct AlwaysCritical;

    impl From<AlwaysCritical> for ANNError {
        fn from(err: AlwaysCritical) -> Self {
            ANNError::opaque(err)
        }
    }

    #[test]
    fn test_caching_error() {
        type TestError = CachingError<Critical, AlwaysCritical>;

        // Cache errors are always critical.
        let err = CachingError::<Generic, AlwaysCritical>::Cache(AlwaysCritical);
        assert!(matches!(
            err.to_ranked(),
            RankedError::Error(CachingError::Cache(AlwaysCritical))
        ));

        // Transient correctly forwards calls.
        let counters = Arc::new(Counters::default());

        let make_transient = || Transient(TransientErr::new(&counters, 10));

        <_ as TransientError<TestError>>::acknowledge(make_transient(), "");
        assert_eq!(counters.acknowledge.load(Ordering::Relaxed), 1);

        <_ as TransientError<TestError>>::acknowledge_with(make_transient(), || "");
        assert_eq!(counters.acknowledge_with.load(Ordering::Relaxed), 1);

        let err = <_ as TransientError<TestError>>::escalate(make_transient(), "").expect_inner();
        assert_eq!(counters.escalate.load(Ordering::Relaxed), 1);
        assert_eq!(err.0, 10);

        let err =
            <_ as TransientError<TestError>>::escalate_with(make_transient(), || "").expect_inner();
        assert_eq!(counters.escalate.load(Ordering::Relaxed), 1);
        assert_eq!(err.0, 10);
    }

    #[test]
    fn test_caching_error_to_ranked() {
        type Top = CachingError<Generic, AlwaysCritical>;
        type Crit = CachingError<Critical, AlwaysCritical>;

        let err = Top::Cache(AlwaysCritical);

        // Cache Errors
        assert!(
            matches!(
                err.to_ranked(),
                RankedError::Error(CachingError::<Critical, AlwaysCritical>::Cache(
                    AlwaysCritical
                ))
            ),
            "cache errors are always critical"
        );

        assert!(
            matches!(Top::from_error(Crit::Cache(AlwaysCritical)), Top::Cache(_)),
            "reassembling from Cache should preserve Cache"
        );

        let counters = Arc::new(Counters::default());

        // Inner - transient.
        let err = Top::Inner(Generic::Transient(TransientErr::new(&counters, 5)));
        assert!(
            matches!(
                err.to_ranked(),
                RankedError::Transient(Transient(TransientErr { .. }))
            ),
            "transient inner errors are transient"
        );

        assert!(
            matches!(
                Top::from_transient(Transient(TransientErr::new(&counters, 5))),
                Top::Inner(Generic::Transient(_)),
            ),
            "transient errors are still tranient",
        );

        // Inner - critical.
        let err = Top::Inner(Generic::Critical(Critical(2)));
        assert!(
            matches!(
                err.to_ranked(),
                RankedError::Error(CachingError::<Critical, AlwaysCritical>::Inner(_))
            ),
            "critical errors are critical"
        );

        assert!(
            matches!(
                Top::from_error(Crit::Inner(Critical(2))),
                Top::Inner(Generic::Critical(_)),
            ),
            "critical errors are still critical",
        );
    }
}
