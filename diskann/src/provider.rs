/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The [`DataProvider`] trait encompasses the following concepts:
//!
//! * Storage for arbitrary data that is contextually accessed through accessors constructed
//!   by [strategies](crate::graph::glue).
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
//! * [`HasId`]: Traits such as [`NeighborAccessor`] and [`DelegateNeighbors`] all need to
//!   interact with the underlying [`DataProvider`] using an internal ID type.
//!
//!   The [`HasId`] trait provides a common base-trait for these related concepts, which
//!   ensures implementers only need to define and constrain it once.

use std::ops::Deref;

use crate::{ANNResult, error::ToRanked, graph::AdjacencyList, utils::VectorId};

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
///
/// Indexing algorithms expose overloadable "strategies" that allow data providers to
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

    /// The operation guard returned by [`SetElement::set_element`] to notify `self` if an
    /// operation fails.
    ///
    /// This is required to be `'static` for multi-insert compatibility (where it is required
    /// to cross spawn boundaries).
    type Guard: Guard<Id = Self::InternalId> + 'static;

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
pub trait SetElement<T>: DataProvider {
    /// The kind of error yielded by `set_element`.
    type SetError: ToRanked + std::fmt::Debug + Send + Sync + 'static;

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
        element: T,
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

///////////////////////
// Neighbor Accessor //
///////////////////////

/// An accessor that provides random-access neighbor retrieval from a graph.
///
/// Generally, neighbor access and data access are logically decoupled, being served from
/// different stores. However, there are situations where data and neighbors are
/// interleaved in the underlying storage medium.
///
/// As such, data accessors used in conjunction with graph operations need to additionally
/// provide an implementation of this trait.
///
/// To avoid repeating implementations for every accessor flavor, the trait
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
        &mut self,
        id: Self::Id,
        neighbors: &mut AdjacencyList<Self::Id>,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;
}

/// A mutable extension of [`NeighborAccessor`] that enables the underlying graph to be
/// mutated.
///
/// Generally, data accessors should implement [`DelegateNeighbor`] instead of extending this
/// trait if graph and data access are naturally decoupled.
pub trait NeighborAccessorMut: NeighborAccessor {
    /// Overwrite the neighbor list for the node associated with `id`.
    fn set_neighbors(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;

    /// Append all entries in `neighbors` tothe neighbor list currently associated with `id`.
    ///
    /// The behavior when the resulting list exceeds some pre-configured capacity or
    /// contains duplicates is implementation defined.
    fn append_vector(
        &mut self,
        id: Self::Id,
        neighbors: &[Self::Id],
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send;

    /// A potentially optimized bulk implementation of [`Self::set_neighbors`].
    ///
    /// For each `id`/`neighbors` pair in `iter`, set the adjacency list for the node associated
    /// with `id` to `neighbors`.
    ///
    /// Implementations are allowed to commit entries out of order.
    fn set_neighbors_bulk<I, T>(
        &mut self,
        iter: I,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        I: Iterator<Item = (Self::Id, T)> + Send,
        T: Deref<Target = [Self::Id]> + Send,
    {
        async move {
            for (vector_id, neighbors) in iter {
                self.set_neighbors(vector_id, neighbors.deref()).await?;
            }
            Ok(())
        }
    }
}

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        task,
    };

    use pin_project::{pin_project, pinned_drop};

    use super::*;

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
}
