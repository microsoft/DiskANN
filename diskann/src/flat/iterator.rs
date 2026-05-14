/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sequential ("flat") access primitives.
//!
//! This module defines the traits that flat-search algorithms use to walk every element
//! of a [`DataProvider`](crate::provider::DataProvider) once.
//!
//! * [`DistancesUnordered`]: the single trait flat search consumes. It takes a
//!   pre-built query computer and a callback, applies the callback to every
//!   `(id, distance)` pair in the provider, and is the only trait an in-memory
//!   visitor (such as [`crate::flat::test::provider::Visitor`]) needs to implement.
//!   The super-traits [`HasId`] and [`BuildQueryComputer`] define the id and
//!   query-computer types.
//!
//! * [`FlatIterator`]: a convenient entry point for backends whose natural shape is
//!   element-at-a-time iteration. The trait exposes a single `next` method and an
//!   associated `Element<'_>` type that must be [`Reborrow`]able to the `ElementRef<'_>`
//!   exposed via the [`HasElementRef`] super-trait.
//!
//! * [`Iterated`]: bridges any [`FlatIterator`] implementation into a
//!   [`DistancesUnordered`] by looping over [`FlatIterator::next`] and scoring each
//!   element with the supplied computer.

use std::fmt::Debug;

use diskann_utils::{Reborrow, future::SendFuture};
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::ToRanked,
    provider::{BuildQueryComputer, HasElementRef, HasId},
};

/// Fused iterate-and-score primitive over the elements of a flat index.
///
/// Implementations drive an entire scan over the underlying data, scoring each
/// element with the supplied [`BuildQueryComputer::QueryComputer`] and invoking
/// `f` with the resulting `(id, distance)` pair. The super-trait
/// [`BuildQueryComputer<T>`] supplies the computer type.
pub trait DistancesUnordered<T>: HasId + BuildQueryComputer<T> + Send + Sync {
    /// The error type yielded by [`Self::distances_unordered`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Drive the entire scan, scoring each element with `computer` and invoking `f`
    /// with the resulting `(id, distance)` pair.
    fn distances_unordered<F>(
        &mut self,
        computer: &<Self as BuildQueryComputer<T>>::QueryComputer,
        f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(<Self as HasId>::Id, f32);
}

//////////////
// Iterator //
//////////////

/// A lending, asynchronous iterator over the elements of a flat index.
///
/// Implementations provide element-at-a-time access via [`Self::next`]. Providers that
/// only implement `FlatIterator` can be wrapped in [`Iterated`] to obtain an
/// [`OnElementsUnordered`] implementation automatically.
pub trait FlatIterator: HasId + HasElementRef + Send + Sync {
    /// The concrete element returned by [`Self::next`]. Reborrows to [`Self::ElementRef`].
    type Element<'a>: for<'b> Reborrow<'b, Target = <Self as HasElementRef>::ElementRef<'b>>
        + Send
        + Sync
    where
        Self: 'a;

    /// The error type yielded by [`Self::next`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Advance the iterator and asynchronously yield the next `(id, element)` pair.
    ///
    /// Returns `Ok(None)` when the scan is exhausted. The yielded element borrows from
    /// the iterator and is invalidated by the next call to `next`.
    #[allow(clippy::type_complexity)]
    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;
}

/////////////
// Default //
/////////////

/// Bridges a [`FlatIterator`] into a [`DistancesUnordered`] by looping over
/// [`FlatIterator::next`], reborrowing each element, and scoring it with the
/// supplied computer.
///
/// This is the default adapter for providers that implement element-at-a-time
/// iteration. Providers that can do better (prefetching, SIMD batching, bulk
/// I/O) should implement [`DistancesUnordered`] directly.
pub struct Iterated<I> {
    inner: I,
}

impl<I> Iterated<I> {
    /// Wrap an iterator to produce a [`DistancesUnordered`] implementation.
    pub fn new(inner: I) -> Self {
        Self { inner }
    }

    /// Unwrap, returning the inner iterator.
    pub fn into_inner(self) -> I {
        self.inner
    }
}

impl<I: HasId> HasId for Iterated<I> {
    type Id = I::Id;
}

impl<I: HasElementRef> HasElementRef for Iterated<I> {
    type ElementRef<'a> = I::ElementRef<'a>;
}

/// Forwards the inner iterator's [`BuildQueryComputer`] impl through the wrapper
/// so that callers (and the [`DistancesUnordered`] blanket below) can obtain the
/// query computer from the [`Iterated`] adapter directly.
impl<I, T> BuildQueryComputer<T> for Iterated<I>
where
    I: BuildQueryComputer<T>,
{
    type QueryComputerError = I::QueryComputerError;
    type QueryComputer = I::QueryComputer;

    fn build_query_computer(
        &self,
        from: T,
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner.build_query_computer(from)
    }
}

/// The blanket implementation of [`DistancesUnordered`] for any
/// [`FlatIterator`] that also exposes a [`BuildQueryComputer`].
impl<I, T> DistancesUnordered<T> for Iterated<I>
where
    I: FlatIterator + BuildQueryComputer<T> + Send + Sync,
{
    type Error = I::Error;

    fn distances_unordered<F>(
        &mut self,
        computer: &Self::QueryComputer,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(<Self as HasId>::Id, f32),
    {
        async move {
            while let Some((id, element)) = self.inner.next().await? {
                let dist = computer.evaluate_similarity(element.reborrow());
                f(id, dist);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt::Debug,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use diskann_utils::Reborrow;
    use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};

    use super::*;
    use crate::{
        ANNError, always_escalate,
        error::Infallible,
        provider::{BuildQueryComputer, HasElementRef, HasId},
        utils::VectorRepr,
    };

    ///////////////////////////
    // Shared sample dataset //
    ///////////////////////////

    /// Canonical sample dataset shared by every contract test below.
    fn sample_items() -> Vec<(u32, Vec<f32>)> {
        vec![
            (10, vec![0.0, 0.0]),
            (11, vec![1.0, 0.0]),
            (12, vec![0.0, 2.0]),
        ]
    }

    /// Backing store of `[f32]` vectors, used by every element-shape fixture
    /// below to cover [`FlatIterator::Element`] variants without re-implementing
    /// the data layout each time.
    struct Store {
        items: Vec<(u32, Vec<f32>)>,
    }

    impl Store {
        fn sample() -> Self {
            Self {
                items: sample_items(),
            }
        }
    }

    ///////////////////////
    // Common impl macro //
    ///////////////////////

    /// Implement [`HasId`], [`HasElementRef`], and [`BuildQueryComputer`] for an
    /// iterator type. Every fixture in this module shares these impls — only
    /// [`FlatIterator::Element`] varies. The [`DistancesUnordered`] impl on
    /// `Iterated<$T>` comes from the blanket impl in the parent module, so does
    /// not need to be repeated here.
    macro_rules! common_iterator_impls {
        ($T:ty) => {
            impl HasId for $T {
                type Id = u32;
            }

            impl HasElementRef for $T {
                type ElementRef<'a> = &'a [f32];
            }

            impl BuildQueryComputer<&[f32]> for $T {
                type QueryComputerError = Infallible;
                type QueryComputer = <f32 as VectorRepr>::QueryDistance;

                fn build_query_computer(
                    &self,
                    from: &[f32],
                ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
                    Ok(f32::query_distance(from, Metric::L2))
                }
            }
        };
    }

    /////////////////////////////////
    // Allocating: Element = Vec   //
    /////////////////////////////////

    /// `Element<'a> = Vec<f32>` — owns its data, reborrows to `&'a [f32]`.
    /// Mirrors the `Allocating` accessor in [`crate::provider`].
    struct Allocating<'a> {
        store: &'a Store,
        cursor: usize,
    }

    impl<'a> Allocating<'a> {
        fn new(store: &'a Store) -> Self {
            Self { store, cursor: 0 }
        }
    }

    common_iterator_impls!(Allocating<'_>);

    impl FlatIterator for Allocating<'_> {
        type Element<'a>
            = Vec<f32>
        where
            Self: 'a;
        type Error = Infallible;

        fn next(
            &mut self,
        ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
            async move {
                let i = self.cursor;
                if i >= self.store.items.len() {
                    return Ok(None);
                }
                self.cursor += 1;
                let (id, ref v) = self.store.items[i];
                Ok(Some((id, v.clone())))
            }
        }
    }

    ////////////////////////////////////////////////
    // Forwarding: Element = &'store [f32]        //
    ////////////////////////////////////////////////

    /// `Element<'a> = &'store [f32]` — borrows directly out of the underlying
    /// store. The element lifetime is tied to the *store* (not the iterator),
    /// proving the trait supports forwarding accessors. Mirrors the
    /// `Forwarding` accessor in [`crate::provider`].
    struct Forwarding<'store> {
        store: &'store Store,
        cursor: usize,
    }

    impl<'store> Forwarding<'store> {
        fn new(store: &'store Store) -> Self {
            Self { store, cursor: 0 }
        }
    }

    common_iterator_impls!(Forwarding<'_>);

    impl<'store> FlatIterator for Forwarding<'store> {
        type Element<'a>
            = &'store [f32]
        where
            Self: 'a;
        type Error = Infallible;

        fn next(
            &mut self,
        ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
            async move {
                let i = self.cursor;
                if i >= self.store.items.len() {
                    return Ok(None);
                }
                self.cursor += 1;
                let (id, ref v) = self.store.items[i];
                Ok(Some((id, v.as_slice())))
            }
        }
    }

    /////////////////////////////////////////////////////////
    // Wrapping: Element = guard-shaped non-ref `Wrapped`  //
    /////////////////////////////////////////////////////////

    /// A guard-shaped element that reborrows to `&'b [f32]` and counts its own
    /// drops. Mirrors the `Wrapping` accessor's `Wrapped<'a>` in
    /// [`crate::provider`], plus a [`Drop`] hook to verify the [`Iterated`]
    /// adapter does not leak guards.
    struct Wrapped<'g> {
        data: &'g [f32],
        drop_count: Arc<AtomicUsize>,
    }

    impl<'b> Reborrow<'b> for Wrapped<'_> {
        type Target = &'b [f32];
        fn reborrow(&'b self) -> Self::Target {
            self.data
        }
    }

    impl Drop for Wrapped<'_> {
        fn drop(&mut self) {
            self.drop_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// `Element<'a> = Wrapped<'a>`.
    struct Wrapping<'a> {
        store: &'a Store,
        cursor: usize,
        drop_count: Arc<AtomicUsize>,
    }

    impl<'a> Wrapping<'a> {
        fn new(store: &'a Store) -> Self {
            Self {
                store,
                cursor: 0,
                drop_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    common_iterator_impls!(Wrapping<'_>);

    impl FlatIterator for Wrapping<'_> {
        type Element<'a>
            = Wrapped<'a>
        where
            Self: 'a;
        type Error = Infallible;

        fn next(
            &mut self,
        ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
            async move {
                let i = self.cursor;
                if i >= self.store.items.len() {
                    return Ok(None);
                }
                self.cursor += 1;
                let (id, ref v) = self.store.items[i];
                Ok(Some((
                    id,
                    Wrapped {
                        data: v.as_slice(),
                        drop_count: self.drop_count.clone(),
                    },
                )))
            }
        }
    }

    /////////////////////////////////////////////////
    // Sharing: Element = &'a [f32] via local buf  //
    /////////////////////////////////////////////////

    /// `Element<'a> = &'a [f32]` — copies into an internal buffer per `next()`
    /// to avoid per-call allocation. Mirrors the `Sharing` accessor in
    /// [`crate::provider`].
    struct Sharing<'a> {
        store: &'a Store,
        cursor: usize,
        buf: Vec<f32>,
    }

    impl<'a> Sharing<'a> {
        fn new(store: &'a Store) -> Self {
            Self {
                store,
                cursor: 0,
                buf: Vec::new(),
            }
        }
    }

    common_iterator_impls!(Sharing<'_>);

    impl FlatIterator for Sharing<'_> {
        type Element<'a>
            = &'a [f32]
        where
            Self: 'a;
        type Error = Infallible;

        fn next(
            &mut self,
        ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
            async move {
                let i = self.cursor;
                if i >= self.store.items.len() {
                    return Ok(None);
                }
                self.cursor += 1;
                let (id, ref v) = self.store.items[i];
                self.buf.clear();
                self.buf.extend_from_slice(v);
                Ok(Some((id, self.buf.as_slice())))
            }
        }
    }

    ////////////////////////
    // Failing iterator   //
    ////////////////////////

    /// A critical (non-recoverable) error type the [`Failing`] iterator yields.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
    #[error("synthetic iterator failure at id {0}")]
    struct Boom(u32);

    always_escalate!(Boom);

    impl From<Boom> for ANNError {
        #[track_caller]
        fn from(boom: Boom) -> ANNError {
            ANNError::opaque(boom)
        }
    }

    /// `Element<'a> = &'a [f32]`, but `next()` returns `Err(Boom(id))` exactly
    /// once after `fail_after` successful yields. Used to verify error
    /// propagation through [`Iterated::on_elements_unordered`].
    struct Failing<'a> {
        store: &'a Store,
        cursor: usize,
        fail_after: usize,
    }

    impl HasId for Failing<'_> {
        type Id = u32;
    }

    impl HasElementRef for Failing<'_> {
        type ElementRef<'a> = &'a [f32];
    }

    impl BuildQueryComputer<&[f32]> for Failing<'_> {
        type QueryComputerError = Infallible;
        type QueryComputer = <f32 as VectorRepr>::QueryDistance;

        fn build_query_computer(
            &self,
            from: &[f32],
        ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
            Ok(f32::query_distance(from, Metric::L2))
        }
    }

    impl FlatIterator for Failing<'_> {
        type Element<'a>
            = &'a [f32]
        where
            Self: 'a;
        type Error = Boom;

        fn next(
            &mut self,
        ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>> {
            async move {
                let i = self.cursor;
                if i >= self.store.items.len() {
                    return Ok(None);
                }
                self.cursor += 1;
                let (id, ref v) = self.store.items[i];
                if i == self.fail_after {
                    return Err(Boom(id));
                }
                Ok(Some((id, v.as_slice())))
            }
        }
    }

    /////////////
    // Helpers //
    /////////////

    /// Build the canonical `(id, distance)` ground-truth list for a query under
    /// L2, against [`sample_items`].
    fn expected_distances(query: &[f32]) -> Vec<(u32, f32)> {
        let computer = f32::query_distance(query, Metric::L2);
        sample_items()
            .into_iter()
            .map(|(id, v)| (id, computer.evaluate_similarity(v.as_slice())))
            .collect()
    }

    ///////////
    // Tests //
    ///////////

    /// The blanket [`DistancesUnordered`] impl on [`Iterated`] produces the
    /// correct `(id, distance)` pairs for every supported
    /// [`FlatIterator::Element`] shape: owning, forwarding, guard-wrapped, and
    /// shared-buffer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distances_unordered() {
        let store = Store::sample();
        let query = vec![0.5_f32, 0.9];
        let expected = expected_distances(&query);

        async fn run<I>(mut visitor: Iterated<I>, query: &[f32], expected: &[(u32, f32)])
        where
            I: FlatIterator<Id = u32> + Send + Sync,
            I: for<'a> HasElementRef<ElementRef<'a> = &'a [f32]>,
            Iterated<I>: HasId<Id = u32> + for<'q> DistancesUnordered<&'q [f32]>,
        {
            let computer = visitor.build_query_computer(query).unwrap();
            let mut seen: Vec<(u32, f32)> = Vec::new();
            visitor
                .distances_unordered(&computer, |id, d| seen.push((id, d)))
                .await
                .unwrap();
            assert_eq!(seen, expected);
        }

        // Allocating: Element = Vec<f32> (owns).
        run(Iterated::new(Allocating::new(&store)), &query, &expected).await;

        // Forwarding: Element = &'store [f32] (borrows from store).
        run(Iterated::new(Forwarding::new(&store)), &query, &expected).await;

        // Round-trip through `Iterated::into_inner` to exercise the unwrap path.
        let recovered = Iterated::new(Forwarding::new(&store)).into_inner();
        run(Iterated::new(recovered), &query, &expected).await;

        // Wrapping: Element = Wrapped<'a> (guard-shaped non-ref).
        run(Iterated::new(Wrapping::new(&store)), &query, &expected).await;

        // Sharing: Element = &'a [f32] (per-call internal buffer).
        run(Iterated::new(Sharing::new(&store)), &query, &expected).await;
    }

    /// An error returned mid-iteration by [`FlatIterator::next`] propagates up
    /// through the [`Iterated`] adapter's [`DistancesUnordered`] impl, and the
    /// closure stops being invoked at the failure point.
    #[tokio::test]
    async fn failures_midstream() {
        let store = Store::sample();
        let mut visitor = Iterated::new(Failing {
            store: &store,
            cursor: 0,
            fail_after: 1, // Yield item 0 successfully, fail on item 1.
        });

        let query = vec![0.0_f32, 0.0];
        let computer = visitor.build_query_computer(query.as_slice()).unwrap();

        let mut seen: Vec<u32> = Vec::new();
        let err = visitor
            .distances_unordered(&computer, |id, _d| seen.push(id))
            .await
            .expect_err("Failing iterator must surface its error");

        assert_eq!(err, Boom(11));
        assert_eq!(
            seen,
            vec![10],
            "the closure must only see items yielded before the failure",
        );
    }
}
