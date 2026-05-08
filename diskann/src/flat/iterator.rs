/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sequential ("flat") access primitives.
//!
//! This module defines the traits that flat-search algorithms use to walk every element
//! of a [`DataProvider`](crate::provider::DataProvider) once.
//!
//! * [`OnElementsUnordered`]: the lowest-level entry point and the only required trait
//!   to implement. It is a single-method trait that applies a caller-supplied closure
//!   to every `(id, element ref)` pair in the provider. The super-traits [`HasId`] and
//!   [`HasElementRef`] define the concrete id and element reference types.
//!
//! * [`DistancesUnordered`]: a sub-trait of [`OnElementsUnordered`] that takes a query
//!   computer and a closure (typically to filter results through a priority queue) and
//!   applies the closure to the `(id, distance)` pair for every element, with each
//!   distance computed using the supplied computer.
//!
//! * [`FlatIterator`]: a convenient entry point for backends whose natural shape is
//!   element-at-a-time iteration. The trait exposes a single `next` method and an
//!   associated `Element<'_>` type that must be [`Reborrow`]able to the `ElementRef<'_>`
//!   exposed via the [`HasElementRef`] super-trait.
//!
//! * [`Iterated`]: bridges any [`FlatIterator`] implementation into an
//!   [`OnElementsUnordered`] by looping over [`FlatIterator::next`].

use std::fmt::Debug;

use diskann_utils::{Reborrow, future::SendFuture};
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::ToRanked,
    provider::{BuildQueryComputer, HasElementRef, HasId},
};

/// Callback-driven sequential scan over the elements of a flat index.
///
/// Algorithms see only `(Id, ElementRef)` pairs and treat the stream as opaque.
pub trait OnElementsUnordered: HasId + HasElementRef + Send + Sync {
    /// The error type yielded by [`Self::on_elements_unordered`].
    type Error: ToRanked + Debug + Send + Sync + 'static;

    /// Drive the entire scan, invoking `f` for each yielded element.
    fn on_elements_unordered<F>(&mut self, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + for<'a> FnMut(Self::Id, <Self as HasElementRef>::ElementRef<'a>);
}

/// Extension of [`OnElementsUnordered`] that drives the scan with a query computer.
///
/// The computer is produced by the visitor's [`BuildQueryComputer<T>`] impl,
/// and invokes a callback with `(id, distance)` pairs.
///
/// This fuses the scan with a pre-processed query computer and runs over a
/// streaming visitor. It pulls the computer type from the implementor's own
/// [`BuildQueryComputer<T>`] impl.
///
/// The default implementation delegates to [`OnElementsUnordered::on_elements_unordered`],
/// calling `computer.evaluate_similarity` on each element.
pub trait DistancesUnordered<T>: OnElementsUnordered + BuildQueryComputer<T> {
    /// Drive the entire scan, scoring each element with `computer` and invoking `f` with
    /// the resulting `(id, distance)` pair.
    fn distances_unordered<F>(
        &mut self,
        computer: &<Self as BuildQueryComputer<T>>::QueryComputer,
        mut f: F,
    ) -> impl SendFuture<Result<(), <Self as OnElementsUnordered>::Error>>
    where
        F: Send + FnMut(<Self as HasId>::Id, f32),
    {
        self.on_elements_unordered(move |id, element| {
            let dist = computer.evaluate_similarity(element);
            f(id, dist);
        })
    }
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

/// Bridges a [`FlatIterator`] into an [`OnElementsUnordered`] by looping over
/// [`FlatIterator::next`] and reborrowing each element into the closure.
///
/// This is the default adapter for providers that implement element-at-a-time iteration.
/// Providers that can do better (prefetching, SIMD batching, bulk I/O) should implement
/// [`OnElementsUnordered`] directly.
pub struct Iterated<I> {
    inner: I,
}

impl<I> Iterated<I> {
    /// Wrap an iterator to produce an [`OnElementsUnordered`] implementation.
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

impl<I> OnElementsUnordered for Iterated<I>
where
    I: FlatIterator + HasId + Send + Sync,
{
    type Error = I::Error;

    fn on_elements_unordered<F>(&mut self, mut f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + for<'a> FnMut(Self::Id, Self::ElementRef<'a>),
    {
        async move {
            while let Some((id, element)) = self.inner.next().await? {
                f(id, element.reborrow());
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

    /// Implement [`HasId`], [`HasElementRef`], [`BuildQueryComputer`], and
    /// [`DistancesUnordered`] for an iterator type. Every fixture in this module
    /// shares these impls — only [`FlatIterator::Element`] varies.
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

            // Forward `BuildQueryComputer` through the `Iterated` adapter so the
            // `DistancesUnordered` supertrait bound is satisfied.
            impl BuildQueryComputer<&[f32]> for Iterated<$T> {
                type QueryComputerError = Infallible;
                type QueryComputer = <f32 as VectorRepr>::QueryDistance;

                fn build_query_computer(
                    &self,
                    from: &[f32],
                ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
                    Ok(f32::query_distance(from, Metric::L2))
                }
            }

            impl DistancesUnordered<&[f32]> for Iterated<$T> {}
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

    /// Drive `visitor.on_elements_unordered` to completion and assert the
    /// yielded `(id, element)` pairs equal [`sample_items`] in iteration order.
    async fn check_visitor<V>(visitor: &mut V)
    where
        V: OnElementsUnordered + HasId<Id = u32>,
        V: for<'a> HasElementRef<ElementRef<'a> = &'a [f32]>,
        V::Error: Debug,
    {
        let mut out = Vec::new();
        visitor
            .on_elements_unordered(|id, e: &[f32]| out.push((id, e.to_vec())))
            .await
            .unwrap();
        assert_eq!(out, sample_items());
    }

    ///////////
    // Tests //
    ///////////

    /// `Iterated::on_elements_unordered` is correct for every supported
    /// [`FlatIterator::Element`] shape: owning, forwarding, guard-wrapped, and
    /// shared-buffer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn default_implementations() {
        let store = Store::sample();

        // Allocating: Element = Vec<f32> (owns).
        check_visitor(&mut Iterated::new(Allocating::new(&store))).await;

        // Forwarding: Element = &'store [f32] (borrows from store).
        check_visitor(&mut Iterated::new(Forwarding::new(&store))).await;

        let recovered = Iterated::new(Forwarding::new(&store)).into_inner();

        check_visitor(&mut Iterated::new(recovered)).await;

        // Wrapping: Element = Wrapped<'a> (guard-shaped non-ref).
        check_visitor(&mut Iterated::new(Wrapping::new(&store))).await;

        // Sharing: Element = &'a [f32] (per-call internal buffer).
        check_visitor(&mut Iterated::new(Sharing::new(&store))).await;
    }

    /// The default body of [`DistancesUnordered::distances_unordered`] produces
    /// `(id, computer.evaluate_similarity(elem))` pairs for every element shape.
    #[tokio::test]
    async fn distances_unordered() {
        let store = Store::sample();
        let query = vec![0.5_f32, 0.9];
        let computer = f32::query_distance(&query, Metric::L2);
        let expected = sample_items()
            .into_iter()
            .map(|(id, v)| (id, computer.evaluate_similarity(v.as_slice())))
            .collect::<Vec<_>>();

        async fn run<I>(mut visitor: Iterated<I>, query: &[f32], expected: &[(u32, f32)])
        where
            I: FlatIterator<Id = u32> + Send + Sync,
            I: for<'a> HasElementRef<ElementRef<'a> = &'a [f32]>,
            Iterated<I>: HasId<Id = u32>
                + for<'q> BuildQueryComputer<
                    &'q [f32],
                    QueryComputerError = Infallible,
                    QueryComputer = <f32 as VectorRepr>::QueryDistance,
                > + for<'q> DistancesUnordered<&'q [f32]>,
        {
            let computer = visitor.build_query_computer(query).unwrap();
            let mut seen: Vec<(u32, f32)> = Vec::new();
            visitor
                .distances_unordered(&computer, |id, d| seen.push((id, d)))
                .await
                .unwrap();
            assert_eq!(seen, expected);
        }

        run(Iterated::new(Allocating::new(&store)), &query, &expected).await;
        run(Iterated::new(Forwarding::new(&store)), &query, &expected).await;
        run(Iterated::new(Wrapping::new(&store)), &query, &expected).await;
        run(Iterated::new(Sharing::new(&store)), &query, &expected).await;
    }

    /// An error returned mid-iteration by [`FlatIterator::next`] propagates up
    /// through [`Iterated::on_elements_unordered`], and the closure stops being
    /// invoked at the failure point.
    #[tokio::test]
    async fn failures_midstream() {
        let store = Store::sample();
        let mut visitor = Iterated::new(Failing {
            store: &store,
            cursor: 0,
            fail_after: 1, // Yield item 0 successfully, fail on item 1.
        });

        let mut seen: Vec<u32> = Vec::new();
        let err = visitor
            .on_elements_unordered(|id, _e: &[f32]| seen.push(id))
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
