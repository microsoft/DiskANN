/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
//! Lending-iterator entry point for flat-search providers.
//!
//! * [`FlatIterator`]: a convenient entry point for backends whose natural shape is
//!   element-at-a-time iteration. The trait exposes a single `next` method, an
//!   associated `ElementRef<'a>` GAT (the element shape used for distance scoring),
//!   and an associated `Element<'a>` that must be [`Reborrow`]able into `ElementRef<'b>`.
//!
//! * [`Iterated`]: bridges any [`FlatIterator`] implementation into a
//!   [`DistancesUnordered<C>`](super::DistancesUnordered) for any computer `C` whose
//!   [`PreprocessedDistanceFunction`] target matches the iterator's `ElementRef`.

use std::fmt::Debug;

use diskann_utils::{Reborrow, future::SendFuture};
use diskann_vector::PreprocessedDistanceFunction;

use crate::{error::ToRanked, flat::DistancesUnordered, provider::HasId};

/// A lending, asynchronous iterator over the elements of a flat index.
///
/// Implementations provide element-at-a-time access via [`Self::next`]. Providers that
/// only implement `FlatIterator` can be wrapped in [`Iterated`] to obtain a default
/// [`DistancesUnordered<C>`] implementation for any computer `C` whose
/// [`PreprocessedDistanceFunction`] target matches `Self::ElementRef`.
pub trait FlatIterator: HasId + Send + Sync {
    /// Lifetime is intentionally unconstrained so it can appear under HRTB without
    /// inducing a `'static` bound on `Self`.
    type ElementRef<'a>;

    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a;

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

/// Bridges a [`FlatIterator`] into a [`DistancesUnordered<C>`] by looping over
/// [`FlatIterator::next`], reborrowing each element, and scoring it with the supplied
/// computer.
pub struct Iterated<I> {
    inner: I,
}

impl<I> Iterated<I> {
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

/// The blanket implementation of [`DistancesUnordered<C>`] for any
/// [`FlatIterator`] paired with a computer that scores its `ElementRef`.
impl<I, C> DistancesUnordered<C> for Iterated<I>
where
    I: FlatIterator + Send + Sync,
    C: for<'b> PreprocessedDistanceFunction<I::ElementRef<'b>, f32> + Send + Sync,
{
    type ElementRef<'a> = I::ElementRef<'a>;
    type Error = I::Error;

    fn distances_unordered<F>(
        &mut self,
        computer: &C,
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
    use crate::{ANNError, always_escalate, error::Infallible, provider::HasId, utils::VectorRepr};

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

    /// Backing store of `[f32]` vectors, used by every element-shape fixture below.
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

    /////////////////////////////////
    // Allocating: Element = Vec   //
    /////////////////////////////////

    /// `Element<'a> = Vec<f32>` — owns its data, reborrows to `&'a [f32]`.
    struct Allocating<'a> {
        store: &'a Store,
        cursor: usize,
    }

    impl<'a> Allocating<'a> {
        fn new(store: &'a Store) -> Self {
            Self { store, cursor: 0 }
        }
    }

    impl HasId for Allocating<'_> {
        type Id = u32;
    }

    impl FlatIterator for Allocating<'_> {
        type ElementRef<'a> = &'a [f32];
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

    /// `Element<'a> = &'store [f32]` — borrows directly out of the underlying store.
    struct Forwarding<'store> {
        store: &'store Store,
        cursor: usize,
    }

    impl<'store> Forwarding<'store> {
        fn new(store: &'store Store) -> Self {
            Self { store, cursor: 0 }
        }
    }

    impl HasId for Forwarding<'_> {
        type Id = u32;
    }

    impl<'store> FlatIterator for Forwarding<'store> {
        type ElementRef<'a> = &'a [f32];
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

    /// A guard-shaped element that reborrows to `&'b [f32]` and counts its own drops.
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

    impl HasId for Wrapping<'_> {
        type Id = u32;
    }

    impl FlatIterator for Wrapping<'_> {
        type ElementRef<'a> = &'a [f32];
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

    /// `Element<'a> = &'a [f32]` — copies into an internal buffer per `next()` to avoid
    /// per-call allocation.
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

    impl HasId for Sharing<'_> {
        type Id = u32;
    }

    impl FlatIterator for Sharing<'_> {
        type ElementRef<'a> = &'a [f32];
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

    /// `Element<'a> = &'a [f32]`, but `next()` returns `Err(Boom(id))` exactly once
    /// after `fail_after` successful yields.
    struct Failing<'a> {
        store: &'a Store,
        cursor: usize,
        fail_after: usize,
    }

    impl HasId for Failing<'_> {
        type Id = u32;
    }

    impl FlatIterator for Failing<'_> {
        type ElementRef<'a> = &'a [f32];
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

    ///////////
    // Tests //
    ///////////

    /// The blanket [`DistancesUnordered`] impl on [`Iterated`] produces the correct
    /// `(id, distance)` pairs for every supported [`FlatIterator::Element`] shape:
    /// owning, forwarding, guard-wrapped, and shared-buffer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distances_unordered_default_impls() {
        let store = Store::sample();
        let query = vec![0.5_f32, 0.9];
        let computer = f32::query_distance(&query, Metric::L2);
        let expected: Vec<(u32, f32)> = sample_items()
            .into_iter()
            .map(|(id, v)| (id, computer.evaluate_similarity(v.as_slice())))
            .collect();

        async fn run<I>(
            mut visitor: Iterated<I>,
            computer: &<f32 as VectorRepr>::QueryDistance,
            expected: &[(u32, f32)],
        ) where
            I: FlatIterator<Id = u32> + Send + Sync,
            I: for<'a> FlatIterator<ElementRef<'a> = &'a [f32]>,
        {
            let mut seen: Vec<(u32, f32)> = Vec::new();
            visitor
                .distances_unordered(computer, |id, d| seen.push((id, d)))
                .await
                .unwrap();
            assert_eq!(seen, expected);
        }

        // Allocating: Element = Vec<f32> (owns).
        run(Iterated::new(Allocating::new(&store)), &computer, &expected).await;

        // Forwarding: Element = &'store [f32] (borrows from store).
        run(Iterated::new(Forwarding::new(&store)), &computer, &expected).await;

        // Round-trip through `Iterated::into_inner` to exercise the unwrap path.
        let recovered = Iterated::new(Forwarding::new(&store)).into_inner();
        run(Iterated::new(recovered), &computer, &expected).await;

        // Wrapping: Element = Wrapped<'a> (guard-shaped non-ref).
        run(Iterated::new(Wrapping::new(&store)), &computer, &expected).await;

        // Sharing: Element = &'a [f32] (per-call internal buffer).
        run(Iterated::new(Sharing::new(&store)), &computer, &expected).await;
    }

    /// An error returned mid-iteration by [`FlatIterator::next`] propagates up through
    /// the [`Iterated`] adapter's [`DistancesUnordered`] impl, and the closure stops
    /// being invoked at the failure point.
    #[tokio::test]
    async fn failures_midstream() {
        let store = Store::sample();
        let mut visitor = Iterated::new(Failing {
            store: &store,
            cursor: 0,
            fail_after: 1, // Yield item 0 successfully, fail on item 1.
        });

        let query = vec![0.0_f32, 0.0];
        let computer = f32::query_distance(&query, Metric::L2);

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
