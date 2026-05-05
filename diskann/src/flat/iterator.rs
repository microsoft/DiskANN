/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`OnElementsUnordered`] — the sequential access primitive for accessing a flat index.
//! [`DistancesUnordered`] — sub-trait of [`OnElementsUnordered`] and [`BuildQueryComputer`]
//! that computes distances over the elements in the flat index.
//!
//! [`FlatIterator`] — a lending async iterator that can be bridged into
//! [`OnElementsUnordered`] via [`Iterated`].

use diskann_utils::{Reborrow, future::SendFuture};
use diskann_vector::PreprocessedDistanceFunction;

use crate::{
    error::StandardError,
    provider::{BuildQueryComputer, HasElementRef, HasId},
};

/// Callback-driven sequential scan over the elements of a flat index.
///
/// `OnElementsUnordered` is the streaming counterpart to [`crate::provider::Accessor`].
/// Where an accessor exposes random retrieval by id, this trait exposes a *sequential*
/// walk that invokes a caller-supplied closure for every element.
///
/// Algorithms see only `(Id, ElementRef)` pairs and treat the stream as opaque.
pub trait OnElementsUnordered: HasId + HasElementRef + Send + Sync {
    /// The error type yielded by [`Self::on_elements_unordered`].
    type Error: StandardError;

    /// Drive the entire scan, invoking `f` for each yielded element.
    fn on_elements_unordered<F>(&mut self, f: F) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + for<'a> FnMut(Self::Id, <Self as HasElementRef>::ElementRef<'a>);
}

/// Extension of [`OnElementsUnordered`] that drives the scan with a query computer
/// produced by the visitor's [`BuildQueryComputer<T>`] impl, invoking a callback with
/// `(id, distance)` pairs.
///
/// This fuses the scan with a pre-processed query computer and runs over a
/// streaming visitor. It pulls the computer type from the implementor's own
/// [`BuildQueryComputer<T>`] impl.
///
/// The default implementation delegates to [`OnElementsUnordered::on_elements_unordered`],
/// calling `computer.evaluate_similarity` on each element.
///
/// ## Note
///
/// This is the flat analog to [`crate::provider::DistancesUnordered`] which runs over
/// a random-access [`crate::provider::Accessor`].
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
/// only implement `FlatIterator` can be wrapped in [`DefaultIteratedOperator`] to obtain
/// an [`OnElementsUnordered`] implementation automatically.
pub trait FlatIterator: HasId + HasElementRef + Send + Sync {
    /// The concrete element returned by [`Self::next`]. Reborrows to [`Self::ElementRef`].
    type Element<'a>: for<'b> Reborrow<'b, Target = <Self as HasElementRef>::ElementRef<'b>>
        + Send
        + Sync
    where
        Self: 'a;

    /// The error type yielded by [`Self::next`].
    type Error: StandardError;

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
