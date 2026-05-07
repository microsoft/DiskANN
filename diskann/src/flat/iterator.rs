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
/// `OnElementsUnordered` is the streaming counterpart to [`crate::provider::Accessor`].
/// Where an accessor exposes random retrieval by id, this trait exposes a *sequential*
/// walk that invokes a caller-supplied closure for every element.
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
