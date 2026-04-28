/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIterator`] — the sequential access primitive for flat search.

use diskann_utils::{Reborrow, future::SendFuture};

use crate::{error::StandardError, provider::HasId};

/// A lending, asynchronous iterator over the elements of a flat index.
///
/// `FlatIterator` is the streaming counterpart to [`crate::provider::Accessor`]. Where an
/// accessor exposes random retrieval by id, a flat iterator exposes a *sequential* walk —
/// each call to [`Self::next`] advances an internal cursor and yields the next element.
///
/// Like [`crate::provider::Accessor::get_element`], advancing the cursor is **async**: it
/// may need to await an I/O fetch (e.g., reading the next disk page, awaiting a network
/// response, etc.). Iterators backed by purely in-memory data should return a ready
/// future.
///
/// The iterator is responsible for:
/// - Choosing the iteration order (buffer-sequential, hash-walked, partitioned, …).
/// - Skipping items that should not be visible to the algorithm (deleted, obsolete, …).
/// - Holding any borrows / locks needed to keep the underlying storage alive.
///
/// Algorithms see only `(Id, ElementRef)` pairs and treat the stream as opaque.
///
/// # `Element` vs `ElementRef`
///
/// Same pattern as [`crate::provider::Accessor`]:
///
/// - `Element<'a>` is the type returned by `next`. Its lifetime is bound to the iterator
///   borrow at the call site, so only one element is live at a time.
/// - `ElementRef<'a>` is an unconstrained-lifetime reborrow used in distance-function
///   bounds. Required to keep [HRTB](https://doc.rust-lang.org/nomicon/hrtb.html) bounds
///   on query computers from forcing `Self: 'static`.
///
/// # Hot path
///
/// Algorithms drive the scan via [`Self::on_elements_unordered`]. The provided
/// implementation simply loops over [`Self::next`]; iterators that can amortize
/// per-element cost (prefetching the next chunk, batching distance computation,
/// performing SIMD-friendly bulk reads) should override it.
pub trait FlatIterator: HasId + Send + Sync {
    /// A reference to a yielded element with an unconstrained lifetime, suitable for
    /// distance-function HRTB bounds.
    type ElementRef<'a>;

    /// The concrete element returned by [`Self::next`]. Reborrows to [`Self::ElementRef`].
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a;

    /// The error type yielded by [`Self::next`] and [`Self::on_elements_unordered`].
    type Error: StandardError;

    /// Advance the iterator and asynchronously yield the next `(id, element)` pair.
    ///
    /// Returns `Ok(None)` when the scan is exhausted. The yielded element borrows from
    /// the iterator and is invalidated by the next call to `next`.
    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;

    /// Drive the entire scan, invoking `f` for each yielded element.
    ///
    /// The default implementation loops over [`Self::next`]. Implementations that benefit
    /// from bulk dispatch (prefetching, batched SIMD distance computation, etc.) should
    /// override this method.
    ///
    /// The order of invocation is unspecified and may differ between calls. The closure
    /// `f` is **synchronous**; if you need to await inside the per-element handler, drive
    /// the iterator manually with [`Self::next`].
    fn on_elements_unordered<F>(
        &mut self,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + for<'a> FnMut(Self::Id, Self::ElementRef<'a>),
    {
        async move {
            while let Some((id, element)) = self.next().await? {
                f(id, element.reborrow());
            }
            Ok(())
        }
    }
}

