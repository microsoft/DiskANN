/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`FlatIterator`] — the sequential access primitive for accessing a flat index.

use diskann_utils::{Reborrow, future::SendFuture};

use crate::{error::StandardError, provider::HasId};

/// A lending, asynchronous iterator over the elements of a flat index.
///
/// `FlatIterator` is the streaming counterpart to [`crate::provider::Accessor`]. Where an
/// accessor exposes random retrieval by id, a flat iterator exposes a *sequential* walk —
/// each call to [`Self::next`] advances an internal cursor and yields the next element.
///
/// Algorithms see only `(Id, ElementRef)` pairs and treat the stream as opaque.
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
    #[allow(clippy::type_complexity)]
    fn next(
        &mut self,
    ) -> impl SendFuture<Result<Option<(Self::Id, Self::Element<'_>)>, Self::Error>>;

    /// Drive the entire scan, invoking `f` for each yielded element.
    ///
    /// The default implementation loops over [`Self::next`].
    fn on_elements_unordered<F>(&mut self, mut f: F) -> impl SendFuture<Result<(), Self::Error>>
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
