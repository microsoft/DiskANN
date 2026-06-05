/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use diskann_utils::Reborrow;

/////////////
// Exports //
/////////////

pub mod map;
pub use map::Map;

////////////
// Traits //
////////////

/// Read-only view into a [`PruneAccessor`].
///
/// Used by `occlude_list` and distance computations.
///
/// See Also: [`PruneAccessor::fill`], [`Map`], [`map::View`].
pub trait View<I> {
    /// The reborrowed element with an arbitrarily short lifetime.
    ///
    /// Corresponds to [`crate::provider::Accessor::ElementRef`].
    type ElementRef<'a>;

    /// The element returned from [`Self::get`].
    ///
    /// The reborrow bound allows this type to be decoupled from the type provided to
    /// distance computers, and as a proxy for `Copy`.
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    /// Retrieve element associated with `id` if available.
    ///
    /// Users can expect that if `Self` was constructed from [`PruneAccessor::fill`], then
    /// `id` will belong to the iterator argument of that function.
    ///
    /// Retrieval should be reasonably efficient, but is not called in the hot-loop of prune.
    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}
