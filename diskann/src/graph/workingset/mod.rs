/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! The working-set is a core scratch data structure used during the pruning
//!

use diskann_utils::{Reborrow, future::SendFuture};

use crate::{ANNError, provider::Accessor};

/////////////
// Exports //
/////////////

pub mod map;
pub use map::Map;

////////////
// Traits //
////////////

/// Populate a working set with element data from an accessor.
///
/// The accessor fetches elements (via `get_element` or bulk operations) and inserts
/// them into the provided state. Fill clears the state first — elements already present
/// in the seed overlay are skipped.
pub trait Fill<Set>: Accessor {
    type Error: Into<ANNError> + std::fmt::Debug + Send + Sync;

    type View<'a>: for<'b> View<Self::Id, ElementRef<'b> = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a,
        Set: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        set: &'a mut Set,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::View<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a;
}

pub trait AsWorkingSet<Set> {
    fn as_working_set(&self, capacity: usize) -> Set;
}

/// Read-only view into a working set. Used by `occlude_list` and distance computations.
pub trait View<I> {
    type ElementRef<'a>;
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}

#[derive(Debug, Clone, Copy)]
pub struct Unseeded;
