/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::hash::Hash;
use std::sync::Arc;

use diskann_utils::{
    Reborrow,
    future::{AsyncFriendly, SendFuture},
    views::{self, Matrix},
};
use hashbrown::hash_map;

use crate::{
    ANNError,
    error::{RankedError, ToRanked, TransientError},
    provider::Accessor,
};

/////////////
// Exports //
/////////////

pub mod map;
pub use map::{Map, ScopedMapView, MapSeed};

////////////
// Traits //
////////////

/// Populate a working set `State` with element data from an accessor.
///
/// This is the successor to `FillSet`. The accessor knows how to fetch elements
/// (via `get_element` or bulk operations) and insert them into the provided state.
/// Fill is additive — elements already present in state (including any batch overlay)
/// are skipped.
pub trait Fill<State>: Accessor {
    type Error: Into<ANNError> + std::fmt::Debug + Send + Sync;

    type Set<'a>: for<'b> ScopedMap<Self::Id, ElementRef<'b> = Self::ElementRef<'b>> + Send + Sync
    where
        Self: 'a,
        State: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        state: &'a mut State,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::Set<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = Self::Id> + Clone + Send + Sync,
        Self: 'a;
}

pub trait AsWorkingSet<WorkingSet> {
    fn as_working_set(&self, capacity: usize) -> WorkingSet;
}

/// Read-only view into a working set. Used by `occlude_list` and distance computations.
pub trait ScopedMap<I> {
    type ElementRef<'a>;
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}

#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

