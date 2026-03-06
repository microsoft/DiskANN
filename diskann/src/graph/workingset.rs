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

// =========
// = Traits
// =========

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

/// Blanket implementation of [`Fill`] for [`Map`]-backed working sets.
///
/// This covers the common case where the accessor's `Extended` type is stored directly
/// in the map. Accessors that need custom fill logic (e.g. hybrid full-precision/quantized)
/// should use a different `State` type and provide their own `Fill` impl.
impl<A> Fill<Map<A::Id, A::Extended>> for A
where
    A: Accessor<Id: Hash + Eq, Extended: 'static>,
{
    type Error = <A::GetError as ToRanked>::Error;
    type Set<'a>
        = ScopedMapView<'a, A::Id, A::Extended>
    where
        Self: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        state: &'a mut Map<A::Id, A::Extended>,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::Set<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = A::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        fill_map(self, state, itr)
    }
}

/// Read-only view into a working set. Used by `occlude_list` and distance computations.
pub trait ScopedMap<I> {
    type ElementRef<'a>;
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>
    where
        Self: 'a;

    fn get(&self, id: I) -> Option<Self::Element<'_>>;
}

/// Immutable batch overlay for pre-seeded elements (e.g. from multi-insert).
///
/// Object-safe by design so it can be stored as `Arc<dyn Batch<K, V>>`.
pub trait Batch<K, V>: Send + Sync {
    fn get(&self, key: &K) -> Option<&V>;

    fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K, V> Batch<K, V> for hashbrown::HashMap<K, V>
where
    K: Hash + Eq + Send + Sync,
    V: Send + Sync,
{
    fn get(&self, key: &K) -> Option<&V> {
        hashbrown::HashMap::get(self, key)
    }

    fn contains_key(&self, key: &K) -> bool {
        hashbrown::HashMap::contains_key(self, key)
    }
}

// =============================
// = Map (default working set)
// =============================

/// Default working set state backed by a `HashMap` with an optional immutable batch overlay.
///
/// For single insert, `batch` is `None`. For multi-insert, the batch is pre-seeded by
/// `MultiInsertStrategy` with elements from the insertion batch, avoiding redundant fetches
/// during [`Fill`].
///
/// Batch keys and map keys are kept disjoint: insertions for keys already present in
/// the batch are silently skipped.
pub struct Map<K, V> {
    map: hashbrown::HashMap<K, V>,
    batch: Option<Arc<dyn Batch<K, V>>>,
}

impl<K, V> Map<K, V> {
    pub fn new() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            batch: None,
        }
    }

    pub fn with_batch(batch: Arc<dyn Batch<K, V>>) -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            batch: Some(batch),
        }
    }
}

impl<K, V> Default for Map<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Map<K, V>
where
    K: Hash + Eq,
{
    /// Look up an element, checking the batch overlay first, then the map.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.batch
            .as_ref()
            .and_then(|b| b.get(key))
            .or_else(|| self.map.get(key))
    }

    /// Mutable access to the map's own entries. Returns `None` for batch-only keys.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    /// Insert into the mutable map layer. If the key exists in the batch, this is
    /// a no-op (batch keys and map keys are disjoint).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.batch.as_ref().map_or(false, |b| b.contains_key(&key)) {
            return None;
        }
        self.map.insert(key, value)
    }

    /// Returns `true` if the key is in the batch overlay or the map.
    pub fn contains_key(&self, key: &K) -> bool {
        self.batch.as_ref().map_or(false, |b| b.contains_key(key)) || self.map.contains_key(key)
    }

    /// Batch-aware entry API. Returns [`Entry::Occupied`] if the key is in the
    /// batch or the map, [`Entry::Vacant`] only if absent from both.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        if self.batch.as_ref().map_or(false, |b| b.contains_key(&key)) {
            return Entry::Occupied;
        }
        match self.map.entry(key) {
            hash_map::Entry::Occupied(_) => Entry::Occupied,
            hash_map::Entry::Vacant(v) => Entry::Vacant(VacantEntry { entry: v }),
        }
    }

    /// Borrow as a [`ScopedMap`]-compatible view.
    pub fn scoped(&self) -> ScopedMapView<'_, K, V> {
        ScopedMapView { map: self }
    }
}

/// Result of [`Map::entry`]. `Occupied` carries no data — the primary consumer
/// (fill) only needs to know whether to skip.
pub enum Entry<'a, K, V> {
    Occupied,
    Vacant(VacantEntry<'a, K, V>),
}

pub struct VacantEntry<'a, K, V> {
    entry: hash_map::VacantEntry<'a, K, V>,
}

impl<K, V> VacantEntry<'_, K, V> {
    pub fn insert(self, value: V)
    where
        K: Hash + Eq,
    {
        self.entry.insert(value);
    }

    pub fn key(&self) -> &K {
        self.entry.key()
    }
}

// ================================
// = ScopedMapView (Map → ScopedMap)
// ================================

/// A [`ScopedMap`] view over a [`Map`], checking batch then map on each lookup.
pub struct ScopedMapView<'a, K, V> {
    map: &'a Map<K, V>,
}

impl<K, V> ScopedMap<K> for ScopedMapView<'_, K, V>
where
    K: Hash + Eq,
    V: Send + Sync + 'static + for<'a> Reborrow<'a>,
{
    type ElementRef<'a> = <V as Reborrow<'a>>::Target;
    type Element<'a>
        = Ref<'a, V>
    where
        Self: 'a;

    fn get(&self, id: K) -> Option<Self::Element<'_>> {
        self.map.get(&id).map(Ref)
    }
}

// ==============
// = Ref wrapper
// ==============

#[derive(Debug, Clone, Copy)]
pub struct Ref<'a, T>(pub(crate) &'a T);

impl<'a, T> Reborrow<'a> for Ref<'_, T>
where
    T: Reborrow<'a>,
{
    type Target = T::Target;
    fn reborrow(&'a self) -> T::Target {
        self.0.reborrow()
    }
}

// ===============
// = Fill helpers
// ===============

pub fn identity<T>(x: T) -> T {
    x
}

/// Fill a [`Map`] using `accessor.get_element` for each missing key.
///
/// Transient errors are acknowledged and skipped. Only critical errors are propagated.
pub fn fill_map<'a, A, Itr>(
    accessor: &'a mut A,
    map: &'a mut Map<A::Id, A::Extended>,
    itr: Itr,
) -> impl SendFuture<Result<ScopedMapView<'a, A::Id, A::Extended>, <A::GetError as ToRanked>::Error>>
where
    A: Accessor,
    Itr: ExactSizeIterator<Item = A::Id> + Send + Sync,
{
    fill_map_projected(accessor, map, itr, identity)
}

/// Fill a [`Map`] using a projection from `Accessor::Extended` to `V`.
///
/// Transient errors are acknowledged and skipped. Only critical errors are propagated.
pub fn fill_map_projected<'a, A, V, Itr, F>(
    accessor: &'a mut A,
    map: &'a mut Map<A::Id, V>,
    itr: Itr,
    projection: F,
) -> impl SendFuture<Result<ScopedMapView<'a, A::Id, V>, <A::GetError as ToRanked>::Error>>
where
    A: Accessor,
    V: Send,
    A::Id: Hash + Eq,
    Itr: ExactSizeIterator<Item = A::Id> + Send + Sync,
    F: Fn(A::Extended) -> V + Send,
{
    async move {
        for i in itr {
            match map.entry(i) {
                Entry::Occupied => { /* in batch or already fetched */ }
                Entry::Vacant(vacant) => match accessor.get_element(i).await {
                    Ok(element) => {
                        vacant.insert(projection(element.into()));
                    }
                    Err(local_error) => match local_error.to_ranked() {
                        RankedError::Transient(transient) => {
                            transient.acknowledge(
                                "transient error during fill; element will be absent from working set",
                            );
                        }
                        RankedError::Error(critical) => {
                            return Err(critical);
                        }
                    },
                },
            }
        }
        Ok(map.scoped())
    }
}

///////////////
// Dense Set //
///////////////

#[derive(Debug, Clone)]
pub struct DenseMap<K, T> {
    pub data: Matrix<T>,
    pub indices: hashbrown::HashMap<K, usize>,
}

impl<K, T> DenseMap<K, T>
where
    K: Hash + Eq + Copy + AsyncFriendly,
    T: Default + AsyncFriendly,
{
    pub fn new(dim: usize) -> Self {
        Self {
            data: Matrix::new(views::Init(|| T::default()), 0, dim),
            indices: hashbrown::HashMap::new(),
        }
    }

    pub fn fill<A, Itr, F>(
        &mut self,
        accessor: &mut A,
        itr: Itr,
        mut projection: F,
    ) -> impl SendFuture<Result<(), A::GetError>>
    where
        A: Accessor<Id = K>,
        Itr: ExactSizeIterator<Item = K> + Send + Sync,
        F: FnMut(&mut [T], A::Element<'_>) + Send,
    {
        async move {
            if itr.len() > self.data.nrows() {
                let dim = self.data.ncols();
                self.data = Matrix::new(views::Init(|| T::default()), itr.len(), dim);
            }

            self.indices.clear();
            for (pos, (row, i)) in std::iter::zip(self.data.row_iter_mut(), itr).enumerate() {
                projection(row, accessor.get_element(i).await?);
                self.indices.insert(i, pos);
            }
            Ok(())
        }
    }

    pub fn project<P>(&self, projection: P) -> ScopedDenseMap<'_, P, K, T>
    where
        P: Projection<T>,
    {
        ScopedDenseMap {
            map: &self,
            projection,
        }
    }
}

pub trait Projection<T>: Send + Sync
where
    T: AsyncFriendly,
{
    type ElementRef<'a>;
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>>;
    fn project<'a>(&self, raw: &'a [T]) -> Self::Element<'a>;
}

pub struct ScopedDenseMap<'a, P, K, T> {
    map: &'a DenseMap<K, T>,
    projection: P,
}

impl<P, K, T> ScopedMap<K> for ScopedDenseMap<'_, P, K, T>
where
    P: Projection<T>,
    K: Hash + Eq,
    T: AsyncFriendly,
{
    type ElementRef<'a> = P::ElementRef<'a>;
    type Element<'a>
        = P::Element<'a>
    where
        Self: 'a;

    fn get(&self, id: K) -> Option<Self::Element<'_>> {
        self.map
            .indices
            .get(&id)
            .map(|row| self.projection.project(self.map.data.row(*row)))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl<T> Projection<T> for Identity
where
    T: AsyncFriendly,
{
    type ElementRef<'a> = &'a [T];
    type Element<'a> = &'a [T];

    fn project<'a>(&self, raw: &'a [T]) -> Self::Element<'a> {
        raw
    }
}
