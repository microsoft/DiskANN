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

#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

impl<K, V> AsWorkingSet<Map<K, V>> for Unseeded {
    fn as_working_set(&self, capacity: usize) -> Map<K, V> {
        Map::with_capacity(capacity)
    }
}

// ============
// = MapSeed
// ============

/// Seed for [`Map`]-backed working sets.
///
/// Wraps an `Arc<HashMap<K, V>>` containing pre-projected batch elements.
/// Cheap to clone (just an `Arc` bump). Convert to a [`Map`] via
/// [`AsWorkingSet::as_working_set`].
#[derive(Clone)]
pub struct MapSeed<K, V> {
    overlay: Arc<hashbrown::HashMap<K, V>>,
}

impl<K, V> MapSeed<K, V>
where
    K: Hash + Eq,
{
    /// Create a seed from a [`glue::Batch`](super::glue::Batch) and an iterator
    /// of IDs assigned to each batch element.
    ///
    /// Each `(index, id)` pair from the iterator is mapped through `project` to produce
    /// the stored value.
    pub fn from_batch<B, F>(
        batch: &Arc<B>,
        ids: impl ExactSizeIterator<Item = K>,
        project: F,
    ) -> Self
    where
        B: super::glue::Batch,
        F: Fn(B::Element<'_>) -> V,
    {
        let overlay: hashbrown::HashMap<K, V> = ids
            .enumerate()
            .map(|(i, id)| (id, project(batch.get(i))))
            .collect();
        Self {
            overlay: Arc::new(overlay),
        }
    }

    /// Create an empty seed (no batch overlay).
    pub fn empty() -> Self {
        Self {
            overlay: Arc::new(hashbrown::HashMap::new()),
        }
    }
}

impl<K, V> AsWorkingSet<Map<K, V>> for MapSeed<K, V>
where
    K: Hash + Eq,
{
    fn as_working_set(&self, capacity: usize) -> Map<K, V> {
        Map {
            map: hashbrown::HashMap::with_capacity(capacity),
            seed: Some(Arc::clone(&self.overlay)),
        }
    }
}

// =============================
// = Map (default working set)
// =============================

/// Default working set state backed by a `HashMap` with an optional immutable seed overlay.
///
/// For single insert, `seed` is `None`. For multi-insert, the seed is pre-populated by
/// [`MapSeed::from_batch`] with elements from the insertion batch, avoiding redundant
/// fetches during [`Fill`].
///
/// Seed keys and map keys are kept disjoint: insertions for keys already present in
/// the seed are silently skipped.
pub struct Map<K, V> {
    map: hashbrown::HashMap<K, V>,
    seed: Option<Arc<hashbrown::HashMap<K, V>>>,
}

impl<K, V> Map<K, V> {
    pub fn new() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            seed: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: hashbrown::HashMap::with_capacity(capacity),
            seed: None,
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
    /// Look up an element, checking the seed overlay first, then the map.
    pub fn get(&self, key: &K) -> Option<&V> {
        self.seed
            .as_ref()
            .and_then(|s| s.get(key))
            .or_else(|| self.map.get(key))
    }

    /// Mutable access to the map's own entries. Returns `None` for seed-only keys.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Insert into the mutable map layer. If the key exists in the seed, this is
    /// a no-op (seed keys and map keys are disjoint).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.seed.as_ref().map_or(false, |s| s.contains_key(&key)) {
            return None;
        }
        self.map.insert(key, value)
    }

    /// Returns `true` if the key is in the seed overlay or the map.
    pub fn contains_key(&self, key: &K) -> bool {
        self.seed.as_ref().map_or(false, |s| s.contains_key(key)) || self.map.contains_key(key)
    }

    /// Seed-aware entry API.
    ///
    /// Returns [`Entry::Seeded`] if the key is in the seed overlay,
    /// [`Entry::Occupied`] if it is in the mutable map layer, or
    /// [`Entry::Vacant`] if absent from both.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        if let Some(value) = self.seed.as_ref().and_then(|s| s.get(&key)) {
            return Entry::Seeded(value);
        }
        match self.map.entry(key) {
            hash_map::Entry::Occupied(o) => Entry::Occupied(OccupiedEntry { entry: o }),
            hash_map::Entry::Vacant(v) => Entry::Vacant(VacantEntry { entry: v }),
        }
    }

    /// Borrow as a [`ScopedMap`]-compatible view.
    pub fn scoped(&self) -> ScopedMapView<'_, K, V> {
        ScopedMapView { map: self }
    }
}

/// Result of [`Map::entry`].
///
/// This is a three-state entry:
/// - [`Seeded`](Entry::Seeded): present in the immutable seed overlay. The value can be
///   inspected but not mutated.
/// - [`Occupied`](Entry::Occupied): present in the mutable map layer. The value can be
///   read and written.
/// - [`Vacant`](Entry::Vacant): absent from both layers.
pub enum Entry<'a, K, V> {
    /// The key exists in the immutable seed overlay.
    Seeded(&'a V),
    /// The key exists in the mutable map layer.
    Occupied(OccupiedEntry<'a, K, V>),
    /// The key is absent from both layers.
    Vacant(VacantEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V> {
    entry: hash_map::OccupiedEntry<'a, K, V>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn get(&self) -> &V {
        self.entry.get()
    }

    pub fn get_mut(&mut self) -> &mut V {
        self.entry.get_mut()
    }

    pub fn into_mut(self) -> &'a mut V {
        self.entry.into_mut()
    }
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

/// A [`ScopedMap`] view over a [`Map`], checking seed then map on each lookup.
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
    V: Send + Sync,
    A::Id: Hash + Eq,
    Itr: ExactSizeIterator<Item = A::Id> + Send + Sync,
    F: Fn(A::Extended) -> V + Send,
{
    async move {
        for i in itr {
            match map.entry(i) {
                Entry::Seeded(_) | Entry::Occupied(_) => { /* in batch or already fetched */ }
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
