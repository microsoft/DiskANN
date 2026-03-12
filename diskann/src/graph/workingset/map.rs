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
    graph::glue,
};

use super::{Unseeded, Fill, AsWorkingSet, ScopedMap};

/// Blanket implementation of [`Fill`] for [`Map`]-backed working sets.
///
/// This covers the common case where the accessor's `Extended` type is stored directly
/// in the map. Accessors that need custom fill logic (e.g. hybrid full-precision/quantized)
/// should use a different `State` type and provide their own `Fill` impl.
impl<A, V, P> Fill<Map<A::Id, V, P>> for A
where
    P: Projection,
    A: for<'a> Accessor<Id: Hash + Eq, ElementRef<'a> = P::ElementRef<'a>>,
    V: Project<P> + Send + Sync + 'static,
    for<'a> A::ElementRef<'a>: Into<V>,
{
    type Error = <A::GetError as ToRanked>::Error;
    type Set<'a>
        = ScopedMapView<'a, A::Id, V, P>
    where
        Self: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        state: &'a mut Map<A::Id, V, P>,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::Set<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = A::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        fill_map(self, state, itr)
    }
}

impl<K, V, P: Projection> AsWorkingSet<Map<K, V, P>> for Unseeded {
    fn as_working_set(&self, capacity: usize) -> Map<K, V, P> {
        Map::with_capacity(capacity)
    }
}

/////////////////
// Projections //
/////////////////

pub trait Projection: Send + Sync + 'static {
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send;
    type ElementRef<'a>;
}

#[derive(Debug)]
pub struct Ref<T: ?Sized>(std::marker::PhantomData<T>);

impl<T> Clone for Ref<T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Ref<T> where T: ?Sized {}

impl<T> Projection for Ref<T>
where
    T: ?Sized + Send + Sync + 'static,
{
    type Element<'a> = &'a T;
    type ElementRef<'a> = &'a T;
}

#[derive(Debug)]
pub struct Reborrowed<T>(std::marker::PhantomData<T>)
where
    T: ?Sized;

impl<T> Clone for Reborrowed<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Reborrowed<T> {}

impl<T> Projection for Reborrowed<T>
where
    T: for<'a> Reborrow<'a> + ?Sized + Send + Sync + 'static,
{
    type Element<'a> = AsReborrowed<'a, T>;
    type ElementRef<'a> = <T as Reborrow<'a>>::Target;
}

#[derive(Debug)]
pub struct AsReborrowed<'a, T>(&'a T)
where
    T: ?Sized;

impl<T> Clone for AsReborrowed<'_, T>
where
    T: ?Sized,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for AsReborrowed<'_, T> where T: ?Sized {}

impl<'a, T> Reborrow<'a> for AsReborrowed<'_, T>
where
    T: Reborrow<'a> + ?Sized,
{
    type Target = T::Target;
    fn reborrow(&'a self) -> Self::Target {
        self.0.reborrow()
    }
}

impl<T> Project<Reborrowed<T>> for T
where
    T: for<'a> Reborrow<'a> + ?Sized + Send + Sync + 'static,
{
    fn project(&self) -> AsReborrowed<'_, T> {
        AsReborrowed(self)
    }
}

pub trait Project<P>
where
    P: Projection,
{
    fn project(&self) -> P::Element<'_>;
}

impl<T> Project<Ref<T>> for Box<T>
where
    T: Send + Sync + 'static + ?Sized,
{
    fn project(&self) -> &T {
        &self
    }
}

impl<T> Project<Ref<[T]>> for Vec<T>
where
    T: Send + Sync + 'static,
{
    fn project(&self) -> &[T] {
        &self
    }
}

// ============
// = MapSeed
// ============

/// Object-safe seed overlay trait, parameterized by [`Projection`].
///
/// The seed stores pre-populated elements (e.g. from a multi-insert batch) and
/// returns them as `P::Element<'_>` — the same type that [`ScopedMapView`] produces
/// from the fill layer via [`Project`]. This decouples the seed's storage type from
/// the fill layer's value type `V`.
pub trait Batch<K, P: Projection>: Send + Sync {
    fn get(&self, key: &K) -> Option<P::Element<'_>>;

    fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }
}

impl<K, V, P> Batch<K, P> for hashbrown::HashMap<K, V>
where
    K: Hash + Eq + Send + Sync,
    V: Project<P> + Send + Sync,
    P: Projection,
{
    fn get(&self, key: &K) -> Option<P::Element<'_>> {
        hashbrown::HashMap::get(self, key).map(|v| v.project())
    }

    fn contains_key(&self, key: &K) -> bool {
        hashbrown::HashMap::contains_key(self, key)
    }
}

/// Zero-copy seed overlay backed by a [`glue::Batch`](glue::Batch).
///
/// Stores the batch behind an `Arc` and an index map from keys to batch positions.
/// Lookups convert batch elements to `P::Element` via `Into`, which is by-value and
/// avoids the double-reference problem that `Project::project(&self)` would have.
struct BatchOverlay<K, B> {
    batch: Arc<B>,
    indices: hashbrown::HashMap<K, usize>,
}

impl<K, B, P> Batch<K, P> for BatchOverlay<K, B>
where
    K: Hash + Eq + Send + Sync,
    B: for<'a> glue::Batch<Element<'a>: Into<P::Element<'a>>>,
    P: Projection,
{
    fn get(&self, key: &K) -> Option<P::Element<'_>> {
        self.indices.get(key).map(|&idx| self.batch.get(idx).into())
    }

    fn contains_key(&self, key: &K) -> bool {
        self.indices.contains_key(key)
    }
}

/// Seed for [`Map`]-backed working sets.
///
/// Wraps an `Arc<dyn Batch<K, P>>` backed by a zero-copy [`BatchOverlay`] over a
/// `glue::Batch`. Cheap to clone (just an `Arc` bump). Convert to a [`Map`] via
/// [`AsWorkingSet::as_working_set`].
pub struct MapSeed<K, P: Projection> {
    overlay: Arc<dyn Batch<K, P>>,
}

impl<K, P: Projection> Clone for MapSeed<K, P> {
    fn clone(&self) -> Self {
        Self {
            overlay: Arc::clone(&self.overlay),
        }
    }
}

impl<K, P> MapSeed<K, P>
where
    K: Hash + Eq + Send + Sync + 'static,
    P: Projection,
{
    /// Create a zero-copy seed backed directly by a [`glue::Batch`](glue::Batch).
    ///
    /// Lookups go through the batch via an index map — no element data is copied.
    /// Requires that batch elements convert to `P::Element` via `Into`.
    pub fn from_batch<B>(batch: &Arc<B>, ids: impl ExactSizeIterator<Item = K>) -> Self
    where
        B: for<'a> glue::Batch<Element<'a>: Into<P::Element<'a>>>,
    {
        let indices: hashbrown::HashMap<K, usize> =
            ids.enumerate().map(|(i, id)| (id, i)).collect();
        Self {
            overlay: Arc::new(BatchOverlay {
                batch: Arc::clone(batch),
                indices,
            }),
        }
    }

    // TODO(mark): better empty() implementation
    pub fn empty() -> Self {
        todo!("MapSeed::empty needs a cleaner implementation")
    }
}

impl<K, V, P> AsWorkingSet<Map<K, V, P>> for MapSeed<K, P>
where
    P: Projection,
{
    fn as_working_set(&self, capacity: usize) -> Map<K, V, P> {
        Map {
            map: hashbrown::HashMap::with_capacity(capacity),
            seed: Some(Arc::clone(&self.overlay)),
            _projection: std::marker::PhantomData,
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
pub struct Map<K, V, P: Projection = Reborrowed<V>> {
    map: hashbrown::HashMap<K, V>,
    seed: Option<Arc<dyn Batch<K, P>>>,
    _projection: std::marker::PhantomData<P>,
}

impl<K, V, P: Projection> Map<K, V, P> {
    pub fn new() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            seed: None,
            _projection: std::marker::PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: hashbrown::HashMap::with_capacity(capacity),
            seed: None,
            _projection: std::marker::PhantomData,
        }
    }
}

impl<K, V, P: Projection> Default for Map<K, V, P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, P> Map<K, V, P>
where
    K: Hash + Eq,
    P: Projection,
{
    /// Look up an element in the fill (mutable) layer only.
    ///
    /// Does NOT check the seed overlay. For unified lookups, use
    /// [`scoped()`](Self::scoped) which returns a [`ScopedMapView`].
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
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
    /// Returns [`Entry::Seeded`] if the key is in the seed overlay (carrying the
    /// projected element), [`Entry::Occupied`] if it is in the mutable map layer, or
    /// [`Entry::Vacant`] if absent from both.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, P> {
        if let Some(element) = self.seed.as_ref().and_then(|s| s.get(&key)) {
            return Entry::Seeded(element);
        }
        match self.map.entry(key) {
            hash_map::Entry::Occupied(o) => Entry::Occupied(OccupiedEntry { entry: o }),
            hash_map::Entry::Vacant(v) => Entry::Vacant(VacantEntry { entry: v }),
        }
    }

    /// Borrow as a [`ScopedMap`]-compatible view.
    pub fn scoped(&self) -> ScopedMapView<'_, K, V, P> {
        ScopedMapView { map: self }
    }
}

/// Result of [`Map::entry`].
///
/// This is a three-state entry:
/// - [`Seeded`](Entry::Seeded): present in the immutable seed overlay. Carries the
///   projected element (`P::Element`).
/// - [`Occupied`](Entry::Occupied): present in the mutable map layer. The value can be
///   read and written.
/// - [`Vacant`](Entry::Vacant): absent from both layers.
pub enum Entry<'a, K, V, P: Projection> {
    /// The key exists in the immutable seed overlay.
    Seeded(P::Element<'a>),
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

/// A [`ScopedMap`] view over a [`Map`], checking seed then fill layer on each lookup.
pub struct ScopedMapView<'a, K, V, P: Projection = Reborrowed<V>> {
    map: &'a Map<K, V, P>,
}

impl<K, V, P> ScopedMap<K> for ScopedMapView<'_, K, V, P>
where
    K: Hash + Eq,
    P: Projection,
    V: Project<P> + Send + Sync + 'static,
{
    type ElementRef<'a> = P::ElementRef<'a>;
    type Element<'a>
        = P::Element<'a>
    where
        Self: 'a;

    fn get(&self, id: K) -> Option<Self::Element<'_>> {
        // Check seed overlay first (zero-copy from batch), then fill layer (owned, projected).
        self.map
            .seed
            .as_ref()
            .and_then(|s| s.get(&id))
            .or_else(|| self.map.map.get(&id).map(|v| v.project()))
    }
}

/// Fill a [`Map`] using `accessor.get_element` for each missing key.
///
/// Transient errors are acknowledged and skipped. Only critical errors are propagated.
pub fn fill_map<'a, A, P, V, Itr>(
    accessor: &'a mut A,
    map: &'a mut Map<A::Id, V, P>,
    itr: Itr,
) -> impl SendFuture<Result<ScopedMapView<'a, A::Id, V, P>, <A::GetError as ToRanked>::Error>>
where
    A: Accessor,
    V: Send + Sync,
    Itr: ExactSizeIterator<Item = A::Id> + Send + Sync,
    for<'b> A::ElementRef<'b>: Into<V>,
    P: Projection,
{
    fill_map_projected(accessor, map, itr, |element| element.reborrow().into())
}

/// Fill a [`Map`] using a projection from `Accessor::Extended` to `V`.
///
/// Transient errors are acknowledged and skipped. Only critical errors are propagated.
pub fn fill_map_projected<'a, A, V, P, Itr, F>(
    accessor: &'a mut A,
    map: &'a mut Map<A::Id, V, P>,
    itr: Itr,
    projection: F,
) -> impl SendFuture<Result<ScopedMapView<'a, A::Id, V, P>, <A::GetError as ToRanked>::Error>>
where
    A: Accessor<Id: Hash + Eq>,
    V: Send + Sync,
    Itr: ExactSizeIterator<Item = A::Id> + Send + Sync,
    F: Fn(A::Element<'_>) -> V + Send,
    P: Projection,
{
    async move {
        for i in itr {
            match map.entry(i) {
                Entry::Seeded(_) | Entry::Occupied(_) => { /* in batch or already fetched */ }
                Entry::Vacant(vacant) => match accessor.get_element(i).await {
                    Ok(element) => {
                        vacant.insert(projection(element));
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

