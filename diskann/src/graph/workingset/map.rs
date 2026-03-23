/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

//! The default implementation of a [workingset](super).
//!
//! This is meant to work out of the box with enough knobs to cover common cases.
//! Supported workflows include:
//!
//! * Controlling reuse of the working set across multiple prunes.
//! * Zero-copy seeding from batches passed to
//!   [`multi_insert`](crate::graph::DiskANNIndex::multi_insert).
//!
//! # Guidance
//!
//! The data structure here has a lot of knobs. This section contains guidance for what
//! defaults to use for developers depending on needs.
//!
//! ### Single insert, no need to tweak working set fills
//!
//! Use [`Map`] with the default projection. For example, if your internal ID type is `u32`
//! and your [`Accessor`] element type is convertible to a `Box<[T]>`, then use
//! `Map<u32, Box<[T]>>`, constructed as follows:
//!
//! ```
//! use diskann::graph::workingset::{Map, map::{Builder, Capacity}};
//!
//! let capacity = 10;
//! let mut map: Map<u32, Box<[f32]>> = Builder::new(Capacity::Default).build(capacity);
//! map.insert(1, Box::new([1.0]));
//! assert_eq!(&**map.get(&1).unwrap(), &[1.0]);
//! ```
//!
//! ### Multi insert, element types are slice-like
//!
//! If using [`multi_insert`](crate::graph::DiskANNIndex::multi_insert) and your elements
//! look like slices, then the [`Ref`] [projection](Projection) can be used. See the section
//! on element projection down below.
//!
//! # Concepts
//!
//! ## Working Set Reuse
//!
//! The core indexing algorithm will reuse the same working-set object (e.g. [`Map`]) across
//! multiple rounds of prune. Users can take advantage of this reuse to use the working set
//! as a cache, reusing vector retrieved from a previous round to reduce the total number of
//! vectors retrieved. This needs to be balanced with the potential extra memory overhead of
//! the cache.
//!
//! To that end, [`Map`] provides four different operating modes, controlled by the
//! [`Capacity`] enum:
//!
//! * [`Capacity::None`]: Do not reuse any vectors. In this case, calls to [`Map::fill`] will
//!   completely clear the set before any vectors are added.
//!
//! * [`Capacity::Default`]: Set the capacity of the [`Map`] to
//!   [`max_occlusion_size`](crate::graph::Config::max_occlusion_size). This allows
//!   some amount of reuse while keeping an upper bound on the amount of memory occupied by
//!   the map to that which prune can use anyways.
//!
//! * [`Capacity::Some`]: Override the allowed capacity of the [`Map`]. Enabling more
//!   aggressive caching than [`Capacity::Default`] while still limiting memory growth.
//!
//! * [`Capacity::Unbounded`]: Never clear entries from the cache. This can cause unbounded
//!   growth, so use with care.
//!
//! In any case where the working set is used as a cache - there's the possibility of stale
//! entries being resident in the working set. The core algorithm will only retain working
//! sets for the duration of a logical operation (adding of back-edges, for example) and will
//! not persist these sets across operations. This minimizes the risk of stale entries
//! negatively impacting graph quality.
//!
//! ### Eviction Policy
//!
//! [`Map`] uses a scheme to evict older entries during [`fill`](Map::fill)
//! (see also [`Map::prepare`]). First, the elements in the provided
//! iterator that already exist in the map are pinned. Unpinned entries will be evicted
//! until there is enough room to store all entries in the iterator. Finally, all missing
//! entries will be fetched.
//!
//! ## Zero Copy Batch Overlays
//!
//! [`glue::MultiInsertStrategy::finish`] provides the batch of inserted vectors for use as a
//! "seed" for the final working set. In situations where
//! [intra-batch candidates](`crate::graph::config::IntraBatchCandidates`) are used, this
//! provides an opportunity to use the existing data provided to multi-insert directly, rather
//! than fetching vectors from the underlying provider every time.
//!
//! This module provides functionality in the form of [`Overlay`]. The simplest constructor
//! to use is [`Overlay::from_batch`] which is designed to work natively with the
//! [`glue::Batch`] trait required by multi-insert. This can be fed to a [`Builder`] which
//! ultimately creates a [`Map`].
//!
//! ### Element Projection
//!
//! One fundamental problem of the overlay is that the way data is stored in the overlay and
//! the way it is stored natively in [`Map`] might be different. To accommodate this, the
//! [`Projection`] trait is introduced. Briefly, this allows decoupling of the storage in
//! [`Map`] and the overlay:
//!
//! ```
//! use std::sync::Arc;
//! use diskann_utils::views::Matrix;
//!
//! use diskann::graph::workingset::{Map, View, map::{Overlay, Ref, Builder, Capacity}};
//!
//! let batch = Matrix::<u32>::row_vector(Box::new([1, 2, 3]));
//!
//! // Construct an "overlay" with just one id.
//! //
//! // The overlay uses the `Ref` projection, yielding slices.
//! let overlay = Overlay::<u32, Ref<[u32]>>::from_batch(Arc::new(batch), [5]);
//! assert_eq!(overlay.get(&5).unwrap(), &[1, 2, 3]);
//!
//! // Next - we build a `Map` that stores its entries as `Box<[u32]>`.
//! let mut map: Map<_, Box<[u32]>, _> = Builder::new(Capacity::Default)
//!     .with_overlay(overlay)
//!     .build(5);
//!
//! map.insert(10, Box::new([10, 20, 30]));
//!
//! // Since we have an overlay, the map has an entry for id "5" as well as the "10" we
//! // just inserted.
//! let v = map.view();
//! assert_eq!(v.get(5).unwrap(), &[1, 2, 3]);
//! assert_eq!(v.get(10).unwrap(), &[10, 20, 30]);
//! ```

use std::{fmt::Debug, hash::Hash, sync::Arc};

use diskann_utils::{Reborrow, future::SendFuture};
use hashbrown::hash_map;

use crate::{
    error::{RankedError, ToRanked, TransientError},
    graph::glue,
    provider::Accessor,
};

use super::{AsWorkingSet, Fill};

/////////
// Map //
/////////

/// Default [working set](super) state backed by a `HashMap` with an optional overlay for
/// [`multi_insert`](crate::graph::glue::MultiInsertStrategy).
///
/// This struct uses [`Builder`] as its constructor, which implements [`AsWorkingSet`] for
/// multi-insert compatibility.
///
/// Additionally, [working set reuse](super#reuse-in-a-workingset) is optionally supported and
/// is controlled by the [`Capacity`] enum. See the [module level docs](self#working-set-reuse)
/// for more details.
///
/// When working set reuse is enabled, older entries are evicted using an unspecified
/// algorithm with the guarantee that items of the [`Fill`] iterator that are already present
/// will not be evicted.
///
/// For zero-copy multi-insert compatibility, an [`Overlay`] should be provided to the [`Builder`].
#[derive(Debug)]
pub struct Map<K, V, P: Projection = Reborrowed<V>> {
    /// Entries local to this `Map`.
    map: hashbrown::HashMap<K, Generation<V>>,

    /// Potential seed from an input batch.
    overlay: Option<Overlay<K, P>>,

    /// The target capacity to trim down each generation. The `map` may exceed this
    /// capacity if a call to `fill` exceeds the capacity on its own.
    ///
    /// Note the following:
    ///
    /// * `capacity` of zero triggers a clear on every fill operation.
    /// * `capacity` of `None` results in an unbounded map.
    capacity: Option<usize>,

    /// The current generation. We use this to control eviction of older entries.
    generation: u32,
}

impl<K, V, P: Projection> Map<K, V, P> {
    fn new_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }
}

impl<K, V, P> Map<K, V, P>
where
    K: Hash + Eq,
    P: Projection,
{
    /// Promote any items in `itr` to the current generation - returning the number of
    /// items that were promoted.
    fn promote<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: ExactSizeIterator<Item = K>,
    {
        itr.filter(|k| {
            if let Some(tagged) = self.map.get_mut(k) {
                tagged.generation = self.generation;
                true
            } else {
                false
            }
        })
        .count()
    }

    /// Evict up to `count` items from the map that do not belong to the current generation.
    ///
    /// Returns the number of items actually evicted.
    fn evict(&mut self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }

        let mut remaining = count;
        self.map.retain(|_, v| {
            if remaining == 0 {
                // Done with eviction - now we just need to ride out the rest of the `retain`
                // process.
                true
            } else if v.generation != self.generation {
                // Cannot underflow.
                remaining -= 1;
                false
            } else {
                // This item belong to the current generation - keep it.
                true
            }
        });

        count - remaining
    }

    /// Prepare the map for the next [`fill`](Self::fill) cycle.
    ///
    /// Pins entries whose keys appear in `itr` and evicts unpinned entries as needed
    /// to stay within the configured [`Capacity`]. Called automatically by [`fill`](Self::fill);
    /// only call directly if using [`fill_with`](Self::fill_with).
    pub fn prepare<Itr>(&mut self, itr: Itr)
    where
        Itr: ExactSizeIterator<Item = K>,
    {
        self.new_generation();

        // Only bother if a capacity was specified. Otherwise, we don't need to bother at
        // all with generations.
        if let Some(capacity) = self.capacity {
            if capacity == 0 {
                // Do not retain **any** items from previous iterations.
                self.map.clear();
            } else {
                let len = itr.len();
                let promoted = self.promote(itr);

                // This operation cannot underflow since `promoted` should be at most `len`.
                //
                // However, incorrect implementations of `ExactSizeIterator` could lie about
                // the length. This will catch such degenerate implementations in debug builds.
                //
                // In release builds - this is still fine because the wrapped result is used
                // as an upper bound, limited by the current size of `map` which is **not**
                // degenerate.
                let extra_needed = len - promoted;
                let final_size = self.map.len().saturating_add(extra_needed);

                if let Some(to_evict) = final_size.checked_sub(capacity) {
                    self.evict(to_evict);
                }
            }
        }
    }
}

impl<K, V, P> Map<K, V, P>
where
    K: Copy + Hash + Eq + Send + Sync,
    V: Send + Sync,
    P: Projection,
{
    /// Fill using `accessor.get_element` for each missing key.
    ///
    /// Calls [`prepare`](Self::prepare) to pin and evict entries, then fetches any
    /// keys still missing. For incremental fills that skip preparation, use
    /// [`fill_with`](Self::fill_with) directly.
    ///
    /// Transient errors are acknowledged and skipped. Only critical errors are propagated.
    pub fn fill<'a, A, Itr>(
        &'a mut self,
        accessor: &'a mut A,
        itr: Itr,
    ) -> impl SendFuture<Result<View<'a, K, V, P>, <A::GetError as ToRanked>::Error>>
    where
        A: for<'b> Accessor<Id = K, ElementRef<'b>: Into<V>>,
        Itr: ExactSizeIterator<Item = K> + Clone + Send + Sync,
    {
        self.prepare(itr.clone());
        self.fill_with(accessor, itr, |element| element.into())
    }

    /// Fill using a projection from `Accessor::Element` to `V`.
    ///
    /// Transient errors are acknowledged and skipped. Only critical errors are propagated.
    pub fn fill_with<'a, A, Itr, F>(
        &'a mut self,
        accessor: &'a mut A,
        itr: Itr,
        f: F,
    ) -> impl SendFuture<Result<View<'a, K, V, P>, <A::GetError as ToRanked>::Error>>
    where
        A: Accessor<Id = K>,
        Itr: ExactSizeIterator<Item = K> + Send + Sync,
        F: Fn(A::ElementRef<'_>) -> V + Send,
    {
        async move {
            for i in itr {
                match self.entry(i) {
                    Entry::Seeded(_) | Entry::Occupied(_) => { /* nothing to do */ }
                    Entry::Vacant(vacant) => match accessor.get_element(i).await {
                        Ok(element) => {
                            vacant.insert(f(element.reborrow()));
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
            Ok(self.view())
        }
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
    /// [`view()`](Self::view) which returns a [`View`].
    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key).map(|v| v.value())
    }

    /// Mutable access to the map's own entries. Returns `None` for seed-only keys.
    ///
    /// NOTE: Using this interface sets the generation of the associated value (if any)
    /// to the current generation.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key).map(|v| {
            v.generation = self.generation;
            v.value_mut()
        })
    }

    /// Remove all entries from the mutable layer, preserving capacity configuration.
    ///
    /// Callers should prefer [`prepare`](Self::prepare) instead if cache-like behavior
    /// is needed.
    pub fn clear(&mut self) {
        self.map.clear()
    }

    /// Insert into the mutable map layer. If the key exists in the seed, this is
    /// a no-op (seed keys and map keys are disjoint).
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.overlay.as_ref().is_some_and(|s| s.contains_key(&key)) {
            return None;
        }
        self.map
            .insert(key, Generation::new(value, self.generation))
            .map(|v| v.into_inner())
    }

    /// Returns `true` if the key is in the seed overlay or the map.
    pub fn contains_key(&self, key: &K) -> bool {
        self.overlay.as_ref().is_some_and(|s| s.contains_key(key)) || self.map.contains_key(key)
    }

    /// Seed-aware entry API.
    ///
    /// Returns [`Entry::Seeded`] if the key is in the seed overlay (carrying the
    /// projected element), [`Entry::Occupied`] if it is in the mutable map layer, or
    /// [`Entry::Vacant`] if absent from both.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, P> {
        if let Some(element) = self.overlay.as_ref().and_then(|s| s.get(&key)) {
            return Entry::Seeded(element);
        }
        let generation = self.generation;
        match self.map.entry(key) {
            hash_map::Entry::Occupied(mut o) => {
                // Side effect: Set the entry's generation to the current generation.
                //
                // Most of the time, this will happen after a call to `prepare()` and so
                // the generation should already match.
                //
                // Doing it this way decreases the workload on the caller needing to manually
                // bump the generation.
                o.get_mut().generation = self.generation;
                Entry::Occupied(OccupiedEntry { entry: o })
            }
            hash_map::Entry::Vacant(v) => Entry::Vacant(VacantEntry {
                entry: v,
                generation,
            }),
        }
    }

    /// Borrow as a [`View`].
    pub fn view(&self) -> View<'_, K, V, P> {
        View { map: self }
    }

    //-----------------------//
    // Internal Test Helpers //
    //-----------------------//

    #[cfg(test)]
    fn generation_of(&self, k: K) -> u32 {
        self.map.get(&k).unwrap().generation
    }
}

#[derive(Debug)]
struct Generation<T> {
    generation: u32,
    value: T,
}

impl<T> Generation<T> {
    fn new(value: T, generation: u32) -> Self {
        Self { generation, value }
    }

    fn value(&self) -> &T {
        &self.value
    }

    fn value_mut(&mut self) -> &mut T {
        &mut self.value
    }

    fn into_inner(self) -> T {
        self.value
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

/// An entry in the mutable map layer that already has a value.
pub struct OccupiedEntry<'a, K, V> {
    entry: hash_map::OccupiedEntry<'a, K, Generation<V>>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Returns a reference to the value.
    pub fn get(&self) -> &V {
        self.entry.get().value()
    }

    /// Returns a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        self.entry.get_mut().value_mut()
    }

    /// Converts into a mutable reference to the value with the entry's lifetime.
    pub fn into_mut(self) -> &'a mut V {
        self.entry.into_mut().value_mut()
    }
}

/// An entry in the mutable map layer with no existing value.
pub struct VacantEntry<'a, K, V> {
    entry: hash_map::VacantEntry<'a, K, Generation<V>>,
    generation: u32,
}

impl<K, V> VacantEntry<'_, K, V> {
    /// Insert a value, stamped with the current generation.
    pub fn insert(self, value: V)
    where
        K: Hash + Eq,
    {
        self.entry.insert(Generation::new(value, self.generation));
    }

    /// Returns a reference to this entry's key.
    pub fn key(&self) -> &K {
        self.entry.key()
    }
}

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
    type View<'a>
        = View<'a, A::Id, V, P>
    where
        Self: 'a;

    fn fill<'a, Itr>(
        &'a mut self,
        map: &'a mut Map<A::Id, V, P>,
        itr: Itr,
    ) -> impl SendFuture<Result<Self::View<'a>, Self::Error>>
    where
        Itr: ExactSizeIterator<Item = A::Id> + Clone + Send + Sync,
        Self: 'a,
    {
        map.fill(self, itr)
    }
}

/////////////////
// Projections //
/////////////////

/// Defines how stored values `V` are projected to element types for lookups.
///
/// A projection decouples the owned storage type (`V`) from the borrowed view type
/// returned by [`View::get`](super::View::get). For example, `Ref<[T]>` projects
/// `Box<[T]>` to `&[T]`, while `Reborrowed<V>` uses `V`'s own [`Reborrow`] impl.
pub trait Projection: Send + Sync + 'static {
    type Element<'a>: for<'b> Reborrow<'b, Target = Self::ElementRef<'b>> + Send;
    type ElementRef<'a>;
}

/// Projection that borrows stored values as `&T` slices.
///
/// Use this when `V = Box<[T]>` and the view should yield `&[T]`.
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

/// Default projection that uses the value's [`Reborrow`] impl.
///
/// This is the default `P` parameter for [`Map`]. It delegates to `V::reborrow()`,
/// wrapping the value in [`AsReborrowed`] so the view's element type matches
/// the accessor's `ElementRef`.
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

/// Wrapper that implements [`Reborrow`] by delegating to the inner value.
///
/// Used by the [`Reborrowed`] projection to provide a [`Reborrow`]-compatible element
/// type from a `&V` reference.
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

/// Convert a stored value `V` to the projection's element type for read access.
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
        self
    }
}

impl<T> Project<Ref<[T]>> for Vec<T>
where
    T: Send + Sync + 'static,
{
    fn project(&self) -> &[T] {
        self
    }
}

/////////////
// Overlay //
/////////////

/// An immutable, zero-copy layer to provide to [`Builder`] and [`Map`].
///
/// Elements in an overlay take precedence when used in [`Map`].
///
/// This struct is cheap to clone.
///
/// ```
/// use std::sync::Arc;
/// use diskann_utils::views::Matrix;
///
/// use diskann::graph::workingset::map::{Overlay, Ref};
///
/// let data = Matrix::<u32>::column_vector(Box::new([10, 20, 30]));
/// let ids = [5, 3, 2];
///
/// // The `Ref<[u32]>` projection yields slices, which is compatible with a `Matrix`.
/// let overlay = Overlay::<u32, Ref<[u32]>>::from_batch(Arc::new(data), ids);
///
/// assert_eq!(overlay.get(&5).unwrap(), &[10]);
/// assert_eq!(overlay.get(&3).unwrap(), &[20]);
/// assert_eq!(overlay.get(&2).unwrap(), &[30]);
///
/// assert!(overlay.get(&1).is_none());
/// assert!(overlay.get(&4).is_none());
/// ```
///
/// See [`Builder`] for the [`Seed`](crate::graph::glue::MultiInsertStrategy::Seed) that
/// should be used for [`Map`].
///
/// # Construction
///
/// Users will typically use [`Self::from_batch`] for multi-insert scenarios. Additional
/// construction using the [`MapLike`] trait are provided for more fine-grained control.
#[derive(Debug)]
pub struct Overlay<K, P: Projection> {
    overlay: Arc<dyn MapLike<K, P>>,
}

impl<K, P: Projection> Clone for Overlay<K, P> {
    fn clone(&self) -> Self {
        Self {
            overlay: self.overlay.clone(),
        }
    }
}

impl<K, P> Overlay<K, P>
where
    P: Projection,
{
    /// Wrap an existing [`MapLike`] implementor as an overlay.
    pub fn new(overlay: Arc<dyn MapLike<K, P>>) -> Self {
        Self { overlay }
    }

    /// Look up a key in the overlay.
    pub fn get(&self, key: &K) -> Option<P::Element<'_>> {
        self.overlay.get(key)
    }

    /// Returns `true` if the overlay contains the key.
    pub fn contains_key(&self, key: &K) -> bool {
        self.overlay.contains_key(key)
    }
}

impl<K, P> Overlay<K, P>
where
    K: Hash + Eq + Send + Sync + 'static,
    P: Projection,
{
    /// Create a zero-copy seed backed directly by a [`glue::Batch`].
    ///
    /// Lookups go through the batch via an index map — no element data is copied.
    /// Requires that batch elements convert to `P::Element` via `Into`.
    pub fn from_batch<B, I>(batch: Arc<B>, ids: I) -> Self
    where
        I: IntoIterator<Item = K, IntoIter: ExactSizeIterator>,
        B: for<'a> glue::Batch<Element<'a>: Into<P::Element<'a>>> + Debug,
        K: Debug,
    {
        let indices: hashbrown::HashMap<K, usize> =
            ids.into_iter().enumerate().map(|(i, id)| (id, i)).collect();
        Self {
            overlay: Arc::new(BatchOverlay { batch, indices }),
        }
    }
}

/// Object-safe seed overlay trait, parameterized by [`Projection`].
///
/// The seed stores pre-populated elements (e.g. from a multi-insert batch) and
/// returns them as `P::Element<'_>` — the same type that [`View`] produces
/// from the fill layer via [`Project`]. This decouples the seed's storage type from
/// the fill layer's value type `V`.
pub trait MapLike<K, P: Projection>: Debug + Send + Sync {
    fn get(&self, key: &K) -> Option<P::Element<'_>>;
    fn contains_key(&self, key: &K) -> bool;
}

impl<K, V, P> MapLike<K, P> for hashbrown::HashMap<K, V>
where
    K: Hash + Eq + Debug + Send + Sync,
    V: Project<P> + Debug + Send + Sync,
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
#[derive(Debug)]
struct BatchOverlay<K, B> {
    batch: Arc<B>,
    indices: hashbrown::HashMap<K, usize>,
}

impl<K, B, P> MapLike<K, P> for BatchOverlay<K, B>
where
    K: Hash + Eq + Debug + Send + Sync,
    B: for<'a> glue::Batch<Element<'a>: Into<P::Element<'a>>> + Debug,
    P: Projection,
{
    fn get(&self, key: &K) -> Option<P::Element<'_>> {
        self.indices.get(key).map(|&idx| self.batch.get(idx).into())
    }

    fn contains_key(&self, key: &K) -> bool {
        self.indices.contains_key(key)
    }
}

/////////////
// Builder //
/////////////

/// Builder for [`Map`]-backed working sets.
///
/// Doubles as the `Seed` type for [`AsWorkingSet<Map>`]: strategies return a `Builder`
/// from [`MultiInsertStrategy::finish`](crate::graph::glue::MultiInsertStrategy::finish),
/// and the index calls [`as_working_set`](AsWorkingSet::as_working_set) to materialize it.
///
/// # Examples
///
/// ```ignore
/// // Unseeded, clear every fill:
/// let map: Map<u32, Box<[f32]>> = Builder::new(Capacity::None).build(128);
///
/// // Seeded, use caller's default capacity:
/// let map: Map<u32, Box<[f32]>, Ref<[f32]>> = Builder::new(Capacity::Default)
///     .with_overlay(overlay)
///     .build(128);
/// ```
#[derive(Debug)]
pub struct Builder<K, P: Projection> {
    overlay: Option<Overlay<K, P>>,
    capacity: Capacity,
}

impl<K, P: Projection> Builder<K, P> {
    /// Create a builder with the given eviction policy.
    pub fn new(capacity: Capacity) -> Self {
        Self {
            overlay: None,
            capacity,
        }
    }

    /// Attach a zero-copy seed overlay to the resulting [`Map`].
    pub fn with_overlay(mut self, overlay: Overlay<K, P>) -> Self {
        self.overlay = Some(overlay);
        self
    }

    /// Materialize the [`Map`].
    ///
    /// `default_capacity` is used when `self.capacity` is [`Capacity::Default`].
    pub fn build<V>(self, default_capacity: usize) -> Map<K, V, P> {
        let capacity = self.capacity.resolve(default_capacity);
        Map {
            map: hashbrown::HashMap::with_capacity(capacity.unwrap_or(0)),
            overlay: self.overlay,
            capacity,
            generation: 0,
        }
    }
}

impl<K, P: Projection> Clone for Builder<K, P> {
    fn clone(&self) -> Self {
        Self {
            overlay: self.overlay.clone(),
            capacity: self.capacity,
        }
    }
}

impl<K, V, P> AsWorkingSet<Map<K, V, P>> for Builder<K, P>
where
    P: Projection,
{
    fn as_working_set(&self, capacity: usize) -> Map<K, V, P> {
        self.clone().build(capacity)
    }
}

/// Controls how a [`Builder`] resolves the capacity of the constructed [`Map`].
///
/// See [working set reuse](self#working-set-reuse) for a detailed description.
#[derive(Debug, Clone, Copy)]
pub enum Capacity {
    /// No capacity — clear the map on every fill. Equivalent to `Some(0)`.
    None,
    /// Use the default capacity provided by the caller (typically `max_occlusion_size`).
    Default,
    /// Use a specific capacity.
    Some(usize),
    /// Unbounded — never evict entries.
    Unbounded,
}

impl Capacity {
    /// Resolve to a concrete capacity. Returns [`Option::None`] for
    /// [`Unbounded`](Capacity::Unbounded).
    pub fn resolve(self, default: usize) -> Option<usize> {
        match self {
            Self::None => Some(0),
            Self::Default => Some(default),
            Self::Some(actual) => Some(actual),
            Self::Unbounded => Option::None,
        }
    }
}

//////////
// View //
//////////

/// The [`super::View`] implementation for [`Map`], constructed with [`Map::view`].
///
/// If an [`Overlay`] is used with the associated [`Map`], elements in the overlay will be
/// given precedence.
///
/// If [working set reuse](self#working-set-reuse) is enabled, then all elements within
/// the [`Map`] are visible, not just those that belong to the most recent [`Fill`].
#[derive(Debug)]
pub struct View<'a, K, V, P: Projection = Reborrowed<V>> {
    map: &'a Map<K, V, P>,
}

impl<K, V, P> super::View<K> for View<'_, K, V, P>
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
            .overlay
            .as_ref()
            .and_then(|s| s.get(&id))
            .or_else(|| self.map.map.get(&id).map(|v| v.value().project()))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use diskann_utils::views::Matrix;

    use crate::graph::{
        test::provider::Accessor as TestAccessor, workingset::View as WorkingSetView,
    };

    /// Convenience alias matching the test provider's working set type.
    type TestMap = Map<u32, Box<[f32]>, Ref<[f32]>>;

    /// Convenience alias matching the test provider's working set type.
    type TestMapProjected = Map<u32, Box<[f32]>, TestProjection>;

    /// Build an unbounded test map (no eviction). Convenience for unit tests.
    fn unbounded_map() -> TestMap {
        Builder::new(Capacity::Unbounded).build(0)
    }

    /// Build an unbounded projected test map. Convenience for unit tests.
    fn unbounded_map_projected() -> TestMapProjected {
        Builder::new(Capacity::Unbounded).build(0)
    }

    /// Build a seeded test map with the given overlay and capacity.
    fn seeded_map(overlay: Overlay<u32, Ref<[f32]>>, capacity: Capacity) -> TestMap {
        Builder::new(capacity).with_overlay(overlay).build(16)
    }

    /// Build a seeded projected test map with the given overlay and capacity.
    fn seeded_map_projected(
        overlay: Overlay<u32, TestProjection>,
        capacity: Capacity,
    ) -> TestMapProjected {
        Builder::new(capacity).with_overlay(overlay).build(16)
    }

    /// The `TestProjection` ensures that we can apply an arbitrary transformation on the
    #[derive(Debug, Clone, Copy)]
    struct TestProjection;

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum Source {
        Project,
        Into,
    }

    #[derive(Debug, PartialEq)]
    struct Wrapped<'a> {
        data: &'a [f32],
        source: Source,
    }

    impl<'a> Wrapped<'a> {
        fn new(data: &'a [f32], source: Source) -> Self {
            Self { data, source }
        }
    }

    impl<'a> Reborrow<'a> for Wrapped<'_> {
        type Target = &'a [f32];
        fn reborrow(&'a self) -> Self::Target {
            self.data
        }
    }

    impl Projection for TestProjection {
        type Element<'a> = Wrapped<'a>;
        type ElementRef<'a> = &'a [f32];
    }

    impl Project<TestProjection> for Box<[f32]> {
        fn project(&self) -> Wrapped<'_> {
            Wrapped {
                data: self,
                source: Source::Project,
            }
        }
    }

    impl<'a> From<&'a [f32]> for Wrapped<'a> {
        fn from(data: &'a [f32]) -> Self {
            Self {
                data,
                source: Source::Into,
            }
        }
    }

    /// Build a 3-row × 2-col matrix for testing with the following layout:
    /// ```text
    /// 1.0  2.0
    /// 3.0  4.0
    /// 5.0  6.0
    /// ```
    fn test_matrix() -> Matrix<f32> {
        Matrix::try_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice(), 3, 2).unwrap()
    }

    type TestOverlay = Overlay<u32, Ref<[f32]>>;

    fn test_overlay() -> (Arc<Matrix<f32>>, TestOverlay) {
        let batch = Arc::new(test_matrix());
        let ids = [10u32, 20, 30];
        let overlay = Overlay::from_batch(batch.clone(), ids);
        (batch, overlay)
    }

    fn test_overlay_projected() -> (Arc<Matrix<f32>>, Overlay<u32, TestProjection>) {
        let batch = Arc::new(test_matrix());
        let ids = [10u32, 20, 30];
        let overlay = Overlay::from_batch(batch.clone(), ids);
        (batch, overlay)
    }

    //----------//
    // Capacity //
    //----------//

    #[test]
    fn test_capacity() {
        let c = Capacity::None;
        assert_eq!(c.resolve(0), Some(0));
        assert_eq!(c.resolve(50), Some(0));
        assert_eq!(c.resolve(100), Some(0));

        let c = Capacity::Default;
        assert_eq!(c.resolve(0), Some(0));
        assert_eq!(c.resolve(50), Some(50));
        assert_eq!(c.resolve(100), Some(100));

        for i in [5, 10, 15] {
            let c = Capacity::Some(i);
            assert_eq!(c.resolve(0), Some(i));
            assert_eq!(c.resolve(50), Some(i));
            assert_eq!(c.resolve(100), Some(i));
        }

        let c = Capacity::Unbounded;
        assert_eq!(c.resolve(0), None);
        assert_eq!(c.resolve(50), None);
        assert_eq!(c.resolve(100), None);
    }

    //-----------------------//
    // Map basics (unseeded) //
    //-----------------------//

    #[test]
    fn insert_and_get() {
        let mut map = unbounded_map();
        assert_eq!(map.generation, 0);

        // Empty Map
        assert!(map.get(&1).is_none());
        assert!(map.get(&2).is_none());
        assert!(map.get(&3).is_none());

        assert!(!map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(!map.contains_key(&3));

        // Insert key 1
        assert!(map.insert(1, vec![1.0, 2.0].into_boxed_slice()).is_none());
        assert_eq!(&**map.get(&1).unwrap(), &[1.0, 2.0]);

        assert!(map.get(&2).is_none());
        assert!(map.get(&3).is_none());

        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(!map.contains_key(&3));

        // -- Check generations
        assert_eq!(map.generation_of(1), 0);

        // Here - we reach into a private method to bump the generation and verify that
        // `insert` respects the generation.
        map.new_generation();
        assert_eq!(map.generation, 1);

        // Insert Key 3
        assert!(map.insert(3, vec![2.0, 3.0].into_boxed_slice()).is_none());
        assert_eq!(&**map.get(&1).unwrap(), &[1.0, 2.0]);
        assert!(map.get(&2).is_none());
        assert_eq!(&**map.get(&3).unwrap(), &[2.0, 3.0]);

        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(map.contains_key(&3));

        // -- Check Gneerations
        assert_eq!(map.generation_of(1), 0);
        assert_eq!(map.generation_of(3), 1);
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut map = unbounded_map();
        map.insert(1, Box::new([1.0]));
        assert_eq!(map.generation_of(1), 0);
        map.new_generation();

        let old = map.insert(1, Box::new([2.0]));
        assert_eq!(&*old.unwrap(), &[1.0]);
        assert_eq!(&**map.get(&1).unwrap(), &[2.0]);
        assert_eq!(map.generation_of(1), 1);
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut map = unbounded_map();
        map.insert(1, Box::new([0.0, 0.0]));
        assert_eq!(map.generation_of(1), 0);

        map.new_generation();
        map.get_mut(&1).unwrap()[0] = 42.0;
        assert_eq!(&**map.get(&1).unwrap(), &[42.0, 0.0]);
        assert_eq!(
            map.generation_of(1),
            1,
            "`get_mut` implicitly bumps the generation"
        );
    }

    #[test]
    fn clear_empties_map() {
        let mut map = unbounded_map();
        map.insert(1, Box::new([1.0]));
        map.insert(2, Box::new([2.0]));
        map.clear();
        assert!(!map.contains_key(&1));
        assert!(!map.contains_key(&2));
    }

    //----------------------//
    // Entry API (unseeded) //
    //----------------------//

    #[test]
    fn entry_vacant_then_occupied() {
        let mut map = unbounded_map();
        map.new_generation();

        // Initially vacant.
        match map.entry(1) {
            Entry::Vacant(v) => {
                assert_eq!(*v.key(), 1);
                v.insert(Box::new([1.0]));
            }
            _ => panic!("expected Vacant"),
        }

        // Now occupied.
        match map.entry(1) {
            Entry::Occupied(o) => {
                assert_eq!(&**o.get(), &[1.0]);
            }
            _ => panic!("expected Occupied"),
        }

        assert_eq!(
            map.generation_of(1),
            1,
            "vacant inserts match current generation"
        );
    }

    #[test]
    fn occupied_entry_mut() {
        let mut map = unbounded_map();

        // Go through `into_mut()`.
        map.insert(1, Box::new([0.0]));
        assert_eq!(map.generation_of(1), 0);

        map.new_generation();
        match map.entry(1) {
            Entry::Occupied(o) => {
                let val = o.into_mut();
                val[0] = 99.0;
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&1).unwrap(), &[99.0]);
        assert_eq!(
            map.generation_of(1),
            1,
            "OccupiedEntry implicitly increments generation"
        );

        map.new_generation();
        match map.entry(1) {
            Entry::Occupied(mut o) => {
                let val = o.get_mut();
                val[0] = 42.0;
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&1).unwrap(), &[42.0]);
        assert_eq!(
            map.generation_of(1),
            2,
            "OccupiedEntry implicitly increments generation"
        );
    }

    //-----------------//
    // View (unseeded) //
    //-----------------//

    #[test]
    fn view_returns_projected_element() {
        // Standard Slice Decay
        {
            let mut map = unbounded_map();
            map.insert(1, Box::new([1.0, 2.0]));

            // Views don't care about generations.
            map.new_generation();
            let view = map.view();

            let element = view.get(1).unwrap();
            assert_eq!(element, &[1.0, 2.0]);
            assert!(view.get(999).is_none());
        }

        // Projection - note that the types inserted are still `Box<[f32]>`.
        {
            let mut map = unbounded_map_projected();
            map.insert(1, Box::new([1.0, 2.0]));

            // Views don't care about generations.
            map.new_generation();
            let view = map.view();

            let element = view.get(1).unwrap();
            assert_eq!(element, Wrapped::new(&[1.0, 2.0], Source::Project));
            assert!(view.get(999).is_none());
        }
    }

    //------------------------//
    // Overlay / BatchOverlay //
    //------------------------//

    #[test]
    fn overlay_from_batch_lookups() {
        // Standard Ref
        {
            let (_batch, overlay) = test_overlay();

            // Each ID maps to its row in the matrix.
            assert_eq!(overlay.get(&10).unwrap(), &[1.0, 2.0]);
            assert_eq!(overlay.get(&20).unwrap(), &[3.0, 4.0]);
            assert_eq!(overlay.get(&30).unwrap(), &[5.0, 6.0]);
            assert!(overlay.contains_key(&10));
            assert!(overlay.contains_key(&20));

            // Missing key.
            assert!(overlay.get(&99).is_none());
            assert!(!overlay.contains_key(&99));
        }

        // Projected
        {
            let (_batch, overlay) = test_overlay_projected();
            let source = Source::Into;

            // Each ID maps to its row in the matrix.
            assert_eq!(overlay.get(&10).unwrap(), Wrapped::new(&[1.0, 2.0], source));
            assert_eq!(overlay.get(&20).unwrap(), Wrapped::new(&[3.0, 4.0], source));
            assert_eq!(overlay.get(&30).unwrap(), Wrapped::new(&[5.0, 6.0], source));
            assert!(overlay.contains_key(&10));
            assert!(overlay.contains_key(&20));

            // Missing key.
            assert!(overlay.get(&99).is_none());
            assert!(!overlay.contains_key(&99));
        }
    }

    #[test]
    fn overlay_is_cheap_to_clone() {
        let (_batch, overlay) = test_overlay();
        let cloned = overlay.clone();
        // Both see the same data.
        assert_eq!(cloned.get(&10).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn overlay_from_batch_zero_copy() {
        // Verify that the batch data is NOT copied — the overlay holds an Arc to the same
        // allocation. We can confirm by checking that the returned slices point into the
        // original matrix's memory.
        let batch = Arc::new(test_matrix());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(batch.clone(), [10u32, 20, 30]);

        let from_overlay: &[f32] = overlay.get(&10).unwrap();
        let from_batch: &[f32] = batch.row(0);
        assert!(std::ptr::eq(from_overlay, from_batch));
    }

    //-------------------------------------------//
    // Map with seed overlay (three-state entry) //
    //-------------------------------------------//

    #[test]
    fn seeded_map_entry_seeded() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map = seeded_map_projected(overlay, Capacity::Default);
        assert_eq!(map.generation, 0);

        match map.entry(99) {
            Entry::Vacant(v) => {
                assert_eq!(*v.key(), 99);
                // Don't fill the vacant entry.
            }
            _ => panic!("expected Vacant for key not in overlay or map"),
        }

        match map.entry(10) {
            Entry::Seeded(element) => {
                assert_eq!(element, Wrapped::new(&[1.0, 2.0], Source::Into));
            }
            _ => panic!("expected Seeded for key present in overlay"),
        }

        match map.entry(99) {
            Entry::Vacant(v) => {
                v.insert(Box::new([5.0]));
            }
            _ => panic!("expected Vacant"),
        }
        assert_eq!(&**map.get(&99).unwrap(), &[5.0]);
        assert_eq!(map.generation_of(99), 0);

        map.new_generation();
        match map.entry(99) {
            Entry::Occupied(v) => {
                *v.into_mut() = Box::new([10.0]);
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&99).unwrap(), &[10.0]);
        assert_eq!(map.generation_of(99), 1);
    }

    #[test]
    fn seeded_map_insert_skips_seeded_key() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map = seeded_map_projected(overlay, Capacity::Default);

        // Inserting a key that's in the seed is a no-op (returns None).
        assert!(
            map.insert(10, Box::new([99.0])).is_none(),
            "The key 10 should not exist"
        );

        // The seed value is unchanged.
        let view = map.view();
        assert_eq!(
            view.get(10).unwrap(),
            Wrapped::new(&[1.0, 2.0], Source::Into)
        );

        assert!(
            map.insert(10, Box::new([99.0])).is_none(),
            "The key 10 should not still exist"
        );
    }

    #[test]
    fn seeded_map_insert_works_for_non_seeded() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map = seeded_map_projected(overlay, Capacity::Default);

        map.new_generation();
        map.insert(99, Box::new([7.0, 8.0]));
        assert_eq!(map.generation_of(99), 1);

        assert!(map.contains_key(&10));
        assert!(map.contains_key(&20));
        assert!(map.contains_key(&30));
        assert!(map.contains_key(&99));
        assert!(!map.contains_key(&1));

        assert!(map.get(&10).is_none());
        assert!(map.get(&20).is_none());
        assert!(map.get(&30).is_none());
        assert_eq!(
            map.get(&99).unwrap().as_ref(),
            &[7.0, 8.0],
            "get does not check the overlay"
        );
        assert!(!map.contains_key(&1));
    }

    //-----------------------------//
    // View with seed (seed-first) //
    //-----------------------------//

    #[test]
    fn view_seed_first_priority() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map = seeded_map_projected(overlay, Capacity::Default);

        // Insert a different value for key 10 into the fill layer — but seed takes priority
        // in the view. (In practice, insert() blocks this, but we test the view's dispatch.)
        // Since insert() is a no-op for seeded keys, insert into a non-seeded key instead.
        map.insert(99, Box::new([7.0, 8.0]));

        let view = map.view();
        // Seed key — served from overlay.
        assert_eq!(
            view.get(10).unwrap(),
            Wrapped::new(&[1.0, 2.0], Source::Into)
        );
        // Map key — served from fill layer.
        assert_eq!(
            view.get(99).unwrap(),
            Wrapped::new(&[7.0, 8.0], Source::Project)
        );
        // Missing — None.
        assert!(view.get(1).is_none());
    }

    #[test]
    fn view_after_clear_still_has_seed() {
        let (_batch, overlay) = test_overlay();
        let mut map = seeded_map(overlay, Capacity::Default);
        map.insert(99, Box::new([7.0]));

        map.clear();

        let view = map.view();
        // Seed survives clear.
        assert_eq!(view.get(10).unwrap(), &[1.0, 2.0]);
        // Fill layer is gone.
        assert!(view.get(99).is_none());
    }

    //-------------//
    // Generations //
    //-------------//

    #[test]
    fn test_promote() {
        let mut map: TestMap = Builder::new(Capacity::Default).build(5);
        map.insert(1, Box::new([1.0]));
        map.insert(2, Box::new([2.0]));
        map.insert(3, Box::new([3.0]));

        // No matches - nothing to promote
        map.new_generation();
        assert_eq!(map.promote([5, 6, 7].into_iter()), 0);
        for i in [1, 2, 3] {
            assert_eq!(
                map.generation_of(i),
                0,
                "generation should remain unchanged"
            );
        }

        // All matches - everything should promote.
        //
        // Include some extra keys not present in the map as well.
        map.new_generation();
        assert_eq!(map.promote([5, 2, 6, 3, 7, 1].into_iter()), 3);
        for i in [1, 2, 3] {
            assert_eq!(map.generation_of(i), 2);
        }

        // Partial matches - some update, others don't.
        map.new_generation();
        assert_eq!(map.promote([1, 4, 2, 10].into_iter()), 2);
        assert_eq!(map.generation_of(1), 3);
        assert_eq!(map.generation_of(2), 3);
        assert_eq!(map.generation_of(3), 2, "this entry was not promoted");

        // Cross generations all get promoted to the same.
        map.new_generation();
        assert_eq!(map.promote([1, 3].into_iter()), 2);
        assert_eq!(map.generation_of(1), 4);
        assert_eq!(map.generation_of(2), 3, "this entry was not promoted");
        assert_eq!(map.generation_of(3), 4);
    }

    #[test]
    fn test_evict() {
        let mut map: TestMap = Builder::new(Capacity::Default).build(5);

        fn reset(map: &mut TestMap) {
            map.clear();
            map.insert(1, Box::new([1.0]));
            map.insert(2, Box::new([2.0]));
            map.insert(3, Box::new([3.0]));
        }

        reset(&mut map);

        // All inserted items belong to the current generation - eviction should do nothing.
        for count in 0..10 {
            assert_eq!(
                map.evict(count),
                0,
                "all items belong to the current generation"
            );
            assert_eq!(&**map.get(&1).unwrap(), [1.0]);
            assert_eq!(&**map.get(&2).unwrap(), [2.0]);
            assert_eq!(&**map.get(&3).unwrap(), [3.0]);
        }

        // If we bump the generation but request no evictions - nothing should happen.
        map.new_generation(); // generation 1
        assert_eq!(map.evict(0), 0);
        assert_eq!(map.map.len(), 3);

        // Now - we bump the generation of entry 2 and request one entry to be evicted.
        let _ = map.get_mut(&2).unwrap();
        assert_eq!(map.generation_of(2), 1);
        assert_eq!(map.evict(1), 1);
        assert_eq!(map.map.len(), 2);
        assert!(map.contains_key(&2));

        // Bump the generation of entry 2 and request two entries to be evicted.
        reset(&mut map);
        map.new_generation(); // generation 2
        let _ = map.get_mut(&2).unwrap();
        assert_eq!(map.evict(2), 2);
        assert_eq!(map.map.len(), 1);
        assert!(map.contains_key(&2));

        // Request that more gets evicted and that the count is truncated by entries not
        // belonging to the current generation.
        reset(&mut map);
        map.new_generation(); // generation 3
        let _ = map.get_mut(&1).unwrap();
        assert_eq!(map.evict(10), 2);
        assert_eq!(map.map.len(), 1);
        assert!(map.contains_key(&1));

        // Check that *all* entries can be evicted.
        reset(&mut map);
        map.new_generation(); // generation 4
        assert_eq!(map.evict(10), 3);
        assert!(map.map.is_empty());
    }

    #[test]
    fn test_prepare_no_capacity() {
        let mut map: TestMap = Builder::new(Capacity::None).build(100);

        map.insert(1, Box::new([1.0]));
        map.insert(2, Box::new([2.0]));
        map.insert(3, Box::new([3.0]));

        // Even if we pass an empty iterator - this is sufficient to clear the entries.
        map.prepare([].into_iter());
        assert!(map.map.is_empty());
    }

    #[test]
    fn test_prepare_unbounded() {
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        map.insert(1, Box::new([1.0]));
        map.insert(2, Box::new([2.0]));
        map.insert(3, Box::new([3.0]));

        // Even if we pass an empty iterator - this is sufficient to clear the entries.
        map.prepare(1..100);
        assert_eq!(map.map.len(), 3, "unbounded maps skip preparation");
    }

    #[test]
    fn test_prepare_bounded() {
        // Capacity of 4: the map can hold up to 4 entries across generations.
        let mut map: TestMap = Builder::new(Capacity::Default).build(4);

        fn reset(map: &mut TestMap) {
            map.clear();
            map.insert(1, Box::new([1.0]));
            map.insert(2, Box::new([2.0]));
            map.insert(3, Box::new([3.0]));
            map.insert(4, Box::new([4.0]));
            assert_eq!(map.map.len(), 4);
        }

        // -- Case 1: insert 4 entries (at capacity)
        //
        // In this test - we have a mix of IDs in the map and out of the map.
        reset(&mut map);

        // IDs 1 and 2 get promoted, 3 and 4 get evicted.
        map.prepare([1, 2, 5, 6].into_iter());
        assert_eq!(map.generation, 1, "prepare should increment generation");
        assert_eq!(map.map.len(), 2, "evicted 2 old-gen entries to make room");

        // The promoted entries survive.
        assert_eq!(map.generation_of(1), 1);
        assert_eq!(map.generation_of(2), 1);

        // The non-promoted entries were evicted.
        assert!(map.get(&3).is_none(), "old-gen entry 3 should be evicted");
        assert!(map.get(&4).is_none(), "old-gen entry 4 should be evicted");

        // -- Case 2: request only IDs already present
        reset(&mut map);
        map.prepare([1, 2].into_iter());
        assert_eq!(map.generation, 2);
        assert_eq!(
            map.map.len(),
            4,
            "since we don't exceed capacity - no eviction was necessary"
        );
        assert_eq!(map.generation_of(1), 2);
        assert_eq!(map.generation_of(2), 2);
        assert_eq!(map.generation_of(3), 1);
        assert_eq!(map.generation_of(4), 1);

        // -- Case 3: request more IDs than capacity
        //
        // Entries 1 and 2 get promoted, everything else gets cleaned up.
        reset(&mut map);
        map.prepare([1, 2, 10, 11, 12, 13].into_iter());
        assert_eq!(map.generation, 3);
        assert_eq!(map.map.len(), 2, "keep items present in the iterator");

        // -- Case 4: empty iterator.
        //
        // No items should be evicted if we're already at capacity.
        reset(&mut map);
        map.prepare([].into_iter());
        assert_eq!(map.generation, 4);
        assert_eq!(map.map.len(), 4, "at capacity — old entries survive");
        assert_eq!(map.generation_of(1), 3);
        assert_eq!(map.generation_of(2), 3);
        assert_eq!(map.generation_of(3), 3);
        assert_eq!(map.generation_of(4), 3);

        // -- Case 5: empty iterator, recovery
        //
        // In this situation, the hash map is already over capacity (maybe a previous `fill`
        // added too many candidates). Here - we verify that it gets shrunk down to capacity
        // even if an empty iterator is provided.
        reset(&mut map);
        map.insert(5, Box::new([5.0]));
        map.insert(6, Box::new([6.0]));
        map.insert(7, Box::new([7.0]));
        map.prepare([].into_iter());
        assert_eq!(map.generation, 5);
        assert_eq!(map.map.len(), 4, "evict down to capacity");

        // -- Case 5: Disjoint, final size exceeds capacity but itertor does not.
        reset(&mut map);
        map.prepare([20, 21, 22].into_iter());
        assert_eq!(map.generation, 6);
        assert_eq!(map.map.len(), 1, "evicted 1 old-gen entry to make room");

        // -- Case 6: New iterator completely disjoint and above capacity.
        reset(&mut map);
        map.prepare([20, 21, 22, 23, 24].into_iter());
        assert_eq!(map.generation, 7);
        assert!(map.map.is_empty());
    }

    //---------------------//
    // Projection (Ref<T>) //
    //---------------------//

    #[test]
    fn project_ref_box_slice() {
        let value: Box<[f32]> = Box::new([1.0, 2.0, 3.0]);
        let projected: &[f32] = <_ as Project<Ref<[f32]>>>::project(&value);
        assert_eq!(projected, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn project_ref_vec() {
        let value = vec![4.0f32, 5.0];
        let projected: &[f32] = <_ as Project<Ref<[f32]>>>::project(&value);
        assert_eq!(projected, &[4.0, 5.0]);
    }

    //----------------------------//
    // Projection (Reborrowed<T>) //
    //----------------------------//

    #[test]
    fn project_reborrowed_box_slice() {
        let value: Box<[f32]> = Box::new([1.0, 2.0]);
        let projected = <_ as Project<Reborrowed<Box<[f32]>>>>::project(&value);
        let reborrowed: &[f32] = projected.reborrow();
        assert_eq!(reborrowed, &[1.0, 2.0]);
    }

    //---------------------//
    // MapLike for HashMap //
    //---------------------//

    #[test]
    fn hashmap_as_maplike() {
        let mut hm = hashbrown::HashMap::<u32, Box<[f32]>>::new();
        hm.insert(1, Box::new([10.0, 20.0]));
        hm.insert(2, Box::new([30.0]));

        {
            let map_like: &dyn MapLike<u32, Ref<[f32]>> = &hm;
            assert_eq!(map_like.get(&1).unwrap(), &[10.0, 20.0]);
            assert_eq!(map_like.get(&2).unwrap(), &[30.0]);
            assert!(map_like.get(&3).is_none());
            assert!(map_like.contains_key(&1));
            assert!(!map_like.contains_key(&3));

            let overlay = Overlay::<u32, Ref<[f32]>>::new(Arc::new(hm.clone()));
            assert_eq!(overlay.get(&1).unwrap(), &[10.0, 20.0]);
            assert_eq!(overlay.get(&2).unwrap(), &[30.0]);
            assert!(overlay.get(&3).is_none());
            assert!(overlay.contains_key(&1));
            assert!(!overlay.contains_key(&3));
        }

        {
            let map_like: &dyn MapLike<u32, TestProjection> = &hm;
            let source = Source::Project;
            assert_eq!(
                map_like.get(&1).unwrap(),
                Wrapped::new(&[10.0, 20.0], source)
            );
            assert_eq!(map_like.get(&2).unwrap(), Wrapped::new(&[30.0], source));
            assert!(map_like.get(&3).is_none());
            assert!(map_like.contains_key(&1));
            assert!(!map_like.contains_key(&3));

            let overlay = Overlay::<u32, TestProjection>::new(Arc::new(hm));
            assert_eq!(
                overlay.get(&1).unwrap(),
                Wrapped::new(&[10.0, 20.0], source)
            );
            assert_eq!(overlay.get(&2).unwrap(), Wrapped::new(&[30.0], source));
            assert!(overlay.get(&3).is_none());
            assert!(overlay.contains_key(&1));
            assert!(!overlay.contains_key(&3));
        }
    }

    //---------//
    // Builder //
    //---------//

    #[test]
    fn builder_unseeded_creates_empty_map() {
        let ws: TestMap = Builder::new(Capacity::Default).build(32);
        assert!(!ws.contains_key(&1));
        // No overlay.
        let view = ws.view();
        assert!(view.get(1).is_none());
    }

    #[test]
    fn builder_as_working_set() {
        let ws: TestMap = Builder::new(Capacity::Default).as_working_set(32);
        assert!(!ws.contains_key(&1));
        let view = ws.view();
        assert!(view.get(1).is_none());
    }

    //------------//
    // Edge cases //
    //------------//

    #[test]
    fn overlay_from_batch_empty() {
        let batch = Arc::new(Matrix::try_from(Box::new([]), 0, 2).unwrap());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(batch, std::iter::empty());
        assert!(overlay.get(&0).is_none());
        assert!(!overlay.contains_key(&0));
    }

    #[test]
    fn overlay_from_batch_single_element() {
        let batch = Arc::new(Matrix::try_from(Box::new([1.0, 2.0]), 1, 2).unwrap());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(batch, [42u32]);
        assert_eq!(overlay.get(&42).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn overlay_from_batch_duplicate_ids_last_wins() {
        // When the same ID appears multiple times, the last index wins (HashMap behavior).
        let batch = Arc::new(test_matrix()); // rows: [1,2], [3,4], [5,6]
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(batch, [10u32, 10, 10]);
        // Last occurrence: index 2 → row [5,6]
        assert_eq!(overlay.get(&10).unwrap(), &[5.0, 6.0]);
    }

    //------------------------------//
    // Projection struct Clone/Copy //
    //------------------------------//

    #[test]
    #[expect(clippy::clone_on_copy)]
    fn ref_clone() {
        let r = Ref::<[f32]>(std::marker::PhantomData);
        let _ = r.clone();
    }

    #[test]
    #[expect(clippy::clone_on_copy)]
    fn reborrowed_clone() {
        let r = Reborrowed::<Box<[f32]>>(std::marker::PhantomData);
        let _ = r.clone();
    }

    #[test]
    #[expect(clippy::clone_on_copy, reason = "explicitly testing Clone impl")]
    fn as_reborrowed_clone() {
        let value: Box<[f32]> = Box::new([3.0, 4.0]);
        let ar = AsReborrowed(&*value);
        let _ = ar.clone();
    }

    //----------------------------------//
    // Fill / fill_with (async, Tier 1) //
    //----------------------------------//

    /// Create a grid-backed provider (1-D, size 5).
    ///
    /// IDs 0–4 have vectors `[0.0]` .. `[4.0]`, start point is `u32::MAX`.
    fn fill_provider() -> crate::graph::test::provider::Provider {
        use crate::graph::test::synthetic::Grid;
        crate::graph::test::provider::Provider::grid(Grid::One, 5).unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_happy_path() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        let current = map.generation;
        let view = map
            .fill(&mut accessor, [0u32, 1, 2].into_iter())
            .await
            .unwrap();

        assert_eq!(view.get(0).unwrap(), &[0.0]);
        assert_eq!(view.get(1).unwrap(), &[1.0]);
        assert_eq!(view.get(2).unwrap(), &[2.0]);
        assert!(view.get(99).is_none());

        assert_eq!(
            map.generation,
            current + 1,
            "`fill` should bump the generation"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_clears_previous_entries() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: Map<u32, Box<[f32]>, Ref<[f32]>> = Builder::new(Capacity::None).build(0);

        // First fill with IDs 0 and 1.
        let view = map
            .fill(&mut accessor, [0u32, 1].into_iter())
            .await
            .unwrap();

        assert!(view.get(0).is_some());
        assert!(view.get(1).is_some());
        assert!(view.get(2).is_none());

        // Second fill with only ID 2 — previous entries should be cleared.
        let view = map.fill(&mut accessor, [2u32].into_iter()).await.unwrap();
        assert!(view.get(0).is_none(), "fill should have cleared id 0");
        assert!(view.get(1).is_none(), "fill should have cleared id 1");
        assert_eq!(view.get(2).unwrap(), &[2.0]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_with_preserves_entries() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        // Populate with ID 0.
        let view = map
            .fill_with(&mut accessor, [0u32].into_iter(), |e| e.into())
            .await
            .unwrap();
        assert!(view.get(0).is_some());

        // fill_with with ID 1 — should NOT clear ID 0.
        let view = map
            .fill_with(&mut accessor, [1u32].into_iter(), |e| e.into())
            .await
            .unwrap();
        assert_eq!(
            view.get(0).unwrap(),
            &[0.0],
            "fill_with should preserve id 0"
        );
        assert_eq!(view.get(1).unwrap(), &[1.0]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_skips_transient_errors() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::flaky(&provider, std::collections::HashSet::from([1]));
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        // ID 1 is transient — should be skipped, not propagated.
        let view = map
            .fill(&mut accessor, [0u32, 1, 2].into_iter())
            .await
            .unwrap();
        assert_eq!(view.get(0).unwrap(), &[0.0]);
        assert!(view.get(1).is_none(), "transient ID should be absent");
        assert_eq!(view.get(2).unwrap(), &[2.0]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_propagates_critical_errors() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        // ID 99 doesn't exist — critical InvalidId error.
        let err = map
            .fill(&mut accessor, [0u32, 99].into_iter())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("99"),
            "error should mention the invalid id: {msg}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_with_skips_occupied_entries() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        // Pre-insert a sentinel value for ID 0.
        map.insert(0, Box::new([99.0]));

        // fill_with should skip the occupied entry.
        let view = map
            .fill_with(&mut accessor, [0u32, 1].into_iter(), |e| e.into())
            .await
            .unwrap();

        // ID 0 retains its pre-inserted sentinel.
        assert_eq!(
            view.get(0).unwrap(),
            &[99.0],
            "occupied entry should be preserved"
        );
        assert_eq!(view.get(1).unwrap(), &[1.0]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_with_skips_seeded_entries() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);

        // Seed with a batch containing a different value for ID 0.
        let batch = Arc::new(Matrix::try_from(Box::new([99.0, 88.0]), 2, 1).unwrap());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(batch, [0u32, 1].into_iter());
        let mut map = seeded_map(overlay, Capacity::Unbounded);

        // fill_with requests IDs 0 and 2. ID 0 is seeded → skip, ID 2 is filled.
        let view = map
            .fill_with(&mut accessor, [0u32, 2].into_iter(), |e| e.into())
            .await
            .unwrap();

        // ID 0 comes from the seed (batch row 0 = [99.0]), NOT the accessor.
        assert_eq!(view.get(0).unwrap(), &[99.0]);
        // ID 2 was filled from the accessor.
        assert_eq!(view.get(2).unwrap(), &[2.0]);
        // Verify ID 0 is NOT in the fill layer.
        assert!(
            map.get(&0).is_none(),
            "ID 0 should only be in the seed, not the fill layer"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_empty_iterator() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        let view = map.fill(&mut accessor, std::iter::empty()).await.unwrap();
        assert!(view.get(0).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blanket_fill_trait() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Builder::new(Capacity::Unbounded).build(0);

        // Exercise the blanket Fill<Map> impl.
        let view = <_ as Fill<TestMap>>::fill(&mut accessor, &mut map, [0u32, 1, 2].into_iter())
            .await
            .unwrap();

        assert_eq!(view.get(0).unwrap(), &[0.0]);
        assert_eq!(view.get(1).unwrap(), &[1.0]);
        assert_eq!(view.get(2).unwrap(), &[2.0]);
    }
}
