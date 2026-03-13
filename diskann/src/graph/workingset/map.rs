/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use std::{fmt::Debug, hash::Hash, sync::Arc};

use diskann_utils::{Reborrow, future::SendFuture};
use hashbrown::hash_map;

use crate::{
    error::{RankedError, ToRanked, TransientError},
    graph::glue,
    provider::Accessor,
};

use super::{AsWorkingSet, Fill, Unseeded};

/////////
// Map //
/////////

/// Default working set state backed by a `HashMap` with an optional immutable seed overlay.
///
/// For single insert, `seed` is `None`. For multi-insert, the seed is pre-populated by
/// [`Overlay::from_batch`] with elements from the insertion batch, avoiding redundant
/// fetches during [`Fill`].
///
/// Seed keys and map keys are kept disjoint: insertions for keys already present in
/// the seed are silently skipped.
#[derive(Debug)]
pub struct Map<K, V, P: Projection = Reborrowed<V>> {
    map: hashbrown::HashMap<K, V>,
    overlay: Option<Overlay<K, P>>,
}

impl<K, V, P: Projection> Map<K, V, P> {
    pub fn new() -> Self {
        Self {
            map: hashbrown::HashMap::new(),
            overlay: None,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            map: hashbrown::HashMap::with_capacity(capacity),
            overlay: None,
        }
    }

    pub fn with_capacity_and(capacity: usize, overlay: Overlay<K, P>) -> Self {
        Self {
            map: hashbrown::HashMap::with_capacity(capacity),
            overlay: Some(overlay),
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
    /// Clears the map before filling. For incremental fills that preserve existing
    /// entries, use [`fill_with`](Self::fill_with) directly.
    ///
    /// Transient errors are acknowledged and skipped. Only critical errors are propagated.
    pub fn fill<'a, A, Itr>(
        &'a mut self,
        accessor: &'a mut A,
        itr: Itr,
    ) -> impl SendFuture<Result<View<'a, K, V, P>, <A::GetError as ToRanked>::Error>>
    where
        A: for<'b> Accessor<Id = K, ElementRef<'b>: Into<V>>,
        Itr: ExactSizeIterator<Item = K> + Send + Sync,
    {
        self.clear();
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
                    Entry::Seeded(_) | Entry::Occupied(_) => { /* in batch or already fetched */ }
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
    /// [`view()`](Self::scoped) which returns a [`View`].
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
        if self.overlay.as_ref().is_some_and(|s| s.contains_key(&key)) {
            return None;
        }
        self.map.insert(key, value)
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
        match self.map.entry(key) {
            hash_map::Entry::Occupied(o) => Entry::Occupied(OccupiedEntry { entry: o }),
            hash_map::Entry::Vacant(v) => Entry::Vacant(VacantEntry { entry: v }),
        }
    }

    /// Borrow as a [`View`].
    pub fn view(&self) -> View<'_, K, V, P> {
        View { map: self }
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

/// An entry in the mutable map layer with no existing value.
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

impl<K, V, P: Projection> AsWorkingSet<Map<K, V, P>> for Unseeded {
    fn as_working_set(&self, capacity: usize) -> Map<K, V, P> {
        Map::with_capacity(capacity)
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

// =========
// = Overlay
// =========

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

/// Seed for [`Map`]-backed working sets.
///
/// Wraps an `Arc<dyn MapLike<K, P>>` backed by a zero-copy [`BatchOverlay`] over a
/// `glue::Batch`. Cheap to clone (just an `Arc` bump). Convert to a [`Map`] via
/// [`AsWorkingSet::as_working_set`].
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
    pub fn new(overlay: Arc<dyn MapLike<K, P>>) -> Self {
        Self { overlay }
    }

    pub fn get(&self, key: &K) -> Option<P::Element<'_>> {
        self.overlay.get(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.overlay.contains_key(key)
    }
}

impl<K, P> Overlay<K, P>
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
        B: for<'a> glue::Batch<Element<'a>: Into<P::Element<'a>>> + Debug,
        K: Debug,
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
}

impl<K, V, P> AsWorkingSet<Map<K, V, P>> for Overlay<K, P>
where
    P: Projection,
{
    fn as_working_set(&self, capacity: usize) -> Map<K, V, P> {
        Map::with_capacity_and(capacity, self.clone())
    }
}

//////////
// View //
//////////

/// A read-only view over a [`Map`], checking seed then fill layer on each lookup.
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
            .or_else(|| self.map.map.get(&id).map(|v| v.project()))
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
        let overlay = Overlay::from_batch(&batch, ids.into_iter());
        (batch, overlay)
    }

    fn test_overlay_projected() -> (Arc<Matrix<f32>>, Overlay<u32, TestProjection>) {
        let batch = Arc::new(test_matrix());
        let ids = [10u32, 20, 30];
        let overlay = Overlay::from_batch(&batch, ids.into_iter());
        (batch, overlay)
    }

    //-----------------------//
    // Map basics (unseeded) //
    //-----------------------//

    #[test]
    fn insert_and_get() {
        let mut map = TestMap::new();

        // Empty Map
        assert!(map.get(&1).is_none());
        assert!(map.get(&2).is_none());
        assert!(map.get(&3).is_none());

        assert!(!map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(!map.contains_key(&3));

        // Insert key !
        assert!(map.insert(1, vec![1.0, 2.0].into_boxed_slice()).is_none());
        assert_eq!(&**map.get(&1).unwrap(), &[1.0, 2.0]);
        assert!(map.get(&2).is_none());
        assert!(map.get(&3).is_none());

        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(!map.contains_key(&3));

        // Insert Key 3
        assert!(map.insert(3, vec![2.0, 3.0].into_boxed_slice()).is_none());
        assert_eq!(&**map.get(&1).unwrap(), &[1.0, 2.0]);
        assert!(map.get(&2).is_none());
        assert_eq!(&**map.get(&3).unwrap(), &[2.0, 3.0]);

        assert!(map.contains_key(&1));
        assert!(!map.contains_key(&2));
        assert!(map.contains_key(&3));
    }

    #[test]
    fn insert_overwrites_existing() {
        let mut map = TestMap::new();
        map.insert(1, Box::new([1.0]));
        let old = map.insert(1, Box::new([2.0]));
        assert_eq!(&*old.unwrap(), &[1.0]);
        assert_eq!(&**map.get(&1).unwrap(), &[2.0]);
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut map = TestMap::new();
        map.insert(1, Box::new([0.0, 0.0]));
        map.get_mut(&1).unwrap()[0] = 42.0;
        assert_eq!(&**map.get(&1).unwrap(), &[42.0, 0.0]);
    }

    #[test]
    fn clear_empties_map() {
        let mut map = TestMap::new();
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
        let mut map = TestMap::new();

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
    }

    #[test]
    fn occupied_entry_mut() {
        let mut map = TestMap::new();

        // Go through `into_mut()`.
        map.insert(1, Box::new([0.0]));
        match map.entry(1) {
            Entry::Occupied(o) => {
                let val = o.into_mut();
                val[0] = 99.0;
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&1).unwrap(), &[99.0]);

        match map.entry(1) {
            Entry::Occupied(mut o) => {
                let val = o.get_mut();
                val[0] = 42.0;
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&1).unwrap(), &[42.0]);
    }

    //-----------------//
    // View (unseeded) //
    //-----------------//

    #[test]
    fn view_returns_projected_element() {
        // Standard Slice Decay
        {
            let mut map = TestMap::new();
            map.insert(1, Box::new([1.0, 2.0]));
            let view = map.view();

            let element = view.get(1).unwrap();
            assert_eq!(element, &[1.0, 2.0]);
            assert!(view.get(999).is_none());
        }

        // Projection - note that the types inserted are still `Box<[f32]>`.
        {
            let mut map = TestMapProjected::new();
            map.insert(1, Box::new([1.0, 2.0]));
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
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(&batch, [10u32, 20, 30].into_iter());

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
        let mut map: TestMapProjected = overlay.as_working_set(16);

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

        match map.entry(99) {
            Entry::Occupied(v) => {
                *v.into_mut() = Box::new([10.0]);
            }
            _ => panic!("expected Occupied"),
        }
        assert_eq!(&**map.get(&99).unwrap(), &[10.0]);
    }

    #[test]
    fn seeded_map_entry_vacant_for_unknown() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map: TestMapProjected = overlay.as_working_set(16);

        match map.entry(99) {
            Entry::Vacant(v) => {
                assert_eq!(*v.key(), 99);
            }
            _ => panic!("expected Vacant for key not in overlay or map"),
        }
    }

    #[test]
    fn seeded_map_insert_skips_seeded_key() {
        let (_batch, overlay) = test_overlay_projected();
        let mut map: TestMapProjected = overlay.as_working_set(16);

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
        let mut map: TestMapProjected = overlay.as_working_set(16);

        map.insert(99, Box::new([7.0, 8.0]));

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
        let mut map: TestMapProjected = overlay.as_working_set(16);

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
        let mut map: TestMap = overlay.as_working_set(16);
        map.insert(99, Box::new([7.0]));

        map.clear();

        let view = map.view();
        // Seed survives clear.
        assert_eq!(view.get(10).unwrap(), &[1.0, 2.0]);
        // Fill layer is gone.
        assert!(view.get(99).is_none());
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

    //----------//
    // Unseeded //
    //----------//

    #[test]
    fn unseeded_creates_empty_map() {
        let ws: TestMap = Unseeded.as_working_set(32);
        assert!(!ws.contains_key(&1));
        // No overlay.
        let view = ws.view();
        assert!(view.get(1).is_none());
    }

    //------------//
    // Edge cases //
    //------------//

    #[test]
    fn overlay_from_batch_empty() {
        let batch = Arc::new(Matrix::try_from(Box::new([]), 0, 2).unwrap());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(&batch, std::iter::empty());
        assert!(overlay.get(&0).is_none());
        assert!(!overlay.contains_key(&0));
    }

    #[test]
    fn overlay_from_batch_single_element() {
        let batch = Arc::new(Matrix::try_from(Box::new([1.0, 2.0]), 1, 2).unwrap());
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(&batch, [42u32].into_iter());
        assert_eq!(overlay.get(&42).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn overlay_from_batch_duplicate_ids_last_wins() {
        // When the same ID appears multiple times, the last index wins (HashMap behavior).
        let batch = Arc::new(test_matrix()); // rows: [1,2], [3,4], [5,6]
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(&batch, [10u32, 10, 10].into_iter());
        // Last occurrence: index 2 → row [5,6]
        assert_eq!(overlay.get(&10).unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn with_capacity_and_default() {
        let map = TestMap::with_capacity(100);
        assert!(!map.contains_key(&1));

        let map2 = TestMap::default();
        assert!(!map2.contains_key(&1));
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
        let mut map: TestMap = Map::new();

        let view = map
            .fill(&mut accessor, [0u32, 1, 2].into_iter())
            .await
            .unwrap();

        assert_eq!(view.get(0).unwrap(), &[0.0]);
        assert_eq!(view.get(1).unwrap(), &[1.0]);
        assert_eq!(view.get(2).unwrap(), &[2.0]);
        assert!(view.get(99).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fill_clears_previous_entries() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: Map<u32, Box<[f32]>, Ref<[f32]>> = Map::new();

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
        let mut map: TestMap = Map::new();

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
        let mut map: TestMap = Map::new();

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
        let mut map: TestMap = Map::new();

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
        let mut map: TestMap = Map::new();

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
        let overlay = Overlay::<u32, Ref<[f32]>>::from_batch(&batch, [0u32, 1].into_iter());
        let mut map: TestMap = overlay.as_working_set(16);

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
        let mut map: TestMap = Map::new();

        let view = map.fill(&mut accessor, std::iter::empty()).await.unwrap();
        assert!(view.get(0).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn blanket_fill_trait() {
        let provider = fill_provider();
        let mut accessor = TestAccessor::new(&provider);
        let mut map: TestMap = Map::new();

        // Exercise the blanket Fill<Map> impl.
        let view = <_ as Fill<TestMap>>::fill(&mut accessor, &mut map, [0u32, 1, 2].into_iter())
            .await
            .unwrap();

        assert_eq!(view.get(0).unwrap(), &[0.0]);
        assert_eq!(view.get(1).unwrap(), &[1.0]);
        assert_eq!(view.get(2).unwrap(), &[2.0]);
    }
}
