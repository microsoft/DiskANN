/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    num::NonZeroUsize,
};

use crate::neighbor::{
    Neighbor,
    queue::{
        BestCandidatesIterator, NeighborPriorityQueue, NeighborPriorityQueueIdType, NeighborQueue,
    },
};

/// Trait combining all required bounds for attribute value types.
/// This trait is automatically implemented for any type that satisfies all the bounds.
pub trait Attribute: Hash + Eq + Copy + Default + Debug + Display + Send + Sync {}

// Blanket implementation: any type satisfying these bounds automatically implements Attribute
impl<T> Attribute for T where T: Hash + Eq + Copy + Default + Debug + Display + Send + Sync {}

/// Trait for neighbor ids that carry their own diversity attribute.
///
/// Instead of looking up attributes through an external provider, the attribute
/// is obtained directly from the id. Ids that do not have an attribute return
/// `None` and are skipped by [`DiverseNeighborQueue`].
pub trait DiverseId: NeighborPriorityQueueIdType + Hash {
    /// The attribute value type carried by this id.
    type Attribute: Attribute;

    /// Get the attribute value carried by this id, or `None` if it has none.
    fn attribute(&self) -> Option<Self::Attribute>;
}

/// A wrapper type pairing a vector id with an attribute value.
///
/// This is a convenience [`DiverseId`] implementation for cases where the
/// attribute is stored alongside the vector id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VectorIdWithAttribute<I, A>
where
    I: NeighborPriorityQueueIdType,
    A: Attribute,
{
    pub id: I,
    pub attribute: A,
}

impl<I, A> VectorIdWithAttribute<I, A>
where
    I: NeighborPriorityQueueIdType,
    A: Attribute,
{
    pub fn new(id: I, attribute: A) -> Self {
        Self { id, attribute }
    }
}

impl<I, A> Display for VectorIdWithAttribute<I, A>
where
    I: NeighborPriorityQueueIdType,
    A: Attribute,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.id, self.attribute)
    }
}

impl<I, A> DiverseId for VectorIdWithAttribute<I, A>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
{
    type Attribute = A;

    fn attribute(&self) -> Option<Self::Attribute> {
        Some(self.attribute)
    }
}

/// A diverse neighbor priority queue that wraps a standard NeighborPriorityQueue
/// and delegates all operations to it. This provides a foundation for implementing
/// diversity-aware search algorithms while maintaining the same interface.
///
/// The queue is generic over an id type `I` that carries its own diversity
/// attribute via the [`DiverseId`] trait, so no external attribute provider is
/// required.
#[derive(Debug, Clone)]
pub struct DiverseNeighborQueue<I>
where
    I: DiverseId,
{
    /// The underlying priority queue that handles all core operations.
    global_queue: NeighborPriorityQueue<I>,
    /// Map from attribute value to local neighbor priority queue.
    local_queue_map: HashMap<I::Attribute, NeighborPriorityQueue<I>>,
    /// The calculated diverse_results_l for local queues
    diverse_results_l: usize,
    /// The target number of diverse results (k_value for diversity)
    diverse_results_k: usize,
}

impl<I> DiverseNeighborQueue<I>
where
    I: DiverseId,
{
    /// Create a new DiverseNeighborQueue with the specified capacity.
    ///
    /// This will implicitly set `l_value` to the provided capacity.
    pub fn new(l_value: usize, k_value: NonZeroUsize, diverse_results_k: usize) -> Self {
        let diverse_results_l = diverse_results_k * l_value / k_value.get();
        Self {
            global_queue: NeighborPriorityQueue::new(l_value),
            local_queue_map: HashMap::new(),
            diverse_results_l,
            diverse_results_k,
        }
    }

    /// Post-process the queues to keep only diverse_results_k items.
    /// This method should be called after search to trim down the results.
    ///
    /// For each local queue, this iterates through items beyond diverse_results_k
    /// (where N is the queue size) to collect IDs for removal, then performs
    /// in-place compaction on the global queue.
    pub fn post_process(&mut self) {
        use hashbrown::HashSet;

        // Step 1: Identify items to remove, build a HashSet, and truncate local queues
        let mut removed_items = HashSet::new();

        for local_queue in self.local_queue_map.values_mut() {
            if local_queue.size() > self.diverse_results_k {
                // Items are sorted, so items from diverse_results_k onwards are worst
                removed_items.extend(
                    local_queue
                        .iter()
                        .skip(self.diverse_results_k)
                        .map(|n| n.id),
                );

                // Truncate the local queue immediately
                local_queue.truncate(self.diverse_results_k);
            }
        }

        // Step 2: Compact global queue using the filter
        if !removed_items.is_empty() {
            self.global_queue
                .retain(|neighbor| !removed_items.contains(&neighbor.id));
        }
    }
}

impl<I> NeighborQueue<I> for DiverseNeighborQueue<I>
where
    I: DiverseId,
{
    type Iter<'a>
        = BestCandidatesIterator<'a, I, Self>
    where
        Self: 'a,
        I: 'a;

    fn insert(&mut self, nbr: Neighbor<I>) {
        // Get the attribute value carried by the neighbor's id.
        // We explicitly skip neighbors without attributes (returning None) rather than using
        // unwrap_or_default(), because using a default value would conflate "missing attribute"
        // with "attribute value 0" (or whatever the default is). This could violate diversity
        // constraints by incorrectly grouping neighbors without attributes together with
        // neighbors that legitimately have the default attribute value.
        let Some(attribute_value) = nbr.id.attribute() else {
            return;
        };

        // Ensure local queue exists for this attribute and get mutable reference
        let local_queue = self
            .local_queue_map
            .entry(attribute_value)
            .or_insert_with(|| NeighborPriorityQueue::new(self.diverse_results_l));

        let local_queue_full = local_queue.is_full();
        let global_queue_full = self.global_queue.is_full();

        if !local_queue_full && !global_queue_full {
            // Case 1: Both local queue and global queue have space
            local_queue.insert(nbr);
            self.global_queue.insert(nbr);
        } else if local_queue_full {
            // Case 2: Local queue is full
            if nbr.distance < local_queue.get(self.diverse_results_l - 1).distance {
                // Get the worst neighbor in the local queue
                let worst_neighbor = local_queue.get(self.diverse_results_l - 1);

                // Remove worst neighbor from global queue using the remove method
                self.global_queue.remove(worst_neighbor);

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr);
            }
        } else if !local_queue_full && global_queue_full {
            // Case 3: Local queue has space but global queue is full
            let l_size = self.global_queue.search_l();
            if nbr.distance < self.global_queue.get(l_size - 1).distance {
                let worst_global = self.global_queue.get(l_size - 1);
                // The attribute of the worst global neighbor comes from its id.
                let attribute_of_worst_global = worst_global.id.attribute();

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr);

                // Remove worst neighbor from its local queue
                if let Some(attribute_of_worst_global) = attribute_of_worst_global
                    && let Some(local_queue) =
                        self.local_queue_map.get_mut(&attribute_of_worst_global)
                {
                    local_queue.remove(worst_global);
                }
            }
        }
    }

    fn get(&self, index: usize) -> Neighbor<I> {
        self.global_queue.get(index)
    }

    fn closest_notvisited(&mut self) -> Option<Neighbor<I>> {
        self.global_queue.closest_notvisited()
    }

    fn has_notvisited_node(&self) -> bool {
        self.global_queue.has_notvisited_node()
    }

    fn size(&self) -> usize {
        self.global_queue.size()
    }

    fn capacity(&self) -> usize {
        self.global_queue.capacity()
    }

    fn search_l(&self) -> usize {
        self.global_queue.search_l()
    }

    fn clear(&mut self) {
        self.global_queue.clear();
        self.local_queue_map.clear();
    }

    fn iter(&self) -> BestCandidatesIterator<'_, I, Self> {
        let sz = self.global_queue.search_l().min(self.global_queue.size());
        BestCandidatesIterator::new(sz, self)
    }
}

/// Unwrap a [`VectorIdWithAttribute`]-keyed neighbor back to its raw id.
fn unwrap_attribute<I, A>(nbr: Neighbor<VectorIdWithAttribute<I, A>>) -> Neighbor<I>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
{
    Neighbor::new(nbr.id.id, nbr.distance)
}

/// A [`NeighborQueue`] adapter that attaches diversity attributes to raw ids at
/// the search pipeline boundary.
///
/// The graph search pipeline produces bare ids (`I`) that do not carry an
/// attribute. This adapter applies a lookup function to each id, wraps it in a
/// [`VectorIdWithAttribute`] so the attribute travels with the id, and delegates
/// the diversity bookkeeping to an inner [`DiverseNeighborQueue`]. Results are
/// unwrapped back to the original id type `I`, so the rest of the pipeline is
/// unaffected.
///
/// Ids for which the lookup returns `None` have no attribute and are skipped,
/// matching the behavior of [`DiverseNeighborQueue`].
pub struct DiverseAttributeQueue<I, A, F>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
    F: Fn(I) -> Option<A> + Send + Sync,
{
    inner: DiverseNeighborQueue<VectorIdWithAttribute<I, A>>,
    attribute_of: F,
}

impl<I, A, F> DiverseAttributeQueue<I, A, F>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
    F: Fn(I) -> Option<A> + Send + Sync,
{
    /// Create a new adapter wrapping a [`DiverseNeighborQueue`].
    ///
    /// `attribute_of` maps a raw id to its attribute, or `None` if it has none.
    pub fn new(
        l_value: usize,
        k_value: NonZeroUsize,
        diverse_results_k: usize,
        attribute_of: F,
    ) -> Self {
        Self {
            inner: DiverseNeighborQueue::new(l_value, k_value, diverse_results_k),
            attribute_of,
        }
    }

    /// Trim each attribute's results down to `diverse_results_k`.
    ///
    /// See [`DiverseNeighborQueue::post_process`].
    pub fn post_process(&mut self) {
        self.inner.post_process();
    }
}

impl<I, A, F> Debug for DiverseAttributeQueue<I, A, F>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
    F: Fn(I) -> Option<A> + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiverseAttributeQueue")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl<I, A, F> NeighborQueue<I> for DiverseAttributeQueue<I, A, F>
where
    I: NeighborPriorityQueueIdType + Hash,
    A: Attribute,
    F: Fn(I) -> Option<A> + Send + Sync,
{
    type Iter<'a>
        = std::iter::Map<
        BestCandidatesIterator<
            'a,
            VectorIdWithAttribute<I, A>,
            DiverseNeighborQueue<VectorIdWithAttribute<I, A>>,
        >,
        fn(Neighbor<VectorIdWithAttribute<I, A>>) -> Neighbor<I>,
    >
    where
        Self: 'a,
        I: 'a;

    fn insert(&mut self, nbr: Neighbor<I>) {
        // Look up the attribute for this id. Ids without an attribute are skipped.
        let Some(attribute) = (self.attribute_of)(nbr.id) else {
            return;
        };
        self.inner.insert(Neighbor::new(
            VectorIdWithAttribute::new(nbr.id, attribute),
            nbr.distance,
        ));
    }

    fn get(&self, index: usize) -> Neighbor<I> {
        unwrap_attribute(self.inner.get(index))
    }

    fn closest_notvisited(&mut self) -> Option<Neighbor<I>> {
        self.inner.closest_notvisited().map(unwrap_attribute)
    }

    fn has_notvisited_node(&self) -> bool {
        self.inner.has_notvisited_node()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn search_l(&self) -> usize {
        self.inner.search_l()
    }

    fn clear(&mut self) {
        self.inner.clear();
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.inner
            .iter()
            .map(unwrap_attribute as fn(Neighbor<VectorIdWithAttribute<I, A>>) -> Neighbor<I>)
    }
}

#[cfg(test)]
mod diverse_priority_queue_test {
    use super::*;

    /// A test id type that carries an optional attribute value, used to exercise
    /// both the "has attribute" and "missing attribute" code paths.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
    struct TestId {
        id: u32,
        attribute: Option<u32>,
    }

    impl TestId {
        fn new(id: u32, attribute: Option<u32>) -> Self {
            Self { id, attribute }
        }
    }

    impl Display for TestId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.id)
        }
    }

    impl DiverseId for TestId {
        type Attribute = u32;

        fn attribute(&self) -> Option<Self::Attribute> {
            self.attribute
        }
    }

    // Type alias for tests to make them more readable
    type TestDiverseQueue = DiverseNeighborQueue<TestId>;

    /// Build a neighbor whose id carries the attribute `id / 3`.
    ///
    /// This mirrors the original test setup where vectors 0-2 had attribute 0,
    /// 3-5 had attribute 1, and so on.
    fn nbr(id: u32, distance: f32) -> Neighbor<TestId> {
        Neighbor::new(TestId::new(id, Some(id / 3)), distance)
    }

    /// Build a neighbor whose id carries an explicit attribute value.
    fn nbr_attr(id: u32, attribute: u32, distance: f32) -> Neighbor<TestId> {
        Neighbor::new(TestId::new(id, Some(attribute)), distance)
    }

    /// Build a neighbor whose id has no attribute.
    fn nbr_none(id: u32, distance: f32) -> Neighbor<TestId> {
        Neighbor::new(TestId::new(id, None), distance)
    }

    #[test]
    fn test_new() {
        let queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);
        assert_eq!(queue.diverse_results_l, 10);
    }

    #[test]
    fn test_insert_single_attribute() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        // Insert neighbors with IDs 0, 1, 2 (all have attribute 0)
        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));
        queue.insert(nbr(2, 1.5));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 1);
        assert!(queue.local_queue_map.contains_key(&0));
    }

    #[test]
    fn test_insert_multiple_attributes() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        // Insert neighbors with different attributes
        queue.insert(nbr(0, 1.0)); // attribute 0
        queue.insert(nbr(3, 0.8)); // attribute 1
        queue.insert(nbr(6, 1.2)); // attribute 2

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 3);
        assert!(queue.local_queue_map.contains_key(&0));
        assert!(queue.local_queue_map.contains_key(&1));
        assert!(queue.local_queue_map.contains_key(&2));
    }

    #[test]
    fn test_insert_maintains_order() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));
        queue.insert(nbr(2, 1.5));

        // Neighbors should be sorted by distance
        assert_eq!(queue.get(0).id.id, 1); // distance 0.5
        assert_eq!(queue.get(1).id.id, 0); // distance 1.0
        assert_eq!(queue.get(2).id.id, 2); // distance 1.5
    }

    #[test]
    fn test_insert_local_queue_full() {
        // l_value=20, k_value=20, diverse_results_k=3 => diverse_results_l = 3 * 20 / 20 = 3
        // All IDs 10-13 share attribute 0.
        let mut queue = TestDiverseQueue::new(20, NonZeroUsize::new(20).unwrap(), 3);

        // Fill up the local queue for attribute 0 (diverse_results_l = 3)
        queue.insert(nbr_attr(10, 0, 1.0));
        queue.insert(nbr_attr(11, 0, 0.8));
        queue.insert(nbr_attr(12, 0, 1.2));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map[&0].size(), 3);

        // Try to insert a better neighbor with same attribute (different ID)
        queue.insert(nbr_attr(13, 0, 0.5)); // Better distance, should replace worst

        assert_eq!(queue.size(), 3); // Size should remain same
        assert_eq!(queue.get(0).id.id, 13); // Best is now the new one with distance 0.5
    }

    #[test]
    fn test_insert_inner_queue_full() {
        let mut queue = TestDiverseQueue::new(3, NonZeroUsize::new(5).unwrap(), 5);

        // Fill up inner queue (capacity = 3)
        queue.insert(nbr(0, 1.0)); // attribute 0
        queue.insert(nbr(3, 0.8)); // attribute 1
        queue.insert(nbr(6, 1.2)); // attribute 2

        assert_eq!(queue.size(), 3);

        // Insert a better neighbor with a new attribute
        queue.insert(nbr(9, 0.5)); // attribute 3, better distance

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id.id, 9); // Best should be the new one
    }

    #[test]
    fn test_get() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));

        let n = queue.get(0);
        assert_eq!(n.id.id, 1);
        assert_eq!(n.distance, 0.5);

        let n = queue.get(1);
        assert_eq!(n.id.id, 0);
        assert_eq!(n.distance, 1.0);
    }

    #[test]
    fn test_closest_notvisited() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));
        queue.insert(nbr(2, 1.5));

        assert!(queue.has_notvisited_node());

        let n = queue.closest_notvisited().unwrap();
        assert_eq!(n.id.id, 1); // Best unvisited
        assert_eq!(n.distance, 0.5);

        assert!(queue.has_notvisited_node());

        let n = queue.closest_notvisited().unwrap();
        assert_eq!(n.id.id, 0); // Next best

        let n = queue.closest_notvisited().unwrap();
        assert_eq!(n.id.id, 2); // Last one

        assert!(!queue.has_notvisited_node());
        assert!(queue.closest_notvisited().is_none());
    }

    #[test]
    fn test_has_notvisited_node() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        assert!(!queue.has_notvisited_node());

        queue.insert(nbr(0, 1.0));
        assert!(queue.has_notvisited_node());

        assert!(queue.closest_notvisited().is_some());
        assert!(!queue.has_notvisited_node());
        assert!(queue.closest_notvisited().is_none());
    }

    #[test]
    fn test_size() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        assert_eq!(queue.size(), 0);

        queue.insert(nbr(0, 1.0));
        assert_eq!(queue.size(), 1);

        queue.insert(nbr(1, 0.5));
        assert_eq!(queue.size(), 2);
    }

    #[test]
    fn test_capacity() {
        let queue = TestDiverseQueue::new(15, NonZeroUsize::new(5).unwrap(), 5);
        assert_eq!(queue.capacity(), 15);
    }

    #[test]
    fn test_search_l() {
        let queue = TestDiverseQueue::new(20, NonZeroUsize::new(5).unwrap(), 5);
        assert_eq!(queue.search_l(), 20);
    }

    #[test]
    fn test_clear() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(3, 0.5));
        queue.insert(nbr(6, 1.5));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 3);

        queue.clear();

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.local_queue_map.len(), 0);
    }

    #[test]
    fn test_iter_candidates() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));
        queue.insert(nbr(2, 1.5));

        let candidates: Vec<_> = queue.iter().collect();

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].id.id, 1);
        assert_eq!(candidates[1].id.id, 0);
        assert_eq!(candidates[2].id.id, 2);
    }

    #[test]
    fn test_inner_and_inner_mut() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr(0, 1.0));

        // Test direct access to global_queue
        assert_eq!(queue.global_queue.size(), 1);

        // Test mutable access to global_queue
        queue.global_queue.clear();
        assert_eq!(queue.global_queue.size(), 0);
    }

    #[test]
    fn test_vector_id_with_attribute() {
        let vid_attr = VectorIdWithAttribute::new(42u32, 7);
        assert_eq!(vid_attr.id, 42);
        assert_eq!(vid_attr.attribute, 7);
        assert_eq!(vid_attr.attribute(), Some(7));

        let formatted = format!("{}", vid_attr);
        assert_eq!(formatted, "(42, 7)");
    }

    #[test]
    fn test_diverse_queue_complex_scenario() {
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 3);

        // Insert multiple neighbors with different attributes
        queue.insert(nbr(0, 1.0)); // attribute 0
        queue.insert(nbr(1, 0.5)); // attribute 0
        queue.insert(nbr(2, 1.5)); // attribute 0
        queue.insert(nbr(3, 0.8)); // attribute 1
        queue.insert(nbr(4, 1.2)); // attribute 1
        queue.insert(nbr(6, 0.7)); // attribute 2

        assert_eq!(queue.size(), 6);

        // Insert ID 17 with attribute 0 and a better distance; it should become the best.
        queue.insert(nbr_attr(17, 0, 0.3));

        assert_eq!(queue.get(0).id.id, 17);
        assert_eq!(queue.get(0).distance, 0.3);
    }

    #[test]
    fn test_post_process() {
        // Create queue with l_value=20, k_value=5, diverse_results_k=2
        // This gives diverse_results_l = 2 * 20 / 5 = 8.
        // IDs 0-2 have attribute 0, IDs 3-5 have attribute 1, IDs 6-8 have attribute 2.
        let mut queue = TestDiverseQueue::new(20, NonZeroUsize::new(5).unwrap(), 2);

        // Insert more than diverse_results_k items for each attribute
        // Attribute 0
        queue.insert(nbr(0, 1.0));
        queue.insert(nbr(1, 0.5));
        queue.insert(nbr(2, 1.5));

        // Attribute 1
        queue.insert(nbr(3, 0.8));
        queue.insert(nbr(4, 1.2));
        queue.insert(nbr(5, 0.6));

        // Attribute 2
        queue.insert(nbr(6, 0.7));
        queue.insert(nbr(7, 1.1));
        queue.insert(nbr(8, 0.9));

        // Before post_process, we should have all 9 items
        assert_eq!(queue.size(), 9);
        assert_eq!(queue.local_queue_map[&0].size(), 3);
        assert_eq!(queue.local_queue_map[&1].size(), 3);
        assert_eq!(queue.local_queue_map[&2].size(), 3);

        // Call post_process to trim to diverse_results_k (2) per attribute
        queue.post_process();

        // After post_process, each local queue should have at most 2 items
        assert_eq!(queue.local_queue_map[&0].size(), 2);
        assert_eq!(queue.local_queue_map[&1].size(), 2);
        assert_eq!(queue.local_queue_map[&2].size(), 2);

        // Global queue should have 6 items total (2 per attribute * 3 attributes)
        assert_eq!(queue.size(), 6);

        // Verify the best items from each attribute are kept
        // Attribute 0: best are ID 1 (0.5) and ID 0 (1.0), worst ID 2 (1.5) should be removed
        assert_eq!(queue.local_queue_map[&0].get(0).id.id, 1);
        assert_eq!(queue.local_queue_map[&0].get(0).distance, 0.5);
        assert_eq!(queue.local_queue_map[&0].get(1).id.id, 0);
        assert_eq!(queue.local_queue_map[&0].get(1).distance, 1.0);

        // Attribute 1: best are ID 5 (0.6) and ID 3 (0.8), worst ID 4 (1.2) should be removed
        assert_eq!(queue.local_queue_map[&1].get(0).id.id, 5);
        assert_eq!(queue.local_queue_map[&1].get(0).distance, 0.6);
        assert_eq!(queue.local_queue_map[&1].get(1).id.id, 3);
        assert_eq!(queue.local_queue_map[&1].get(1).distance, 0.8);

        // Attribute 2: best are ID 6 (0.7) and ID 8 (0.9), worst ID 7 (1.1) should be removed
        assert_eq!(queue.local_queue_map[&2].get(0).id.id, 6);
        assert_eq!(queue.local_queue_map[&2].get(0).distance, 0.7);
        assert_eq!(queue.local_queue_map[&2].get(1).id.id, 8);
        assert_eq!(queue.local_queue_map[&2].get(1).distance, 0.9);

        // Verify global queue has the correct items in sorted order
        assert_eq!(queue.get(0).id.id, 1); // 0.5
        assert_eq!(queue.get(1).id.id, 5); // 0.6
        assert_eq!(queue.get(2).id.id, 6); // 0.7
        assert_eq!(queue.get(3).id.id, 3); // 0.8
        assert_eq!(queue.get(4).id.id, 8); // 0.9
        assert_eq!(queue.get(5).id.id, 0); // 1.0
    }

    #[test]
    fn test_skip_neighbors_without_attributes() {
        // Test that neighbors without attributes are silently skipped
        // rather than being conflated with attribute value 0.
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        // Insert neighbors, including one without an attribute
        queue.insert(nbr_attr(0, 0, 1.0)); // Has attribute 0
        queue.insert(nbr_attr(1, 0, 0.5)); // Has attribute 0
        queue.insert(nbr_attr(2, 1, 0.8)); // Has attribute 1
        queue.insert(nbr_none(3, 0.3)); // No attribute - should be skipped
        queue.insert(nbr_attr(4, 0, 1.2)); // Has attribute 0

        // Queue should only contain 4 items (ID 3 was skipped)
        assert_eq!(queue.size(), 4, "Expected 4 items, ID 3 should be skipped");

        // Verify the local queue for attribute 0 has 3 items (IDs 0, 1, 4)
        assert_eq!(
            queue.local_queue_map[&0].size(),
            3,
            "Attribute 0 should have 3 items"
        );

        // Verify the local queue for attribute 1 has 1 item (ID 2)
        assert_eq!(
            queue.local_queue_map[&1].size(),
            1,
            "Attribute 1 should have 1 item"
        );

        // Verify ID 3 (without attribute) is not in the queue
        let ids: Vec<u32> = queue.iter().map(|n| n.id.id).collect();
        assert!(!ids.contains(&3), "ID 3 should not be in the queue");
        assert_eq!(
            ids,
            vec![1, 2, 0, 4],
            "Queue should contain IDs 1,2,0,4 in order of distance"
        );
    }

    #[test]
    fn test_attribute_zero_vs_missing_attribute() {
        // Verify that attribute value 0 is distinct from missing attributes.
        let mut queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5);

        queue.insert(nbr_attr(0, 0, 1.0)); // Has attribute 0
        queue.insert(nbr_none(1, 0.5)); // Missing attribute - should be skipped
        queue.insert(nbr_attr(2, 0, 0.8)); // Has attribute 0

        // Only IDs 0 and 2 should be in the queue
        assert_eq!(queue.size(), 2);
        assert_eq!(queue.local_queue_map.len(), 1);
        assert!(queue.local_queue_map.contains_key(&0));
        assert_eq!(queue.local_queue_map[&0].size(), 2);

        // Verify ID 1 is not in the queue
        let ids: Vec<u32> = queue.iter().map(|n| n.id.id).collect();
        assert_eq!(ids, vec![2, 0], "Queue should only contain IDs 2 and 0");
    }
}
