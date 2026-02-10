/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    num::NonZeroUsize,
    sync::Arc,
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

/// A wrapper type for (VectorIdType, attribute) tuples to implement required traits
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
    fn new(id: I, attribute: A) -> Self {
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

/// A diverse neighbor priority queue that wraps a standard NeighborPriorityQueue
/// and delegates all operations to it. This provides a foundation for implementing
/// diversity-aware search algorithms while maintaining the same interface.
///
/// This struct serves as a wrapper around NeighborPriorityQueue and can be extended
/// in the future to implement diversity constraints or other specialized behaviors.
#[derive(Debug, Clone)]
pub struct DiverseNeighborQueue<P>
where
    P: AttributeValueProvider,
{
    /// The underlying priority queue that handles all core operations
    /// Stores VectorIdWithAttribute which contains (VectorIdType, attribute_value)
    global_queue: NeighborPriorityQueue<VectorIdWithAttribute<P::Id, P::Value>>,
    /// Map from attribute_id to local neighbor priority queue
    local_queue_map: HashMap<P::Value, NeighborPriorityQueue<P::Id>>,
    /// Attribute value provider for managing diversity attributes
    attribute_provider: Arc<P>,
    /// The calculated diverse_results_l for local queues
    diverse_results_l: usize,
    /// The target number of diverse results (k_value for diversity)
    diverse_results_k: usize,
}

impl<P> DiverseNeighborQueue<P>
where
    P: AttributeValueProvider,
{
    /// Create a new DiverseNeighborQueue with the specified capacity.
    ///
    /// This will implicitly set `l_value` to the provided capacity.
    pub fn new(
        l_value: usize,
        k_value: NonZeroUsize,
        diverse_results_k: usize,
        attribute_provider: Arc<P>,
    ) -> Self {
        let diverse_results_l = diverse_results_k * l_value / k_value.get();
        Self {
            global_queue: NeighborPriorityQueue::new(l_value),
            local_queue_map: HashMap::new(),
            attribute_provider,
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
                .retain(|neighbor| !removed_items.contains(&neighbor.id.id));
        }
    }
}

impl<P> NeighborQueue<P::Id> for DiverseNeighborQueue<P>
where
    P: AttributeValueProvider,
{
    type Iter<'a>
        = BestCandidatesIterator<'a, P::Id, Self>
    where
        Self: 'a,
        P::Id: 'a;

    fn insert(&mut self, nbr: Neighbor<P::Id>) {
        // Get the attribute value for the current neighbor.
        // We explicitly skip neighbors without attributes (returning None) rather than using
        // unwrap_or_default(), because using a default value would conflate "missing attribute"
        // with "attribute value 0" (or whatever the default is). This could violate diversity
        // constraints by incorrectly grouping neighbors without attributes together with
        // neighbors that legitimately have the default attribute value.
        let Some(attribute_value) = self.attribute_provider.get(nbr.id) else {
            return;
        };

        // Ensure local queue exists for this attribute and get mutable reference
        let local_queue = self
            .local_queue_map
            .entry(attribute_value)
            .or_insert_with(|| NeighborPriorityQueue::new(self.diverse_results_l));

        let local_queue_full = local_queue.is_full();
        let global_queue_full = self.global_queue.is_full();

        // Create a neighbor with VectorIdWithAttribute for global_queue
        let nbr_with_attribute = Neighbor::new(
            VectorIdWithAttribute::new(nbr.id, attribute_value),
            nbr.distance,
        );

        if !local_queue_full && !global_queue_full {
            // Case 1: Both local queue and global queue have space
            local_queue.insert(nbr);
            self.global_queue.insert(nbr_with_attribute);
        } else if local_queue_full {
            // Case 2: Local queue is full
            if nbr.distance < local_queue.get(self.diverse_results_l - 1).distance {
                // Get the worst neighbor in the local queue
                let worst_neighbor = local_queue.get(self.diverse_results_l - 1);
                // Create the corresponding neighbor with attribute for removal from global queue
                let worst_neighbor_with_attribute = Neighbor::new(
                    VectorIdWithAttribute::new(worst_neighbor.id, attribute_value),
                    worst_neighbor.distance,
                );

                // Remove worst neighbor from global queue using the remove method
                self.global_queue.remove(worst_neighbor_with_attribute);

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr_with_attribute);
            }
        } else if !local_queue_full && global_queue_full {
            // Case 3: Local queue has space but global queue is full
            let l_size = self.global_queue.search_l();
            if nbr.distance < self.global_queue.get(l_size - 1).distance {
                let worst_global = self.global_queue.get(l_size - 1);
                // Extract the attribute from VectorIdWithAttribute
                let attribute_of_worst_global = worst_global.id.attribute;

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr_with_attribute);

                // Remove worst neighbor from its local queue
                if let Some(local_queue) = self.local_queue_map.get_mut(&attribute_of_worst_global)
                {
                    let worst_neighbor_without_attribute =
                        Neighbor::new(worst_global.id.id, worst_global.distance);
                    local_queue.remove(worst_neighbor_without_attribute);
                }
            }
        }
    }

    fn get(&self, index: usize) -> Neighbor<P::Id> {
        let neighbor_with_attribute = self.global_queue.get(index);
        Neighbor::new(
            neighbor_with_attribute.id.id,
            neighbor_with_attribute.distance,
        )
    }

    fn closest_notvisited(&mut self) -> Neighbor<P::Id> {
        let neighbor_with_attribute = self.global_queue.closest_notvisited();
        Neighbor::new(
            neighbor_with_attribute.id.id,
            neighbor_with_attribute.distance,
        )
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

    fn iter(&self) -> BestCandidatesIterator<'_, P::Id, Self> {
        let sz = self.global_queue.search_l().min(self.global_queue.size());
        BestCandidatesIterator::new(sz, self)
    }
}

/// Trait for providing attribute values for vector IDs.
/// Implementations of this trait can be used with diverse search to retrieve
/// attribute values for vectors during search operations.
pub trait AttributeValueProvider: crate::provider::HasId + Send + Sync + std::fmt::Debug {
    type Value: Attribute;

    /// Get the attribute value for a given vector ID.
    ///
    /// # Arguments
    /// * `id` - The vector ID
    ///
    /// # Returns
    /// * `Option<Self::Value>` - The attribute value if it exists, None otherwise
    fn get(&self, id: Self::Id) -> Option<Self::Value>;
}

#[cfg(test)]
mod diverse_priority_queue_test {
    use super::*;

    /// A test attribute value provider that stores attribute values for vector IDs.
    /// This is a simple in-memory store using a HashMap for testing purposes.
    #[derive(Debug, Clone)]
    struct TestAttributeValueProvider {
        /// Map from vector_id to attribute value
        attributes: HashMap<u32, u32>,
    }

    impl TestAttributeValueProvider {
        /// Create a new empty TestAttributeValueProvider.
        fn new() -> Self {
            Self {
                attributes: HashMap::new(),
            }
        }

        /// Insert an attribute value for a given vector ID.
        fn insert(&mut self, vector_id: u32, attribute_value: u32) {
            self.attributes.insert(vector_id, attribute_value);
        }
    }

    impl crate::provider::HasId for TestAttributeValueProvider {
        type Id = u32;
    }

    impl AttributeValueProvider for TestAttributeValueProvider {
        type Value = u32;

        fn get(&self, id: Self::Id) -> Option<Self::Value> {
            self.attributes.get(&id).copied()
        }
    }

    impl Default for TestAttributeValueProvider {
        fn default() -> Self {
            Self::new()
        }
    }

    // Type alias for tests to make them more readable
    type TestDiverseQueue = DiverseNeighborQueue<TestAttributeValueProvider>;

    /// Helper function to create a test attribute provider wrapped in Arc
    fn create_test_attribute_provider() -> Arc<TestAttributeValueProvider> {
        let mut provider = TestAttributeValueProvider::new();
        // Set up attributes: vectors 0-2 have attribute 0, vectors 3-5 have attribute 1, etc.
        for i in 0..20 {
            provider.insert(i, i / 3);
        }
        Arc::new(provider)
    }

    #[test]
    fn test_new() {
        let attribute_provider = create_test_attribute_provider();
        let queue = TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);
        assert_eq!(queue.diverse_results_l, 10);
    }

    #[test]
    fn test_insert_single_attribute() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        // Insert neighbors with IDs 0, 1, 2 (all have attribute 0)
        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 1.5));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 1);
        assert!(queue.local_queue_map.contains_key(&0));
    }

    #[test]
    fn test_insert_multiple_attributes() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        // Insert neighbors with different attributes
        queue.insert(Neighbor::new(0, 1.0)); // attribute 0
        queue.insert(Neighbor::new(3, 0.8)); // attribute 1
        queue.insert(Neighbor::new(6, 1.2)); // attribute 2

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 3);
        assert!(queue.local_queue_map.contains_key(&0));
        assert!(queue.local_queue_map.contains_key(&1));
        assert!(queue.local_queue_map.contains_key(&2));
    }

    #[test]
    fn test_insert_maintains_order() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 1.5));

        // Neighbors should be sorted by distance
        assert_eq!(queue.get(0).id, 1); // distance 0.5
        assert_eq!(queue.get(1).id, 0); // distance 1.0
        assert_eq!(queue.get(2).id, 2); // distance 1.5
    }

    #[test]
    fn test_insert_local_queue_full() {
        let mut attribute_provider = TestAttributeValueProvider::new();
        // All IDs 10-15 have the same attribute (attribute 0)
        for i in 10..=15 {
            attribute_provider.insert(i, 0);
        }
        // l_value=20, k_value=20, diverse_results_k=3 => diverse_results_l = 3 * 20 / 20 = 3
        let mut queue = TestDiverseQueue::new(
            20,
            NonZeroUsize::new(20).unwrap(),
            3,
            Arc::new(attribute_provider),
        );

        // Fill up the local queue for attribute 0 (diverse_results_l = 3)
        queue.insert(Neighbor::new(10, 1.0));
        queue.insert(Neighbor::new(11, 0.8));
        queue.insert(Neighbor::new(12, 1.2));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map[&0].size(), 3);

        // Try to insert a better neighbor with same attribute (different ID)
        queue.insert(Neighbor::new(13, 0.5)); // Better distance, should replace worst

        assert_eq!(queue.size(), 3); // Size should remain same
        assert_eq!(queue.get(0).id, 13); // Best is now the new one with distance 0.5
    }

    #[test]
    fn test_insert_inner_queue_full() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(3, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        // Fill up inner queue (capacity = 3)
        queue.insert(Neighbor::new(0, 1.0)); // attribute 0
        queue.insert(Neighbor::new(3, 0.8)); // attribute 1
        queue.insert(Neighbor::new(6, 1.2)); // attribute 2

        assert_eq!(queue.size(), 3);

        // Insert a better neighbor with a new attribute
        queue.insert(Neighbor::new(9, 0.5)); // attribute 3, better distance

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 9); // Best should be the new one
    }

    #[test]
    fn test_get() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));

        let nbr = queue.get(0);
        assert_eq!(nbr.id, 1);
        assert_eq!(nbr.distance, 0.5);

        let nbr = queue.get(1);
        assert_eq!(nbr.id, 0);
        assert_eq!(nbr.distance, 1.0);
    }

    #[test]
    fn test_closest_notvisited() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 1.5));

        assert!(queue.has_notvisited_node());

        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 1); // Best unvisited
        assert_eq!(nbr.distance, 0.5);

        assert!(queue.has_notvisited_node());

        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 0); // Next best

        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 2); // Last one

        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_has_notvisited_node() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        assert!(!queue.has_notvisited_node());

        queue.insert(Neighbor::new(0, 1.0));
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_size() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        assert_eq!(queue.size(), 0);

        queue.insert(Neighbor::new(0, 1.0));
        assert_eq!(queue.size(), 1);

        queue.insert(Neighbor::new(1, 0.5));
        assert_eq!(queue.size(), 2);
    }

    #[test]
    fn test_capacity() {
        let attribute_provider = create_test_attribute_provider();
        let queue = TestDiverseQueue::new(15, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);
        assert_eq!(queue.capacity(), 15);
    }

    #[test]
    fn test_search_l() {
        let attribute_provider = create_test_attribute_provider();
        let queue = TestDiverseQueue::new(20, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);
        assert_eq!(queue.search_l(), 20);
    }

    #[test]
    fn test_clear() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(3, 0.5));
        queue.insert(Neighbor::new(6, 1.5));

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 3);

        queue.clear();

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.local_queue_map.len(), 0);
    }

    #[test]
    fn test_iter_candidates() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 1.5));

        let candidates: Vec<_> = queue.iter().collect();

        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].id, 1);
        assert_eq!(candidates[1].id, 0);
        assert_eq!(candidates[2].id, 2);
    }

    #[test]
    fn test_inner_and_inner_mut() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 5, attribute_provider);

        queue.insert(Neighbor::new(0, 1.0));

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

        let formatted = format!("{}", vid_attr);
        assert_eq!(formatted, "(42, 7)");
    }

    #[test]
    fn test_attribute_value_provider() {
        let mut provider = TestAttributeValueProvider::new();

        assert_eq!(provider.get(0), None);

        provider.insert(0, 10);
        assert_eq!(provider.get(0), Some(10));

        provider.insert(5, 20);
        assert_eq!(provider.get(5), Some(20));

        // Update existing value
        provider.insert(0, 15);
        assert_eq!(provider.get(0), Some(15));
    }

    #[test]
    fn test_attribute_value_provider_default() {
        let provider = TestAttributeValueProvider::default();
        assert_eq!(provider.get(0), None);
    }

    #[test]
    fn test_diverse_queue_complex_scenario() {
        let attribute_provider = create_test_attribute_provider();
        let mut queue =
            TestDiverseQueue::new(10, NonZeroUsize::new(5).unwrap(), 3, attribute_provider);

        // Insert multiple neighbors with different attributes
        queue.insert(Neighbor::new(0, 1.0)); // attribute 0
        queue.insert(Neighbor::new(1, 0.5)); // attribute 0
        queue.insert(Neighbor::new(2, 1.5)); // attribute 0
        queue.insert(Neighbor::new(3, 0.8)); // attribute 1
        queue.insert(Neighbor::new(4, 1.2)); // attribute 1
        queue.insert(Neighbor::new(6, 0.7)); // attribute 2

        assert_eq!(queue.size(), 6);

        // Try to add more to attribute 0 when its local queue is full (use unique ID 17)
        // ID 17 has attribute 5, so let's use ID 15 which has attribute 5 but we'll manually set attribute 0
        let mut attribute_provider_updated = TestAttributeValueProvider::new();
        for i in 0..20 {
            attribute_provider_updated.insert(i, i / 3);
        }
        attribute_provider_updated.insert(17, 0); // Set ID 17 to have attribute 0

        // Create a new queue with updated provider
        let mut queue2 = TestDiverseQueue::new(
            10,
            NonZeroUsize::new(5).unwrap(),
            3,
            Arc::new(attribute_provider_updated),
        );
        queue2.insert(Neighbor::new(0, 1.0)); // attribute 0
        queue2.insert(Neighbor::new(1, 0.5)); // attribute 0
        queue2.insert(Neighbor::new(2, 1.5)); // attribute 0
        queue2.insert(Neighbor::new(3, 0.8)); // attribute 1
        queue2.insert(Neighbor::new(4, 1.2)); // attribute 1
        queue2.insert(Neighbor::new(6, 0.7)); // attribute 2

        // Now insert ID 17 with attribute 0 and better distance
        queue2.insert(Neighbor::new(17, 0.3)); // Should replace worst in attribute 0

        // Verify best neighbor is from the new insertion
        assert_eq!(queue2.get(0).id, 17);
        assert_eq!(queue2.get(0).distance, 0.3);
    }

    #[test]
    fn test_post_process() {
        let mut attribute_provider = TestAttributeValueProvider::new();
        // Set up attributes: IDs 0-2 have attribute 0, IDs 3-5 have attribute 1, IDs 6-8 have attribute 2
        for i in 0..9 {
            attribute_provider.insert(i, i / 3);
        }

        // Create queue with l_value=20, k_value=5, diverse_results_k=2
        // This gives diverse_results_l = 2 * 20 / 5 = 8
        let mut queue = TestDiverseQueue::new(
            20,
            NonZeroUsize::new(5).unwrap(),
            2,
            Arc::new(attribute_provider),
        );

        // Insert more than diverse_results_k items for each attribute
        // Attribute 0
        queue.insert(Neighbor::new(0, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
        queue.insert(Neighbor::new(2, 1.5));

        // Attribute 1
        queue.insert(Neighbor::new(3, 0.8));
        queue.insert(Neighbor::new(4, 1.2));
        queue.insert(Neighbor::new(5, 0.6));

        // Attribute 2
        queue.insert(Neighbor::new(6, 0.7));
        queue.insert(Neighbor::new(7, 1.1));
        queue.insert(Neighbor::new(8, 0.9));

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
        assert_eq!(queue.local_queue_map[&0].get(0).id, 1);
        assert_eq!(queue.local_queue_map[&0].get(0).distance, 0.5);
        assert_eq!(queue.local_queue_map[&0].get(1).id, 0);
        assert_eq!(queue.local_queue_map[&0].get(1).distance, 1.0);

        // Attribute 1: best are ID 5 (0.6) and ID 3 (0.8), worst ID 4 (1.2) should be removed
        assert_eq!(queue.local_queue_map[&1].get(0).id, 5);
        assert_eq!(queue.local_queue_map[&1].get(0).distance, 0.6);
        assert_eq!(queue.local_queue_map[&1].get(1).id, 3);
        assert_eq!(queue.local_queue_map[&1].get(1).distance, 0.8);

        // Attribute 2: best are ID 6 (0.7) and ID 8 (0.9), worst ID 7 (1.1) should be removed
        assert_eq!(queue.local_queue_map[&2].get(0).id, 6);
        assert_eq!(queue.local_queue_map[&2].get(0).distance, 0.7);
        assert_eq!(queue.local_queue_map[&2].get(1).id, 8);
        assert_eq!(queue.local_queue_map[&2].get(1).distance, 0.9);

        // Verify global queue has the correct items in sorted order
        assert_eq!(queue.get(0).id, 1); // 0.5
        assert_eq!(queue.get(1).id, 5); // 0.6
        assert_eq!(queue.get(2).id, 6); // 0.7
        assert_eq!(queue.get(3).id, 3); // 0.8
        assert_eq!(queue.get(4).id, 8); // 0.9
        assert_eq!(queue.get(5).id, 0); // 1.0
    }

    #[test]
    fn test_skip_neighbors_without_attributes() {
        // Test that neighbors without attributes are silently skipped
        // rather than being conflated with attribute value 0
        let mut attribute_provider = TestAttributeValueProvider::new();

        // Set up some vectors with attributes
        attribute_provider.insert(0, 0); // ID 0 has attribute 0
        attribute_provider.insert(1, 0); // ID 1 has attribute 0
        attribute_provider.insert(2, 1); // ID 2 has attribute 1
        // ID 3 has no attribute (not in the map)
        attribute_provider.insert(4, 0); // ID 4 has attribute 0

        let mut queue = TestDiverseQueue::new(
            10,
            NonZeroUsize::new(5).unwrap(),
            5,
            Arc::new(attribute_provider),
        );

        // Insert neighbors, including one without an attribute
        queue.insert(Neighbor::new(0, 1.0)); // Has attribute 0
        queue.insert(Neighbor::new(1, 0.5)); // Has attribute 0
        queue.insert(Neighbor::new(2, 0.8)); // Has attribute 1
        queue.insert(Neighbor::new(3, 0.3)); // No attribute - should be skipped
        queue.insert(Neighbor::new(4, 1.2)); // Has attribute 0

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
        let ids: Vec<u32> = queue.iter().map(|n| n.id).collect();
        assert!(!ids.contains(&3), "ID 3 should not be in the queue");
        assert_eq!(
            ids,
            vec![1, 2, 0, 4],
            "Queue should contain IDs 1,2,0,4 in order of distance"
        );
    }

    #[test]
    fn test_attribute_zero_vs_missing_attribute() {
        // Verify that attribute value 0 is distinct from missing attributes
        let mut attribute_provider = TestAttributeValueProvider::new();

        // ID 0 explicitly has attribute 0
        attribute_provider.insert(0, 0);
        // ID 1 has no attribute (missing)
        // ID 2 explicitly has attribute 0
        attribute_provider.insert(2, 0);

        let mut queue = TestDiverseQueue::new(
            10,
            NonZeroUsize::new(5).unwrap(),
            5,
            Arc::new(attribute_provider),
        );

        queue.insert(Neighbor::new(0, 1.0)); // Has attribute 0
        queue.insert(Neighbor::new(1, 0.5)); // Missing attribute - should be skipped
        queue.insert(Neighbor::new(2, 0.8)); // Has attribute 0

        // Only IDs 0 and 2 should be in the queue
        assert_eq!(queue.size(), 2);
        assert_eq!(queue.local_queue_map.len(), 1);
        assert!(queue.local_queue_map.contains_key(&0));
        assert_eq!(queue.local_queue_map[&0].size(), 2);

        // Verify ID 1 is not in the queue
        let ids: Vec<u32> = queue.iter().map(|n| n.id).collect();
        assert_eq!(ids, vec![2, 0], "Queue should only contain IDs 2 and 0");
    }
}
