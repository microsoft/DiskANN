/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, hash::Hash, num::NonZeroUsize};

use crate::{
    neighbor::{
        Neighbor,
        queue::{
            BestCandidatesIterator, NeighborPriorityQueue, NeighborPriorityQueueIdType,
            NeighborQueue,
        },
    },
    utils::IntoUsize,
};

/// A wrapper type for (VectorIdType, attribute) tuples to implement required traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct VectorIdWithAttr<VectorIdType: NeighborPriorityQueueIdType> {
    pub id: VectorIdType,
    pub attr: u32,
}

impl<VectorIdType: NeighborPriorityQueueIdType> VectorIdWithAttr<VectorIdType> {
    fn new(id: VectorIdType, attr: u32) -> Self {
        Self { id, attr }
    }
}

impl<VectorIdType: NeighborPriorityQueueIdType> std::fmt::Display
    for VectorIdWithAttr<VectorIdType>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.id, self.attr)
    }
}

/// A diverse neighbor priority queue that wraps a standard NeighborPriorityQueue
/// and delegates all operations to it. This provides a foundation for implementing
/// diversity-aware search algorithms while maintaining the same interface.
///
/// This struct serves as a wrapper around NeighborPriorityQueue and can be extended
/// in the future to implement diversity constraints or other specialized behaviors.
#[derive(Debug, Clone)]
pub struct DiverseNeighborQueue<VectorIdType: NeighborPriorityQueueIdType + Hash> {
    /// The underlying priority queue that handles all core operations
    /// Stores VectorIdWithAttr which contains (VectorIdType, attr_value)
    global_queue: NeighborPriorityQueue<VectorIdWithAttr<VectorIdType>>,
    /// Map from attribute_id to local neighbor priority queue
    local_queue_map: HashMap<u32, NeighborPriorityQueue<VectorIdType>>,
    /// Attribute value provider for managing diversity attributes
    attr_provider: PlaceholderAttributeValueProvider,
    /// The calculated diverse_results_l for local queues
    diverse_results_l: usize,
    /// The target number of diverse results (k_value for diversity)
    diverse_results_k: usize,
}

impl<VectorIdType: NeighborPriorityQueueIdType + Hash> DiverseNeighborQueue<VectorIdType> {
    /// Create a new DiverseNeighborQueue with the specified capacity.
    ///
    /// This will implicitly set `l_value` to the provided capacity.
    pub fn new(
        l_value: usize,
        k_value: NonZeroUsize,
        diverse_results_k: usize,
        attr_provider: PlaceholderAttributeValueProvider,
    ) -> Self {
        let diverse_results_l = diverse_results_k * l_value / k_value.get();
        Self {
            global_queue: NeighborPriorityQueue::new(l_value),
            local_queue_map: HashMap::new(),
            attr_provider,
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

impl<VectorIdType: NeighborPriorityQueueIdType + Hash + IntoUsize> NeighborQueue<VectorIdType>
    for DiverseNeighborQueue<VectorIdType>
{
    type Iter<'a>
        = BestCandidatesIterator<'a, VectorIdType, Self>
    where
        Self: 'a,
        VectorIdType: 'a;

    fn insert(&mut self, nbr: Neighbor<VectorIdType>) {
        // Get the attribute_id for the current neighbor
        // TODO: This is a hack to user placeholder attribute provider. To be replaced with actual provider in future PRs.
        let cur_id_usize = nbr.id.into_usize();
        let attr_value = self.attr_provider.get(cur_id_usize).unwrap_or(0);

        // Ensure local queue exists for this attribute and get mutable reference
        let local_queue = self
            .local_queue_map
            .entry(attr_value)
            .or_insert_with(|| NeighborPriorityQueue::new(self.diverse_results_l));

        let local_queue_full = local_queue.is_full();
        let global_queue_full = self.global_queue.is_full();

        // Create a neighbor with VectorIdWithAttr for global_queue
        let nbr_with_attr = Neighbor::new(VectorIdWithAttr::new(nbr.id, attr_value), nbr.distance);

        if !local_queue_full && !global_queue_full {
            // Case 1: Both local queue and global queue have space
            local_queue.insert(nbr);
            self.global_queue.insert(nbr_with_attr);
        } else if local_queue_full {
            // Case 2: Local queue is full
            if nbr.distance < local_queue.get(self.diverse_results_l - 1).distance {
                // Get the worst neighbor in the local queue
                let worst_neighbor = local_queue.get(self.diverse_results_l - 1);
                // Create the corresponding neighbor with attribute for removal from global queue
                let worst_neighbor_with_attr = Neighbor::new(
                    VectorIdWithAttr::new(worst_neighbor.id, attr_value),
                    worst_neighbor.distance,
                );

                // Remove worst neighbor from global queue using the remove method
                self.global_queue.remove(worst_neighbor_with_attr);

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr_with_attr);
            }
        } else if !local_queue_full && global_queue_full {
            // Case 3: Local queue has space but global queue is full
            let l_size = self.global_queue.search_l();
            if nbr.distance < self.global_queue.get(l_size - 1).distance {
                let worst_global = self.global_queue.get(l_size - 1);
                // Extract the attribute from VectorIdWithAttr
                let attr_of_worst_global = worst_global.id.attr;

                // Insert new neighbor into both queues
                local_queue.insert(nbr);
                self.global_queue.insert(nbr_with_attr);

                // Remove worst neighbor from its local queue
                if let Some(local_queue) = self.local_queue_map.get_mut(&attr_of_worst_global) {
                    let worst_neighbor_without_attr =
                        Neighbor::new(worst_global.id.id, worst_global.distance);
                    local_queue.remove(worst_neighbor_without_attr);
                }
            }
        }
    }

    fn get(&self, index: usize) -> Neighbor<VectorIdType> {
        let neighbor_with_attr = self.global_queue.get(index);
        Neighbor::new(neighbor_with_attr.id.id, neighbor_with_attr.distance)
    }

    fn closest_notvisited(&mut self) -> Neighbor<VectorIdType> {
        let neighbor_with_attr = self.global_queue.closest_notvisited();
        Neighbor::new(neighbor_with_attr.id.id, neighbor_with_attr.distance)
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

    fn iter(&self) -> BestCandidatesIterator<'_, VectorIdType, Self> {
        let sz = self.global_queue.search_l().min(self.global_queue.size());
        BestCandidatesIterator::new(sz, self)
    }
}

/// A placeholder attribute value provider that stores attribute values for vector IDs.
/// This is a simple in-memory store using a HashMap.
/// TODO: Remove this and use actual attribute value provider in future PRs.
#[derive(Debug, Clone)]
pub struct PlaceholderAttributeValueProvider {
    /// Map from vector_id to attribute value
    attributes: HashMap<usize, u32>,
}

impl PlaceholderAttributeValueProvider {
    /// Create a new empty PlaceholderAttributeValueProvider.
    pub fn new() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }

    /// Insert an attribute value for a given vector ID.
    ///
    /// # Arguments
    /// * `vector_id` - The vector ID
    /// * `attr_value` - The attribute value to store
    pub fn insert(&mut self, vector_id: usize, attr_value: u32) {
        self.attributes.insert(vector_id, attr_value);
    }

    /// Get the attribute value for a given vector ID.
    ///
    /// # Arguments
    /// * `vector_id` - The vector ID
    ///
    /// # Returns
    /// * `Option<u32>` - The attribute value if it exists, None otherwise
    pub fn get(&self, vector_id: usize) -> Option<u32> {
        self.attributes.get(&vector_id).copied()
    }
}

impl Default for PlaceholderAttributeValueProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod diverse_priority_queue_test {
    use super::*;

    /// Helper function to create a test attribute provider
    fn create_test_attr_provider() -> PlaceholderAttributeValueProvider {
        let mut provider = PlaceholderAttributeValueProvider::new();
        // Set up attributes: vectors 0-2 have attr 0, vectors 3-5 have attr 1, etc.
        for i in 0..20 {
            provider.insert(i, (i / 3) as u32);
        }
        provider
    }

    #[test]
    fn test_new() {
        let attr_provider = create_test_attr_provider();
        let queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);
        assert_eq!(queue.diverse_results_l, 10);
    }

    #[test]
    fn test_insert_single_attribute() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        // Insert neighbors with different attributes
        queue.insert(Neighbor::new(0, 1.0)); // attr 0
        queue.insert(Neighbor::new(3, 0.8)); // attr 1
        queue.insert(Neighbor::new(6, 1.2)); // attr 2

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.local_queue_map.len(), 3);
        assert!(queue.local_queue_map.contains_key(&0));
        assert!(queue.local_queue_map.contains_key(&1));
        assert!(queue.local_queue_map.contains_key(&2));
    }

    #[test]
    fn test_insert_maintains_order() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let mut attr_provider = PlaceholderAttributeValueProvider::new();
        // All IDs 10-15 have the same attribute (attr 0)
        for i in 10..=15 {
            attr_provider.insert(i, 0);
        }
        // l_value=20, k_value=20, diverse_results_k=3 => diverse_results_l = 3 * 20 / 20 = 3
        let mut queue =
            DiverseNeighborQueue::<u32>::new(20, NonZeroUsize::new(20).unwrap(), 3, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(3, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        // Fill up inner queue (capacity = 3)
        queue.insert(Neighbor::new(0, 1.0)); // attr 0
        queue.insert(Neighbor::new(3, 0.8)); // attr 1
        queue.insert(Neighbor::new(6, 1.2)); // attr 2

        assert_eq!(queue.size(), 3);

        // Insert a better neighbor with a new attribute
        queue.insert(Neighbor::new(9, 0.5)); // attr 3, better distance

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 9); // Best should be the new one
    }

    #[test]
    fn test_get() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        assert!(!queue.has_notvisited_node());

        queue.insert(Neighbor::new(0, 1.0));
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_size() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        assert_eq!(queue.size(), 0);

        queue.insert(Neighbor::new(0, 1.0));
        assert_eq!(queue.size(), 1);

        queue.insert(Neighbor::new(1, 0.5));
        assert_eq!(queue.size(), 2);
    }

    #[test]
    fn test_capacity() {
        let attr_provider = create_test_attr_provider();
        let queue =
            DiverseNeighborQueue::<u32>::new(15, NonZeroUsize::new(5).unwrap(), 5, attr_provider);
        assert_eq!(queue.capacity(), 15);
    }

    #[test]
    fn test_search_l() {
        let attr_provider = create_test_attr_provider();
        let queue =
            DiverseNeighborQueue::<u32>::new(20, NonZeroUsize::new(5).unwrap(), 5, attr_provider);
        assert_eq!(queue.search_l(), 20);
    }

    #[test]
    fn test_clear() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

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
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 5, attr_provider);

        queue.insert(Neighbor::new(0, 1.0));

        // Test direct access to global_queue
        assert_eq!(queue.global_queue.size(), 1);

        // Test mutable access to global_queue
        queue.global_queue.clear();
        assert_eq!(queue.global_queue.size(), 0);
    }

    #[test]
    fn test_vector_id_with_attr() {
        let vid_attr = VectorIdWithAttr::new(42u32, 7);
        assert_eq!(vid_attr.id, 42);
        assert_eq!(vid_attr.attr, 7);

        let formatted = format!("{}", vid_attr);
        assert_eq!(formatted, "(42, 7)");
    }

    #[test]
    fn test_placeholder_attribute_value_provider() {
        let mut provider = PlaceholderAttributeValueProvider::new();

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
    fn test_placeholder_attribute_value_provider_default() {
        let provider = PlaceholderAttributeValueProvider::default();
        assert_eq!(provider.get(0), None);
    }

    #[test]
    fn test_diverse_queue_complex_scenario() {
        let attr_provider = create_test_attr_provider();
        let mut queue =
            DiverseNeighborQueue::<u32>::new(10, NonZeroUsize::new(5).unwrap(), 3, attr_provider);

        // Insert multiple neighbors with different attributes
        queue.insert(Neighbor::new(0, 1.0)); // attr 0
        queue.insert(Neighbor::new(1, 0.5)); // attr 0
        queue.insert(Neighbor::new(2, 1.5)); // attr 0
        queue.insert(Neighbor::new(3, 0.8)); // attr 1
        queue.insert(Neighbor::new(4, 1.2)); // attr 1
        queue.insert(Neighbor::new(6, 0.7)); // attr 2

        assert_eq!(queue.size(), 6);

        // Try to add more to attribute 0 when its local queue is full (use unique ID 17)
        // ID 17 has attr 5, so let's use ID 15 which has attr 5 but we'll manually set attr 0
        let mut attr_provider_updated = PlaceholderAttributeValueProvider::new();
        for i in 0..20 {
            attr_provider_updated.insert(i, (i / 3) as u32);
        }
        attr_provider_updated.insert(17, 0); // Set ID 17 to have attr 0

        // Create a new queue with updated provider
        let mut queue2 = DiverseNeighborQueue::<u32>::new(
            10,
            NonZeroUsize::new(5).unwrap(),
            3,
            attr_provider_updated,
        );
        queue2.insert(Neighbor::new(0, 1.0)); // attr 0
        queue2.insert(Neighbor::new(1, 0.5)); // attr 0
        queue2.insert(Neighbor::new(2, 1.5)); // attr 0
        queue2.insert(Neighbor::new(3, 0.8)); // attr 1
        queue2.insert(Neighbor::new(4, 1.2)); // attr 1
        queue2.insert(Neighbor::new(6, 0.7)); // attr 2

        // Now insert ID 17 with attr 0 and better distance
        queue2.insert(Neighbor::new(17, 0.3)); // Should replace worst in attr 0

        // Verify best neighbor is from the new insertion
        assert_eq!(queue2.get(0).id, 17);
        assert_eq!(queue2.get(0).distance, 0.3);
    }

    #[test]
    fn test_post_process() {
        let mut attr_provider = PlaceholderAttributeValueProvider::new();
        // Set up attributes: IDs 0-2 have attr 0, IDs 3-5 have attr 1, IDs 6-8 have attr 2
        for i in 0..9 {
            attr_provider.insert(i, (i / 3) as u32);
        }

        // Create queue with l_value=20, k_value=5, diverse_results_k=2
        // This gives diverse_results_l = 2 * 20 / 5 = 8
        let mut queue =
            DiverseNeighborQueue::<u32>::new(20, NonZeroUsize::new(5).unwrap(), 2, attr_provider);

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
}
