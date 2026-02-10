/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::{HashMap, HashSet, VecDeque};

/// Bookkeeping data structures for slot management in dynamic operations
///
/// This struct manages the relationship between tags, slots, and empty/deleted slots
/// to maintain consistency during dynamic index operations.
#[derive(Debug)]
pub struct TagSlotManager {
    /// Queue of available empty slots
    pub empty_slots: VecDeque<u32>,
    /// Set of deleted slots pending cleanup
    pub deleted_slots: HashSet<u32>,
    /// Mapping from tag ID to slot ID
    pub tag_to_slot: HashMap<usize, u32>,
    /// Mapping from slot ID to tag ID
    pub slot_to_tag: HashMap<u32, usize>,
    /// The total capacity (number of slots)
    capacity: usize,
}

impl TagSlotManager {
    /// Create new slot bookkeeping with given capacity
    pub fn new(max_capacity: usize) -> Self {
        Self {
            empty_slots: (0..(max_capacity - 1) as u32).collect(), // Queue, initialize with 0, ..., max_capacity - 2 to leave room for 1 frozen point.
            deleted_slots: HashSet::new(),
            tag_to_slot: HashMap::new(),
            slot_to_tag: HashMap::new(),
            capacity: max_capacity,
        }
    }

    /// Get n empty slots from the queue
    ///
    /// Returns a vector of slot IDs that have been removed from the empty_slots queue.
    /// If there are insufficient empty slots, returns an error.
    pub fn get_n_empty_slots(&mut self, n: usize) -> anyhow::Result<Vec<u32>> {
        if self.empty_slots.len() < n {
            return Err(anyhow::anyhow!(
                "Insufficient empty slots: requested {}, available {}",
                n,
                self.empty_slots.len()
            ));
        }

        let slots: Vec<u32> = (0..n)
            .filter_map(|_| self.empty_slots.pop_front())
            .collect();

        Ok(slots)
    }

    /// Return the number of deleted elements.
    pub fn num_deleted(&self) -> usize {
        self.deleted_slots.len()
    }

    /// Return the number of active elements (both valid and deleted).
    pub fn num_active(&self) -> usize {
        self.capacity() - self.empty_slots.len()
    }

    /// Return the total number of elements this object can manage.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Find slots corresponding to a range of tag IDs
    ///
    /// Returns a vector of slot IDs for the given tag range.
    /// If any tag is not found in the mapping, returns an error.
    pub fn find_slots_by_tags(
        &self,
        tag_range: std::ops::Range<usize>,
    ) -> anyhow::Result<Vec<u32>> {
        tag_range
            .map(|tag| {
                self.tag_to_slot
                    .get(&tag)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Tag {} not found in slot mapping", tag))
            })
            .collect()
    }

    /// Assign slots to a range of tags
    ///
    /// Creates bidirectional mappings between tags and slots for the given range.
    /// Takes ownership of the slots vector for efficiency.
    pub fn assign_slots_to_tags(
        &mut self,
        tag_range: std::ops::Range<usize>,
        slots: Vec<u32>,
    ) -> anyhow::Result<()> {
        if tag_range.len() != slots.len() {
            return Err(anyhow::anyhow!(
                "Tag range length ({}) does not match slots length ({})",
                tag_range.len(),
                slots.len()
            ));
        }

        for (tag, slot) in tag_range.zip(slots) {
            self.tag_to_slot.insert(tag, slot);
            self.slot_to_tag.insert(slot, tag);
        }

        Ok(())
    }

    /// Mark tags as deleted by moving their slots to the deleted_slots set
    ///
    /// For each tag in the range, finds the corresponding slot and adds it to deleted_slots.
    /// The tag-to-slot mappings remain intact until background cleaning occurs.
    pub fn mark_tags_deleted(&mut self, tag_range: std::ops::Range<usize>) -> anyhow::Result<()> {
        for tag in tag_range {
            if let Some(&slot) = self.tag_to_slot.get(&tag) {
                self.deleted_slots.insert(slot);
            } else {
                return Err(anyhow::anyhow!(
                    "Tag {} not found in slot mapping for delete",
                    tag
                ));
            }
        }
        Ok(())
    }

    /// Consolidate deleted slots back to empty slots after background cleaning
    ///
    /// This method processes all deleted slots by:
    /// 1. Removing their bidirectional tag-slot mappings
    /// 2. Moving the slots back to the empty_slots queue
    /// 3. Clearing the deleted_slots set
    ///
    /// Should be called after background cleaning operations complete.
    pub fn consolidate(&mut self) {
        let slots_to_consolidate: Vec<u32> = self.deleted_slots.drain().collect();

        slots_to_consolidate.into_iter().for_each(|slot| {
            // Remove bidirectional mappings if they exist
            if let Some(tag_id) = self.slot_to_tag.remove(&slot) {
                self.tag_to_slot.remove(&tag_id);
            }
            // Return slot to available pool
            self.empty_slots.push_back(slot);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl TagSlotManager {
        /// Check invariants between bookkeeping data structures (testing only)
        fn check_invariants(&self, max_capacity: usize, context: &str) -> anyhow::Result<()> {
            // 1. Every key in slot_to_tag must also exist in slot_to_tag
            for &slot in self.slot_to_tag.keys() {
                if !self.slot_to_tag.contains_key(&slot) {
                    return Err(anyhow::anyhow!(
                        "[{}] Invariant 1 violated: slot {} exists in slot_to_tag but not in slot_to_tag",
                        context, slot
                    ));
                }
            }

            // 2. Every tag that exists in slot_to_tag as a value must exist in tag_to_slot and mapping back to the same slot
            for (&slot, &tag_id) in self.slot_to_tag.iter() {
                match self.tag_to_slot.get(&tag_id) {
                    Some(&mapped_slot) => {
                        if mapped_slot != slot {
                            return Err(anyhow::anyhow!(
                                "[{}] Invariant 3 violated: slot {} maps to tag {} but tag {} maps back to slot {}",
                                context, slot, tag_id, tag_id, mapped_slot
                            ));
                        }
                    }
                    None => {
                        return Err(anyhow::anyhow!(
                            "[{}] Invariant 3 violated: slot {} maps to tag {} but tag {} not found in tag_to_slot",
                            context, slot, tag_id, tag_id
                        ));
                    }
                }
            }

            // 3. Every key in slot_to_tag must NOT exist in empty_slots
            // Every key between 0 and max_capacity - 1 must be either in slot_to_tag or in empty_slots
            let empty_slots_set: HashSet<u32> = self.empty_slots.iter().copied().collect();

            for &slot in self.slot_to_tag.keys() {
                if empty_slots_set.contains(&slot) {
                    return Err(anyhow::anyhow!(
                        "[{}] Invariant 4a violated: slot {} exists in both slot_to_tag and empty_slots",
                        context, slot
                    ));
                }
            }

            for slot in 0..(max_capacity - 1) {
                let slot_u32: u32 = slot.try_into().unwrap();
                let in_slot_to_tag = self.slot_to_tag.contains_key(&slot_u32);
                let in_empty_slots = empty_slots_set.contains(&slot_u32);

                if !in_slot_to_tag && !in_empty_slots {
                    return Err(anyhow::anyhow!(
                        "[{}] Invariant 4b violated: slot {} is neither in slot_to_tag nor in empty_slots",
                        context, slot
                    ));
                }

                if in_slot_to_tag && in_empty_slots {
                    return Err(anyhow::anyhow!(
                        "[{}] Invariant 4c violated: slot {} exists in both slot_to_tag and empty_slots",
                        context, slot
                    ));
                }
            }

            // 4. Every key in deleted_slots must NOT be in empty_slots
            for &deleted_slot in self.deleted_slots.iter() {
                if empty_slots_set.contains(&deleted_slot) {
                    return Err(anyhow::anyhow!(
                        "[{}] Invariant 4 violated: slot {} exists in both deleted_slots and empty_slots",
                        context, deleted_slot
                    ));
                }
            }

            Ok(())
        }
    }

    /// Helper function to simulate insert operation
    fn mock_insert(
        bookkeeping: &mut TagSlotManager,
        tag_range: std::ops::Range<usize>,
        max_capacity: usize,
        context: &str,
    ) -> anyhow::Result<Vec<u32>> {
        let num_slots = tag_range.len();
        let slots = bookkeeping.get_n_empty_slots(num_slots)?;
        bookkeeping.assign_slots_to_tags(tag_range, slots.clone())?;
        bookkeeping.check_invariants(max_capacity, context)?;
        Ok(slots)
    }

    /// Helper function to simulate delete operation
    fn mock_delete(
        bookkeeping: &mut TagSlotManager,
        tag_range: std::ops::Range<usize>,
        max_capacity: usize,
        context: &str,
    ) -> anyhow::Result<Vec<u32>> {
        let slots = bookkeeping.find_slots_by_tags(tag_range.clone())?;
        bookkeeping.mark_tags_deleted(tag_range)?;
        bookkeeping.check_invariants(max_capacity, context)?;
        Ok(slots)
    }

    /// Helper function to simulate replace operation (read-only)
    fn mock_replace(
        bookkeeping: &TagSlotManager,
        tag_range: std::ops::Range<usize>,
        max_capacity: usize,
        context: &str,
    ) -> anyhow::Result<Vec<u32>> {
        let slots = bookkeeping.find_slots_by_tags(tag_range)?;
        bookkeeping.check_invariants(max_capacity, context)?;
        Ok(slots)
    }

    #[test]
    fn test_slot_bookkeeping_integration() {
        let max_capacity = 10;
        let mut bookkeeping = TagSlotManager::new(max_capacity);

        // Initial state check
        bookkeeping
            .check_invariants(max_capacity, "initial state")
            .unwrap();
        assert_eq!(bookkeeping.empty_slots.len(), max_capacity - 1); // Left slot for frozen point
        assert_eq!(bookkeeping.deleted_slots.len(), 0);
        assert_eq!(bookkeeping.tag_to_slot.len(), 0);
        assert_eq!(bookkeeping.slot_to_tag.len(), 0);

        // Test insert operation: request slots and assign to tags
        let insert_slots = mock_insert(
            &mut bookkeeping,
            100..103,
            max_capacity,
            "after first insert",
        )
        .unwrap();
        assert_eq!(insert_slots.len(), 3);
        assert_eq!(insert_slots, vec![0, 1, 2]); // Should get first 3 slots, this depends on the fresh state of the test and the order preservation by VecDeque

        assert_eq!(bookkeeping.empty_slots.len(), max_capacity - 1 - 3); // 6 slots left
        assert_eq!(bookkeeping.tag_to_slot.len(), 3);
        assert_eq!(bookkeeping.slot_to_tag.len(), 3);
        assert_eq!(bookkeeping.tag_to_slot[&100], 0);
        assert_eq!(bookkeeping.tag_to_slot[&101], 1);
        assert_eq!(bookkeeping.tag_to_slot[&102], 2);

        // Test another insert operation
        let insert_slots_2 = mock_insert(
            &mut bookkeeping,
            200..202,
            max_capacity,
            "after second insert",
        )
        .unwrap();
        assert_eq!(insert_slots_2, vec![3, 4]); // Depends on order preservation of VecDeque.

        assert_eq!(bookkeeping.empty_slots.len(), max_capacity - 1 - 5); // 4 slots left
        assert_eq!(bookkeeping.tag_to_slot.len(), 5);

        // Test replace operation: only find slots by tags (no state change)
        let replace_slots =
            mock_replace(&bookkeeping, 100..102, max_capacity, "after replace lookup").unwrap();
        assert_eq!(replace_slots, vec![0, 1]);

        // State should be unchanged after replace lookup
        assert_eq!(bookkeeping.tag_to_slot.len(), 5);
        assert_eq!(bookkeeping.deleted_slots.len(), 0);

        // Test delete operation: find slots and mark them deleted
        let delete_slots = mock_delete(
            &mut bookkeeping,
            100..102,
            max_capacity,
            "after delete marking",
        )
        .unwrap();
        assert_eq!(delete_slots, vec![0, 1]);

        // Tag mappings should still exist, but slots should be marked deleted
        assert_eq!(bookkeeping.tag_to_slot.len(), 5);
        assert_eq!(bookkeeping.deleted_slots.len(), 2);
        assert!(bookkeeping.deleted_slots.contains(&0));
        assert!(bookkeeping.deleted_slots.contains(&1));

        // Test another insert while having deleted slots
        let insert_slots_3 = mock_insert(
            &mut bookkeeping,
            300..301,
            max_capacity,
            "after insert with deleted slots",
        )
        .unwrap();
        assert_eq!(insert_slots_3, vec![5]);

        // Test consolidate operation
        bookkeeping.consolidate();
        bookkeeping
            .check_invariants(max_capacity, "after consolidate")
            .unwrap();

        // After consolidate: deleted slots should be moved to empty_slots
        assert_eq!(bookkeeping.deleted_slots.len(), 0);
        assert_eq!(bookkeeping.tag_to_slot.len(), 4); // 200..202 + 300..301
        assert_eq!(bookkeeping.slot_to_tag.len(), 4);
        assert_eq!(bookkeeping.empty_slots.len(), max_capacity - 1 - 4); // 5 slots available

        // Deleted tag mappings should be removed
        assert!(!bookkeeping.tag_to_slot.contains_key(&100));
        assert!(!bookkeeping.tag_to_slot.contains_key(&101));
        assert!(!bookkeeping.slot_to_tag.contains_key(&0));
        assert!(!bookkeeping.slot_to_tag.contains_key(&1));

        // Remaining mappings should be intact
        assert_eq!(bookkeeping.tag_to_slot[&200], 3);
        assert_eq!(bookkeeping.tag_to_slot[&201], 4);
        assert_eq!(bookkeeping.tag_to_slot[&300], 5);

        // Test getting recycled slots success after consolidate
        let _ = mock_insert(
            &mut bookkeeping,
            400..402,
            max_capacity,
            "after using recycled slots",
        )
        .unwrap();
    }

    #[test]
    fn test_error_conditions() {
        let max_capacity = 5;
        let mut bookkeeping = TagSlotManager::new(max_capacity);

        // Test insufficient empty slots
        let result = bookkeeping.get_n_empty_slots(10);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Insufficient empty slots"));

        // Test mismatched tag range and slots length
        let slots = vec![0, 1];
        let result = bookkeeping.assign_slots_to_tags(100..104, slots); // 4 tags, 2 slots
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not match slots length"));

        // Test finding non-existent tag
        let result = bookkeeping.find_slots_by_tags(999..1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Tag 999 not found"));

        // Test marking non-existent tag as deleted
        let result = bookkeeping.mark_tags_deleted(999..1000);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Tag 999 not found"));
    }

    #[test]
    fn test_complex_interleaved_operations() {
        let max_capacity = 20;
        let mut bookkeeping = TagSlotManager::new(max_capacity);

        // Insert batch 1: tags 1000-1004 (5 slots)
        let slots1 =
            mock_insert(&mut bookkeeping, 1000..1005, max_capacity, "batch 1 insert").unwrap();
        assert_eq!(slots1.len(), 5);

        // Insert batch 2: tags 2000-2002 (3 slots)
        let slots2 =
            mock_insert(&mut bookkeeping, 2000..2003, max_capacity, "batch 2 insert").unwrap();
        assert_eq!(slots2.len(), 3);

        // Delete partial batch 1: tags 1001-1003
        let deleted_slots = mock_delete(
            &mut bookkeeping,
            1001..1004,
            max_capacity,
            "partial delete batch 1",
        )
        .unwrap();
        assert_eq!(deleted_slots.len(), 3);

        // Replace operation on remaining batch 1
        let replace_slots =
            mock_replace(&bookkeeping, 1000..1001, max_capacity, "replace operation").unwrap();
        assert_eq!(replace_slots.len(), 1);

        // Insert batch 3 while having deleted slots
        let slots3 = mock_insert(
            &mut bookkeeping,
            3000..3002,
            max_capacity,
            "batch 3 insert with deleted",
        )
        .unwrap();
        assert_eq!(slots3.len(), 2);

        // Delete entire batch 2
        let deleted_slots2 = mock_delete(
            &mut bookkeeping,
            2000..2003,
            max_capacity,
            "delete entire batch 2",
        )
        .unwrap();
        assert_eq!(deleted_slots2.len(), 3);

        // Consolidate
        bookkeeping.consolidate();
        bookkeeping
            .check_invariants(max_capacity, "after consolidation")
            .unwrap();

        // Verify final state
        assert_eq!(bookkeeping.deleted_slots.len(), 0);
        assert_eq!(bookkeeping.tag_to_slot.len(), 4); // 1000, 1004, 3000, 3001
        assert_eq!(bookkeeping.slot_to_tag.len(), 4);

        // Insert using recycled slots
        let recycled_slots = mock_insert(
            &mut bookkeeping,
            4000..4006,
            max_capacity,
            "final insert with recycled",
        )
        .unwrap();
        assert_eq!(recycled_slots.len(), 6);
    }
}
