/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};
use std::collections::HashSet;
use std::marker::PhantomData;

use super::Neighbor;

/// Shared trait for type the generic `I` parameter used by the
/// `NeighborPeriorityQueue`.
pub trait NeighborPriorityQueueIdType:
    Default + Eq + Clone + Copy + std::fmt::Debug + std::fmt::Display + std::hash::Hash + Send + Sync
{
}

/// Any type that implements all the individual requirements for
/// `NeighborPriorityQueueIdType` implements the full trait.
impl<T> NeighborPriorityQueueIdType for T where
    T: Default + Eq + Clone + Copy + std::fmt::Debug + std::fmt::Display + std::hash::Hash + Send + Sync
{
}

/// Trait defining the interface for a neighbor priority queue.
///
/// This trait abstracts the core functionality of a priority queue that manages
/// neighbors ordered by distance, supporting both fixed-size and resizable queues.
pub trait NeighborQueue<I: NeighborPriorityQueueIdType>: std::fmt::Debug + Send + Sync {
    /// The iterator type returned by `iter()`.
    type Iter<'a>: ExactSizeIterator<Item = Neighbor<I>> + Send + Sync
    where
        Self: 'a,
        I: 'a;

    /// Insert a neighbor into the priority queue.
    fn insert(&mut self, nbr: Neighbor<I>);

    /// Get the neighbor at the specified index.
    fn get(&self, index: usize) -> Neighbor<I>;

    /// Get the closest unvisited neighbor.
    fn closest_notvisited(&mut self) -> Neighbor<I>;

    /// Check whether there is an unvisited node.
    fn has_notvisited_node(&self) -> bool;

    /// Get the current size of the priority queue.
    fn size(&self) -> usize;

    /// Get the capacity of the priority queue.
    fn capacity(&self) -> usize;

    /// Get the current search parameter L.
    fn search_l(&self) -> usize;

    /// Clear the priority queue, resetting size and cursor.
    fn clear(&mut self);

    /// Return an iterator over the best candidates.
    fn iter(&self) -> Self::Iter<'_>;

    /// Return the first node (by distance order) that is not visited and not in `submitted`,
    /// scanning positions 0..min(size, search_l). Does not modify any state.
    fn peek_best_unsubmitted(&self, _submitted: &HashSet<I>) -> Option<Neighbor<I>> {
        None
    }

    /// Find the node with matching `id`, mark it visited, and advance the cursor if needed.
    /// Returns true if found and marked, false otherwise.
    fn mark_visited_by_id(&mut self, _id: &I) -> bool {
        false
    }
}

/// Neighbor priority Queue based on the distance to the query node
///
/// This queue two collections under the hood, ids and distances, instead of a single
/// collection of neighbors in order support SIMD processing.
///
/// Performance is critical for this data structure - please benchmark any changes
#[derive(Debug, Clone)]
pub struct NeighborPriorityQueue<I: NeighborPriorityQueueIdType> {
    /// The size of the priority queue
    size: usize,

    /// The capacity of the priority queue
    capacity: usize,

    /// The current notvisited neighbor whose distance is smallest among all notvisited neighbor
    cursor: usize,

    /// The neighbor (id, visited) collection.
    /// These are stored together to make inserts cheaper.
    id_visiteds: Vec<(I, bool)>,

    /// The neighbor distance collection
    distances: Vec<f32>,

    /// The flag to indicate whether the queue has unbounded capacity/will be resized on insertion
    auto_resizable: bool,

    // Search parameter L: search stops once we have explored the L best candidates.
    search_param_l: usize,
}

impl<I: NeighborPriorityQueueIdType> NeighborPriorityQueue<I> {
    /// Create NeighborPriorityQueue with capacity.
    ///
    /// This will implicitly set `search_param_l` to the provided capacity.
    pub fn new(search_param_l: usize) -> Self {
        Self {
            size: 0,
            capacity: search_param_l,
            cursor: 0,
            id_visiteds: Vec::with_capacity(search_param_l),
            distances: Vec::with_capacity(search_param_l),
            auto_resizable: false,
            search_param_l,
        }
    }

    //  Create a auto resizable(unbounded capacity) NeighborPriorityQueue with the initial capacity as search parameter L
    pub fn auto_resizable_with_search_param_l(search_param_l: usize) -> Self {
        Self {
            size: 0,
            capacity: search_param_l,
            cursor: 0,
            id_visiteds: Vec::with_capacity(search_param_l),
            distances: Vec::with_capacity(search_param_l),
            auto_resizable: true,
            search_param_l,
        }
    }

    /// Inserts item with order.
    ///
    /// There are two behaviors based on resizable queue or not.
    /// 1) If fixed size queue then the item will be dropped if queue is full / already exist in queue / it has a greater distance than the last item.
    /// 2) If resizable queue then the capacity of the queue will be increased by 50%(1.5x) if the queue is full.
    ///
    /// The set cursor that is used to pop() the next item will be set to the lowest index of an unvisited item.
    /// Due to the performance sensitiveness of this function - we don't check for uniqueness of the item.
    /// Inserting the same item twice will cause undefined behavior.
    pub fn insert(&mut self, nbr: Neighbor<I>) {
        self.dbgassert_unique_insert(nbr.id);

        if self.auto_resizable {
            if self.size == self.capacity {
                self.reserve(1.max(self.capacity >> 1)); // 1.5x capacity
            }
        } else if self.size == self.capacity && self.get_unchecked(self.size - 1) < nbr {
            return;
        }

        let insert_idx = if self.size > 0 {
            self.get_lower_bound(&nbr)
        } else {
            0
        };

        if self.size == self.capacity {
            self.id_visiteds.truncate(self.size - 1);
            self.distances.truncate(self.size - 1);
            self.size -= 1;
        }

        self.id_visiteds.insert(insert_idx, (nbr.id, false));
        self.distances.insert(insert_idx, nbr.distance);

        self.size += 1;

        debug_assert!(self.size == self.id_visiteds.len());
        debug_assert!(self.size == self.distances.len());

        if insert_idx < self.cursor {
            self.cursor = insert_idx;
        }
    }

    /// Extracts the first min(L, size_of_queue) best candidates from the priority queue moving
    /// them to the result array and returns the count of extracted elements. The rest of the
    /// candidates are shifted to the beginning of the array and the size and capacity are
    /// updated accordingly.
    pub fn extract_best_l_candidates(&mut self, result: &mut [Neighbor<I>]) -> usize {
        let extract_size = self.search_param_l.min(self.size);

        // Copy the first L best candidates to the result vector
        for (i, res) in result.iter_mut().enumerate().take(extract_size) {
            *res = Neighbor::new(self.id_visiteds[i].0, self.distances[i]);
        }

        // Remove the first L best candidates from the priority queue
        self.id_visiteds.drain(0..extract_size);
        self.distances.drain(0..extract_size);

        // Update the size and cursor of the priority queue
        self.size -= extract_size;
        self.cursor = 0;

        extract_size
    }

    /// Drain candidates from the front, signaling that they have been consumed.
    pub fn drain_best(&mut self, count: usize) {
        let count = count.min(self.size);
        self.id_visiteds.drain(0..count);
        self.distances.drain(0..count);
        self.size -= count;
        self.cursor = 0;
    }

    /// Return an immutable iterator over the best candidates in the priority queue.
    ///
    /// The length of the returned iterator will be the minimum of current size and the
    /// requested `search_param_l`.
    ///
    /// The full type of the returned iterator should be ignored (though it can be relied
    /// on implemented `ExactSizeIterator`) to allow future modifications to the
    /// implemention of `NeighborPriorityQueue`.
    pub fn iter(&self) -> BestCandidatesIterator<'_, I, Self> {
        let sz = self.search_param_l.min(self.size);
        BestCandidatesIterator::new(sz, self)
    }

    /// Remove a neighbor from the priority queue.
    /// Returns true if the neighbor was found and removed, false otherwise.
    pub fn remove(&mut self, nbr: Neighbor<I>) -> bool {
        if self.size == 0 {
            return false;
        }

        // Use get_lower_bound to find where the neighbor with this distance would be
        let index = self.get_lower_bound(&nbr);

        // Check if we found the exact neighbor (both id and distance must match)
        if index < self.size && self.get_unchecked(index).id == nbr.id {
            // Remove the neighbor from both collections
            self.id_visiteds.remove(index);
            self.distances.remove(index);
            self.size -= 1;

            // Adjust cursor if necessary
            if index < self.cursor && self.cursor > 0 {
                self.cursor -= 1;
            }

            debug_assert!(self.size == self.id_visiteds.len());
            debug_assert!(self.size == self.distances.len());

            return true;
        }

        false
    }

    /// Get the lower bound of the neighbor - the index of the first neighbor
    /// that has a distance greater than or equal to the target neighbor
    /// PERFORMANCE: This function is performance critical - please benchmark any changes
    fn get_lower_bound(&mut self, nbr: &Neighbor<I>) -> usize {
        diskann_wide::alias!(f32s = f32x8);
        let target = f32s::splat(diskann_wide::ARCH, nbr.distance);

        let mut index = 0;

        // Check 16 items at a time
        while index + 16 <= self.size {
            let search =
                unsafe { f32s::load_simd(diskann_wide::ARCH, self.distances.as_ptr().add(index)) };
            let offset1 = search.ge_simd(target).first();
            let search = unsafe {
                f32s::load_simd(diskann_wide::ARCH, self.distances.as_ptr().add(index + 8))
            };
            let offset2 = search.ge_simd(target).first();

            match (offset1, offset2) {
                (Some(offset), _) => return index + offset,
                (None, Some(offset)) => return index + 8 + offset,
                _ => (),
            }

            index += 16;
        }

        // Check the remaining items
        if index + 8 <= self.size {
            let search =
                unsafe { f32s::load_simd(diskann_wide::ARCH, self.distances.as_ptr().add(index)) };
            let offset = search.ge_simd(target).first();
            if let Some(offset) = offset {
                return index + offset;
            }
            index += 8;
        }

        if index < self.size {
            let search = unsafe {
                f32s::load_simd_first(
                    diskann_wide::ARCH,
                    self.distances.as_ptr().add(index),
                    self.size - index,
                )
            };
            let offset = search.ge_simd(target).first();
            if let Some(offset) = offset {
                return index + offset;
            }
        }

        self.size
    }

    /// Get the neighbor at index - SAFETY: index must be less than size
    fn get_unchecked(&self, index: usize) -> Neighbor<I> {
        debug_assert!(index < self.size);
        let id = unsafe { self.id_visiteds.get_unchecked(index).0 };
        let distance = unsafe { *self.distances.get_unchecked(index) };
        Neighbor::new(id, distance)
    }

    // Get the neighbor at index.
    pub fn get(&self, index: usize) -> Neighbor<I> {
        assert!(index < self.size, "index out of bounds");
        self.get_unchecked(index)
    }

    /// Get the closest and notvisited neighbor
    pub fn closest_notvisited(&mut self) -> Neighbor<I> {
        let current = self.cursor;
        self.set_visited(current, true);

        // Look for the next notvisited neighbor
        self.cursor += 1;
        while self.cursor < self.size && self.get_visited(self.cursor) {
            self.cursor += 1;
        }
        self.get_unchecked(current)
    }

    /// Whether there is notvisited node or not
    pub fn has_notvisited_node(&self) -> bool {
        self.cursor < self.search_param_l.min(self.size)
    }

    /// Get the size of the NeighborPriorityQueue
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the capacity of the NeighborPriorityQueue
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current search L.
    pub fn search_l(&self) -> usize {
        self.search_param_l
    }

    /// Adjust the size of the queue for a new `search_param_l` value.
    ///
    /// If `search_param_l` is less than the current size, the contents of the buffer will
    /// be truncated.
    pub fn reconfigure(&mut self, search_param_l: usize) {
        self.search_param_l = search_param_l;
        if search_param_l < self.size {
            self.id_visiteds.truncate(search_param_l);
            self.distances.truncate(search_param_l);
            self.size = search_param_l;
            self.cursor = self.cursor.min(search_param_l);
        } else if search_param_l > self.capacity {
            // Grow the backing store.
            let additional = search_param_l - self.size;
            self.id_visiteds.reserve(additional);
            self.distances.reserve(additional);
        }
        self.capacity = search_param_l;
    }

    /// Reserve additional space in the buffer.
    ///
    /// Note that this function probably does not do what you want and it thus marked
    /// as private. It should only be called when requesting additional space for dynamic
    /// resizing.
    ///
    /// Most of the time, you want `reconfigure`.
    fn reserve(&mut self, additional: usize) {
        self.id_visiteds.reserve(additional);
        self.distances.reserve(additional);
        self.capacity += additional;
    }

    /// Set size (and cursor) to 0. This must be called to reset the queue when reusing
    /// between searched.
    pub fn clear(&mut self) {
        self.id_visiteds.clear();
        self.distances.clear();
        self.size = 0;
        self.cursor = 0;
    }

    fn set_visited(&mut self, index: usize, flag: bool) {
        // SAFETY: index must be less than size
        assert!(index <= self.size);
        assert!(self.size <= self.capacity);
        unsafe { self.id_visiteds.get_unchecked_mut(index) }.1 = flag;
    }

    fn get_visited(&self, index: usize) -> bool {
        // SAFETY: index must be less than size
        assert!(index < self.size);
        unsafe { self.id_visiteds.get_unchecked(index).1 }
    }

    /// Return whether or not the queue is auto resizeable (for paged search).
    pub fn is_resizable(&self) -> bool {
        self.auto_resizable
    }

    /// Check if the queue is full.
    pub fn is_full(&self) -> bool {
        !self.auto_resizable && self.size == self.capacity
    }

    #[cfg(debug_assertions)]
    fn dbgassert_unique_insert(&self, id: I) {
        for i in 0..self.size {
            debug_assert!(
                self.id_visiteds[i].0 != id,
                "Neighbor with ID {} already exists in the priority queue",
                id
            );
        }
    }

    #[cfg(not(debug_assertions))]
    fn dbgassert_unique_insert(&self, _id: I) {}

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns `false`.
    /// This method operates in place, visiting each element exactly once in the
    /// original order, and preserves the order of the retained elements.
    ///
    /// After this operation, the cursor is reset to 0 and all retained elements
    /// are marked as unvisited, since the compaction invalidates the previous
    /// visited state.
    ///
    /// # Arguments
    /// * `f` - A predicate that returns `true` for items to keep
    #[cfg(feature = "experimental_diversity_search")]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Neighbor<I>) -> bool,
    {
        if self.size == 0 {
            return;
        }

        let mut write_idx = 0;

        // Iterate through and compact in-place
        for read_idx in 0..self.size {
            // SAFETY: read_idx is guaranteed to be < self.size by the loop bounds
            let neighbor = self.get_unchecked(read_idx);

            // If this item should be kept, move it to write position
            if f(&neighbor) {
                if write_idx != read_idx {
                    self.id_visiteds[write_idx] = self.id_visiteds[read_idx];
                    self.distances[write_idx] = self.distances[read_idx];
                }
                // Reset visited state since compaction invalidates previous state
                self.id_visiteds[write_idx].1 = false;
                write_idx += 1;
            }
        }

        // Truncate the vectors and update size and cursor
        self.truncate(write_idx);
    }

    /// Shortens the queue, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than or equal to the queue's current length, this has no effect.
    /// The cursor is always reset to 0 since truncation invalidates the cursor state.
    ///
    /// # Arguments
    /// * `len` - The new length of the queue
    #[cfg(feature = "experimental_diversity_search")]
    pub fn truncate(&mut self, len: usize) {
        let new_size = len;
        if new_size < self.size {
            self.id_visiteds.truncate(new_size);
            self.distances.truncate(new_size);
            self.size = new_size;
            self.cursor = 0;
        }
    }

    /// Return the first node that is not visited and not in `submitted`,
    /// scanning positions 0..min(size, search_param_l). Does not modify any state.
    pub fn peek_best_unsubmitted(&self, submitted: &HashSet<I>) -> Option<Neighbor<I>> {
        let limit = self.search_param_l.min(self.size);
        for i in 0..limit {
            let (id, visited) = self.id_visiteds[i];
            if !visited && !submitted.contains(&id) {
                return Some(Neighbor::new(id, self.distances[i]));
            }
        }
        None
    }

    /// Find the node with matching `id`, mark it visited, and advance the cursor if needed.
    /// Returns true if found and marked, false otherwise.
    pub fn mark_visited_by_id(&mut self, id: &I) -> bool {
        for i in 0..self.size {
            if self.id_visiteds[i].0 == *id {
                self.id_visiteds[i].1 = true;
                // If the cursor was pointing at this node, advance past visited nodes
                if self.cursor == i {
                    self.cursor += 1;
                    while self.cursor < self.size && self.get_visited(self.cursor) {
                        self.cursor += 1;
                    }
                }
                return true;
            }
        }
        false
    }
}

impl<I: NeighborPriorityQueueIdType> NeighborQueue<I> for NeighborPriorityQueue<I> {
    type Iter<'a>
        = BestCandidatesIterator<'a, I, Self>
    where
        Self: 'a,
        I: 'a;

    fn insert(&mut self, nbr: Neighbor<I>) {
        self.insert(nbr)
    }

    fn get(&self, index: usize) -> Neighbor<I> {
        self.get(index)
    }

    fn closest_notvisited(&mut self) -> Neighbor<I> {
        self.closest_notvisited()
    }

    fn has_notvisited_node(&self) -> bool {
        self.has_notvisited_node()
    }

    fn size(&self) -> usize {
        self.size()
    }

    fn capacity(&self) -> usize {
        self.capacity()
    }

    fn search_l(&self) -> usize {
        self.search_l()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.iter()
    }

    fn peek_best_unsubmitted(&self, submitted: &HashSet<I>) -> Option<Neighbor<I>> {
        self.peek_best_unsubmitted(submitted)
    }

    fn mark_visited_by_id(&mut self, id: &I) -> bool {
        self.mark_visited_by_id(id)
    }
}

/// Enable the following syntax for iteration over the valid elements in the queue.
/// ```
/// use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
/// let mut queue = NeighborPriorityQueue::<u32>::new(2);
/// queue.insert(Neighbor::new(1, 3.0));
/// queue.insert(Neighbor::new(2, 2.0));
/// queue.insert(Neighbor::new(3, 1.0));
/// for i in &queue {
///    println!("Neighbor = {:?}", i);
/// }
/// ```
impl<'a, I> IntoIterator for &'a NeighborPriorityQueue<I>
where
    I: NeighborPriorityQueueIdType,
{
    type Item = Neighbor<I>;
    type IntoIter = BestCandidatesIterator<'a, I, NeighborPriorityQueue<I>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct BestCandidatesIterator<'a, I, Q>
where
    I: NeighborPriorityQueueIdType,
    Q: NeighborQueue<I> + ?Sized,
{
    cursor: usize,
    size: usize,
    queue: &'a Q,
    _phantom: PhantomData<I>,
}

impl<'a, I, Q> BestCandidatesIterator<'a, I, Q>
where
    I: NeighborPriorityQueueIdType,
    Q: NeighborQueue<I> + ?Sized,
{
    pub fn new(size: usize, queue: &'a Q) -> Self {
        Self {
            cursor: 0,
            size,
            queue,
            _phantom: PhantomData,
        }
    }
}

impl<I, Q> Iterator for BestCandidatesIterator<'_, I, Q>
where
    I: NeighborPriorityQueueIdType,
    Q: NeighborQueue<I> + ?Sized,
{
    type Item = Neighbor<I>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.size {
            let result = self.queue.get(self.cursor);
            self.cursor += 1;
            Some(result)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.size - self.cursor;
        (remaining, Some(remaining))
    }
}

impl<I, Q> ExactSizeIterator for BestCandidatesIterator<'_, I, Q>
where
    I: NeighborPriorityQueueIdType,
    Q: NeighborQueue<I> + ?Sized,
{
}

#[cfg(test)]
mod neighbor_priority_queue_test {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_reconfigure() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);

        queue.reconfigure(20);
        assert_eq!(queue.capacity(), 20);
        assert_eq!(queue.search_l(), 20);

        queue.reconfigure(20);
        assert_eq!(queue.capacity(), 20);
        assert_eq!(queue.search_l(), 20);

        queue.reconfigure(10);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);
    }

    #[test]
    fn test_insert() {
        let mut queue = NeighborPriorityQueue::new(3);
        assert_eq!(queue.size(), 0);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        queue.insert(Neighbor::new(3, 0.9));
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(2).id, 1);
        queue.insert(Neighbor::new(4, 2.0)); // should be dropped as queue is full and distance is greater than last item

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 2); // node id in queue should be [2,3,1]
        assert_eq!(queue.get(1).id, 3);
        assert_eq!(queue.get(2).id, 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_repeat_insert_panics() {
        let mut queue = NeighborPriorityQueue::new(10);
        assert_eq!(queue.size(), 0);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(1, 0.5));
    }

    #[test]
    fn test_is_sorted() {
        let mut queue = NeighborPriorityQueue::new(40);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let data: Vec<f32> = (0..60).map(|_| rng.random_range(-1.0..1.0)).collect();
        for i in 0..60 {
            queue.insert(Neighbor::new(i, data[i as usize]));
        }

        for i in 0..39 {
            assert!(queue.get(i).distance <= queue.get(i + 1).distance);
        }
    }

    #[test]
    fn test_index() {
        let mut queue = NeighborPriorityQueue::new(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        assert_eq!(queue.get(0).id, 2);
        assert_eq!(queue.get(0).distance, 0.5);
    }

    #[test]
    fn test_visit() {
        let mut queue = NeighborPriorityQueue::new(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert!(!queue.get_visited(0));
        queue.insert(Neighbor::new(3, 1.5)); // node id in queue should be [2,1,3]
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 2);
        assert_eq!(nbr.distance, 0.5);
        assert!(queue.get_visited(0)); // super unfortunate test. We know based on above id 2 should be 0th index
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 1);
        assert_eq!(nbr.distance, 1.0);
        assert!(queue.get_visited(1));
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 3);
        assert_eq!(nbr.distance, 1.5);
        assert!(queue.get_visited(2));
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_clear_queue() {
        let mut queue = NeighborPriorityQueue::new(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        assert!(queue.has_notvisited_node());
        queue.clear();
        assert_eq!(queue.size(), 0);
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_reserve() {
        let mut queue = NeighborPriorityQueue::<u32>::new(5);
        queue.reconfigure(10);
        assert_eq!(queue.id_visiteds.len(), 0);
        assert_eq!(queue.distances.len(), 0);
        assert_eq!(queue.capacity, 10);
    }

    #[test]
    fn test_set_capacity() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);
        queue.reconfigure(5);
        assert_eq!(queue.capacity, 5);
        assert_eq!(queue.id_visiteds.len(), 0);
        assert_eq!(queue.distances.len(), 0);

        queue.reconfigure(11);
        assert_eq!(queue.capacity, 11);
    }

    #[test]
    fn test_resizable_with_initial_capacity() {
        let resizable_queue = NeighborPriorityQueue::<u32>::auto_resizable_with_search_param_l(10);

        assert_eq!(resizable_queue.capacity(), 10);
        assert_eq!(resizable_queue.size(), 0);
        assert!(resizable_queue.auto_resizable);
        assert_eq!(resizable_queue.id_visiteds.len(), 0);
        assert_eq!(resizable_queue.distances.len(), 0);
    }

    #[test]
    fn test_insert_on_full_queue() {
        let mut fixed_queue = NeighborPriorityQueue::new(5);
        fixed_queue.insert(Neighbor::new(5, 0.5));
        fixed_queue.insert(Neighbor::new(2, 0.2));
        fixed_queue.insert(Neighbor::new(4, 0.4));
        fixed_queue.insert(Neighbor::new(1, 0.1));
        fixed_queue.insert(Neighbor::new(3, 0.3));

        // this one should be dropped
        fixed_queue.insert(Neighbor::new(6, 0.6));
        assert_eq!(fixed_queue.get(4).id, 5);
        // verify capacity and size are unchanged
        assert_eq!(fixed_queue.capacity(), 5);
        assert_eq!(fixed_queue.size(), 5);

        // this one pushes out id=5
        fixed_queue.insert(Neighbor::new(35, 0.35));
        assert_eq!(fixed_queue.get(4).id, 4);
        // verify capacity and size are unchanged
        assert_eq!(fixed_queue.capacity(), 5);
        assert_eq!(fixed_queue.size(), 5);
    }

    #[test]
    fn test_reconfigure_after_insert() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(5, 0.5));
        queue.insert(Neighbor::new(2, 0.2));
        queue.insert(Neighbor::new(4, 0.4));
        queue.insert(Neighbor::new(1, 0.1));
        queue.insert(Neighbor::new(3, 0.3));

        let _: Neighbor<u32> = queue.closest_notvisited();
        let _: Neighbor<u32> = queue.closest_notvisited();
        let _: Neighbor<u32> = queue.closest_notvisited();
        let _: Neighbor<u32> = queue.closest_notvisited();

        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.size(), 5);
        assert_eq!(queue.search_l(), 5);
        assert_eq!(queue.cursor, 4);

        queue.reconfigure(3);

        assert_eq!(queue.capacity(), 3);
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.search_l(), 3);
        assert_eq!(queue.cursor, 3);

        queue.reconfigure(5);

        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.search_l(), 5);
        assert_eq!(queue.cursor, 3);
    }

    #[test]
    fn test_insert_on_resizable_queue() {
        let mut resizable_queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(2);

        resizable_queue.insert(Neighbor::new(1, 1.0));
        resizable_queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(resizable_queue.size(), 2);
        assert_eq!(resizable_queue.capacity(), 2);

        resizable_queue.insert(Neighbor::new(3, 0.9));
        assert_eq!(resizable_queue.size(), 3);
        assert_eq!(resizable_queue.capacity(), 3);

        resizable_queue.insert(Neighbor::new(4, 2.0));
        assert_eq!(resizable_queue.size(), 4);
        assert_eq!(resizable_queue.capacity(), 4);
    }

    #[test]
    fn test_extract_best_l_candidates() {
        let mut queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 0.1));
        queue.insert(Neighbor::new(4, 5.0));
        queue.insert(Neighbor::new(5, 0.2));

        let mut result = vec![Neighbor::default(); 3];
        queue.extract_best_l_candidates(&mut result);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].id, 3);
        assert_eq!(result[1].id, 5);
        assert_eq!(result[2].id, 2);
        assert_eq!(queue.size(), 2);
        assert_eq!(queue.cursor, 0);
    }

    #[test]
    fn test_iter() {
        let mut queue = NeighborPriorityQueue::<u32>::auto_resizable_with_search_param_l(3);
        assert_eq!(queue.size(), 0);

        let mut iter = (&queue).into_iter(); // use the `&queue` syntax to test trait implementaiton.

        // Require that the iterator implements `ExactSizedIterator`.
        let iter_dyn: &mut dyn ExactSizeIterator<Item = Neighbor<u32>> = &mut iter;
        assert_eq!(iter_dyn.len(), 0);
        assert!(iter_dyn.next().is_none());

        // Iterator should now have length one.
        queue.insert(Neighbor::new(1, 1.0));
        assert_eq!(queue.size(), 1);
        let mut iter = queue.iter();
        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next().unwrap().id, 1);
        assert!(iter.next().is_none());

        // Queue up many more and try again.
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 0.1));
        queue.insert(Neighbor::new(4, 5.0));
        queue.insert(Neighbor::new(5, 0.2));
        let mut iter = queue.iter();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next().unwrap().id, 3);
        assert_eq!(iter.next().unwrap().id, 5);
        assert_eq!(iter.next().unwrap().id, 2);
        assert!(iter.next().is_none());

        // Test iteration syntax.
        for (i, neighbor) in (&queue).into_iter().enumerate() {
            assert_eq!(neighbor.id, queue.get(i).id);
        }

        // After clearing - the view iterator should correctly be empty.
        queue.clear();
        let mut iter = queue.iter();
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_has_notvisited_node_fixed_size_queue() {
        let mut queue = NeighborPriorityQueue::new(3);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 2, /*search_param_l=*/ 3, /*cursor=*/ 0,
        );
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert!(queue.has_notvisited_node());
        queue.closest_notvisited();
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 2, /*search_param_l=*/ 3, /*cursor=*/ 2,
        );
        assert!(!queue.has_notvisited_node());

        queue.insert(Neighbor::new(3, 0.1));
        queue.insert(Neighbor::new(4, 5.0));
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 3, /*search_param_l=*/ 3, /*cursor=*/ 0,
        );
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 3, /*search_param_l=*/ 3, /*cursor=*/ 3,
        );
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_has_notvisited_node_fixed_size_queue_with_mannual_resize() {
        let mut queue = NeighborPriorityQueue::new(3);
        queue.reconfigure(5);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 0.1));
        queue.insert(Neighbor::new(4, 5.0));
        queue.insert(Neighbor::new(5, 0.2));
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 5, /*search_param_l=*/ 5, /*cursor=*/ 0,
        );

        for i in 1..=5 {
            assert!(queue.has_notvisited_node());
            queue.closest_notvisited();
            assert_queue_size_search_param_l_cursor(
                &queue, /*size=*/ 5, /*search_param_l=*/ 5, /*cursor=*/ i,
            );
        }

        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_has_notvisited_auto_resizable_queue() {
        let mut queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(3);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 2, /*search_param_l=*/ 3, /*cursor=*/ 0,
        );
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert!(queue.has_notvisited_node());
        queue.closest_notvisited();
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 2, /*search_param_l=*/ 3, /*cursor=*/ 2,
        );
        assert!(!queue.has_notvisited_node());

        queue.insert(Neighbor::new(3, 0.1));
        queue.insert(Neighbor::new(4, 5.0));
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 4, /*search_param_l=*/ 3, /*cursor=*/ 0,
        );
        assert!(queue.has_notvisited_node());

        queue.closest_notvisited();
        assert_queue_size_search_param_l_cursor(
            &queue, /*size=*/ 4, /*search_param_l=*/ 3, /*cursor=*/ 3,
        );
        assert!(!queue.has_notvisited_node());
    }

    fn assert_queue_size_search_param_l_cursor(
        queue: &NeighborPriorityQueue<u32>,
        size: usize,
        search_param_l: usize,
        cursor: usize,
    ) {
        assert_eq!(queue.size(), size);
        assert_eq!(queue.search_param_l, search_param_l);
        assert_eq!(queue.cursor, cursor);
    }

    #[cfg(not(miri))]
    #[test]
    fn insertion_is_in_sorted_order() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let a: Vec<f32> = (0..100)
            .map(|_| rng.random_range(-1000000.0..1000000.0))
            .collect();
        for i in 0..100usize {
            let capacity = i + 1;
            let mut queue = NeighborPriorityQueue::new(capacity);
            for (j, &v) in a.iter().enumerate() {
                queue.insert(Neighbor::new(j as u32, v));
            }
            for j in 0..capacity - 1 {
                assert!(queue.get(j).distance <= queue.get(j + 1).distance);
            }
        }
    }

    // Tests for the NeighborQueue trait implementation
    #[test]
    fn test_trait_implementation_basic_operations() {
        // Note: With GAT (generic associated types), NeighborQueue is not dyn-compatible,
        // so we test using the concrete type through the trait methods
        let mut queue = NeighborPriorityQueue::new(5);

        // Test initial state
        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.search_l(), 5);
        assert!(!queue.has_notvisited_node());

        // Test insert and basic accessors
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));

        assert_eq!(queue.size(), 3);
        assert!(queue.has_notvisited_node());

        // Test get - should be sorted by distance
        assert_eq!(queue.get(0).id, 2); // distance 0.5
        assert_eq!(queue.get(1).id, 1); // distance 1.0
        assert_eq!(queue.get(2).id, 3); // distance 1.5

        // Test closest_notvisited
        let closest = queue.closest_notvisited();
        assert_eq!(closest.id, 2);
        assert_eq!(closest.distance, 0.5);

        // Test clear
        queue.clear();
        assert_eq!(queue.size(), 0);
        assert!(!queue.has_notvisited_node());

        // Test iter through the trait
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        let mut iter = NeighborQueue::iter(&queue);
        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next().unwrap().id, 2);
        assert_eq!(iter.next().unwrap().id, 1);
    }

    #[test]
    fn test_trait_implementation_drain() {
        // Use concrete type instead of trait object since drain_best is not in trait
        let mut queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(3);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.1));
        queue.insert(Neighbor::new(5, 2.0));

        assert_eq!(queue.size(), 5);

        // Test drain_best (called on concrete type, not through trait)
        queue.drain_best(3);
        assert_eq!(queue.size(), 2);
        // After draining the 3 best, the remaining should be the worse ones
        assert_eq!(queue.get(0).id, 3); // distance 1.5
        assert_eq!(queue.get(1).id, 5); // distance 2.0
    }

    #[test]
    fn test_trait_implementation_reconfigure() {
        // Use concrete type instead of trait object since reconfigure is not in trait
        let mut queue = NeighborPriorityQueue::new(5);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));

        assert_eq!(queue.capacity(), 5);
        assert_eq!(queue.search_l(), 5);
        assert_eq!(queue.size(), 3);

        // Reconfigure to smaller capacity (called on concrete type, not through trait)
        queue.reconfigure(2);
        assert_eq!(queue.capacity(), 2);
        assert_eq!(queue.search_l(), 2);
        assert_eq!(queue.size(), 2); // Should be truncated

        // Reconfigure to larger capacity
        queue.reconfigure(10);
        assert_eq!(queue.capacity(), 10);
        assert_eq!(queue.search_l(), 10);
        assert_eq!(queue.size(), 2); // Size should remain the same
    }

    #[test]
    fn test_trait_polymorphism() {
        // Test that we can use different queue types through the trait
        fn test_queue_operations<T: NeighborQueue<u32>>(mut queue: T) {
            queue.insert(Neighbor::new(1, 1.0));
            queue.insert(Neighbor::new(2, 0.5));

            assert_eq!(queue.size(), 2);
            assert!(queue.has_notvisited_node());

            let closest = queue.closest_notvisited();
            assert_eq!(closest.id, 2);
        }

        // Test with regular queue
        let fixed_queue = NeighborPriorityQueue::new(5);
        test_queue_operations(fixed_queue);

        // Test with auto-resizable queue
        let resizable_queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(5);
        test_queue_operations(resizable_queue);
    }

    #[test]
    fn test_remove() {
        let mut queue = NeighborPriorityQueue::new(10);

        // Test removing from empty queue
        assert!(!queue.remove(Neighbor::new(1, 1.0)));

        // Add some neighbors
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));
        queue.insert(Neighbor::new(5, 2.0));

        assert_eq!(queue.size(), 5);
        // Queue should be sorted: [4(0.3), 2(0.5), 1(1.0), 3(1.5), 5(2.0)]

        // Test removing an element from the middle
        assert!(queue.remove(Neighbor::new(1, 1.0)));
        assert_eq!(queue.size(), 4);
        assert_eq!(queue.get(0).id, 4); // 0.3
        assert_eq!(queue.get(1).id, 2); // 0.5
        assert_eq!(queue.get(2).id, 3); // 1.5
        assert_eq!(queue.get(3).id, 5); // 2.0

        // Test removing the first element
        assert!(queue.remove(Neighbor::new(4, 0.3)));
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 2); // 0.5
        assert_eq!(queue.get(1).id, 3); // 1.5
        assert_eq!(queue.get(2).id, 5); // 2.0

        // Test removing the last element
        assert!(queue.remove(Neighbor::new(5, 2.0)));
        assert_eq!(queue.size(), 2);
        assert_eq!(queue.get(0).id, 2); // 0.5
        assert_eq!(queue.get(1).id, 3); // 1.5

        // Test removing non-existent neighbor (wrong id)
        assert!(!queue.remove(Neighbor::new(99, 0.5)));
        assert_eq!(queue.size(), 2);

        // Test removing non-existent neighbor (wrong distance)
        assert!(!queue.remove(Neighbor::new(2, 99.0)));
        assert_eq!(queue.size(), 2);

        // Remove remaining elements
        assert!(queue.remove(Neighbor::new(2, 0.5)));
        assert_eq!(queue.size(), 1);
        assert!(queue.remove(Neighbor::new(3, 1.5)));
        assert_eq!(queue.size(), 0);

        // Try removing from empty queue again
        assert!(!queue.remove(Neighbor::new(1, 1.0)));
    }

    #[test]
    fn test_remove_with_cursor() {
        let mut queue = NeighborPriorityQueue::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));

        // Queue: [4(0.3), 2(0.5), 1(1.0), 3(1.5)]

        // Visit some nodes to advance cursor
        queue.closest_notvisited(); // Visits 4, cursor = 1
        queue.closest_notvisited(); // Visits 2, cursor = 2

        assert_eq!(queue.cursor, 2);

        // Remove an element before the cursor
        assert!(queue.remove(Neighbor::new(4, 0.3)));
        // Cursor should be adjusted down
        assert_eq!(queue.cursor, 1);
        assert_eq!(queue.size(), 3);

        // Remove an element at or after the cursor
        assert!(queue.remove(Neighbor::new(1, 1.0)));
        // Cursor should remain the same
        assert_eq!(queue.cursor, 1);
        assert_eq!(queue.size(), 2);

        // Verify remaining elements
        assert_eq!(queue.get(0).id, 2); // 0.5
        assert_eq!(queue.get(1).id, 3); // 1.5
    }

    #[test]
    fn test_remove_maintains_sorted_order() {
        let mut queue = NeighborPriorityQueue::new(10);

        // Insert elements
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));
        queue.insert(Neighbor::new(5, 2.0));
        queue.insert(Neighbor::new(6, 0.8));

        // Remove multiple elements
        queue.remove(Neighbor::new(3, 1.5));
        queue.remove(Neighbor::new(4, 0.3));

        // Verify queue is still sorted
        for i in 0..queue.size() - 1 {
            assert!(queue.get(i).distance <= queue.get(i + 1).distance);
        }

        // Verify exact order
        assert_eq!(queue.get(0).id, 2); // 0.5
        assert_eq!(queue.get(1).id, 6); // 0.8
        assert_eq!(queue.get(2).id, 1); // 1.0
        assert_eq!(queue.get(3).id, 5); // 2.0
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_retain() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        // Insert elements
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));
        queue.insert(Neighbor::new(5, 2.0));
        queue.insert(Neighbor::new(6, 0.8));

        assert_eq!(queue.size(), 6);

        // Keep only elements with distance <= 1.0
        queue.retain(|nbr| nbr.distance <= 1.0);

        assert_eq!(queue.size(), 4);

        // Verify remaining elements in sorted order
        assert_eq!(queue.get(0).id, 4); // 0.3
        assert_eq!(queue.get(1).id, 2); // 0.5
        assert_eq!(queue.get(2).id, 6); // 0.8
        assert_eq!(queue.get(3).id, 1); // 1.0

        // Keep elements with id >= 3 (removes 2 and 1)
        queue.retain(|nbr| nbr.id >= 3);

        assert_eq!(queue.size(), 2);

        // Verify ids 4 and 6 remain
        assert_eq!(queue.get(0).id, 4);
        assert_eq!(queue.get(1).id, 6);
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_retain_empty() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        // Retain on empty queue should not panic
        queue.retain(|_| true);
        assert_eq!(queue.size(), 0);
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_retain_remove_all() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));

        // Remove all elements (keep none)
        queue.retain(|_| false);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.cursor, 0);
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_retain_remove_none() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));

        // Keep all elements
        queue.retain(|_| true);

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 2); // 0.5
        assert_eq!(queue.get(1).id, 1); // 1.0
        assert_eq!(queue.get(2).id, 3); // 1.5
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_retain_resets_visited_state() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));

        // Mark some nodes as visited
        queue.closest_notvisited(); // marks 4 (0.3) as visited
        queue.closest_notvisited(); // marks 2 (0.5) as visited

        assert_eq!(queue.cursor, 2);
        assert!(!queue.has_notvisited_node() || queue.cursor < queue.size());

        // Retain only elements with distance <= 1.0 (keeps 4, 2, 1)
        queue.retain(|nbr| nbr.distance <= 1.0);

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.cursor, 0); // Cursor should be reset

        // All elements should be marked as unvisited after retain
        assert!(queue.has_notvisited_node());

        // Verify we can traverse all elements again
        let first = queue.closest_notvisited();
        assert_eq!(first.id, 4); // 0.3
        assert_eq!(queue.cursor, 1);

        let second = queue.closest_notvisited();
        assert_eq!(second.id, 2); // 0.5
        assert_eq!(queue.cursor, 2);

        let third = queue.closest_notvisited();
        assert_eq!(third.id, 1); // 1.0
        assert_eq!(queue.cursor, 3);
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_truncate() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));
        queue.insert(Neighbor::new(5, 2.0));

        assert_eq!(queue.size(), 5);

        // Truncate to 3 elements
        queue.truncate(3);

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.get(0).id, 4); // 0.3
        assert_eq!(queue.get(1).id, 2); // 0.5
        assert_eq!(queue.get(2).id, 1); // 1.0
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_truncate_larger_size() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));

        // Truncate to larger size should not change anything
        queue.truncate(10);

        assert_eq!(queue.size(), 2);
    }

    #[test]
    #[cfg(feature = "experimental_diversity_search")]
    fn test_truncate_with_cursor() {
        let mut queue = NeighborPriorityQueue::<u32>::new(10);

        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 0.3));

        // Advance cursor
        queue.closest_notvisited(); // cursor = 1
        queue.closest_notvisited(); // cursor = 2

        assert_eq!(queue.cursor, 2);

        // Truncate to size smaller than cursor
        queue.truncate(1);

        assert_eq!(queue.size(), 1);
        assert_eq!(queue.cursor, 0); // cursor is always reset to 0
    }

    #[test]
    fn test_peek_best_unsubmitted_basic() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)]

        let submitted = HashSet::new();
        let result = queue.peek_best_unsubmitted(&submitted);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, 2); // closest unvisited, unsubmitted
    }

    #[test]
    fn test_peek_best_unsubmitted_skips_submitted() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)]

        let mut submitted = HashSet::new();
        submitted.insert(2u32);
        let result = queue.peek_best_unsubmitted(&submitted);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, 1); // 2 is submitted, so next is 1
    }

    #[test]
    fn test_peek_best_unsubmitted_skips_visited() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)]

        queue.closest_notvisited(); // visits 2

        let submitted = HashSet::new();
        let result = queue.peek_best_unsubmitted(&submitted);
        assert!(result.is_some());
        assert_eq!(result.unwrap().id, 1); // 2 is visited, so next is 1
    }

    #[test]
    fn test_peek_best_unsubmitted_none_when_all_excluded() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));

        let mut submitted = HashSet::new();
        submitted.insert(1u32);
        submitted.insert(2u32);
        let result = queue.peek_best_unsubmitted(&submitted);
        assert!(result.is_none());
    }

    #[test]
    fn test_peek_best_unsubmitted_respects_search_l() {
        let mut queue = NeighborPriorityQueue::auto_resizable_with_search_param_l(2);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        queue.insert(Neighbor::new(4, 2.0));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5), 4(2.0)], search_l=2

        let mut submitted = HashSet::new();
        submitted.insert(2u32);
        submitted.insert(1u32);
        // Both nodes within search_l window are submitted
        let result = queue.peek_best_unsubmitted(&submitted);
        assert!(result.is_none());
    }

    #[test]
    fn test_peek_best_unsubmitted_does_not_modify_state() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));

        let submitted = HashSet::new();
        let _ = queue.peek_best_unsubmitted(&submitted);
        let _ = queue.peek_best_unsubmitted(&submitted);

        // Cursor should still be at 0 (no state modification)
        assert_eq!(queue.cursor, 0);
        assert!(queue.has_notvisited_node());
    }

    #[test]
    fn test_peek_best_unsubmitted_empty_queue() {
        let queue = NeighborPriorityQueue::<u32>::new(5);
        let submitted = HashSet::new();
        assert!(queue.peek_best_unsubmitted(&submitted).is_none());
    }

    #[test]
    fn test_mark_visited_by_id_basic() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)]

        assert!(queue.mark_visited_by_id(&1));
        assert!(queue.get_visited(1)); // id=1 is at index 1
    }

    #[test]
    fn test_mark_visited_by_id_not_found() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));

        assert!(!queue.mark_visited_by_id(&99));
    }

    #[test]
    fn test_mark_visited_by_id_advances_cursor() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)], cursor=0

        // Mark the node at cursor (id=2 at index 0)
        assert!(queue.mark_visited_by_id(&2));
        // Cursor should advance past this visited node to index 1
        assert_eq!(queue.cursor, 1);
    }

    #[test]
    fn test_mark_visited_by_id_cursor_skips_consecutive_visited() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)], cursor=0

        // Visit id=1 (index 1) first - cursor stays at 0
        assert!(queue.mark_visited_by_id(&1));
        assert_eq!(queue.cursor, 0);

        // Now visit id=2 (index 0, where cursor is) - cursor should skip past both visited nodes
        assert!(queue.mark_visited_by_id(&2));
        assert_eq!(queue.cursor, 2); // skips index 0 (visited) and index 1 (visited)
    }

    #[test]
    fn test_mark_visited_by_id_does_not_move_cursor_for_non_cursor_node() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)], cursor=0

        // Mark id=3 (index 2) as visited - cursor should stay at 0
        assert!(queue.mark_visited_by_id(&3));
        assert_eq!(queue.cursor, 0);
    }

    #[test]
    fn test_peek_and_mark_workflow() {
        let mut queue = NeighborPriorityQueue::new(5);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        // Queue sorted: [2(0.5), 1(1.0), 3(1.5)]

        let mut submitted = HashSet::new();

        // Peek - should return id=2
        let node = queue.peek_best_unsubmitted(&submitted).unwrap();
        assert_eq!(node.id, 2);
        submitted.insert(node.id);

        // Peek again - should return id=1 (2 is submitted)
        let node = queue.peek_best_unsubmitted(&submitted).unwrap();
        assert_eq!(node.id, 1);
        submitted.insert(node.id);

        // Mark id=2 as visited (IO completed)
        assert!(queue.mark_visited_by_id(&2));

        // Peek - should return id=3 (2 visited, 1 submitted)
        let node = queue.peek_best_unsubmitted(&submitted).unwrap();
        assert_eq!(node.id, 3);
    }
}
