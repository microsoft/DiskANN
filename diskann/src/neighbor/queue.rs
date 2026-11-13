/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
};

#[cfg(target_arch = "x86_64")]
use diskann_wide::{SIMDMask, SIMDPartialOrd, SIMDVector};

use super::Neighbor;

/// Shared trait for the generic `I` parameter used by `NeighborQueue`.
pub trait NeighborPriorityQueueIdType: Eq + Clone + Copy + Debug + Display + Send + Sync {}

/// Any type that implements all the individual requirements for
/// `NeighborPriorityQueueIdType` implements the full trait.
impl<T> NeighborPriorityQueueIdType for T where T: Eq + Clone + Copy + Debug + Display + Send + Sync {}

/// Trait defining the interface for a neighbor priority queue.
///
/// This trait abstracts the core functionality of a priority queue that manages
/// neighbors ordered by distance, supporting both fixed-size and resizable queues.
pub trait NeighborQueue<I: NeighborPriorityQueueIdType>: Debug + Send + Sync {
    /// The iterator type returned by `iter()`.
    type Iter<'a>: ExactSizeIterator<Item = Neighbor<I>> + Send + Sync
    where
        Self: 'a,
        I: 'a;

    /// Insert a neighbor into the priority queue.
    fn insert(&mut self, nbr: Neighbor<I>);

    /// Insert unique neighbors into the priority queue, clearing the buffer without
    /// changing its capacity.
    fn batch_insert(&mut self, neighbors: &mut Vec<Neighbor<I>>) {
        for neighbor in neighbors.drain(..) {
            self.insert(neighbor);
        }
    }

    /// Get the neighbor at the specified index.
    fn get(&self, index: usize) -> Neighbor<I>;

    /// Get the closest unvisited neighbor.
    fn closest_notvisited(&mut self) -> Option<Neighbor<I>>;

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
        if nbr.distance.is_nan() {
            // We don't support NaN distances. If we see one, we ignore the insert since we can't determine where it belongs in the sorted order.
            return;
        }

        self.dbgassert_unique_insert(nbr.id);

        if self.auto_resizable {
            if self.size == self.capacity {
                self.reserve(1.max(self.capacity >> 1)); // 1.5x capacity
            }
        } else if self.size == self.capacity
            && self.get_unchecked(self.size - 1).distance() < nbr.distance()
        {
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

    /// Insert unique neighbors into the priority queue, clearing the buffer without
    /// changing its capacity.
    ///
    /// As with [`Self::insert`], IDs must be unique; NaN distances are ignored.
    pub fn batch_insert(&mut self, neighbors: &mut Vec<Neighbor<I>>) {
        if self.auto_resizable {
            for neighbor in neighbors.drain(..) {
                self.insert(neighbor);
            }
            return;
        }

        if self.capacity == 0 {
            neighbors.clear();
            return;
        }

        #[cfg(debug_assertions)]
        for (i, neighbor) in neighbors.iter().enumerate() {
            if !neighbor.distance.is_nan() {
                self.dbgassert_unique_insert(neighbor.id);
                debug_assert!(
                    neighbors[..i]
                        .iter()
                        .all(|other| other.distance.is_nan() || other.id != neighbor.id),
                    "Neighbor with ID {} occurs more than once in the batch",
                    neighbor.id
                );
            }
        }

        let worst = (self.size == self.capacity).then(|| self.distances[self.size - 1]);
        neighbors.retain(|neighbor| {
            !neighbor.distance.is_nan()
                && worst.is_none_or(|distance| neighbor.distance <= distance)
        });
        if neighbors.is_empty() {
            return;
        }

        // Later equal-distance candidates must precede earlier ones.
        neighbors.reverse();
        neighbors.sort_by(super::ord::fast_distance);

        self.merge_sorted(neighbors);
        neighbors.clear();
    }

    /// Merge a nonempty, NaN-free batch sorted by ascending distance.
    /// Candidates precede old entries on ties; ties within the batch keep its order.
    fn merge_sorted(&mut self, neighbors: &[Neighbor<I>]) {
        let candidate_count = neighbors.len();
        let first = neighbors[0];
        let prefix_len = self.get_lower_bound(&first);
        let mut cursor = self.cursor.min(prefix_len);
        // Cursor-resetting operations may leave it pointing at a visited entry.
        // The unchanged prefix retains its visited flags; the first new entry is unvisited.
        while cursor < prefix_len && self.id_visiteds[cursor].1 {
            cursor += 1;
        }

        let merged_size = self.capacity.min(self.size + candidate_count);
        // Partition the combined top-L so neither discarded tail needs to be merged.
        let mut min_old = prefix_len.max(merged_size.saturating_sub(candidate_count));
        let mut max_old = self.size.min(merged_size);
        while min_old < max_old {
            let old_count = min_old + (max_old - min_old) / 2;
            let new_count = merged_size - old_count;
            if old_count < self.size
                && new_count > 0
                && self.distances[old_count] < neighbors[new_count - 1].distance
            {
                min_old = old_count + 1;
            } else {
                max_old = old_count;
            }
        }
        let (mut old_remaining, mut new_remaining) = (min_old, merged_size - min_old);

        self.id_visiteds.truncate(old_remaining);
        self.distances.truncate(old_remaining);
        self.id_visiteds.resize(merged_size, (first.id, false));
        self.distances.resize(merged_size, 0.0);
        // Merge backwards without overwriting unread old entries.
        // On ties, write the old entry to the higher slot so new entries stay first.
        while new_remaining > 1 {
            let destination = old_remaining + new_remaining - 1;
            let neighbor = neighbors[new_remaining - 1];
            if old_remaining > 0 && self.distances[old_remaining - 1] >= neighbor.distance {
                old_remaining -= 1;
                self.id_visiteds[destination] = self.id_visiteds[old_remaining];
                self.distances[destination] = self.distances[old_remaining];
            } else {
                new_remaining -= 1;
                self.id_visiteds[destination] = (neighbor.id, false);
                self.distances[destination] = neighbor.distance;
            }
        }
        if new_remaining == 1 {
            // Only the first candidate remains; shift the old suffix in one move.
            self.id_visiteds
                .copy_within(prefix_len..old_remaining, prefix_len + 1);
            self.distances
                .copy_within(prefix_len..old_remaining, prefix_len + 1);
            self.id_visiteds[prefix_len] = (first.id, false);
            self.distances[prefix_len] = first.distance;
        }
        self.size = merged_size;
        self.cursor = cursor;
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
    #[cfg(target_arch = "x86_64")]
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

    /// Get the lower bound of the neighbor - the index of the first neighbor
    /// that has a distance greater than or equal to the target neighbor
    /// PERFORMANCE: This function is performance critical - please benchmark any changes
    #[cfg(not(target_arch = "x86_64"))]
    fn get_lower_bound(&mut self, nbr: &Neighbor<I>) -> usize {
        let target = nbr.distance;

        for (i, d) in self.distances.iter().take(self.size).enumerate() {
            if *d >= target {
                return i;
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

    /// Get the closest and notvisited neighbor. This function returns None if there are no notvisited neighbors.
    pub fn closest_notvisited(&mut self) -> Option<Neighbor<I>> {
        // has_notvisited_node() ensures that cursor is less than size and thus get_unchecked call inside set_visited() is safe to call. Without this check, set_visited() would panic if cursor is equal to size.
        if !self.has_notvisited_node() {
            return None;
        }

        let current = self.cursor;
        self.set_visited(current, true);

        // Look for the next notvisited neighbor
        self.cursor += 1;
        while self.cursor < self.size && self.get_visited(self.cursor) {
            self.cursor += 1;
        }

        Some(self.get_unchecked(current))
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
        assert!(index < self.size);
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

    fn batch_insert(&mut self, neighbors: &mut Vec<Neighbor<I>>) {
        self.batch_insert(neighbors)
    }

    fn get(&self, index: usize) -> Neighbor<I> {
        self.get(index)
    }

    fn closest_notvisited(&mut self) -> Option<Neighbor<I>> {
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

    #[track_caller]
    fn assert_same_queue<I: NeighborPriorityQueueIdType>(
        actual: &NeighborPriorityQueue<I>,
        expected: &NeighborPriorityQueue<I>,
    ) {
        assert_eq!(actual.size, expected.size);
        assert_eq!(actual.cursor, expected.cursor);
        assert_eq!(actual.id_visiteds, expected.id_visiteds);
        assert!(
            actual
                .distances
                .iter()
                .map(|d| d.to_bits())
                .eq(expected.distances.iter().map(|d| d.to_bits())),
            "distance bit patterns differ"
        );
    }

    #[test]
    fn batch_insert_matches_sequential_insertion() {
        let cases: &[(usize, usize, usize, &[f32])] = &[
            (16, 0, 0, &[2.0, 1.0]),
            (16, 7, 3, &[4.5, 1.5, 8.0]),
            (4, 4, 2, &[3.5, -1.0, 1.5, 8.0, 2.5]),
            (4, 4, 1, &[0.0, -0.0, 1.0, 1.0]),
            (16, 16, 16, &[f32::NAN, 7.5]),
            (
                16,
                3,
                2,
                &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0],
            ),
            (128, 128, 3, &[63.5, -1.0, 200.0]),
        ];
        for resizable in [false, true] {
            for &(capacity, initial, visited, distances) in cases {
                let mut reference = if resizable {
                    NeighborPriorityQueue::auto_resizable_with_search_param_l(capacity)
                } else {
                    NeighborPriorityQueue::new(capacity)
                };
                for id in 0..initial {
                    reference.insert(Neighbor::new(id, id as f32));
                }
                for _ in 0..visited {
                    reference.closest_notvisited();
                }
                let mut actual = reference.clone();
                let mut candidates: Vec<_> = distances
                    .iter()
                    .enumerate()
                    .map(|(i, &distance)| Neighbor::new(initial + i, distance))
                    .collect();
                for &candidate in &candidates {
                    reference.insert(candidate);
                }
                let buffer_capacity = candidates.capacity();
                let allocation = candidates.as_ptr();
                actual.batch_insert(&mut candidates);
                assert!(candidates.is_empty());
                assert_eq!(candidates.capacity(), buffer_capacity);
                assert_eq!(candidates.as_ptr(), allocation);
                assert_same_queue(&actual, &reference);
            }
        }
    }

    #[test]
    fn batch_insert_handles_empty_and_filtered_batches() {
        let cases: &[(usize, &[f32])] = &[
            (0, &[]),
            (0, &[f32::NEG_INFINITY, f32::NAN]),
            (4, &[]),
            (4, &[f32::NAN]),
            (4, &[2.0]),
        ];
        for &(capacity, distances) in cases {
            let mut queue = NeighborPriorityQueue::new(capacity);
            for id in 0..capacity {
                queue.insert(Neighbor::new(id, 1.0));
            }
            let reference = queue.clone();
            let mut candidates: Vec<_> = distances
                .iter()
                .enumerate()
                .map(|(i, &distance)| Neighbor::new(capacity + i, distance))
                .collect();
            let buffer_capacity = candidates.capacity();
            let allocation = candidates.as_ptr();
            NeighborQueue::batch_insert(&mut queue, &mut candidates);
            assert!(candidates.is_empty());
            assert_eq!(candidates.capacity(), buffer_capacity);
            assert_eq!(candidates.as_ptr(), allocation);
            assert_eq!(queue.capacity(), capacity);
            assert_same_queue(&queue, &reference);
        }
    }

    #[test]
    fn batch_insert_normalizes_cursor_after_drain() {
        let mut queue = NeighborPriorityQueue::new(8);
        for id in 0..6 {
            queue.insert(Neighbor::new(id, id as f32));
        }
        for _ in 0..6 {
            queue.closest_notvisited();
        }
        queue.drain_best(2);
        queue.batch_insert(&mut vec![Neighbor::new(6, 10.0)]);

        assert_eq!(queue.cursor, 4);
        assert_eq!(
            queue.closest_notvisited().map(Neighbor::as_tuple),
            Some((6, 10.0))
        );
        assert!(!queue.has_notvisited_node());
    }

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
        let nbr = queue.closest_notvisited().unwrap();
        assert_eq!(nbr.id, 2);
        assert_eq!(nbr.distance, 0.5);
        assert!(queue.get_visited(0)); // super unfortunate test. We know based on above id 2 should be 0th index
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited().unwrap();
        assert_eq!(nbr.id, 1);
        assert_eq!(nbr.distance, 1.0);
        assert!(queue.get_visited(1));
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited().unwrap();
        assert_eq!(nbr.id, 3);
        assert_eq!(nbr.distance, 1.5);
        assert!(queue.get_visited(2));
        assert!(!queue.has_notvisited_node());
        assert!(queue.closest_notvisited().is_none());
    }

    #[test]
    fn test_closest_notvisited_when_no_notvisited_nodes_left() {
        let mut queue = NeighborPriorityQueue::new(1);
        queue.insert(Neighbor::new(1, 1.0));
        assert!(queue.closest_notvisited().is_some());
        assert!(queue.closest_notvisited().is_none());
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

        assert!(queue.closest_notvisited().is_some());
        assert!(queue.closest_notvisited().is_some());
        assert!(queue.closest_notvisited().is_some());
        assert!(queue.closest_notvisited().is_some());

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

        assert!(queue.closest_notvisited().is_some());
        assert!(queue.has_notvisited_node());
        assert!(queue.closest_notvisited().is_some());
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

        assert!(queue.closest_notvisited().is_some());
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
            assert!(queue.closest_notvisited().is_some());
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

        assert!(queue.closest_notvisited().is_some());
        assert!(queue.has_notvisited_node());
        assert!(queue.closest_notvisited().is_some());
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

        assert!(queue.closest_notvisited().is_some());
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
        let closest = queue.closest_notvisited().unwrap();
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

            let closest = queue.closest_notvisited().unwrap();
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
        assert!(queue.closest_notvisited().is_some()); // Visits 4, cursor = 1
        assert!(queue.closest_notvisited().is_some()); // Visits 2, cursor = 2

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
    fn test_insert_neighbors_with_infinity_distance() {
        let mut queue = NeighborPriorityQueue::new(5);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 5);

        for id in 0..2 {
            queue.insert(Neighbor::new(id, f32::INFINITY));
        }

        assert_eq!(queue.size(), 2);
        assert_eq!(queue.capacity(), 5);

        for id in 2..10 {
            queue.insert(Neighbor::new(id, f32::INFINITY));
        }

        assert_eq!(queue.size(), 5);
        assert_eq!(queue.capacity(), 5);

        assert!(queue.get(0).id >= 0, "First element should be retrievable");
    }

    #[test]
    fn test_normal_distances_should_push_infinity_distances_away_from_queue() {
        let mut queue = NeighborPriorityQueue::new(5);

        assert_eq!(queue.size(), 0);
        assert_eq!(queue.capacity(), 5);

        for id in 0..=4 {
            queue.insert(Neighbor::new(id, f32::INFINITY));
        }

        assert_eq!(queue.size(), 5);
        assert_eq!(queue.capacity(), 5);

        assert!(queue.get(0).id >= 0, "First element should be retrievable");

        for id in 5..=7 {
            queue.insert(Neighbor::new(id, id as f32));
        }

        assert_eq!(queue.size(), 5);
        assert_eq!(queue.capacity(), 5);

        // The normal distance neighbors should be at the front of the queue
        assert_eq!(queue.get(0).id, 5);
        assert_eq!(queue.get(1).id, 6);
        assert_eq!(queue.get(2).id, 7);

        // The infinity distance neighbors should be pushed to the end of the queue
        assert_eq!(queue.get(3).id, 4);
        assert_eq!(queue.get(4).id, 3);
    }

    #[test]
    fn test_insert_neighbor_with_nan_distance_is_ignored() {
        let mut queue = NeighborPriorityQueue::new(5);

        assert_eq!(queue.size(), 0);
        queue.insert(Neighbor::new(0, f32::NAN));
        assert_eq!(queue.size(), 0);
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
        assert!(queue.closest_notvisited().is_some()); // marks 4 (0.3) as visited
        assert!(queue.closest_notvisited().is_some()); // marks 2 (0.5) as visited

        assert_eq!(queue.cursor, 2);
        assert!(!queue.has_notvisited_node() || queue.cursor < queue.size());

        // Retain only elements with distance <= 1.0 (keeps 4, 2, 1)
        queue.retain(|nbr| nbr.distance <= 1.0);

        assert_eq!(queue.size(), 3);
        assert_eq!(queue.cursor, 0); // Cursor should be reset

        // All elements should be marked as unvisited after retain
        assert!(queue.has_notvisited_node());

        // Verify we can traverse all elements again
        let first = queue.closest_notvisited().unwrap();
        assert_eq!(first.id, 4); // 0.3
        assert_eq!(queue.cursor, 1);

        let second = queue.closest_notvisited().unwrap();
        assert_eq!(second.id, 2); // 0.5
        assert_eq!(queue.cursor, 2);

        let third = queue.closest_notvisited().unwrap();
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
        assert!(queue.closest_notvisited().is_some()); // cursor = 1
        assert!(queue.closest_notvisited().is_some()); // cursor = 2

        assert_eq!(queue.cursor, 2);

        // Truncate to size smaller than cursor
        queue.truncate(1);

        assert_eq!(queue.size(), 1);
        assert_eq!(queue.cursor, 0); // cursor is always reset to 0
    }
}
