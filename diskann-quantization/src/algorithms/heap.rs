/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

/// A fixed-size heap data structure that operates in place on non-empty mutable slices.
///
/// The heap size never changes after creation, and only supports updating the maximum element.
pub struct SliceHeap<'a, T: Ord + Copy> {
    data: &'a mut [T],
}

#[derive(Debug, Error)]
#[error("heap cannot be constructed from an empty slice")]
pub struct EmptySlice;

impl<'a, T: Ord + Copy> SliceHeap<'a, T> {
    /// Creates a new `SliceHeap` from a mutable slice.
    /// The slice is assumed to be unordered initially and will be heapified.
    ///
    /// # Errors
    ///
    /// Returns `EmptySlice` if the input slice is empty.
    pub fn new(data: &'a mut [T]) -> Result<Self, EmptySlice> {
        if data.is_empty() {
            return Err(EmptySlice);
        }

        let mut heap = SliceHeap { data };
        heap.heapify();
        Ok(heap)
    }

    /// Creates a new `SliceHeap` from a mutable slice without heapifying.
    /// Use this if you know the slice is already in heap order.
    ///
    /// # Errors
    ///
    /// Returns `EmptySlice` if the input slice is empty.
    pub fn new_unchecked(data: &'a mut [T]) -> Result<Self, EmptySlice> {
        if data.is_empty() {
            return Err(EmptySlice);
        }

        Ok(SliceHeap { data })
    }

    /// Returns the number of elements in the heap.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Always returns `false` as the heap can never be empty
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Returns a reference to the greatest element in the heap, or `None` if empty.
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Updates the root element in place and restores the heap property.
    /// This allows direct mutation of the maximum element.
    ///
    /// Since the heap cannot be empty (enforced by construction), this operation always succeeds.
    pub fn update_root<F>(&mut self, update_fn: F)
    where
        F: FnOnce(&mut T),
    {
        // SAFETY: The heap is guaranteed to be unempty.
        let root = unsafe { self.data.get_unchecked_mut(0) };
        update_fn(root);
        self.sift_down(0);
    }

    /// Converts the entire slice into a heap.
    pub fn heapify(&mut self) {
        if self.data.len() <= 1 {
            return;
        }

        // Start from the last non-leaf node and sift down
        let start = (self.data.len() - 2) / 2;
        for i in (0..=start).rev() {
            self.sift_down(i);
        }
    }

    /// Returns a slice of all heap elements in heap order (not sorted order).
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Get the element as position `pos`.
    ///
    /// # Safety
    ///
    /// `pos < self.len()` (checked in debug mode).
    unsafe fn get_unchecked(&self, pos: usize) -> &T {
        debug_assert!(pos < self.len());
        self.data.get_unchecked(pos)
    }

    /// Swap the two elements as positions `a` and `b`.
    ///
    /// # Safety
    ///
    /// All the following must hold (these are checked in debug mode):
    ///
    /// 1. `a < self.len()`.
    /// 2. `b < self.len()`.
    /// 3. `a != b`.
    unsafe fn swap_unchecked(&mut self, a: usize, b: usize) {
        debug_assert!(a < self.len());
        debug_assert!(b < self.len());
        debug_assert!(a != b);
        let base = self.data.as_mut_ptr();

        // SAFETY: The safety requirements of this function imply that the pointer arithmetic
        // is valid and that the non-overlapping criteria are satisfied.
        unsafe { std::ptr::swap_nonoverlapping(base.add(a), base.add(b), 1) }
    }

    /// The implementation of this function is largely copied from `sift_down_range` in
    /// https://doc.rust-lang.org/src/alloc/collections/binary_heap/mod.rs.html#776.
    ///
    /// Since we've constrainted `T: Copy`, we don't need to worry about the `Hole` helper
    /// data structures.
    fn sift_down(&mut self, mut pos: usize) {
        const {
            assert!(
                std::mem::size_of::<T>() != 0,
                "cannot operate on a `SliceHeap` with a zero sized type"
            )
        };

        let len = self.len();

        // Since the maximum allocation size is `isize::MAX`, the maximum value that `pos`
        // can be while satisfying the safety requirements is `isize::MAX`.
        //
        // This means that `2 * pos + 1 == usize::MAX` so this operation never overflows.
        let mut child = 2 * pos + 1;

        // Loop Invariant: child == 2 * pos + 1
        while child <= len.saturating_sub(2) {
            // compare with the greater of the two children
            // SAFETY: We have the following:
            //  * `child >= 1`: By loop invariant. If we enter this loop, then we're
            //    guaranteed that `len >= 3`.
            //  * `child < self.len() - 1` and thus `child + 1 < self.len()` - so both are
            //    valid indices.
            child += unsafe { self.get_unchecked(child) <= self.get_unchecked(child + 1) } as usize;

            // If we are already in order, stop.
            //
            // SAFETY: `child` is now either the old `child` or the old `child + 1`
            // We already proven that both are `< self.len()`.
            //
            // Furthermore, since `pos < child` (no matter which one is chosen), `pos` is
            // also in-bounds.
            if unsafe { self.get_unchecked(pos) >= self.get_unchecked(child) } {
                return;
            }

            // SAFETY: We've proven that `pos` and `child` are in-bounds. Since
            //  * `child = 2 * pos + 1 > pos`.
            //  * `child + 1 = 2 * pos + 2 > pos`.
            // we are guaranteed that `pos != child`.
            unsafe { self.swap_unchecked(pos, child) };
            pos = child;
            child = 2 * pos + 1;
        }

        // SAFETY: We've explicitly checked that `child < self.len()` and from the loop
        // invariante above, `pos < child`. So both accesses are in-bounds.
        if child == len - 1 && unsafe { self.get_unchecked(pos) < self.get_unchecked(child) } {
            // SAFETY: We've proved that `pos` and `child` are in-bounds. From the loop
            // invariant above, `pos != child`, so the swap is valid.
            unsafe { self.swap_unchecked(pos, child) };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    #[test]
    fn test_basic_heap_creation() {
        let mut data = [3, 1, 4, 1, 5, 9, 2, 6];
        let heap = SliceHeap::new(&mut data).unwrap();

        assert_eq!(heap.len(), 8);
        assert!(!heap.is_empty());
        assert_eq!(heap.peek(), Some(&9));
    }

    #[test]
    fn test_update_root() {
        let mut data = [3, 1, 4, 1, 5, 9, 2, 6];
        let mut heap = SliceHeap::new(&mut data).unwrap();

        // Update max (9) to 5
        heap.update_root(|x| {
            assert_eq!(*x, 9);
            *x = 5
        });

        assert_eq!(heap.peek(), Some(&6));

        // Update max to 10 (should become new max)
        heap.update_root(|x| {
            assert_eq!(*x, 6);
            *x = 10
        });
        assert_eq!(heap.peek(), Some(&10));

        // If we update to the same value, it should remain in place.
        heap.update_root(|x| {
            assert_eq!(*x, 10);
            *x = 10;
        });
        assert_eq!(heap.peek(), Some(&10));

        // Update max to 1 (should sink to bottom)
        heap.update_root(|x| {
            assert_eq!(*x, 10);
            *x = 1
        });
        assert_eq!(heap.peek(), Some(&5));
    }

    #[test]
    fn test_empty_heap() {
        let mut data: [i32; 0] = [];
        let result = SliceHeap::new(&mut data);

        assert!(matches!(result, Err(EmptySlice)));

        let result_unchecked = SliceHeap::new_unchecked(&mut data);
        assert!(matches!(result_unchecked, Err(EmptySlice)));
    }

    #[test]
    fn test_single_element() {
        let mut data = [42];
        let mut heap = SliceHeap::new(&mut data).unwrap();

        assert_eq!(heap.len(), 1);
        assert_eq!(heap.peek(), Some(&42));

        heap.update_root(|x| *x = 100);
        assert_eq!(heap.peek(), Some(&100));

        heap.update_root(|x| *x = 10);
        assert_eq!(heap.peek(), Some(&10));
    }

    #[test]
    fn test_heapify() {
        let mut data = [1, 2, 3, 4, 5];
        let mut heap = SliceHeap::new_unchecked(&mut data).unwrap(); // Not heapified

        // Manually heapify
        heap.heapify();

        assert_eq!(heap.peek(), Some(&5));

        // Verify heap property by updating max to minimum and checking order
        heap.update_root(|x| *x = 0);
        assert_eq!(heap.peek(), Some(&4));

        heap.update_root(|x| *x = 0);
        assert_eq!(heap.peek(), Some(&3));
    }

    #[test]
    fn test_heap_property_maintained() {
        let mut data = [10, 8, 9, 4, 7, 5, 3, 2, 1, 6];
        let mut heap = SliceHeap::new(&mut data).unwrap();

        // Repeatedly update max with smaller values
        for new_val in (1..10).rev() {
            heap.update_root(|x| *x = new_val);

            // Verify heap property: parent >= children
            let slice = heap.as_slice();
            for i in 0..slice.len() {
                let left = 2 * i + 1;
                let right = 2 * i + 2;

                if left < slice.len() {
                    assert!(
                        slice[i] >= slice[left],
                        "Heap property violated: parent {} < left child {}",
                        slice[i],
                        slice[left]
                    );
                }

                if right < slice.len() {
                    assert!(
                        slice[i] >= slice[right],
                        "Heap property violated: parent {} < right child {}",
                        slice[i],
                        slice[right]
                    );
                }
            }
        }
    }

    fn fuzz_test_impl(heap_size: usize, num_operations: usize, rng: &mut StdRng) {
        // Generate initial data
        let mut slice_data: Vec<i32> = (0..heap_size)
            .map(|_| rng.random_range(-100..100))
            .collect();

        // Create heaps
        let mut binary_heap: BinaryHeap<i32> = slice_data.iter().copied().collect();
        let mut slice_heap = SliceHeap::new(&mut slice_data).unwrap();

        // Verify initial state
        assert_eq!(slice_heap.peek().copied(), binary_heap.peek().copied());

        // Perform random operations
        for iteration in 0..num_operations {
            // Generate a random new value for the maximum element
            let new_value = rng.random_range(-200..200);

            // Update slice heap
            let slice_old_max = slice_heap.peek().copied();
            slice_heap.update_root(|x| *x = new_value);
            let slice_new_max = slice_heap.peek().copied();

            // Update binary heap (remove max, add new value)
            let binary_old_max = binary_heap.pop();
            binary_heap.push(new_value);
            let binary_new_max = binary_heap.peek().copied();

            // Verify they have the same maximum
            assert_eq!(
                slice_old_max, binary_old_max,
                "Iteration {}: Old maxima differ after updating {} to {}. SliceHeap old max: {:?}, BinaryHeap old max: {:?}",
                iteration, slice_old_max.unwrap_or(0), new_value, slice_old_max, binary_old_max
            );

            assert_eq!(
                slice_new_max, binary_new_max,
                "Iteration {}: Maxima differ after updating {} to {}. SliceHeap max: {:?}, BinaryHeap max: {:?}",
                iteration, slice_old_max.unwrap_or(0), new_value, slice_new_max, binary_new_max
            );

            // Verify heap property is maintained in slice heap
            verify_heap_property(slice_heap.as_slice());

            // Occasionally verify that both heaps contain the same elements (when sorted)
            if iteration % 100 == 0 {
                let mut slice_elements: Vec<i32> = slice_heap.as_slice().to_vec();
                slice_elements.sort_unstable();
                slice_elements.reverse(); // Sort descending

                let mut binary_elements: Vec<i32> = binary_heap.clone().into_sorted_vec();
                binary_elements.reverse(); // BinaryHeap::into_sorted_vec() returns ascending, we want descending

                assert_eq!(
                    slice_elements, binary_elements,
                    "Iteration {}: Heap contents differ when sorted",
                    iteration
                );
            }
        }
    }

    #[test]
    fn fuzz_test_against_binary_heap() {
        let mut rng = StdRng::seed_from_u64(0x0d270403030e30bb);

        // Heap of size 1.
        fuzz_test_impl(1, 101, &mut rng);

        // Heap of size 2.
        fuzz_test_impl(2, 101, &mut rng);

        // Heap size not power of two.
        fuzz_test_impl(1000, 1000, &mut rng);

        // Heap size power of two.
        fuzz_test_impl(128, 1000, &mut rng);
    }

    #[test]
    fn fuzz_test_edge_cases() {
        let mut rng = StdRng::seed_from_u64(123);

        // Test with small heaps
        for heap_size in 1..=10 {
            let mut data: Vec<i32> = (0..heap_size)
                .map(|_| rng.random_range(-100..100))
                .collect();
            let mut heap = SliceHeap::new(&mut data).unwrap();

            // Perform random updates
            for _ in 0..50 {
                let new_value = rng.random_range(-200..200);
                heap.update_root(|x| *x = new_value);

                // Verify heap property
                verify_heap_property(heap.as_slice());

                // Verify max is actually the maximum
                let max = heap.peek().unwrap();
                assert!(
                    heap.as_slice().iter().all(|&x| x <= *max),
                    "Max element {} is not actually the maximum in heap: {:?}",
                    max,
                    heap.as_slice()
                );
            }
        }
    }

    /// Helper function to verify the heap property holds for a slice
    fn verify_heap_property(slice: &[i32]) {
        for i in 0..slice.len() {
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left < slice.len() {
                assert!(
                    slice[i] >= slice[left],
                    "Heap property violated: parent {} at index {} < left child {} at index {}. Full heap: {:?}",
                    slice[i], i, slice[left], left, slice
                );
            }

            if right < slice.len() {
                assert!(
                    slice[i] >= slice[right],
                    "Heap property violated: parent {} at index {} < right child {} at index {}. Full heap: {:?}",
                    slice[i], i, slice[right], right, slice
                );
            }
        }
    }
}
