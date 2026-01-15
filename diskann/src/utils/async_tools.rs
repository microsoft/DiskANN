/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    num::NonZeroUsize,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use thiserror::Error;

use crate::ANNError;

//////////////////////
// VectorIdBoxSlice //
//////////////////////

/// An auxiliary type for an ID/Vector pair.
#[derive(Debug)]
pub struct VectorIdBoxSlice<I, T> {
    pub vector_id: I,
    pub vector: Box<[T]>,
}

impl<I, T> VectorIdBoxSlice<I, T> {
    pub fn new(vector_id: I, vector: Box<[T]>) -> Self {
        Self { vector_id, vector }
    }
}

////////////
// Around //
////////////

/// Return an iterator over a window of `slice` centered around `i` without yielding the
/// element at position `i`. The window will wrap around `slice`, treating it as a circular
/// buffer.
///
/// The iterator will have a length `L` of the minimum of `len` and `slice.len() - 1`.
/// If `L` is odd, then the window is biased left. For example, if `slice.len() == 6`,
/// `i == 3` and `len == 3`, then the elements at position 1, 2, and 4 will be returned.
///
/// ```text
///                              i
///                              |
///         +-------------------------------------+
/// slice   |    0    1    2     3     4     5    |
///         +-------------------------------------+
///                   |    |           |
///                   +----+-----------+
///              Items yielded if `len == 3`.
/// ```
///
/// # Corner Case Handling
///
/// * If `slice.is_empty()`, the returned iterator will be empty.
/// * If `i >= slice.len()` it will be silently truncated `slice.len() - 1`.
/// * If `len` exceeds `slice.len() - 1` (recall we skip position `1`), it will be truncated
///   to `slice.len() - 1`.
///
/// This corner case handling means this function is always safe to call, but may not yield
/// the expected results if `i` or `len` are out-of-bounds.
pub(crate) fn around<T>(slice: &[T], i: usize, len: usize) -> Around<'_, T> {
    Around::new(slice, i, len)
}

/// Iterator for sampling a slice in a window around a fixed position. See [`around`] for
/// details.
#[derive(Debug)]
pub(crate) struct Around<'a, T> {
    slice: &'a [T],
    skip: usize,
    position: usize,
    remaining: usize,
}

impl<'a, T> Around<'a, T> {
    fn new(slice: &'a [T], i: usize, len: usize) -> Self {
        let max = slice.len();

        // Special-case length 0.
        if max == 0 {
            return Self {
                slice,
                skip: 0,
                position: 0,
                remaining: 0,
            };
        }

        // Subtraction cannot underflow because `max >= 1`.
        let len = len.min(max - 1);

        // `div_ceil` is what biases odd lengths to the left of `i`.
        let half = len.div_ceil(2);

        let i = i.min(max - 1);

        // Starting poisition with wrap around logic.
        let position = if i >= half {
            i - half
        } else {
            max - (half - i)
        };

        Self {
            slice,
            skip: i,
            position,
            remaining: len,
        }
    }

    // Increment with wrap around.
    fn inc(&mut self) -> usize {
        let current = self.position;
        let next = current + 1;
        if next == self.slice.len() {
            self.position = 0
        } else {
            self.position = next
        }
        current
    }
}

impl<'a, T> Iterator for Around<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            let mut i = self.inc();
            if i == self.skip {
                i = self.inc();
            }

            // SAFETY: `i` is guaranteed to be strictly less than `self.slice.len()` because
            // `Self::inc` maintains this property.
            debug_assert!(i < self.slice.len());
            Some(unsafe { self.slice.get_unchecked(i) })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// Our implementation of `size_hint` is exact.
impl<T> ExactSizeIterator for Around<'_, T> {}

///////////////////////////
// Dynamic Load Balancer //
///////////////////////////

/// A utility for dynamically processing elements in a slice `T` across multiple threads.
///
/// This works by using a local `AtomicUsize` to track which indices inside the slice have
/// been processed and is useful when the work to be performed on each `T` is not suitable
/// for static partitioning.
///
/// This can occur when the work is highly variable.
#[derive(Debug)]
pub(crate) struct DynamicBalancer<T> {
    items: Arc<[T]>,
    current: AtomicUsize,
}

impl<T> DynamicBalancer<T> {
    /// Construdct a new `DynamicBalancer` over `items`. The first invocation of `next` will
    /// return index `0` (assuming `items` is non-empty).
    pub(crate) fn new(items: Arc<[T]>) -> Self {
        Self {
            items,
            current: AtomicUsize::new(0),
        }
    }

    /// Retrieve the next unclamied item and return a tuple `(item, index)` containing said
    /// item and its position in the underlying slice.
    ///
    /// If all items have been claimed, return `None`.
    ///
    /// This function yields all items in the underlying slice exactly once.
    pub(crate) fn next(&self) -> Option<(&T, usize)> {
        let i = self.current.fetch_add(1, Ordering::Relaxed);

        // There is technically a race here where enough concurrent calls to `next` in
        // between the retrieval of `i` and "putting it back" that the counter wraps around.
        //
        // The odds of that happening, however, are exceedingly rare.
        match self.items.get(i) {
            Some(v) => Some((v, i)),
            None => {
                // This branch will never be hit in a realistic program.
                if i == usize::MAX {
                    std::process::abort();
                }

                self.current.fetch_sub(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Return all work items as a slice.
    pub(crate) fn all(&self) -> &[T] {
        &self.items
    }

    /// Return the number of work items.
    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }
}

////////////////
// arc_chunks //
////////////////

/// Return an iterator over `chunk_size` elements of the slice at a time, starting at the
/// beginning of the slice.
///
/// The chunks are `ArcChunk<T>` slices that do not overlap. If `chunk_size` does not divide
/// the length of the slice, then the last chunk will not have length `chunk_size`.
///
/// Unlike `std::slice::chunks`, the items yielded by this iterator are `'static` and thus
/// may be given to spawned tasks.
pub(crate) fn arc_chunks<T>(
    arc_slice: Arc<[T]>,
    chunk_size: NonZeroUsize,
) -> impl Iterator<Item = ArcChunk<T>> {
    let slice_len = arc_slice.len();
    let num_chunks = slice_len.div_ceil(chunk_size.into());
    (0..num_chunks).map(move |chunk| {
        let chunk_size: usize = chunk_size.into();
        let start = chunk_size * chunk;
        let stop = (start + chunk_size).min(slice_len);
        ArcChunk::new(arc_slice.clone(), start..stop)
    })
}

#[derive(Debug)]
pub(crate) struct ArcChunk<T> {
    /// The data we are iterating over.
    data: Arc<[T]>,
    /// The sub-chunk of `data` that this batch represents..
    chunk: Range<usize>,
}

// Manually implement `Clone` so it works even if `T` is not constrained to be `Clone`.
impl<T> Clone for ArcChunk<T> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            chunk: self.chunk.clone(),
        }
    }
}

impl<T> ArcChunk<T> {
    pub(crate) fn new(data: Arc<[T]>, chunk: Range<usize>) -> Self {
        assert!(chunk.end <= data.len(), "range is invalid for data");
        Self { data, chunk }
    }

    /// Return the length of this slice chunk.
    pub(crate) fn len(&self) -> usize {
        self.chunk.len()
    }

    /// Gat the underlying slice chunk.
    pub(crate) fn get_chunk(&self) -> &[T] {
        &self.data[self.chunk.clone()]
    }

    /// Get the `i`th entry in the slice chunk.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub(crate) fn get(&self, i: usize) -> &T {
        &self.get_chunk()[i]
    }
}

//////////////
// Parition //
//////////////

/// Partition a range `0..nitems` into `ntasks` disjoint ranges so that:
///
/// * The union of the returned ranges for `task` in `0..ntasks` is `0..nitems`
/// * The ranges for different values of `task` are disjoint.
/// * The length of any two ranges differs by at most 1.
///
/// Returns an error if `tasks >= ntasks`.
pub fn partition(
    nitems: usize,
    ntasks: NonZeroUsize,
    task: usize,
) -> Result<std::ops::Range<usize>, PartitionError> {
    if ntasks.get() <= task {
        return Err(PartitionError {
            ntasks: ntasks.get(),
            task,
        });
    }

    Ok(partition_impl(nitems, ntasks, task))
}

/// An iterator that yields disjoint sub-ranges that partition a given range.
///
/// This is the iterator version of [`partition()`]. It yields exactly `ntasks` ranges
/// that collectively cover `range` without overlaps.
///
/// See [`partition()`] for details on the partitioning algorithm.
#[derive(Debug, Clone)]
pub struct PartitionIter {
    nitems: usize,
    ntasks: NonZeroUsize,
    current: usize,
}

impl PartitionIter {
    pub fn new(nitems: usize, ntasks: NonZeroUsize) -> Self {
        Self {
            nitems,
            ntasks,
            current: 0,
        }
    }
}

impl Iterator for PartitionIter {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.ntasks.get() {
            None
        } else {
            let sub_range = partition_impl(self.nitems, self.ntasks, self.current);
            self.current += 1;
            Some(sub_range)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ntasks.get() - self.current;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PartitionIter {}

/// Internal helper function to calculate a partition range
///
/// This function assumes that task < ntasks.get().
fn partition_impl(nitems: usize, ntasks: NonZeroUsize, task: usize) -> std::ops::Range<usize> {
    let k = nitems / ntasks.get();
    let m = nitems - k * ntasks.get();

    if task >= m {
        let start = m * (k + 1) + (task - m) * k;
        let stop = start + k;
        start..stop
    } else {
        let start = task * (k + 1);
        let stop = start + k + 1;
        start..stop
    }
}

#[derive(Debug, Error)]
#[error("task id {task} must be less than the number of tasks {ntasks}")]
pub struct PartitionError {
    ntasks: usize,
    task: usize,
}

impl From<PartitionError> for ANNError {
    fn from(err: PartitionError) -> Self {
        Self::log_async_error(err)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // Test when the chunk-size evenly divides the range.
    #[test]
    fn test_chunks_even() {
        let x: Arc<[usize]> = (0..12).collect();
        let ptr = x.as_ptr();

        let mut chunks = arc_chunks(x.clone(), NonZeroUsize::new(4).unwrap());
        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 4);
        assert_eq!(c.get_chunk(), &[0, 1, 2, 3]);
        for i in 0..4 {
            assert_eq!(c.get(i), &i)
        }

        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 4);
        assert_eq!(c.get_chunk(), &[4, 5, 6, 7]);
        for i in 0..4 {
            assert_eq!(c.get(i), &(i + 4))
        }

        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 4);
        assert_eq!(c.get_chunk(), &[8, 9, 10, 11]);
        for i in 0..4 {
            assert_eq!(c.get(i), &(i + 8))
        }

        let c = chunks.next();
        assert!(c.is_none());
    }

    // Test when the chunk-size does not evenly divide the range.
    #[test]
    fn test_chunks_odd() {
        let x: Arc<[usize]> = (0..11).collect();
        let ptr = x.as_ptr();

        let mut chunks = arc_chunks(x.clone(), NonZeroUsize::new(4).unwrap());
        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 4);
        assert_eq!(c.get_chunk(), &[0, 1, 2, 3]);
        for i in 0..4 {
            assert_eq!(c.get(i), &i)
        }

        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 4);
        assert_eq!(c.get_chunk(), &[4, 5, 6, 7]);
        for i in 0..4 {
            assert_eq!(c.get(i), &(i + 4))
        }

        let c = chunks.next().unwrap();
        assert_eq!(c.data.as_ptr(), ptr, "expected the pointer to be preserved");
        assert_eq!(c.len(), 3);
        assert_eq!(c.get_chunk(), &[8, 9, 10]);
        for i in 0..3 {
            assert_eq!(c.get(i), &(i + 8))
        }

        let c = chunks.next();
        assert!(c.is_none());
    }

    #[test]
    #[should_panic]
    fn test_chunks_panics() {
        let x: Arc<[usize]> = (0..11).collect();
        let mut chunks = arc_chunks(x.clone(), NonZeroUsize::new(4).unwrap());
        let c = chunks.next().unwrap();
        assert_eq!(c.len(), 4);
        let _: &usize = c.get(4);
    }

    #[test]
    fn test_around_empty() {
        let empty: &[usize] = &[];
        for entry in 0..5 {
            for len in 0..5 {
                let itr = around(empty, entry, len);

                // ExactSizeIter
                assert_eq!(itr.len(), 0);
                assert_eq!(itr.size_hint(), (0, Some(0)));

                let v: Vec<_> = itr.copied().collect();
                assert_eq!(&*v, empty);
            }
        }

        for len in 1..5 {
            let v: Vec<usize> = (0..len).collect();
            for entry in 0..10 {
                let itr = around(&v, entry, 0);

                assert_eq!(itr.len(), 0);
                assert_eq!(itr.size_hint(), (0, Some(0)));

                let v: Vec<_> = itr.copied().collect();
                assert_eq!(&*v, empty);
            }
        }
    }

    #[test]
    fn test_around_cases() {
        struct TestCase {
            entry: usize,
            len: usize,
            expected: &'static [usize],
        }

        impl TestCase {
            fn new(entry: usize, len: usize, expected: &'static [usize]) -> Self {
                Self {
                    entry,
                    len,
                    expected,
                }
            }
        }

        let tests = [
            // Lenght 1
            (
                vec![0],
                vec![
                    // Length 1
                    TestCase::new(0, 1, &[]),
                    TestCase::new(1, 1, &[]), // out-of-bounds
                    // Length 2
                    TestCase::new(0, 2, &[]),
                    TestCase::new(1, 2, &[]), // out-of-bounds
                ],
            ),
            (
                vec![0, 1],
                vec![
                    // Length 1
                    TestCase::new(0, 1, &[1]),
                    TestCase::new(1, 1, &[0]),
                    TestCase::new(2, 1, &[0]), // out-of-bounds
                    // Length 2 -- same behavior as 1
                    TestCase::new(0, 2, &[1]),
                    TestCase::new(1, 2, &[0]),
                    TestCase::new(2, 2, &[0]), // out-of-bounds
                ],
            ),
            (
                vec![0, 1, 2],
                vec![
                    // Length 1
                    TestCase::new(0, 1, &[2]),
                    TestCase::new(1, 1, &[0]),
                    TestCase::new(2, 1, &[1]),
                    TestCase::new(3, 1, &[1]),
                    // Length 2
                    TestCase::new(0, 2, &[2, 1]),
                    TestCase::new(1, 2, &[0, 2]),
                    TestCase::new(2, 2, &[1, 0]),
                    TestCase::new(3, 2, &[1, 0]), // out-of-bounds
                    // Length 3 - same behavior as 2
                    TestCase::new(0, 2, &[2, 1]),
                    TestCase::new(1, 2, &[0, 2]),
                    TestCase::new(2, 2, &[1, 0]),
                    TestCase::new(3, 2, &[1, 0]), // out-of-bounds
                ],
            ),
            (
                vec![0, 1, 2, 3],
                vec![
                    // Lenght 1
                    TestCase::new(0, 1, &[3]),
                    TestCase::new(1, 1, &[0]),
                    TestCase::new(2, 1, &[1]),
                    TestCase::new(3, 1, &[2]),
                    TestCase::new(4, 1, &[2]), // test out-of-bounds truncation
                    TestCase::new(5, 1, &[2]), // test out-of-bounds turncation
                    // Length 2
                    TestCase::new(0, 2, &[3, 1]),
                    TestCase::new(1, 2, &[0, 2]),
                    TestCase::new(2, 2, &[1, 3]),
                    TestCase::new(3, 2, &[2, 0]),
                    TestCase::new(4, 2, &[2, 0]), // test out-of-bounds truncation
                    TestCase::new(5, 2, &[2, 0]), // test out-of-bounds truncation
                    // Length 3
                    TestCase::new(0, 3, &[2, 3, 1]),
                    TestCase::new(1, 3, &[3, 0, 2]),
                    TestCase::new(2, 3, &[0, 1, 3]),
                    TestCase::new(3, 3, &[1, 2, 0]),
                    TestCase::new(4, 3, &[1, 2, 0]), // test out-of-bounds truncation
                    TestCase::new(5, 3, &[1, 2, 0]), // test out-of-bounds truncation
                    // Length 4 - same behavior as 3
                    TestCase::new(0, 4, &[2, 3, 1]),
                    TestCase::new(1, 4, &[3, 0, 2]),
                    TestCase::new(2, 4, &[0, 1, 3]),
                    TestCase::new(3, 4, &[1, 2, 0]),
                    TestCase::new(4, 4, &[1, 2, 0]), // test out-of-bounds truncation
                    TestCase::new(5, 4, &[1, 2, 0]), // test out-of-bounds truncation
                ],
            ),
        ];

        for (source, cases) in tests {
            println!("source = {:?}", source);
            for TestCase {
                entry,
                len,
                expected,
            } in cases
            {
                println!(
                    "entry = {}, len = {}, expected = {:?}",
                    entry, len, expected
                );
                let itr = around(&source, entry, len);

                let expected_len = expected.len();

                // ExactSizeIter
                assert_eq!(itr.len(), expected_len);
                assert_eq!(itr.size_hint(), (expected_len, Some(expected_len)));

                let v: Vec<_> = itr.copied().collect();
                assert_eq!(&*v, expected);
            }
        }
    }

    #[test]
    #[should_panic(expected = "range is invalid for data")]
    fn test_arc_chunk_new_panics() {
        let x: Arc<[usize]> = Arc::new([]);
        let _: ArcChunk<usize> = ArcChunk::new(x, 0..1);
    }

    fn test_partitioning<F>(get_ranges: F)
    where
        F: Fn(usize, NonZeroUsize) -> Vec<Range<usize>>,
    {
        // One Task.
        for nitems in [0, 1, 10, 20, 100] {
            let ntasks = NonZeroUsize::new(1).unwrap();
            let ranges = get_ranges(nitems, ntasks);
            assert_eq!(ranges.len(), 1);
            assert_eq!(ranges[0], 0..nitems);
        }

        // Two Tasks.
        {
            let ntasks = NonZeroUsize::new(2).unwrap();

            let ranges = get_ranges(10, ntasks);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[0], 0..5);
            assert_eq!(ranges[1], 5..10);

            let ranges = get_ranges(11, ntasks);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[0], 0..6);
            assert_eq!(ranges[1], 6..11);

            let ranges = get_ranges(12, ntasks);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[0], 0..6);
            assert_eq!(ranges[1], 6..12);

            let ranges = get_ranges(1, ntasks);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[0], 0..1);
            assert_eq!(ranges[1], 1..1);

            let ranges = get_ranges(0, ntasks);
            assert_eq!(ranges.len(), 2);
            assert_eq!(ranges[0], 0..0);
            assert_eq!(ranges[1], 0..0);
        }

        // Three Tasks.
        {
            let ntasks = NonZeroUsize::new(3).unwrap();
            let ranges = get_ranges(0, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..0);
            assert_eq!(ranges[1], 0..0);
            assert_eq!(ranges[2], 0..0);

            let ranges = get_ranges(1, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..1);
            assert!(ranges[1].is_empty());
            assert!(ranges[2].is_empty());

            let ranges = get_ranges(2, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..1);
            assert_eq!(ranges[1], 1..2);
            assert!(ranges[2].is_empty());

            let ranges = get_ranges(3, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..1);
            assert_eq!(ranges[1], 1..2);
            assert_eq!(ranges[2], 2..3);

            let ranges = get_ranges(4, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..2);
            assert_eq!(ranges[1], 2..3);
            assert_eq!(ranges[2], 3..4);

            let ranges = get_ranges(10, ntasks);
            assert_eq!(ranges.len(), 3);
            assert_eq!(ranges[0], 0..4);
            assert_eq!(ranges[1], 4..7);
            assert_eq!(ranges[2], 7..10);
        }

        // Check Coverage.
        for nitems in 0..100 {
            for ntasks in 1..100 {
                let ntasks = NonZeroUsize::new(ntasks).unwrap();
                let ranges = get_ranges(nitems, ntasks);
                let base_len = ranges[0].len();

                let mut seen = HashSet::new();
                for (task, range) in ranges.iter().enumerate() {
                    assert!(
                        range.len() == base_len || range.len() + 1 == base_len,
                        "Range {} has a length of {} while the base range has length {}\
                         for nitems = {}, ntasks = {}",
                        task,
                        range.len(),
                        base_len,
                        nitems,
                        ntasks,
                    );

                    for r in range.clone() {
                        if !seen.insert(r) {
                            panic!(
                                "Saw {} twice for nitems = {}, ntasks = {} for task {}, \
                                 which returned range {:?}",
                                r, nitems, ntasks, task, range
                            );
                        }
                    }
                }

                // Ensure that all items were seen.
                for r in 0..nitems {
                    if !seen.contains(&r) {
                        panic!(
                            "Did not see item {} for nitems = {}, ntasks = {}",
                            r, nitems, ntasks
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_partition() {
        // Test the partition function
        test_partitioning(|nitems, ntasks| {
            (0..ntasks.get())
                .map(|task| partition(nitems, ntasks, task).unwrap())
                .collect()
        });

        // Test error cases
        for nitems in [0, 1, 10] {
            for ntask_count in 1..3 {
                let ntasks = NonZeroUsize::new(ntask_count).unwrap();
                assert!(partition(nitems, ntasks, ntasks.get()).is_err());
            }
        }
    }

    #[test]
    fn test_partition_iter() {
        // Test the PartitionIter
        test_partitioning(|nitems, ntasks| PartitionIter::new(nitems, ntasks).collect());

        // Test ExactSizeIterator implementation
        {
            let ntasks = NonZeroUsize::new(4).unwrap();
            let iter = PartitionIter::new(30, ntasks);
            assert_eq!(iter.len(), 4);

            let mut iter = PartitionIter::new(30, ntasks);
            assert_eq!(iter.len(), 4);
            iter.next();
            assert_eq!(iter.len(), 3);
        }
    }
}
