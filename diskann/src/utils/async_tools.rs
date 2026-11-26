/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::convert_error;

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
#[derive(Debug, Clone)]
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

convert_error!(PartitionError);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

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

    fn test_partitioning<F>(get_ranges: F)
    where
        F: Fn(usize, NonZeroUsize) -> Vec<std::ops::Range<usize>>,
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
