/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared helpers for regional split implementations.
/// Items stored contiguously in consecutive buckets, each with a key.
///
/// Invariants: `items` and `keys` have equal lengths. `offsets` starts at zero, is
/// non-decreasing, and ends at most at `items.len()`.
pub(super) struct Buckets<I> {
    items: Vec<I>,
    keys: Vec<usize>,
    offsets: Vec<usize>,
}

impl<I: Copy> Buckets<I> {
    pub(super) fn new() -> Self {
        Self {
            items: Vec::new(),
            keys: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Number of sealed buckets.
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Add `item` with `key` to the open bucket.
    pub(super) fn push(&mut self, item: I, key: usize) {
        self.items.push(item);
        self.keys.push(key);
    }

    /// Close the open bucket.
    pub(super) fn seal(&mut self) {
        self.offsets.push(self.items.len());
    }

    /// The items in `bucket`, or `None` if it is not sealed.
    #[cfg_attr(not(test), expect(dead_code, reason = "split uses unchecked access"))]
    pub(super) fn get(&self, bucket: usize) -> Option<&[I]> {
        if bucket < self.len() {
            // SAFETY: `bucket < self.len()` was just checked.
            unsafe { Some(self.get_unchecked(bucket)) }
        } else {
            None
        }
    }

    /// # Safety
    ///
    /// `bucket` must be less than [`Self::len`].
    pub(super) unsafe fn get_unchecked(&self, bucket: usize) -> &[I] {
        debug_assert!(bucket < self.len());
        // SAFETY: `bucket + 1 < offsets.len()` by the caller's contract. The offsets
        // invariant makes `start <= end <= items.len()`.
        unsafe {
            let start = *self.offsets.get_unchecked(bucket);
            let end = *self.offsets.get_unchecked(bucket + 1);
            self.items.get_unchecked(start..end)
        }
    }

    /// Stable regrouping into `count` buckets by key; items keep their keys.
    ///
    /// # Panics
    ///
    /// Panics if a key is not less than `count`.
    pub(super) fn regroup(&self, count: usize) -> Self {
        assert_eq!(self.items.len(), self.keys.len());

        let mut offsets = vec![0; count + 1];
        for &key in &self.keys {
            offsets[1..][key] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let mut next = offsets.clone();
        let mut items = self.items.clone();
        let mut keys = self.keys.clone();
        for (&item, &key) in self.items.iter().zip(&self.keys) {
            // SAFETY: Counting checked `key < count`, so `key < next.len()`.
            // Key `key` occurs exactly `offsets[key + 1] - offsets[key]` times,
            // so its cursor stays below`offsets[key + 1] <= offsets[count] == items.len() == keys.len()`.
            unsafe {
                let slot = next.get_unchecked_mut(key);
                *items.get_unchecked_mut(*slot) = item;
                *keys.get_unchecked_mut(*slot) = key;
                *slot += 1;
            }
        }
        Self {
            items,
            keys,
            offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Buckets;

    fn buckets(buckets: &Buckets<u32>) -> Vec<Vec<u32>> {
        (0..buckets.len())
            .map(|i| buckets.get(i).unwrap().to_vec())
            .collect()
    }

    #[test]
    fn is_stable() {
        let mut moved = Buckets::new();
        for (item, key) in [(10, 2), (11, 0), (12, 2)] {
            moved.push(item, key);
        }
        moved.seal();
        moved.seal();
        moved.push(13, 2);
        moved.seal();

        assert_eq!(buckets(&moved), [vec![10, 11, 12], vec![], vec![13]]);
        let incoming = moved.regroup(4);
        assert_eq!(
            buckets(&incoming),
            [vec![11], vec![], vec![10, 12, 13], vec![]]
        );
        assert_eq!(incoming.keys, [0, 2, 2, 2]);
        assert!(moved.get(3).is_none());
        assert!(incoming.get(4).is_none());
    }

    #[test]
    #[should_panic]
    fn rejects_out_of_range() {
        let mut moved = Buckets::new();
        moved.push(10_u32, usize::MAX);
        moved.seal();
        let _ = moved.regroup(1);
    }
}
