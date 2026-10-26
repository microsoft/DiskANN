/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lists taking part in one regional split, the neighborhood of each parent, and the
//! bucketed storage that split policies share.

use std::{hash::Hash, iter::FusedIterator, slice};

use hashbrown::{HashMap, hash_map::Entry};

/// Distinct parents and retained neighbors in one split, with each parent's neighborhood.
///
/// Parents and neighbors are disjoint and kept in insertion order. Each neighborhood
/// lists distinct neighbors in link order. Splits can use neighborhoods to restrict
/// which lists exchange points.
#[derive(Debug, Clone)]
pub(crate) struct Plan<L> {
    parents: Vec<L>,
    neighbors: Vec<L>,
    /// Neighborhood of every parent, keyed by parent.
    links: HashMap<L, Vec<L>>,
}

impl<L> Plan<L>
where
    L: Copy + Eq + Hash,
{
    /// Start a plan for `parents`. Repeated parents are recorded once.
    pub(crate) fn new<P: IntoIterator<Item = L>>(parents: P) -> Self {
        let parents = parents.into_iter();
        let capacity = parents.size_hint().0;

        let mut plan = Self {
            parents: Vec::with_capacity(capacity),
            neighbors: Vec::new(),
            links: HashMap::with_capacity(capacity),
        };

        for parent in parents {
            if let Entry::Vacant(entry) = plan.links.entry(parent) {
                entry.insert(Vec::new());
                plan.parents.push(parent);
            }
        }

        plan
    }

    /// Record `list` in `parent`'s neighborhood.
    ///
    /// # Panics
    ///
    /// Panics if `parent` is not a parent
    pub(crate) fn link(&mut self, parent: L, list: L) {
        assert!(!self.links.contains_key(&list),);

        if let Some(n) = self.links.get_mut(&parent)
            && !n.contains(&list)
        {
            n.push(list);

            if !self.neighbors.contains(&list) {
                self.neighbors.push(list);
            }
        }
    }

    pub(crate) fn parents(&self) -> &[L] {
        &self.parents
    }

    pub(crate) fn neighbors(&self) -> &[L] {
        &self.neighbors
    }

    /// Every list in the plan: parents, then neighbors.
    pub(crate) fn lists(&self) -> Lists<'_, L> {
        Lists(self.parents.iter(), self.neighbors.iter())
    }

    /// Neighbors linked to `parent`, or `None` if it is not a parent.
    pub(crate) fn neighborhood(&self, parent: L) -> Option<&[L]> {
        self.links.get(&parent).map(Vec::as_slice)
    }
}

/// Iterator over a [`Plan`]'s parents, then its neighbors.
#[derive(Debug, Clone)]
pub(crate) struct Lists<'a, L>(slice::Iter<'a, L>, slice::Iter<'a, L>);

impl<L: Copy> Iterator for Lists<'_, L> {
    type Item = L;

    fn next(&mut self) -> Option<L> {
        self.0.next().or_else(|| self.1.next()).copied()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<L: Copy> ExactSizeIterator for Lists<'_, L> {
    fn len(&self) -> usize {
        self.0.len() + self.1.len()
    }
}

impl<L: Copy> FusedIterator for Lists<'_, L> {}

/////////////
// Buckets //
/////////////

/// Items stored contiguously in consecutive buckets, each with a key.
///
/// Invariants:
/// - `items` and `keys` have equal lengths.
/// - `offsets` starts at zero, is non-decreasing, and ends at most at `items.len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
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
        debug_assert_eq!(self.items.len(), self.keys.len());

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
    use super::{Buckets, Plan};

    #[test]
    fn neighborhoods_record_distinct_links_in_order() {
        let mut plan = Plan::new([5_u32, 3, 5]);
        for (parent, list) in [(5, 9), (3, 7), (5, 7), (5, 9)] {
            plan.link(parent, list);
        }

        assert_eq!(plan.parents(), [5, 3]);
        assert_eq!(plan.neighbors(), [9, 7]);
        assert_eq!(plan.neighborhood(5), Some(&[9, 7][..]));
        assert_eq!(plan.neighborhood(3), Some(&[7][..]));
        assert_eq!(plan.neighborhood(9), None);
    }

    #[test]
    fn parents_without_links_have_empty_neighborhoods() {
        let plan = Plan::new([1_u32, 2]);
        assert_eq!(plan.parents(), [1, 2]);
        assert!(plan.neighbors().is_empty());
        assert_eq!(plan.neighborhood(1), Some(&[][..]));

        let plan = Plan::<u32>::new([]);
        assert!(plan.parents().is_empty());
        assert_eq!(plan.neighborhood(0), None);
    }

    #[test]
    #[should_panic(expected = "IVF plans can only link from a parent")]
    fn links_require_a_parent() {
        let mut plan = Plan::new([1_u32]);
        plan.link(1, 2);
        plan.link(2, 3);
    }

    #[test]
    #[should_panic(expected = "IVF plans cannot link two parents")]
    fn parents_cannot_link_parents() {
        let mut plan = Plan::new([1_u32, 2]);
        plan.link(1, 2);
    }

    #[test]
    #[should_panic(expected = "IVF plans cannot link two parents")]
    fn parents_cannot_link_themselves() {
        let mut plan = Plan::new([1_u32]);
        plan.link(1, 1);
    }

    #[test]
    fn lists_yield_parents_then_neighbors() {
        let plan = Plan {
            parents: vec![5_u32, 3],
            neighbors: vec![9, 7],
            links: Default::default(),
        };
        let mut lists = plan.lists();
        assert_eq!(lists.len(), 4);
        assert_eq!(lists.clone().collect::<Vec<_>>(), [5, 3, 9, 7]);

        assert_eq!(lists.nth(2), Some(9));
        assert_eq!(lists.len(), 1);
        assert_eq!(lists.next(), Some(7));
        assert_eq!(lists.next(), None);
        assert_eq!(lists.next(), None);
        assert_eq!(lists.size_hint(), (0, Some(0)));

        assert_eq!(Plan::<u32>::new([]).lists().len(), 0);
    }

    fn buckets(buckets: &Buckets<u32>) -> Vec<Vec<u32>> {
        (0..buckets.len())
            .map(|i| buckets.get(i).unwrap().to_vec())
            .collect()
    }

    #[test]
    fn buckets_regroup_stably() {
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
    fn buckets_reject_out_of_range_keys() {
        let mut moved = Buckets::new();
        moved.push(10_u32, usize::MAX);
        moved.seal();
        let _ = moved.regroup(1);
    }
}
