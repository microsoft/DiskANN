/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lists taking part in one regional split, the neighborhood of each parent, and the
//! bucketed storage that split policies share.

use std::{hash::Hash, iter::FusedIterator, slice};

use hashbrown::{HashMap, hash_map::Entry};

/// Distinct parents and retained neighbors in one split, and the links between them.
///
/// Lists are identified by their position in [`Self::lists`]: parents first, then
/// neighbors, each in insertion order. Links are symmetric and only join a parent to a
/// neighbor: a parent links to the neighbors selected for it, and a neighbor links to
/// the parents that selected it.
#[derive(Debug, Clone)]
pub(crate) struct Plan<L> {
    parents: Vec<L>,
    neighbors: Vec<L>,
    /// Position of every list.
    positions: HashMap<L, usize>,
    /// Linked positions of every list, in link order.
    /// linking parent index -> indexes of neighbors linked  
    parent_links: Vec<Vec<usize>>,
    /// linking neighbor index -> indexes of parents linked
    neighbor_links: Vec<Vec<usize>>,
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
            positions: HashMap::with_capacity(capacity),
            parent_links: Vec::with_capacity(capacity),
            neighbor_links: Vec::with_capacity(capacity),
        };

        for parent in parents {
            if let Entry::Vacant(entry) = plan.positions.entry(parent) {
                entry.insert(plan.parent_links.len());

                plan.parents.push(parent);
                plan.parent_links.push(Vec::new());
            }
        }

        plan
    }

    pub(crate) fn parents(&self) -> &[L] {
        &self.parents
    }

    pub(crate) fn neighbors(&self) -> &[L] {
        &self.neighbors
    }

    /// Positions linked to the list at position `s`, in link order.
    ///
    /// # Panics
    ///
    /// Panics if `s` is not less than `self.lists().len()`.
    pub(crate) fn parent_links(&self, s: usize) -> &[usize] {
        &self.parent_links[s]
    }

    pub(crate) fn neighbor_links(&self, s: usize) -> &[usize] {
        &self.neighbor_links[s]
    }

    /// Link `parent` and `list`; a list not seen before becomes a neighbor.
    ///
    /// Repeated links are recorded once.
    ///
    /// # Panics
    ///
    /// Panics if `parent` is not a parent, or if `list` is a parent.
    pub(crate) fn link(&mut self, parent: L, list: L) {
        let i = self.positions.get(&parent).copied().unwrap_or(usize::MAX);
        assert!(
            i < self.parents.len(),
            "IVF plans can only link from a parent"
        );

        let j = match self.positions.entry(list) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(self.neighbor_links.len());
                self.neighbors.push(list);
                self.neighbor_links.push(Vec::new());

                self.neighbor_links.len() - 1
            }
        };

        if !self.parent_links[i].contains(&j) {
            self.parent_links[i].push(j);
            self.neighbor_links[j].push(i);
        }
    }
}

/////////////
// Buckets //
/////////////

/// Destination of a moved point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dst {
    /// Fitted child, by row.
    Child(usize),
    /// Retained neighbor, by index in [`Plan::neighbors`].
    Neighbor(usize),
}

/// Dense numbering for typed bucket keys.
pub(super) trait Layout: Copy {
    type Key: Copy;

    /// Number of buckets.
    fn len(&self) -> usize;

    /// Bucket number of `key`, below [`Self::len`].
    ///
    /// Implementations should panic for keys outside the layout rather than alias
    /// another key's bucket.
    fn index(&self, key: Self::Key) -> usize;
}

/// Destinations of one split: retained neighbors, then children.
#[derive(Debug, Clone, Copy)]
pub(super) struct Destinations {
    pub(super) neighbors: usize,
    pub(super) children: usize,
}

impl Layout for Destinations {
    type Key = Dst;

    fn len(&self) -> usize {
        self.neighbors + self.children
    }

    /// # Panics
    ///
    /// Panics if the neighbor or child is out of range.
    fn index(&self, key: Dst) -> usize {
        match key {
            Dst::Neighbor(n) => {
                assert!(n < self.neighbors, "neighbor {n} is out of range");
                n
            }
            Dst::Child(c) => {
                assert!(c < self.children, "child {c} is out of range");
                self.neighbors + c
            }
        }
    }
}

/// Items recorded in consecutive source buckets, each destined for a key of `L`.
///
/// Invariants:
/// - `items` and `keys` have equal lengths.
/// - `offsets` starts at zero, is non-decreasing, and ends at most at `items.len()`.
#[derive(Debug, Clone)]
pub(super) struct Buckets<I, L> {
    layout: L,
    items: Vec<I>,
    /// Bucket number of each item's key.
    keys: Vec<usize>,
    offsets: Vec<usize>,
}

impl<I: Copy, L: Layout> Buckets<I, L> {
    pub(super) fn new(layout: L) -> Self {
        Self {
            layout,
            items: Vec::new(),
            keys: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Number of sealed source buckets.
    pub(super) fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Add `item`, destined for `key`, to the open source bucket.
    ///
    /// # Panics
    ///
    /// Panics if `key` is outside the layout.
    pub(super) fn push(&mut self, item: I, key: L::Key) {
        let index = self.layout.index(key);

        self.items.push(item);
        self.keys.push(index);
    }

    /// Close the open source bucket.
    pub(super) fn seal(&mut self) {
        self.offsets.push(self.items.len());
    }

    /// Stable grouping of every recorded item, sealed or not, by key.
    pub(super) fn group(&self) -> Groups<I, L> {
        debug_assert_eq!(self.items.len(), self.keys.len());

        let mut offsets = vec![0; self.layout.len() + 1];
        for &key in &self.keys {
            offsets[key + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }

        let mut next = offsets.clone();
        let mut items = self.items.clone();

        for (&item, &key) in self.items.iter().zip(&self.keys) {
            // SAFETY: Counting indexed `offsets[key + 1]`, so `key < next.len()`. Key `key`
            // occurs exactly `offsets[key + 1] - offsets[key]` times, so its cursor stays
            // below `offsets[key + 1] <= items.len()`.
            unsafe {
                let slot = next.get_unchecked_mut(key);
                *items.get_unchecked_mut(*slot) = item;
                *slot += 1;
            }
        }

        Groups {
            layout: self.layout.clone(),
            items,
            offsets,
        }
    }
}

/// Items grouped by key; see [`Buckets::regroup`].
#[derive(Debug)]
pub(super) struct Groups<I, L> {
    layout: L,
    items: Vec<I>,
    offsets: Vec<usize>,
}

impl<I, L: Layout> Groups<I, L> {
    /// Items destined for `key`, in recording order.
    ///
    /// # Panics
    ///
    /// Panics if `key` is outside the layout.
    pub(super) fn get(&self, key: L::Key) -> &[I] {
        let index = self.layout.index(key);
        // SAFETY: index correctly returned

        let start = self.offsets[index];
        let end = self.offsets[index + 1];

        &self.items[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::{Buckets, Destinations, Dst, Layout, Plan};

    #[test]
    fn links_are_symmetric_distinct_and_in_order() {
        let mut plan = Plan::new([5_u32, 3, 5]);
        for (parent, list) in [(5, 9), (3, 7), (5, 7), (5, 9)] {
            plan.link(parent, list);
        }

        assert_eq!(plan.parents(), [5, 3]);
        assert_eq!(plan.neighbors(), [9, 7]);
        assert_eq!(plan.parent_links(0), [0, 1]);
        assert_eq!(plan.parent_links(1), [1]);
        assert_eq!(plan.neighbor_links(0), [0]);
        assert_eq!(plan.neighbor_links(1), [1, 0]);
    }

    #[test]
    fn parents_without_links_have_none() {
        let plan = Plan::new([1_u32, 2]);
        assert_eq!(plan.parents(), [1, 2]);
        assert!(plan.neighbors().is_empty());
        assert!(plan.parent_links(1).is_empty());
        assert!(Plan::<u32>::new([]).parents().is_empty());
    }

    #[test]
    #[should_panic(expected = "IVF plans can only link from a parent")]
    fn links_require_a_parent() {
        let mut plan = Plan::new([1_u32]);
        plan.link(2, 3);
    }

    #[test]
    fn buckets_group_stably_by_destination() {
        let mut moved = Buckets::new(Destinations {
            neighbors: 2,
            children: 2,
        });
        for (item, key) in [
            (10, Dst::Child(0)),
            (11, Dst::Neighbor(0)),
            (12, Dst::Child(0)),
        ] {
            moved.push(item, key);
        }
        moved.seal();
        moved.seal();
        moved.push(13, Dst::Child(0));
        moved.seal();
        moved.push(14, Dst::Neighbor(1));
        assert_eq!(moved.len(), 3);

        // Unsealed items are grouped too.
        let incoming = moved.group();
        assert_eq!(incoming.get(Dst::Neighbor(0)), [11]);
        assert_eq!(incoming.get(Dst::Neighbor(1)), [14]);
        assert_eq!(incoming.get(Dst::Child(0)), [10, 12, 13]);
        assert!(incoming.get(Dst::Child(1)).is_empty());
    }

    #[test]
    #[should_panic(expected = "neighbor 2 is out of range")]
    fn destinations_reject_out_of_range_neighbors() {
        let mut moved = Buckets::new(Destinations {
            neighbors: 2,
            children: 2,
        });
        moved.push(10_u32, Dst::Neighbor(2));
    }

    #[test]
    #[should_panic(expected = "child 2 is out of range")]
    fn destinations_reject_out_of_range_children() {
        let mut moved = Buckets::new(Destinations {
            neighbors: 2,
            children: 2,
        });
        moved.push(10_u32, Dst::Child(2));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn grouping_rejects_keys_outside_the_layout() {
        #[derive(Clone, Copy)]
        struct Broken;

        impl Layout for Broken {
            type Key = ();

            fn len(&self) -> usize {
                1
            }

            fn index(&self, (): ()) -> usize {
                1
            }
        }

        let mut moved = Buckets::new(Broken);
        moved.push(10_u32, ());
        let _ = moved.group();
    }
}
