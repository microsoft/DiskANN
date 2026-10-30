/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lists taking part in reassignment and the links between them.

use std::hash::Hash;

use hashbrown::{HashMap, hash_map::Entry};

/// Distinct lists in insertion order and the symmetric links between them.
///
/// Lists are identified by their index in [`Self::lists`]. During reassignment, a list's
/// points may move only to lists it links to.
#[derive(Debug, Clone)]
pub(crate) struct Links<L> {
    ids: Vec<L>,
    /// Index of every list in `ids`.
    positions: HashMap<L, usize>,
    /// Linked indices of every list, in link order.
    links: Vec<Vec<usize>>,
}

impl<L> Links<L>
where
    L: Copy + Eq + Hash,
{
    /// Start with `lists` and no links. Repeated lists are recorded once.
    pub(crate) fn new<P: IntoIterator<Item = L>>(lists: P) -> Self {
        let lists = lists.into_iter();
        let capacity = lists.size_hint().0;

        let mut links = Self {
            ids: Vec::with_capacity(capacity),
            positions: HashMap::with_capacity(capacity),
            links: Vec::with_capacity(capacity),
        };
        for list in lists {
            links.insert(list);
        }

        links
    }

    pub(crate) fn lists(&self) -> &[L] {
        &self.ids
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Indices linked to the list at index `i`, in link order.
    ///
    /// # Panics
    ///
    /// Panics if `i` is not less than `self.lists().len()`.
    pub(crate) fn links(&self, i: usize) -> &[usize] {
        &self.links[i]
    }

    /// Link `from` and `to` in both directions; a list not seen before is added.
    ///
    /// Repeated links are recorded once.
    ///
    /// # Panics
    ///
    /// Panics if `from` is not recorded, or if `from` and `to` are the same list.
    pub(crate) fn link(&mut self, from: L, to: L) {
        let i = self.positions.get(&from).copied();
        assert!(i.is_some(), "Links can only link from a recorded list");
        assert!(from != to, "Links cannot link a list to itself");

        let (i, j) = (i.unwrap_or_default(), self.insert(to));

        if !self.links[i].contains(&j) {
            self.links[i].push(j);
            self.links[j].push(i);
        }
    }

    /// The index of `id`, added at the end if it is new.
    fn insert(&mut self, id: L) -> usize {
        match self.positions.entry(id) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                entry.insert(self.ids.len());
                self.ids.push(id);
                self.links.push(Vec::new());

                self.ids.len() - 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Links;

    #[test]
    fn links_are_symmetric_distinct_and_in_order() {
        let mut links = Links::new([5_u32, 3, 5]);
        for (from, to) in [(5, 9), (3, 7), (5, 7), (5, 9), (7, 3), (3, 5)] {
            links.link(from, to);
        }

        assert_eq!(links.lists(), [5, 3, 9, 7]);
        assert_eq!(links.links(0), [2, 3, 1]);
        assert_eq!(links.links(1), [3, 0]);
        assert_eq!(links.links(2), [0]);
        assert_eq!(links.links(3), [1, 0]);
    }

    #[test]
    fn lists_without_links_have_none() {
        let links = Links::new([1_u32, 2]);
        assert_eq!(links.lists(), [1, 2]);
        assert!(links.links(1).is_empty());
        assert!(Links::<u32>::new([]).lists().is_empty());
    }

    #[test]
    #[should_panic(expected = "Links can only link from a recorded list")]
    fn links_require_a_recorded_list() {
        let mut links = Links::new([1_u32]);
        links.link(2, 3);
    }

    #[test]
    #[should_panic(expected = "Links cannot link a list to itself")]
    fn lists_cannot_link_themselves() {
        let mut links = Links::new([1_u32]);
        links.link(1, 1);
    }
}
