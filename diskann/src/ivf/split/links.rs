/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Lists taking part in reassignment and the links between them.

use std::hash::Hash;

use hashbrown::{HashMap, hash_map::Entry};

/// Distinct lists in insertion order and the symmetric links between them.
///
/// Lists are identified by their index in [`Self::lists`].
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

    /// Return the recorded lists, in insertion order.
    pub(crate) fn lists(&self) -> &[L] {
        &self.ids
    }

    /// Return the number of recorded lists.
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
