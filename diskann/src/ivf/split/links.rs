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

impl<L> Links<L> {
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
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    use super::*;

    /// Check list count, first-seen order and every adjacency list in link order.
    #[track_caller]
    fn assert_links(links: &Links<u32>, expected: &[(u32, Vec<u32>)]) {
        let ids: Vec<u32> = expected.iter().map(|e| e.0).collect();
        assert_eq!(links.lists(), ids);
        assert_eq!(links.len(), expected.len());

        for (i, (id, neighbors)) in expected.iter().enumerate() {
            let actual: Vec<u32> = links.links(i).iter().map(|&j| links.lists()[j]).collect();
            assert_eq!(&actual, neighbors, "links from list {id}");
        }
    }

    #[test]
    fn new_records_distinct_lists_in_first_seen_order() {
        let links = Links::new([5, 3, 5, 7, 3]);
        assert_links(&links, &[(5, vec![]), (3, vec![]), (7, vec![])]);

        assert_links(&Links::new([]), &[]);
    }

    #[test]
    fn link_is_symmetric_and_recorded_once() {
        let mut links = Links::new([1, 2]);
        links.link(1, 2);
        links.link(2, 1);
        links.link(1, 2);
        assert_links(&links, &[(1, vec![2]), (2, vec![1])]);
    }

    #[test]
    fn link_appends_new_targets_in_link_order() {
        let mut links = Links::new([1]);
        links.link(1, 9);
        links.link(1, 4);
        assert_links(&links, &[(1, vec![9, 4]), (9, vec![1]), (4, vec![1])]);

        // A list that `link` added can then link from.
        links.link(9, 4);
        assert_links(&links, &[(1, vec![9, 4]), (9, vec![1, 4]), (4, vec![1, 9])]);
    }

    #[test]
    #[should_panic(expected = "Links can only link from a recorded list")]
    fn link_from_unrecorded_list_panics() {
        Links::new([1u32]).link(2, 1);
    }

    #[test]
    #[should_panic(expected = "Links cannot link a list to itself")]
    fn link_to_itself_panics() {
        Links::new([1u32]).link(1, 1);
    }

    /// Compare random links with a vector-based model, preserving first-seen list and edge
    /// order.
    fn link_fuzz_impl(domain: Uniform<u32>, nlinks: usize, rng: &mut StdRng) {
        let initial: Vec<u32> = (0..4).map(|_| domain.sample(rng)).collect();
        let mut links = Links::new(initial.iter().copied());

        let mut model: Vec<(u32, Vec<u32>)> = Vec::new();
        let position = |model: &[(u32, Vec<u32>)], id: u32| model.iter().position(|e| e.0 == id);
        for &id in &initial {
            if position(&model, id).is_none() {
                model.push((id, Vec::new()));
            }
        }

        for _ in 0..nlinks {
            let from = model[rng.random_range(0..model.len())].0;
            let to = domain.sample(rng);
            if from == to {
                continue;
            }
            links.link(from, to);

            let j = position(&model, to).unwrap_or_else(|| {
                model.push((to, Vec::new()));
                model.len() - 1
            });
            let i = position(&model, from).unwrap();
            if !model[i].1.contains(&to) {
                model[i].1.push(to);
                model[j].1.push(from);
            }
        }

        assert_links(&links, &model);
    }

    #[test]
    fn link_fuzz() {
        let mut rng = StdRng::seed_from_u64(0xa54ff53a5f1d36f1);
        for size in [4, 32] {
            let domain = Uniform::new(0, size).unwrap();
            for _ in 0..10 {
                link_fuzz_impl(domain, 100, &mut rng);
            }
        }
    }
}
