/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Flat grouped layouts shared by the partition updates and the online planner.

use std::ops::Range;

/// Values stored in consecutive groups numbered from zero.
///
/// Group `i` owns `values[ends[i - 1]..ends[i]]` and may be empty.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Csr<V> {
    ends: Vec<usize>,
    values: Vec<V>,
}

impl<V> Csr<V> {
    /// Append a group holding `values`.
    pub(super) fn push(&mut self, values: impl IntoIterator<Item = V>) {
        self.values.extend(values);
        self.ends.push(self.values.len());
    }

    /// Number of groups.
    pub(super) fn num_groups(&self) -> usize {
        self.ends.len()
    }

    /// Positions in [`Self::values`] owned by `group`.
    pub(super) fn range(&self, group: usize) -> Range<usize> {
        let start = group.checked_sub(1).map_or(0, |prev| self.ends[prev]);
        start..self.ends[group]
    }

    /// Values of `group`.
    pub(super) fn group(&self, group: usize) -> &[V] {
        &self.values[self.range(group)]
    }

    /// Every value, group by group.
    pub(super) fn values(&self) -> &[V] {
        &self.values
    }
}

impl Csr<usize> {
    /// For each target below `targets`, the groups containing it in ascending order.
    ///
    /// Every value must be below `targets`.
    pub(super) fn transpose(&self, targets: usize) -> Self {
        let mut ends = vec![0; targets];
        for &target in &self.values {
            ends[target] += 1;
        }
        let mut next = Vec::with_capacity(targets);
        let mut total = 0;
        for end in &mut ends {
            next.push(total);
            total += *end;
            *end = total;
        }

        let mut values = vec![0; total];
        for group in 0..self.num_groups() {
            for &target in self.group(group) {
                values[next[target]] = group;
                next[target] += 1;
            }
        }
        Self { ends, values }
    }
}

/// Values grouped under ascending unique keys, ascending within each group.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Grouped<K, V> {
    keys: Box<[K]>,
    groups: Csr<V>,
}

impl<K: Copy + Ord, V: Copy + Ord> Grouped<K, V> {
    /// Group distinct `pairs` by key.
    pub(super) fn from_pairs(mut pairs: Vec<(K, V)>) -> Self {
        pairs.sort_unstable();
        debug_assert!(
            pairs.windows(2).all(|pair| pair[0] != pair[1]),
            "grouped pairs must be distinct"
        );

        let same_key = |a: &(K, V), b: &(K, V)| a.0 == b.0;
        let keys = pairs.chunk_by(same_key).map(|group| group[0].0).collect();
        let ends = pairs
            .chunk_by(same_key)
            .scan(0, |end, group| {
                *end += group.len();
                Some(*end)
            })
            .collect();
        let values = pairs.into_iter().map(|(_, value)| value).collect();
        Self {
            keys,
            groups: Csr { ends, values },
        }
    }

    /// Number of values across all groups.
    pub(super) fn len(&self) -> usize {
        self.groups.values.len()
    }

    /// Group keys, ascending.
    pub(super) fn keys(&self) -> &[K] {
        &self.keys
    }

    /// The values under `key`, or nothing if `key` is absent.
    pub(super) fn get(&self, key: K) -> &[V] {
        self.keys
            .binary_search(&key)
            .map_or(&[], |index| self.groups.group(index))
    }

    /// The groups at positions `range`, with their keys.
    pub(super) fn groups(&self, range: Range<usize>) -> impl ExactSizeIterator<Item = (K, &[V])> {
        range.map(|index| (self.keys[index], self.groups.group(index)))
    }

    /// Every group with its key, in ascending key order.
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = (K, &[V])> {
        self.groups(0..self.keys.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups<V: Clone>(csr: &Csr<V>) -> Vec<Vec<V>> {
        (0..csr.num_groups())
            .map(|group| csr.group(group).to_vec())
            .collect()
    }

    #[test]
    fn csr_push_keeps_empty_groups() {
        let mut csr = Csr::default();
        csr.push([1, 2]);
        csr.push([]);
        csr.push([3]);
        assert_eq!(groups(&csr), vec![vec![1, 2], vec![], vec![3]]);
        assert_eq!(csr.range(2), 2..3);
        assert_eq!(csr.values(), &[1, 2, 3]);
    }

    #[test]
    fn transpose_lists_containing_groups() {
        let mut csr = Csr::default();
        csr.push([1, 2]);
        csr.push([]);
        csr.push([2]);
        let transposed = csr.transpose(4);
        assert_eq!(
            groups(&transposed),
            vec![vec![], vec![0], vec![0, 2], vec![]]
        );
    }

    #[test]
    fn grouped_get_finds_groups_by_key() {
        let grouped = Grouped::from_pairs(vec![(7, 2), (5, 1), (7, 0)]);
        assert_eq!(grouped.keys(), &[5, 7]);
        assert_eq!(grouped.get(7), &[0, 2]);
        assert_eq!(grouped.get(6), &[] as &[i32]);
        assert_eq!(grouped.len(), 3);
    }
}
