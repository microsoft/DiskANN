/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sorted Neighbor Vector

use std::ops::Deref;

use crate::neighbor::{self, Neighbor};

/// A utility that asserts the contained neighbors are sorted by distance.
#[derive(Debug)]
pub(crate) struct SortedNeighbors<'a, I>(&'a [Neighbor<I>]);

impl<I> Clone for SortedNeighbors<'_, I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for SortedNeighbors<'_, I> {}

impl<'a, I> SortedNeighbors<'a, I> {
    /// Create a new `SortedNeighbors` around `neighbors` truncated to `max` length.
    ///
    /// As a by-product calling this method, `neighbors` will be resized to at most
    /// `max` and be sorted.
    pub(crate) fn new(neighbors: &'a mut Vec<Neighbor<I>>, max: usize) -> Self {
        // Here- we use `select_nth_unstable` to get the `position` index in the correct
        // location. We can then sort the prefix slice returned by that API.
        //
        // The dance with the index calculation is to ensure we do not hit
        // `select_nth_unstalbe`'s panic condition.
        //
        // If the checked subtraction fails, it's because either `max == 0` or
        // `neighbors.len() == 0`. In either case, the resulting slice will be empty
        // and there's no actual work to be done.
        if let Some(position) = max.min(neighbors.len()).checked_sub(1) {
            let (prefix, _, _) =
                neighbors.select_nth_unstable_by(position, neighbor::ord::fast_distance);
            prefix.sort_unstable_by(neighbor::ord::fast_distance)
        }

        neighbors.truncate(max);
        Self(&*neighbors)
    }

    /// Apply the projection `f` to each element in `self` and store the result in `other`.
    ///
    /// The returned [`SortedNeighbors`] inherits the sorted property from `self`.
    ///
    /// # Side Effects
    ///
    /// This method removes all pre-existing elements from `storage`.
    pub(crate) fn map_in<'b, F, J>(
        self,
        storage: &'b mut Vec<Neighbor<J>>,
        mut f: F,
    ) -> SortedNeighbors<'b, J>
    where
        F: FnMut(&I) -> J,
    {
        storage.clear();
        storage.extend(self.iter().map(|n| Neighbor::new(f(n.id()), *n.distance())));
        SortedNeighbors(storage)
    }
}

impl<I> Deref for SortedNeighbors<'_, I> {
    type Target = [Neighbor<I>];
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::cmp::assert_eq_verbose;

    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};

    #[test]
    fn test_empty_neighbors() {
        let mut v = Vec::<Neighbor<u32>>::new();
        for i in 0..10 {
            let sorted = SortedNeighbors::new(&mut v, i);
            assert!(sorted.is_empty());
        }
    }

    #[test]
    fn test_sorted_neighbors() {
        let reference = [
            Neighbor::new(1, 0.1),
            Neighbor::new(2, 0.2),
            Neighbor::new(3, 0.3),
            Neighbor::new(4, 0.4),
            Neighbor::new(5, 0.5),
            Neighbor::new(6, 0.6),
            Neighbor::new(7, 0.7),
            Neighbor::new(8, 0.8),
            Neighbor::new(9, 0.9),
            Neighbor::new(10, 1.0),
        ];

        let mut rng = StdRng::seed_from_u64(0xd6152fb91c744f54);

        let ntrials = 10;
        for max in 0..reference.len() + 2 {
            for _ in 0..ntrials {
                let mut shuffled = reference.to_vec();
                shuffled.shuffle(&mut rng);

                let sorted = SortedNeighbors::new(&mut shuffled, max);

                let expected_len = reference.len().min(max);
                assert_eq!(sorted.len(), expected_len);
                assert_eq_verbose!(&sorted[..expected_len], &reference[..expected_len]);

                // Changes are visible on the taken vector.
                assert_eq!(shuffled.len(), expected_len)
            }
        }
    }

    #[test]
    fn test_map() {
        let messages = ["a", "b", "c", "d", "e", "f"];

        let mut storage = vec![Neighbor::new("foo", 1.0), Neighbor::new("bar", 0.0)];

        {
            let mut neighbors = vec![
                Neighbor::new(0usize, 5.0f32),
                Neighbor::new(1, 4.0),
                Neighbor::new(2, 3.0),
                Neighbor::new(3, 2.0),
                Neighbor::new(4, 1.0),
                Neighbor::new(5, 0.0),
            ];

            let sorted = SortedNeighbors::new(&mut neighbors, 6);
            let cache = sorted.map_in(&mut storage, |id: &usize| messages[*id]);

            assert_eq_verbose!(
                *cache,
                [
                    Neighbor::new("f", 0.0f32),
                    Neighbor::new("e", 1.0),
                    Neighbor::new("d", 2.0),
                    Neighbor::new("c", 3.0),
                    Neighbor::new("b", 4.0),
                    Neighbor::new("a", 5.0),
                ]
                .as_slice()
            );
        }

        {
            let mut neighbors = Vec::<Neighbor<usize>>::new();
            let sorted = SortedNeighbors::new(&mut neighbors, 10);
            let cache = sorted.map_in(&mut storage, |id: &usize| messages[*id]);
            assert!(cache.is_empty());
        }
    }
}
