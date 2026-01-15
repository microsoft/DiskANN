/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Sorted Neighbor Vector

use std::ops::Deref;

use crate::neighbor::Neighbor;

/// A utility that asserts the contained neighbors are sorted by distance.
#[derive(Debug)]
pub struct SortedNeighbors<'a, I>(&'a [Neighbor<I>])
where
    I: Default + Eq;

impl<'a, I> SortedNeighbors<'a, I>
where
    I: Default + Eq + std::fmt::Debug,
{
    /// Create a new `SortedNeighbors` around `neighbors` truncated to `max` length.
    ///
    /// As a by-product calling this method, `neighbors` will be resized to at most
    /// `max` and be sorted.
    pub fn new(neighbors: &'a mut Vec<Neighbor<I>>, max: usize) -> Self {
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
            let (prefix, _, _) = neighbors.select_nth_unstable(position);
            prefix.sort_unstable()
        }

        neighbors.truncate(max);
        Self(&*neighbors)
    }
}

impl<I> Deref for SortedNeighbors<'_, I>
where
    I: Default + Eq,
{
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
                assert_eq!(&sorted[..expected_len], &reference[..expected_len],);

                // Changes are visible on the taken vector.
                assert_eq!(shuffled.len(), expected_len)
            }
        }
    }
}
