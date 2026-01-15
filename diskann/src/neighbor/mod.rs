/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Imports
use std::{cmp::Ordering, fmt::Debug};

use crate::graph::{SearchOutputBuffer, search_output_buffer};

// Exports
mod queue;
pub use queue::{NeighborPriorityQueue, NeighborPriorityQueueIdType, NeighborQueue};

#[cfg(feature = "experimental_diversity_search")]
mod diverse_priority_queue;
#[cfg(feature = "experimental_diversity_search")]
pub use diverse_priority_queue::{
    DiverseNeighborQueue, PlaceholderAttributeValueProvider, VectorIdWithAttr,
};

//////////////
// Neighbor //
//////////////

/// Neighbor node
#[derive(Debug, Clone, Copy)]
pub struct Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq,
{
    /// The id of the node
    pub id: VectorIdType,

    /// The distance from the query node to current node
    pub distance: f32,
}

impl<VectorIdType> Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq,
{
    /// Create the neighbor node and it has not been visited
    pub fn new(id: VectorIdType, distance: f32) -> Self {
        Self { id, distance }
    }

    /// Return the contents of `self` as a tuple.
    pub fn as_tuple(self) -> (VectorIdType, f32) {
        (self.id, self.distance)
    }
}

impl<VectorIdType> Default for Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq,
{
    fn default() -> Self {
        Self {
            id: VectorIdType::default(),
            distance: 0.0_f32,
        }
    }
}

impl<VectorIdType> PartialEq for Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<VectorIdType> Eq for Neighbor<VectorIdType> where VectorIdType: Default + Eq {}

/// PERF SENSITIVE: does not do well with comparing item with self.
/// Not doing so, allows for a 1% gain. So use it with care.
impl<VectorIdType> Ord for Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq + Debug,
{
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert!(
            self.id.ne(&other.id),
            "Neighbor id should not be equal: {:?}, {:?}",
            self.id,
            other.id
        );
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// PERF SENSITIVE: does not do well with comparing item with self.
/// Not doing so, allows for a 1% gain. So use it with care.
impl<VectorIdType> PartialOrd for Neighbor<VectorIdType>
where
    VectorIdType: Default + Eq + Debug,
{
    #[inline]
    fn lt(&self, other: &Self) -> bool {
        debug_assert!(
            self.id.ne(&other.id),
            "Neighbor id should not be equal: {:?}, {:?}",
            self.id,
            other.id
        );
        self.distance < other.distance
    }

    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I> SearchOutputBuffer<I> for [Neighbor<I>]
where
    I: Default + Eq,
{
    fn size_hint(&self) -> Option<usize> {
        Some(self.len())
    }

    fn set(
        &mut self,
        i: usize,
        id: I,
        distance: f32,
    ) -> Result<(), search_output_buffer::IndexOutOfBounds> {
        match self.get_mut(i) {
            None => Err(search_output_buffer::IndexOutOfBounds::new(i)),
            Some(slot) => {
                *slot = Neighbor::new(id, distance);
                Ok(())
            }
        }
    }

    fn set_from<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: Iterator<Item = (I, f32)>,
    {
        let mut count = 0;
        std::iter::zip(self.iter_mut(), itr).for_each(|(o, i)| {
            *o = Neighbor::new(i.0, i.1);
            count += 1;
        });
        count
    }
}

#[cfg(test)]
mod neighbor_test {
    use super::*;

    #[test]
    fn eq_lt_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(1, 1.1);

        assert!(n1 != n2);
        assert!(n1 < n2);
        assert!(n1 == n3);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn cmp_same_id_panics() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(1, 1.1);

        // This should panic - since the ids are the same.
        let _: bool = n1 < n2;
    }

    #[test]
    fn gt_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);

        let test = n2 > n1;
        assert!(test);
    }

    #[test]
    fn le_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);

        let test = n1 <= n2;
        assert!(test);
    }

    #[test]
    fn cmp_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(3, 1.1);

        assert_eq!(n1.cmp(&n2), Ordering::Less);
        assert_eq!(n2.cmp(&n1), Ordering::Greater);
        assert_eq!(n1.cmp(&n3), Ordering::Equal);
    }

    #[test]
    fn test_search_output_buffer() {
        fn test_scalar_interface(buffer: &mut [Neighbor<u32>]) {
            let len = buffer.len();
            assert_eq!(buffer.size_hint(), Some(len));
            // All of these should work okay.
            for i in 0..len {
                buffer.set(i, i as u32, i as f32).unwrap();
            }

            // Setting one past the end should yield an error.
            let err = buffer.set(len, 0, 0.0).unwrap_err();
            assert_eq!(err.to_string(), format!("index {} is out-of-bounds", len));

            // Check that the ids and distances are set correctly.
            for (i, n) in buffer.iter().enumerate() {
                assert_eq!(i, n.id as usize);
                assert_eq!(i as f32, n.distance);
            }
        }

        // Scalar Interface
        for len in 0..20 {
            let mut buffer = vec![Neighbor::<u32>::default(); len];
            test_scalar_interface(&mut buffer);
        }

        // Iterator Interface
        for len in 0..10 {
            for input_len in 0..10 {
                let mut buffer = vec![Neighbor::<u32>::default(); len];

                let source: Vec<_> = (0..input_len).map(|i| (i as u32, i as f32)).collect();

                let count = buffer.set_from(source.into_iter());
                // The assigned count should be the minimum of the input and output lengths.
                assert_eq!(count, input_len.min(len));
                for (i, n) in buffer.iter().take(count).enumerate() {
                    assert_eq!(i, n.id as usize);
                    assert_eq!(i as f32, n.distance);
                }

                // THe upper values should be untouched.
                for neighbor in buffer.iter().skip(count) {
                    assert_eq!(neighbor.id, 0);
                    assert_eq!(neighbor.distance, 0.0);
                }
            }
        }
    }
}
