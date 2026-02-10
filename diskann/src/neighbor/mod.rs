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
    Attribute, AttributeValueProvider, DiverseNeighborQueue, VectorIdWithAttribute,
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

/// A [`SearchOutputBuffer`] wrapper around `&mut [Neighbor<I>]`. This can be used to
/// populate such a mutable slice as the result of [`crate::graph::DiskANNIndex::search`].
#[derive(Debug)]
pub struct BackInserter<'a, I>
where
    I: Default + Eq,
{
    buffer: &'a mut [Neighbor<I>],
    position: usize,
}

impl<'a, I> BackInserter<'a, I>
where
    I: Default + Eq,
{
    /// Construct a new [`BackInserter`] around the provided slice.
    ///
    /// THe buffer will have a capacity equal to the length of `buffer`.
    pub fn new(buffer: &'a mut [Neighbor<I>]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    /// Return the overall capacity of the buffer buffer.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

impl<I> SearchOutputBuffer<I> for BackInserter<'_, I>
where
    I: Default + Eq,
{
    fn size_hint(&self) -> Option<usize> {
        // We maintain the invariant that `self.position <= self.buffer.len()`, so this
        // subtraction should not underflow.
        Some(self.buffer.len() - self.position)
    }

    fn push(&mut self, id: I, distance: f32) -> search_output_buffer::BufferState {
        if self.position == self.buffer.len() {
            return search_output_buffer::BufferState::Full;
        }

        self.buffer[self.position] = Neighbor::new(id, distance);
        self.position += 1;

        // Return `Full` if we added the last item.
        if self.position == self.buffer.len() {
            search_output_buffer::BufferState::Full
        } else {
            search_output_buffer::BufferState::Available
        }
    }

    fn current_len(&self) -> usize {
        self.position
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (I, f32)>,
    {
        let mut i = 0;
        std::iter::zip(self.buffer.iter_mut().skip(self.position), itr).for_each(
            |(neighbor, (id, distance))| {
                i += 1;
                *neighbor = Neighbor::new(id, distance);
            },
        );

        self.position += i;

        i
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
        const MAX_LENGTH: usize = 5;

        // Helps with typing.
        fn f(i: usize) -> Neighbor<u32> {
            Neighbor::new(i as u32, i as f32)
        }

        // All `push`.
        {
            let mut buffer = [Neighbor::<u32>::default(); MAX_LENGTH];
            let mut inserter = BackInserter::new(&mut buffer);

            assert_eq!(inserter.capacity(), MAX_LENGTH);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH));
            assert_eq!(inserter.current_len(), 0);

            assert!(inserter.push(1, 1.0).is_available());
            assert_eq!(inserter.current_len(), 1);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH - 1));

            assert!(inserter.push(2, 2.0).is_available());
            assert_eq!(inserter.current_len(), 2);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH - 2));

            assert!(inserter.push(3, 3.0).is_available());
            assert_eq!(inserter.current_len(), 3);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH - 3));

            assert!(inserter.push(4, 4.0).is_available());
            assert_eq!(inserter.current_len(), 4);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH - 4));

            // This should error since further attempts will not work.
            assert!(inserter.push(5, 5.0).is_full());
            assert_eq!(inserter.current_len(), 5);
            assert_eq!(inserter.size_hint(), Some(0));

            assert!(inserter.push(6, 6.0).is_full());
            assert_eq!(inserter.current_len(), 5);
            assert_eq!(inserter.size_hint(), Some(0));

            assert_eq!(&buffer, &[f(1), f(2), f(3), f(4), f(5)]);
        }

        // All `iterator`.
        {
            let mut buffer = [Neighbor::<u32>::default(); MAX_LENGTH];
            let mut inserter = BackInserter::new(&mut buffer);
            assert_eq!(inserter.capacity(), MAX_LENGTH);
            assert_eq!(inserter.size_hint(), Some(MAX_LENGTH));
            assert_eq!(inserter.current_len(), 0);

            let set = inserter.extend([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)]);
            assert_eq!(set, MAX_LENGTH);
            assert_eq!(inserter.current_len(), MAX_LENGTH);
            assert_eq!(inserter.size_hint(), Some(0));

            // Ensure that `pushing` respects the limit.
            assert!(inserter.push(7, 7.0).is_full());

            let set = inserter.extend([(10, 10.0), (20, 20.0)]);
            assert_eq!(set, 0, "no more items can be added");

            assert_eq!(&buffer, &[f(1), f(2), f(3), f(4), f(5)]);
        }

        // Mixture
        {
            let mut buffer = [Neighbor::<u32>::default(); MAX_LENGTH];
            let mut inserter = BackInserter::new(&mut buffer);

            assert!(inserter.push(1, 1.0).is_available());

            let set = inserter.extend([(2, 2.0), (3, 3.0)]);
            assert_eq!(set, 2, "only two items were pushed");

            assert_eq!(inserter.current_len(), 3);
            assert_eq!(inserter.size_hint(), Some(2));

            assert!(inserter.push(4, 4.0).is_available());
            assert_eq!(inserter.current_len(), 4);
            assert_eq!(inserter.size_hint(), Some(1));

            let set = inserter.extend([(5, 5.0), (6, 6.0)]);
            assert_eq!(
                set, 1,
                "there should only be room for one more item in the buffer"
            );
            assert_eq!(inserter.current_len(), 5);
            assert_eq!(inserter.size_hint(), Some(0));

            assert_eq!(&buffer, &[f(1), f(2), f(3), f(4), f(5)]);
        }
    }
}
