/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

// Imports
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

/// A pairing of opaque ID with a distance.
///
/// For all intents and purposes, this is a simple aggregate with minimal additional semantics.
///
/// # Sorting
///
/// An exceedingly common algorithmic operation is to sort [`Neighbor`]s according to some
/// protocol. Usually, this is by distance (from smallest to largest) ignoring the ID entirely,
/// but certain situations call for different orderings. This raises two problems:
///
/// 1. Accurately specifying the protocol.
/// 2. Dealing with the
///    [ordering semantics](https://doc.rust-lang.org/std/cmp/trait.Ord.html#examples-of-incorrect-ord-implementations)
///    of floating point numbers.
///
/// To that end, the functions in the [`ord`] submodule should be used in combination with
/// the standard library's sorting methods that accept explicit comparison functions like
/// [`std::slice::sort_by`].
///
/// ```rust
/// use diskann::neighbor::{self, Neighbor};
///
/// let mut neighbors = [
///     Neighbor::new("a", 10.0),
///     Neighbor::new("b", 5.0),
///     Neighbor::new("c", 7.0),
///     Neighbor::new("d", 5.0),
/// ];
///
/// // Sort by increasing distance.
/// neighbors.sort_by(neighbor::ord::fast_distance);
/// assert_eq!(
///     neighbors.map(Neighbor::as_tuple),
///     [("b", 5.0), ("d", 5.0), ("c", 7.0), ("a", 10.0)]
/// );
///
/// // Sort reverse distance + IDs.
/// neighbors.sort_by(neighbor::ord::reverse(neighbor::ord::fast_distance_total));
/// assert_eq!(
///     neighbors.map(Neighbor::as_tuple),
///     [("a", 10.0), ("c", 7.0), ("d", 5.0), ("b", 5.0)]
/// );
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct Neighbor<I, D = f32> {
    id: I,
    distance: D,
}

impl<I, D> Neighbor<I, D> {
    /// Create a [`Neighbor`] with `id` and `distance`.
    #[inline]
    pub fn new(id: I, distance: D) -> Self {
        Self { id, distance }
    }

    /// Return the ID and distance in `self` as a tuple.
    #[inline]
    pub fn as_tuple(self) -> (I, D) {
        (self.id, self.distance)
    }

    /// Return the distance.
    #[inline]
    pub fn distance(&self) -> &D {
        &self.distance
    }

    /// Return the ID.
    #[inline]
    pub fn id(&self) -> &I {
        &self.id
    }
}

#[cfg(test)]
impl<I, D> crate::test::cmp::VerboseEq for Neighbor<I, D>
where
    I: crate::test::cmp::VerboseEq,
    D: crate::test::cmp::VerboseEq,
{
    #[inline(never)]
    #[track_caller]
    fn verbose_eq(&self, other: &Self) -> crate::ANNResult<()> {
        if let Err(err) = (self.id).verbose_eq(&other.id) {
            return Err(err.context(crate::test::cmp::Field("id")));
        }

        if let Err(err) = (self.distance).verbose_eq(&other.distance) {
            return Err(err.context(crate::test::cmp::Field("distance")));
        }

        Ok(())
    }
}

pub mod ord {
    //! Methods for ordering [`Neighbor`]s.

    use super::Neighbor;

    /// Return the ordering between `x` and `y` according just to distance.
    ///
    /// This is a fast, semi-approximate method whose behavior is unspecified when either
    /// distance is [`f32::NAN`].
    ///
    /// ```rust
    /// use diskann::neighbor::{Neighbor, ord::fast_distance};
    ///
    /// let x = Neighbor::new(10, 5.0);
    /// let y = Neighbor::new(11, 4.0);
    ///
    /// assert!(fast_distance(&x, &y).is_gt());
    /// assert!(fast_distance(&y, &x).is_lt());
    /// assert!(fast_distance(&x, &x).is_eq());
    ///
    /// let z = Neighbor::new(12, f32::NAN);
    ///
    /// // The following line can return any `Ordering`.
    /// // neighbor::ord::fast_distance(&z, &z);
    /// ```
    pub fn fast_distance<I>(x: &Neighbor<I>, y: &Neighbor<I>) -> std::cmp::Ordering {
        x.distance()
            .partial_cmp(y.distance())
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// Return the ordering between `x` and `y` according to first distance then ID.
    ///
    /// This is a fast, semi-approximate method whose behavior is unspecified when either
    /// distance is [`f32::NAN`].
    ///
    /// ```rust
    /// use diskann::neighbor::{Neighbor, ord::fast_distance_total};
    ///
    /// let x = Neighbor::new(10, 5.0);
    /// let y = Neighbor::new(11, 4.0);
    /// let z = Neighbor::new(12, 4.0);
    ///
    /// // Different distance - ID doesn't matter.
    /// assert!(fast_distance_total(&x, &y).is_gt());
    /// assert!(fast_distance_total(&y, &x).is_lt());
    /// assert!(fast_distance_total(&x, &x).is_eq());
    ///
    /// // Same distance then compares by ID.
    /// assert!(fast_distance_total(&y, &z).is_lt());
    /// assert!(fast_distance_total(&z, &y).is_gt());
    /// ```
    pub fn fast_distance_total<I>(x: &Neighbor<I>, y: &Neighbor<I>) -> std::cmp::Ordering
    where
        I: Ord,
    {
        fast_distance(x, y).then(x.id().cmp(y.id()))
    }

    /// A combinator for comparisons that reverses the ordering.
    ///
    /// This can be useful in situations where higher distances need to be ordered first.
    ///
    /// ```rust
    /// use diskann::neighbor::{Neighbor, ord::{reverse, fast_distance}};
    ///
    /// let mut neighbors = [
    ///     Neighbor::new(1, 1.0),
    ///     Neighbor::new(2, 2.0),
    ///     Neighbor::new(3, 3.0),
    /// ];
    ///
    /// neighbors.sort_by(reverse(fast_distance));
    ///
    /// assert_eq!(
    ///     neighbors.map(Neighbor::as_tuple),
    ///     [(3, 3.0), (2, 2.0), (1, 1.0)]
    /// );
    /// ```
    pub fn reverse<F, I>(f: F) -> impl Fn(&Neighbor<I>, &Neighbor<I>) -> std::cmp::Ordering
    where
        F: Fn(&Neighbor<I>, &Neighbor<I>) -> std::cmp::Ordering,
    {
        move |x: &Neighbor<I>, y: &Neighbor<I>| f(x, y).reverse()
    }
}

/// A [`SearchOutputBuffer`] wrapper around `&mut [Neighbor<I>]`. This can be used to
/// populate such a mutable slice as the result of [`crate::graph::DiskANNIndex::search`].
#[derive(Debug)]
pub struct BackInserter<'a, I> {
    buffer: &'a mut [Neighbor<I>],
    position: usize,
}

impl<'a, I> BackInserter<'a, I> {
    /// Construct a new [`BackInserter`] around the provided slice.
    ///
    /// The buffer will have a capacity equal to the length of `buffer`.
    pub fn new(buffer: &'a mut [Neighbor<I>]) -> Self {
        Self {
            buffer,
            position: 0,
        }
    }

    /// Return the overall capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }
}

impl<I> SearchOutputBuffer<I> for BackInserter<'_, I> {
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

impl<I> SearchOutputBuffer<I> for Vec<Neighbor<I>> {
    fn size_hint(&self) -> Option<usize> {
        None
    }

    fn push(&mut self, id: I, distance: f32) -> search_output_buffer::BufferState {
        self.push(Neighbor::new(id, distance));
        search_output_buffer::BufferState::Available
    }

    fn current_len(&self) -> usize {
        self.len()
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (I, f32)>,
    {
        let before = self.len();
        Extend::extend(
            self,
            itr.into_iter().map(|(id, dist)| Neighbor::new(id, dist)),
        );
        self.len() - before
    }
}

#[cfg(test)]
mod neighbor_test {
    use super::*;

    use crate::test::cmp::assert_eq_verbose;

    #[test]
    fn fast_distance() {
        let n1 = Neighbor::new(1, 1.0);
        let n2 = Neighbor::new(2, 2.0);

        assert!(ord::fast_distance(&n1, &n2).is_lt());
        assert!(ord::fast_distance(&n2, &n1).is_gt());
        assert!(ord::reverse(ord::fast_distance)(&n1, &n2).is_gt());
        assert!(ord::reverse(ord::fast_distance)(&n2, &n1).is_lt());

        assert!(ord::fast_distance(&n1, &n1).is_eq());
        assert!(ord::fast_distance(&n2, &n2).is_eq());
        assert!(ord::reverse(ord::fast_distance)(&n1, &n1).is_eq());
        assert!(ord::reverse(ord::fast_distance)(&n2, &n2).is_eq());

        // The following tests the behavior of NAN.
        //
        // This **must not** be taken as a guarantee of stability for this behavior.
        let nan = Neighbor::new(3, f32::NAN);

        assert!(ord::fast_distance(&n1, &nan).is_eq());
        assert!(ord::fast_distance(&nan, &n1).is_eq());
        assert!(ord::fast_distance(&nan, &nan).is_eq());

        assert!(ord::reverse(ord::fast_distance)(&n1, &nan).is_eq());
        assert!(ord::reverse(ord::fast_distance)(&nan, &n1).is_eq());
        assert!(ord::reverse(ord::fast_distance)(&nan, &nan).is_eq());
    }

    #[test]
    fn fast_distance_total() {
        let n1 = Neighbor::new(1, 1.0);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(3, 2.0);
        let n4 = Neighbor::new(4, 3.0);

        assert!(ord::fast_distance_total(&n1, &n1).is_eq());
        assert!(ord::fast_distance_total(&n1, &n2).is_lt());
        assert!(ord::fast_distance_total(&n1, &n3).is_lt());
        assert!(ord::fast_distance_total(&n1, &n4).is_lt());

        assert!(ord::fast_distance_total(&n2, &n1).is_gt());
        assert!(ord::fast_distance_total(&n2, &n2).is_eq());
        assert!(ord::fast_distance_total(&n2, &n3).is_lt());
        assert!(ord::fast_distance_total(&n2, &n4).is_lt());

        assert!(ord::fast_distance_total(&n3, &n1).is_gt());
        assert!(ord::fast_distance_total(&n3, &n2).is_gt());
        assert!(ord::fast_distance_total(&n3, &n3).is_eq());
        assert!(ord::fast_distance_total(&n3, &n4).is_lt());

        assert!(ord::fast_distance_total(&n4, &n1).is_gt());
        assert!(ord::fast_distance_total(&n4, &n2).is_gt());
        assert!(ord::fast_distance_total(&n4, &n3).is_gt());
        assert!(ord::fast_distance_total(&n4, &n4).is_eq());
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

            assert_eq_verbose!(buffer, [f(1), f(2), f(3), f(4), f(5)]);
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

            assert_eq_verbose!(buffer, [f(1), f(2), f(3), f(4), f(5)]);
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

            assert_eq_verbose!(buffer, [f(1), f(2), f(3), f(4), f(5)]);
        }
    }

    #[test]
    fn test_vec_neighbor_search_output_buffer() {
        use crate::graph::search_output_buffer::SearchOutputBuffer;

        let mut buf: Vec<Neighbor<u32>> = Vec::new();
        assert_eq!(SearchOutputBuffer::<u32>::size_hint(&buf), None);
        assert_eq!(SearchOutputBuffer::<u32>::current_len(&buf), 0);

        // push grows unboundedly
        assert!(SearchOutputBuffer::push(&mut buf, 1, 0.5).is_available());
        assert!(SearchOutputBuffer::push(&mut buf, 2, 1.0).is_available());
        assert_eq!(SearchOutputBuffer::<u32>::current_len(&buf), 2);
        assert_eq_verbose!(buf[0], Neighbor::new(1, 0.5));
        assert_eq_verbose!(buf[1], Neighbor::new(2, 1.0));

        // extend appends and returns count
        let count = SearchOutputBuffer::extend(&mut buf, vec![(3u32, 1.5), (4, 2.0), (5, 2.5)]);
        assert_eq!(count, 3);
        assert_eq!(SearchOutputBuffer::<u32>::current_len(&buf), 5);
        assert_eq_verbose!(buf[4], Neighbor::new(5, 2.5));
    }
}
