/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A generalized trait for extracting search results.
///
/// Putting this behind a trait allows multiple different output containers to use common
/// interfaces. This, in turn, can allow search to return IDs and distances as separate
/// buffers (see [`IdDistance`]), or as a contiguous [`crate::neighbor::BackInserter`].
///
/// It also makes it easier to testing search post-processing routines as they should target
/// an output implementing `SearchOutputBuffer`, simplifying how they can be tested.
pub trait SearchOutputBuffer<I, D = f32> {
    /// Return a hint on the remaining size of the buffer.
    ///
    /// `None` may be returned in instances where the output has unknown or unbounded size.
    fn size_hint(&self) -> Option<usize>;

    /// Push an `id` and `distance` pair to the next position in the buffer.
    ///
    /// Returns a [`BufferState`] to indicate whether future insertions will succeed.
    ///
    /// Unlike the iterator interface, implementations should return [`BufferState::Full`]
    /// if **future** insertions will fail to prevent unnecessary work.
    fn push(&mut self, id: I, distance: D) -> BufferState;

    /// Return the number of items pushed into the buffer.
    fn current_len(&self) -> usize;

    /// Set from an iterator, returning the number of positions filled.
    ///
    /// The entire iterator may not be consumed if there is insufficient capacity in `self`.
    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (I, D)>;
}

/// Indicate whether future calls to [`SearchOutputBuffer::push`] will succeed or not.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "This type indicates whether the output buffer is full or not."]
pub enum BufferState {
    /// There is capacity available in the buffer.
    Available,

    /// The buffer is full. Future calls to [`SearchOutputBuffer::push`] or
    /// [`SearchOutputBuffer::extend`] will fail.
    Full,
}

impl BufferState {
    /// Return `true` if `self == Self::Available`. Otherwise return `false`.
    pub fn is_available(self) -> bool {
        self == Self::Available
    }

    /// Return `true` if `self == Self::Full`. Otherwise return `false`.
    pub fn is_full(self) -> bool {
        self == Self::Full
    }
}

/// A [`SearchOutputBuffer`] that maintains two separate slices for IDs and distances.
///
/// # Internal Invariants
///
/// Both `ids` and `distances` must have the same length.
#[derive(Debug)]
pub struct IdDistance<'a, I> {
    ids: &'a mut [I],
    distances: &'a mut [f32],
    position: usize,
}

impl<'a, I> IdDistance<'a, I> {
    /// Construct a new [`IdDistance`] around the two slices.
    ///
    /// # Panics
    ///
    /// Panics if the two slices have different lengths.
    pub fn new(ids: &'a mut [I], distances: &'a mut [f32]) -> Self {
        assert_eq!(
            ids.len(),
            distances.len(),
            "ids and distances should have the same length"
        );
        Self {
            ids,
            distances,
            position: 0,
        }
    }

    /// The length of **both** internal slices.
    pub fn capacity(&self) -> usize {
        self.ids.len()
    }
}

impl<I> SearchOutputBuffer<I> for IdDistance<'_, I> {
    fn size_hint(&self) -> Option<usize> {
        Some(self.capacity() - self.position)
    }

    fn push(&mut self, id: I, distance: f32) -> BufferState {
        if self.position == self.capacity() {
            return BufferState::Full;
        }

        self.ids[self.position] = id;
        self.distances[self.position] = distance;
        self.position += 1;

        // Return `Full` if we added the last item.
        if self.position == self.capacity() {
            BufferState::Full
        } else {
            BufferState::Available
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
        let p = self.position;
        std::iter::zip(
            self.ids.iter_mut().skip(p),
            self.distances.iter_mut().skip(p),
        )
        .zip(itr)
        .for_each(|((i_out, d_out), (i_in, d_in))| {
            i += 1;
            *i_out = i_in;
            *d_out = d_in;
        });
        self.position += i;
        i
    }
}

#[derive(Debug)]
pub struct IdDistanceAssociatedData<'a, I, A> {
    ids: &'a mut [I],
    distances: &'a mut [f32],
    associated_data: &'a mut [A],
    position: usize,
}

impl<'a, I, A> IdDistanceAssociatedData<'a, I, A> {
    pub fn new(ids: &'a mut [I], distances: &'a mut [f32], associated_data: &'a mut [A]) -> Self {
        assert_eq!(
            ids.len(),
            distances.len(),
            "ids and distances should have the same length"
        );
        assert_eq!(
            ids.len(),
            associated_data.len(),
            "ids and associated_data should have the same length"
        );
        Self {
            ids,
            distances,
            associated_data,
            position: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.ids.len()
    }
}

impl<I: Copy + Send, A: Clone + Send> SearchOutputBuffer<(I, A), f32>
    for IdDistanceAssociatedData<'_, I, A>
{
    fn size_hint(&self) -> Option<usize> {
        Some(self.capacity() - self.position)
    }

    fn push(&mut self, item: (I, A), distance: f32) -> BufferState {
        if self.position == self.capacity() {
            return BufferState::Full;
        }

        let (id, assoc) = item;
        self.ids[self.position] = id;
        self.distances[self.position] = distance;
        self.associated_data[self.position] = assoc;
        self.position += 1;

        // Return `Full` if we added the last item.
        if self.position == self.capacity() {
            BufferState::Full
        } else {
            BufferState::Available
        }
    }

    fn current_len(&self) -> usize {
        self.position
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = ((I, A), f32)>,
    {
        let mut i = 0;
        let p = self.position;
        for (((id_out, dist_out), assoc_out), ((id, assoc), dist)) in self
            .ids
            .iter_mut()
            .skip(p)
            .zip(self.distances.iter_mut().skip(p))
            .zip(self.associated_data.iter_mut().skip(p))
            .zip(itr)
        {
            *id_out = id;
            *dist_out = dist;
            *assoc_out = assoc;
            i += 1;
        }
        self.position += i;
        i
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    // Test that the constructor panics on unequal lengths.
    #[test]
    #[should_panic(expected = "ids and distances should have the same length")]
    fn test_new_panics() {
        let mut ids = vec![0u32; 10];
        let mut distances = vec![0.0f32; 9];
        IdDistance::new(&mut ids, &mut distances);
    }

    #[test]
    fn test_id_distance() {
        const MAX_LENGTH: usize = 5;

        // All `push`.
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut buffer = IdDistance::new(&mut ids, &mut distances);

            assert_eq!(buffer.capacity(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH));
            assert_eq!(buffer.current_len(), 0);

            assert!(buffer.push(1, 1.0).is_available());
            assert_eq!(buffer.current_len(), 1);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 1));

            assert!(buffer.push(2, 2.0).is_available());
            assert_eq!(buffer.current_len(), 2);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 2));

            assert!(buffer.push(3, 3.0).is_available());
            assert_eq!(buffer.current_len(), 3);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 3));

            assert!(buffer.push(4, 4.0).is_available());
            assert_eq!(buffer.current_len(), 4);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 4));

            // This should error since further attempts will not work.
            assert!(buffer.push(5, 5.0).is_full());
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert!(buffer.push(6, 6.0).is_full());
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert_eq!(&ids, &[1, 2, 3, 4, 5]);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        }

        // All `iterator`.
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut buffer = IdDistance::new(&mut ids, &mut distances);

            assert_eq!(buffer.capacity(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH));
            assert_eq!(buffer.current_len(), 0);

            let set = buffer.extend([(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0), (6, 6.0)]);
            assert_eq!(set, MAX_LENGTH);
            assert_eq!(buffer.current_len(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(0));

            // Ensure that `pushing` respects the limit.
            assert!(buffer.push(7, 7.0).is_full());

            let set = buffer.extend([(10, 10.0), (20, 20.0)]);
            assert_eq!(set, 0, "no more items can be added");

            assert_eq!(&ids, &[1, 2, 3, 4, 5]);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        }

        // Mixture
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut buffer = IdDistance::new(&mut ids, &mut distances);

            assert!(buffer.push(1, 1.0).is_available());

            let set = buffer.extend([(2, 2.0), (3, 3.0)]);
            assert_eq!(set, 2, "only two items were pushed");

            assert_eq!(buffer.current_len(), 3);
            assert_eq!(buffer.size_hint(), Some(2));

            assert!(buffer.push(4, 4.0).is_available());
            assert_eq!(buffer.current_len(), 4);
            assert_eq!(buffer.size_hint(), Some(1));

            let set = buffer.extend([(5, 5.0), (6, 6.0)]);
            assert_eq!(
                set, 1,
                "there should only be room for one more item in the buffer"
            );
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert_eq!(&ids, &[1, 2, 3, 4, 5],);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        }
    }

    #[test]
    #[should_panic(expected = "ids and associated_data should have the same length")]
    fn test_id_distance_associated_data_panics_on_different_id_associated_data_lengths() {
        let mut ids = vec![0u32; 10];
        let mut distances = vec![0.0f32; 10];
        let mut associated_data = vec![0u32; 9];
        IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated_data);
    }

    #[test]
    fn test_id_distance_associated() {
        const MAX_LENGTH: usize = 5;

        // All `push`.
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut associated = [0u32; MAX_LENGTH];
            let mut buffer =
                IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated);

            assert_eq!(buffer.capacity(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH));
            assert_eq!(buffer.current_len(), 0);

            assert!(buffer.push((1, 10), 1.0).is_available());
            assert_eq!(buffer.current_len(), 1);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 1));

            assert!(buffer.push((2, 20), 2.0).is_available());
            assert_eq!(buffer.current_len(), 2);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 2));

            assert!(buffer.push((3, 30), 3.0).is_available());
            assert_eq!(buffer.current_len(), 3);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 3));

            assert!(buffer.push((4, 40), 4.0).is_available());
            assert_eq!(buffer.current_len(), 4);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH - 4));

            // This should error since further attempts will not work.
            assert!(buffer.push((5, 50), 5.0).is_full());
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert!(buffer.push((6, 60), 6.0).is_full());
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert_eq!(&ids, &[1, 2, 3, 4, 5]);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(&associated, &[10, 20, 30, 40, 50]);
        }

        // All `iterator`.
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut associated = [0u32; MAX_LENGTH];
            let mut buffer =
                IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated);

            assert_eq!(buffer.capacity(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(MAX_LENGTH));
            assert_eq!(buffer.current_len(), 0);

            let set = buffer.extend([
                ((1, 10), 1.0),
                ((2, 20), 2.0),
                ((3, 30), 3.0),
                ((4, 40), 4.0),
                ((5, 50), 5.0),
                ((6, 60), 6.0),
            ]);
            assert_eq!(set, MAX_LENGTH);
            assert_eq!(buffer.current_len(), MAX_LENGTH);
            assert_eq!(buffer.size_hint(), Some(0));

            // Ensure that `pushing` respects the limit.
            assert!(buffer.push((7, 70), 7.0).is_full());

            let set = buffer.extend([((10, 100), 10.0), ((20, 200), 20.0)]);
            assert_eq!(set, 0, "no more items can be added");

            assert_eq!(&ids, &[1, 2, 3, 4, 5]);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(&associated, &[10, 20, 30, 40, 50]);
        }

        // Mixture
        {
            let mut ids = [0u32; MAX_LENGTH];
            let mut distances = [0.0f32; MAX_LENGTH];
            let mut associated = [0u32; MAX_LENGTH];
            let mut buffer =
                IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated);

            assert!(buffer.push((1, 10), 1.0).is_available());

            let set = buffer.extend([((2, 20), 2.0), ((3, 30), 3.0)]);
            assert_eq!(set, 2, "only two items were pushed");

            assert_eq!(buffer.current_len(), 3);
            assert_eq!(buffer.size_hint(), Some(2));

            assert!(buffer.push((4, 40), 4.0).is_available());
            assert_eq!(buffer.current_len(), 4);
            assert_eq!(buffer.size_hint(), Some(1));

            let set = buffer.extend([((5, 50), 5.0), ((6, 60), 6.0)]);
            assert_eq!(
                set, 1,
                "there should only be room for one more item in the buffer"
            );
            assert_eq!(buffer.current_len(), 5);
            assert_eq!(buffer.size_hint(), Some(0));

            assert_eq!(&ids, &[1, 2, 3, 4, 5],);
            assert_eq!(&distances, &[1.0, 2.0, 3.0, 4.0, 5.0]);
            assert_eq!(&associated, &[10, 20, 30, 40, 50],);
        }
    }
}
