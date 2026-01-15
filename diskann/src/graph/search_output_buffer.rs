/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

/// A generalized trait for extracting search results.
///
/// Putting this behind a trait allows multiple different output containers to use common
/// interfaces. This, in turn, can allow search to return IDs and distances as separate
/// buffers (see [`IdDistance`]), or as a contiguous `[Neighbor<_>]`, and more.
///
/// It also makes it easier to testing search post-processing routines as they should target
/// an output implementing `SearchOutputBuffer`, simplifying how they can be tested.
pub trait SearchOutputBuffer<I, D = f32> {
    /// Return a hint on the size of the buffer.
    ///
    /// Implementations should guarantee that if `size_hint` returns `Some`, then
    /// `set` is valid for all indices up to the returned value (but unsafe code should
    /// not rely on this).
    ///
    /// `None` may be returned in instances where the output has unbounded size.
    fn size_hint(&self) -> Option<usize>;

    /// Set position `i` in the buffer, returning an error is `i` is out of bounds.
    fn set(&mut self, i: usize, id: I, distance: D) -> Result<(), IndexOutOfBounds>;

    /// Set from an iterator, returning the number of positions filled.
    fn set_from<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: Iterator<Item = (I, D)>;
}

#[derive(Debug, Clone, Copy, Error)]
#[error("index {0} is out-of-bounds")]
pub struct IndexOutOfBounds(usize);

impl IndexOutOfBounds {
    pub fn new(index: usize) -> Self {
        Self(index)
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
        Self { ids, distances }
    }

    /// The length of **both** internal slices.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl<I> SearchOutputBuffer<I> for IdDistance<'_, I> {
    fn size_hint(&self) -> Option<usize> {
        Some(self.len())
    }

    fn set(&mut self, i: usize, id: I, distance: f32) -> Result<(), IndexOutOfBounds> {
        if i >= self.len() {
            Err(IndexOutOfBounds::new(i))
        } else {
            self.ids[i] = id;
            self.distances[i] = distance;
            Ok(())
        }
    }

    fn set_from<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: Iterator<Item = (I, f32)>,
    {
        let mut i = 0;
        std::iter::zip(self.ids.iter_mut(), self.distances.iter_mut())
            .zip(itr)
            .for_each(|((i_out, d_out), (i_in, d_in))| {
                i += 1;
                *i_out = i_in;
                *d_out = d_in;
            });
        i
    }
}

#[derive(Debug)]
pub struct IdDistanceAssociatedData<'a, I, A> {
    ids: &'a mut [I],
    distances: &'a mut [f32],
    associated_data: &'a mut [A],
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
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl<I: Copy + Send, A: Clone + Send> SearchOutputBuffer<(I, A), f32>
    for IdDistanceAssociatedData<'_, I, A>
{
    fn size_hint(&self) -> Option<usize> {
        Some(self.len())
    }

    fn set(&mut self, i: usize, item: (I, A), distance: f32) -> Result<(), IndexOutOfBounds> {
        let (id, assoc) = item;
        if i >= self.len() {
            Err(IndexOutOfBounds::new(i))
        } else {
            self.ids[i] = id;
            self.distances[i] = distance;
            self.associated_data[i] = assoc;
            Ok(())
        }
    }

    fn set_from<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: Iterator<Item = ((I, A), f32)>,
    {
        let mut i = 0;
        for (((id_out, dist_out), assoc_out), ((id, assoc), dist)) in self
            .ids
            .iter_mut()
            .zip(self.distances.iter_mut())
            .zip(self.associated_data.iter_mut())
            .zip(itr)
        {
            *id_out = id;
            *dist_out = dist;
            *assoc_out = assoc;
            i += 1;
        }

        i
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_is_err<T>(_: &T)
    where
        T: std::error::Error,
    {
    }

    #[test]
    fn index_error_is_error() {
        let err = IndexOutOfBounds::new(0);
        assert_is_err(&err);
    }

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
        // Scalar Interface
        for len in 0..20 {
            let mut ids = vec![0u32; len];
            let mut distances = vec![0.0; len];
            let mut buffer = IdDistance::new(&mut ids, &mut distances);

            assert_eq!(buffer.len(), len);
            assert_eq!(buffer.size_hint(), Some(len));

            // All of these should work okay.
            for i in 0..len {
                buffer.set(i, i as u32, i as f32).unwrap();
            }

            // Setting one past the end should yield an error.
            let err = buffer.set(len, 0, 0.0).unwrap_err();
            assert_is_err(&err);
            assert_eq!(err.to_string(), format!("index {} is out-of-bounds", len));

            // Check that the ids and distances are set correctly.
            for (i, id) in ids.iter().enumerate() {
                assert_eq!(i, *id as usize);
            }
            for (i, distance) in distances.iter().enumerate() {
                assert_eq!(i as f32, *distance);
            }
        }

        // Iterator Interface
        for len in 0..10 {
            for input_len in 0..10 {
                let mut ids = vec![0u32; len];
                let mut distances = vec![0.0; len];
                let mut buffer = IdDistance::new(&mut ids, &mut distances);

                let source: Vec<_> = (0..input_len).map(|i| (i as u32, i as f32)).collect();

                let count = buffer.set_from(source.into_iter());
                // The assigned count should be the minimum of the input and output lengths.
                assert_eq!(count, input_len.min(len));
                for (i, (id, dist)) in std::iter::zip(ids.iter(), distances.iter())
                    .take(count)
                    .enumerate()
                {
                    assert_eq!(i, *id as usize);
                    assert_eq!(i as f32, *dist);
                }

                // THe upper values should be untouched.
                for i in count..len {
                    assert_eq!(ids[i], 0);
                    assert_eq!(distances[i], 0.0);
                }
            }
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
    fn test_id_distance_associated_data_set() {
        // Test scalar interface
        for len in 0..20 {
            let mut ids = vec![0u32; len];
            let mut distances = vec![0.0; len];
            let mut associated_data = vec![0u32; len];
            let mut buffer =
                IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated_data);

            assert_eq!(buffer.len(), len);
            assert_eq!(buffer.size_hint(), Some(len));

            // All of these should work okay
            for i in 0..len {
                buffer.set(i, (i as u32, i as u32), i as f32).unwrap();
            }

            // Setting one past the end should yield an error
            let err = buffer.set(len, (0, 999u32), 0.0).unwrap_err();
            assert_is_err(&err);
            assert_eq!(err.to_string(), format!("index {} is out-of-bounds", len));

            // Check that ids, distances, and associated data are set correctly
            for (i, id) in ids.iter().enumerate() {
                assert_eq!(i, *id as usize);
            }
            for (i, distance) in distances.iter().enumerate() {
                assert_eq!(i as f32, *distance);
            }
            for (i, data) in associated_data.iter().enumerate() {
                assert_eq!(i as u32, *data);
            }
        }
    }

    #[test]
    fn test_id_distance_associated_data_set_from() {
        // Iterator Interface
        for len in 0..10 {
            for input_len in 0..10 {
                let mut ids = vec![0u32; len];
                let mut distances = vec![0.0; len];
                let mut associated_data = vec![0u32; len];
                let mut buffer =
                    IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated_data);

                let source: Vec<_> = (0..input_len)
                    .map(|i| ((i as u32, i as u32), i as f32))
                    .collect();

                let count = buffer.set_from(source.into_iter());

                // The assigned count should be the minimum of the input and output lengths
                assert_eq!(count, input_len.min(len));

                for i in 0..count {
                    assert_eq!(i, ids[i] as usize);
                    assert_eq!(i as f32, distances[i]);
                    assert_eq!(i as u32, associated_data[i]);
                }

                // The upper values should be untouched
                for i in count..len {
                    assert_eq!(ids[i], 0);
                    assert_eq!(distances[i], 0.0);
                    assert_eq!(0u32, associated_data[i]);
                }
            }
        }
    }

    #[test]
    fn test_id_distance_is_empty() {
        // Test with empty buffers
        let mut empty_ids: Vec<u32> = Vec::new();
        let mut empty_distances: Vec<f32> = Vec::new();
        let buffer_empty = IdDistance::new(&mut empty_ids, &mut empty_distances);
        assert!(buffer_empty.is_empty());

        // Test with non-empty buffers
        let mut ids = vec![0u32; 5];
        let mut distances = vec![0.0f32; 5];
        let buffer_non_empty = IdDistance::new(&mut ids, &mut distances);
        assert!(!buffer_non_empty.is_empty());
    }

    #[test]
    fn test_id_distance_associated_data_is_empty() {
        // Test with empty buffers
        let mut empty_ids: Vec<u32> = Vec::new();
        let mut empty_distances: Vec<f32> = Vec::new();
        let mut empty_associated_data: Vec<u32> = Vec::new();
        let buffer_empty = IdDistanceAssociatedData::new(
            &mut empty_ids,
            &mut empty_distances,
            &mut empty_associated_data,
        );
        assert!(buffer_empty.is_empty());

        // Test with non-empty buffers
        let mut ids = vec![0u32; 5];
        let mut distances = vec![0.0f32; 5];
        let mut associated_data = vec![0u32; 5];
        let buffer_non_empty =
            IdDistanceAssociatedData::new(&mut ids, &mut distances, &mut associated_data);
        assert!(!buffer_non_empty.is_empty());
    }
}
