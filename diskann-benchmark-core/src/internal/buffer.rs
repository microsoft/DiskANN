/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::graph::{SearchOutputBuffer, search_output_buffer::BufferState};

/// A [`SearchOutputBuffer`] implementation that either references a slice in-place for
/// fixed sized outputs or references a growable vector.
///
/// We put this behind an enum to cut down on the amount of code that gets monomorphized.
///
/// If the underlying buffer is a slice, the number of written elements is tracked separately and
/// is available via [`Buffer::current_len`]. When the underlying buffer is a `Vec`, the length
/// of the vector is used.
#[derive(Debug)]
pub(crate) struct Buffer<'a, I>(Inner<'a, I>);

#[derive(Debug)]
enum Inner<'a, I> {
    Slice { slice: &'a mut [I], written: usize },
    Vec(&'a mut Vec<I>),
}

impl<'a, I> Buffer<'a, I> {
    /// Construct a buffer that writes into `slice`.
    pub(crate) fn slice(slice: &'a mut [I]) -> Self {
        Self(Inner::Slice { slice, written: 0 })
    }

    /// Construct a vector. This clear `v` when called.
    pub(crate) fn vector(v: &'a mut Vec<I>) -> Self {
        v.clear();
        Self(Inner::Vec(v))
    }

    /// Return the number of elements that have been written to the buffer.
    ///
    /// This is useful to track instances where searches return fewer than the
    /// requested number of ids.
    pub(crate) fn current_len(&self) -> usize {
        match &self.0 {
            Inner::Slice { written, .. } => *written,
            Inner::Vec(vec) => vec.len(),
        }
    }
}

impl<I, D> SearchOutputBuffer<I, D> for Buffer<'_, I> {
    fn size_hint(&self) -> Option<usize> {
        match &self.0 {
            Inner::Slice { slice, written } => Some(slice.len() - written),
            Inner::Vec(_) => None,
        }
    }

    fn current_len(&self) -> usize {
        <Buffer<I>>::current_len(self)
    }

    fn push(&mut self, id: I, _distance: D) -> BufferState {
        match &mut self.0 {
            Inner::Slice { slice, written } => match slice.get_mut(*written) {
                Some(slot) => {
                    *slot = id;
                    *written += 1;
                    if *written == slice.len() {
                        BufferState::Full
                    } else {
                        BufferState::Available
                    }
                }
                None => BufferState::Full,
            },
            Inner::Vec(vec) => {
                vec.push(id);
                BufferState::Available
            }
        }
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (I, D)>,
    {
        match &mut self.0 {
            Inner::Slice { slice, written } => match slice.get_mut(*written..) {
                Some(left) => {
                    let count = std::iter::zip(left.iter_mut(), itr)
                        .map(|(dst, src)| {
                            *dst = src.0;
                        })
                        .count();
                    *written += count;
                    count
                }
                None => 0,
            },
            Inner::Vec(vec) => {
                let before = vec.len();
                vec.extend(itr.into_iter().map(|i| i.0));
                vec.len() - before
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn size_hint<T>(buffer: &T) -> Option<usize>
    where
        T: SearchOutputBuffer<u32, f32>,
    {
        buffer.size_hint()
    }

    #[test]
    fn test_slice_buffer_creation() {
        let mut data = [0u32; 5];
        let buffer = Buffer::slice(&mut data);

        assert_eq!(buffer.current_len(), 0);
        assert_eq!(size_hint(&buffer), Some(5));
    }

    #[test]
    fn test_vec_buffer_creation() {
        let mut vec = vec![1u32, 2, 3, 4];
        let buffer = Buffer::vector(&mut vec);

        assert_eq!(buffer.current_len(), 0);
        assert_eq!(size_hint(&buffer), None);
        assert!(vec.is_empty(), "vector should be cleared on construction");
    }

    #[test]
    fn test_slice_buffer_set_single_element() {
        let mut data = [0u32; 5];
        let mut buffer = Buffer::slice(&mut data);

        assert_eq!(buffer.push(42, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 1);
        assert_eq!(data, [42, 0, 0, 0, 0]);
    }

    #[test]
    fn test_slice_buffer_set_updates_written() {
        let mut data = [0u32; 5];
        let mut buffer = Buffer::slice(&mut data);

        assert_eq!(buffer.current_len(), 0);
        assert_eq!(size_hint(&buffer), Some(5));

        assert_eq!(buffer.push(100, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 1);
        assert_eq!(size_hint(&buffer), Some(4));

        assert_eq!(buffer.push(200, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 2);
        assert_eq!(size_hint(&buffer), Some(3));

        assert_eq!(buffer.push(300, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 3);
        assert_eq!(size_hint(&buffer), Some(2));

        assert_eq!(buffer.push(400, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 4);
        assert_eq!(size_hint(&buffer), Some(1));

        assert_eq!(buffer.push(500, 0.0), BufferState::Full);
        assert_eq!(buffer.current_len(), 5);
        assert_eq!(size_hint(&buffer), Some(0));

        assert_eq!(buffer.push(600, 0.0), BufferState::Full);
        assert_eq!(buffer.current_len(), 5);
        assert_eq!(size_hint(&buffer), Some(0));

        // Check that `data` was written properly.
        assert_eq!(data, [100, 200, 300, 400, 500]);
    }

    #[test]
    fn test_slice_buffer_empty() {
        let mut data = [0u32; 0];
        let mut buffer = Buffer::slice(&mut data);

        assert_eq!(buffer.push(42, 0.0), BufferState::Full);
    }

    #[test]
    fn test_vec_buffer_push() {
        let mut vec = vec![0u32, 0, 0, 0, 0];
        let mut buffer = Buffer::vector(&mut vec);
        assert_eq!(
            size_hint(&buffer),
            None,
            "vector-type buffers have no upper bound"
        );

        assert_eq!(buffer.push(42, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 1);

        assert_eq!(buffer.push(50, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 2);

        assert_eq!(buffer.push(3, 0.0), BufferState::Available);
        assert_eq!(buffer.current_len(), 3);

        assert_eq!(&vec, &[42, 50, 3]);
    }

    #[test]
    fn test_slice_buffer_set_from() {
        let mut data = [0u32; 5];
        let mut buffer = Buffer::slice(&mut data);

        let items = vec![(10, 1.0), (20, 2.0), (30, 3.0)];
        let count = buffer.extend(items);

        assert_eq!(count, 3);
        assert_eq!(buffer.current_len(), 3);
        assert_eq!(data, [10, 20, 30, 0, 0]);
    }

    #[test]
    fn test_slice_buffer_set_from_more_than_capacity() {
        let mut data = [0u32; 3];
        let mut buffer = Buffer::slice(&mut data);

        let items = vec![(10, 1.0), (20, 2.0), (30, 3.0), (40, 4.0), (50, 5.0)];
        let count = buffer.extend(items);

        // Only first 3 items should be written
        assert_eq!(count, 3);
        assert_eq!(buffer.current_len(), 3);
        assert_eq!(data, [10, 20, 30]);
    }

    #[test]
    fn test_vec_buffer_extend() {
        let mut vec = Vec::<u32>::new();
        let mut buffer = Buffer::vector(&mut vec);

        let items = vec![(100, 1.0), (200, 2.0)];
        let count = buffer.extend(items);

        assert_eq!(count, 2);
        assert_eq!(buffer.current_len(), 2);
        assert_eq!(&vec, &[100, 200]);
    }

    #[test]
    fn test_vec_buffer_extend_cascades() {
        let mut vec = vec![1u32, 2, 3, 4, 5];
        let mut buffer = Buffer::vector(&mut vec);

        let items = vec![(10, 1.0), (20, 2.0)];
        assert_eq!(buffer.extend(items), 2);

        let items = vec![(21, 1.0), (22, 2.0)];
        assert_eq!(buffer.extend(items), 2);

        assert_eq!(
            buffer.extend::<[(u32, f32); 0]>([]),
            0,
            "empty iterator should add nothing"
        );

        // Previous items should be cleared
        assert_eq!(&vec, &[10, 20, 21, 22]);
    }

    #[test]
    fn test_slice_push_and_extend_combinations() {
        // Push then extend
        let mut data = [0u32; 5];
        let mut buffer = Buffer::slice(&mut data);
        assert_eq!(buffer.push(1, 0.0), BufferState::Available);
        assert_eq!(buffer.push(2, 0.0), BufferState::Available);
        assert_eq!(buffer.extend(vec![(3, 0.0), (4, 0.0)]), 2);
        assert_eq!(data, [1, 2, 3, 4, 0]);

        // Extend then push to fill
        let mut data = [0u32; 5];
        let mut buffer = Buffer::slice(&mut data);
        buffer.extend(vec![(10, 0.0), (20, 0.0)]);
        assert_eq!(buffer.push(30, 0.0), BufferState::Available);
        assert_eq!(buffer.push(40, 0.0), BufferState::Available);
        assert_eq!(buffer.push(50, 0.0), BufferState::Full);
        assert_eq!(data, [10, 20, 30, 40, 50]);

        // Interleaved operations
        let mut data = [0u32; 6];
        let mut buffer = Buffer::slice(&mut data);
        assert_eq!(buffer.push(1, 0.0), BufferState::Available);
        assert_eq!(buffer.extend(vec![(2, 0.0), (3, 0.0)]), 2);
        assert_eq!(buffer.push(4, 0.0), BufferState::Available);
        assert_eq!(buffer.extend(vec![(5, 0.0), (6, 0.0), (7, 0.0)]), 2);
        assert_eq!(data, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_slice_push_and_extend_at_capacity() {
        // Extend fills buffer, push returns Full
        let mut data = [0u32; 2];
        let mut buffer = Buffer::slice(&mut data);
        buffer.extend(vec![(1, 0.0), (2, 0.0)]);
        assert_eq!(buffer.push(99, 0.0), BufferState::Full);
        assert_eq!(data, [1, 2]);

        // Push fills buffer, extend returns 0
        let mut data = [0u32; 2];
        let mut buffer = Buffer::slice(&mut data);
        assert_eq!(buffer.push(1, 0.0), BufferState::Available);
        assert_eq!(buffer.push(2, 0.0), BufferState::Full);
        assert_eq!(buffer.extend(vec![(99, 0.0)]), 0);
        assert_eq!(data, [1, 2]);

        // Extend truncates when exceeding remaining capacity after push
        let mut data = [0u32; 4];
        let mut buffer = Buffer::slice(&mut data);
        assert_eq!(buffer.push(1, 0.0), BufferState::Available);
        assert_eq!(buffer.push(2, 0.0), BufferState::Available);
        assert_eq!(
            buffer.extend(vec![(3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)]),
            2
        );
        assert_eq!(data, [1, 2, 3, 4]);
    }

    #[test]
    fn test_vec_push_and_extend_combinations() {
        let mut vec = Vec::<u32>::new();
        let mut buffer = Buffer::vector(&mut vec);

        // Interleave push and extend - vec has no capacity limit
        assert_eq!(buffer.push(1, 0.0), BufferState::Available);
        assert_eq!(buffer.extend(vec![(2, 0.0), (3, 0.0)]), 2);
        assert_eq!(buffer.push(4, 0.0), BufferState::Available);
        assert_eq!(buffer.extend::<[(u32, f32); 0]>([]), 0); // empty extend
        assert_eq!(buffer.extend(vec![(5, 0.0)]), 1);

        assert_eq!(&vec, &[1, 2, 3, 4, 5]);
    }
}
