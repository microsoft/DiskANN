/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_utils::{DenseData, MutDenseData};
use std::ops::{Index, IndexMut};
use thiserror::Error;

//////////////////
// ChunkOffsets //
//////////////////

/// A wrapper class for PQ chunk offsets.
///
/// Upon construction, this class guarantees that the underlying chunk offset plan is valid.
/// A valid PQ chunk offset plan records the starting offsets of each chunk such that chunk
/// `i` of a slice `x` can be accessed using `x[offsets[i]..offsets[i+1]]`.
///
/// In particular, a valid PQ chunk offset plan has the following properties:
///
/// * It has a length of at least 2.
/// * Its first entry is 0.
/// * For `i` in `0..offsets.len()`, `offsets[i] < offsets[i+1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkOffsetsBase<T>
where
    T: DenseData<Elem = usize>,
{
    // Pre-compute the associated dimension for better locality.
    //
    // We could extract this as `offsets.last().unwrap() - 1`, but hoisting it out into
    // the struct means it is readily available when needed.
    dim: NonZeroUsize,
    // Chunk Offsets
    offsets: T,
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ChunkOffsetError {
    #[error("offsets must have a length of at least 2, found {0}")]
    LengthNotAtLeastTwo(usize),
    #[error("offsets must begin at 0, not {0}")]
    DoesNotBeginWithZero(usize),
    #[error(
        "offsets must be strictly increasing, \
         instead entry {start_val} at position {start} is followed by {next_val}"
    )]
    NonMonotonic {
        start_val: usize,
        start: usize,
        next_val: usize,
    },
}

/// Error returned by [`ChunkOffsets::partition`].
#[derive(Error, Debug)]
#[error("num_chunks {num_chunks} must not exceed dim {dim}")]
pub struct PartitionError {
    pub num_chunks: usize,
    pub dim: usize,
}

/// Error returned by [`ChunkOffsetsView::partition_into`].
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum PartitionIntoError {
    #[error("scratch must have a length of at least 2, found {0}")]
    ScratchTooSmall(usize),
    #[error(transparent)]
    PartitionError(#[from] PartitionError),
}

impl<T> ChunkOffsetsBase<T>
where
    T: DenseData<Elem = usize>,
{
    /// Construct a new `ChunkOffset` from a raw slice.
    ///
    /// Returns an error if:
    /// * The length of `offsets` is less than 2.
    /// * The entries in `offsets` are not strictly increasing.
    pub fn new(offsets: T) -> Result<Self, ChunkOffsetError> {
        let slice = offsets.as_slice();

        // Validate that the length is correct.
        let len = slice.len();
        if len < 2 {
            return Err(ChunkOffsetError::LengthNotAtLeastTwo(len));
        }

        // Check that we don't start at zero.
        let start = slice[0];
        if start != 0 {
            return Err(ChunkOffsetError::DoesNotBeginWithZero(start));
        }

        // What follows is a convoluted dance to safely get the source dimension as a
        // `NonZeroUsize` while validating the monotonicity of the offsets in the provided
        // slice.
        //
        // We can seed the `NonZeroUsize` with the knowledge that we've already checked
        // that the length of the `slice` vector is at least two.
        let mut last: NonZeroUsize = match NonZeroUsize::new(slice[1]) {
            Some(x) => Ok(x),
            None => Err(ChunkOffsetError::NonMonotonic {
                start_val: start,
                start: 0,
                next_val: 0,
            }),
        }?;

        // Now that we've successfully initialized a `NonZeroUsize` - we can use it going
        // forward.
        //
        // Validate that `offsets` is monotonic.
        for i in 2..slice.len() {
            let start_val = slice[i - 1];
            let next_val = NonZeroUsize::new(slice[i]);
            last = match next_val {
                Some(next_val) => {
                    if start_val >= next_val.get() {
                        Err(ChunkOffsetError::NonMonotonic {
                            start_val,
                            start: i - 1,
                            next_val: next_val.get(),
                        })
                    } else {
                        Ok(next_val)
                    }
                }
                // If we hit this case, then `slice[i]` was zero.
                None => Err(ChunkOffsetError::NonMonotonic {
                    start_val,
                    start: i - 1,
                    next_val: 0,
                }),
            }?;
        }

        // The last entry in `offset` is one-past the end.
        Ok(Self { dim: last, offsets })
    }

    /// Return the number of chunks associated with this mapping.
    ///
    /// This will be one-less than the length of the provided slice.
    pub fn len(&self) -> usize {
        // This invariant should hold by construction, and allows us to safely subtract
        // 1 from the length of the underlying `offsets` span.
        debug_assert!(self.offsets.as_slice().len() >= 2);
        self.offsets.as_slice().len() - 1
    }

    /// Return whether the offsets are empty.
    pub fn is_empty(&self) -> bool {
        // by class invariant, there must always be at least one chunk.
        false
    }

    /// Return the dimensionality of the vector data associated with this chunking schema.
    pub fn dim(&self) -> usize {
        self.dim.get()
    }

    /// Return the dimensionality of the vector data associated with this chunking schema.
    ///
    /// By class invariant, the dimensionality must be nonzero, and this expressed in the
    /// retuen type.
    ///
    /// This method cannot fail and will not panic.
    pub fn dim_nonzero(&self) -> NonZeroUsize {
        self.dim
    }

    /// Return a range containing the start and one-past-the-end indices for chunk `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.len()`.
    pub fn at(&self, i: usize) -> core::ops::Range<usize> {
        assert!(
            i < self.len(),
            "index {i} must be less than len {}",
            self.len()
        );
        let slice = self.offsets.as_slice();
        slice[i]..slice[i + 1]
    }

    /// Return `self` as a view.
    pub fn as_view(&self) -> ChunkOffsetsView<'_> {
        ChunkOffsetsBase {
            dim: self.dim,
            offsets: self.offsets.as_slice(),
        }
    }

    /// Return a `'static` copy of `self`.
    pub fn to_owned(&self) -> ChunkOffsets {
        ChunkOffsetsBase {
            dim: self.dim,
            offsets: self.offsets.as_slice().into(),
        }
    }

    /// Return the underlying data as a slice.
    pub fn as_slice(&self) -> &[usize] {
        self.offsets.as_slice()
    }
}

pub type ChunkOffsetsView<'a> = ChunkOffsetsBase<&'a [usize]>;
pub type ChunkOffsets = ChunkOffsetsBase<Box<[usize]>>;

/// Allow chunk offsets view to be converted directly to slices.
impl<'a> From<ChunkOffsetsView<'a>> for &'a [usize] {
    fn from(view: ChunkOffsetsView<'a>) -> Self {
        view.offsets
    }
}

impl ChunkOffsets {
    /// Build a chunk-offset plan that partitions `dim` into `num_chunks`
    /// near-equal chunks. The first `dim.get() % num_chunks.get()` chunks are
    /// one element larger than the rest.
    ///
    /// Returns an error if the requested partition is not valid (e.g.
    /// `num_chunks.get() > dim.get()`).
    pub fn partition(dim: NonZeroUsize, num_chunks: NonZeroUsize) -> Result<Self, PartitionError> {
        if num_chunks.get() > dim.get() {
            return Err(PartitionError {
                num_chunks: num_chunks.get(),
                dim: dim.get(),
            });
        }
        let mut offsets = vec![0usize; num_chunks.get() + 1].into_boxed_slice();
        fill_chunk_offsets(dim, &mut offsets);
        Ok(Self { dim, offsets })
    }
}

impl<'a> ChunkOffsetsView<'a> {
    /// Fill the caller-owned `scratch` buffer with the partition for `dim`
    /// into `scratch.len() - 1` chunks and return a validated view borrowing it.
    ///
    /// See [`ChunkOffsets::partition`] for the partitioning rule.
    ///
    /// Returns an error if `scratch.len() < 2` or if the requested partition is not
    /// valid (e.g. `scratch.len() - 1 > dim.get()`).
    pub fn partition_into(
        dim: NonZeroUsize,
        scratch: &'a mut [usize],
    ) -> Result<Self, PartitionIntoError> {
        if scratch.len() < 2 {
            return Err(PartitionIntoError::ScratchTooSmall(scratch.len()));
        }
        let num_chunks = scratch.len() - 1;
        if num_chunks > dim.get() {
            return Err(PartitionError {
                num_chunks,
                dim: dim.get(),
            }
            .into());
        }

        fill_chunk_offsets(dim, scratch);
        Ok(Self {
            dim,
            offsets: scratch,
        })
    }
}

/// Internal helper: fill `offsets` with the prefix-sum
/// partitioning of `dimensions` into `num_chunks` near-equal chunks, where
/// `num_chunks = offsets.len() - 1`.
///
/// The first `dimensions.get() % num_chunks` chunks are one element larger than the
/// rest, so each chunk has size `dimensions.get() / num_chunks` or
/// `dimensions.get() / num_chunks + 1` and the total covers `[0, dimensions.get()]`.
///
/// # Panics
///
/// Panics if `offsets.len() <= 1`.
fn fill_chunk_offsets(dimensions: NonZeroUsize, offsets: &mut [usize]) {
    let num_chunks = offsets.len() - 1;
    let dimensions = dimensions.get();
    let mut chunk_offset: usize = 0;
    offsets[0] = chunk_offset;
    for chunk_index in 0..num_chunks {
        chunk_offset += dimensions / num_chunks;
        if chunk_index < (dimensions % num_chunks) {
            chunk_offset += 1;
        }
        offsets[chunk_index + 1] = chunk_offset;
    }
}

///////////////
// ChunkView //
///////////////

/// A view over a slice that partitions the slice into chunks corresponding to a valid
/// PQ chunking configuration.
///
/// This class maintains the invariant that the provided chunking configuration if valid
/// and that the data being partitioned has the correct length.
#[derive(Debug, Clone, Copy)]
pub struct ChunkViewImpl<'a, T>
where
    T: DenseData,
{
    data: T,
    offsets: ChunkOffsetsView<'a>,
}

#[derive(Error, Debug)]
#[non_exhaustive]
#[error(
    "error in chunk view construction, got a slice of length {got} but \
         the provided chunking schema expects a length of {should}"
)]
pub struct ChunkViewError {
    got: usize,
    should: usize,
}

impl<'a, T> ChunkViewImpl<'a, T>
where
    T: DenseData,
{
    /// Construct a new `ChunkView`.
    ///
    /// Returns an error if `data.len() != offsets.dim()`.
    pub fn new<U>(data: U, offsets: ChunkOffsetsView<'a>) -> Result<Self, ChunkViewError>
    where
        T: From<U>,
    {
        let data: T = data.into();

        // Use the `offsets` as the source of truth because it is more likely that a
        // `ChunkOffsetsView` will be constructed once and reused, so is more likely to be
        // the desired outcome.
        let got = data.as_slice().len();
        let should = offsets.dim();
        if got != should {
            Err(ChunkViewError { got, should })
        } else {
            Ok(Self { data, offsets })
        }
    }

    /// Return the number of partitions in the chunking view.
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Return the `i`th chunk of the view.
///
/// # Panics
///
/// Panics if `i >= self.len()`.
impl<T> Index<usize> for ChunkViewImpl<'_, T>
where
    T: DenseData,
{
    type Output = [T::Elem];

    fn index(&self, i: usize) -> &Self::Output {
        &(self.data.as_slice())[self.offsets.at(i)]
    }
}

/// Return the `i`th chunk of the view.
///
/// # Panics
///
/// Panics if `i >= self.len()`.
impl<T> IndexMut<usize> for ChunkViewImpl<'_, T>
where
    T: MutDenseData,
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut (self.data.as_mut_slice())[self.offsets.at(i)]
    }
}

pub type ChunkView<'a, T> = ChunkViewImpl<'a, &'a [T]>;
pub type MutChunkView<'a, T> = ChunkViewImpl<'a, &'a mut [T]>;

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::lazy_format;

    /// This function is only callable with copyable types.
    ///
    /// This lets us test for types we expect to be `Copy`.
    fn is_copyable<T: Copy>(_x: T) -> bool {
        true
    }

    //////////////////
    // ChunkOffsets //
    //////////////////

    #[test]
    fn chunk_offset_happy_path() {
        let offsets_raw: Vec<usize> = vec![0, 1, 3, 6, 10, 12, 13, 14];
        let offsets = ChunkOffsetsView::new(offsets_raw.as_slice()).unwrap();

        assert_eq!(offsets.len(), offsets_raw.len() - 1);
        assert_eq!(offsets.dim(), *offsets_raw.last().unwrap());
        assert!(!offsets.is_empty());

        assert_eq!(offsets.at(0), 0..1);
        assert_eq!(offsets.at(1), 1..3);
        assert_eq!(offsets.at(2), 3..6);
        assert_eq!(offsets.at(3), 6..10);
        assert_eq!(offsets.at(4), 10..12);
        assert_eq!(offsets.at(5), 12..13);
        assert_eq!(offsets.at(6), 13..14);

        // Finally, make sure the type is copyable.
        assert!(is_copyable(offsets));
        // Make sure `as_slice()` properly round-trips.
        assert_eq!(offsets.as_slice(), offsets_raw.as_slice());

        // `to_owned`
        let offsets_owned = offsets.to_owned();
        assert_eq!(offsets_owned.as_slice(), offsets_raw.as_slice());
        assert_ne!(
            offsets_owned.as_slice().as_ptr(),
            offsets_raw.as_slice().as_ptr()
        );
        assert_eq!(offsets_owned.dim, offsets.dim);

        // `as_view`.
        let offsets_view = offsets_owned.as_view();
        assert_eq!(offsets_view, offsets);
        // ensure that pointers are preserved.
        assert_eq!(
            offsets_view.as_slice().as_ptr(),
            offsets_owned.as_slice().as_ptr()
        );
    }

    #[test]
    #[should_panic(expected = "index 5 must be less than len 3")]
    fn chunk_offset_indexing_panic() {
        let offsets = ChunkOffsets::new(Box::new([0, 1, 2, 3])).unwrap();

        // panics
        let _ = offsets.at(5);
    }

    // Construction errors.
    #[test]
    fn chunk_offset_construction_errors() {
        // Pass an empty slice.
        let offsets = ChunkOffsets::new(Box::new([]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must have a length of at least 2, found 0"
        );

        // Pass a slice with length 1.
        let offsets = ChunkOffsets::new(Box::new([0]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must have a length of at least 2, found 1"
        );

        // Doesn't start with zero.
        let offsets = ChunkOffsets::new(Box::new([10, 11, 12, 13]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must begin at 0, not 10"
        );

        // Non-monotonic cases - zero sized chunk
        let offsets = ChunkOffsets::new(Box::new([0, 10, 20, 30, 30, 40, 41]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must be strictly increasing, instead entry 30 at position 3 \
            is followed by 30"
        );

        // Non-monotonic cases - decreasing size
        let offsets = ChunkOffsets::new(Box::new([0, 10, 9, 10, 20]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must be strictly increasing, instead entry 10 at position 1 \
            is followed by 9"
        );

        // Non-monotonic cases - some dimension after the first is zero.
        let offsets = ChunkOffsets::new(Box::new([0, 10, 11, 12, 0]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must be strictly increasing, instead entry 12 at position 3 \
            is followed by 0"
        );

        // Non-monotonic cases - second entry is zero.
        let offsets = ChunkOffsets::new(Box::new([0, 0, 11, 12, 20]));
        assert_eq!(
            offsets.unwrap_err().to_string(),
            "offsets must be strictly increasing, instead entry 0 at position 0 \
            is followed by 0"
        );
    }

    //////////////////////
    // partition builders //
    //////////////////////

    #[test]
    fn partition_happy_path() {
        let nz = |x: usize| NonZeroUsize::new(x).unwrap();

        // Even split: 9 / 3 = 3 each.
        let offsets = ChunkOffsets::partition(nz(9), nz(3)).unwrap();
        assert_eq!(offsets.as_slice(), &[0, 3, 6, 9]);
        assert_eq!(offsets.dim(), 9);
        assert_eq!(offsets.len(), 3);

        // Uneven split: 8 / 3 = 2 r 2 -> first two chunks get an extra element.
        let offsets = ChunkOffsets::partition(nz(8), nz(3)).unwrap();
        assert_eq!(offsets.as_slice(), &[0, 3, 6, 8]);

        // Single chunk degenerate case.
        let offsets = ChunkOffsets::partition(nz(5), nz(1)).unwrap();
        assert_eq!(offsets.as_slice(), &[0, 5]);

        // dimensions == num_pq_chunks: each chunk is size 1.
        let offsets = ChunkOffsets::partition(nz(4), nz(4)).unwrap();
        assert_eq!(offsets.as_slice(), &[0, 1, 2, 3, 4]);

        // The view-into variant matches the owning constructor; num_chunks is
        // inferred from `scratch.len() - 1`.
        let mut scratch = [0usize; 4];
        let view = ChunkOffsetsView::partition_into(nz(8), &mut scratch).unwrap();
        assert_eq!(view.as_slice(), &[0, 3, 6, 8]);
        assert_eq!(view.dim(), 8);
        assert_eq!(view.len(), 3);
        assert_eq!(scratch.as_slice(), &[0, 3, 6, 8]);
    }

    #[test]
    fn partition_construction_errors() {
        let nz = |x: usize| NonZeroUsize::new(x).unwrap();

        // num_chunks > dim -> TooManyChunks (caught explicitly before partitioning).
        let err = ChunkOffsets::partition(nz(3), nz(5)).unwrap_err();
        assert!(
            matches!(
                err,
                PartitionError {
                    num_chunks: 5,
                    dim: 3
                }
            ),
            "expected TooManyChunks, got {err:?}"
        );

        // Scratch length < 2 -> ScratchTooSmall (cannot infer num_chunks).
        let mut too_short = [0usize; 1];
        let err = ChunkOffsetsView::partition_into(nz(8), &mut too_short).unwrap_err();
        assert!(matches!(err, PartitionIntoError::ScratchTooSmall(1)));

        // num_chunks > dim via the view builder too.
        let mut scratch = [0usize; 6];
        let err = ChunkOffsetsView::partition_into(nz(3), &mut scratch).unwrap_err();
        assert!(matches!(
            err,
            PartitionIntoError::PartitionError(PartitionError {
                num_chunks: 5,
                dim: 3
            })
        ));
    }

    ///////////////
    // ChunkView //
    ///////////////

    fn check_chunk_view<T>(
        view: &ChunkViewImpl<'_, T>,
        data: &[i32],
        offsets: &[usize],
        context: &dyn std::fmt::Display,
    ) where
        T: DenseData<Elem = i32>,
    {
        assert_eq!(view.len(), offsets.len() - 1, "{}", context);

        // Ensure that each yielded slice matches that we retrieve manually.
        for i in 0..view.len() {
            let context = lazy_format!("start = {}, {}", i, context);
            let start = offsets[i];
            let stop = offsets[i + 1];

            let expected = &data[start..stop];
            let retrieved = &view[i];

            assert_eq!(retrieved, expected, "{}", context);
        }
    }

    #[test]
    fn test_immutable_chunkview() {
        let data: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        //                        |-----|  |--|  |--------|  |
        //                          c0      c1       c2      c3
        let offsets: Vec<usize> = vec![0, 3, 5, 9, 10];

        let chunks = ChunkOffsetsView::new(offsets.as_slice()).unwrap();
        let chunk_view = ChunkView::new(data.as_slice(), chunks).unwrap();

        assert_eq!(chunk_view.len(), offsets.len() - 1);
        assert_eq!(chunk_view.len(), chunks.len());

        assert!(is_copyable(chunk_view));
        let context = lazy_format!("chunkview happy path");
        check_chunk_view(&chunk_view, &data, &offsets, &context);
    }

    #[test]
    fn test_chunkview_construction_error() {
        let data: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        //                        |-----|  |--|  |--------|  |
        //                          c0      c1       c2      c3
        let offsets: Vec<usize> = vec![0, 3, 5, 9]; // One too short.

        let chunks = ChunkOffsetsView::new(offsets.as_slice()).unwrap();
        let chunk_view = ChunkView::new(data.as_slice(), chunks);
        assert!(chunk_view.is_err());
        assert_eq!(
            chunk_view.unwrap_err().to_string(),
            "error in chunk view construction, got a slice of length 10 but \
             the provided chunking schema expects a length of 9"
        );
    }

    #[test]
    fn test_mutable_chunkview() {
        let mut data: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        //                            |-----|  |--|  |--------|  |
        //                              c0      c1       c2      c3
        let offsets: Vec<usize> = vec![0, 3, 5, 9, 10];

        // We need to clone the original data before constructing the mutable chunk view
        // to avoid both a mutable and immutable borrow in the checking function.
        let data_clone = data.clone();

        let chunks = ChunkOffsetsView::new(offsets.as_slice()).unwrap();
        let mut chunk_view = MutChunkView::new(data.as_mut_slice(), chunks).unwrap();

        assert_eq!(chunk_view.len(), offsets.len() - 1);
        assert_eq!(chunk_view.len(), chunks.len());

        let context = lazy_format!("mutchunkview happy path");
        check_chunk_view(&chunk_view, &data_clone, &offsets, &context);

        // Make sure that we can assign through the view.
        for i in 0..chunk_view.len() {
            let i_i32: i32 = i.try_into().unwrap();

            chunk_view[i].iter_mut().for_each(|d| *d = i_i32);
        }

        // chunk 0
        assert_eq!(data[0], 0);
        assert_eq!(data[1], 0);
        assert_eq!(data[2], 0);

        // chunk 1
        assert_eq!(data[3], 1);
        assert_eq!(data[4], 1);

        // chunk 2
        assert_eq!(data[5], 2);
        assert_eq!(data[6], 2);
        assert_eq!(data[7], 2);
        assert_eq!(data[8], 2);

        // chunk 3
        assert_eq!(data[9], 3);
    }
}
