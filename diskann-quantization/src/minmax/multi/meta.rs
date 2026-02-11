// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Meta types and compression for MinMax quantized multi-vectors.

use std::ptr::NonNull;

use super::super::vectors::DataMutRef;
use super::super::MinMaxQuantizer;
use crate::bits::{Representation, Unsigned};
use crate::minmax::{self, Data};
use crate::multi_vector::matrix::{
    Defaulted, NewMut, NewOwned, NewRef, Repr, ReprMut, ReprOwned, SliceError,
};
use crate::multi_vector::{LayoutError, Mat, MatMut, MatRef, Standard};
use crate::scalar::InputContainsNaN;
use crate::utils;
use crate::CompressInto;

////////////////
// MinMaxMeta //
////////////////

/// Metadata for MinMax quantized multi-vectors.
///
/// Stores the intrinsic dimension (output dimension after transform) which is
/// needed to interpret each row of the quantized data. The row stride in bytes
/// is computed as [`Data::<NBITS>::canonical_bytes(intrinsic_dim)`].
#[derive(Debug, Clone, Copy)]
pub struct MinMaxMeta<const NBITS: usize> {
    nrows: usize,
    intrinsic_dim: usize,
}

impl<const NBITS: usize> MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    /// Creates new MinMax metadata with the given number of rows and intrinsic dimension.
    pub fn new(nrows: usize, intrinsic_dim: usize) -> Self {
        Self {
            nrows,
            intrinsic_dim,
        }
    }

    /// Returns the `intrinsic_dim`
    pub fn intrinsic_dim(&self) -> usize {
        self.intrinsic_dim
    }

    /// Returns the number of bytes from a canonical repr of
    /// a minmax quantized vector, see [minmax::Data::canonical_bytes]
    pub fn ncols(&self) -> usize {
        Data::<NBITS>::canonical_bytes(self.intrinsic_dim)
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<u8>() * self.nrows() * self.ncols()
    }
}

// SAFETY: The implementation correctly computes row offsets and constructs valid
// DataRef types from the underlying byte buffer. The ncols() method
// returns the canonical byte size for each row, and get_row properly
// slices the buffer at the correct offsets.
unsafe impl<const NBITS: usize> Repr for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Row<'a> = crate::minmax::DataRef<'a, NBITS>;

    fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns `Layout` for for allocating [`crate::minmax::Data`]
    ///
    /// # Safety
    /// - [`crate::minmax::MinMaxCompensation`] does not require alignment
    ///   since it is always accessed by copying bytes first before casting.
    fn layout(&self) -> Result<std::alloc::Layout, LayoutError> {
        Ok(std::alloc::Layout::array::<u8>(
            self.nrows() * self.ncols(),
        )?)
    }

    /// Returns an immutable reference to the i-th row.
    ///
    /// # Safety
    /// - The original pointer must point to a valid allocation of at least
    ///   `std::mem::size_of::<u8>() * self.nrows() * self.ncols()` bytes for lifetime `'a`.
    /// - `i` must be less than `self.nrows`.
    unsafe fn get_row<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::Row<'a> {
        debug_assert!(i < self.nrows);
        let len = self.ncols();

        // SAFETY: If the orginal pointer was initialized correctly, by the bounds check for `i`,
        // `i * self.ncols() .. (i + 1) * self.ncols()` is a valid memory access.
        // SAFETY: `ncols` is initialized using `minmax::Data::canonical_bytes` and `intrinsic_dim`,
        unsafe {
            let row_ptr = ptr.as_ptr().add(i * len);
            let slice = std::slice::from_raw_parts(row_ptr, len);

            minmax::DataRef::<'a, NBITS>::from_canonical_unchecked(slice, self.intrinsic_dim)
        }
    }
}

// SAFETY: The implementation correctly computes row offsets and constructs valid
// DataMutRef types from the underlying byte buffer.
unsafe impl<const NBITS: usize> ReprMut for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type RowMut<'a> = crate::minmax::DataMutRef<'a, NBITS>;

    /// Returns a mutable reference to the i-th row.
    ///
    /// # Safety
    /// - The original pointer must point to a valid allocation of at least
    ///   `std::mem::size_of::<u8>() * self.bytes()` bytes for lifetime `'a`.
    /// - `i` must be less than `self.nrows`.
    /// - Caller must make sure they have exclusive access to the row
    unsafe fn get_row_mut<'a>(self, ptr: NonNull<u8>, i: usize) -> Self::RowMut<'a> {
        debug_assert!(i < self.nrows);
        let len = self.ncols();

        // SAFETY: If the orginal pointer was initialized correctly, by the bounds check for `i`,
        // `i * self.ncols() .. (i + 1) * self.ncols()` is a valid memory access.
        // SAFETY: `ncols` is initialized using `minmax::Data::canonical_bytes` and `intrinsic_dim`,
        // SAFETY: Caller has ensured we have exclusive access to the bytes in row `i`
        unsafe {
            let row_ptr = ptr.as_ptr().add(i * len);
            let slice = std::slice::from_raw_parts_mut(row_ptr, len);

            minmax::DataMutRef::<'a, NBITS>::from_canonical_front_mut_unchecked(
                slice,
                self.intrinsic_dim,
            )
        }
    }
}

// SAFETY: The drop implementation correctly reconstructs a Box from the raw pointer
// using the same size that was used for allocation (self.bytes()), allowing Box
// to properly deallocate the memory.
unsafe impl<const NBITS: usize> ReprOwned for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    /// Deallocates the memory pointed to by `ptr`.
    ///
    /// # Safety:
    /// - The caller guarantees that `ptr` was allocated with the correct layout.
    unsafe fn drop(self, ptr: NonNull<u8>) {
        let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr.as_ptr(), self.bytes());
        let _ = Box::from_raw(slice_ptr);
    }
}

// SAFETY: `ptr` points to a properly sized slice that is compatible with the drop
// logic in `Self as ReprOwned`. Box guarantees that the initial construction
// will be non-null.
unsafe impl<const NBITS: usize> NewOwned<Defaulted> for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Error = crate::error::Infallible;
    fn new_owned(self, _: Defaulted) -> Result<Mat<Self>, Self::Error> {
        let b: Box<[u8]> = (0..self.bytes()).map(|_| u8::default()).collect();

        // SAFETY: Box guarantees that its pointer us non-null.
        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(b)) }.cast::<u8>();

        // SAFETY: `ptr` points to a properly sized slice that is compatible with the drop
        // logic in `Self as ReprOwned`.
        let mat = unsafe { Mat::from_raw_parts(self, ptr) };
        Ok(mat)
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`Repr`] in this case.
unsafe impl<const NBITS: usize> NewRef<u8> for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Error = SliceError;

    fn new_ref(self, slice: &[u8]) -> Result<MatRef<'_, Self>, Self::Error> {
        let expected = self.bytes();
        if slice.len() != expected {
            return Err(SliceError::LengthMismatch {
                expected,
                found: slice.len(),
            });
        }

        // SAFETY: We've verified that the slice has the correct length.
        Ok(unsafe { MatRef::from_raw_parts(self, utils::as_nonnull(slice).cast::<u8>()) })
    }
}

// SAFETY: This checks that the slice has the correct length, which is all that is
// required for [`ReprMut`] in this case since [`minmax::Data`] has no alignment requirement.
unsafe impl<const NBITS: usize> NewMut<u8> for MinMaxMeta<NBITS>
where
    Unsigned: Representation<NBITS>,
{
    type Error = SliceError;

    fn new_mut(self, slice: &mut [u8]) -> Result<MatMut<'_, Self>, Self::Error> {
        let expected = self.bytes();
        if slice.len() != expected {
            return Err(SliceError::LengthMismatch {
                expected,
                found: slice.len(),
            });
        }

        // SAFETY: We've verified that the slice has the correct length.
        Ok(unsafe { MatMut::from_raw_parts(self, utils::as_nonnull_mut(slice).cast::<u8>()) })
    }
}

//////////////////
// CompressInto //
//////////////////

impl<'a, 'b, const NBITS: usize, T>
    CompressInto<MatRef<'a, Standard<T>>, MatMut<'b, MinMaxMeta<NBITS>>> for MinMaxQuantizer
where
    T: Copy + Into<f32>,
    Unsigned: Representation<NBITS>,
{
    type Error = InputContainsNaN;

    type Output = ();

    /// Compress a multi-vector of full-precision vectors into a multi-vector of MinMax quantized vectors.
    ///
    /// This method iterates row by row over the input matrix, quantizing each full-precision
    /// vector into the corresponding row of the output MinMax matrix.
    ///
    /// # Error
    ///
    /// Returns [`InputContainsNaN`] error if any input vector contains `NaN`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `from.num_vectors() != to.num_vectors()`: The input and output must have the same number of vectors.
    /// * `from.vector_dim() != self.dim()`: Each input vector must have the same dimensionality as the quantizer.
    /// * The output intrinsic dimension doesn't match `self.output_dim()`.
    fn compress_into(
        &self,
        from: MatRef<'a, Standard<T>>,
        mut to: MatMut<'b, MinMaxMeta<NBITS>>,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            from.num_vectors(),
            to.num_vectors(),
            "input and output must have the same number of vectors: {} != {}",
            from.num_vectors(),
            to.num_vectors()
        );
        assert_eq!(
            from.vector_dim(),
            self.dim(),
            "input vectors must match quantizer dimension: {} != {}",
            from.vector_dim(),
            self.dim()
        );
        assert_eq!(
            to.repr().intrinsic_dim(),
            self.output_dim(),
            "output intrinsic dimension must match quantizer output dimension: {} != {}",
            to.repr().intrinsic_dim(),
            self.output_dim()
        );

        for (from_row, to_row) in from.rows().zip(to.rows_mut()) {
            // Use the single-vector `CompressInto` implementation
            let _ = <MinMaxQuantizer as CompressInto<&[T], DataMutRef<'_, NBITS>>>::compress_into(
                self, from_row, to_row,
            )?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::transforms::NullTransform;
    use crate::algorithms::Transform;
    use crate::minmax::vectors::DataRef;
    use crate::num::Positive;
    use diskann_utils::{Reborrow, ReborrowMut};
    use std::num::NonZeroUsize;

    /// Test dimensions for comprehensive coverage.
    const TEST_DIMS: &[usize] = &[1, 2, 3, 4, 7, 8, 16, 31, 32, 64];
    /// Test vector counts for multi-vector matrices.
    const TEST_NVECS: &[usize] = &[1, 2, 3, 5, 10];

    /// Macro to generate a single test that runs a generic function for all bitrates (1, 2, 4, 8).
    macro_rules! expand_to_bitrates {
        ($name:ident, $func:ident) => {
            #[test]
            fn $name() {
                $func::<1>();
                $func::<2>();
                $func::<4>();
                $func::<8>();
            }
        };
    }

    // ==================
    // Helper Functions
    // ==================

    /// Creates a quantizer for testing with the given dimension.
    fn make_quantizer(dim: usize) -> MinMaxQuantizer {
        MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        )
    }

    /// Generates deterministic test data for a multi-vector.
    fn generate_test_data(num_vectors: usize, dim: usize) -> Vec<f32> {
        (0..num_vectors * dim)
            .map(|i| {
                let row = i / dim;
                let col = i % dim;
                // Vary values to exercise quantization: use sin/cos patterns
                ((row as f32 + 1.0) * (col as f32 + 0.5)).sin() * 10.0 + (row as f32)
            })
            .collect()
    }

    /// Compresses a single vector using the single-vector CompressInto and returns the bytes.
    fn compress_single_vector<const NBITS: usize>(
        quantizer: &MinMaxQuantizer,
        input: &[f32],
        dim: usize,
    ) -> Vec<u8>
    where
        Unsigned: Representation<NBITS>,
    {
        let row_bytes = Data::<NBITS>::canonical_bytes(dim);
        let mut output = vec![0u8; row_bytes];
        let output_ref =
            crate::minmax::DataMutRef::<NBITS>::from_canonical_front_mut(&mut output, dim).unwrap();
        let _ = <MinMaxQuantizer as CompressInto<&[f32], DataMutRef<'_, NBITS>>>::compress_into(
            quantizer, input, output_ref,
        )
        .expect("single-vector compression should succeed");
        output
    }

    // =============================
    // Construction
    // =============================

    mod construction {
        use super::*;

        /// Tests NewOwned (Mat::new with Defaulted) for various dimensions and sizes.
        fn test_new_owned<const NBITS: usize>()
        where
            Unsigned: Representation<NBITS>,
        {
            for &dim in TEST_DIMS {
                for &num_vectors in TEST_NVECS {
                    let meta = MinMaxMeta::<NBITS>::new(num_vectors, dim);
                    let mat: Mat<MinMaxMeta<NBITS>> =
                        Mat::new(meta, Defaulted).expect("NewOwned should succeed");

                    // Verify num_vectors and intrinsic_dim
                    assert_eq!(mat.num_vectors(), num_vectors);
                    assert_eq!(mat.repr().intrinsic_dim(), dim);

                    // Verify Repr::nrows() and Repr::ncols()
                    let expected_row_bytes = Data::<NBITS>::canonical_bytes(dim);
                    assert_eq!(mat.repr().nrows(), num_vectors);
                    assert_eq!(mat.repr().ncols(), expected_row_bytes);

                    // Verify Repr::layout() returns correct size (nrows * ncols bytes)
                    let expected_bytes = expected_row_bytes * num_vectors;
                    let layout = mat.repr().layout().expect("layout should succeed");
                    assert_eq!(layout.size(), expected_bytes);

                    // Verify we can access all rows
                    for i in 0..num_vectors {
                        let row = mat.get_row(i);
                        assert!(row.is_some(), "row {i} should exist");
                        assert_eq!(row.unwrap().len(), dim);
                    }

                    // Out of bounds access should return None
                    assert!(mat.get_row(num_vectors).is_none());
                }
            }
        }

        expand_to_bitrates!(new_owned, test_new_owned);

        /// Tests NewRef (MatRef::new) for valid slices.
        fn test_new_ref<const NBITS: usize>()
        where
            Unsigned: Representation<NBITS>,
        {
            for &dim in TEST_DIMS {
                for &num_vectors in TEST_NVECS {
                    let meta = MinMaxMeta::<NBITS>::new(num_vectors, dim);
                    let expected_row_bytes = Data::<NBITS>::canonical_bytes(dim);
                    let expected_bytes = expected_row_bytes * num_vectors;

                    // Valid slice
                    let data = vec![0u8; expected_bytes];
                    let mat_ref = MatRef::new(meta, &data);
                    assert!(mat_ref.is_ok(), "NewRef should succeed for correct size");
                    let mat_ref = mat_ref.unwrap();

                    // Verify num_vectors and intrinsic_dim
                    assert_eq!(mat_ref.num_vectors(), num_vectors);
                    assert_eq!(mat_ref.repr().intrinsic_dim(), dim);

                    // Verify Repr::nrows() and Repr::ncols()
                    assert_eq!(mat_ref.repr().nrows(), num_vectors);
                    assert_eq!(mat_ref.repr().ncols(), expected_row_bytes);

                    // Verify Repr::layout() returns correct size
                    let layout = mat_ref.repr().layout().expect("layout should succeed");
                    assert_eq!(layout.size(), expected_bytes);
                }
            }
        }

        expand_to_bitrates!(new_ref, test_new_ref);

        /// Tests NewMut (MatMut::new) for valid slices.
        fn test_new_mut<const NBITS: usize>()
        where
            Unsigned: Representation<NBITS>,
        {
            for &dim in TEST_DIMS {
                for &num_vectors in TEST_NVECS {
                    let meta = MinMaxMeta::<NBITS>::new(num_vectors, dim);
                    let expected_row_bytes = Data::<NBITS>::canonical_bytes(dim);
                    let expected_bytes = expected_row_bytes * num_vectors;

                    let mut data = vec![0u8; expected_bytes];
                    let mat_mut = MatMut::new(meta, &mut data);
                    assert!(mat_mut.is_ok(), "NewMut should succeed for correct size");
                    let mat_mut = mat_mut.unwrap();

                    // Verify num_vectors and intrinsic_dim
                    assert_eq!(mat_mut.num_vectors(), num_vectors);
                    assert_eq!(mat_mut.repr().intrinsic_dim(), dim);

                    // Verify Repr::nrows() and Repr::ncols()
                    assert_eq!(mat_mut.repr().nrows(), num_vectors);
                    assert_eq!(mat_mut.repr().ncols(), expected_row_bytes);

                    // Verify Repr::layout() returns correct size
                    let layout = mat_mut.repr().layout().expect("layout should succeed");
                    assert_eq!(layout.size(), expected_bytes);
                }
            }
        }

        expand_to_bitrates!(new_mut, test_new_mut);

        #[test]
        fn slice_length_mismatch_errors() {
            let dim = 4;
            let num_vectors = 2;
            let meta = MinMaxMeta::<8>::new(num_vectors, dim);
            let expected_bytes = DataRef::<8>::canonical_bytes(dim) * num_vectors;

            // Too short
            let short_data = vec![0u8; expected_bytes - 1];
            let result = MatRef::new(meta, &short_data);
            assert!(
                matches!(result, Err(SliceError::LengthMismatch { .. })),
                "should fail for too-short slice"
            );

            // Too long
            let long_data = vec![0u8; expected_bytes + 1];
            let result = MatRef::new(meta, &long_data);
            assert!(
                matches!(result, Err(SliceError::LengthMismatch { .. })),
                "should fail for too-long slice"
            );

            // Mutable version - too short
            let mut short_mut = vec![0u8; expected_bytes - 1];
            let result = MatMut::new(meta, &mut short_mut);
            assert!(
                matches!(result, Err(SliceError::LengthMismatch { .. })),
                "MatMut should fail for too-short slice"
            );
        }
    }

    // ===========================
    // CompressInto Multi-Vector
    // ===========================

    mod compress_into {
        use super::*;

        /// Tests that multi-vector CompressInto produces identical results to
        /// single-vector CompressInto applied row-by-row.
        fn test_compress_matches_single<const NBITS: usize>()
        where
            Unsigned: Representation<NBITS>,
        {
            for &dim in TEST_DIMS {
                for &num_vectors in TEST_NVECS {
                    let quantizer = make_quantizer(dim);
                    let input_data = generate_test_data(num_vectors, dim);

                    // Multi-vector compression
                    let input_view =
                        MatRef::new(Standard::new(num_vectors, dim).unwrap(), &input_data)
                            .expect("input view creation");

                    let mut multi_mat: Mat<MinMaxMeta<NBITS>> =
                        Mat::new(MinMaxMeta::new(num_vectors, dim), Defaulted)
                            .expect("output mat creation");

                    quantizer
                        .compress_into(input_view, multi_mat.reborrow_mut())
                        .expect("multi-vector compression");

                    // Compare each row with single-vector compression
                    for i in 0..num_vectors {
                        let row_input = &input_data[i * dim..(i + 1) * dim];
                        let expected_bytes =
                            compress_single_vector::<NBITS>(&quantizer, row_input, dim);

                        let actual_row = multi_mat.get_row(i).expect("row should exist");

                        // Compare metadata
                        // SAFETY: expected_bytes was produced by compress_single_vector with correct dim
                        let expected_ref = unsafe {
                            DataRef::<NBITS>::from_canonical_unchecked(&expected_bytes, dim)
                        };
                        assert_eq!(
                            actual_row.meta(),
                            expected_ref.meta(),
                            "metadata mismatch at row {i}, dim={dim}, num_vectors={num_vectors}, NBITS={NBITS}"
                        );

                        // Compare quantized values
                        for j in 0..dim {
                            assert_eq!(
                                actual_row.vector().get(j).unwrap(),
                                expected_ref.vector().get(j).unwrap(),
                                "quantized value mismatch at row {i}, col {j}"
                            );
                        }
                    }
                }
            }
        }

        expand_to_bitrates!(compress_matches_single, test_compress_matches_single);

        /// Tests row iteration after compression.
        fn test_row_iteration<const NBITS: usize>()
        where
            Unsigned: Representation<NBITS>,
        {
            let dim = 8;
            let num_vectors = 5;
            let quantizer = make_quantizer(dim);
            let input_data = generate_test_data(num_vectors, dim);

            let input_view = MatRef::new(Standard::new(num_vectors, dim).unwrap(), &input_data)
                .expect("input view");

            let mut mat: Mat<MinMaxMeta<NBITS>> =
                Mat::new(MinMaxMeta::new(num_vectors, dim), Defaulted).expect("mat creation");

            quantizer
                .compress_into(input_view, mat.reborrow_mut())
                .expect("compression");

            // Test rows() iterator using reborrow
            let view = mat.reborrow();
            let mut count = 0;
            for row in view.rows() {
                assert_eq!(row.len(), dim, "row should have correct dimension");
                count += 1;
            }
            assert_eq!(count, num_vectors);
        }

        expand_to_bitrates!(row_iteration, test_row_iteration);
    }

    // ===========================
    // Error Cases
    // ===========================

    mod error_cases {
        use super::*;

        #[test]
        #[should_panic(expected = "input and output must have the same number of vectors")]
        fn compress_into_vector_count_mismatch() {
            const NBITS: usize = 8;
            let dim = 4;
            let quantizer = make_quantizer(dim);

            // Input has 3 vectors
            let input_data = generate_test_data(3, dim);
            let input_view =
                MatRef::new(Standard::new(3, dim).unwrap(), &input_data).expect("input view");

            // Output has 2 vectors (mismatch)
            let mut mat: Mat<MinMaxMeta<NBITS>> =
                Mat::new(MinMaxMeta::new(2, dim), Defaulted).expect("mat creation");

            let _ = quantizer.compress_into(input_view, mat.reborrow_mut());
        }

        #[test]
        #[should_panic(expected = "input vectors must match quantizer dimension")]
        fn compress_into_input_dim_mismatch() {
            const NBITS: usize = 8;
            let quantizer = make_quantizer(4); // quantizer expects dim=4

            // Input has dim=8 (mismatch)
            let input_data = generate_test_data(2, 8);
            let input_view =
                MatRef::new(Standard::new(2, 8).unwrap(), &input_data).expect("input view");

            // Output correctly has dim=4
            let mut mat: Mat<MinMaxMeta<NBITS>> =
                Mat::new(MinMaxMeta::new(2, 4), Defaulted).expect("mat creation");

            let _ = quantizer.compress_into(input_view, mat.reborrow_mut());
        }

        #[test]
        #[should_panic(
            expected = "output intrinsic dimension must match quantizer output dimension"
        )]
        fn compress_into_output_dim_mismatch() {
            const NBITS: usize = 8;
            let quantizer = make_quantizer(4);

            // Input correctly has dim=4
            let input_data = generate_test_data(2, 4);
            let input_view =
                MatRef::new(Standard::new(2, 4).unwrap(), &input_data).expect("input view");

            // Output has intrinsic_dim=8 (mismatch)
            let row_bytes = Data::<NBITS>::canonical_bytes(8);
            let mut output_data = vec![0u8; 2 * row_bytes];
            let output_view =
                MatMut::new(MinMaxMeta::<NBITS>::new(2, 8), &mut output_data).expect("output view");

            let _ = quantizer.compress_into(input_view, output_view);
        }
    }
}
