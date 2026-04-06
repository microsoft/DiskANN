/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    alloc::{AlignedAllocator, Poly},
    num::PowerOfTwo,
};

/// An aligned, heap-allocated slice.
///
/// This is a [`Poly`] backed by an [`AlignedAllocator`], providing a
/// cache-aligned buffer of elements.
pub type AlignedSlice<T> = Poly<[T], AlignedAllocator>;

/// Creates a new [`AlignedSlice`] with the given capacity and alignment.
/// The allocated memory is set to `T::default()`.
///
/// # Error
///
/// Returns an `IndexError` if the alignment is not a power of two or if the layout is invalid.
pub fn aligned_alloc<T: Default>(capacity: usize, alignment: usize) -> ANNResult<AlignedSlice<T>> {
    let allocator =
        AlignedAllocator::new(PowerOfTwo::new(alignment).map_err(ANNError::log_index_error)?);
    Poly::from_iter((0..capacity).map(|_| T::default()), allocator)
        .map_err(ANNError::log_index_error)
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            // Use smaller allocations for Miri.
            const TEST_SIZE: usize = 100;
        } else {
            const TEST_SIZE: usize = 1_000_000;
        }
    }

    #[test]
    fn create_aligned_alloc_works_32() {
        (0..100).for_each(|_| {
            let size = TEST_SIZE;
            let data = aligned_alloc::<f32>(size, 32).unwrap();
            assert_eq!(data.len(), size, "Capacity should match");

            let ptr = data.as_ptr() as usize;
            assert_eq!(ptr % 32, 0, "Ptr should be aligned to 32");

            // assert that the slice is initialized.
            (0..size).for_each(|i| {
                assert_eq!(data[i], f32::default());
            });

            drop(data);
        });
    }

    #[test]
    fn create_aligned_alloc_works_256() {
        let mut rng = crate::utils::create_rnd_in_tests();

        (0..100).for_each(|_| {
            let n = rng.random::<u8>();
            let size = usize::from(n) + 1;
            let data = aligned_alloc::<u8>(size, 256).unwrap();
            assert_eq!(data.len(), size, "Capacity should match");

            let ptr = data.as_ptr() as usize;
            assert_eq!(ptr % 256, 0, "Ptr should be aligned to 256");

            // assert that the slice is initialized.
            (0..size).for_each(|i| {
                assert_eq!(data[i], u8::default());
            });

            drop(data);
        });
    }

    #[test]
    fn create_zero_length_alloc() {
        let x = aligned_alloc::<f32>(0, 16).unwrap();
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn chunks_mut_test() {
        let size = 10;
        let slice_len = 2;
        let mut data = aligned_alloc::<f32>(size, 32).unwrap();
        let slices: Vec<&mut [f32]> = data[2..8].chunks_mut(slice_len).collect();
        assert_eq!(slices.len(), 3);
        for (i, slice) in slices.into_iter().enumerate() {
            assert_eq!(slice.len(), slice_len);
            slice[0] = i as f32 + 1.0;
            slice[1] = i as f32 + 1.0;
        }
        let expected_arr = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0];
        assert_eq!(data.as_ref(), &expected_arr);
    }
}
