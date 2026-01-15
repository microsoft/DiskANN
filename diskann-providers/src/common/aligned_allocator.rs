/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::{
    ops::{Deref, DerefMut, Range},
    ptr::copy_nonoverlapping,
};

use diskann::{ANNError, ANNResult};
use diskann_quantization::{
    alloc::{AlignedAllocator, Poly},
    num::PowerOfTwo,
};

/// A box that holds a slice but is aligned to the specified layout for potential
/// cache efficiency improvements.
#[derive(Debug)]
pub struct AlignedBoxWithSlice<T> {
    val: Poly<[T], AlignedAllocator>,
}

impl<T> AlignedBoxWithSlice<T>
where
    T: Default,
{
    /// Creates a new `AlignedBoxWithSlice` with the given capacity and alignment.
    /// The allocated memory are set to `T::default()`..
    ///
    /// # Error
    ///
    /// Return IndexError if the alignment is not a power of two or if the layout is invalid.
    pub fn new(capacity: usize, alignment: usize) -> ANNResult<Self> {
        let allocator =
            AlignedAllocator::new(PowerOfTwo::new(alignment).map_err(ANNError::log_index_error)?);
        let val = Poly::from_iter((0..capacity).map(|_| T::default()), allocator)
            .map_err(ANNError::log_index_error)?;
        Ok(Self { val })
    }

    /// Returns a reference to the slice.
    pub fn as_slice(&self) -> &[T] {
        &self.val
    }

    /// Returns a mutable reference to the slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.val
    }

    /// Copies data from the source slice to the destination box.
    pub fn memcpy(&mut self, src: &[T]) -> ANNResult<()> {
        if src.len() > self.val.len() {
            return Err(ANNError::log_index_error(format_args!(
                "source slice is too large (src:{}, dst:{})",
                src.len(),
                self.val.len()
            )));
        }

        // Check that they don't overlap
        let src_ptr = src.as_ptr();
        let src_end = unsafe { src_ptr.add(src.len()) };
        let dst_ptr = self.val.as_mut_ptr();
        let dst_end = unsafe { dst_ptr.add(self.val.len()) };

        if src_ptr < dst_end && src_end > dst_ptr {
            return Err(ANNError::log_index_error("Source and destination overlap"));
        }

        // Call is safe because we checked that
        // 1. the destination is large enough.
        // 2. the source and destination don't overlap.
        unsafe {
            copy_nonoverlapping(src.as_ptr(), self.val.as_mut_ptr(), src.len());
        }

        Ok(())
    }

    /// Split the range of memory into nonoverlapping mutable slices.
    /// The number of returned slices is (range length / slice_len) and each has a length of slice_len.
    pub fn split_into_nonoverlapping_mut_slices(
        &mut self,
        range: Range<usize>,
        slice_len: usize,
    ) -> ANNResult<Vec<&mut [T]>> {
        if !range.len().is_multiple_of(slice_len) || range.end > self.len() {
            return Err(ANNError::log_index_error(format_args!(
                "Cannot split range ({:?}) of AlignedBoxWithSlice (len: {}) into nonoverlapping mutable slices with length {}",
                range,
                self.len(),
                slice_len,
            )));
        }

        let mut slices = Vec::with_capacity(range.len() / slice_len);
        let mut remaining_slice = &mut self.val[range];

        while remaining_slice.len() >= slice_len {
            let (left, right) = remaining_slice.split_at_mut(slice_len);
            slices.push(left);
            remaining_slice = right;
        }

        Ok(slices)
    }
}

impl<T> Deref for AlignedBoxWithSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> DerefMut for AlignedBoxWithSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(miri))]
    use std::{cell::RefCell, rc::Rc};

    use diskann::ANNErrorKind;
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
    fn create_alignedvec_works_32() {
        (0..100).for_each(|_| {
            let size = TEST_SIZE;
            let data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
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
    fn create_alignedvec_works_256() {
        let mut rng = crate::utils::create_rnd_in_tests();

        (0..100).for_each(|_| {
            let n = rng.random::<u8>();
            let size = usize::from(n) + 1;
            let data = AlignedBoxWithSlice::<u8>::new(size, 256).unwrap();
            assert_eq!(data.len(), size, "Capacity should match");

            let ptr = data.as_ptr() as usize;
            assert_eq!(ptr % 256, 0, "Ptr should be aligned to 32");

            // assert that the slice is initialized.
            (0..size).for_each(|i| {
                assert_eq!(data[i], u8::default());
            });

            drop(data);
        });
    }

    #[test]
    fn create_zero_length_box() {
        let x = AlignedBoxWithSlice::<f32>::new(0, 16).unwrap();
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn as_slice_test() {
        let size = TEST_SIZE;
        let data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        // assert that the slice is initialized.
        (0..size).for_each(|i| {
            assert_eq!(data[i], f32::default());
        });

        let slice = data.as_slice();
        (0..size).for_each(|i| {
            assert_eq!(slice[i], f32::default());
        });
    }

    #[test]
    fn as_mut_slice_test() {
        let size = TEST_SIZE;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut_slice = data.as_mut_slice();
        (0..size).for_each(|i| {
            assert_eq!(mut_slice[i], f32::default());
        });
    }

    #[test]
    fn memcpy_test() {
        let size = TEST_SIZE;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut destination = AlignedBoxWithSlice::<f32>::new(size - 2, 32).unwrap();
        let mut_destination = destination.as_mut_slice();
        data.memcpy(mut_destination).unwrap();
        (0..size - 2).for_each(|i| {
            assert_eq!(data[i], mut_destination[i]);
        });
    }

    #[test]
    #[should_panic(expected = "source slice is too large")]
    fn memcpy_panic_test() {
        let size = TEST_SIZE;
        let mut data = AlignedBoxWithSlice::<f32>::new(size - 2, 32).unwrap();
        let mut destination = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let mut_destination = destination.as_mut_slice();
        data.memcpy(mut_destination).unwrap();
    }

    // NOTE: This function is wildly unsafe and miri complains about it before we even
    // get to the internal check.
    //
    // In safe Rust, it is impossible to obtain overlapping mutable and immutable slices,
    // but the check in this function is a safe-guard against UB as a result of unsafe code,
    // so we still need to exercise it.
    #[cfg(not(miri))]
    #[test]
    fn test_memcpy_overlap() {
        let aligned_box = Rc::new(RefCell::new(AlignedBoxWithSlice::new(4, 16).unwrap()));
        aligned_box
            .borrow_mut()
            .as_mut_slice()
            .copy_from_slice(&[1, 2, 3, 4]);

        let src_ptr = aligned_box.as_ptr();
        let src = &unsafe { src_ptr.as_ref().unwrap() }.as_slice()[0..3];

        let result = aligned_box.borrow_mut().memcpy(src);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), ANNErrorKind::IndexError);
    }

    #[test]
    fn split_into_nonoverlapping_mut_slices_test() {
        let size = 10;
        let slice_len = 2;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let slices = data
            .split_into_nonoverlapping_mut_slices(2..8, slice_len)
            .unwrap();
        assert_eq!(slices.len(), 3);
        for (i, slice) in slices.into_iter().enumerate() {
            assert_eq!(slice.len(), slice_len);
            slice[0] = i as f32 + 1.0;
            slice[1] = i as f32 + 1.0;
        }
        let expected_arr = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0];
        assert_eq!(data.as_ref(), &expected_arr);
    }

    #[test]
    fn split_into_nonoverlapping_mut_slices_error_when_indivisible() {
        let size = 10;
        let slice_len = 2;
        let range = 2..7;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let result = data.split_into_nonoverlapping_mut_slices(range.clone(), slice_len);
        assert!(result.is_err_and(|e| e.kind() == ANNErrorKind::IndexError));
    }
}
