/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::Poly;
use crate::alloc::{AlignedAllocator, AllocatorError};

use crate::num::PowerOfTwo;

/// Type alias for an aligned, heap-allocated slice
///
/// Shorthand for `Poly<[T], AlignedAllocator>` which is intended
/// for allocations requiring specific alignment (e.g. cache-line or disk-sector alignment)
pub type AlignedSlice<T> = Poly<[T], AlignedAllocator>;

/// Create a new [`AlignedSlice`] with the given capacity and alignment
/// initialized to `T::default()`
pub fn aligned_slice<T: Default>(
    capacity: usize,
    alignment: PowerOfTwo,
) -> Result<AlignedSlice<T>, AllocatorError> {
    let allocator = AlignedAllocator::new(alignment);
    Poly::from_iter((0..capacity).map(|_| T::default()), allocator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_slice_alignment_32() {
        let data = aligned_slice::<f32>(1000, PowerOfTwo::new(32).unwrap()).unwrap();
        assert_eq!(data.len(), 1000);
        assert_eq!(data.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn aligned_slice_alignment_256() {
        let data = aligned_slice::<u8>(500, PowerOfTwo::new(256).unwrap()).unwrap();
        assert_eq!(data.len(), 500);
        assert_eq!(data.as_ptr() as usize % 256, 0);
    }

    #[test]
    fn aligned_slice_alignment_512() {
        let data = aligned_slice::<u8>(4096, PowerOfTwo::new(512).unwrap()).unwrap();
        assert_eq!(data.len(), 4096);
        assert_eq!(data.as_ptr() as usize % 512, 0);
    }

    #[test]
    fn aligned_slice_zero_length() {
        // Zero-length aligned slices should succeed: `Poly::from_iter`
        // special-cases empty iterators and returns an empty slice.
        let data = aligned_slice::<f32>(0, PowerOfTwo::new(16).unwrap()).unwrap();
        assert!(data.is_empty());
        assert_eq!(data.len(), 0);
    }

    #[test]
    fn aligned_slice_default_initialized() {
        let data = aligned_slice::<f32>(100, PowerOfTwo::new(64).unwrap()).unwrap();
        for &val in data.iter() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn aligned_slice_deref_mut() {
        let mut data = aligned_slice::<f32>(4, PowerOfTwo::new(32).unwrap()).unwrap();
        data[0] = 1.0;
        data[1] = 2.0;
        assert_eq!(&data[..2], &[1.0, 2.0]);
    }
}
