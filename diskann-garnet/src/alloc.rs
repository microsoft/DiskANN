/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::alloc::{AllocatorCore, AllocatorError, GlobalAllocator};
use std::ptr::NonNull;

/// Custom allocator that over-aligns to 8 bytes. This is needed since Garnet will hand us byte slices for f32 data
/// that may be unaligned, so we need an allocator to make owned, aligned byte containers.
#[derive(Debug, Clone, Copy)]
pub struct AlignToEight;

unsafe impl AllocatorCore for AlignToEight {
    #[inline]
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        // Bump up the alignment.
        let layout = layout.align_to(8).map_err(|_| AllocatorError)?;
        GlobalAllocator.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<[u8]>, layout: std::alloc::Layout) {
        // Lint: The given `layout` **should** be the same as that passed to `allocate`,
        // which must have succeeded for the pointer to be valid in the first place.
        #[allow(clippy::expect_used)]
        let layout = layout.align_to(8).expect("invalid layout provided");
        unsafe { GlobalAllocator.deallocate(ptr, layout) }
    }
}

#[cfg(test)]
mod test {
    use crate::alloc::AlignToEight;
    use diskann_quantization::alloc::Poly;

    #[test]
    fn test_align_8() {
        let poly = Poly::broadcast(0u8, 128, AlignToEight).unwrap();
        assert!((poly.as_ptr() as usize).is_multiple_of(8));
    }
}
