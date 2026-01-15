/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr::NonNull;

use thiserror::Error;

use super::{AllocatorCore, AllocatorError, GlobalAllocator};
use crate::num::PowerOfTwo;

/// An [`AllocatorCore`] that allocates memory aligned to at least a specified alignment.
///
/// This can be useful for large allocations that need a predictable base alignment.
#[derive(Debug, Clone, Copy)]
pub struct AlignedAllocator {
    /// This represents a power of 2.
    alignment: u8,
}

impl AlignedAllocator {
    /// Construct a new allocator that uses the given alignment.
    #[inline]
    pub const fn new(alignment: PowerOfTwo) -> Self {
        Self {
            // CAST: `trailing_zeros` returns as most 63 (because we've removed 0), so
            // the conversion is always lossless.
            alignment: alignment.raw().trailing_zeros() as u8,
        }
    }

    #[inline]
    pub const fn alignment(&self) -> usize {
        1usize << (self.alignment as usize)
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("alignment {0} must be a power of two")]
pub struct NotPowerOfTwo(usize);

// SAFETY: We are making the alignment potentially stricter before forwarding to the
// `GlobalAllocator`.
unsafe impl AllocatorCore for AlignedAllocator {
    #[inline]
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        // Bump up the alignment.
        let layout = layout
            .align_to(self.alignment())
            .map_err(|_| AllocatorError)?;
        GlobalAllocator.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<[u8]>, layout: std::alloc::Layout) {
        // Lint: The given `layout` **should** be the same as that passed to `allocate`,
        // which must have succeeded for the pointer to be valid in the first place.
        #[allow(clippy::expect_used)]
        let layout = layout
            .align_to(self.alignment())
            .expect("invalid layout provided");
        GlobalAllocator.deallocate(ptr, layout)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_allocator() {
        let powers_of_two = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ];
        let trials = 10;
        for power in powers_of_two {
            let alloc = AlignedAllocator::new(PowerOfTwo::new(power).unwrap());
            assert_eq!(alloc.alignment(), power);

            // Test allocation.
            struct Guard<'a> {
                ptr: NonNull<[u8]>,
                layout: std::alloc::Layout,
                allocator: &'a AlignedAllocator,
            }

            impl Drop for Guard<'_> {
                fn drop(&mut self) {
                    // SAFETY: We immediately pass allocated pointer to the guard, along
                    // with the allocator and layout.
                    unsafe { self.allocator.deallocate(self.ptr, self.layout) }
                }
            }

            for trial in 1..(trials + 1) {
                let layout = std::alloc::Layout::from_size_align(trial, power).unwrap();
                let ptr = alloc.allocate(layout).unwrap();

                // Ensure we deallocate if we panic.
                let _guard = Guard {
                    ptr,
                    layout,
                    allocator: &alloc,
                };

                assert_eq!(ptr.len(), trial);
                assert_eq!(
                    (ptr.cast::<u8>().as_ptr() as usize) % power,
                    0,
                    "ptr {:?} is not aligned to {}",
                    ptr,
                    power
                );
            }
        }
    }
}
