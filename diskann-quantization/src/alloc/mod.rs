/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{alloc::Layout, ptr::NonNull};

mod aligned;
mod bump;
mod poly;
mod traits;

pub use aligned::{AlignedAllocator, NotPowerOfTwo};
pub use bump::BumpAllocator;
pub use poly::{CompoundError, Poly, TrustedIter, poly};
pub use traits::{Allocator, AllocatorCore, AllocatorError};

use crate::num::PowerOfTwo;

/// An aligned, heap-allocated slice.
///
/// This is a [`Poly`] backed by an [`AlignedAllocator`], providing a
/// cache-aligned buffer of elements.
pub type AlignedSlice<T> = Poly<[T], AlignedAllocator>;

/// Creates a new [`AlignedSlice`] with the given capacity and alignment.
/// The allocated memory is set to `T::default()`.
///
/// # Errors
///
/// Returns an [`AllocatorError`] if the layout is invalid or allocation fails.
pub fn aligned_slice<T: Default>(
    capacity: usize,
    alignment: PowerOfTwo,
) -> Result<AlignedSlice<T>, AllocatorError> {
    let allocator = AlignedAllocator::new(alignment);
    Poly::from_iter((0..capacity).map(|_| T::default()), allocator)
}

/// A handle to Rust's global allocator. This type does not support allocations of size 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalAllocator;

// SAFETY: This is a simple wrapper around Rust's built-in allocation and deallocation
// methods, augmented slightly to handle zero sized layouts by returning a dangling pointer.
//
// The returned slice from `allocate` always has the exact size and alignment as `layout`.
unsafe impl AllocatorCore for GlobalAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        if layout.size() == 0 {
            return Err(AllocatorError);
        }

        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = std::ptr::slice_from_raw_parts_mut(ptr, layout.size());
        NonNull::new(ptr).ok_or(AllocatorError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<[u8]>, layout: Layout) {
        // SAFETY: The caller has the responsibility to ensure that `ptr` and `layout`
        // came from a previous allocation.
        unsafe { std::alloc::dealloc(ptr.as_ptr().cast::<u8>(), layout) }
    }
}

////////////
// Scoped //
////////////

trait DebugAllocator: AllocatorCore + std::fmt::Debug {}
impl<T> DebugAllocator for T where T: AllocatorCore + std::fmt::Debug {}

/// A dynamic wrapper around an `AllocatorCore` that provides the guarantee that all
/// allocated object are tied to a given scope.
///
/// Additionally, this can allow the use of an allocator that is not `Clone` in contexts
/// where a clonable allocator is needed (provided the scoping limitations are acceptable).
#[derive(Debug, Clone, Copy)]
pub struct ScopedAllocator<'a> {
    allocator: &'a dyn DebugAllocator,
}

impl<'a> ScopedAllocator<'a> {
    /// Construct a new `ScopedAllocator` around the provided `allocator`.
    pub const fn new<T>(allocator: &'a T) -> Self
    where
        T: AllocatorCore + std::fmt::Debug,
    {
        Self { allocator }
    }
}

impl ScopedAllocator<'static> {
    /// A convenience method for construcing a `ScopedAllocator` around the [`GlobalAllocator`]
    /// for cases where a more specialized allocator is not needed.
    pub const fn global() -> Self {
        Self {
            allocator: &GlobalAllocator,
        }
    }
}

// SAFETY: This allocator simply delegates to the underlying allocator.
unsafe impl AllocatorCore for ScopedAllocator<'_> {
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError> {
        self.allocator.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<[u8]>, layout: std::alloc::Layout) {
        // SAFETY: Inherited from caller.
        unsafe { self.allocator.deallocate(ptr, layout) }
    }
}

///////////////
// Try Clone //
///////////////

/// A trait like [`Clone`] that allows graceful allocation failure.
///
/// # NOTE
///
/// Keep this `pub(crate)` for now because we do not want general users of the crate
/// relying on the current implementations for [`Poly`]. In particular, the base case should
/// be `Poly<T> where T: TryClone` instead of `Poly<T> where T: Clone`.
pub(crate) trait TryClone: Sized {
    /// Returns a duplicate of the value.
    fn try_clone(&self) -> Result<Self, AllocatorError>;
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn test_alloc<T>() {
        let alloc = GlobalAllocator;

        let layout = Layout::new::<T>();
        let ptr = alloc.allocate(layout).unwrap();

        assert_eq!(ptr.len(), layout.size());
        assert_eq!(ptr.len(), std::mem::size_of::<T>());
        assert_eq!((ptr.as_ptr().cast::<u8>() as usize) % layout.align(), 0);

        // SAFETY: `ptr` was obtained from this allocator with the specified `layout`.
        unsafe { alloc.deallocate(ptr, layout) };
    }

    #[test]
    fn test_global_allocator() {
        assert!(GlobalAllocator.allocate(Layout::new::<()>()).is_err());

        test_alloc::<(u8,)>();
        test_alloc::<(u8, u8)>();
        test_alloc::<(u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8, u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8, u8, u8, u8, u8)>();
        test_alloc::<(u8, u8, u8, u8, u8, u8, u8, u8, u8)>();

        test_alloc::<(u16,)>();
        test_alloc::<(u16, u16)>();
        test_alloc::<(u16, u16, u16)>();
        test_alloc::<(u16, u16, u16, u16)>();
        test_alloc::<(u16, u16, u16, u16, u16)>();

        test_alloc::<(u32,)>();
        test_alloc::<(u32, u32)>();
        test_alloc::<(u32, u32, u32)>();

        test_alloc::<String>();
    }

    #[test]
    fn aligned_slice_f32_alignment_32() {
        let alignment = PowerOfTwo::new(32).unwrap();
        let size = 1000;
        let data = aligned_slice::<f32>(size, alignment).unwrap();
        assert_eq!(data.len(), size);
        assert_eq!(data.as_ptr() as usize % 32, 0);
        for val in data.iter() {
            assert_eq!(*val, f32::default());
        }
    }

    #[test]
    fn aligned_slice_u8_alignment_256() {
        let alignment = PowerOfTwo::new(256).unwrap();
        let size = 100;
        let data = aligned_slice::<u8>(size, alignment).unwrap();
        assert_eq!(data.len(), size);
        assert_eq!(data.as_ptr() as usize % 256, 0);
        for val in data.iter() {
            assert_eq!(*val, u8::default());
        }
    }

    #[test]
    fn aligned_slice_zero_length() {
        let alignment = PowerOfTwo::new(16).unwrap();
        let x = aligned_slice::<f32>(0, alignment).unwrap();
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn aligned_slice_chunks_mut() {
        let alignment = PowerOfTwo::new(32).unwrap();
        let size = 10;
        let slice_len = 2;
        let mut data = aligned_slice::<f32>(size, alignment).unwrap();
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
