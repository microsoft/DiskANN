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
pub use poly::{poly, CompoundError, Poly, TrustedIter};
pub use traits::{Allocator, AllocatorCore, AllocatorError};

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
        self.allocator.deallocate(ptr, layout)
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
}
