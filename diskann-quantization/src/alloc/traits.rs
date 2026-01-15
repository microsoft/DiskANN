/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr::NonNull;

use thiserror::Error;

/// Indicate that an allocation error has occurred.
///
/// This type is limited in what it can contain because additional context
/// inevitably requires more memory allocation, which is what we're trying to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("unknown allocation error")]
pub struct AllocatorError;

/// A dynamic memory allocator for use with [`crate::alloc::Poly`].
///
/// # Safety
///
/// Implementations must ensure that if `allocate` succeeds, the returned slice has a length
/// of at least `layout.size()` bytes and an alignment of at least `layout.align()`. If this
/// cannot be satisfied, then an error **must** be returned.
pub unsafe trait AllocatorCore {
    /// Allocate space for at least `layout.size()` bytes aligned to at least
    /// `layout.align()`. Returns an error if the requested size or alignment is not
    /// possible with this allocator.
    fn allocate(&self, layout: std::alloc::Layout) -> Result<NonNull<[u8]>, AllocatorError>;

    /// Deallocation companion to `allocate`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that
    ///
    /// 1. `ptr` is "currently allocated" from the allocator.
    ///    See: <https://doc.rust-lang.org/std/alloc/trait.Allocator.html#currently-allocated-memory>
    /// 2. `ptr` has the same base pointer as the slice-pointer returned from [`Self::allocate`].
    /// 3. `layout` is the same layout that was passed to [`Self::allocate`] for this pointer.
    unsafe fn deallocate(&self, ptr: NonNull<[u8]>, layout: std::alloc::Layout);
}

/// A dynamic memory allocator for use with [`crate::alloc::Poly`].
///
/// Users should implement [`AllocatorCore`] instead and use the blanket implementation for
/// the full cloneable allocator.
pub trait Allocator: AllocatorCore + Clone {}

impl<T> Allocator for T where T: AllocatorCore + Clone {}
