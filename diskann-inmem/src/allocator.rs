/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Allocators for backing memory of in-mem indexes

use std::{alloc::Layout, fmt::Debug, ptr::NonNull};

/// Allocates zeroed backing memory.
///
/// # Safety
///
/// Return uniquely owned, zeroed memory in a single allocation satisfying `layout`.
/// Keep it valid until deallocation by this instance with the original layout.
/// Support concurrent calls and cross-thread access and deallocation.
pub unsafe trait BufferAllocator: Debug + Send + Sync {
    /// Allocate a nonzero layout, returning `None` if it cannot be allocated.
    fn allocate_zeroed(&self, layout: Layout) -> Option<NonNull<u8>>;

    /// Release an allocation.
    ///
    /// # Safety
    ///
    /// `ptr` must be a live allocation from this instance for `layout`, with no
    /// remaining accesses.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// Delegates to Rust's global allocator.
#[derive(Debug, Default)]
pub struct DefaultAllocator;

// SAFETY: Rust's global allocator satisfies the allocator contract.
unsafe impl BufferAllocator for DefaultAllocator {
    fn allocate_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }
        // SAFETY: The layout is nonzero.
        NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: Inherited from caller.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
    }
}

/// Requests transparent huge pages on Linux on a best-effort basis.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub struct HugePageAllocator {
    // This is native page size for a system, not the size of huge pages.
    page_size: usize,
}

#[cfg(target_os = "linux")]
impl HugePageAllocator {
    /// Discover the system's base page size.
    ///
    /// # Errors
    ///
    /// Returns an error if the system page size is unavailable or invalid.
    pub fn new() -> std::io::Result<Self> {
        // SAFETY: sysconf does not access caller-provided memory.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        let page_size = usize::try_from(page_size)
            .ok()
            .filter(|size| size.is_power_of_two())
            .ok_or_else(|| std::io::Error::other("unable to determine a valid system page size"))?;
        Ok(Self { page_size })
    }

    fn backing_layout(&self, layout: Layout) -> Option<Layout> {
        Layout::from_size_align(layout.size(), layout.align().max(self.page_size))
            .ok()
            .map(|layout| layout.pad_to_align())
    }
}

#[cfg(target_os = "linux")]
// SAFETY: The padded allocation is aligned and zeroed. Its layout is recomputed
// from the original layout and fixed page size on deallocation.
unsafe impl BufferAllocator for HugePageAllocator {
    fn allocate_zeroed(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() == 0 {
            return None;
        }
        let backing = self.backing_layout(layout)?;
        // SAFETY: `backing` is nonzero.
        let ptr = NonNull::new(unsafe { std::alloc::alloc(backing) })?;
        // SAFETY: The allocation is page-aligned and covers `backing.size()` bytes.
        let _ = unsafe { libc::madvise(ptr.as_ptr().cast(), backing.size(), libc::MADV_HUGEPAGE) };
        // SAFETY: We own `backing.size()` writable bytes.
        unsafe { ptr.as_ptr().write_bytes(0, backing.size()) };
        Some(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        #[expect(
            clippy::expect_used,
            reason = "a successfully allocated layout must have a valid backing layout"
        )]
        let backing = self
            .backing_layout(layout)
            .expect("original backing layout must be valid");
        // SAFETY: The caller provides the original allocation; `backing` is its layout.
        unsafe { DefaultAllocator.deallocate(ptr, backing) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn huge_page_backing_layout() {
        let allocator = HugePageAllocator::new().unwrap();
        for size in [1, allocator.page_size - 1, allocator.page_size + 1] {
            for align in [1, allocator.page_size, allocator.page_size * 2] {
                let requested = Layout::from_size_align(size, align).unwrap();
                let backing = allocator.backing_layout(requested).unwrap();
                assert!(backing.size() >= size);
                assert!(backing.size().is_multiple_of(allocator.page_size));
                assert!(backing.align() >= align.max(allocator.page_size));
            }
        }
        let oversized = Layout::from_size_align(isize::MAX as usize, 1).unwrap();
        assert!(allocator.backing_layout(oversized).is_none());
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn huge_page_allocator_rejects_empty_layout() {
        let allocator = HugePageAllocator::new().unwrap();
        assert!(allocator.allocate_zeroed(Layout::new::<()>()).is_none());
    }

    #[test]
    fn default_allocator_rejects_empty_layout() {
        assert!(
            DefaultAllocator
                .allocate_zeroed(Layout::new::<()>())
                .is_none()
        );
    }
}
