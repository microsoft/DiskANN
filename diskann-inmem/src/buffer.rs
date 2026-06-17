/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{alloc::Layout, marker::PhantomData, ptr::NonNull};

use crate::num::{Align, Bytes};

/// An unsynchronized row-store for raw data.
///
/// The backing data is stored as a raw pointers and interacted with via [`RawSlice`], which
/// is also raw pointer based. Careful use of this struct enables safe use of
/// [`RawSlice::as_slice`], [`RawSlice::as_mut_slice`], and other accesses from multiple
/// threads without undefined behavior.
///
/// Note that `Buffer` is unconditionally `Send` and `Sync`.
#[derive(Debug)]
pub(crate) struct Buffer {
    ptr: NonNull<u8>,
    stride: Bytes,
    entries: usize,
    layout: Layout,
}

impl Buffer {
    /// Construct a new [`Buffer`] capable of holding `entries` with each entry occupying
    /// exactly `bytes_per_entry`. Subsequent entries are separated by exactly
    /// `bytes_per_entry` bytes. The base point will be aligned to at least `align`.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of bytes `bytes_per_entry * entries` rounded up to
    /// the next multiple of `align` exceeds `isize::MAX`.
    pub(crate) fn new(entries: usize, bytes_per_entry: Bytes, align: Align) -> Result<Self, BufferError> {
        // If we overflow `usize::MAX`, we will definitely overflow `isize::MAX`.
        let bytes = bytes_per_entry.checked_mul(entries).ok_or(BufferError)?;

        // Since `align` is constrained to be a power of two, the only way this fails is
        // if we overflow `isize::MAX`.
        let layout = std::alloc::Layout::from_size_align(bytes.value(), align.value())
            .map_err(|_: std::alloc::LayoutError| BufferError)?;

        let ptr = if layout.size() == 0 {
            std::ptr::dangling_mut()
        } else {
            // SAFETY: `layout.size()` is non-zero.
            unsafe { std::alloc::alloc_zeroed(layout) }
        };

        let ptr = match NonNull::new(ptr) {
            Some(ptr) => ptr,
            None => std::alloc::handle_alloc_error(layout),
        };

        Ok(Self {
            ptr,
            stride: bytes_per_entry,
            entries,
            layout,
        })
    }

    /// Return the number of entries in this [`Buffer`].
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.entries
    }

    /// Return the number of bytes for each entry.
    #[inline]
    pub(crate) fn stride(&self) -> Bytes {
        self.stride
    }

    /// Return the `i`th entry without bounds checking.
    ///
    /// The returned [`RawSlice`] is guaranteed to have a length of [`Self::stride`] and
    /// begin at `self.as_ptr().add(self.stride().value() * i)`.
    ///
    /// # Safety
    ///
    /// `i` must be less than [`len`](Self::len).
    #[inline]
    pub(crate) unsafe fn get_unchecked(&self, i: usize) -> RawSlice<'_> {
        debug_assert!(i < self.entries);
        let ptr = unsafe { self.ptr.add(self.stride().value() * i) };
        RawSlice {
            ptr,
            len: self.stride,
            _lifetime: PhantomData,
        }
    }

    #[cfg(test)]
    pub(crate) fn get(&self, i: usize) -> Option<RawSlice<'_>> {
        if i >= self.entries {
            None
        } else {
            // SAFETY: We have validated that `i < self.entries`. This does two things:
            //
            // 1. Ensure that the multiplication will not overflow.
            // 2. Ensures that the computed offset is within the original allocation.
            Some(unsafe { self.get_unchecked(i) })
        }
    }

    #[cfg(test)]
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // If the layout size is zero, there's nothing to do because we hold a dangling pointer.
        if self.layout.size() != 0 {
            // SAFETY: This is the same pointer and allocation that was previously returned
            // from a successful `alloc_zeroed`.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), self.layout) }
        }
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub(crate) struct BufferError;

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("requested allocation exceeds `isize::MAX`")
    }
}

impl std::error::Error for BufferError {}

// SAFETY: We're safe to pass around the `Buffer`. It's just use of the returned `RawSlice`
// that needs to be arbitrated.
unsafe impl Send for Buffer {}

// SAFETY: We're safe to pass around the `Buffer`. It's just use of the returned `RawSlice`
// that needs to be arbitrated.
unsafe impl Sync for Buffer {}

/// A raw entry in [`Buffer`].
///
/// The memory in the range `[RawSlice::as_ptr(), RawSlice::as_ptr().add(slice.len()))` is
/// guaranteed to be within a single alive allocation.
///
/// This has borrowing semantics of a raw pointer.
#[derive(Debug)]
pub(crate) struct RawSlice<'a> {
    ptr: NonNull<u8>,
    len: Bytes,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> RawSlice<'a> {
    /// Create a new [`RawSlice`].
    ///
    /// # Safety
    ///
    /// The memory `[ptr, ptr.add(len.value()))` must be part of a single allocation for
    /// the duration of the lifetime `'a`.
    ///
    /// However, this has the semantics of a pointer: multiple threads can hold a [`RawSlice`]
    /// to the same piece of memory without undefined behavior.
    unsafe fn new(ptr: NonNull<u8>, len: Bytes) -> Self {
        Self {
            ptr,
            len,
            _lifetime: PhantomData,
        }
    }

    /// Create a new slice to the first `n.min(self.len())` bytes of `self`.
    #[inline]
    pub(crate) fn truncate(&self, n: Bytes) -> RawSlice<'a> {
        // SAFETY: The `min` operation ensures we provide an argument <= `self.len()`.
        unsafe { self.truncate_unchecked(self.len.min(n)) }
    }

    /// Shorten the slice to the `n`.
    ///
    /// # Safety
    ///
    /// `n` must be less than or equal to `self.len()`.
    #[inline]
    pub(crate) unsafe fn truncate_unchecked(&self, n: Bytes) -> RawSlice<'a> {
        debug_assert!(n <= self.len);

        // SAFETY: Inherited from the caller.
        unsafe { Self::new(self.ptr, n) }
    }

    /// Split `self` into two as `([ptr, ptr.add(m)), [ptr.add(m), ptr.add(self.len())))`
    /// where `m = n.min(self.len())`.
    #[inline]
    pub(crate) fn split(&self, n: Bytes) -> (RawSlice<'a>, RawSlice<'a>) {
        // SAFETY: The argument is <= `self.len()`.
        unsafe { self.split_unchecked(self.len.min(n)) }
    }

    /// Split `self` into two as `([ptr, ptr.add(n)), [ptr.add(n), ptr.add(self.len())))`
    ///
    /// # Safety
    ///
    /// `n` must be less than or equal to `self.len()`.
    #[inline]
    pub(crate) unsafe fn split_unchecked(&self, n: Bytes) -> (RawSlice<'a>, RawSlice<'a>) {
        debug_assert!(n <= self.len);
        unsafe {
            (
                Self::new(self.ptr, n),
                Self::new(self.ptr.add(n.value()), self.len.unchecked_sub(n)),
            )
        }
    }

    /// Return the length of the slice in bytes.
    #[inline]
    pub(crate) fn len(&self) -> Bytes {
        self.len
    }

    /// Return the base [`NonNull`] pointer of the slice.
    pub(crate) fn as_non_null(&self) -> NonNull<u8> {
        self.ptr
    }

    /// Return the base pointer of the slice as `*const u8`.
    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr().cast_const()
    }

    /// Return the base pointer of the slice as `*mut u8`.
    ///
    /// This returns a mutable pointer regardless of the receiver's mutability, matching
    /// the raw-pointer semantics of [`RawSlice`].
    pub(crate) fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Materialize the raw slice as a true shared slice.
    ///
    /// # Safety
    ///
    /// Correct adherence to the API of [`RawSlice`] will ensure that the memory behind the
    /// materialized slice resides within a single allocation.
    ///
    /// However, it is the responsibility of the caller to ensure that materializing this
    /// slice does not violate Rust's borrowing rules.
    #[inline]
    pub(crate) unsafe fn as_slice(&self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len.value()) }
    }

    /// Materialize the raw slice as a true mutable slice.
    ///
    /// # Safety
    ///
    /// Correct adherence to the API of [`RawSlice`] will ensure that the memory behind the
    /// materialized slice resides within a single allocation.
    ///
    /// However, it is the responsibility of the caller to ensure that materializing this
    /// slice does not violate Rust's borrowing rules.
    #[inline]
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len.value()) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{sync::Barrier, thread};

    #[derive(Debug)]
    struct Ctx {
        entries: usize,
        bytes_per_entry: Bytes,
        align: Align,
    }

    impl std::fmt::Display for Ctx {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "entries = {}, bytes_per_entry = {}, align = {}",
                self.entries, self.bytes_per_entry, self.align
            )
        }
    }

    fn test_buffer_inner(entries: usize, bytes_per_entry: Bytes, align: Align) {
        let ctx = Ctx {
            entries,
            bytes_per_entry,
            align,
        };
        let mut buffer = Buffer::new(entries, bytes_per_entry, align).unwrap();

        // Initial Checks
        assert_eq!(buffer.len(), entries, "{}", ctx);
        assert_eq!(buffer.stride(), bytes_per_entry, "{}", ctx);

        if entries != 0 && !bytes_per_entry.is_zero() {
            let addr = buffer.as_ptr() as usize;
            assert!(
                addr.is_multiple_of(align.value()),
                "pointer address {:#x} must be a multiple of the requested alignment: {}",
                addr,
                ctx,
            );
        }

        // Verify zero initialization
        assert_is_zeroed(&mut buffer, &ctx);

        // Check Slice Methods
        check_slice_methods(&mut buffer, &ctx);

        // Check that concurrent mutation is allowed.
        //
        // This is mainly a Miri check.
        zero(&mut buffer);
        check_threaded(&mut buffer, &ctx);
    }

    fn zero(buffer: &mut Buffer) {
        // SAFETY NOTE: Exclusive reference to `buffer` guarantees no concurrent mutation.
        for i in 0..buffer.len() {
            let mut raw_slice = buffer.get(i).unwrap();
            assert_eq!(raw_slice.len(), buffer.stride());

            let slice = unsafe { raw_slice.as_mut_slice() };
            assert_eq!(slice.len(), buffer.stride().value());
            slice.fill(0);
        }
    }

    fn assert_is_zeroed(buffer: &mut Buffer, ctx: &Ctx) {
        // SAFETY NOTE: Exclusive reference to `buffer` guarantees no concurrent mutation.
        // All `unsafe` calls below rely on this guarantee.

        for i in 0..buffer.len() {
            let raw_slice = buffer.get(i).unwrap();
            assert_eq!(raw_slice.len(), buffer.stride());

            assert_eq!(raw_slice.as_non_null().as_ptr(), raw_slice.as_mut_ptr());
            assert_eq!(
                raw_slice.as_non_null().as_ptr().cast_const(),
                raw_slice.as_ptr()
            );

            assert_eq!(
                raw_slice.as_ptr(),
                buffer
                    .as_ptr()
                    .wrapping_add(buffer.stride().checked_mul(i).unwrap().value()),
                "stride mismatch - {}",
                ctx
            );

            let slice = unsafe { raw_slice.as_slice() };
            assert_eq!(slice.len(), buffer.stride().value());
            assert!(slice.iter().all(|&i| i == 0), "{}", ctx);
        }

        // Verify that bounds-checking works.
        assert!(buffer.get(buffer.len()).is_none(), "{}", ctx);
    }

    fn check_slice_methods(buffer: &mut Buffer, ctx: &Ctx) {
        // SAFETY NOTE: We take `buffer` by exclusive reference to guarantee that there
        // is no possibility of concurrent mutation outside this method. All `unsafe` calls
        // below rely on this guarantee unless otherwise noted.

        if buffer.len() == 0 {
            return;
        }

        let mut raw = buffer.get(0).unwrap();
        let base: u8 = 5;
        let base_usize: usize = base.into();

        // truncate //

        iota(unsafe { raw.as_mut_slice() }, base);
        for i in 0..raw.len().value() + base_usize {
            let expected = i.min(raw.len().value());

            let truncated = raw.truncate(Bytes::new(i));
            assert_eq!(truncated.len().value(), expected, "{}", ctx);
            assert!(is_iota(unsafe { truncated.as_slice() }, base), "{}", ctx);
        }

        // split //

        for i in 0..raw.len().value() + base_usize {
            let first = i.min(raw.len().value());
            let last = raw.len().value() - first;

            let (mut prefix, mut suffix) = raw.split(Bytes::new(i));

            assert_eq!(prefix.len().value(), first, "{}", ctx);
            assert_eq!(suffix.len().value(), last, "{}", ctx);

            assert!(is_iota(unsafe { prefix.as_slice() }, base), "{}", ctx);
            assert!(
                is_iota(unsafe { suffix.as_slice() }, base.wrapping_add(i as u8)),
                "{}",
                ctx
            );

            // Verify it's okay to mutate two disjoint slices concurrently.
            //
            // SAFETY: `prefix` and `suffix` are non-overlapping sub-ranges of the same
            // entry, so materializing both as mutable is sound.
            {
                let prefix = unsafe { prefix.as_mut_slice() };
                let suffix = unsafe { suffix.as_mut_slice() };
                suffix.fill(0);
                prefix.fill(0);
            }

            assert!(unsafe { raw.as_slice() }.iter().all(|i| *i == 0), "{}", ctx);
            iota(unsafe { raw.as_mut_slice() }, base);
        }
    }

    fn check_threaded(buffer: &mut Buffer, ctx: &Ctx) {
        let spawns = buffer.len();

        // The goal here is to ensure that threads hold concurrent mutable references to
        // disjoint entries within the `Buffer` and that when the mutate them concurrently,
        // we get a coherent result.
        let pre = &Barrier::new(spawns);
        let post = &Barrier::new(spawns);
        {
            let borrowed: &Buffer = buffer;
            thread::scope(|s| {
                for i in 0..spawns {
                    s.spawn(move || {
                        // SAFETY: The top level method has an exclusive reference to the buffer.
                        //
                        // This loop by construction accesses disjoint offsets. This is sufficient
                        // to guarantee exclusivity for this thread.
                        let slice = unsafe { borrowed.get(i).unwrap().as_mut_slice() };
                        pre.wait();
                        iota(slice, i as u8);
                        post.wait();
                    });
                }
            });
        }

        for i in 0..spawns {
            let slice = unsafe { buffer.get(i).unwrap().as_slice() };
            assert!(is_iota(slice, i as u8), "i = {} -- {}", i, ctx);
        }
    }

    fn iota(x: &mut [u8], base: u8) {
        for (i, v) in x.iter_mut().enumerate() {
            *v = base.wrapping_add(i as u8);
        }
    }

    #[must_use]
    fn is_iota(x: &[u8], base: u8) -> bool {
        for (i, v) in x.iter().enumerate() {
            if *v != base.wrapping_add(i as u8) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_buffer() {
        let entries = [0, 1, 2, 5];
        let bytes_per_entry = [0, 1, 2, 5, 10].map(Bytes::new);
        let align = [Align::_1, Align::_64];

        for entries in entries {
            for bytes_per_entry in bytes_per_entry {
                for align in align {
                    test_buffer_inner(entries, bytes_per_entry, align);
                }
            }
        }
    }

    #[test]
    fn test_buffer_overflow_mul() {
        // entries * bytes_per_entry overflows usize.
        let result = Buffer::new(usize::MAX, Bytes::new(2), Align::_1);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_overflow_layout() {
        // Total size exceeds isize::MAX (Layout rejects this).
        let result = Buffer::new(isize::MAX as usize, Bytes::new(2), Align::_1);
        assert!(result.is_err());
    }
}
