/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Utilities for prefetching.

use crate::num::Bytes;

/// A validated [`Prefetch`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Checked<P>(P);

impl<P> Checked<P>
where
    P: Prefetch,
{
    /// Construct a new [`Checked`] containing `prefetcher`, validating the prefetcher against
    /// `len`.
    pub(crate) fn new(prefetcher: P, len: Bytes) -> Result<Self, InvalidPrefetch> {
        prefetcher.check(len)?;
        Ok(Self(prefetcher))
    }

    /// Prefetch the slice defined by `[ptr, ptr.add(len.value()))`.
    ///
    /// # Safety
    ///
    /// * The slice must point to valid memory within a single allocation. There are no
    ///   aliasing requirements.
    ///
    /// * `self` must be compatible with `len`, either through [`Self::new`] or [`Self::check`].
    pub(crate) unsafe fn prefetch(self, ptr: *const u8, len: Bytes) {
        debug_assert!(self.check(len).is_ok());

        // SAFETY: Inherited from caller.
        unsafe { self.0.prefetch(ptr, len) }
    }

    /// Check if `self` can prefetch slices of length `len`.
    pub(crate) fn check(self, len: Bytes) -> Result<(), InvalidPrefetch> {
        self.0.check(len)
    }

    #[cfg(test)]
    fn safe_prefetch(self, x: &[u8]) -> Result<(), InvalidPrefetch> {
        let bytes = Bytes::new(x.len());
        self.check(bytes)?;

        // SAFETY: We've checked the length, and slices satisfy the memory and lifetime
        // requirements.
        unsafe { self.prefetch(x.as_ptr(), bytes) };
        Ok(())
    }
}

/// Prefetch contiguous chunks of memory.
///
/// Prefetchers are created ahead of time and reused. This allows specialized prefetchers
/// (e.g. fully or partially unrolled) to be created.
///
/// # Safety
///
/// The function [`Self::check`] **must** be accurate. A successful return from
/// [`Self::check`] must imply that [`Self::prefetch`] on a valid slice of that length is safe.
pub(crate) unsafe trait Prefetch:
    std::fmt::Debug + Send + Sync + 'static + Copy
{
    /// Check that slices of length `len` are compatible with this prefetcher.
    ///
    /// If this function returns `Ok(())`, calling [`Self::prefetch`] with a valid slice of
    /// length `len` must be safe.
    fn check(self, len: Bytes) -> Result<(), InvalidPrefetch>;

    /// Prefetch the slice defined by `[ptr, ptr.add(len.value()))`.
    ///
    /// # Safety
    ///
    /// * The slice must point to valid memory within a single allocation. There are no
    ///   aliasing requirements.
    ///
    /// * [`Self::check`] must return `Ok(())` for `len`.
    unsafe fn prefetch(self, ptr: *const u8, len: Bytes);
}

/// A call to [`Prefetch::check`] failed.
#[derive(Debug)]
pub(crate) struct InvalidPrefetch(());

impl InvalidPrefetch {
    const fn new() -> Self {
        Self(())
    }
}

impl std::fmt::Display for InvalidPrefetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid prefetch")
    }
}

impl std::error::Error for InvalidPrefetch {}

diskann::convert_error!(InvalidPrefetch);

/// Prefetch data using a simple `for` loop.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Loop(());

impl Loop {
    /// Construct a new `Loop`.
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

// SAFETY: The `Loop` prefetcher is compatible with all lengths of slices as long as the
// provided slice is valid.
unsafe impl Prefetch for Loop {
    fn check(self, _len: Bytes) -> Result<(), InvalidPrefetch> {
        Ok(())
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8, len: Bytes) {
        // SAFETY: Inherited from caller.
        unsafe { prefetch(ptr, len.value()) }
    }
}

/// A prefetcher for a fixed number of bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Unrolled<const BYTES: usize>(());

impl<const BYTES: usize> Unrolled<BYTES> {
    /// Construct a new `Unrolled`.
    pub(crate) const fn new() -> Self {
        Self(())
    }
}

// SAFETY: This prefetcher is only valid for slices of length `BYTES`, which is correctly
// reported in the implementation of `check`.
unsafe impl<const BYTES: usize> Prefetch for Unrolled<BYTES> {
    fn check(self, bytes: Bytes) -> Result<(), InvalidPrefetch> {
        if bytes == Bytes::new(BYTES) {
            Ok(())
        } else {
            Err(InvalidPrefetch::new())
        }
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8, _len: Bytes) {
        debug_assert!(self.check(_len).is_ok());

        // SAFETY: Inherited from caller.
        unsafe { prefetch(ptr, BYTES) }
    }
}

//------------------------//
// Architecture Dependent //
//------------------------//

/// Prefetch `len` bytes beginning at `ptr`.
///
/// Prefetch locations are spaced one cache-line width apart. The final location is
/// prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn prefetch(ptr: *const u8, len: usize) {
    use std::arch::x86_64::*;

    // Fetch the final location (the one containing the tag) first.
    let stride = Bytes::CACHELINE.value();
    let ptr = ptr.cast::<i8>();
    let lines = len.div_ceil(stride);
    if lines == 0 {
        return;
    }

    // SAFETY: Inherited from caller.
    unsafe { _mm_prefetch(ptr.add(stride * (lines - 1)), _MM_HINT_T0) };
    for i in 0..(lines - 1) {
        // SAFETY: Inherited from caller.
        unsafe {
            _mm_prefetch(ptr.add(stride * i), _MM_HINT_T0);
        }
    }
}

/// Prefetch `len` bytes beginning at `ptr`.
///
/// Prefetch locations are spaced one cache-line width apart. The final location is
/// prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) unsafe fn prefetch(_ptr: *const u8, _len: usize) {}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    // The safety of this test is mainly dependent on running under Miri.
    #[test]
    fn test_loop() {
        let p = Loop::new();
        for i in 0..=1024 {
            let v = vec![0u8; i];

            let checked = Checked::new(p, Bytes::new(v.len())).unwrap();
            checked.safe_prefetch(&v).unwrap();
        }
    }

    fn unrolled_prefetch<const BYTES: usize>() {
        let p = Unrolled::<BYTES>::new();

        // Happy Path
        {
            let c = Checked::new(p, Bytes::new(BYTES)).unwrap();
            let v = vec![0u8; BYTES];
            c.safe_prefetch(&v).unwrap();
        }

        if let Some(under) = BYTES.checked_sub(1) {
            assert!(Checked::new(p, Bytes::new(under)).is_err());
            let v = vec![0u8; under];
            let c = Checked::new(p, Bytes::new(BYTES)).unwrap();
            assert!(c.safe_prefetch(&v).is_err());
        }

        if let Some(over) = BYTES.checked_add(1) {
            assert!(Checked::new(p, Bytes::new(over)).is_err());
            let v = vec![0u8; over];
            let c = Checked::new(p, Bytes::new(BYTES)).unwrap();
            assert!(c.safe_prefetch(&v).is_err());
        }
    }

    #[test]
    fn test_unrolled() {
        unrolled_prefetch::<0>();
        unrolled_prefetch::<63>();
        unrolled_prefetch::<64>();
        unrolled_prefetch::<65>();
        unrolled_prefetch::<127>();
        unrolled_prefetch::<128>();
        unrolled_prefetch::<129>();
    }
}
