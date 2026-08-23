/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::num::Bytes;

pub(crate) unsafe trait Prefetch:
    std::fmt::Debug + Send + Sync + 'static + Copy
{
    /// Check that slices of length `bytes` are compatible with this prefetcher.
    fn check(self, bytes: Bytes);
    unsafe fn prefetch(self, ptr: *const u8);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NoPrefetch;

unsafe impl Prefetch for NoPrefetch {
    fn check(self, bytes: Bytes) {}

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8) {}
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Loop(Bytes);

impl Loop {
    pub(crate) const fn new(bytes: Bytes) -> Self {
        Self(bytes)
    }
}

unsafe impl Prefetch for Loop {
    fn check(self, bytes: Bytes) {
        assert!(bytes == self.0);
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8) {
        unsafe { prefetch(ptr, self.0.value()) }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Unrolled<const BYTES: usize>;

impl<const BYTES: usize> Unrolled<BYTES> {
    pub(crate) const fn new() -> Self {
        Self
    }
}

unsafe impl<const BYTES: usize> Prefetch for Unrolled<BYTES> {
    fn check(self, bytes: Bytes) {
        assert_eq!(bytes, Bytes::new(BYTES));
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8) {
        unsafe { prefetch(ptr, BYTES) }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Binned<const LINES: usize>;

impl<const LINES: usize> Binned<LINES> {
    pub(crate) const fn new() -> Self {
        Self
    }
}

unsafe impl<const LINES: usize> Prefetch for Binned<LINES> {
    fn check(self, bytes: Bytes) {
        let lower = Bytes::CACHELINE
            .checked_mul(LINES.saturating_sub(1))
            .unwrap();
        let upper = Bytes::CACHELINE.checked_mul(LINES).unwrap();

        assert!(bytes > lower);
        assert!(bytes <= upper);
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8) {
        unsafe { prefetch(ptr, Bytes::CACHELINE.value() * LINES) }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BinnedPlus<const LINES: usize>(Bytes);

impl<const LINES: usize> BinnedPlus<LINES> {
    pub(crate) const fn new(bytes: Bytes) -> Self {
        assert!(bytes.value() > Bytes::CACHELINE.value() * LINES);
        Self(bytes)
    }
}

unsafe impl<const LINES: usize> Prefetch for BinnedPlus<LINES> {
    fn check(self, bytes: Bytes) {
        assert_eq!(bytes, self.0)
    }

    #[inline(always)]
    unsafe fn prefetch(self, ptr: *const u8) {
        use std::arch::x86_64::*;

        let ptr = ptr.cast::<i8>();

        unsafe { _mm_prefetch(ptr.add(self.0.value()), _MM_HINT_T0) };

        let stride = Bytes::CACHELINE.value();
        for i in 0..LINES {
            unsafe { _mm_prefetch(ptr.add(stride * i), _MM_HINT_T0) };
        }
    }
}

// #[derive(Debug, Clone, Copy)]
// pub(crate) struct JumpTable {
//     bytes: Bytes,
//     back: usize,
//     last: usize,
// }
//
// impl JumpTable {
//     pub(crate) fn new(bytes: Bytes) -> Self {
//         let stride = Bytes::CACHELINE.value();
//         let lines = bytes.value().div_ceil(stride);
//
//         let back = 7 * lines.min(8);
//         let last = if lines > 8 {
//             stride * (lines - 1)
//         } else {
//             0
//         };
//
//         Self {
//             bytes,
//             back,
//             last,
//         }
//     }
// }
//
// unsafe impl Prefetch for JumpTable {
//     fn bytes(self) -> Bytes {
//         self.bytes
//     }
//
//     #[inline(always)]
//     unsafe fn prefetch(self, ptr: *const u8) {
//         unsafe { prefetch_up_to_8(ptr, self.back, self.last) }
//     }
// }

/// Prefetch `len` bytes beginning at `ptr`.
///
/// The last cache line prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn prefetch(ptr: *const u8, len: usize) {
    use std::arch::x86_64::*;

    // Fetch the last cache line (the one with the tag) first.
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub unsafe fn prefetch_up_to_8(ptr: *const u8, back: usize, last: usize) {
    use std::arch::x86_64::*;

    // const STRIDE: usize = Bytes::CACHELINE.value();
    // const PREFETCH_INSTRUCTION_BYTES: usize = 7;

    if last != 0 {
        unsafe {
            _mm_prefetch(ptr.cast::<i8>().add(last), _MM_HINT_T0);
        }
    }

    let ptr = ptr.wrapping_sub(128);
    unsafe {
        std::arch::asm! {
            // Obtain the address of the label - the base of our prefetch table.
            "lea {tmp}, [rip + 3f]",
            "sub {tmp}, {back}",
            "notrack jmp {tmp}",
            "2:",
            "prefetcht0 byte ptr [{base} + 576]",
            "prefetcht0 byte ptr [{base} + 512]",
            "prefetcht0 byte ptr [{base} + 448]",
            "prefetcht0 byte ptr [{base} + 384]",
            "prefetcht0 byte ptr [{base} + 320]",
            "prefetcht0 byte ptr [{base} + 256]",
            "prefetcht0 byte ptr [{base} + 192]",
            "prefetcht0 byte ptr [{base} + 128]",
            "3:",
            back = in(reg) back,
            base = in(reg_abcd) ptr,
            tmp = out(reg) _,
            options(readonly, nostack, preserves_flags),
        }
    }
}

/// Prefetch `len` bytes beginning at `ptr`.
///
/// The last cache line prefetched first, followed by the rest in ascending order.
///
/// # Safety
///
/// The memory range `[ptr, ptr.add(len))` must be valid.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) unsafe fn prefetch(_ptr: *const u8, _len: usize) {}

///////////
// Tests //
///////////

// #
// [cfg(test)]
// mod test {
//     use super::*;
//
//     #[test]
//     fn test_prefetch_up_to_8() {
//         let v = vec![0u8; 600];
//         for lines in 0..10 {
//             unsafe { prefetch_up_to_8(v.as_ptr(), lines) };
//         }
//     }
// }
