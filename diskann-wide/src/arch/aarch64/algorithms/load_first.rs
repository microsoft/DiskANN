/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::arch::aarch64::*;

use crate::arch::aarch64::Neon;

////////////////
// Load First //
////////////////

//-------------//
// 64-bit wide //
//-------------//

/// Load the first `first` elements from `ptr` into a `uint8x8_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn u8x8(_: Neon, ptr: *const u8, first: usize) -> uint8x8_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vcreate_u8(load_first_of_8_bytes(ptr, first)) }
}

/// Load the first `first` elements from `ptr` into an `int8x8_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn i8x8(_: Neon, ptr: *const i8, first: usize) -> int8x8_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpret_s8_u8(vcreate_u8(load_first_of_8_bytes(ptr.cast::<u8>(), first))) }
}

/// Load the first `first` elements from `ptr` into a `float32x2_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn f32x2(
    arch: Neon,
    ptr: *const f32,
    first: usize,
) -> float32x2_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpret_f32_u32(load_first_32x2(arch, ptr.cast::<u32>(), first)) }
}

//--------------//
// 128-bit wide //
//--------------//

/// Load the first `first` elements from `ptr` into a `uint8x16_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn u8x16(
    arch: Neon,
    ptr: *const u8,
    first: usize,
) -> uint8x16_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { load_first_of_16_bytes(arch, ptr, first) }
}

/// Load the first `first` elements from `ptr` into an `int8x16_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn i8x16(
    arch: Neon,
    ptr: *const i8,
    first: usize,
) -> int8x16_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_s8_u8(u8x16(arch, ptr.cast::<u8>(), first)) }
}

/// Load the first `first` elements from `ptr` into a `uint16x8_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn u16x8(
    arch: Neon,
    ptr: *const u16,
    first: usize,
) -> uint16x8_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_u16_u8(load_first_of_16_bytes(arch, ptr.cast::<u8>(), 2 * first)) }
}

/// Load the first `first` elements from `ptr` into an `int16x8_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn i16x8(
    arch: Neon,
    ptr: *const i16,
    first: usize,
) -> int16x8_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_s16_u16(u16x8(arch, ptr.cast::<u16>(), first)) }
}

/// Load the first `first` elements from `ptr` into a `uint32x4_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn u32x4(
    arch: Neon,
    ptr: *const u32,
    first: usize,
) -> uint32x4_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { load_first_32x4(arch, ptr, first) }
}

/// Load the first `first` elements from `ptr` into an `int32x4_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn i32x4(
    arch: Neon,
    ptr: *const i32,
    first: usize,
) -> int32x4_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_s32_u32(u32x4(arch, ptr.cast::<u32>(), first)) }
}

/// Load the first `first` elements from `ptr` into a `float32x4_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn f32x4(
    arch: Neon,
    ptr: *const f32,
    first: usize,
) -> float32x4_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_f32_u32(u32x4(arch, ptr.cast::<u32>(), first)) }
}

/// Load the first `first` elements from `ptr` into a `uint64x2_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn u64x2(
    arch: Neon,
    ptr: *const u64,
    first: usize,
) -> uint64x2_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { load_first_64x2(arch, ptr, first) }
}

/// Load the first `first` elements from `ptr` into an `int64x2_t` register.
///
/// # Safety
///
/// The caller must ensure `[ptr, ptr + first)` is readable. The presence of `Neon`
/// enables the use of "neon" intrinsics.
#[inline(always)]
pub(in crate::arch::aarch64) unsafe fn i64x2(
    arch: Neon,
    ptr: *const i64,
    first: usize,
) -> int64x2_t {
    // SAFETY: Pointer access inherited from caller. `Neon` enables "neon" intrinsics.
    unsafe { vreinterpretq_s64_u64(u64x2(arch, ptr.cast::<u64>(), first)) }
}

////////////////////
// Implementation //
////////////////////

/// Efficiently load the first `8 < bytes < 16` bytes from `ptr` without accessing memory
/// outside of `[ptr, ptr + bytes)`.
///
/// Uses two overlapping 8-byte loads combined with `TBL` to shift the high portion into
/// position, mirroring the x86 `PSHUFB` technique.
///
/// # Safety
///
/// * `bytes` must be in the range `(8, 16)`.
/// * The memory in `[ptr, ptr + bytes)` must be readable and valid.
#[inline(always)]
unsafe fn load_8_to_16_bytes(arch: Neon, ptr: *const u8, bytes: usize) -> uint8x16_t {
    debug_assert!(bytes > 8 && bytes < 16);

    // Two overlapping 8-byte loads: [ptr, ptr+8) and [ptr+bytes-8, ptr+bytes).
    //
    // `lo` occupies the lower 64 bits of a 128-bit register (upper half is zero).
    // We need to shift `hi` right by `bytes - 8` positions so that its first valid
    // byte aligns with byte `bytes - 8` of `lo`, then OR the two together.
    //
    // We achieve this with `vqtbl1q_u8`: build an identity index vector and subtract
    // `bytes - 8`. Lanes that underflow get their high bit set, which `TBL` maps to
    // zero — exactly what we want for the overlapping region.
    //
    // SAFETY: Both reads are within `[ptr, ptr + bytes)`. The intrinsics require NEON.
    unsafe {
        let base = vcombine_u8(
            vcreate_u8(0x0706050403020100),
            vcreate_u8(0x0F0E0D0C0B0A0908),
        );

        let lo = vcombine_u8(vld1_u8(ptr), vcreate_u8(0));
        let hi = vcombine_u8(vld1_u8(ptr.add(bytes - 8)), vcreate_u8(0));
        let shift = vmovq_n_u8((bytes - 8) as u8);
        let mask = vsubq_u8(base, shift);

        // Miri does not support the `vqtbl1q_u8` instruction.
        //
        // Because we want `Miri` to see the loads, we just emulate the shift portion of
        // the operation.
        let combined = if cfg!(miri) {
            use crate::{SIMDVector, arch::aarch64::u8x16};
            let lo = u8x16::from_underlying(arch, lo).to_array();
            let hi = u8x16::from_underlying(arch, hi).to_array();

            let combined: [u8; 16] = core::array::from_fn(|i| {
                if i < 8 {
                    lo[i]
                } else if i < bytes {
                    hi[i - (bytes - 8)]
                } else {
                    0
                }
            });

            u8x16::from_array(arch, combined).to_underlying()
        } else {
            vqtbl1q_u8(hi, mask)
        };

        vorrq_u8(lo, combined)
    }
}

/// Load the first `bytes` bytes from `ptr` into a `u64`.
///
/// Bytes beyond `bytes` are zero. This is efficient for small loads (≤8 bytes) because
/// it stays entirely in general-purpose registers with no SIMD involvement.
///
/// # Safety
///
/// * The memory in `[ptr, ptr + bytes.min(8))` must be readable and valid.
#[inline(always)]
unsafe fn load_first_of_8_bytes(ptr: *const u8, bytes: usize) -> u64 {
    // SAFETY: All reads are within `[ptr, ptr + bytes)`, which the caller asserts is valid.
    unsafe {
        if bytes >= 8 {
            std::ptr::read_unaligned(ptr as *const u64)
        } else if bytes >= 4 {
            let lo = std::ptr::read_unaligned(ptr as *const u32) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(bytes - 4) as *const u32) as u64;
            lo | (hi << ((bytes - 4) * 8))
        } else if bytes >= 2 {
            let lo = std::ptr::read_unaligned(ptr as *const u16) as u64;
            let hi = std::ptr::read_unaligned(ptr.add(bytes - 2) as *const u16) as u64;
            lo | (hi << ((bytes - 2) * 8))
        } else if bytes == 1 {
            std::ptr::read(ptr) as u64
        } else {
            0
        }
    }
}

/// Load the first `bytes` bytes from `ptr` into a 128-bit Neon register.
///
/// For full loads (≥16), uses a single `vld1q_u8`. For 8 < bytes < 16, uses the two-load
/// shuffle technique. For ≤8 bytes, uses GPR-based overlapping reads.
///
/// # Safety
///
/// * The memory in `[ptr, ptr + bytes)` must be readable and valid.
/// * Memory at and above `ptr + bytes` will not be accessed.
#[inline(always)]
unsafe fn load_first_of_16_bytes(arch: Neon, ptr: *const u8, bytes: usize) -> uint8x16_t {
    if bytes >= 16 {
        // SAFETY: Full load is valid since `bytes >= 16`.
        return unsafe { vld1q_u8(ptr) };
    }

    if bytes > 8 {
        // SAFETY: `bytes` is in `(8, 16)` and `[ptr, ptr + bytes)` is valid.
        return unsafe { load_8_to_16_bytes(arch, ptr, bytes) };
    }

    // SAFETY: `bytes` is in `[0, 8]` and `[ptr, ptr + bytes)` is valid.
    //
    // The presence of `Neon` enables the use of "neon" intrinsics.
    unsafe {
        let v = load_first_of_8_bytes(ptr, bytes);
        vcombine_u8(vcreate_u8(v), vcreate_u8(0))
    }
}

/// Load the first `first` elements of a 32-bit type from `ptr` into a 128-bit register.
///
/// # Safety
///
/// * The memory in `[ptr, ptr + first)` must be readable and valid (element-wise).
/// * Memory at and above `ptr + first` will not be accessed.
#[inline(always)]
unsafe fn load_first_32x4(_: Neon, ptr: *const u32, first: usize) -> uint32x4_t {
    // SAFETY: All reads are within `[ptr, ptr + first)`.
    //
    // The presence of `Neon` enables the use of "neon" intrinsics.
    unsafe {
        if first >= 4 {
            vld1q_u32(ptr)
        } else if first == 3 {
            let lo = vld1_u32(ptr);
            let hi = vld1_lane_u32(ptr.add(2), vcreate_u32(0), 0);
            vcombine_u32(lo, hi)
        } else if first == 2 {
            vcombine_u32(vld1_u32(ptr), vcreate_u32(0))
        } else if first == 1 {
            vcombine_u32(vcreate_u32(ptr.read_unaligned() as u64), vcreate_u32(0))
        } else {
            vmovq_n_u32(0)
        }
    }
}

/// Load the first `first` elements of a 32-bit type from `ptr` into a 64-bit register.
///
/// # Safety
///
/// * The memory in `[ptr, ptr + first)` must be readable and valid (element-wise).
/// * Memory at and above `ptr + first` will not be accessed.
#[inline(always)]
unsafe fn load_first_32x2(_: Neon, ptr: *const u32, first: usize) -> uint32x2_t {
    // SAFETY: All reads are within `[ptr, ptr + first)`.
    //
    // The presence of `Neon` enables the use of "neon" intrinsics.
    unsafe {
        if first >= 2 {
            vld1_u32(ptr)
        } else if first == 1 {
            vcreate_u32(ptr.read_unaligned() as u64)
        } else {
            vmov_n_u32(0)
        }
    }
}

/// Load the first `first` elements of a 64-bit type from `ptr` into a 128-bit register.
///
/// # Safety
///
/// * The memory in `[ptr, ptr + first)` must be readable and valid (element-wise).
/// * Memory at and above `ptr + first` will not be accessed.
#[inline(always)]
unsafe fn load_first_64x2(_: Neon, ptr: *const u64, first: usize) -> uint64x2_t {
    // SAFETY: All reads are within `[ptr, ptr + first)`.
    //
    // The presence of `Neon` enables the use of "neon" intrinsics.
    unsafe {
        if first >= 2 {
            vld1q_u64(ptr)
        } else if first == 1 {
            vcombine_u64(vld1_u64(ptr), vcreate_u64(0))
        } else {
            vmovq_n_u64(0)
        }
    }
}
