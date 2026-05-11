/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Low-level functions
//!
//! The methods here are meant to be primitives used by the distance functions for the
//! various scalar-quantized-like quantizers.
//!
//! As such, they typically return integer distance results since they largely operate over
//! raw bit-slices.
//!
//! ## Micro-architecture Mapping
//!
//! There are two interfaces for interacting with the distance primitives:
//!
//! * [`diskann_wide::arch::Target2`]: A micro-architecture aware interface where the target
//!   micro-architecture is provided as an explicit argument.
//!
//!   This can be used in conjunction with [`diskann_wide::Architecture::run2`] to apply the
//!   necessary target-features to opt-into newer architecture code generation when
//!   compiling the whole binary for an older architecture.
//!
//!   This interface is also composable with micro-architecture dispatching done higher in
//!   the callstack, and so should be preferred when incorporating into quantizer distance
//!   computations.
//!
//! * [`diskann_vector::PureDistanceFunction`]: If micro-architecture awareness is not needed,
//!   this provides a simple interface targeting [`diskann_wide::ARCH`] (the current compilation
//!   architecture).
//!
//!   This interface will always yield a binary compatible with the compilation architecture
//!   target, but will not enable faster code-paths when compiling for older architectures.
//!
//! The following table summarizes the implementation status of kernels. All kernels have
//! `diskann_wide::arch::Scalar` implementation fallbacks.
//!
//! Implementation Kind:
//!
//! * "Fallback": A fallback implementation using scalar indexing.
//!
//! * "Optimized": A better implementation than "fallback" that does not contain
//!   target-depeendent code, instead relying on compiler optimizations.
//!
//!   Micro-architecture dispatch is still relevant as it allows the compiler to generate
//!   better code for newer machines.
//!
//! * "Yes": Architecture specific SIMD implementation exists.
//!
//! * "No": Architecture specific implementation does not exist - the next most-specific
//!   implementation is used. For example, if a `x86-64-v3` implementation does not exist,
//!   then the "scalar" implementation will be used instead.
//!
//! Type Aliases
//!
//! * `USlice<N>`: `BitSlice<N, Unsigned, Dense>`
//! * `TSlice<N>`: `BitSlice<N, Unsigned, BitTranspose>`
//! * `BSlice`: `BitSlice<1, Binary, Dense>`
//!
//! * `MV<T>`: [`diskann_vector::MathematicalValue<T>`]
//!
//! ### Inner Product
//!
//! | LHS           | RHS           | Result    | Scalar    | x86-64-v3     | x86-64-v4 | Neon      |
//! |---------------|---------------|-----------|-----------|---------------|-----------|-----------|
//! | `USlice<1>`   | `USlice<1>`   | `MV<u32>` | Optimized | Optimized     | Uses V3   | Optimized |
//! | `USlice<2>`   | `USlice<2>`   | `MV<u32>` | Fallback  | Yes           | Yes       | Fallback  |
//! | `USlice<3>`   | `USlice<3>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<4>`   | `USlice<4>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `USlice<5>`   | `USlice<5>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<6>`   | `USlice<6>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<7>`   | `USlice<7>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<8>`   | `USlice<8>`   | `MV<u32>` | Yes       | Yes           | Yes       | Fallback  |
//! |               |               |           |           |               |           |           |
//! | `USlice<8>`   | `USlice<4>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `USlice<8>`   | `USlice<2>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `USlice<8>`   | `USlice<1>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! |               |               | `       ` |           |               |           |           |
//! | `TSlice<4>`   | `USlice<1>`   | `MV<u32>` | Optimized | Optimized     | Optimized | Optimized |
//! |               |               | `       ` |           |               |           |           |
//! | `&[f32]`      | `USlice<1>`   | `MV<f32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<2>`   | `MV<f32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<3>`   | `MV<f32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<4>`   | `MV<f32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<5>`   | `MV<f32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<6>`   | `MV<f32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<7>`   | `MV<f32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `&[f32]`      | `USlice<8>`   | `MV<f32>` | Fallback  | No            | Uses V3   | Fallback  |
//!
//! ### Squared L2
//!
//! | LHS           | RHS           | Result    | Scalar    | x86-64-v3     | x86-64-v4 | Neon      |
//! |---------------|---------------|-----------|-----------|---------------|-----------|-----------|
//! | `USlice<1>`   | `USlice<1>`   | `MV<u32>` | Optimized | Optimized     | Uses V3   | Optimized |
//! | `USlice<2>`   | `USlice<2>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `USlice<3>`   | `USlice<3>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<4>`   | `USlice<4>`   | `MV<u32>` | Fallback  | Yes           | Uses V3   | Fallback  |
//! | `USlice<5>`   | `USlice<5>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<6>`   | `USlice<6>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<7>`   | `USlice<7>`   | `MV<u32>` | Fallback  | No            | Uses V3   | Fallback  |
//! | `USlice<8>`   | `USlice<8>`   | `MV<u32>` | Yes       | Yes           | Yes       | Fallback  |
//!
//! ### Hamming
//!
//! | LHS           | RHS           | Result    | Scalar    | x86-64-v3     | x86-64-v4 | Neon      |
//! |---------------|---------------|-----------|-----------|---------------|-----------|-----------|
//! | `BSlice`      | `BSlice`      | `MV<u32>` | Optimized | Optimized     | Uses V3   | Optimized |

use diskann_vector::PureDistanceFunction;
use diskann_wide::{ARCH, Architecture, arch::Target2};
#[cfg(target_arch = "x86_64")]
use diskann_wide::{
    SIMDCast, SIMDDotProduct, SIMDMulAdd, SIMDReinterpret, SIMDSumTree, SIMDVector,
};

use super::{Binary, BitSlice, BitTranspose, Dense, Representation, Unsigned};
use crate::distances::{Hamming, InnerProduct, MV, MathematicalResult, SquaredL2, check_lengths};

// Convenience alias.
type USlice<'a, const N: usize, Perm = Dense> = BitSlice<'a, N, Unsigned, Perm>;

/// Retarget the architectures via the `retarget` inherent method.
///
/// * [`diskann_wide::arch::x86_64::V3`] -> [`diskann_wide::arch::Scalar`]
/// * [`diskann_wide::arch::x86_64::V4`] -> [`diskann_wide::arch::x86_64::V3`]
/// * [`diskann_wide::arch::aarch64::Neon`] -> [`diskann_wide::arch::Scalar`]
macro_rules! retarget {
    ($arch:path, $op:ty, ($N:literal, $M:literal)) => {
        impl Target2<
            $arch,
            MathematicalResult<u32>,
            USlice<'_, $N>,
            USlice<'_, $M>,
        > for $op {
            #[inline(always)]
            fn run(
                self,
                arch: $arch,
                x: USlice<'_, $N>,
                y: USlice<'_, $M>
            ) -> MathematicalResult<u32> {
                self.run(arch.retarget(), x, y)
            }
        }
    };
    ($arch:path, $op:ty, $N:literal) => {
        retarget!($arch, $op, ($N, $N));
    };
    ($arch:path, $op:ty, $($args:tt),+ $(,)?) => {
        $(retarget!($arch, $op, $args);)+
    };
}

/// Impledment [`diskann_vector::PureDistanceFunction`] using the current compilation architecture
macro_rules! dispatch_pure {
    ($op:ty, ($N:literal, $M:literal)) => {
        impl PureDistanceFunction<USlice<'_, $N>, USlice<'_, $M>, MathematicalResult<u32>> for $op {
            #[inline(always)]
            fn evaluate(x: USlice<'_, $N>, y: USlice<'_, $M>) -> MathematicalResult<u32> {
                (diskann_wide::ARCH).run2(Self, x, y)
            }
        }
    };
    ($op:ty, $N:literal) => {
        dispatch_pure!($op, ($N, $N));
    };
    ($op:ty, $($args:tt),+ $(,)?) => {
        $(dispatch_pure!($op, $args);)+
    }
}

/// Load 1 byte beginning at `ptr` and invoke `f` with that byte.
///
/// # Safety
///
/// * The memory range `[ptr, ptr + 1)` (in bytes) must be dereferencable.
/// * `ptr` does not need to be aligned.
#[cfg(target_arch = "x86_64")]
unsafe fn load_one<F, R>(ptr: *const u32, mut f: F) -> R
where
    F: FnMut(u32) -> R,
{
    // SAFETY: Caller asserts that one byte is readable.
    f(unsafe { ptr.cast::<u8>().read_unaligned() }.into())
}

/// Load 2 bytes beginning at `ptr` and invoke `f` with the value.
///
/// # Safety
///
/// * The memory range `[ptr, ptr + 2)` (in bytes) must be dereferencable.
/// * `ptr` does not need to be aligned.
#[cfg(target_arch = "x86_64")]
unsafe fn load_two<F, R>(ptr: *const u32, mut f: F) -> R
where
    F: FnMut(u32) -> R,
{
    // SAFETY: Caller asserts that two bytes are readable.
    f(unsafe { ptr.cast::<u16>().read_unaligned() }.into())
}

/// Load 3 bytes beginning at `ptr` and invoke `f` with the value.
///
/// # Safety
///
/// * The memory range `[ptr, ptr + 3)` (in bytes) must be dereferencable.
/// * `ptr` does not need to be aligned.
#[cfg(target_arch = "x86_64")]
unsafe fn load_three<F, R>(ptr: *const u32, mut f: F) -> R
where
    F: FnMut(u32) -> R,
{
    // SAFETY: Caller asserts that three bytes are readable. This loads the first two.
    let lo: u32 = unsafe { ptr.cast::<u16>().read_unaligned() }.into();
    // SAFETY: Caller asserts that three bytes are readable. This loads the third.
    let hi: u32 = unsafe { ptr.cast::<u8>().add(2).read_unaligned() }.into();
    f(lo | hi << 16)
}

/// Load 4 bytes beginning at `ptr` and invoke `f` with the value.
///
/// # Safety
///
/// * The memory range `[ptr, ptr + 4)` (in bytes) must be dereferencable.
/// * `ptr` does not need to be aligned.
#[cfg(target_arch = "x86_64")]
unsafe fn load_four<F, R>(ptr: *const u32, mut f: F) -> R
where
    F: FnMut(u32) -> R,
{
    // SAFETY: Caller asserts that four bytes are readable.
    f(unsafe { ptr.read_unaligned() })
}

////////////////////////////
// Distances on BitSlices //
////////////////////////////

/// Operations to apply to 1-bit encodings.
///
/// The general structure of 1-bit vector operations is the same, but the element wise
/// operator is different. This trait encapsulates the differences in behavior required
/// for different distance function.
///
/// The exact operations to apply depending on the representation of the bit encoding.
trait BitVectorOp<Repr>
where
    Repr: Representation<1>,
{
    /// Apply the op to all bits in the 64-bit arguments.
    fn on_u64(x: u64, y: u64) -> u32;

    /// Apply the op to all bits in the 8-bit arguments.
    ///
    /// NOTE: Implementations must have the correct behavior when the upper bits of `x`
    /// and `y` are set to 0 when handling epilogues.
    fn on_u8(x: u8, y: u8) -> u32;
}

/// Computing Squared-L2 amounts to evaluating the pop-count of a bitwise `xor`.
impl BitVectorOp<Unsigned> for SquaredL2 {
    #[inline(always)]
    fn on_u64(x: u64, y: u64) -> u32 {
        (x ^ y).count_ones()
    }
    #[inline(always)]
    fn on_u8(x: u8, y: u8) -> u32 {
        (x ^ y).count_ones()
    }
}

/// Computing Squared-L2 amounts to evaluating the pop-count of a bitwise `xor`.
impl BitVectorOp<Binary> for Hamming {
    #[inline(always)]
    fn on_u64(x: u64, y: u64) -> u32 {
        (x ^ y).count_ones()
    }
    #[inline(always)]
    fn on_u8(x: u8, y: u8) -> u32 {
        (x ^ y).count_ones()
    }
}

/// The implementation as `and` is not straight-forward.
///
/// Recall that scalar quantization encodings are unsigned, so "0" is zero and "1" is some
/// non-zero value.
///
/// When computing the inner product, `0 * x == 0` for all `x` and only `x * x` has a
/// non-zero value. Therefore, the elementwise op is an `and` and not `xnor`.
impl BitVectorOp<Unsigned> for InnerProduct {
    #[inline(always)]
    fn on_u64(x: u64, y: u64) -> u32 {
        (x & y).count_ones()
    }
    #[inline(always)]
    fn on_u8(x: u8, y: u8) -> u32 {
        (x & y).count_ones()
    }
}

/// A general algorithm for applying a bitwise operand to two dense bit vectors of equal
/// but arbitrary length.
///
/// NOTE: The `inline(always)` attribute is required to inheret the caller's target-features.
#[inline(always)]
fn bitvector_op<Op, Repr>(
    x: BitSlice<'_, 1, Repr>,
    y: BitSlice<'_, 1, Repr>,
) -> MathematicalResult<u32>
where
    Repr: Representation<1>,
    Op: BitVectorOp<Repr>,
{
    let len = check_lengths!(x, y)?;

    let px: *const u64 = x.as_ptr().cast();
    let py: *const u64 = y.as_ptr().cast();

    let mut i = 0;
    let mut s: u32 = 0;

    // Work in groups of 64
    let blocks = len / 64;
    while i < blocks {
        // SAFETY: We know at least 64-bits (8-bytes) are valid from this offset (by
        // guarantee of the `BitSlice`). All bit-patterns of a `u64` are valid, `u64: Copy`,
        // and an `unaligned` read is used.
        let vx = unsafe { px.add(i).read_unaligned() };

        // SAFETY: The same logic applies to `y` because:
        // 1. It has the same type as `x`.
        // 2. We've verified that it has the same length as `x`.
        let vy = unsafe { py.add(i).read_unaligned() };

        s += Op::on_u64(vx, vy);
        i += 1;
    }

    // Work in groups of 8
    i *= 8;
    let px: *const u8 = x.as_ptr();
    let py: *const u8 = y.as_ptr();

    let blocks = len / 8;
    while i < blocks {
        // SAFETY: The underlying pointer is a `*const u8` and we have checked that this
        // offset is within the bounds of the slice underlying the bitslice.
        let vx = unsafe { px.add(i).read_unaligned() };

        // SAFETY: The same logic applies to `y` because:
        // 1. It has the same type as `x`.
        // 2. We've verified that it has the same length as `x`.
        let vy = unsafe { py.add(i).read_unaligned() };
        s += Op::on_u8(vx, vy);
        i += 1;
    }

    if i * 8 != len {
        // SAFETY: The underlying slice is readable in the range
        // `[px, px + floor(len / 8) + 1)`. This accesses `px + floor(len / 8)`.
        let vx = unsafe { px.add(i).read_unaligned() };

        // SAFETY: Same as above.
        let vy = unsafe { py.add(i).read_unaligned() };
        let m = (0x01u8 << (len - 8 * i)) - 1;

        s += Op::on_u8(vx & m, vy & m)
    }
    Ok(MV::new(s))
}

/// Compute the hamming distance between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
impl PureDistanceFunction<BitSlice<'_, 1, Binary>, BitSlice<'_, 1, Binary>, MathematicalResult<u32>>
    for Hamming
{
    fn evaluate(x: BitSlice<'_, 1, Binary>, y: BitSlice<'_, 1, Binary>) -> MathematicalResult<u32> {
        bitvector_op::<Hamming, Binary>(x, y)
    }
}

///////////////
// SquaredL2 //
///////////////

/// Compute the squared L2 distance between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This can directly invoke the methods implemented in `vector` because
/// `BitSlice<'_, 8, Unsigned>` is isomorphic to `&[u8]`.
impl<A> Target2<A, MathematicalResult<u32>, USlice<'_, 8>, USlice<'_, 8>> for SquaredL2
where
    A: Architecture,
    diskann_vector::distance::SquaredL2: for<'a> Target2<A, MV<f32>, &'a [u8], &'a [u8]>,
{
    #[inline(always)]
    fn run(self, arch: A, x: USlice<'_, 8>, y: USlice<'_, 8>) -> MathematicalResult<u32> {
        check_lengths!(x, y)?;

        let r: MV<f32> = <_ as Target2<_, _, _, _>>::run(
            diskann_vector::distance::SquaredL2 {},
            arch,
            x.as_slice(),
            y.as_slice(),
        );

        Ok(MV::new(r.into_inner() as u32))
    }
}

/// Compute the squared L2 distance between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This implementation is optimized around x86 with the AVX2 vector extension.
/// Specifically, we try to hit `Wide::<i32, 8> as SIMDDotProduct<Wide<i16, 8>>` so we can
/// hit the `_mm256_madd_epi16` intrinsic.
///
/// Also note that AVX2 does not have 16-bit integer bit-shift instructions. Instead, we
/// have to use 32-bit integer shifts and then bit-cast to 16-bit intrinsics.
/// This works because we need to apply the same shift to all lanes.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 4>, USlice<'_, 4>>
    for SquaredL2
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 4>,
        y: USlice<'_, 4>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        let blocks = len / 8;
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x000f000f);
            while i + 8 < blocks {
                // SAFETY: We have checked that `i + 8 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 8 * std::mem::size_of::<u32>())` is valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: i16s = (vx & mask).reinterpret_simd();
                let wy: i16s = (vy & mask).reinterpret_simd();
                let d = wx - wy;
                s0 = s0.dot_simd(d, d);

                let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
                let d = wx - wy;
                s1 = s1.dot_simd(d, d);

                let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
                let d = wx - wy;
                s2 = s2.dot_simd(d, d);

                let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
                let d = wx - wy;
                s3 = s3.dot_simd(d, d);

                i += 8;
            }

            let remainder = blocks - i;

            // SAFETY: At least one value of type `u32` is valid for an unaligned starting
            // at offset `i`. The exact number is computed as `remainder`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u32s::load_simd_first(arch, px_u32.add(i), remainder) };

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u32s::load_simd_first(arch, py_u32.add(i), remainder) };

            let wx: i16s = (vx & mask).reinterpret_simd();
            let wy: i16s = (vy & mask).reinterpret_simd();
            let d = wx - wy;
            s0 = s0.dot_simd(d, d);

            let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
            let d = wx - wy;
            s1 = s1.dot_simd(d, d);

            let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
            let d = wx - wy;
            s2 = s2.dot_simd(d, d);

            let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
            let d = wx - wy;
            s3 = s3.dot_simd(d, d);

            i += remainder;

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
        }

        // Convert blocks to indexes.
        i *= 8;

        // Deal with the remainder the slow way.
        if i != len {
            // Outline the fallback routine to keep code-generation at this level cleaner.
            #[inline(never)]
            fn fallback(x: USlice<'_, 4>, y: USlice<'_, 4>, from: usize) -> u32 {
                let mut s: i32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as i32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as i32;
                    let d = ix - iy;
                    s += d * d;
                }
                s as u32
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

/// Compute the squared L2 distance between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This implementation is optimized for x86 with the AVX-512 vector extension.
/// It scales the V3 approach to 512-bit registers: we load data as `u32x16`, shift and
/// mask to extract 4-bit nibbles at 16-bit granularity (`0x000f000f` mask), reinterpret
/// as `i16x32`, compute differences, and use `_mm512_madd_epi16` via `dot_simd` to
/// accumulate squared differences into `i32x16`.
///
/// AVX-512 does not have 16-bit integer bit-shift instructions, so we use 32-bit integer
/// shifts and then bit-cast to 16-bit intrinsics, which works because we apply the same
/// shift to all lanes.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V4, MathematicalResult<u32>, USlice<'_, 4>, USlice<'_, 4>>
    for SquaredL2
{
    #[expect(non_camel_case_types)]
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V4,
        x: USlice<'_, 4>,
        y: USlice<'_, 4>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        type i32s = <diskann_wide::arch::x86_64::V4 as Architecture>::i32x16;
        type u32s = <diskann_wide::arch::x86_64::V4 as Architecture>::u32x16;
        type i16s = <diskann_wide::arch::x86_64::V4 as Architecture>::i16x32;

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        let blocks = len / 8;
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x000f000f);
            while i + 16 < blocks {
                // SAFETY: We have checked that `i + 16 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 16 * std::mem::size_of::<u32>())` is
                // valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: i16s = (vx & mask).reinterpret_simd();
                let wy: i16s = (vy & mask).reinterpret_simd();
                let d = wx - wy;
                s0 = s0.dot_simd(d, d);

                let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
                let d = wx - wy;
                s1 = s1.dot_simd(d, d);

                let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
                let d = wx - wy;
                s2 = s2.dot_simd(d, d);

                let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
                let d = wx - wy;
                s3 = s3.dot_simd(d, d);

                i += 16;
            }

            let remainder = blocks - i;

            // SAFETY: At least one value of type `u32` is valid for an unaligned starting
            // at offset `i`. The exact number is computed as `remainder`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u32s::load_simd_first(arch, px_u32.add(i), remainder) };

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u32s::load_simd_first(arch, py_u32.add(i), remainder) };

            let wx: i16s = (vx & mask).reinterpret_simd();
            let wy: i16s = (vy & mask).reinterpret_simd();
            let d = wx - wy;
            s0 = s0.dot_simd(d, d);

            let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
            let d = wx - wy;
            s1 = s1.dot_simd(d, d);

            let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
            let d = wx - wy;
            s2 = s2.dot_simd(d, d);

            let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
            let d = wx - wy;
            s3 = s3.dot_simd(d, d);

            i += remainder;

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
        }

        // Convert blocks to indexes.
        i *= 8;

        // Deal with the remainder the slow way.
        if i != len {
            // Outline the fallback routine to keep code-generation at this level cleaner.
            #[inline(never)]
            fn fallback(x: USlice<'_, 4>, y: USlice<'_, 4>, from: usize) -> u32 {
                let mut s: i32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as i32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as i32;
                    let d = ix - iy;
                    s += d * d;
                }
                s as u32
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

/// Compute the squared L2 distance between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This implementation is optimized around x86 with the AVX2 vector extension.
/// Specifically, we try to hit `Wide::<i32, 8> as SIMDDotProduct<Wide<i16, 8>>` so we can
/// hit the `_mm256_madd_epi16` intrinsic.
///
/// Also note that AVX2 does not have 16-bit integer bit-shift instructions. Instead, we
/// have to use 32-bit integer shifts and then bit-cast to 16-bit intrinsics.
/// This works because we need to apply the same shift to all lanes.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 2>, USlice<'_, 2>>
    for SquaredL2
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 2>,
        y: USlice<'_, 2>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        let blocks = len / 16;
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x00030003);
            while i + 8 < blocks {
                // SAFETY: We have checked that `i + 8 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 8 * std::mem::size_of::<u32>())` is valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: i16s = (vx & mask).reinterpret_simd();
                let wy: i16s = (vy & mask).reinterpret_simd();
                let d = wx - wy;
                s0 = s0.dot_simd(d, d);

                let wx: i16s = (vx >> 2 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 2 & mask).reinterpret_simd();
                let d = wx - wy;
                s1 = s1.dot_simd(d, d);

                let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
                let d = wx - wy;
                s2 = s2.dot_simd(d, d);

                let wx: i16s = (vx >> 6 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 6 & mask).reinterpret_simd();
                let d = wx - wy;
                s3 = s3.dot_simd(d, d);

                let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
                let d = wx - wy;
                s0 = s0.dot_simd(d, d);

                let wx: i16s = (vx >> 10 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 10 & mask).reinterpret_simd();
                let d = wx - wy;
                s1 = s1.dot_simd(d, d);

                let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
                let d = wx - wy;
                s2 = s2.dot_simd(d, d);

                let wx: i16s = (vx >> 14 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 14 & mask).reinterpret_simd();
                let d = wx - wy;
                s3 = s3.dot_simd(d, d);

                i += 8;
            }

            let remainder = blocks - i;

            // SAFETY: At least one value of type `u32` is valid for an unaligned starting
            // at offset `i`. The exact number is computed as `remainder`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u32s::load_simd_first(arch, px_u32.add(i), remainder) };

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u32s::load_simd_first(arch, py_u32.add(i), remainder) };
            let wx: i16s = (vx & mask).reinterpret_simd();
            let wy: i16s = (vy & mask).reinterpret_simd();
            let d = wx - wy;
            s0 = s0.dot_simd(d, d);

            let wx: i16s = (vx >> 2 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 2 & mask).reinterpret_simd();
            let d = wx - wy;
            s1 = s1.dot_simd(d, d);

            let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
            let d = wx - wy;
            s2 = s2.dot_simd(d, d);

            let wx: i16s = (vx >> 6 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 6 & mask).reinterpret_simd();
            let d = wx - wy;
            s3 = s3.dot_simd(d, d);

            let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
            let d = wx - wy;
            s0 = s0.dot_simd(d, d);

            let wx: i16s = (vx >> 10 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 10 & mask).reinterpret_simd();
            let d = wx - wy;
            s1 = s1.dot_simd(d, d);

            let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
            let d = wx - wy;
            s2 = s2.dot_simd(d, d);

            let wx: i16s = (vx >> 14 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 14 & mask).reinterpret_simd();
            let d = wx - wy;
            s3 = s3.dot_simd(d, d);

            i += remainder;

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
        }

        // Convert blocks to indexes.
        i *= 16;

        // Deal with the remainder the slow way.
        if i != len {
            // Outline the fallback routine to keep code-generation at this level cleaner.
            #[inline(never)]
            fn fallback(x: USlice<'_, 2>, y: USlice<'_, 2>, from: usize) -> u32 {
                let mut s: i32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as i32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as i32;
                    let d = ix - iy;
                    s += d * d;
                }
                s as u32
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

/// Compute the squared L2 distance between bitvectors `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
impl<A> Target2<A, MathematicalResult<u32>, USlice<'_, 1>, USlice<'_, 1>> for SquaredL2
where
    A: Architecture,
{
    fn run(self, _: A, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MathematicalResult<u32> {
        bitvector_op::<Self, Unsigned>(x, y)
    }
}

/// An implementation for L2 distance that uses scalar indexing for the implementation.
macro_rules! impl_fallback_l2 {
    ($N:literal) => {
        /// Compute the squared L2 distance between `x` and `y`.
        ///
        /// Returns an error if the arguments have different lengths.
        ///
        /// # Performance
        ///
        /// This function uses a generic implementation and therefore is not very fast.
        impl Target2<diskann_wide::arch::Scalar, MathematicalResult<u32>, USlice<'_, $N>, USlice<'_, $N>> for SquaredL2 {
            #[inline(never)]
            fn run(
                self,
                _: diskann_wide::arch::Scalar,
                x: USlice<'_, $N>,
                y: USlice<'_, $N>
            ) -> MathematicalResult<u32> {
                let len = check_lengths!(x, y)?;

                let mut accum: i32 = 0;
                for i in 0..len {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix: i32 = unsafe { x.get_unchecked(i) } as i32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy: i32 = unsafe { y.get_unchecked(i) } as i32;
                    let diff = ix - iy;
                    accum += diff * diff;
                }
                Ok(MV::new(accum as u32))
            }
        }
    };
    ($($N:literal),+ $(,)?) => {
        $(impl_fallback_l2!($N);)+
    };
}

impl_fallback_l2!(7, 6, 5, 4, 3, 2);

#[cfg(target_arch = "x86_64")]
retarget!(diskann_wide::arch::x86_64::V3, SquaredL2, 7, 6, 5, 3);

#[cfg(target_arch = "x86_64")]
retarget!(diskann_wide::arch::x86_64::V4, SquaredL2, 7, 6, 5, 3, 2);

dispatch_pure!(SquaredL2, 1, 2, 3, 4, 5, 6, 7, 8);
#[cfg(target_arch = "aarch64")]
retarget!(
    diskann_wide::arch::aarch64::Neon,
    SquaredL2,
    7,
    6,
    5,
    4,
    3,
    2
);

///////////////////
// Inner Product //
///////////////////

/// Compute the inner product between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This can directly invoke the methods implemented in `vector` because
/// `BitSlice<'_, 8, Unsigned>` is isomorphic to `&[u8]`.
impl<A> Target2<A, MathematicalResult<u32>, USlice<'_, 8>, USlice<'_, 8>> for InnerProduct
where
    A: Architecture,
    diskann_vector::distance::InnerProduct: for<'a> Target2<A, MV<f32>, &'a [u8], &'a [u8]>,
{
    #[inline(always)]
    fn run(self, arch: A, x: USlice<'_, 8>, y: USlice<'_, 8>) -> MathematicalResult<u32> {
        check_lengths!(x, y)?;
        let r: MV<f32> = <_ as Target2<_, _, _, _>>::run(
            diskann_vector::distance::InnerProduct {},
            arch,
            x.as_slice(),
            y.as_slice(),
        );

        Ok(MV::new(r.into_inner() as u32))
    }
}

/// Compute the inner product between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This is optimized around the `__mm512_dpbusd_epi32` VNNI instruction, which computes the
/// pairwise dot product between vectors of 8-bit integers and accumulates groups of 4 with
/// an `i32` accumulation vector.
///
/// One quirk of this instruction is that one argument must be unsigned and the other must
/// be signed. Since thie kernsl works on 2-bit integers, this is not a limitation. Just
/// something to be aware of.
///
/// Since AVX512 does not have an 8-bit shift instruction, we generally load data as
/// `u32x16` (which has a native shift) and bit-cast it to `u8x64` as needed.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V4, MathematicalResult<u32>, USlice<'_, 2>, USlice<'_, 2>>
    for InnerProduct
{
    #[expect(non_camel_case_types)]
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V4,
        x: USlice<'_, 2>,
        y: USlice<'_, 2>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        type i32s = <diskann_wide::arch::x86_64::V4 as Architecture>::i32x16;
        type u32s = <diskann_wide::arch::x86_64::V4 as Architecture>::u32x16;
        type u8s = <diskann_wide::arch::x86_64::V4 as Architecture>::u8x64;
        type i8s = <diskann_wide::arch::x86_64::V4 as Architecture>::i8x64;

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        let blocks = len.div_ceil(16);
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x03030303);
            while i + 16 < blocks {
                // SAFETY: We have checked that `i + 16 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 16 * std::mem::size_of::<u32>())` is valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: u8s = (vx & mask).reinterpret_simd();
                let wy: i8s = (vy & mask).reinterpret_simd();
                s0 = s0.dot_simd(wx, wy);

                let wx: u8s = ((vx >> 2) & mask).reinterpret_simd();
                let wy: i8s = ((vy >> 2) & mask).reinterpret_simd();
                s1 = s1.dot_simd(wx, wy);

                let wx: u8s = ((vx >> 4) & mask).reinterpret_simd();
                let wy: i8s = ((vy >> 4) & mask).reinterpret_simd();
                s2 = s2.dot_simd(wx, wy);

                let wx: u8s = ((vx >> 6) & mask).reinterpret_simd();
                let wy: i8s = ((vy >> 6) & mask).reinterpret_simd();
                s3 = s3.dot_simd(wx, wy);

                i += 16;
            }

            // Here
            // * `len / 4` gives the number of full bytes
            // * `4 * i` gives the number of bytes processed.
            let remainder = len / 4 - 4 * i;

            // SAFETY: At least `remainder` bytes are valid starting at an offset of `i`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u8s::load_simd_first(arch, px_u32.add(i).cast::<u8>(), remainder) };
            let vx: u32s = vx.reinterpret_simd();

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u8s::load_simd_first(arch, py_u32.add(i).cast::<u8>(), remainder) };
            let vy: u32s = vy.reinterpret_simd();

            let wx: u8s = (vx & mask).reinterpret_simd();
            let wy: i8s = (vy & mask).reinterpret_simd();
            s0 = s0.dot_simd(wx, wy);

            let wx: u8s = ((vx >> 2) & mask).reinterpret_simd();
            let wy: i8s = ((vy >> 2) & mask).reinterpret_simd();
            s1 = s1.dot_simd(wx, wy);

            let wx: u8s = ((vx >> 4) & mask).reinterpret_simd();
            let wy: i8s = ((vy >> 4) & mask).reinterpret_simd();
            s2 = s2.dot_simd(wx, wy);

            let wx: u8s = ((vx >> 6) & mask).reinterpret_simd();
            let wy: i8s = ((vy >> 6) & mask).reinterpret_simd();
            s3 = s3.dot_simd(wx, wy);

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
            i = (4 * i) + remainder;
        }

        // Convert blocks to indexes.
        i *= 4;

        // Deal with the remainder the slow way.
        debug_assert!(len - i <= 3);
        let rest = (len - i).min(3);
        if i != len {
            for j in 0..rest {
                // SAFETY: `i` is guaranteed to be less than `x.len()`.
                let ix = unsafe { x.get_unchecked(i + j) } as u32;
                // SAFETY: `i` is guaranteed to be less than `y.len()`.
                let iy = unsafe { y.get_unchecked(i + j) } as u32;
                s += ix * iy;
            }
        }

        Ok(MV::new(s))
    }
}

/// Compute the inner product between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This implementation is optimized around x86 with the AVX2 vector extension.
/// Specifically, we try to hit `Wide::<i32, 8> as SIMDDotProduct<Wide<i16, 8>>` so we can
/// hit the `_mm256_madd_epi16` intrinsic.
///
/// Also note that AVX2 does not have 16-bit integer bit-shift instructions. Instead, we
/// have to use 32-bit integer shifts and then bit-cast to 16-bit intrinsics.
/// This works because we need to apply the same shift to all lanes.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 4>, USlice<'_, 4>>
    for InnerProduct
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 4>,
        y: USlice<'_, 4>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        let blocks = len / 8;
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x000f000f);
            while i + 8 < blocks {
                // SAFETY: We have checked that `i + 8 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 8 * std::mem::size_of::<u32>())` is valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: i16s = (vx & mask).reinterpret_simd();
                let wy: i16s = (vy & mask).reinterpret_simd();
                s0 = s0.dot_simd(wx, wy);

                let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
                s1 = s1.dot_simd(wx, wy);

                let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
                s2 = s2.dot_simd(wx, wy);

                let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
                s3 = s3.dot_simd(wx, wy);

                i += 8;
            }

            let remainder = blocks - i;

            // SAFETY: At least one value of type `u32` is valid for an unaligned starting
            // at offset `i`. The exact number is computed as `remainder`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u32s::load_simd_first(arch, px_u32.add(i), remainder) };

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u32s::load_simd_first(arch, py_u32.add(i), remainder) };

            let wx: i16s = (vx & mask).reinterpret_simd();
            let wy: i16s = (vy & mask).reinterpret_simd();
            s0 = s0.dot_simd(wx, wy);

            let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
            s1 = s1.dot_simd(wx, wy);

            let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
            s2 = s2.dot_simd(wx, wy);

            let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
            s3 = s3.dot_simd(wx, wy);

            i += remainder;

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
        }

        // Convert blocks to indexes.
        i *= 8;

        // Deal with the remainder the slow way.
        if i != len {
            // Outline the fallback routine to keep code-generation at this level cleaner.
            #[inline(never)]
            fn fallback(x: USlice<'_, 4>, y: USlice<'_, 4>, from: usize) -> u32 {
                let mut s: u32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as u32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as u32;
                    s += ix * iy;
                }
                s
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

/// Compute the inner product between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This is optimized around the `__mm512_dpbusd_epi32` VNNI instruction, which computes the
/// pairwise dot product between vectors of 8-bit integers and accumulates groups of 4 with
/// an `i32` accumulation vector.
///
/// For 4-bit values, each byte holds 2 nibbles. We load data as `u32x16`, mask with
/// `0x0f0f0f0f` to extract the low nibbles as bytes, and shift right by 4 then mask to
/// extract the high nibbles. This gives us `u8x64` / `i8x64` operands for VNNI, requiring
/// only 2 shift positions instead of 4 for the V3 `madd_epi16` approach.
///
/// Since AVX-512 does not have an 8-bit shift instruction, we load data as `u32x16`
/// (which has a native shift) and bit-cast to `u8x64` as needed.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V4, MathematicalResult<u32>, USlice<'_, 4>, USlice<'_, 4>>
    for InnerProduct
{
    #[expect(non_camel_case_types)]
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V4,
        x: USlice<'_, 4>,
        y: USlice<'_, 4>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        type i32s = <diskann_wide::arch::x86_64::V4 as Architecture>::i32x16;
        type u32s = <diskann_wide::arch::x86_64::V4 as Architecture>::u32x16;
        type u8s = <diskann_wide::arch::x86_64::V4 as Architecture>::u8x64;
        type i8s = <diskann_wide::arch::x86_64::V4 as Architecture>::i8x64;

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        // Each u32 holds 8 nibbles = 8 four-bit values.
        let blocks = len.div_ceil(8);
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x0f0f0f0f);
            while i + 16 < blocks {
                // SAFETY: We have checked that `i + 16 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 16 * std::mem::size_of::<u32>())` is
                // valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: u8s = (vx & mask).reinterpret_simd();
                let wy: i8s = (vy & mask).reinterpret_simd();
                s0 = s0.dot_simd(wx, wy);

                let wx: u8s = ((vx >> 4) & mask).reinterpret_simd();
                let wy: i8s = ((vy >> 4) & mask).reinterpret_simd();
                s1 = s1.dot_simd(wx, wy);

                i += 16;
            }

            // Here
            // * `len / 2` gives the number of full bytes (2 nibbles per byte)
            // * `4 * i` gives the number of bytes processed (4 bytes per u32 × i u32s).
            let remainder = len / 2 - 4 * i;

            // SAFETY: At least `remainder` bytes are valid starting at an offset of `i`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u8s::load_simd_first(arch, px_u32.add(i).cast::<u8>(), remainder) };
            let vx: u32s = vx.reinterpret_simd();

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u8s::load_simd_first(arch, py_u32.add(i).cast::<u8>(), remainder) };
            let vy: u32s = vy.reinterpret_simd();

            let wx: u8s = (vx & mask).reinterpret_simd();
            let wy: i8s = (vy & mask).reinterpret_simd();
            s0 = s0.dot_simd(wx, wy);

            let wx: u8s = ((vx >> 4) & mask).reinterpret_simd();
            let wy: i8s = ((vy >> 4) & mask).reinterpret_simd();
            s1 = s1.dot_simd(wx, wy);

            s = (s0 + s1).sum_tree() as u32;
            i = (4 * i) + remainder;
        }

        // Convert bytes to nibble indexes.
        i *= 2;

        // Deal with the remainder the slow way (at most 1 element).
        debug_assert!(len - i <= 1);
        if i != len {
            // SAFETY: `i` is guaranteed to be less than `x.len()`.
            let ix = unsafe { x.get_unchecked(i) } as u32;
            // SAFETY: `i` is guaranteed to be less than `y.len()`.
            let iy = unsafe { y.get_unchecked(i) } as u32;
            s += ix * iy;
        }

        Ok(MV::new(s))
    }
}

/// Compute the inner product between `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
///
/// # Implementation Notes
///
/// This implementation is optimized around x86 with the AVX2 vector extension.
/// Specifically, we try to hit `Wide::<i32, 8> as SIMDDotProduct<Wide<i16, 8>>` so we can
/// hit the `_mm256_madd_epi16` intrinsic.
///
/// Also note that AVX2 does not have 16-bit integer bit-shift instructions. Instead, we
/// have to use 32-bit integer shifts and then bit-cast to 16-bit intrinsics.
/// This works because we need to apply the same shift to all lanes.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 2>, USlice<'_, 2>>
    for InnerProduct
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 2>,
        y: USlice<'_, 2>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px_u32: *const u32 = x.as_ptr().cast();
        let py_u32: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        // The number of 32-bit blocks over the underlying slice.
        let blocks = len / 16;
        if i < blocks {
            let mut s0 = i32s::default(arch);
            let mut s1 = i32s::default(arch);
            let mut s2 = i32s::default(arch);
            let mut s3 = i32s::default(arch);
            let mask = u32s::splat(arch, 0x00030003);
            while i + 8 < blocks {
                // SAFETY: We have checked that `i + 8 < blocks` which means the address
                // range `[px_u32 + i, px_u32 + i + 8 * std::mem::size_of::<u32>())` is valid.
                //
                // The load has no alignment requirements.
                let vx = unsafe { u32s::load_simd(arch, px_u32.add(i)) };

                // SAFETY: The same logic applies to `y` because:
                // 1. It has the same type as `x`.
                // 2. We've verified that it has the same length as `x`.
                let vy = unsafe { u32s::load_simd(arch, py_u32.add(i)) };

                let wx: i16s = (vx & mask).reinterpret_simd();
                let wy: i16s = (vy & mask).reinterpret_simd();
                s0 = s0.dot_simd(wx, wy);

                let wx: i16s = (vx >> 2 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 2 & mask).reinterpret_simd();
                s1 = s1.dot_simd(wx, wy);

                let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
                s2 = s2.dot_simd(wx, wy);

                let wx: i16s = (vx >> 6 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 6 & mask).reinterpret_simd();
                s3 = s3.dot_simd(wx, wy);

                let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
                s0 = s0.dot_simd(wx, wy);

                let wx: i16s = (vx >> 10 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 10 & mask).reinterpret_simd();
                s1 = s1.dot_simd(wx, wy);

                let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
                s2 = s2.dot_simd(wx, wy);

                let wx: i16s = (vx >> 14 & mask).reinterpret_simd();
                let wy: i16s = (vy >> 14 & mask).reinterpret_simd();
                s3 = s3.dot_simd(wx, wy);

                i += 8;
            }

            let remainder = blocks - i;

            // SAFETY: At least one value of type `u32` is valid for an unaligned starting
            // at offset `i`. The exact number is computed as `remainder`.
            //
            // The predicated load is guaranteed not to access memory after `remainder` and
            // has no alignment requirements.
            let vx = unsafe { u32s::load_simd_first(arch, px_u32.add(i), remainder) };

            // SAFETY: The same logic applies to `y` because:
            // 1. It has the same type as `x`.
            // 2. We've verified that it has the same length as `x`.
            let vy = unsafe { u32s::load_simd_first(arch, py_u32.add(i), remainder) };
            let wx: i16s = (vx & mask).reinterpret_simd();
            let wy: i16s = (vy & mask).reinterpret_simd();
            s0 = s0.dot_simd(wx, wy);

            let wx: i16s = (vx >> 2 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 2 & mask).reinterpret_simd();
            s1 = s1.dot_simd(wx, wy);

            let wx: i16s = (vx >> 4 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 4 & mask).reinterpret_simd();
            s2 = s2.dot_simd(wx, wy);

            let wx: i16s = (vx >> 6 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 6 & mask).reinterpret_simd();
            s3 = s3.dot_simd(wx, wy);

            let wx: i16s = (vx >> 8 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 8 & mask).reinterpret_simd();
            s0 = s0.dot_simd(wx, wy);

            let wx: i16s = (vx >> 10 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 10 & mask).reinterpret_simd();
            s1 = s1.dot_simd(wx, wy);

            let wx: i16s = (vx >> 12 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 12 & mask).reinterpret_simd();
            s2 = s2.dot_simd(wx, wy);

            let wx: i16s = (vx >> 14 & mask).reinterpret_simd();
            let wy: i16s = (vy >> 14 & mask).reinterpret_simd();
            s3 = s3.dot_simd(wx, wy);

            i += remainder;

            s = ((s0 + s1) + (s2 + s3)).sum_tree() as u32;
        }

        // Convert blocks to indexes.
        i *= 16;

        // Deal with the remainder the slow way.
        if i != len {
            // Outline the fallback routine to keep code-generation at this level cleaner.
            #[inline(never)]
            fn fallback(x: USlice<'_, 2>, y: USlice<'_, 2>, from: usize) -> u32 {
                let mut s: u32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as u32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as u32;
                    s += ix * iy;
                }
                s
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

/// Compute the inner product between bitvectors `x` and `y`.
///
/// Returns an error if the arguments have different lengths.
impl<A> Target2<A, MathematicalResult<u32>, USlice<'_, 1>, USlice<'_, 1>> for InnerProduct
where
    A: Architecture,
{
    #[inline(always)]
    fn run(self, _: A, x: USlice<'_, 1>, y: USlice<'_, 1>) -> MathematicalResult<u32> {
        bitvector_op::<Self, Unsigned>(x, y)
    }
}

/// An implementation for inner products that uses scalar indexing for the implementation.
macro_rules! impl_fallback_ip {
    (($N:literal, $M:literal)) => {
        /// Compute the inner product between `x` and `y`.
        ///
        /// Returns an error if the arguments have different lengths.
        ///
        /// # Performance
        ///
        /// This function uses a generic implementation and therefore is not very fast.
        impl Target2<diskann_wide::arch::Scalar, MathematicalResult<u32>, USlice<'_, $N>, USlice<'_, $M>> for InnerProduct {
            #[inline(never)]
            fn run(
                self,
                _: diskann_wide::arch::Scalar,
                x: USlice<'_, $N>,
                y: USlice<'_, $M>
            ) -> MathematicalResult<u32> {
                let len = check_lengths!(x, y)?;

                let mut accum: u32 = 0;
                for i in 0..len {
                    // SAFETY: `i` is guaranteed to be less than `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as u32;
                    // SAFETY: `i` is guaranteed to be less than `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as u32;
                    accum += ix * iy;
                }
                Ok(MV::new(accum))
            }
        }
    };
    ($N:literal) => {
        impl_fallback_ip!(($N, $N));
    };
    ($($args:tt),+ $(,)?) => {
        $(impl_fallback_ip!($args);)+
    };
}

impl_fallback_ip!(7, 6, 5, 4, 3, 2, (8, 4), (8, 2), (8, 1));

#[cfg(target_arch = "x86_64")]
retarget!(diskann_wide::arch::x86_64::V3, InnerProduct, 7, 6, 5, 3);

#[cfg(target_arch = "x86_64")]
retarget!(
    diskann_wide::arch::x86_64::V4,
    InnerProduct,
    7,
    6,
    5,
    3,
    (8, 4),
    (8, 2),
    (8, 1)
);

dispatch_pure!(
    InnerProduct,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    (8, 8),
    (8, 4),
    (8, 2),
    (8, 1)
);

//////////////////////////////////////////
// Heterogeneous USlice<8> × USlice<M> //
/////////////////////////////////////////
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 8>, USlice<'_, 4>>
    for InnerProduct
{
    /// Computes the inner product of 8-bit unsigned × 4-bit unsigned vectors using V3 intrinsics.
    ///
    /// # Strategy
    ///
    /// Unpack each 16-byte chunk of `y` into 32 nibble values via [`unpack_half_bytes`],
    /// then multiply with the corresponding 32 bytes of `x` using `_mm256_maddubs_epi16`
    /// (u8 × u8 → i16, pairwise horizontal add).
    ///
    /// The main loop is 4× unrolled: four i16 products are summed in i16 before a single
    /// `_mm256_madd_epi16(…, 1)` widens to i32. This is safe because
    /// 4 × (255 × 15 × 2) = 30_600 < i16::MAX.
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 8>,
        y: USlice<'_, 4>,
    ) -> MathematicalResult<u32> {
        use std::arch::x86_64::_mm256_maddubs_epi16;

        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u8s_16 = <diskann_wide::arch::x86_64::V3>::u8x16);
        diskann_wide::alias!(u8s_32 = <diskann_wide::arch::x86_64::V3>::u8x32);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px: *const u8 = x.as_ptr();
        let py: *const u8 = y.as_ptr();

        let mut i: usize = 0;
        let mut s: u32 = 0;

        #[inline(always)]
        fn unpack_half(input: u8s_16) -> u8s_32 {
            let combined = diskann_wide::LoHi::new(input, input >> 4).zip::<u8s_32>();
            combined & u8s_32::splat(input.arch(), (1u8 << 4) - 1)
        }

        // Each block processes 32 elements: 32 bytes from x, 16 packed bytes from y.
        let blocks = len / 32;
        if blocks > 0 {
            let mut acc = i32s::default(arch);

            let products = |x: u8s_32, y: u8s_32| -> i16s {
                // SAFETY: `arch` is V3 (AVX2), which provides `_mm256_maddubs_epi16`.
                i16s::from_underlying(arch, unsafe {
                    _mm256_maddubs_epi16(x.to_underlying(), y.to_underlying())
                })
            };

            let ones = i16s::splat(arch, 1);

            // Main loop: 4× unrolled, processing 128 elements per iteration.
            //
            // `products` returns i16 lanes, each at most 255 × 15 × 2 = 7_650.
            // We sum 4 such values in i16 before widening to i32:
            //   4 × 7_650 = 30_600 < i16::MAX (32_767)
            while i + 4 <= blocks {
                // Block 0
                // SAFETY: `i + 4 <= blocks` guarantees at least 4 full blocks remain.
                // Each block needs 32 bytes from `px` and 16 bytes from `py`.
                let vx = unsafe { u8s_32::load_simd(arch, px.add(32 * i)) };
                // SAFETY: `i + 4 <= blocks` guarantees 16 bytes readable at `py.add(16 * i)`.
                let vy = unsafe { u8s_16::load_simd(arch, py.add(16 * i)) };
                let m0 = products(vx, unpack_half(vy));

                // Block 1
                // SAFETY: `i + 1 < i + 4 <= blocks`.
                let vx = unsafe { u8s_32::load_simd(arch, px.add(32 * (i + 1))) };
                // SAFETY: same bound; 16 bytes readable at `py.add(16 * (i + 1))`.
                let vy = unsafe { u8s_16::load_simd(arch, py.add(16 * (i + 1))) };
                let m1 = products(vx, unpack_half(vy));

                // Block 2
                // SAFETY: `i + 2 < i + 4 <= blocks`.
                let vx = unsafe { u8s_32::load_simd(arch, px.add(32 * (i + 2))) };
                // SAFETY: same bound; 16 bytes readable at `py.add(16 * (i + 2))`.
                let vy = unsafe { u8s_16::load_simd(arch, py.add(16 * (i + 2))) };
                let m2 = products(vx, unpack_half(vy));

                // Block 3
                // SAFETY: `i + 3 < i + 4 <= blocks`.
                let vx = unsafe { u8s_32::load_simd(arch, px.add(32 * (i + 3))) };
                // SAFETY: same bound; 16 bytes readable at `py.add(16 * (i + 3))`.
                let vy = unsafe { u8s_16::load_simd(arch, py.add(16 * (i + 3))) };
                let m3 = products(vx, unpack_half(vy));

                acc = acc.dot_simd(m0 + m1 + m2 + m3, ones);
                i += 4;
            }

            // Drain remaining full 32-element blocks (0..3 iterations).
            while i < blocks {
                // SAFETY: `i < blocks` guarantees 32 bytes from `px` and 16 bytes
                // from `py` are readable at this offset.
                let vx = unsafe { u8s_32::load_simd(arch, px.add(32 * i)) };
                // SAFETY: `i < blocks` guarantees 16 bytes readable at `py.add(16 * i)`.
                let vy = unsafe { u8s_16::load_simd(arch, py.add(16 * i)) };
                acc = acc.dot_simd(products(vx, unpack_half(vy)), ones);
                i += 1;
            }

            s = acc.sum_tree() as u32;
        }

        // Convert block count to element index.
        i *= 32;

        // Scalar fallback for the remaining < 32 elements.
        if i != len {
            #[inline(never)]
            fn fallback(x: USlice<'_, 8>, y: USlice<'_, 4>, from: usize) -> u32 {
                let mut s: u32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is bounded by `x.len()`.
                    let ix = unsafe { x.get_unchecked(i) } as u32;
                    // SAFETY: `i` is bounded by `x.len()` which equals `y.len()`.
                    let iy = unsafe { y.get_unchecked(i) } as u32;
                    s += ix * iy;
                }
                s
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 8>, USlice<'_, 2>>
    for InnerProduct
{
    /// Computes the inner product of 8-bit unsigned × 2-bit unsigned vectors using AVX2.
    ///
    /// # Strategy
    ///
    /// Unpack each 16-byte chunk of `y` into 64 crumb values via a two-level cascade:
    /// first [`unpack_half_bytes`] splits bytes into nibbles, then a second pass splits
    /// nibbles into crumbs (masked with `0x03`). Each unpacked half is paired with 32
    /// bytes of `x` and multiplied via `_mm256_maddubs_epi16`.
    ///
    /// The main loop is 4× unrolled: eight i16 products (4 blocks × 2 halves) are summed
    /// in i16 before a single `_mm256_madd_epi16(…, 1)` widens to i32. This is safe
    /// because 8 × (255 × 3 × 2) = 12_240 < i16::MAX.
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 8>,
        y: USlice<'_, 2>,
    ) -> MathematicalResult<u32> {
        use diskann_wide::SplitJoin;
        use std::arch::x86_64::_mm256_maddubs_epi16;

        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u8s_16 = <diskann_wide::arch::x86_64::V3>::u8x16);
        diskann_wide::alias!(u8s_32 = <diskann_wide::arch::x86_64::V3>::u8x32);
        diskann_wide::alias!(i16s = <diskann_wide::arch::x86_64::V3>::i16x16);

        let px: *const u8 = x.as_ptr();
        let py: *const u8 = y.as_ptr();

        let mut i: usize = 0;
        let mut s: u32 = 0;

        // Each block processes 64 elements: 64 bytes from x, 16 packed bytes from y.
        let blocks = len / 64;
        if blocks > 0 {
            let mut acc = i32s::default(arch);

            let products = |x: u8s_32, y: u8s_32| -> i16s {
                // SAFETY: `arch` is V3 (AVX2), which provides `_mm256_maddubs_epi16`.
                i16s::from_underlying(arch, unsafe {
                    _mm256_maddubs_epi16(x.to_underlying(), y.to_underlying())
                })
            };

            #[inline(always)]
            fn unpack_sub<const N: u8>(input: u8s_16) -> u8s_32 {
                let combined = diskann_wide::LoHi::new(input, input >> N).zip::<u8s_32>();
                combined & u8s_32::splat(input.arch(), (1u8 << N) - 1)
            }

            let unpack_crumbs = |x: u8s_16| -> (u8s_32, u8s_32) {
                // Level 1: nibble split → 32 × 4-bit values in a u8x32.
                let nibbles = unpack_sub::<4>(x);

                // Split the u8x32 into two u8x16 halves (lo = elements 0..15,
                // hi = elements 16..31), then apply Level 2 crumb split to each.
                let diskann_wide::LoHi { lo, hi } = nibbles.split();
                let lower = unpack_sub::<2>(lo);
                let upper = unpack_sub::<2>(hi);

                (lower, upper)
            };

            let ones = i16s::splat(arch, 1);

            // Main loop: 4× unrolled, processing 256 elements per iteration.
            //
            // `products` returns i16 lanes, each at most 255 × 3 × 2 = 1_530.
            // Each iteration produces 4 blocks × 2 products = 8 values.
            // We sum all 8 in i16 before widening to i32:
            //   8 × 1_530 = 12_240 < i16::MAX (32_767)
            while i + 4 <= blocks {
                // Block 0
                // SAFETY: `i + 4 <= blocks` guarantees at least 4 full blocks remain.
                // Each block needs 64 bytes from `px` and 16 bytes from `py`.
                let (vx0, vx1, (vy0, vy1)) = unsafe {
                    (
                        u8s_32::load_simd(arch, px.add(64 * i)),
                        u8s_32::load_simd(arch, px.add(64 * i + 32)),
                        unpack_crumbs(u8s_16::load_simd(arch, py.add(16 * i))),
                    )
                };
                let m0a = products(vx0, vy0);
                let m0b = products(vx1, vy1);

                // Block 1
                // SAFETY: `i + 1 < i + 4 <= blocks`.
                let (vx0, vx1, (vy0, vy1)) = unsafe {
                    (
                        u8s_32::load_simd(arch, px.add(64 * (i + 1))),
                        u8s_32::load_simd(arch, px.add(64 * (i + 1) + 32)),
                        unpack_crumbs(u8s_16::load_simd(arch, py.add(16 * (i + 1)))),
                    )
                };
                let m1a = products(vx0, vy0);
                let m1b = products(vx1, vy1);

                // Block 2
                // SAFETY: `i + 2 < i + 4 <= blocks`.
                let (vx0, vx1, (vy0, vy1)) = unsafe {
                    (
                        u8s_32::load_simd(arch, px.add(64 * (i + 2))),
                        u8s_32::load_simd(arch, px.add(64 * (i + 2) + 32)),
                        unpack_crumbs(u8s_16::load_simd(arch, py.add(16 * (i + 2)))),
                    )
                };
                let m2a = products(vx0, vy0);
                let m2b = products(vx1, vy1);

                // Block 3
                // SAFETY: `i + 3 < i + 4 <= blocks`.
                let (vx0, vx1, (vy0, vy1)) = unsafe {
                    (
                        u8s_32::load_simd(arch, px.add(64 * (i + 3))),
                        u8s_32::load_simd(arch, px.add(64 * (i + 3) + 32)),
                        unpack_crumbs(u8s_16::load_simd(arch, py.add(16 * (i + 3)))),
                    )
                };
                let m3a = products(vx0, vy0);
                let m3b = products(vx1, vy1);

                acc = acc.dot_simd((m0a + m0b + m1a + m1b) + (m2a + m2b + m3a + m3b), ones);
                i += 4;
            }

            // Drain remaining full 64-element blocks (0..3 iterations).
            while i < blocks {
                // SAFETY: `i < blocks` guarantees 64 bytes from `px` and 16 bytes
                // from `py` are readable at this offset.
                let (vx0, vx1, (vy0, vy1)) = unsafe {
                    (
                        u8s_32::load_simd(arch, px.add(64 * i)),
                        u8s_32::load_simd(arch, px.add(64 * i + 32)),
                        unpack_crumbs(u8s_16::load_simd(arch, py.add(16 * i))),
                    )
                };
                acc = acc.dot_simd(products(vx0, vy0) + products(vx1, vy1), ones);
                i += 1;
            }

            s = acc.sum_tree() as u32;
        }

        // Convert block count to element index.
        i *= 64;

        // Scalar fallback for the remaining < 64 elements.
        if i != len {
            #[inline(never)]
            fn fallback(x: USlice<'_, 8>, y: USlice<'_, 2>, from: usize) -> u32 {
                let mut s: u32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is in `from..x.len()`, which equals `y.len()`.
                    let (ix, iy) =
                        unsafe { (x.get_unchecked(i) as u32, y.get_unchecked(i) as u32) };
                    s += ix * iy;
                }
                s
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<u32>, USlice<'_, 8>, USlice<'_, 1>>
    for InnerProduct
{
    /// Computes the inner product of 8-bit unsigned × 1-bit unsigned vectors using V3 intrinsics.
    ///
    /// For each 32-element block we load 32 bytes from `x` and 4 bytes (32 bits) from `y`.
    /// ANDing the data with the mask created from 4 bytes from `y` zeroes unselected lanes.
    /// Finally, `_mm256_sad_epu8` horizontally sums the masked bytes in groups of 8.
    ///
    /// The main loop is 4× unrolled, processing 128 elements per iteration.
    ///
    /// ## Overflow
    ///
    /// Each `sad` output lane holds at most `8 × 255 = 2_040`. Accumulated across `d/32`
    /// blocks, the per-lane max is `(d/32) × 2_040`. At dim = 3072: `96 × 2_040 = 195_840`,
    /// well within i32 range.
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: USlice<'_, 8>,
        y: USlice<'_, 1>,
    ) -> MathematicalResult<u32> {
        use diskann_wide::{FromInt, SIMDMask};
        use std::arch::x86_64::_mm256_sad_epu8;

        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);
        diskann_wide::alias!(u8s_32 = <diskann_wide::arch::x86_64::V3>::u8x32);

        type Mask32 = diskann_wide::BitMask<32, diskann_wide::arch::x86_64::V3>;
        type Mask8x32 = diskann_wide::arch::x86_64::v3::masks::mask8x32;

        let px: *const u8 = x.as_ptr();
        let py: *const u8 = y.as_ptr();

        let mut i: usize = 0;
        let mut s: u32 = 0;

        // Each block processes 32 elements: 32 bytes from x, 4 bytes (32 bits) from y.
        let blocks = len / 32;
        if blocks > 0 {
            let mut acc = i32s::default(arch);
            let zero = u8s_32::default(arch);

            // Expand 32 bits of `y` to a byte mask, AND with `x`, and horizontally sum
            // the masked bytes via `_mm256_sad_epu8`.
            //
            // Returns an `i32x8` where lanes 0, 2, 4, 6 hold group sums (each ≤ 2040)
            // and lanes 1, 3, 5, 7 are zero.
            let masked_sad = |vx: u8s_32, bits: u32| -> i32s {
                // Expand 32-bit mask → 32 bytes of 0xFF/0x00.
                let byte_mask: Mask8x32 = Mask32::from_int(arch, bits).into();

                // AND data with mask: zeroes lanes where the bit is 0.
                let masked = vx & u8s_32::from_underlying(arch, byte_mask.to_underlying());

                // Horizontal byte sum in groups of 8 → 4 × u64 partial sums.
                // SAFETY: `arch` is V3 (AVX2), `_mm256_sad_epu8` is available.
                i32s::from_underlying(arch, unsafe {
                    _mm256_sad_epu8(masked.to_underlying(), zero.to_underlying())
                })
            };

            // Main loop: 4× unrolled, processing 128 elements per iteration.
            while i + 4 <= blocks {
                // SAFETY: `i + 4 <= blocks` guarantees at least 4 full blocks remain.
                // Each block needs 32 bytes from `px` and 4 bytes from `py`.
                let s0 = unsafe {
                    let vx = u8s_32::load_simd(arch, px.add(32 * i));
                    let bits = (py.add(4 * i) as *const u32).read_unaligned();
                    masked_sad(vx, bits)
                };

                // SAFETY: `i + 1 < i + 4 <= blocks`.
                let s1 = unsafe {
                    let vx = u8s_32::load_simd(arch, px.add(32 * (i + 1)));
                    let bits = (py.add(4 * (i + 1)) as *const u32).read_unaligned();
                    masked_sad(vx, bits)
                };

                // SAFETY: `i + 2 < i + 4 <= blocks`.
                let s2 = unsafe {
                    let vx = u8s_32::load_simd(arch, px.add(32 * (i + 2)));
                    let bits = (py.add(4 * (i + 2)) as *const u32).read_unaligned();
                    masked_sad(vx, bits)
                };

                // SAFETY: `i + 3 < i + 4 <= blocks`.
                let s3 = unsafe {
                    let vx = u8s_32::load_simd(arch, px.add(32 * (i + 3)));
                    let bits = (py.add(4 * (i + 3)) as *const u32).read_unaligned();
                    masked_sad(vx, bits)
                };

                acc = acc + s0 + s1 + s2 + s3;
                i += 4;
            }

            // Drain remaining full 32-element blocks (0..3 iterations).
            while i < blocks {
                // SAFETY: `i < blocks` guarantees 32 bytes from `px` and 4 bytes from `py`
                // are readable at this offset.
                let si = unsafe {
                    let vx = u8s_32::load_simd(arch, px.add(32 * i));
                    let bits = (py.add(4 * i) as *const u32).read_unaligned();
                    masked_sad(vx, bits)
                };
                acc = acc + si;
                i += 1;
            }

            s = acc.sum_tree() as u32;
        }

        // Convert block count to element index.
        i *= 32;

        // Scalar fallback for the remaining < 32 elements.
        if i != len {
            #[inline(never)]
            fn fallback(x: USlice<'_, 8>, y: USlice<'_, 1>, from: usize) -> u32 {
                let mut s: u32 = 0;
                for i in from..x.len() {
                    // SAFETY: `i` is in `from..x.len()`, which equals `y.len()`.
                    let (ix, iy) =
                        unsafe { (x.get_unchecked(i) as u32, y.get_unchecked(i) as u32) };
                    s += ix * iy;
                }
                s
            }
            s += fallback(x, y, i);
        }

        Ok(MV::new(s))
    }
}

#[cfg(target_arch = "aarch64")]
retarget!(
    diskann_wide::arch::aarch64::Neon,
    InnerProduct,
    7,
    6,
    5,
    4,
    3,
    2,
    (8, 4),
    (8, 2),
    (8, 1)
);

//////////////////
// BitTranspose //
//////////////////

/// The strategy is to compute the inner product `<x, y>` by decomposing the problem into
/// groups of 64-dimensions.
///
/// For each group, we load the 64-bits of `y` into a word `bits`. And the four 64-bit words
/// of the group in `x` in `b0`, `b1`, b2`, and `b3`.
///
/// Note that bit `i` in `b0` is bit-0 of the `i`-th value in ths group. Likewise, bit `i`
/// in `b1` is bit-1 of the same word.
///
/// This means that we can compute the partial inner product for this group as
/// ```math
/// (bits & b0).count_ones()                // Contribution of bit 0
///     + 2 * (bits & b1).count_ones()      // Contribution of bit 1
///     + 4 * (bits & b2).count_ones()      // Contribution of bit 2
///     + 8 * (bits & b3).count_ones()      // Contribution of bit 3
/// ```
/// We process as many full groups as we can.
///
/// To handle the remainder, we need to be careful about acessing `y` because `BitSlice`
/// only guarantees the validity of reads at the byte level. That is - we cannot assume that
/// a full 64-bit read is valid.
///
/// The bit-tranposed `x`, on the other hand, guarantees allocations in blocks of
/// 4 * 64-bits, so it can be treated as normal.
impl<A> Target2<A, MathematicalResult<u32>, USlice<'_, 4, BitTranspose>, USlice<'_, 1, Dense>>
    for InnerProduct
where
    A: Architecture,
{
    #[inline(always)]
    fn run(
        self,
        _: A,
        x: USlice<'_, 4, BitTranspose>,
        y: USlice<'_, 1, Dense>,
    ) -> MathematicalResult<u32> {
        let len = check_lengths!(x, y)?;

        // We work in blocks of 64 element.
        //
        // The `BitTranspose` guarantees read are valid in blocks of 64 elements (32 byte).
        // However, the `Dense` representation only pads to bytes.
        // Our strategy for dealing with fewer than 64 remaining elements is to reconstruct
        // a 64-bit integer from bytes.
        let px: *const u64 = x.as_ptr().cast();
        let py: *const u64 = y.as_ptr().cast();

        let mut i = 0;
        let mut s: u32 = 0;

        let blocks = len / 64;
        while i < blocks {
            // SAFETY: `y` is valid for at least `blocks` 64-bit reads and `i < blocks`.
            let bits = unsafe { py.add(i).read_unaligned() };

            // SAFETY: The layout for `x` is grouped into 32-byte blocks. We've ensured that
            // the lengths of the two vectors are the same, so we know that `x` has at least
            // `blocks` such regions.
            //
            // This loads the first 64-bits of block `i` where `i < blocks`.
            let b0 = unsafe { px.add(4 * i).read_unaligned() };
            s += (bits & b0).count_ones();

            // SAFETY: This loads the second 64-bit word of block `i`.
            let b1 = unsafe { px.add(4 * i + 1).read_unaligned() };
            s += (bits & b1).count_ones() << 1;

            // SAFETY: This loads the third 64-bit word of block `i`.
            let b2 = unsafe { px.add(4 * i + 2).read_unaligned() };
            s += (bits & b2).count_ones() << 2;

            // SAFETY: This loads the fourth 64-bit word of block `i`.
            let b3 = unsafe { px.add(4 * i + 3).read_unaligned() };
            s += (bits & b3).count_ones() << 3;

            i += 1;
        }

        // If the input length is a multiple of 64 - then we're done.
        if 64 * i == len {
            return Ok(MV::new(s));
        }

        // Convert blocks to bytes.
        let k = i * 8;

        // Unpack the last elements from the bit-vector.
        //
        // SAFETY: The length of the 1-bit BitSlice is `ceil(len / 8)`. This computation
        // effectively computes `ceil((64 * floor(len / 64)) / 8)`, which is less.
        let py = unsafe { py.cast::<u8>().add(k) };
        let bytes_remaining = y.bytes() - k;
        let mut bits: u64 = 0;

        // Code - generation: Applying `min(8)` gives a constant upper-bound to the
        // compiler, allowing better code-generation.
        for j in 0..bytes_remaining.min(8) {
            // SAFETY: Starting at `py`, there are `bytes_remaining` valid bytes. This
            // accesses all of them.
            bits += (unsafe { py.add(j).read() } as u64) << (8 * j);
        }

        // Because the upper-bits of the last loaded byte can contain indeterminate bits,
        // we must mask out all out-of-bounds bits.
        bits &= (0x01u64 << (len - (64 * i))) - 1;

        // Combine with the remainders.
        //
        // SAFETY: The `BitTranspose` permutation always allocates in granularies of blocks.
        // This loads the first 64-bit word of the last block.
        let b0 = unsafe { px.add(4 * i).read_unaligned() };
        s += (bits & b0).count_ones();

        // SAFETY: This loads the second 64-bit word of the last block.
        let b1 = unsafe { px.add(4 * i + 1).read_unaligned() };
        s += (bits & b1).count_ones() << 1;

        // SAFETY: This loads the third 64-bit word of the last block.
        let b2 = unsafe { px.add(4 * i + 2).read_unaligned() };
        s += (bits & b2).count_ones() << 2;

        // SAFETY: This loads the fourth 64-bit word of the last block.
        let b3 = unsafe { px.add(4 * i + 3).read_unaligned() };
        s += (bits & b3).count_ones() << 3;

        Ok(MV::new(s))
    }
}

impl
    PureDistanceFunction<USlice<'_, 4, BitTranspose>, USlice<'_, 1, Dense>, MathematicalResult<u32>>
    for InnerProduct
{
    fn evaluate(
        x: USlice<'_, 4, BitTranspose>,
        y: USlice<'_, 1, Dense>,
    ) -> MathematicalResult<u32> {
        (diskann_wide::ARCH).run2(Self, x, y)
    }
}

////////////////////
// Full Precision //
////////////////////

/// The main trick here is avoiding explicit conversion from 1 bit integers to 32-bit
/// floating-point numbers by using `_mm256_permutevar_ps`, which performs a shuffle on two
/// independent 128-bit lanes of `f32` values in a register `A` using the lower 2-bits of
/// each 32-bit integer in a register `B`.
///
/// Importantly, this instruction only takes a single cycle and we can avoid any kind of
/// masking. Going the route of conversion would require and `AND` operation to isolate
/// bottom bits and a somewhat lengthy 32-bit integer to `f32` conversion instruction.
///
/// The overall strategy broadcasts a 32-bit integer (consisting of 32, 1-bit values) across
/// 8 lanes into a register `A`.
///
/// Each lane is then shifted by a different amount so:
///
/// * Lane 0 has value 0 as its least significant bit (LSB)
/// * Lane 1 has value 1 as its LSB.
/// * Lane 2 has value 2 as its LSB.
/// * etc.
///
/// These LSB's are used to power the shuffle function to convert to `f32` values (either
/// 0.0 or 1.0) and we can FMA as needed.
///
/// To process the next group of 8 values, we shift all lanes in `A` by 8-bits so lane 0
/// has value 8 as its LSB, lane 1 has value 9 etc.
///
/// A total of three shifts are applied to extract all 32 1-bit value as `f32` in order.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<f32>, &[f32], USlice<'_, 1>>
    for InnerProduct
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: &[f32],
        y: USlice<'_, 1>,
    ) -> MathematicalResult<f32> {
        let len = check_lengths!(x, y)?;

        use std::arch::x86_64::*;

        diskann_wide::alias!(f32s = <diskann_wide::arch::x86_64::V3>::f32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);

        // Replicate 0s and 1s so we effectively get a shuffle that only depends on the
        // bottom bit (instead of the lowest 2).
        let values = f32s::from_array(arch, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        // Shifts required to offset each lane.
        let variable_shifts = u32s::from_array(arch, [0, 1, 2, 3, 4, 5, 6, 7]);

        let px: *const f32 = x.as_ptr();
        let py: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s = f32s::default(arch);

        let prep = |v: u32| -> u32s { u32s::splat(arch, v) >> variable_shifts };
        let to_f32 = |v: u32s| -> f32s {
            // SAFETY: The `_mm256_permutevar_ps` instruction requires the AVX extension,
            // which the presence of the `x86_64::V3` architecture guarantees is available.
            f32s::from_underlying(arch, unsafe {
                _mm256_permutevar_ps(values.to_underlying(), v.to_underlying())
            })
        };

        // Data is processed in groups of 32 elements.
        let blocks = len / 32;
        if i < blocks {
            let mut s0 = f32s::default(arch);
            let mut s1 = f32s::default(arch);

            while i < blocks {
                // SAFETY: `i < blocks` implies 32-bits are readable from this offset.
                let iy = prep(unsafe { py.add(i).read_unaligned() });

                // SAFETY: `i < blocks` implies 32 f32 values are readable beginning at `32*i`.
                let ix0 = unsafe { f32s::load_simd(arch, px.add(32 * i)) };
                // SAFETY: See above.
                let ix1 = unsafe { f32s::load_simd(arch, px.add(32 * i + 8)) };
                // SAFETY: See above.
                let ix2 = unsafe { f32s::load_simd(arch, px.add(32 * i + 16)) };
                // SAFETY: See above.
                let ix3 = unsafe { f32s::load_simd(arch, px.add(32 * i + 24)) };

                s0 = ix0.mul_add_simd(to_f32(iy), s0);
                s1 = ix1.mul_add_simd(to_f32(iy >> 8), s1);
                s0 = ix2.mul_add_simd(to_f32(iy >> 16), s0);
                s1 = ix3.mul_add_simd(to_f32(iy >> 24), s1);

                i += 1;
            }
            s = s0 + s1;
        }

        let remainder = len % 32;
        if remainder != 0 {
            let tail = if len % 8 == 0 { 8 } else { len % 8 };

            // SAFETY: Because `remainder != 0`, there is valid memory beginning at the
            // offset `blocks`, so this addition remains within an allocated object.
            let py = unsafe { py.add(blocks) };

            if remainder <= 8 {
                // SAFETY: Non-zero remainder implies at least one byte is readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_one(py, |iy| {
                        let iy = prep(iy);
                        let ix = f32s::load_simd_first(arch, px.add(32 * blocks), tail);
                        s = ix.mul_add_simd(to_f32(iy), s);
                    })
                }
            } else if remainder <= 16 {
                // SAFETY: At least two bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_two(py, |iy| {
                        let iy = prep(iy);
                        let ix0 = f32s::load_simd(arch, px.add(32 * blocks));
                        let ix1 = f32s::load_simd_first(arch, px.add(32 * blocks + 8), tail);
                        s = ix0.mul_add_simd(to_f32(iy), s);
                        s = ix1.mul_add_simd(to_f32(iy >> 8), s);
                    })
                }
            } else if remainder <= 24 {
                // SAFETY: At least three bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_three(py, |iy| {
                        let iy = prep(iy);

                        let ix0 = f32s::load_simd(arch, px.add(32 * blocks));
                        let ix1 = f32s::load_simd(arch, px.add(32 * blocks + 8));
                        let ix2 = f32s::load_simd_first(arch, px.add(32 * blocks + 16), tail);

                        s = ix0.mul_add_simd(to_f32(iy), s);
                        s = ix1.mul_add_simd(to_f32(iy >> 8), s);
                        s = ix2.mul_add_simd(to_f32(iy >> 16), s);
                    })
                }
            } else {
                // SAFETY: At least four bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_four(py, |iy| {
                        let iy = prep(iy);

                        let ix0 = f32s::load_simd(arch, px.add(32 * blocks));
                        let ix1 = f32s::load_simd(arch, px.add(32 * blocks + 8));
                        let ix2 = f32s::load_simd(arch, px.add(32 * blocks + 16));
                        let ix3 = f32s::load_simd_first(arch, px.add(32 * blocks + 24), tail);

                        s = ix0.mul_add_simd(to_f32(iy), s);
                        s = ix1.mul_add_simd(to_f32(iy >> 8), s);
                        s = ix2.mul_add_simd(to_f32(iy >> 16), s);
                        s = ix3.mul_add_simd(to_f32(iy >> 24), s);
                    })
                }
            }
        }

        Ok(MV::new(s.sum_tree()))
    }
}

/// The strategy used here is almost identical to that used for 1-bit distances. The main
/// difference is that now we use the full 2-bit shuffle capabilities of `_mm256_permutevar_ps`
/// and ths relatives sizes of the shifts are slightly different.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<f32>, &[f32], USlice<'_, 2>>
    for InnerProduct
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: &[f32],
        y: USlice<'_, 2>,
    ) -> MathematicalResult<f32> {
        let len = check_lengths!(x, y)?;

        use std::arch::x86_64::*;

        diskann_wide::alias!(f32s = <diskann_wide::arch::x86_64::V3>::f32x8);
        diskann_wide::alias!(u32s = <diskann_wide::arch::x86_64::V3>::u32x8);

        // This is the lookup table mapping 2-bit patterns to their equivalent `f32`
        // representation. The AVX2 shuffle only applies within each 128-bit group of the
        // full 256-bit register, so we replicate the contents.
        let values = f32s::from_array(arch, [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);

        // Shifts required to get logical dimensions shifted to the lower 2-bits of each lane.
        let variable_shifts = u32s::from_array(arch, [0, 2, 4, 6, 8, 10, 12, 14]);

        let px: *const f32 = x.as_ptr();
        let py: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s = f32s::default(arch);

        let prep = |v: u32| -> u32s { u32s::splat(arch, v) >> variable_shifts };
        let to_f32 = |v: u32s| -> f32s {
            // SAFETY: The `_mm256_permutevar_ps` instruction requires the AVX extension,
            // which the presense of the `x86_64::V3` architecture guarantees is available.
            f32s::from_underlying(arch, unsafe {
                _mm256_permutevar_ps(values.to_underlying(), v.to_underlying())
            })
        };

        let blocks = len / 16;
        if blocks != 0 {
            let mut s0 = f32s::default(arch);
            let mut s1 = f32s::default(arch);

            // Process 32 elements.
            while i + 2 <= blocks {
                // SAFETY: `i + 2 <= blocks` implies `py.add(i)` is in-bounds and readable
                // for 4 unaligned bytes.
                let iy = prep(unsafe { py.add(i).read_unaligned() });

                // SAFETY: Same logic as above, just applied to `f32` values instead of
                // packed bits.
                let (ix0, ix1) = unsafe {
                    (
                        f32s::load_simd(arch, px.add(16 * i)),
                        f32s::load_simd(arch, px.add(16 * i + 8)),
                    )
                };

                s0 = ix0.mul_add_simd(to_f32(iy), s0);
                s1 = ix1.mul_add_simd(to_f32(iy >> 16), s1);

                // SAFETY: `i + 2 <= blocks` implies `py.add(i + 1)` is in-bounds and readable
                // for 4 unaligned bytes.
                let iy = prep(unsafe { py.add(i + 1).read_unaligned() });

                // SAFETY: Same logic as above.
                let (ix0, ix1) = unsafe {
                    (
                        f32s::load_simd(arch, px.add(16 * (i + 1))),
                        f32s::load_simd(arch, px.add(16 * (i + 1) + 8)),
                    )
                };

                s0 = ix0.mul_add_simd(to_f32(iy), s0);
                s1 = ix1.mul_add_simd(to_f32(iy >> 16), s1);

                i += 2;
            }

            // Process 16 elements
            if i < blocks {
                // SAFETY: `i < blocks` implies `py.add(i)` is in-bounds and readable for
                // 4 unaligned bytes.
                let iy = prep(unsafe { py.add(i).read_unaligned() });

                // SAFETY: Same logic as above.
                let (ix0, ix1) = unsafe {
                    (
                        f32s::load_simd(arch, px.add(16 * i)),
                        f32s::load_simd(arch, px.add(16 * i + 8)),
                    )
                };

                s0 = ix0.mul_add_simd(to_f32(iy), s0);
                s1 = ix1.mul_add_simd(to_f32(iy >> 16), s1);
            }

            s = s0 + s1;
        }

        let remainder = len % 16;
        if remainder != 0 {
            let tail = if len % 8 == 0 { 8 } else { len % 8 };
            // SAFETY: Non-zero remainder implies there are readable bytes after the offset
            // `blocks`, so the addition is valid.
            let py = unsafe { py.add(blocks) };

            if remainder <= 4 {
                // SAFETY: Non-zero remainder implies at least one byte is readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_one(py, |iy| {
                        let iy = prep(iy);
                        let ix = f32s::load_simd_first(arch, px.add(16 * blocks), tail);
                        s = ix.mul_add_simd(to_f32(iy), s);
                    });
                }
            } else if remainder <= 8 {
                // SAFETY: At least two bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_two(py, |iy| {
                        let iy = prep(iy);
                        let ix = f32s::load_simd_first(arch, px.add(16 * blocks), tail);
                        s = ix.mul_add_simd(to_f32(iy), s);
                    });
                }
            } else if remainder <= 12 {
                // SAFETY: At least three bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_three(py, |iy| {
                        let iy = prep(iy);
                        let ix0 = f32s::load_simd(arch, px.add(16 * blocks));
                        let ix1 = f32s::load_simd_first(arch, px.add(16 * blocks + 8), tail);
                        s = ix0.mul_add_simd(to_f32(iy), s);
                        s = ix1.mul_add_simd(to_f32(iy >> 16), s);
                    });
                }
            } else {
                // SAFETY: At least four bytes are readable for `py`.
                // The same logic applies to the SIMD loads.
                unsafe {
                    load_four(py, |iy| {
                        let iy = prep(iy);
                        let ix0 = f32s::load_simd(arch, px.add(16 * blocks));
                        let ix1 = f32s::load_simd_first(arch, px.add(16 * blocks + 8), tail);
                        s = ix0.mul_add_simd(to_f32(iy), s);
                        s = ix1.mul_add_simd(to_f32(iy >> 16), s);
                    });
                }
            }
        }

        Ok(MV::new(s.sum_tree()))
    }
}

/// The strategy here is similar to the 1 and 2-bit strategies. However, instead of using
/// `_mm256_permutevar_ps`, we now go directly for 32-bit integer to 32-bit floating point.
///
/// This is because the shuffle intrinsic only supports 2-bit shuffles.
#[cfg(target_arch = "x86_64")]
impl Target2<diskann_wide::arch::x86_64::V3, MathematicalResult<f32>, &[f32], USlice<'_, 4>>
    for InnerProduct
{
    #[inline(always)]
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        x: &[f32],
        y: USlice<'_, 4>,
    ) -> MathematicalResult<f32> {
        let len = check_lengths!(x, y)?;

        diskann_wide::alias!(f32s = <diskann_wide::arch::x86_64::V3>::f32x8);
        diskann_wide::alias!(i32s = <diskann_wide::arch::x86_64::V3>::i32x8);

        let variable_shifts = i32s::from_array(arch, [0, 4, 8, 12, 16, 20, 24, 28]);
        let mask = i32s::splat(arch, 0x0f);

        let to_f32 = |v: u32| -> f32s {
            ((i32s::splat(arch, v as i32) >> variable_shifts) & mask).simd_cast()
        };

        let px: *const f32 = x.as_ptr();
        let py: *const u32 = y.as_ptr().cast();

        let mut i = 0;
        let mut s = f32s::default(arch);

        let blocks = len / 8;
        while i < blocks {
            // SAFETY: `i < blocks` implies that 8 `f32` values are readable from `8*i`.
            let ix = unsafe { f32s::load_simd(arch, px.add(8 * i)) };
            // SAFETY: Same logic as above - but applied to the packed bits.
            let iy = to_f32(unsafe { py.add(i).read_unaligned() });
            s = ix.mul_add_simd(iy, s);

            i += 1;
        }

        let remainder = len % 8;
        if remainder != 0 {
            let f = |iy| {
                // SAFETY: The epilogue handles at most 8 values. Since the remainder is
                // non-zero, the pointer arithmetic is in-bounds and `load_simd_first` will
                // avoid accessing the out-of-bounds elements.
                let ix = unsafe { f32s::load_simd_first(arch, px.add(8 * blocks), remainder) };
                s = ix.mul_add_simd(to_f32(iy), s);
            };

            // SAFETY: Non-zero remainder means there are readable bytes from the offset
            // `blocks`.
            let py = unsafe { py.add(blocks) };

            if remainder <= 2 {
                // SAFETY: Non-zero remainder less than 2 implies that one byte is readable.
                unsafe { load_one(py, f) };
            } else if remainder <= 4 {
                // SAFETY: At least two bytes are readable from `py`.
                unsafe { load_two(py, f) };
            } else if remainder <= 6 {
                // SAFETY: At least three bytes are readable from `py`.
                unsafe { load_three(py, f) };
            } else {
                // SAFETY: At least four bytes are readable from `py`.
                unsafe { load_four(py, f) };
            }
        }

        Ok(MV::new(s.sum_tree()))
    }
}

impl<const N: usize>
    Target2<diskann_wide::arch::Scalar, MathematicalResult<f32>, &[f32], USlice<'_, N>>
    for InnerProduct
where
    Unsigned: Representation<N>,
{
    /// A fallback implementation that uses scaler indexing to retrieve values from
    /// the corresponding `BitSlice`.
    #[inline(always)]
    fn run(
        self,
        _: diskann_wide::arch::Scalar,
        x: &[f32],
        y: USlice<'_, N>,
    ) -> MathematicalResult<f32> {
        check_lengths!(x, y)?;

        let mut s = 0.0;
        for (i, x) in x.iter().enumerate() {
            // SAFETY: We've ensured that `x.len() == y.len()`, so this access is
            // always inbounds.
            let y = unsafe { y.get_unchecked(i) } as f32;
            s += x * y;
        }

        Ok(MV::new(s))
    }
}

/// Implement `Target2` for higher architecture in terms of the scalar fallback.
macro_rules! ip_retarget {
    ($arch:path, $N:literal) => {
        impl Target2<$arch, MathematicalResult<f32>, &[f32], USlice<'_, $N>>
            for InnerProduct
        {
            #[inline(always)]
            fn run(
                self,
                arch: $arch,
                x: &[f32],
                y: USlice<'_, $N>,
            ) -> MathematicalResult<f32> {
                self.run(arch.retarget(), x, y)
            }
        }
    };
    ($arch:path, $($Ns:literal),*) => {
        $(ip_retarget!($arch, $Ns);)*
    }
}

#[cfg(target_arch = "x86_64")]
ip_retarget!(diskann_wide::arch::x86_64::V3, 3, 5, 6, 7, 8);

#[cfg(target_arch = "x86_64")]
ip_retarget!(diskann_wide::arch::x86_64::V4, 1, 2, 3, 4, 5, 6, 7, 8);

#[cfg(target_arch = "aarch64")]
ip_retarget!(diskann_wide::arch::aarch64::Neon, 1, 2, 3, 4, 5, 6, 7, 8);

/// Delegate the implementation of `PureDistanceFunction` to `diskann_wide::arch::Target2`
/// with the current architectures.
macro_rules! dispatch_full_ip {
    ($N:literal) => {
        /// Compute the inner product between `x` and `y`.
        ///
        /// Returns an error if the arguments have different lengths.
        impl PureDistanceFunction<&[f32], USlice<'_, $N>, MathematicalResult<f32>>
            for InnerProduct
        {
            fn evaluate(x: &[f32], y: USlice<'_, $N>) -> MathematicalResult<f32> {
                Self.run(ARCH, x, y)
            }
        }
    };
    ($($Ns:literal),*) => {
        $(dispatch_full_ip!($Ns);)*
    }
}

dispatch_full_ip!(1, 2, 3, 4, 5, 6, 7, 8);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fmt::Display, sync::LazyLock};

    use diskann_utils::{Reborrow, lazy_format};
    use rand::{
        Rng, SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
        seq::IndexedRandom,
    };

    use super::*;
    use crate::bits::{BoxedBitSlice, Representation, Unsigned};

    type MR = MathematicalResult<u32>;

    #[inline(always)]
    fn should_check_this_dimension(dim: usize) -> bool {
        if cfg!(miri) {
            return dim.is_power_of_two()
                || (dim > 1 && (dim - 1).is_power_of_two())
                || (dim < 64 && (dim % 8 == 7));
        }

        true
    }

    /////////////////////////
    // Unsigned Bit Slices //
    /////////////////////////

    // This test works by generating random integer codes for the compressed vectors,
    // then uses the functions implemented in `vector` to compute the expected result of
    // the computation in "full precision integer space".
    //
    // We verify that the exact same results are returned by each computation.
    fn test_bitslice_distances<const NBITS: usize, R>(
        dim_max: usize,
        trials_per_dim: usize,
        evaluate_l2: &dyn Fn(USlice<'_, NBITS>, USlice<'_, NBITS>) -> MR,
        evaluate_ip: &dyn Fn(USlice<'_, NBITS>, USlice<'_, NBITS>) -> MR,
        context: &str,
        rng: &mut R,
    ) where
        Unsigned: Representation<NBITS>,
        R: Rng,
    {
        let domain = Unsigned::domain_const::<NBITS>();
        let min: i64 = *domain.start();
        let max: i64 = *domain.end();

        let dist = Uniform::new_inclusive(min, max).unwrap();

        for dim in 0..dim_max {
            if !should_check_this_dimension(dim) {
                continue;
            }

            let mut x_reference: Vec<u8> = vec![0; dim];
            let mut y_reference: Vec<u8> = vec![0; dim];

            let mut x = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(dim);
            let mut y = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(dim);

            for trial in 0..trials_per_dim {
                x_reference
                    .iter_mut()
                    .for_each(|i| *i = dist.sample(rng).try_into().unwrap());
                y_reference
                    .iter_mut()
                    .for_each(|i| *i = dist.sample(rng).try_into().unwrap());

                // Fill the input slices with 1's so we can catch situations where we don't
                // correctly handle odd remaining elements.
                x.as_mut_slice().fill(u8::MAX);
                y.as_mut_slice().fill(u8::MAX);

                for i in 0..dim {
                    x.set(i, x_reference[i].into()).unwrap();
                    y.set(i, y_reference[i].into()).unwrap();
                }

                // Check L2
                let expected: MV<f32> =
                    diskann_vector::distance::SquaredL2::evaluate(&*x_reference, &*y_reference);

                let got = evaluate_l2(x.reborrow(), y.reborrow()).unwrap();

                // Integer computations should be exact.
                assert_eq!(
                    expected.into_inner(),
                    got.into_inner() as f32,
                    "failed SquaredL2 for NBITS = {}, dim = {}, trial = {} -- context {}",
                    NBITS,
                    dim,
                    trial,
                    context,
                );

                // Check IP
                let expected: MV<f32> =
                    diskann_vector::distance::InnerProduct::evaluate(&*x_reference, &*y_reference);

                let got = evaluate_ip(x.reborrow(), y.reborrow()).unwrap();

                // Integer computations should be exact.
                assert_eq!(
                    expected.into_inner(),
                    got.into_inner() as f32,
                    "faild InnerProduct for NBITS = {}, dim = {}, trial = {} -- context {}",
                    NBITS,
                    dim,
                    trial,
                    context,
                );
            }
        }

        // Test that we correctly return error types for length mismatches.
        let x = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(10);
        let y = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(11);

        assert!(
            evaluate_l2(x.reborrow(), y.reborrow()).is_err(),
            "context: {}",
            context
        );
        assert!(
            evaluate_l2(y.reborrow(), x.reborrow()).is_err(),
            "context: {}",
            context
        );

        assert!(
            evaluate_ip(x.reborrow(), y.reborrow()).is_err(),
            "context: {}",
            context
        );
        assert!(
            evaluate_ip(y.reborrow(), x.reborrow()).is_err(),
            "context: {}",
            context
        );
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const MAX_DIM: usize = 132;
            const TRIALS_PER_DIM: usize = 1;
        } else {
            const MAX_DIM: usize = 256;
            const TRIALS_PER_DIM: usize = 20;
        }
    }

    // For the bit-slice kernels, we want to use different maximum dimensions for the distance
    // test depending on the implementation of the kernel, and whether or not we are running
    // under Miri.
    //
    // For implementations that use the scalar fallback, we need not set very high bounds
    // (particularly when running under miri) because the implementations are quite simple.
    //
    // However, some SIMD kernels (especially for the lower bit widths), require higher bounds
    // to trigger all possible corner cases.
    static BITSLICE_TEST_BOUNDS: LazyLock<HashMap<Key, Bounds>> = LazyLock::new(|| {
        use ArchKey::{Neon, Scalar, X86_64_V3, X86_64_V4};
        [
            (Key::new(1, Scalar), Bounds::new(64, 64)),
            (Key::new(1, X86_64_V3), Bounds::new(256, 256)),
            (Key::new(1, X86_64_V4), Bounds::new(256, 256)),
            (Key::new(1, Neon), Bounds::new(64, 64)),
            (Key::new(2, Scalar), Bounds::new(64, 64)),
            // Need a higher miri-amount due to the larget block size
            (Key::new(2, X86_64_V3), Bounds::new(512, 300)),
            (Key::new(2, X86_64_V4), Bounds::new(768, 600)), // main loop processes 256 items
            (Key::new(2, Neon), Bounds::new(64, 64)),
            (Key::new(3, Scalar), Bounds::new(64, 64)),
            (Key::new(3, X86_64_V3), Bounds::new(256, 96)),
            (Key::new(3, X86_64_V4), Bounds::new(256, 96)),
            (Key::new(3, Neon), Bounds::new(64, 64)),
            (Key::new(4, Scalar), Bounds::new(64, 64)),
            // Need a higher miri-amount due to the larget block size
            (Key::new(4, X86_64_V3), Bounds::new(256, 150)),
            (Key::new(4, X86_64_V4), Bounds::new(256, 150)),
            (Key::new(4, Neon), Bounds::new(64, 64)),
            (Key::new(5, Scalar), Bounds::new(64, 64)),
            (Key::new(5, X86_64_V3), Bounds::new(256, 96)),
            (Key::new(5, X86_64_V4), Bounds::new(256, 96)),
            (Key::new(5, Neon), Bounds::new(64, 64)),
            (Key::new(6, Scalar), Bounds::new(64, 64)),
            (Key::new(6, X86_64_V3), Bounds::new(256, 96)),
            (Key::new(6, X86_64_V4), Bounds::new(256, 96)),
            (Key::new(6, Neon), Bounds::new(64, 64)),
            (Key::new(7, Scalar), Bounds::new(64, 64)),
            (Key::new(7, X86_64_V3), Bounds::new(256, 96)),
            (Key::new(7, X86_64_V4), Bounds::new(256, 96)),
            (Key::new(7, Neon), Bounds::new(64, 64)),
            (Key::new(8, Scalar), Bounds::new(64, 64)),
            (Key::new(8, X86_64_V3), Bounds::new(256, 96)),
            (Key::new(8, X86_64_V4), Bounds::new(256, 96)),
            (Key::new(8, Neon), Bounds::new(64, 64)),
        ]
        .into_iter()
        .collect()
    });

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum ArchKey {
        Scalar,
        #[expect(non_camel_case_types)]
        X86_64_V3,
        #[expect(non_camel_case_types)]
        X86_64_V4,
        Neon,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Key {
        nbits: usize,
        arch: ArchKey,
    }

    impl Key {
        fn new(nbits: usize, arch: ArchKey) -> Self {
            Self { nbits, arch }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct Bounds {
        standard: usize,
        miri: usize,
    }

    impl Bounds {
        fn new(standard: usize, miri: usize) -> Self {
            Self { standard, miri }
        }

        fn get(&self) -> usize {
            if cfg!(miri) { self.miri } else { self.standard }
        }
    }

    macro_rules! test_bitslice {
        ($name:ident, $nbits:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);

                let max_dim = BITSLICE_TEST_BOUNDS[&Key::new($nbits, ArchKey::Scalar)].get();

                test_bitslice_distances::<$nbits, _>(
                    max_dim,
                    TRIALS_PER_DIM,
                    &|x, y| SquaredL2::evaluate(x, y),
                    &|x, y| InnerProduct::evaluate(x, y),
                    "pure distance function",
                    &mut rng,
                );

                test_bitslice_distances::<$nbits, _>(
                    max_dim,
                    TRIALS_PER_DIM,
                    &|x, y| diskann_wide::arch::Scalar::new().run2(SquaredL2, x, y),
                    &|x, y| diskann_wide::arch::Scalar::new().run2(InnerProduct, x, y),
                    "scalar arch",
                    &mut rng,
                );

                // Architecture Specific.
                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                    let max_dim = BITSLICE_TEST_BOUNDS[&Key::new($nbits, ArchKey::X86_64_V3)].get();
                    test_bitslice_distances::<$nbits, _>(
                        max_dim,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(SquaredL2, x, y),
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "x86-64-v3",
                        &mut rng,
                    );
                }

                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                    let max_dim = BITSLICE_TEST_BOUNDS[&Key::new($nbits, ArchKey::X86_64_V4)].get();
                    test_bitslice_distances::<$nbits, _>(
                        max_dim,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(SquaredL2, x, y),
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "x86-64-v4",
                        &mut rng,
                    );
                }

                #[cfg(target_arch = "aarch64")]
                if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                    let max_dim = BITSLICE_TEST_BOUNDS[&Key::new($nbits, ArchKey::Neon)].get();
                    test_bitslice_distances::<$nbits, _>(
                        max_dim,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(SquaredL2, x, y),
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "neon",
                        &mut rng,
                    );
                }
            }
        };
    }

    test_bitslice!(test_bitslice_distances_8bit, 8, 0xf0330c6d880e08ff);
    test_bitslice!(test_bitslice_distances_7bit, 7, 0x98aa7f2d4c83844f);
    test_bitslice!(test_bitslice_distances_6bit, 6, 0xf2f7ad7a37764b4c);
    test_bitslice!(test_bitslice_distances_5bit, 5, 0xae878d14973fb43f);
    test_bitslice!(test_bitslice_distances_4bit, 4, 0x8d6dbb8a6b19a4f8);
    test_bitslice!(test_bitslice_distances_3bit, 3, 0x8f56767236e58da2);
    test_bitslice!(test_bitslice_distances_2bit, 2, 0xb04f741a257b61af);
    test_bitslice!(test_bitslice_distances_1bit, 1, 0x820ea031c379eab5);

    ///////////////////////////
    // Hamming Bit Distances //
    ///////////////////////////

    fn test_hamming_distances<R>(dim_max: usize, trials_per_dim: usize, rng: &mut R)
    where
        R: Rng,
    {
        let dist: [i8; 2] = [-1, 1];

        for dim in 0..dim_max {
            if !should_check_this_dimension(dim) {
                continue;
            }

            let mut x_reference: Vec<i8> = vec![1; dim];
            let mut y_reference: Vec<i8> = vec![1; dim];

            let mut x = BoxedBitSlice::<1, Binary>::new_boxed(dim);
            let mut y = BoxedBitSlice::<1, Binary>::new_boxed(dim);

            for _ in 0..trials_per_dim {
                x_reference
                    .iter_mut()
                    .for_each(|i| *i = *dist.choose(rng).unwrap());
                y_reference
                    .iter_mut()
                    .for_each(|i| *i = *dist.choose(rng).unwrap());

                // Fill the input slices with 1's so we can catch situations where we don't
                // correctly handle odd remaining elements.
                x.as_mut_slice().fill(u8::MAX);
                y.as_mut_slice().fill(u8::MAX);

                for i in 0..dim {
                    x.set(i, x_reference[i].into()).unwrap();
                    y.set(i, y_reference[i].into()).unwrap();
                }

                // We can check equality by evaluating the L2 distance between the reference
                // vectors.
                //
                // This is proportional to the Hamming distance by a factor of 4 (since the
                // distance betwwen +1 and -1 is 2 - and 2^2 = 4.
                let expected: MV<f32> =
                    diskann_vector::distance::SquaredL2::evaluate(&*x_reference, &*y_reference);
                let got: MV<u32> = Hamming::evaluate(x.reborrow(), y.reborrow()).unwrap();
                assert_eq!(4.0 * (got.into_inner() as f32), expected.into_inner());
            }
        }

        let x = BoxedBitSlice::<1, Binary>::new_boxed(10);
        let y = BoxedBitSlice::<1, Binary>::new_boxed(11);
        assert!(Hamming::evaluate(x.reborrow(), y.reborrow()).is_err());
        assert!(Hamming::evaluate(y.reborrow(), x.reborrow()).is_err());
    }

    #[test]
    fn test_hamming_distance() {
        let mut rng = StdRng::seed_from_u64(0x2160419161246d97);
        test_hamming_distances(MAX_DIM, TRIALS_PER_DIM, &mut rng);
    }

    ///////////////////
    // Heterogeneous //
    ///////////////////

    fn test_bit_transpose_distances<R>(
        dim_max: usize,
        trials_per_dim: usize,
        evaluate_ip: &dyn Fn(USlice<'_, 4, BitTranspose>, USlice<'_, 1>) -> MR,
        context: &str,
        rng: &mut R,
    ) where
        R: Rng,
    {
        let dist_4bit = {
            let domain = Unsigned::domain_const::<4>();
            Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap()
        };

        let dist_1bit = {
            let domain = Unsigned::domain_const::<1>();
            Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap()
        };

        for dim in 0..dim_max {
            if !should_check_this_dimension(dim) {
                continue;
            }

            let mut x_reference: Vec<u8> = vec![0; dim];
            let mut y_reference: Vec<u8> = vec![0; dim];

            let mut x = BoxedBitSlice::<4, Unsigned, BitTranspose>::new_boxed(dim);
            let mut y = BoxedBitSlice::<1, Unsigned, Dense>::new_boxed(dim);

            for trial in 0..trials_per_dim {
                x_reference
                    .iter_mut()
                    .for_each(|i| *i = dist_4bit.sample(rng).try_into().unwrap());
                y_reference
                    .iter_mut()
                    .for_each(|i| *i = dist_1bit.sample(rng).try_into().unwrap());

                // First - pre-set all the values in the bit-slices to 1.
                x.as_mut_slice().fill(u8::MAX);
                y.as_mut_slice().fill(u8::MAX);

                for i in 0..dim {
                    x.set(i, x_reference[i].into()).unwrap();
                    y.set(i, y_reference[i].into()).unwrap();
                }

                // Check IP
                let expected: MV<f32> =
                    diskann_vector::distance::InnerProduct::evaluate(&*x_reference, &*y_reference);

                let got = evaluate_ip(x.reborrow(), y.reborrow());

                // Integer computations should be exact.
                assert_eq!(
                    expected.into_inner(),
                    got.unwrap().into_inner() as f32,
                    "faild InnerProduct for dim = {}, trial = {} -- context {}",
                    dim,
                    trial,
                    context,
                );
            }
        }

        let x = BoxedBitSlice::<4, Unsigned, BitTranspose>::new_boxed(10);
        let y = BoxedBitSlice::<1, Unsigned>::new_boxed(11);
        assert!(
            evaluate_ip(x.reborrow(), y.reborrow()).is_err(),
            "context: {}",
            context
        );

        let y = BoxedBitSlice::<1, Unsigned>::new_boxed(9);
        assert!(
            evaluate_ip(x.reborrow(), y.reborrow()).is_err(),
            "context: {}",
            context
        );
    }

    #[test]
    fn test_bit_transpose_distance() {
        let mut rng = StdRng::seed_from_u64(0xe20e26e926d4b853);

        test_bit_transpose_distances(
            MAX_DIM,
            TRIALS_PER_DIM,
            &|x, y| InnerProduct::evaluate(x, y),
            "pure distance function",
            &mut rng,
        );

        test_bit_transpose_distances(
            MAX_DIM,
            TRIALS_PER_DIM,
            &|x, y| diskann_wide::arch::Scalar::new().run2(InnerProduct, x, y),
            "scalar",
            &mut rng,
        );

        // Architecture Specific.
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            test_bit_transpose_distances(
                MAX_DIM,
                TRIALS_PER_DIM,
                &|x, y| arch.run2(InnerProduct, x, y),
                "x86-64-v3",
                &mut rng,
            );
        }

        // Architecture Specific.
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            test_bit_transpose_distances(
                MAX_DIM,
                TRIALS_PER_DIM,
                &|x, y| arch.run2(InnerProduct, x, y),
                "x86-64-v4",
                &mut rng,
            );
        }

        // Architecture Specific.
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            test_bit_transpose_distances(
                MAX_DIM,
                TRIALS_PER_DIM,
                &|x, y| arch.run2(InnerProduct, x, y),
                "neon",
                &mut rng,
            );
        }
    }

    //////////
    // Full //
    //////////

    fn test_full_distances<const NBITS: usize>(
        dim_max: usize,
        trials_per_dim: usize,
        evaluate_ip: &dyn Fn(&[f32], USlice<'_, NBITS>) -> MathematicalResult<f32>,
        context: &str,
        rng: &mut impl Rng,
    ) where
        Unsigned: Representation<NBITS>,
    {
        // let dist_float = Uniform::new_inclusive(-2.0f32, 2.0f32).unwrap();
        let dist_float = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let dist_bit = {
            let domain = Unsigned::domain_const::<NBITS>();
            Uniform::new_inclusive(*domain.start(), *domain.end()).unwrap()
        };

        for dim in 0..dim_max {
            if !should_check_this_dimension(dim) {
                continue;
            }

            let mut x: Vec<f32> = vec![0.0; dim];

            let mut y_reference: Vec<u8> = vec![0; dim];
            let mut y = BoxedBitSlice::<NBITS, Unsigned, Dense>::new_boxed(dim);

            for trial in 0..trials_per_dim {
                x.iter_mut()
                    .for_each(|i| *i = *dist_float.choose(rng).unwrap());
                y_reference
                    .iter_mut()
                    .for_each(|i| *i = dist_bit.sample(rng).try_into().unwrap());

                // First - pre-set all the values in the bit-slices to 1.
                y.as_mut_slice().fill(u8::MAX);

                let mut expected = 0.0;
                for i in 0..dim {
                    y.set(i, y_reference[i].into()).unwrap();
                    expected += y_reference[i] as f32 * x[i];
                }

                // Check IP
                let got = evaluate_ip(&x, y.reborrow()).unwrap();

                // Integer computations should be exact.
                assert_eq!(
                    expected,
                    got.into_inner(),
                    "faild InnerProduct for dim = {}, trial = {} -- context {}",
                    dim,
                    trial,
                    context,
                );

                // Ensure that using the `Scalar` architecture providers the same
                // results.
                let scalar: MV<f32> = InnerProduct
                    .run(diskann_wide::arch::Scalar, x.as_slice(), y.reborrow())
                    .unwrap();
                assert_eq!(got.into_inner(), scalar.into_inner());
            }
        }

        // Error Checking
        let x = vec![0.0; 10];
        let y = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(11);
        assert!(
            evaluate_ip(x.as_slice(), y.reborrow()).is_err(),
            "context: {}",
            context
        );

        let y = BoxedBitSlice::<NBITS, Unsigned>::new_boxed(9);
        assert!(
            evaluate_ip(x.as_slice(), y.reborrow()).is_err(),
            "context: {}",
            context
        );
    }

    macro_rules! test_full {
        ($name:ident, $nbits:literal, $seed:literal) => {
            #[test]
            fn $name() {
                let mut rng = StdRng::seed_from_u64($seed);

                test_full_distances::<$nbits>(
                    MAX_DIM,
                    TRIALS_PER_DIM,
                    &|x, y| InnerProduct::evaluate(x, y),
                    "pure distance function",
                    &mut rng,
                );

                test_full_distances::<$nbits>(
                    MAX_DIM,
                    TRIALS_PER_DIM,
                    &|x, y| diskann_wide::arch::Scalar::new().run2(InnerProduct, x, y),
                    "scalar",
                    &mut rng,
                );

                // Architecture Specific.
                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                    test_full_distances::<$nbits>(
                        MAX_DIM,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "x86-64-v3",
                        &mut rng,
                    );
                }

                #[cfg(target_arch = "x86_64")]
                if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                    test_full_distances::<$nbits>(
                        MAX_DIM,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "x86-64-v4",
                        &mut rng,
                    );
                }

                #[cfg(target_arch = "aarch64")]
                if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                    test_full_distances::<$nbits>(
                        MAX_DIM,
                        TRIALS_PER_DIM,
                        &|x, y| arch.run2(InnerProduct, x, y),
                        "neon",
                        &mut rng,
                    );
                }
            }
        };
    }

    test_full!(test_full_distance_1bit, 1, 0xe20e26e926d4b853);
    test_full!(test_full_distance_2bit, 2, 0xae9542700aecbf68);
    test_full!(test_full_distance_3bit, 3, 0xfffd04b26bb6068c);
    test_full!(test_full_distance_4bit, 4, 0x86db49fd1a1704ba);
    test_full!(test_full_distance_5bit, 5, 0x3a35dc7fa7931c41);
    test_full!(test_full_distance_6bit, 6, 0x1f69de79e418d336);
    test_full!(test_full_distance_7bit, 7, 0x3fcf17b82dadc5ab);
    test_full!(test_full_distance_8bit, 8, 0x85dcaf48b1399db2);

    ///////////////////////////////////////////
    // Heterogeneous: USlice<8> × USlice<M> //
    ///////////////////////////////////////////

    /// Helper that builds vectors from a fill function and asserts the
    /// inner product matches.
    struct HetCase<const M: usize> {
        x_vals: Vec<i64>,
        y_vals: Vec<i64>,
    }

    impl<const M: usize> HetCase<M>
    where
        Unsigned: Representation<M>,
    {
        fn new(dim: usize, fill: impl FnMut(usize) -> (i64, i64)) -> Self {
            let (x_vals, y_vals) = (0..dim).map(fill).unzip();
            Self { x_vals, y_vals }
        }

        fn check_with(
            &self,
            label: impl Display,
            evaluate: &dyn Fn(USlice<'_, 8>, USlice<'_, M>) -> MR,
        ) {
            let dim = self.x_vals.len();
            let mut x = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
            let mut y = BoxedBitSlice::<M, Unsigned>::new_boxed(dim);
            // Pre-fill with 0xFF to catch trailing-element bugs.
            x.as_mut_slice().fill(u8::MAX);
            y.as_mut_slice().fill(u8::MAX);
            for (i, (&xv, &yv)) in self.x_vals.iter().zip(&self.y_vals).enumerate() {
                x.set(i, xv).unwrap();
                y.set(i, yv).unwrap();
            }
            let expected: u32 = self
                .x_vals
                .iter()
                .zip(&self.y_vals)
                .map(|(&a, &b)| a as u32 * b as u32)
                .sum();
            let got = evaluate(x.reborrow(), y.reborrow()).unwrap().into_inner();
            assert_eq!(expected, got, "{} failed for dim = {}", label, dim);
        }
    }

    /// Fuzz test helper: random vectors across many dimensions.
    fn fuzz_heterogeneous_ip<const M: usize>(
        dim_max: usize,
        trials_per_dim: usize,
        max_val: i64,
        evaluate_ip: &dyn Fn(USlice<'_, 8>, USlice<'_, M>) -> MR,
        context: &str,
        rng: &mut impl Rng,
    ) where
        Unsigned: Representation<M>,
    {
        let dist_8bit = Uniform::new_inclusive(0i64, 255i64).unwrap();
        let dist_mbit = Uniform::new_inclusive(0i64, max_val).unwrap();

        for dim in 0..dim_max {
            for trial in 0..trials_per_dim {
                HetCase::<M>::new(dim, |_| {
                    (dist_8bit.sample(&mut *rng), dist_mbit.sample(&mut *rng))
                })
                .check_with(
                    lazy_format!("IP(8,{}) dim={dim}, trial={trial} -- {context}", M),
                    evaluate_ip,
                );
            }

            // Length mismatch → error.
            let x = BoxedBitSlice::<8, Unsigned>::new_boxed(dim);
            let y = BoxedBitSlice::<M, Unsigned>::new_boxed(dim + 1);
            assert!(
                evaluate_ip(x.reborrow(), y.reborrow()).is_err(),
                "context: {}",
                context,
            );
        }
    }

    /// All values at maximum: x[i] = 255, y[i] = max_val.
    /// Confirms no i16 saturation in vpmaddubsw.
    fn het_test_max_values<const M: usize>(
        max_val: i64,
        context: &str,
        evaluate: &dyn Fn(USlice<'_, 8>, USlice<'_, M>) -> MR,
    ) where
        Unsigned: Representation<M>,
    {
        let dims = [127, 128, 129, 255, 256, 512, 768, 896, 3072];
        for &dim in &dims {
            let case = HetCase::<M>::new(dim, |_| (255, max_val));
            case.check_with(lazy_format!("max-value {context} dim={dim}"), evaluate);
        }
    }

    /// Known-answer tests to catch bit-ordering bugs.
    fn het_test_known_answers<const M: usize>(
        max_val: i64,
        evaluate: &dyn Fn(USlice<'_, 8>, USlice<'_, M>) -> MR,
    ) where
        Unsigned: Representation<M>,
    {
        // _mm256_addubs_epi8 unsigned treatment: x[i] = 200 (> 127), y[i] = max_val.
        // Correct: 200 × max_val per element.
        HetCase::<M>::new(64, |_| (200, max_val)).check_with("vpmaddubsw operand-order", evaluate);

        // Ascending x, constant y.
        let y_val = (max_val / 2).max(1);
        HetCase::<M>::new(128, |i| ((i % 256) as i64, y_val))
            .check_with("ascending-x constant-y", evaluate);

        // Single element (pure scalar fallback).
        HetCase::<M>::new(1, |_| (200, max_val)).check_with("single element", evaluate);
    }

    /// Exhaustive edge-case coverage.
    fn het_test_edge_cases<const M: usize>(
        max_val: i64,
        block_size: usize,
        evaluate: &dyn Fn(USlice<'_, 8>, USlice<'_, M>) -> MR,
    ) where
        Unsigned: Representation<M>,
    {
        let y_half = (max_val / 2).max(1);

        // One side zero.
        HetCase::<M>::new(64, |_| (0, max_val)).check_with("x-zero y-nonzero", evaluate);
        HetCase::<M>::new(64, |_| (255, 0)).check_with("y-zero x-nonzero", evaluate);

        // Every dimension from 0..block_size+1 (scalar fallback boundary).
        for dim in 0..=(block_size + 1) {
            HetCase::<M>::new(dim, |_| (3, y_half)).check_with("uniform fill", evaluate);
        }

        // Exact block boundaries.
        for &dim in &[block_size, 2 * block_size, 4 * block_size, 8 * block_size] {
            HetCase::<M>::new(dim, |_| (100, max_val)).check_with("exact block boundary", evaluate);
        }

        // x varies, y constant.
        HetCase::<M>::new(300, |i| ((i % 256) as i64, 1))
            .check_with("x-varies y-constant", evaluate);

        // x constant, y varies.
        HetCase::<M>::new(300, |i| (1, (i as i64) % (max_val + 1)))
            .check_with("x-constant y-varies", evaluate);

        // Alternating pattern.
        HetCase::<M>::new(128, |i| if i % 2 == 0 { (255, max_val) } else { (0, 0) })
            .check_with("alternating pattern", evaluate);

        // Opposite alternating pattern.
        HetCase::<M>::new(128, |i| if i % 2 == 0 { (0, 0) } else { (255, max_val) })
            .check_with("opposite alternating", evaluate);

        // Large accumulation check for overflow.
        HetCase::<M>::new(1024, |_| (255, max_val)).check_with("large accumulation", evaluate);

        // x > 127 sweep (vpmaddubsw unsigned treatment).
        for x_val in [128i64, 170, 200, 240, 255] {
            HetCase::<M>::new(block_size, move |_| (x_val, y_half))
                .check_with(lazy_format!("x > 127 (x_val={x_val})"), evaluate);
        }

        // Dim = block_size - 1 (no full block, all scalar).
        HetCase::<M>::new(block_size - 1, |i| {
            (
                ((i * 7 + 3) % 256) as i64,
                ((i * 11 + 5) as i64) % (max_val + 1),
            )
        })
        .check_with("dim=block_size-1 (all scalar)", evaluate);

        // 4× unroll boundary exercises.
        let unroll4 = 4 * block_size;
        for &dim in &[
            unroll4,
            unroll4 + 1,
            unroll4 + block_size,
            unroll4 + block_size + 1,
        ] {
            HetCase::<M>::new(dim, |i| {
                (((i + 1) % 256) as i64, ((i + 1) as i64) % (max_val + 1))
            })
            .check_with("unroll boundary", evaluate);
        }
    }

    macro_rules! heterogeneous_ip_tests_8xM {
        (
            mod_name: $mod:ident,
            M: $M:literal,
            max_val: $max_val:literal,
            block_size: $block_size:literal,
            seed_fuzz: $seed_fuzz:literal,
        ) => {
            mod $mod {
                use super::*;

                #[test]
                fn all_ip_dispatches() {
                    let mut rng = StdRng::seed_from_u64($seed_fuzz);

                    fuzz_heterogeneous_ip::<$M>(
                        MAX_DIM,
                        TRIALS_PER_DIM,
                        $max_val,
                        &|x, y| InnerProduct::evaluate(x, y),
                        "pure distance function",
                        &mut rng,
                    );
                    fuzz_heterogeneous_ip::<$M>(
                        MAX_DIM,
                        TRIALS_PER_DIM,
                        $max_val,
                        &|x, y| diskann_wide::arch::Scalar::new().run2(InnerProduct, x, y),
                        "scalar arch",
                        &mut rng,
                    );
                    #[cfg(target_arch = "x86_64")]
                    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                        fuzz_heterogeneous_ip::<$M>(
                            MAX_DIM,
                            TRIALS_PER_DIM,
                            $max_val,
                            &|x, y| arch.run2(InnerProduct, x, y),
                            "x86-64-v3",
                            &mut rng,
                        );
                    }
                    #[cfg(target_arch = "x86_64")]
                    if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                        fuzz_heterogeneous_ip::<$M>(
                            MAX_DIM,
                            TRIALS_PER_DIM,
                            $max_val,
                            &|x, y| arch.run2(InnerProduct, x, y),
                            "x86-64-v4",
                            &mut rng,
                        );
                    }
                }

                #[test]
                fn max_values() {
                    het_test_max_values::<$M>($max_val, "dispatch", &|x, y| {
                        InnerProduct::evaluate(x, y)
                    });
                    #[cfg(target_arch = "x86_64")]
                    if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                        het_test_max_values::<$M>($max_val, "V3", &|x, y| {
                            arch.run2(InnerProduct, x, y)
                        });
                    }
                }

                #[test]
                fn known_answers() {
                    het_test_known_answers::<$M>($max_val, &|x, y| InnerProduct::evaluate(x, y));
                }

                #[test]
                fn edge_cases() {
                    het_test_edge_cases::<$M>($max_val, $block_size, &|x, y| {
                        InnerProduct::evaluate(x, y)
                    });
                }
            }
        };
    }

    heterogeneous_ip_tests_8xM! {
        mod_name: heterogeneous_ip_8x4,
        M: 4,
        max_val: 15,
        block_size: 32,
        seed_fuzz: 0xd3a7f1c09b2e4856,
    }

    heterogeneous_ip_tests_8xM! {
        mod_name: heterogeneous_ip_8x2,
        M: 2,
        max_val: 3,
        block_size: 64,
        seed_fuzz: 0x82c4a6e809f1d3b5,
    }

    heterogeneous_ip_tests_8xM! {
        mod_name: heterogeneous_ip_8x1,
        M: 1,
        max_val: 1,
        block_size: 32,
        seed_fuzz: 0x1b17_a5e7c2d0f839,
    }
}
