/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::convert::{AsMut, AsRef};

use diskann_wide::arch::Target2;
#[cfg(not(target_arch = "aarch64"))]
use diskann_wide::{Architecture, Const, Constant, SIMDCast, SIMDVector};
use half::f16;

/// Perform a numeric cast on a slice of values.
///
/// This trait is intended to have the following numerical behavior:
///
/// 1. If a lossless conversion between types is available, use that.
/// 2. Otherwise, if the two type are floating point types, use a round-to-nearest strategy.
/// 3. Otherwise, try to behave like the Rust `as` numeric cast.
///
/// The main reason we can't just say "behave like "as"" is because Rust does not have
/// a native `f16` type, which this crate supports.
pub trait CastFromSlice<From> {
    fn cast_from_slice(self, from: From);
}

macro_rules! use_simd_cast_from_slice {
    ($from:ty => $to:ty) => {
        impl CastFromSlice<&[$from]> for &mut [$to] {
            #[inline(always)]
            fn cast_from_slice(self, from: &[$from]) {
                SliceCast::<$to, $from>::new().run(diskann_wide::ARCH, self, from)
            }
        }

        impl<const N: usize> CastFromSlice<&[$from; N]> for &mut [$to; N] {
            #[inline(always)]
            fn cast_from_slice(self, from: &[$from; N]) {
                SliceCast::<$to, $from>::new().run(diskann_wide::ARCH, self, from)
            }
        }
    };
}

use_simd_cast_from_slice!(f32 => f16);
use_simd_cast_from_slice!(f16 => f32);

/// A zero-sized type providing implementations of [`diskann_wide::arch::Target2`] to provide
/// platform-dependent conversions between slices of the two generic types.
#[derive(Debug, Default, Clone, Copy)]
pub struct SliceCast<To, From> {
    _marker: std::marker::PhantomData<(To, From)>,
}

impl<To, From> SliceCast<To, From> {
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

// Non-SIMD Instantiations

impl<T, U> Target2<diskann_wide::arch::Scalar, (), T, U> for SliceCast<f16, f32>
where
    T: AsMut<[f16]>,
    U: AsRef<[f32]>,
{
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar, mut to: T, from: U) {
        let to = to.as_mut();
        let from = from.as_ref();
        std::iter::zip(to.iter_mut(), from.iter()).for_each(|(to, from)| {
            *to = diskann_wide::cast_f32_to_f16(*from);
        })
    }
}

impl<T, U> Target2<diskann_wide::arch::Scalar, (), T, U> for SliceCast<f32, f16>
where
    T: AsMut<[f32]>,
    U: AsRef<[f16]>,
{
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar, mut to: T, from: U) {
        let to = to.as_mut();
        let from = from.as_ref();
        std::iter::zip(to.iter_mut(), from.iter()).for_each(|(to, from)| {
            *to = diskann_wide::cast_f16_to_f32(*from);
        })
    }
}

// SIMD Instantiations
#[cfg(target_arch = "x86_64")]
impl<T, U, To, From> Target2<diskann_wide::arch::x86_64::V4, (), T, U> for SliceCast<To, From>
where
    T: AsMut<[To]>,
    U: AsRef<[From]>,
    diskann_wide::arch::x86_64::V4: SIMDConvert<To, From>,
{
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V4, mut to: T, from: U) {
        simd_convert(arch, to.as_mut(), from.as_ref())
    }
}

#[cfg(target_arch = "x86_64")]
impl<T, U, To, From> Target2<diskann_wide::arch::x86_64::V3, (), T, U> for SliceCast<To, From>
where
    T: AsMut<[To]>,
    U: AsRef<[From]>,
    diskann_wide::arch::x86_64::V3: SIMDConvert<To, From>,
{
    #[inline(always)]
    fn run(self, arch: diskann_wide::arch::x86_64::V3, mut to: T, from: U) {
        simd_convert(arch, to.as_mut(), from.as_ref())
    }
}

/////////////////////////////
// General SIMD Conversion //
/////////////////////////////

/// A helper trait to fill in the gaps for the unrolled `simd_convert` method.
#[cfg(target_arch = "x86_64")]
trait SIMDConvert<To, From>: Architecture {
    /// A constant encoding the the SIMD width of the underlying schema.
    type Width: Constant<Type = usize>;

    /// The SIMD Vector for the converted-to type.
    type WideTo: SIMDVector<Arch = Self, Scalar = To, ConstLanes = Self::Width>;

    /// The SIMD Vector for the converted-from type.
    type WideFrom: SIMDVector<Arch = Self, Scalar = From, ConstLanes = Self::Width>;

    /// The method that actually does the vector-wide conversion.
    fn simd_convert(from: Self::WideFrom) -> Self::WideTo;

    /// Delegate routing for handling conversion lengths less than the vector width.
    ///
    /// The canonical implementation uses predicated loads, but implementations may wish
    /// to use a scalar loop instead.
    ///
    /// # Safety
    ///
    /// This trait will only be called when the following guarantees are made:
    ///
    /// * `pto` will point to properly aligned memory that is valid for writes on the
    ///   range `[pto, pto + len)`.
    /// * `pfrom` will point to properly aligned memory that is valid for reads on the
    ///   range `[pfrom, pfrom + len)`.
    /// * The memory ranges covered by `pto` and `pfrom` must not alias.
    #[inline(always)]
    unsafe fn handle_small(self, pto: *mut To, pfrom: *const From, len: usize) {
        let from = Self::WideFrom::load_simd_first(self, pfrom, len);
        let to = Self::simd_convert(from);
        to.store_simd_first(pto, len);
    }

    /// !! Do not extend this function !!
    ///
    /// Due to limitations on how associated constants can be used, we need a function
    /// to access the SIMD width and rely on the compiler to constant propagate the result.
    #[inline(always)]
    fn get_simd_width() -> usize {
        Self::Width::value()
    }
}

#[inline(never)]
#[allow(clippy::panic)]
#[cfg(target_arch = "x86_64")]
fn emit_length_error(xlen: usize, ylen: usize) -> ! {
    panic!(
        "lengths must be equal, instead got: xlen = {}, ylen = {}",
        xlen, ylen
    )
}

/// Convert each element of `from` into its corresponding position in `to` using the
/// conversion rule applied by `S`.
///
/// # Panics
///
/// Panics if `to.len() != from.len()`.
///
/// # Implementation Notes
///
/// This function will only call `A::handle_small` if the total length of the processed
/// slices is less that the underlying SIMD width.
///
/// Otherwise, we take advantage of unaligned operations to avoid dealing with
/// non-full-width chunks.
///
/// For example, if the SIMD width was 4 and the total length was 7, then it would be
/// processed in two chunks of 4 like so:
/// ```text
///      Chunk 0
/// |---------------|
///   0   1   2   3   4   5   6
///             |---------------|
///                   Chunk 1
/// ```
/// This overlapping can only happen at the very end of the slice and only if the length
/// of the slice is not a multiple of the SIMD width used.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
fn simd_convert<A, To, From>(arch: A, to: &mut [To], from: &[From])
where
    A: SIMDConvert<To, From>,
{
    let len = to.len();

    // Keep stack writes to a minimum by explicitly outlining error handling.
    if len != from.len() {
        emit_length_error(len, from.len())
    }

    // Get the SIMD width.
    //
    // We're relying on the compiler to constant propagate this.
    let width = A::get_simd_width();

    let pto = to.as_mut_ptr();
    let pfrom = from.as_ptr();

    // Too short, deal with the small case and return.
    if len < width {
        // SAFETY: We know `pto` and `pfrom` do not alias because of Rust's aliasing
        // rules on `to` and `from.
        //
        // Additionally, we've checked that both spans are valid for `len`.
        unsafe { arch.handle_small(pto, pfrom, len) };
        return;
    }

    const UNROLL: usize = 8;

    let mut i = 0;
    // SAFETY: We emit a bunch of unrolled load and store operations in this loop.
    //
    // All of these operations are safe because the bound `i + UNROLL * width <= len`
    // is checked.
    unsafe {
        while i + UNROLL * width <= len {
            let s0 = A::WideFrom::load_simd(arch, pfrom.add(i));
            A::simd_convert(s0).store_simd(pto.add(i));

            let s1 = A::WideFrom::load_simd(arch, pfrom.add(i + width));
            A::simd_convert(s1).store_simd(pto.add(i + width));

            let s2 = A::WideFrom::load_simd(arch, pfrom.add(i + 2 * width));
            A::simd_convert(s2).store_simd(pto.add(i + 2 * width));

            let s3 = A::WideFrom::load_simd(arch, pfrom.add(i + 3 * width));
            A::simd_convert(s3).store_simd(pto.add(i + 3 * width));

            let s0 = A::WideFrom::load_simd(arch, pfrom.add(i + 4 * width));
            A::simd_convert(s0).store_simd(pto.add(i + 4 * width));

            let s1 = A::WideFrom::load_simd(arch, pfrom.add(i + 5 * width));
            A::simd_convert(s1).store_simd(pto.add(i + 5 * width));

            let s2 = A::WideFrom::load_simd(arch, pfrom.add(i + 6 * width));
            A::simd_convert(s2).store_simd(pto.add(i + 6 * width));

            let s3 = A::WideFrom::load_simd(arch, pfrom.add(i + 7 * width));
            A::simd_convert(s3).store_simd(pto.add(i + 7 * width));

            i += UNROLL * width;
        }
    }

    while i + width <= len {
        // SAFETY: `i + width <= len` ensure that this read is in-bounds.
        let s0 = unsafe { A::WideFrom::load_simd(arch, pfrom.add(i)) };
        let t0 = A::simd_convert(s0);
        // SAFETY: `i + width <= len` ensure that this write is in-bounds.
        unsafe { t0.store_simd(pto.add(i)) };
        i += width;
    }

    // Check if we need to deal with any remaining elements.
    // If so, bump back `i` so we can process a whole chunk.
    if i != len {
        let offset = i - (width - len % width);

        // SAFETY: At this point, we know that `len >= width`, `i < len`, and
        // `len - i == len % width != 0`.
        //
        // Therefore, `offset` is inbounds and `offset + width == len`.
        let s0 = unsafe { A::WideFrom::load_simd(arch, pfrom.add(offset)) };
        let t0 = A::simd_convert(s0);

        // SAFETY: This write is safe for the same reason that the preceeding read is safe.
        unsafe { t0.store_simd(pto.add(offset)) };
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDConvert<f32, f16> for diskann_wide::arch::x86_64::V4 {
    type Width = Const<8>;
    type WideTo = <diskann_wide::arch::x86_64::V4 as Architecture>::f32x8;
    type WideFrom = <diskann_wide::arch::x86_64::V4 as Architecture>::f16x8;

    #[inline(always)]
    fn simd_convert(from: Self::WideFrom) -> Self::WideTo {
        from.into()
    }

    // SAFETY: We only access data in the valid range for `pto` and `pfrom`.
    #[inline(always)]
    unsafe fn handle_small(self, pto: *mut f32, pfrom: *const f16, len: usize) {
        for i in 0..len {
            *pto.add(i) = diskann_wide::cast_f16_to_f32(*pfrom.add(i))
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDConvert<f32, f16> for diskann_wide::arch::x86_64::V3 {
    type Width = Const<8>;
    type WideTo = <diskann_wide::arch::x86_64::V3 as Architecture>::f32x8;
    type WideFrom = <diskann_wide::arch::x86_64::V3 as Architecture>::f16x8;

    #[inline(always)]
    fn simd_convert(from: Self::WideFrom) -> Self::WideTo {
        from.into()
    }

    // SAFETY: We only access data in the valid range for `pto` and `pfrom`.
    #[inline(always)]
    unsafe fn handle_small(self, pto: *mut f32, pfrom: *const f16, len: usize) {
        for i in 0..len {
            *pto.add(i) = diskann_wide::cast_f16_to_f32(*pfrom.add(i))
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDConvert<f16, f32> for diskann_wide::arch::x86_64::V4 {
    type Width = Const<8>;
    type WideTo = <diskann_wide::arch::x86_64::V4 as Architecture>::f16x8;
    type WideFrom = <diskann_wide::arch::x86_64::V4 as Architecture>::f32x8;

    #[inline(always)]
    fn simd_convert(from: Self::WideFrom) -> Self::WideTo {
        from.simd_cast()
    }

    // SAFETY: We only access data in the valid range for `pto` and `pfrom`.
    #[inline(always)]
    unsafe fn handle_small(self, pto: *mut f16, pfrom: *const f32, len: usize) {
        for i in 0..len {
            *pto.add(i) = diskann_wide::cast_f32_to_f16(*pfrom.add(i))
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl SIMDConvert<f16, f32> for diskann_wide::arch::x86_64::V3 {
    type Width = Const<8>;
    type WideTo = <diskann_wide::arch::x86_64::V3 as Architecture>::f16x8;
    type WideFrom = <diskann_wide::arch::x86_64::V3 as Architecture>::f32x8;

    #[inline(always)]
    fn simd_convert(from: Self::WideFrom) -> Self::WideTo {
        from.simd_cast()
    }

    // SAFETY: We only access data in the valid range for `pto` and `pfrom`.
    #[inline(always)]
    unsafe fn handle_small(self, pto: *mut f16, pfrom: *const f32, len: usize) {
        for i in 0..len {
            *pto.add(i) = diskann_wide::cast_f32_to_f16(*pfrom.add(i))
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;

    ////////////////
    // Fuzz tests //
    ////////////////

    trait ReferenceConvert<From> {
        fn reference_convert(self, from: &[From]);
    }

    impl ReferenceConvert<f32> for &mut [f16] {
        fn reference_convert(self, from: &[f32]) {
            assert_eq!(self.len(), from.len());
            std::iter::zip(self.iter_mut(), from.iter()).for_each(|(d, s)| *d = f16::from_f32(*s));
        }
    }

    impl ReferenceConvert<f16> for &mut [f32] {
        fn reference_convert(self, from: &[f16]) {
            assert_eq!(self.len(), from.len());
            std::iter::zip(self.iter_mut(), from.iter()).for_each(|(d, s)| *d = (*s).into());
        }
    }

    fn test_cast_from_slice<To, From>(max_dim: usize, num_trials: usize, rng: &mut StdRng)
    where
        StandardUniform: Distribution<From>,
        To: Default + PartialEq + std::fmt::Debug + Copy,
        From: Default + Copy,
        for<'a, 'b> &'a mut [To]: CastFromSlice<&'b [From]> + ReferenceConvert<From>,
    {
        let distribution = StandardUniform {};
        for dim in 0..=max_dim {
            let mut src = vec![From::default(); dim];
            let mut dst = vec![To::default(); dim];
            let mut dst_reference = vec![To::default(); dim];

            for _ in 0..num_trials {
                src.iter_mut().for_each(|s| *s = distribution.sample(rng));
                dst.cast_from_slice(src.as_slice());
                dst_reference.reference_convert(&src);

                assert_eq!(dst, dst_reference);
            }
        }
    }

    #[test]
    fn test_f32_to_f16_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x0a3bfe052a8ebf98);
        test_cast_from_slice::<f16, f32>(256, 10, &mut rng);
    }

    #[test]
    fn test_f16_to_f32_fuzz() {
        let mut rng = StdRng::seed_from_u64(0x83765b2816321eca);
        test_cast_from_slice::<f32, f16>(256, 10, &mut rng);
    }

    ////////////////
    // Miri Tests //
    ////////////////

    // With Miri, we need to be really careful that we do not hit methods that call
    // `cvtph_ps`...
    #[test]
    fn miri_test_f32_to_f16() {
        for dim in 0..256 {
            println!("processing dim {}", dim);

            let src = vec![f32::default(); dim];
            let mut dst = vec![f16::default(); dim];

            // Just check that all accesses are inbounds.
            dst.cast_from_slice(&src);

            // Scalar conversion.
            SliceCast::<f16, f32>::new().run(
                diskann_wide::arch::Scalar,
                dst.as_mut_slice(),
                src.as_slice(),
            );

            // SIMD conversion
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                    SliceCast::<f16, f32>::new().run(arch, dst.as_mut_slice(), src.as_slice())
                }
            }
        }
    }

    #[test]
    fn miri_test_f16_to_f32() {
        for dim in 0..256 {
            println!("processing dim {}", dim);

            let src = vec![f16::default(); dim];
            let mut dst = vec![f32::default(); dim];

            // Just check that all accesses are inbounds.
            dst.cast_from_slice(&src);

            // Scalar conversion.
            SliceCast::<f32, f16>::new().run(
                diskann_wide::arch::Scalar,
                dst.as_mut_slice(),
                src.as_slice(),
            );

            // SIMD conversion
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                    SliceCast::<f32, f16>::new().run(arch, dst.as_mut_slice(), src.as_slice())
                }
            }
        }
    }
}
