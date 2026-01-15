/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use half::f16;

use super::{
    SplitJoin, SupportedLaneCount,
    arch::{self, emulated::Scalar},
    bitmask::BitMask,
    constant::Const,
    reference::{ReferenceAbs, ReferenceCast, ReferenceScalarOps, ReferenceShifts, TreeReduce},
    traits::{
        ArrayType, SIMDAbs, SIMDCast, SIMDDotProduct, SIMDMask, SIMDMinMax, SIMDMulAdd,
        SIMDPartialEq, SIMDPartialOrd, SIMDReinterpret, SIMDSelect, SIMDSumTree, SIMDVector,
    },
};

/// An emulated SIMD vector.
///
/// The emulated implementation behaves just like an intrinsic, but the APIs are implemented
/// using loops over arrays rather than dispatching to platform specific instructions.
///
/// The idea behind this type is that it can be used on architecture where explicit backend
/// support has not been added, or when an architecture does not support a given type/lengh
/// pair well.
///
/// Furthermore, it can be used when developing new back-ends to provide fallback
/// implementations. This allows new back-ends to be developed one piece as a time instead
/// of all at onces.
///
/// NOTE: The alignment requirements of an emulated vector *will* be different than the
/// alignment requirements an actual intrinsic.
///
/// Higher level code *must not* rely on alignments being compatible across architectures!
#[derive(Debug, Clone, Copy)]
pub struct Emulated<T, const N: usize, A = Scalar>(pub(crate) [T; N], A);

impl<T, const N: usize, A> Emulated<T, N, A> {
    pub fn from_arch_fn<F>(arch: A, f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self(core::array::from_fn(f), arch)
    }
}

impl<T, const N: usize, A> SIMDVector for Emulated<T, N, A>
where
    T: Copy + std::fmt::Debug + Default,
    Const<N>: ArrayType<T, Type = [T; N]>,
    BitMask<N, A>: SIMDMask<Arch = A>,
    A: arch::Sealed,
{
    type Arch = A;
    type Scalar = T;
    type Underlying = [T; N];
    type ConstLanes = Const<N>;
    const LANES: usize = N;
    type Mask = BitMask<N, A>;

    /// The underlying behavior is emulated using loops and is not accelerated by back-end
    /// intrinsics.
    const EMULATED: bool = true;

    /// Return the Scalar architecture.
    fn arch(self) -> A {
        self.1
    }

    fn default(arch: A) -> Self {
        Self([T::default(); N], arch)
    }

    /// Return the underlying array.
    fn to_underlying(self) -> Self::Underlying {
        self.0
    }

    /// Construct from the underlying array.
    fn from_underlying(arch: A, repr: [T; N]) -> Self {
        Self(repr, arch)
    }

    /// Return the underlying array.
    fn to_array(self) -> [T; N] {
        self.0
    }

    /// Construct from the underlying array.
    fn from_array(arch: A, x: [T; N]) -> Self {
        Self(x, arch)
    }

    /// Broadcast the provided scalar across all lanes.
    fn splat(arch: A, value: Self::Scalar) -> Self {
        Self([value; N], arch)
    }

    /// Load all the things.
    #[inline(always)]
    unsafe fn load_simd(arch: A, ptr: *const T) -> Self {
        // SAFETY: The caller asserts that `ptr` is contiguously readable for `N` values.
        Self(
            unsafe { std::ptr::read_unaligned(ptr.cast::<[T; N]>()) },
            arch,
        )
    }

    /// Only load values then the corresponding mask lane is set.
    unsafe fn load_simd_masked_logical(arch: A, ptr: *const T, mask: Self::Mask) -> Self {
        Self::from_arch_fn(arch, |i| {
            if mask.get_unchecked(i) {
                // SAFETY: The caller ensures it's safe to access this offset from `ptr`
                // because the lane in `mask` is set.
                unsafe { std::ptr::read_unaligned(ptr.add(i)) }
            } else {
                T::default()
            }
        })
    }

    /// Only load the first `first` items. Set the rest to zero.
    #[inline(always)]
    unsafe fn load_simd_first(arch: A, ptr: *const T, first: usize) -> Self {
        Self::from_arch_fn(arch, |i| {
            if i < first {
                // SAFETY: The caller ensures it's safe to access the first `first` values
                // beginning at `ptr`.
                unsafe { std::ptr::read_unaligned(ptr.add(i)) }
            } else {
                T::default()
            }
        })
    }

    /// Store all the things.
    #[inline(always)]
    unsafe fn store_simd(self, ptr: *mut T) {
        // SAFETY: The caller asserts that it is safe to write `N` contiguous values to `ptr`.
        unsafe { ptr.cast::<[T; N]>().write_unaligned(self.0) }
    }

    /// Only store values then the corresponding mask lane is set.
    unsafe fn store_simd_masked_logical(self, ptr: *mut T, mask: Self::Mask) {
        for (i, v) in self.0.iter().enumerate() {
            if mask.get_unchecked(i) {
                // SAFETY: The caller asserts it is safe to write to offsets with the
                // corresponding bit mask set.
                unsafe { ptr.add(i).write_unaligned(*v) };
            }
        }
    }

    /// Only store the first `first` items. Set the rest to zero.
    #[inline(always)]
    unsafe fn store_simd_first(self, ptr: *mut T, first: usize) {
        for (i, v) in self.0.iter().enumerate().take(first) {
            // SAFETY: The caller asserts it is safe to write to the first `first` offsets
            // beginning at `ptr`.
            unsafe { ptr.add(i).write_unaligned(*v) };
        }
    }
}

/// Binary Ops
impl<T, const N: usize, A> std::ops::Add for Emulated<T, N, A>
where
    T: ReferenceScalarOps + Copy + std::fmt::Debug + std::default::Default,
    Const<N>: ArrayType<T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_add_(rhs.0[i]))
    }
}

impl<T, const N: usize, A> std::ops::Sub for Emulated<T, N, A>
where
    T: ReferenceScalarOps,
{
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_sub_(rhs.0[i]))
    }
}

impl<T, const N: usize, A> std::ops::Mul for Emulated<T, N, A>
where
    T: ReferenceScalarOps,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_mul_(rhs.0[i]))
    }
}

/// MulAdd
impl<T, const N: usize, A> SIMDMulAdd for Emulated<T, N, A>
where
    T: ReferenceScalarOps,
{
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        Self::from_arch_fn(self.1, |i| {
            self.0[i].expected_fma_(rhs.0[i], accumulator.0[i])
        })
    }
}

/// MinMax
impl<T, const N: usize, A> SIMDMinMax for Emulated<T, N, A>
where
    T: ReferenceScalarOps,
{
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_min_(rhs.0[i]))
    }
    #[inline(always)]
    fn max_simd(self, rhs: Self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_max_(rhs.0[i]))
    }
}

/// Abs
impl<T, const N: usize, A> SIMDAbs for Emulated<T, N, A>
where
    T: ReferenceAbs,
{
    #[inline(always)]
    fn abs_simd(self) -> Self {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_abs_())
    }
}

/// SIMDPartialEq
impl<T, const N: usize, A> SIMDPartialEq for Emulated<T, N, A>
where
    T: PartialEq,
    Self: SIMDVector,
{
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] == other.0[i])
    }

    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] != other.0[i])
    }
}

/// SIMDPartialOrd
impl<T, const N: usize, A> SIMDPartialOrd for Emulated<T, N, A>
where
    T: PartialOrd,
    Self: SIMDVector,
{
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] < other.0[i])
    }

    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] <= other.0[i])
    }

    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] > other.0[i])
    }

    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        Self::Mask::from_fn(self.arch(), |i| self.0[i] >= other.0[i])
    }
}

// Bit Ops
impl<T, const N: usize, A> std::ops::BitAnd for Emulated<T, N, A>
where
    T: std::ops::BitAnd<Output = T> + Copy,
{
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i] & other.0[i])
    }
}

impl<T, const N: usize, A> std::ops::BitOr for Emulated<T, N, A>
where
    T: std::ops::BitOr<Output = T> + Copy,
{
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i] | other.0[i])
    }
}

impl<T, const N: usize, A> std::ops::BitXor for Emulated<T, N, A>
where
    T: std::ops::BitXor<Output = T> + Copy,
{
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, other: Self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i] ^ other.0[i])
    }
}

impl<T, const N: usize, A> std::ops::Not for Emulated<T, N, A>
where
    T: std::ops::Not<Output = T> + Copy,
{
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| !self.0[i])
    }
}

impl<T, const N: usize, A> std::ops::Shl for Emulated<T, N, A>
where
    T: ReferenceShifts,
{
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_shl_(rhs.0[i]))
    }
}

impl<T, const N: usize, A> std::ops::Shl<T> for Emulated<T, N, A>
where
    T: ReferenceShifts,
{
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: T) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_shl_(rhs))
    }
}

impl<T, const N: usize, A> std::ops::Shr for Emulated<T, N, A>
where
    T: ReferenceShifts,
{
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_shr_(rhs.0[i]))
    }
}

impl<T, const N: usize, A> std::ops::Shr<T> for Emulated<T, N, A>
where
    T: ReferenceShifts,
{
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: T) -> Self::Output {
        Self::from_arch_fn(self.1, |i| self.0[i].expected_shr_(rhs))
    }
}

//////////////////
// Dot Products //
//////////////////

// i16 to i32
macro_rules! impl_simd_dot_product_i16_to_i32 {
    ($N:literal, $TwoN:literal) => {
        /// Promote intermediate values to `i32` and then perform accuulation.
        impl<A> SIMDDotProduct<Emulated<i16, $TwoN, A>> for Emulated<i32, $N, A>
        where
            A: arch::Sealed,
        {
            fn dot_simd(
                self,
                left: Emulated<i16, $TwoN, A>,
                right: Emulated<i16, $TwoN, A>,
            ) -> Self {
                self + Self::from_arch_fn(self.1, |i| {
                    let l0: i32 = left.0[2 * i].into();
                    let l1: i32 = left.0[2 * i + 1].into();

                    let r0: i32 = right.0[2 * i].into();
                    let r1: i32 = right.0[2 * i + 1].into();
                    l0.expected_fma_(r0, l1.expected_mul_(r1))
                })
            }
        }
    };
}

//i8/u8 to i32
macro_rules! impl_simd_dot_product_iu8_to_i32 {
    ($N:literal, $TwoN:literal) => {
        /// Promote intermediate values to `i32` and then perform accuulation.
        impl<A> SIMDDotProduct<Emulated<u8, $TwoN, A>, Emulated<i8, $TwoN, A>>
            for Emulated<i32, $N, A>
        where
            A: arch::Sealed,
        {
            fn dot_simd(self, left: Emulated<u8, $TwoN, A>, right: Emulated<i8, $TwoN, A>) -> Self {
                self + Self::from_arch_fn(self.1, |i| {
                    let l0: i32 = left.0[4 * i].into();
                    let l1: i32 = left.0[4 * i + 1].into();
                    let l2: i32 = left.0[4 * i + 2].into();
                    let l3: i32 = left.0[4 * i + 3].into();

                    let r0: i32 = right.0[4 * i].into();
                    let r1: i32 = right.0[4 * i + 1].into();
                    let r2: i32 = right.0[4 * i + 2].into();
                    let r3: i32 = right.0[4 * i + 3].into();

                    let a = l0.expected_fma_(r0, l1.expected_mul_(r1));
                    let b = l2.expected_fma_(r2, l3.expected_mul_(r3));
                    a + b
                })
            }
        }

        impl<A> SIMDDotProduct<Emulated<i8, $TwoN, A>, Emulated<u8, $TwoN, A>>
            for Emulated<i32, $N, A>
        where
            A: arch::Sealed,
        {
            fn dot_simd(self, left: Emulated<i8, $TwoN, A>, right: Emulated<u8, $TwoN, A>) -> Self {
                self.dot_simd(right, left)
            }
        }
    };
}

impl_simd_dot_product_i16_to_i32!(8, 16);
impl_simd_dot_product_i16_to_i32!(16, 32);

impl_simd_dot_product_iu8_to_i32!(8, 32);
impl_simd_dot_product_iu8_to_i32!(16, 64);

////////////
// Select //
////////////

impl<T, const N: usize, A> SIMDSelect<Emulated<T, N, A>> for BitMask<N, A>
where
    T: Copy,
    A: arch::Sealed,
    Const<N>: SupportedLaneCount,
    BitMask<N, A>: SIMDMask<Arch = A>,
    Emulated<T, N, A>: SIMDVector<Mask = BitMask<N, A>>,
{
    #[inline(always)]
    fn select(self, x: Emulated<T, N, A>, y: Emulated<T, N, A>) -> Emulated<T, N, A> {
        Emulated::from_arch_fn(self.arch(), |i| {
            if self.get_unchecked(i) {
                x.0[i]
            } else {
                y.0[i]
            }
        })
    }
}

/////////////
// SumTree //
/////////////

macro_rules! impl_sumtree {
    ($T:ty, $N:literal) => {
        impl<A> SIMDSumTree for Emulated<$T, $N, A>
        where
            A: arch::Sealed,
        {
            #[inline(always)]
            fn sum_tree(self) -> $T {
                self.0.tree_reduce(|x, y| x.expected_add_(y))
            }
        }
    };
    ($T:ty, $($N:literal),* $(,)?) => {
        $(impl_sumtree!($T, $N);)*
    };
}

impl_sumtree!(f32, 1, 2, 4, 8, 16);
impl_sumtree!(i32, 4, 8, 16);
impl_sumtree!(u32, 4, 8, 16);

////////////////
// Conversion //
////////////////

macro_rules! impl_from {
    (f16 => f32, $N:literal) => {
        impl<A> From<Emulated<f16, $N, A>> for Emulated<f32, $N, A> {
            #[inline(always)]
            fn from(value: Emulated<f16, $N, A>) -> Self {
                Emulated(value.0.map(|v| v.reference_cast()), value.1)
            }
        }
    };
    ($from:ty => $to:ty, $N:literal) => {
        impl<A> From<Emulated<$from, $N, A>> for Emulated<$to, $N, A> {
            #[inline(always)]
            fn from(value: Emulated<$from, $N, A>) -> Self {
                Emulated(value.0.map(|v| v.into()), value.1)
            }
        }
    };
}

impl_from!(f16 => f32, 1);
impl_from!(f16 => f32, 2);
impl_from!(f16 => f32, 4);
impl_from!(f16 => f32, 8);
impl_from!(f16 => f32, 16);

impl_from!(u8 => i16, 16);
impl_from!(u8 => i16, 32);

impl_from!(i8 => i16, 16);
impl_from!(i8 => i16, 32);

impl_from!(i8 => i32, 1);
impl_from!(i8 => i32, 4);

impl_from!(u8 => i32, 1);
impl_from!(u8 => i32, 4);

/////////////////
// Reinterpret //
/////////////////

macro_rules! impl_little_endian_transmute_cast {
    (<$from:ty, $Nfrom:literal> => <$to:ty, $Nto:literal>) => {
        #[cfg(target_endian = "little")]
        impl<A> SIMDReinterpret<Emulated<$to, $Nto, A>> for Emulated<$from, $Nfrom, A>
        where
            A: arch::Sealed,
        {
            fn reinterpret_simd(self) -> Emulated<$to, $Nto, A> {
                let array = self.0;
                // # SAFETY: This is only ever instantiated with arrays of primitive
                // types that hold no resources, no padding, and are valid for all
                // possible bit-patterns.
                let casted = unsafe { std::mem::transmute::<[$from; $Nfrom], [$to; $Nto]>(array) };
                Emulated(casted, self.1)
            }
        }
    };
}

impl_little_endian_transmute_cast!(<u32, 8> => <i16, 16>);

impl_little_endian_transmute_cast!(<u32, 16> => <u8, 64>);
impl_little_endian_transmute_cast!(<u32, 16> => <i8, 64>);

impl_little_endian_transmute_cast!(<u8, 64> => <u32, 16>);
impl_little_endian_transmute_cast!(<i8, 64> => <u32, 16>);

/////////////
// Casting //
/////////////

macro_rules! impl_cast {
    ($from:ty => $to:ty, $N:literal) => {
        impl<A> SIMDCast<$to> for Emulated<$from, $N, A>
        where
            A: arch::Sealed,
        {
            type Cast = Emulated<$to, $N, A>;
            #[inline(always)]
            fn simd_cast(self) -> Self::Cast {
                Emulated::from_arch_fn(self.arch(), |i| self.0[i].reference_cast())
            }
        }
    };
}

impl_cast!(f16 => f32, 8);
impl_cast!(f16 => f32, 16);

impl_cast!(f32 => f16, 8);
impl_cast!(f32 => f16, 16);

impl_cast!(i32 => f32, 8);

///////////////
// SplitJoin //
///////////////

macro_rules! impl_splitjoin {
    ($type:ty, $N:literal => $N2:literal) => {
        impl<A> SplitJoin for Emulated<$type, $N, A>
        where
            A: Copy,
        {
            type Halved = Emulated<$type, $N2, A>;

            #[inline(always)]
            fn split(self) -> $crate::LoHi<Self::Halved> {
                let $crate::LoHi { lo, hi } = self.0.split();
                let arch = self.1;
                $crate::LoHi::new(Emulated(lo, arch), Emulated(hi, arch))
            }

            #[inline(always)]
            fn join(lohi: $crate::LoHi<Self::Halved>) -> Self {
                Self($crate::LoHi::new(lohi.lo.0, lohi.hi.0).join(), lohi.lo.1)
            }
        }
    };
}

impl_splitjoin!(i8, 32 => 16);
impl_splitjoin!(i8, 64 => 32);

impl_splitjoin!(i16, 16 => 8);
impl_splitjoin!(i16, 32 => 16);

impl_splitjoin!(i32, 8 => 4);
impl_splitjoin!(i32, 16 => 8);

impl_splitjoin!(u8, 32 => 16);
impl_splitjoin!(u8, 64 => 32);

impl_splitjoin!(u32, 8 => 4);
impl_splitjoin!(u32, 16 => 8);
impl_splitjoin!(u64, 4 => 2);

impl_splitjoin!(f32, 16 => 8);
impl_splitjoin!(f32, 8 => 4);

impl_splitjoin!(f16, 16 => 8);

///////////
// Tests //
///////////

#[cfg(test)]
mod test_emulated {
    use half::f16;

    use super::*;
    use crate::{reference::ReferenceScalarOps, test_utils};

    // Test loading logic - ensure that no out of bounds accesses are made.
    // In particular, this is meant to be run under `Miri` to ensure that our guarantees
    // regarding out-of-bounds accesses are honored.
    #[test]
    fn test_load() {
        // Floating Point
        #[cfg(not(miri))] // Miri does not have ph-to-ps conversion.
        test_utils::test_load_simd::<f16, 8, Emulated<f16, 8>>(Scalar);
        test_utils::test_load_simd::<f32, 4, Emulated<f32, 4>>(Scalar);
        test_utils::test_load_simd::<f32, 8, Emulated<f32, 8>>(Scalar);

        // Unsigned Integers
        test_utils::test_load_simd::<u8, 8, Emulated<u8, 8>>(Scalar);
        test_utils::test_load_simd::<u8, 16, Emulated<u8, 16>>(Scalar);

        test_utils::test_load_simd::<u16, 4, Emulated<u16, 4>>(Scalar);
        test_utils::test_load_simd::<u16, 8, Emulated<u16, 8>>(Scalar);
        test_utils::test_load_simd::<u16, 16, Emulated<u16, 16>>(Scalar);

        test_utils::test_load_simd::<u32, 2, Emulated<u32, 2>>(Scalar);
        test_utils::test_load_simd::<u32, 4, Emulated<u32, 4>>(Scalar);
        test_utils::test_load_simd::<u32, 8, Emulated<u32, 8>>(Scalar);

        // Unsigned Integers
        test_utils::test_load_simd::<i8, 8, Emulated<i8, 8>>(Scalar);
        test_utils::test_load_simd::<i8, 16, Emulated<i8, 16>>(Scalar);

        test_utils::test_load_simd::<i16, 4, Emulated<i16, 4>>(Scalar);
        test_utils::test_load_simd::<i16, 8, Emulated<i16, 8>>(Scalar);
        test_utils::test_load_simd::<i16, 16, Emulated<i16, 16>>(Scalar);

        test_utils::test_load_simd::<i32, 2, Emulated<i32, 2>>(Scalar);
        test_utils::test_load_simd::<i32, 4, Emulated<i32, 4>>(Scalar);
        test_utils::test_load_simd::<i32, 8, Emulated<i32, 8>>(Scalar);
    }

    #[test]
    fn test_store() {
        // Floating Point
        #[cfg(not(miri))] // Miri does not have ph-to-ps conversion.
        test_utils::test_store_simd::<f16, 8, Emulated<f16, 8>>(Scalar);
        test_utils::test_store_simd::<f32, 4, Emulated<f32, 4>>(Scalar);
        test_utils::test_store_simd::<f32, 8, Emulated<f32, 8>>(Scalar);

        // Unsigned Integers
        test_utils::test_store_simd::<u8, 8, Emulated<u8, 8>>(Scalar);
        test_utils::test_store_simd::<u8, 16, Emulated<u8, 16>>(Scalar);

        test_utils::test_store_simd::<u16, 4, Emulated<u16, 4>>(Scalar);
        test_utils::test_store_simd::<u16, 8, Emulated<u16, 8>>(Scalar);
        test_utils::test_store_simd::<u16, 16, Emulated<u16, 16>>(Scalar);

        test_utils::test_store_simd::<u32, 2, Emulated<u32, 2>>(Scalar);
        test_utils::test_store_simd::<u32, 4, Emulated<u32, 4>>(Scalar);
        test_utils::test_store_simd::<u32, 8, Emulated<u32, 8>>(Scalar);

        // Unsigned Integers
        test_utils::test_store_simd::<i8, 8, Emulated<i8, 8>>(Scalar);
        test_utils::test_store_simd::<i8, 16, Emulated<i8, 16>>(Scalar);

        test_utils::test_store_simd::<i16, 4, Emulated<i16, 4>>(Scalar);
        test_utils::test_store_simd::<i16, 8, Emulated<i16, 8>>(Scalar);
        test_utils::test_store_simd::<i16, 16, Emulated<i16, 16>>(Scalar);

        test_utils::test_store_simd::<i32, 2, Emulated<i32, 2>>(Scalar);
        test_utils::test_store_simd::<i32, 4, Emulated<i32, 4>>(Scalar);
        test_utils::test_store_simd::<i32, 8, Emulated<i32, 8>>(Scalar);
    }

    // Only test a subset of constructors as all `Emulated` have the same implementation.
    #[test]
    fn test_constructors() {
        test_utils::ops::test_splat::<u8, 64, Emulated<u8, 64>>(Scalar);
        let x = Emulated::<u32, 8>::default(Scalar);
        assert_eq!(x.to_underlying(), [0; 8]);

        let x = Emulated::<u32, 8>::from_underlying(Scalar, [1; 8]);
        assert_eq!(x.to_underlying(), [1; 8]);
    }

    // Wrap inside `Some` for compatibility with optional tests.
    const SC: Option<Scalar> = Some(Scalar);

    macro_rules! test_emulated {
        ($type:ty, $N:literal) => {
            test_utils::ops::test_add!(Emulated<$type, $N>, 0xba37c3f2cf666f87, SC);
            test_utils::ops::test_sub!(Emulated<$type, $N>, 0xeb755abd230e5d80, SC);
            test_utils::ops::test_mul!(Emulated<$type, $N>, 0x0a24ed76a54c3561, SC);
            test_utils::ops::test_fma!(Emulated<$type, $N>, 0xa906c44505abe9ca, SC);
            test_utils::ops::test_minmax!(Emulated<$type, $N>, 0x959522be5234d492, SC);

            test_utils::ops::test_cmp!(Emulated<$type, $N>, 0x9b58e6cbd8330c2d, SC);
            test_utils::ops::test_select!(Emulated<$type, $N>, 0x610aca3aa4d77c0a, SC);
        };
        (unsigned, $type:ty, $N:literal) => {
            test_emulated!($type, $N);

            test_utils::ops::test_bitops!(Emulated<$type, $N>, 0x14fc7841e66bd162, SC);
        };
        (signed, $type:ty, $N:literal) => {
            test_emulated!($type, $N);

            test_utils::ops::test_bitops!(Emulated<$type, $N>, 0x850435f89f86f3b0, SC);
            test_utils::ops::test_abs!(Emulated<$type, $N>, 0x1842a2b86dfd9ecb, SC);
        };
    }

    // Emulated arithmetic.
    test_emulated!(f32, 1);
    test_emulated!(f32, 4);
    test_emulated!(f32, 8);
    test_emulated!(f32, 16);
    // test_emulated!(f64, 8);

    // unsigned integer
    test_emulated!(unsigned, u8, 16);

    test_emulated!(unsigned, u16, 16);
    test_emulated!(unsigned, u16, 32);

    test_emulated!(unsigned, u32, 1);
    test_emulated!(unsigned, u32, 4);
    test_emulated!(unsigned, u32, 8);
    test_emulated!(unsigned, u32, 16);

    test_emulated!(unsigned, u64, 2);
    test_emulated!(unsigned, u64, 4);
    test_emulated!(unsigned, u64, 8);
    test_emulated!(unsigned, u64, 16);

    // signed integer
    test_emulated!(signed, i8, 8);
    test_emulated!(signed, i8, 16);

    test_emulated!(signed, i16, 8);
    test_emulated!(signed, i16, 16);

    test_emulated!(signed, i32, 1);
    test_emulated!(signed, i32, 4);
    test_emulated!(signed, i32, 8);
    test_emulated!(signed, i32, 16);

    test_emulated!(signed, i64, 2);
    test_emulated!(signed, i64, 4);
    test_emulated!(signed, i64, 8);
    test_emulated!(signed, i64, 16);

    // Dot Products
    test_utils::dot_product::test_dot_product!(
        (Emulated<i16, 16>, Emulated<i16, 16>) => Emulated<i32, 8>, 0x3001f05604e96289, SC
    );
    test_utils::dot_product::test_dot_product!(
        (Emulated<i16, 32>, Emulated<i16, 32>) => Emulated<i32, 16>, 0x137ce7a540d9b1a2, SC
    );

    test_utils::dot_product::test_dot_product!(
        (Emulated<u8, 32>, Emulated<i8, 32>) => Emulated<i32, 8>, 0x3001f05604e96289, SC
    );
    test_utils::dot_product::test_dot_product!(
        (Emulated<i8, 32>, Emulated<u8, 32>) => Emulated<i32, 8>, 0x3001f05604e96289, SC
    );
    test_utils::dot_product::test_dot_product!(
        (Emulated<u8, 64>, Emulated<i8, 64>) => Emulated<i32, 16>, 0x3001f05604e96289, SC
    );
    test_utils::dot_product::test_dot_product!(
        (Emulated<i8, 64>, Emulated<u8, 64>) => Emulated<i32, 16>, 0x3001f05604e96289, SC
    );

    // reductions
    test_utils::ops::test_sumtree!(Emulated<f32, 1>, 0x410bad8207a8ccfc, SC);
    test_utils::ops::test_sumtree!(Emulated<f32, 2>, 0xf2fc4e4bbd193493, SC);
    test_utils::ops::test_sumtree!(Emulated<f32, 4>, 0x8034d5a0cd2be14d, SC);
    test_utils::ops::test_sumtree!(Emulated<f32, 8>, 0x0f075940b7e3732c, SC);
    test_utils::ops::test_sumtree!(Emulated<f32, 16>, 0x5b3cb860e3f02d3c, SC);

    test_utils::ops::test_sumtree!(Emulated<i32, 4>, 0xf8c38f70a807e9d2, SC);
    test_utils::ops::test_sumtree!(Emulated<i32, 8>, 0xf8aa4a7e7a273e80, SC);
    test_utils::ops::test_sumtree!(Emulated<i32, 16>, 0x8d1a467fe835a9c5, SC);

    test_utils::ops::test_sumtree!(Emulated<u32, 4>, 0x5e4cffc86a21e90d, SC);
    test_utils::ops::test_sumtree!(Emulated<u32, 8>, 0xf43f19adb43bc611, SC);
    test_utils::ops::test_sumtree!(Emulated<u32, 16>, 0xa43dfe10aa9de860, SC);

    /////////////////
    // conversions //
    /////////////////

    test_utils::ops::test_lossless_convert!(
        Emulated<i8, 16> => Emulated<i16, 16>, 0x1b4f08a8b741d565, SC
    );
    test_utils::ops::test_lossless_convert!(
        Emulated<i8, 32> => Emulated<i16, 32>, 0xdf6f41eb836d4f46, SC
    );

    test_utils::ops::test_lossless_convert!(
        Emulated<i8, 1> => Emulated<i32, 1>, 0x318ceec0e9798353, SC
    );
    test_utils::ops::test_lossless_convert!(
        Emulated<i8, 4> => Emulated<i32, 4>, 0x9f5e1a437f7e7f3f, SC
    );

    test_utils::ops::test_lossless_convert!(
        Emulated<u8, 16> => Emulated<i16, 16>, 0x96611521fed02f98, SC
    );
    test_utils::ops::test_lossless_convert!(
        Emulated<u8, 32> => Emulated<i16, 32>, 0x6749d3aa94effa04, SC
    );

    test_utils::ops::test_lossless_convert!(
        Emulated<u8, 1> => Emulated<i32, 1>, 0x669cbd5c7bf6184e, SC
    );
    test_utils::ops::test_lossless_convert!(
        Emulated<u8, 4> => Emulated<i32, 4>, 0x75929494c5d333d0, SC
    );

    ///////////
    // Casts //
    ///////////

    test_utils::ops::test_cast!(Emulated<f16, 8> => Emulated<f32, 8>, 0x1e9e37b58fb3f1a8, SC);
    test_utils::ops::test_cast!(Emulated<f16, 16> => Emulated<f32, 16>, 0xd2b068a9bf3f9d24, SC);

    test_utils::ops::test_cast!(Emulated<f32, 8> => Emulated<f16, 8>, 0xe9d2dd426d89699d, SC);
    test_utils::ops::test_cast!(Emulated<f32, 16> => Emulated<f16, 16>, 0x2b637e21afd9ef6c, SC);

    test_utils::ops::test_cast!(Emulated<i32, 8> => Emulated<f32, 8>, 0x2b08e8ec7e49323b, SC);
}
