/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    LoHi, SIMDAbs, SIMDDotProduct, SIMDMask, SIMDMinMax, SIMDMulAdd, SIMDPartialEq, SIMDPartialOrd,
    SIMDSelect, SIMDSumTree, SIMDVector, SplitJoin,
};

#[derive(Debug, Clone, Copy)]
pub struct Doubled<T>(pub(crate) T, pub(crate) T);

impl<T> Doubled<T> {
    #[inline(always)]
    pub fn new(lo: T, hi: T) -> Self {
        Self(lo, hi)
    }
}

// This needs to be a macro because we don't have `generic_const_exprs` so we cannot do
// computation with lanes.
macro_rules! double_vector {
    ($T:ty, $N:literal, $repr:ty) => {
        impl $crate::SIMDVector for $crate::doubled::Doubled<$repr> {
            type Arch = <$repr as $crate::SIMDVector>::Arch;
            type Scalar = $T;
            type Underlying = (
                <$repr as $crate::SIMDVector>::Underlying,
                <$repr as $crate::SIMDVector>::Underlying,
            );
            const LANES: usize = $N;
            type ConstLanes = $crate::Const<$N>;
            type Mask = $crate::doubled::Doubled<<$repr as $crate::SIMDVector>::Mask>;
            const EMULATED: bool = false;

            #[inline(always)]
            fn arch(self) -> Self::Arch {
                self.0.arch()
            }

            #[inline(always)]
            fn to_underlying(self) -> Self::Underlying {
                (self.0.to_underlying(), self.1.to_underlying())
            }

            #[inline(always)]
            fn from_underlying(arch: Self::Arch, value: Self::Underlying) -> Self {
                Self(
                    <$repr as $crate::SIMDVector>::from_underlying(arch, value.0),
                    <$repr as $crate::SIMDVector>::from_underlying(arch, value.1),
                )
            }

            #[inline(always)]
            fn default(arch: Self::Arch) -> Self {
                let v = <$repr as $crate::SIMDVector>::default(arch);
                Self(v, v)
            }

            #[inline(always)]
            fn to_array(self) -> [$T; $N] {
                use $crate::splitjoin::{LoHi, SplitJoin};

                <[$T; $N]>::join(LoHi::new(self.0.to_array(), self.1.to_array()))
            }

            #[inline(always)]
            fn from_array(arch: Self::Arch, x: [$T; $N]) -> Self {
                use $crate::splitjoin::{LoHi, SplitJoin};

                let LoHi { lo, hi } = x.split();
                Self(
                    <$repr as $crate::SIMDVector>::from_array(arch, lo),
                    <$repr as $crate::SIMDVector>::from_array(arch, hi),
                )
            }

            #[inline(always)]
            fn splat(arch: Self::Arch, value: Self::Scalar) -> Self {
                let v = <$repr as $crate::SIMDVector>::splat(arch, value);
                Self(v, v)
            }

            #[inline(always)]
            fn num_lanes() -> usize {
                $N
            }

            #[inline(always)]
            unsafe fn load_simd(arch: Self::Arch, ptr: *const Self::Scalar) -> Self {
                Self(
                    // SAFETY: The caller asserts this pointer access is safe.
                    unsafe { <$repr as $crate::SIMDVector>::load_simd(arch, ptr) },
                    // SAFETY: The caller asserts this pointer access is safe.
                    unsafe {
                        <$repr as $crate::SIMDVector>::load_simd(arch, ptr.wrapping_add({ $N / 2 }))
                    },
                )
            }

            #[inline(always)]
            unsafe fn load_simd_masked_logical(
                arch: Self::Arch,
                ptr: *const Self::Scalar,
                mask: Self::Mask,
            ) -> Self {
                Self(
                    // SAFETY: The caller asserts this pointer access is safe.
                    unsafe {
                        <$repr as $crate::SIMDVector>::load_simd_masked_logical(arch, ptr, mask.0)
                    },
                    // SAFETY: The caller asserts this pointer access is safe.
                    unsafe {
                        <$repr as $crate::SIMDVector>::load_simd_masked_logical(
                            arch,
                            ptr.wrapping_add({ $N / 2 }),
                            mask.1,
                        )
                    },
                )
            }

            #[inline(always)]
            unsafe fn store_simd(self, ptr: *mut Self::Scalar) {
                // SAFETY: The caller asserts this pointer access is safe.
                unsafe { self.0.store_simd(ptr) };
                // SAFETY: The caller asserts this pointer access is safe.
                unsafe { self.1.store_simd(ptr.wrapping_add({ $N / 2 })) };
            }

            #[inline(always)]
            unsafe fn store_simd_masked_logical(
                self,
                ptr: *mut <Self as $crate::SIMDVector>::Scalar,
                mask: <Self as $crate::SIMDVector>::Mask,
            ) {
                // SAFETY: The caller asserts this pointer access is safe.
                unsafe { self.0.store_simd_masked_logical(ptr, mask.0) };
                // SAFETY: The caller asserts this pointer access is safe.
                unsafe {
                    self.1
                        .store_simd_masked_logical(ptr.wrapping_add({ $N / 2 }), mask.1)
                };
            }
        }
    };
}

impl<T: std::ops::Add<Output = T>> std::ops::Add for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl<T: std::ops::Sub<Output = T>> std::ops::Sub for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl<T: std::ops::Mul<Output = T>> std::ops::Mul for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<T: std::ops::BitAnd<Output = T>> std::ops::BitAnd for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0, self.1 & rhs.1)
    }
}

impl<T: std::ops::Not<Output = T>> std::ops::Not for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        Self(!self.0, !self.1)
    }
}

impl<T: std::ops::BitOr<Output = T>> std::ops::BitOr for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0, self.1 | rhs.1)
    }
}

impl<T: std::ops::BitXor<Output = T>> std::ops::BitXor for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0, self.1 ^ rhs.1)
    }
}

impl<T: std::ops::Shr<Output = T>> std::ops::Shr for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(self.0 >> rhs.0, self.1 >> rhs.1)
    }
}

impl<T: std::ops::Shl<Output = T>> std::ops::Shl for Doubled<T> {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(self.0 << rhs.0, self.1 << rhs.1)
    }
}

impl<T: SIMDMinMax> SIMDMinMax for Doubled<T> {
    #[inline(always)]
    fn min_simd(self, rhs: Self) -> Self {
        Self(self.0.min_simd(rhs.0), self.1.min_simd(rhs.1))
    }
    #[inline(always)]
    fn min_simd_standard(self, rhs: Self) -> Self {
        Self(
            self.0.min_simd_standard(rhs.0),
            self.1.min_simd_standard(rhs.1),
        )
    }
    #[inline(always)]
    fn max_simd(self, rhs: Self) -> Self {
        Self(self.0.max_simd(rhs.0), self.1.max_simd(rhs.1))
    }
    #[inline(always)]
    fn max_simd_standard(self, rhs: Self) -> Self {
        Self(
            self.0.max_simd_standard(rhs.0),
            self.1.max_simd_standard(rhs.1),
        )
    }
}

impl<T: SIMDAbs> SIMDAbs for Doubled<T> {
    #[inline(always)]
    fn abs_simd(self) -> Self {
        Self(self.0.abs_simd(), self.1.abs_simd())
    }
}

impl<T: SIMDMulAdd> SIMDMulAdd for Doubled<T> {
    #[inline(always)]
    fn mul_add_simd(self, rhs: Self, accumulator: Self) -> Self {
        Self(
            self.0.mul_add_simd(rhs.0, accumulator.0),
            self.1.mul_add_simd(rhs.1, accumulator.1),
        )
    }
}

impl<T> SIMDSumTree for Doubled<T>
where
    T: SIMDSumTree + std::ops::Add<Output = T>,
    Doubled<T>: SIMDVector<Scalar = T::Scalar>,
{
    #[inline(always)]
    fn sum_tree(self) -> <Self as SIMDVector>::Scalar {
        (self.0 + self.1).sum_tree()
    }
}

impl<T: SIMDPartialEq> SIMDPartialEq for Doubled<T>
where
    Doubled<T>: SIMDVector<Mask = Doubled<T::Mask>>,
    Doubled<T::Mask>: SIMDMask,
{
    #[inline(always)]
    fn eq_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.eq_simd(other.0), self.1.eq_simd(other.1))
    }
    #[inline(always)]
    fn ne_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.ne_simd(other.0), self.1.ne_simd(other.1))
    }
}

impl<T: SIMDPartialOrd> SIMDPartialOrd for Doubled<T>
where
    Doubled<T>: SIMDVector<Mask = Doubled<T::Mask>>,
    Doubled<T::Mask>: SIMDMask,
{
    #[inline(always)]
    fn lt_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.lt_simd(other.0), self.1.lt_simd(other.1))
    }
    #[inline(always)]
    fn le_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.le_simd(other.0), self.1.le_simd(other.1))
    }
    #[inline(always)]
    fn gt_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.gt_simd(other.0), self.1.gt_simd(other.1))
    }
    #[inline(always)]
    fn ge_simd(self, other: Self) -> Self::Mask {
        Doubled(self.0.ge_simd(other.0), self.1.ge_simd(other.1))
    }
}

impl<T, L, R> SIMDDotProduct<Doubled<L>, Doubled<R>> for Doubled<T>
where
    T: SIMDVector,
    L: SIMDVector,
    R: SIMDVector,
    Self: SIMDVector,
    Doubled<L>: SIMDVector,
    Doubled<R>: SIMDVector,
    T: SIMDDotProduct<L, R>,
{
    #[inline(always)]
    fn dot_simd(self, left: Doubled<L>, right: Doubled<R>) -> Self {
        Self(
            self.0.dot_simd(left.0, right.0),
            self.1.dot_simd(left.1, right.1),
        )
    }
}

impl<T> SplitJoin for Doubled<T> {
    type Halved = T;

    #[inline(always)]
    fn split(self) -> LoHi<T> {
        LoHi::new(self.0, self.1)
    }

    #[inline(always)]
    fn join(lohi: LoHi<T>) -> Self {
        Self(lohi.lo, lohi.hi)
    }
}

//////////////
// SIMDMask //
//////////////

macro_rules! double_mask {
    ($N:literal, $repr:ty) => {
        impl $crate::SIMDMask for $crate::doubled::Doubled<$repr> {
            type Arch = <$repr as $crate::SIMDMask>::Arch;
            type Underlying = (
                <$repr as $crate::SIMDMask>::Underlying,
                <$repr as $crate::SIMDMask>::Underlying,
            );
            type BitMask = $crate::BitMask<$N, Self::Arch>;
            const LANES: usize = $N;
            const ISBITS: bool = false;

            #[inline(always)]
            fn arch(self) -> Self::Arch {
                self.0.arch()
            }

            #[inline(always)]
            fn to_underlying(self) -> Self::Underlying {
                (self.0.to_underlying(), self.1.to_underlying())
            }

            #[inline(always)]
            fn from_underlying(arch: Self::Arch, value: Self::Underlying) -> Self {
                Self(
                    <$repr as $crate::SIMDMask>::from_underlying(arch, value.0),
                    <$repr as $crate::SIMDMask>::from_underlying(arch, value.1),
                )
            }

            fn get_unchecked(&self, i: usize) -> bool {
                if i < { $N / 2 } {
                    self.0.get_unchecked(i)
                } else {
                    self.1.get_unchecked(i - { $N / 2 })
                }
            }

            fn keep_first(arch: Self::Arch, i: usize) -> Self {
                let lo = <$repr>::keep_first(arch, i);
                let hi = <$repr>::keep_first(arch, i.saturating_sub({ $N / 2 }));
                Self(lo, hi)
            }
        }

        impl From<$crate::doubled::Doubled<$repr>>
            for $crate::BitMask<$N, <$repr as $crate::SIMDMask>::Arch>
        {
            #[inline(always)]
            fn from(value: $crate::doubled::Doubled<$repr>) -> Self {
                use $crate::splitjoin::{LoHi, SplitJoin};

                Self::join(LoHi::new(value.0.into(), value.1.into()))
            }
        }

        impl From<$crate::BitMask<$N, <$repr as $crate::SIMDMask>::Arch>>
            for $crate::doubled::Doubled<$repr>
        {
            #[inline(always)]
            fn from(value: $crate::BitMask<$N, <$repr as $crate::SIMDMask>::Arch>) -> Self {
                use $crate::splitjoin::{LoHi, SplitJoin};

                let LoHi { lo, hi } = value.split();
                Self(lo.into(), hi.into())
            }
        }
    };
}

impl<T, M> SIMDSelect<Doubled<T>> for Doubled<M>
where
    M: SIMDMask + SIMDSelect<T>,
    T: SIMDVector<Mask = M>,
    Doubled<M>: SIMDMask,
    Doubled<T>: SIMDVector<Mask = Self>,
{
    #[inline(always)]
    fn select(self, x: Doubled<T>, y: Doubled<T>) -> Doubled<T> {
        Doubled(self.0.select(x.0, y.0), self.1.select(x.1, y.1))
    }
}

/// This is needed to work around Rust's orphan rule when implementing bit shifts with
/// a scalar argument.
macro_rules! double_scalar_shift {
    (Doubled<$type:ty>) => {
        impl std::ops::Shl<<$type as $crate::SIMDVector>::Scalar>
            for $crate::doubled::Doubled<$type>
        {
            type Output = Self;
            fn shl(self, rhs: <$type as $crate::SIMDVector>::Scalar) -> Self {
                Self(self.0 << rhs, self.1 << rhs)
            }
        }

        impl std::ops::Shr<<$type as $crate::SIMDVector>::Scalar>
            for $crate::doubled::Doubled<$type>
        {
            type Output = Self;
            fn shr(self, rhs: <$type as $crate::SIMDVector>::Scalar) -> Self {
                Self(self.0 >> rhs, self.1 >> rhs)
            }
        }
    };
}

pub(crate) use double_mask;
pub(crate) use double_vector;

pub(crate) use double_scalar_shift;
