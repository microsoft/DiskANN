/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(crate) trait ReferenceShifts: Copy {
    fn expected_shr_(self, rhs: Self) -> Self;
    fn expected_shl_(self, rhs: Self) -> Self;
}

macro_rules! impl_shifts_unsigned {
    ($type:ty) => {
        impl ReferenceShifts for $type {
            #[inline(always)]
            fn expected_shr_(self, rhs: Self) -> Self {
                if (rhs as usize) >= 8 * std::mem::size_of::<Self>() {
                    0
                } else {
                    self >> rhs
                }
            }

            #[inline(always)]
            fn expected_shl_(self, rhs: Self) -> Self {
                if (rhs as usize) >= 8 * std::mem::size_of::<Self>() {
                    0
                } else {
                    self << rhs
                }
            }
        }
    };
}

macro_rules! impl_shifts_signed {
    ($type:ty) => {
        impl ReferenceShifts for $type {
            #[inline(always)]
            fn expected_shr_(self, rhs: Self) -> Self {
                if rhs < 0 || rhs >= ((8 * std::mem::size_of::<Self>()) as $type) {
                    if self < 0 { -1 } else { 0 }
                } else {
                    self >> rhs
                }
            }

            #[inline(always)]
            fn expected_shl_(self, rhs: Self) -> Self {
                if rhs < 0 || (rhs >= (8 * std::mem::size_of::<Self>()) as $type) {
                    0
                } else {
                    self << rhs
                }
            }
        }
    };
}

impl_shifts_unsigned!(u8);
impl_shifts_unsigned!(u16);
impl_shifts_unsigned!(u32);
impl_shifts_unsigned!(u64);

impl_shifts_signed!(i8);
impl_shifts_signed!(i16);
impl_shifts_signed!(i32);
impl_shifts_signed!(i64);

/// This is the ground truth for how operations behave.
///
/// It is expected that all SIMD backends yield the exact same results as those expressed
/// here.
pub(crate) trait ReferenceScalarOps: Copy {
    fn expected_add_(self, rhs: Self) -> Self;
    fn expected_sub_(self, rhs: Self) -> Self;
    fn expected_mul_(self, rhs: Self) -> Self;
    fn expected_fma_(self, rhs: Self, acc: Self) -> Self;
    fn expected_max_(self, rhs: Self) -> Self;
    fn expected_min_(self, rhs: Self) -> Self;
}

/// Scalar ops for integers.
///
/// Arithmetic is always done with wrapping.
/// Unlike normal Rust, we do not check for overflow in Debug builds.
macro_rules! impl_expected_ops_for_integers {
    ($type:ty) => {
        impl ReferenceScalarOps for $type {
            #[inline(always)]
            fn expected_add_(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
            #[inline(always)]
            fn expected_sub_(self, rhs: Self) -> Self {
                self.wrapping_sub(rhs)
            }
            #[inline(always)]
            fn expected_mul_(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
            #[inline(always)]
            fn expected_fma_(self, rhs: Self, acc: Self) -> Self {
                self.wrapping_mul(rhs).wrapping_add(acc)
            }
            #[inline(always)]
            fn expected_max_(self, rhs: Self) -> Self {
                self.max(rhs)
            }
            #[inline(always)]
            fn expected_min_(self, rhs: Self) -> Self {
                self.min(rhs)
            }
        }
    };
}

/// Scalar ops for floats.
macro_rules! impl_expected_ops_for_floats {
    ($type:ty) => {
        impl ReferenceScalarOps for $type {
            #[inline(always)]
            fn expected_add_(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn expected_sub_(self, rhs: Self) -> Self {
                self - rhs
            }
            #[inline(always)]
            fn expected_mul_(self, rhs: Self) -> Self {
                self * rhs
            }
            /// FMA **must** be done with only a single rounding.
            #[inline(always)]
            fn expected_fma_(self, rhs: Self, acc: Self) -> Self {
                self.mul_add(rhs, acc)
            }

            #[inline(always)]
            fn expected_max_(self, rhs: Self) -> Self {
                self.max(rhs)
            }
            #[inline(always)]
            fn expected_min_(self, rhs: Self) -> Self {
                self.min(rhs)
            }
        }
    };
}

impl_expected_ops_for_integers!(u8);
impl_expected_ops_for_integers!(u16);
impl_expected_ops_for_integers!(u32);
impl_expected_ops_for_integers!(u64);

impl_expected_ops_for_integers!(i8);
impl_expected_ops_for_integers!(i16);
impl_expected_ops_for_integers!(i32);
impl_expected_ops_for_integers!(i64);

// float16 arithmetic operations are not supported natively in hardware.
// impl_expected_ops_for_floats!(f16);
impl_expected_ops_for_floats!(f32);
impl_expected_ops_for_floats!(f64);

///////////////////
// Reference Abs //
///////////////////

pub(crate) trait ReferenceAbs: Copy {
    fn expected_abs_(self) -> Self;
}

macro_rules! impl_abs {
    (integer, $T:ty) => {
        impl ReferenceAbs for $T {
            fn expected_abs_(self) -> Self {
                if self == Self::MIN {
                    self
                } else {
                    self.abs()
                }
            }
        }
    };
    (integer, $($T:ty),* $(,)?) => {
        $(impl_abs!(integer, $T);)*
    };
    ($T:ty) => {
        impl ReferenceAbs for $T {
            fn expected_abs_(self) -> Self {
                self.abs()
            }
        }
    };
    ($($T:ty),* $(,)?) => {
        $(impl_abs!($T);)*
    };
}

impl_abs!(integer, i8, i16, i32, i64);
impl_abs!(f32, f64);

/////////////////////
// Reference Casts //
/////////////////////

/// Reference implementations for numeric conversion.
pub(crate) trait ReferenceCast<To> {
    fn reference_cast(self) -> To;
}

impl ReferenceCast<f32> for half::f16 {
    #[cfg(miri)]
    fn reference_cast(self) -> f32 {
        self.to_f32_const()
    }
    #[cfg(not(miri))]
    #[inline(always)]
    fn reference_cast(self) -> f32 {
        self.into()
    }
}

impl ReferenceCast<half::f16> for f32 {
    #[cfg(miri)]
    fn reference_cast(self) -> half::f16 {
        half::f16::from_f32_const(self)
    }
    #[cfg(not(miri))]
    #[inline(always)]
    fn reference_cast(self) -> half::f16 {
        half::f16::from_f32(self)
    }
}

impl ReferenceCast<f32> for i32 {
    fn reference_cast(self) -> f32 {
        self as f32
    }
}

impl ReferenceCast<u8> for i16 {
    fn reference_cast(self) -> u8 {
        self as u8
    }
}

impl ReferenceCast<i8> for i16 {
    fn reference_cast(self) -> i8 {
        self as i8
    }
}

/// Perform a Miri-safe conversion from `f16` to `f32`.
///
/// This has the same semantics as `f16.into()`, but will not use an intrinsic if running
/// under Miri.
#[inline(always)]
pub fn cast_f16_to_f32(x: half::f16) -> f32 {
    x.reference_cast()
}

/// Perform a Miri-safe cast from `f32` to `f16`.
///
/// This has the same semantics as `f16::from_f32`, but will not use an intrinsic if running
/// under Miri.
///
/// This function rounds to nearest.
#[inline(always)]
pub fn cast_f32_to_f16(x: f32) -> half::f16 {
    x.reference_cast()
}

/// A recursive implementation of reduction by pairwise halving.
///
/// We need to use macros to stamp out implementation for progressively wider arrays because
/// of Rust's current limitations in expressions involving const generics.
pub(crate) trait TreeReduce {
    type Scalar: Copy;
    fn tree_reduce<F>(self, op: F) -> Self::Scalar
    where
        F: Fn(Self::Scalar, Self::Scalar) -> Self::Scalar;
}

impl<T> TreeReduce for [T; 1]
where
    T: Copy,
{
    type Scalar = T;
    #[inline(always)]
    fn tree_reduce<F>(self, _op: F) -> Self::Scalar
    where
        F: Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    {
        self[0]
    }
}

impl<T> TreeReduce for [T; 2]
where
    T: Copy,
{
    type Scalar = T;
    #[inline(always)]
    fn tree_reduce<F>(self, op: F) -> Self::Scalar
    where
        F: Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
    {
        op(self[0], self[1])
    }
}

/// A note on the implementation:
///
/// Generally, the compiler is quite good at optimizing recursive patterns like this and in
/// general, we expect this to compile down to straight assembly without stack-allocating
/// the intermediate arrays.
///
/// An example can be seen here: https://godbolt.org/z/sMcdGbbPG
macro_rules! impl_tree_reduce {
    ($N:literal) => {
        impl<T> TreeReduce for [T; $N]
        where
            T: Copy,
        {
            type Scalar = T;
            #[inline(always)]
            fn tree_reduce<F>(self, op: F) -> Self::Scalar
            where
                F: Fn(Self::Scalar, Self::Scalar) -> Self::Scalar,
            {
                const N2: usize = { $N / 2 };
                let half: [T; N2] = core::array::from_fn(|i| op(self[i], self[N2 + i]));
                half.tree_reduce(op)
            }
        }
    };
}

impl_tree_reduce!(4);
impl_tree_reduce!(8);
impl_tree_reduce!(16);
