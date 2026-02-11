/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![expect(non_camel_case_types)]

use half::f16;

use super::{
    Dispatched1, Dispatched2, Dispatched3, FTarget1, FTarget2, FTarget3, Target, Target1, Target2,
    Target3,
};
use crate::{Architecture, Emulated, SIMDVector, arch, lifetime::AddLifetime};

pub type f16x8 = Emulated<f16, 8>;
pub type f16x16 = Emulated<f16, 16>;

pub type f32x4 = Emulated<f32, 4>;
pub type f32x8 = Emulated<f32, 8>;
pub type f32x16 = Emulated<f32, 16>;

pub type i8x16 = Emulated<i8, 16>;
pub type i8x32 = Emulated<i8, 32>;
pub type i8x64 = Emulated<i8, 64>;

pub type i16x8 = Emulated<i16, 8>;
pub type i16x16 = Emulated<i16, 16>;
pub type i16x32 = Emulated<i16, 32>;

pub type i32x4 = Emulated<i32, 4>;
pub type i32x8 = Emulated<i32, 8>;
pub type i32x16 = Emulated<i32, 16>;

pub type u8x16 = Emulated<u8, 16>;
pub type u8x32 = Emulated<u8, 32>;
pub type u8x64 = Emulated<u8, 64>;

pub type u32x4 = Emulated<u32, 4>;
pub type u32x8 = Emulated<u32, 8>;
pub type u32x16 = Emulated<u32, 16>;

pub type u64x2 = Emulated<u64, 2>;
pub type u64x4 = Emulated<u64, 4>;

/// A safe architecture that is guaranteed to be compatible with the machine that
/// a Rust program was compiled for.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Scalar;

impl Scalar {
    pub const fn new() -> Self {
        Self
    }
}

impl arch::sealed::Sealed for Scalar {}

impl Architecture for Scalar {
    arch::maskdef!();
    arch::typedef!();

    fn level() -> arch::Level {
        arch::Level::scalar()
    }

    fn run<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>,
    {
        f.run(self)
    }

    #[inline(always)]
    fn run_inline<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>,
    {
        f.run(self)
    }

    fn run1<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>,
    {
        f.run(self, x0)
    }

    #[inline(always)]
    fn run1_inline<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>,
    {
        f.run(self, x0)
    }

    fn run2<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>,
    {
        f.run(self, x0, x1)
    }

    #[inline(always)]
    fn run2_inline<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>,
    {
        f.run(self, x0, x1)
    }

    fn run3<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>,
    {
        f.run(self, x0, x1, x2)
    }

    #[inline(always)]
    fn run3_inline<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>,
    {
        f.run(self, x0, x1, x2)
    }

    fn dispatch1<F, R, T0>(self) -> Dispatched1<R, T0>
    where
        T0: AddLifetime,
        F: for<'a> FTarget1<Self, R, T0::Of<'a>>,
    {
        let f: fn(Self, T0::Of<'_>) -> R = |me, x0| F::run(me, x0);

        // SAFETY: It's always safe to construct the `Scalar` architecture. Additionally,
        // since `Scalar` is a `Copy` zero-sized type, it is safe to wink into existence
        // and is ABI compattible with `Hidden`.
        unsafe { arch::hide1(f) }
    }

    fn dispatch2<F, R, T0, T1>(self) -> Dispatched2<R, T0, T1>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>,
    {
        let f: fn(Self, T0::Of<'_>, T1::Of<'_>) -> R = |me, x0, x1| F::run(me, x0, x1);

        // SAFETY: It's always safe to construct the `Scalar` architecture. Additionally,
        // since `Scalar` is a `Copy` zero-sized type, it is safe to wink into existence
        // and is ABI compattible with `Hidden`.
        unsafe { arch::hide2(f) }
    }

    fn dispatch3<F, R, T0, T1, T2>(self) -> Dispatched3<R, T0, T1, T2>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        T2: AddLifetime,
        F: for<'a, 'b, 'c> FTarget3<Self, R, T0::Of<'a>, T1::Of<'b>, T2::Of<'c>>,
    {
        let f: fn(Self, T0::Of<'_>, T1::Of<'_>, T2::Of<'_>) -> R =
            |me, x0, x1, x2| F::run(me, x0, x1, x2);

        // SAFETY: It's always safe to construct the `Scalar` architecture. Additionally,
        // since `Scalar` is a `Copy` zero-sized type, it is safe to wink into existence
        // and is ABI compattible with `Hidden`.
        unsafe { arch::hide3(f) }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_arch() {
        // Ensure that `Scalar::new()` is `const`.
        const ARCH: Scalar = Scalar::new();

        let mut x = 10;
        let y: &str = ARCH.run(|| {
            x += 10;
            "foo"
        });

        assert_eq!(x, 20);
        assert_eq!(y, "foo");

        // Execute this at run time so code-coverage counts it.
        let arch = Scalar::new();
        assert_eq!(arch, ARCH);
    }
}
