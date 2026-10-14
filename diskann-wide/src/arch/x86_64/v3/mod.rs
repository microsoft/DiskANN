/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{Scalar, Target, Target1, Target2, Target3, get_or_set_architecture};
use crate::{
    Architecture, SIMDVector,
    arch::{self, Dispatched1, Dispatched2, Dispatched3, FTarget1, FTarget2, FTarget3, Hidden},
    lifetime::AddLifetime,
};

// float16
pub mod f16x8_;
pub use f16x8_::f16x8;

pub mod f16x16_;
pub use f16x16_::f16x16;

// float32
pub mod f32x4_;
pub use f32x4_::f32x4;

pub mod f32x8_;
pub use f32x8_::f32x8;

pub mod f32x16_;
pub use f32x16_::f32x16;

// int8
pub mod i8x16_;
pub use i8x16_::i8x16;

pub mod i8x32_;
pub use i8x32_::i8x32;

pub mod i8x64_;
pub use i8x64_::i8x64;

// int16
pub mod i16x8_;
pub use i16x8_::i16x8;

pub mod i16x16_;
pub use i16x16_::i16x16;

pub mod i16x32_;
pub use i16x32_::i16x32;

// int32
pub mod i32x4_;
pub use i32x4_::i32x4;

pub mod i32x8_;
pub use i32x8_::i32x8;

pub mod i32x16_;
pub use i32x16_::i32x16;

// uint8
pub mod u8x16_;
pub use u8x16_::u8x16;

pub mod u8x32_;
pub use u8x32_::u8x32;

pub mod u8x64_;
pub use u8x64_::u8x64;

// uint32
pub mod u32x4_;
pub use u32x4_::u32x4;

pub mod u32x8_;
pub use u32x8_::u32x8;

pub mod u32x16_;
pub use u32x16_::u32x16;

// uint64
pub mod u64x2_;
pub use u64x2_::u64x2;

pub mod u64x4_;
pub use u64x4_::u64x4;

pub mod u64x8_;
pub use u64x8_::u64x8;

// Masks
pub mod masks;

// Conversions between intrinsics
mod conversion;

////////////
// x86-v3 //
////////////

/// An [`Architecture`] supporting all the requirements of the
/// [`x86-64-v3`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct V3(Hidden);

impl arch::sealed::Sealed for V3 {}

macro_rules! v3_features {
    ($($args:tt)*) => {
        #[target_feature(enable = "avx2,avx,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave")]
        $($args)*
    }
}

impl V3 {
    /// Construct a new `V3` architecture struct.
    ///
    /// # Safety
    ///
    /// To avoid undefined behavior, this function must only be called on a machine that
    /// supports the feature-set necessary for the `x86-64-v3` target CPU.
    ///
    /// These are:
    ///
    /// * `avx`
    /// * `avx2`
    /// * `bmi1`
    /// * `bmi2`
    /// * `f16c`
    /// * `fma`
    /// * `lzcnt`
    /// * `movbe`
    /// * `osxsave`
    pub const unsafe fn new() -> Self {
        Self(Hidden)
    }

    /// Return an instance of `Self` if it is safe to do so. Otherwise, return `None`.
    pub fn new_checked() -> Option<Self> {
        // SAFETY: `get_or_set_architecture!()` performs the necessary checks.
        unsafe { Self::new_checked_with(get_or_set_architecture!()) }
    }

    /// Retarget for a more conservative architecture.
    pub fn retarget(self) -> Scalar {
        Scalar::new()
    }

    /// Return an instance of `Self` if it is safe to do so. Otherwise, return `None`.
    ///
    /// Unlike `new_checked`, this function checks target features every time and is
    /// provided to allow tests to be decoupled from the architecture dispatch switch.
    #[cfg(test)]
    pub(crate) fn new_checked_uncached() -> Option<Self> {
        // SAFETY: `arch_number()` performs the necessary checks.
        match unsafe { Self::new_checked_with(super::arch_number()) } {
            Some(v) => Some(v),
            None => {
                // Check if the user specifically requested at least this architecture
                // level. If so, panic.
                //
                // SAFETY: We do not actually use the resulting architecture object. We
                // just check if the user specified level would have instantiated it.
                if unsafe { Self::new_checked_with(super::test_arch_number()) }.is_some() {
                    panic!(
                        "V3 architecture was requested but is not compatible on the current hardware"
                    );
                } else {
                    None
                }
            }
        }
    }

    /// # Safety
    ///
    /// The architecture number `version` can only be `ARCH_V3` if runtime checks have
    /// verified that it is safe to do so.
    unsafe fn new_checked_with(version: u64) -> Option<Self> {
        if version >= super::ARCH_V3 {
            // SAFETY: Architecture resolution says we support at least V3.
            Some(unsafe { Self::new() })
        } else {
            None
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_with<F, R>(self, f: F) -> R
        where
            F: Target<Self, R>,
        {
            f.run(self)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        #[inline]
        pub(super) unsafe fn run_with_inline<F, R>(self, f: F) -> R
        where
            F: Target<Self, R>,
        {
            f.run(self)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_with_1<F, T0, R>(self, f: F, x0: T0) -> R
        where
            F: Target1<Self, R, T0>,
        {
            f.run(self, x0)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        #[inline]
        pub(super) unsafe fn run_with_1_inline<F, T0, R>(self, f: F, x0: T0) -> R
        where
            F: Target1<Self, R, T0>,
        {
            f.run(self, x0)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_with_2<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
        where
            F: Target2<Self, R, T0, T1>,
        {
            f.run(self, x0, x1)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        #[inline]
        pub(super) unsafe fn run_with_2_inline<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
        where
            F: Target2<Self, R, T0, T1>,
        {
            f.run(self, x0, x1)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_with_3<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
        where
            F: Target3<Self, R, T0, T1, T2>,
        {
            f.run(self, x0, x1, x2)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        #[inline]
        pub(super) unsafe fn run_with_3_inline<F, T0, T1, T2, R>(
            self,
            f: F,
            x0: T0,
            x1: T1,
            x2: T2,
        ) -> R
        where
            F: Target3<Self, R, T0, T1, T2>,
        {
            f.run(self, x0, x1, x2)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_function_with_1<F, T0, R>(self, x0: T0::Of<'_>) -> R
        where
            T0: AddLifetime,
            F: for<'a> FTarget1<Self, R, T0::Of<'a>>,
        {
            F::run(self, x0)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_function_with_2<F, T0, T1, R>(
            self,
            x0: T0::Of<'_>,
            x1: T1::Of<'_>,
        ) -> R
        where
            T0: AddLifetime,
            T1: AddLifetime,
            F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>,
        {
            F::run(self, x0, x1)
        }
    }

    v3_features! {
        // # Safety
        //
        // The current machine must have all the documented features required for V3.
        pub(super) unsafe fn run_function_with_3<F, T0, T1, T2, R>(
            self,
            x0: T0::Of<'_>,
            x1: T1::Of<'_>,
            x2: T2::Of<'_>,
        ) -> R
        where
            T0: AddLifetime,
            T1: AddLifetime,
            T2: AddLifetime,
            F: for<'a, 'b, 'c> FTarget3<Self, R, T0::Of<'a>, T1::Of<'b>, T2::Of<'c>>,
        {
            F::run(self, x0, x1, x2)
        }
    }
}

impl Architecture for V3 {
    arch::maskdef!();
    arch::typedef!();

    fn level() -> arch::Level {
        arch::Level::v3()
    }

    #[inline(always)]
    fn run<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { Self::run_with(self, f) }
    }

    #[inline(always)]
    fn run_inline<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { Self::run_with_inline(self, f) }
    }

    #[inline(always)]
    fn run1<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_1(f, x0) }
    }

    #[inline(always)]
    fn run1_inline<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_1_inline(f, x0) }
    }

    #[inline(always)]
    fn run2<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_2(f, x0, x1) }
    }

    #[inline(always)]
    fn run2_inline<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_2_inline(f, x0, x1) }
    }

    #[inline(always)]
    fn run3<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_3(f, x0, x1, x2) }
    }

    #[inline(always)]
    fn run3_inline<F, T0, T1, T2, R>(self, f: F, x0: T0, x1: T1, x2: T2) -> R
    where
        F: Target3<Self, R, T0, T1, T2>,
    {
        // SAFETY: The existence of `self` implies that we are V3 compatible and therefore
        // have all the required features.
        unsafe { self.run_with_3_inline(f, x0, x1, x2) }
    }

    fn dispatch1<F, R, T0>(self) -> Dispatched1<R, T0>
    where
        T0: AddLifetime,
        F: for<'a> FTarget1<Self, R, T0::Of<'a>>,
    {
        let f: unsafe fn(Self, T0::Of<'_>) -> R = Self::run_function_with_1::<F, T0, R>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // A `V3` architecture. Additionally, since `V3` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compatible with `Hidden`.
        unsafe { arch::hide1(f) }
    }

    fn dispatch2<F, R, T0, T1>(self) -> Dispatched2<R, T0, T1>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>,
    {
        let f: unsafe fn(Self, T0::Of<'_>, T1::Of<'_>) -> R =
            Self::run_function_with_2::<F, T0, T1, R>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // A `V3` architecture. Additionally, since `V3` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compatible with `Hidden`.
        unsafe { arch::hide2(f) }
    }

    fn dispatch3<F, R, T0, T1, T2>(self) -> Dispatched3<R, T0, T1, T2>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        T2: AddLifetime,
        F: for<'a, 'b, 'c> FTarget3<Self, R, T0::Of<'a>, T1::Of<'b>, T2::Of<'c>>,
    {
        let f: unsafe fn(Self, T0::Of<'_>, T1::Of<'_>, T2::Of<'_>) -> R =
            Self::run_function_with_3::<F, T0, T1, T2, R>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // A `V3` architecture. Additionally, since `V3` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compatible with `Hidden`.
        unsafe { arch::hide3(f) }
    }
}

///////////////////////
// Custom Intrinsics //
///////////////////////

impl V3 {
    /// Multiply unsigned bytes from `x` by signed bytes from `y`, then add adjacent
    /// products with saturation to `i16::MIN..=i16::MAX`.
    ///
    /// Each output lane corresponds to one adjacent input pair, in order.
    /// There is no accumulator.
    ///
    /// See: [`std::arch::x86_64::_mm256_maddubs_epi16`].
    #[inline(always)]
    pub fn mm256_maddubs_epi16(self, x: u8x32, y: i8x32) -> i16x16 {
        // SAFETY: V3 provides AVX2.
        i16x16::from_underlying(self, unsafe {
            std::arch::x86_64::_mm256_maddubs_epi16(x.to_underlying(), y.to_underlying())
        })
    }
}

// Conversions
impl From<V3> for Scalar {
    fn from(_: V3) -> Scalar {
        Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;

    #[test]
    fn test_mm256_maddubs_epi16() {
        if let Some(arch) = V3::new_checked_uncached() {
            let check = move |x: &[u8], y: &[i8]| {
                let got = arch
                    .mm256_maddubs_epi16(
                        u8x32::from_array(arch, x.try_into().unwrap()),
                        i8x32::from_array(arch, y.try_into().unwrap()),
                    )
                    .to_array();
                let expected: [i16; 16] = core::array::from_fn(|i| {
                    let even = i16::from(x[2 * i]) * i16::from(y[2 * i]);
                    let odd = i16::from(x[2 * i + 1]) * i16::from(y[2 * i + 1]);
                    even.saturating_add(odd)
                });
                assert_eq!(got, expected, "x = {x:?}, y = {y:?}");
            };
            check(&[u8::MAX; 32], &[i8::MAX; 32]);
            check(&[u8::MAX; 32], &[i8::MIN; 32]);
            test_utils::driver::drive_binary(&check, (32, 32), 0x7fc14e39a2d06b85);
        }
    }
}
