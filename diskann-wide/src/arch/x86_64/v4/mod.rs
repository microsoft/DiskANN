/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # AVX-512 Support
//!
//! Access to AVX-512 instrinsics is provided by the [`V4`] backend.
//!
//! This backend corresponds to the features required for the [`V3`] architecture in addition
//! to:
//!
//!
//! * `avx512f`
//! * `avx512bw`
//! * `avx512cd`
//! * `avx512vl`
//! * `avx512dq`
//! * `avx512vnni`
//! * `avx512bitalg` (upcoming - pending CI resources)
//! * `avx512vpopcntdq` (upcoming - pending CI resources)
//!
//! Note that the requirement of `avx512vnni`, `avx512bitalg`, and `avx512vpopcntdq` extend
//! beyond the traditional requirements of `x86-64-v4`. These are included to assist with
//! quantized distance functions and to avoid adding an additional backend solely for these
//! features.
//!
//! ## Miri Support
//!
//! [Miri](https://github.com/rust-lang/miri) does not support much if any of the AVX-512
//! feature set, and does not pass the [`is_x86_feature_detected!`] for AVX-512 related
//! features.
//!
//! However, Miri is a great tool for assisting with writing kernels when it comes to
//! validating bounds-checking. The [`V4`] supports running under Miri with a few caveats.
//! First, when using `#[cfg(miri)]`, the type aliases such as [`f32x4`] will be mapped
//! to emulated variants (i.e., [Emulated<f32, 4, V4>`). Use of the standard SIMD functions
//! like arithmetic, loading, storing, etc. will execute as expected, enabling Miri assisted
//! bounds checking for loads and stores.
//!
//! Since Miri does not support the AVX-512 runtime features, the constructor
//! [`V4::new_checked_miri`] can be used to safely construct a [`V4`] architecture if the
//! runtime supports one, or if Miri is present.
//!
//! Note that even when running under Miri, we require the runtime CPU/Miri emulated
//! environment to support [`V3`] since `V3: From<V4>` and we need to maintain the safety
//! of that implementation.
//!
//! The accuracy of the emulated behavior is done via "best effort". Please report any
//! divergences.
//!
//! ### Developer Notes
//!
//! Miri emulation is tackled by redefining the SIMD identifiers like `f32x4`. Under Miri,
//! these are aliased to `Emulated` instead of the native intrinsic types.
//!
//! This means that regardless of Miri, the native intrinsic types are still defined and
//! useable. They are just not mapped via `V4 as Architecture`. Thus, when working in the
//! submodules like `f32x8_` that reference types like `f32x4`, make sure that the symbol
//! `f32x4` is imported via `f32x4_::f32x4` to ensure compilation when using Miri.
//!
//! Additionally, if a direct intrinsic is needed that does not map to the usual traits
//! exposed by this crate, it can be added as an inherent method of the [`V4`] architecture.
//! Be sure to include and test a Miri-compliant emulated version of the intrinsic.

use super::{Scalar, Target, Target1, Target2, Target3, V3, get_or_set_architecture};
use crate::{
    Architecture, SIMDVector,
    arch::{self, Dispatched1, Dispatched2, Dispatched3, FTarget1, FTarget2, FTarget3, Hidden},
    lifetime::AddLifetime,
};

macro_rules! maybe_miri {
    ($mod:ident, $type:ident, $T:ty, $N:literal) => {
        #[cfg(miri)]
        #[allow(non_camel_case_types)]
        pub type $type = crate::Emulated<$T, $N, V4>;

        #[cfg(not(miri))]
        pub use $mod::$type;
    };
}

// float16
pub mod f16x8_;
maybe_miri!(f16x8_, f16x8, half::f16, 8);

pub mod f16x16_;
maybe_miri!(f16x16_, f16x16, half::f16, 16);

// float32
pub mod f32x4_;
maybe_miri!(f32x4_, f32x4, f32, 4);

pub mod f32x8_;
maybe_miri!(f32x8_, f32x8, f32, 8);

pub mod f32x16_;
maybe_miri!(f32x16_, f32x16, f32, 16);

// int8
pub mod i8x16_;
maybe_miri!(i8x16_, i8x16, i8, 16);

pub mod i8x32_;
maybe_miri!(i8x32_, i8x32, i8, 32);

pub mod i8x64_;
maybe_miri!(i8x64_, i8x64, i8, 64);

// int16
pub mod i16x8_;
maybe_miri!(i16x8_, i16x8, i16, 8);

pub mod i16x16_;
maybe_miri!(i16x16_, i16x16, i16, 16);

pub mod i16x32_;
maybe_miri!(i16x32_, i16x32, i16, 32);

// int32
pub mod i32x4_;
maybe_miri!(i32x4_, i32x4, i32, 4);

pub mod i32x8_;
maybe_miri!(i32x8_, i32x8, i32, 8);

pub mod i32x16_;
maybe_miri!(i32x16_, i32x16, i32, 16);

// uint8
pub mod u8x16_;
maybe_miri!(u8x16_, u8x16, u8, 16);

pub mod u8x32_;
maybe_miri!(u8x32_, u8x32, u8, 32);

pub mod u8x64_;
maybe_miri!(u8x64_, u8x64, u8, 64);

// uint32
pub mod u32x4_;
maybe_miri!(u32x4_, u32x4, u32, 4);

pub mod u32x8_;
maybe_miri!(u32x8_, u32x8, u32, 8);

pub mod u32x16_;
maybe_miri!(u32x16_, u32x16, u32, 16);

// uint64
pub mod u64x2_;
maybe_miri!(u64x2_, u64x2, u64, 2);

pub mod u64x4_;
maybe_miri!(u64x4_, u64x4, u64, 4);

// Conversions between intrinsics
mod conversion;

/////////////////
// x86-v4-plus //
/////////////////

/// An [`Architecture`] supporting all the requirements of the
/// [`x86-64-v4`](https://en.wikipedia.org/wiki/X86-64#Microarchitecture_levels) with the
/// additional requirement of
///
/// * `avx512vnni`
/// * `avx512bitalg`
/// * `avx512vpopcntdq`
///
/// Overall, the implications and requirements of `V4` are those of [`V3`] plus:
///
/// * `avx512f`
/// * `avx512bw`
/// * `avx512cd`
/// * `avx512vl`
/// * `avx512dq`
/// * `avx512vnni`
/// * `avx512bitalg` (upcoming - pending CI resources)
/// * `avx512vpopcntdq` (upcoming - pending CI resources)
///
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct V4(Hidden);

impl arch::Sealed for V4 {}

// `miri` does not have the ability to dynamically determing micro-architecture. Thus, to
// run V4 kernels under `Miri` without compiling the whole application of `x86-64-v4`, we
// drop the `target_features`.
//
// This is mostly okay because under `Miri` we use emulated operations in any case.
//
// We lose some ability for `Miri` to identify unsuitable target-features, but that wouldn't
// have worked anyways.
macro_rules! v4_features {
    ($($args:tt)*) => {
        #[cfg_attr(
            not(miri),
            target_feature(
                enable = "avx2,avx,bmi1,bmi2,f16c,fma,lzcnt,movbe,xsave,avx512f,avx512bw,avx512cd,avx512vl,avx512dq,avx512vnni"
            )
        )]
        $($args)*
    }
}

impl V4 {
    /// Construct a new `V4` architecture struct.
    ///
    /// # Safety
    ///
    /// To avoid undefined behavior, this function must only be called on a machine that
    /// supports the feature-set described in the top level [`V4`] documentation.
    pub const unsafe fn new() -> Self {
        Self(Hidden)
    }

    /// Return an instance of `Self` if it is safe to do so. Otherwise, return `None`.
    pub fn new_checked() -> Option<Self> {
        // SAFETY: `get_or_set_architecture!()` performs the necessary checks.
        unsafe { Self::new_checked_with(get_or_set_architecture!()) }
    }

    /// If running under Miri, return `Self`, which will use AVX-512 emulation to help
    /// with debugging tests. Otherwise, returns the results of [`Self::new_checked`].
    pub fn new_checked_miri() -> Option<Self> {
        if cfg!(miri) {
            // Since `V3` is reachable through `V4` - so we inherit the checking of `V3`.
            V3::new_checked().map(|_| {
                // SAFETY: With Miri enabled, AVX-512 emulation is used.
                unsafe { Self::new() }
            })
        } else {
            Self::new_checked()
        }
    }

    /// Retarget for a more conservative architecture.
    pub fn retarget(self) -> V3 {
        // SAFETY: `V4` is a superset of `V3` - so an instance of `V4` asserts the present.
        unsafe { V3::new() }
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
                        "V4 architecture was requested but is not compatible on the current hardare"
                    );
                } else {
                    None
                }
            }
        }
    }

    /// # Safety
    ///
    /// The architecture number `version` can only be `ARCH_V4` with runtime checks having
    /// verified that it is safe to do so.
    unsafe fn new_checked_with(version: u64) -> Option<Self> {
        if version >= super::ARCH_V4 {
            // SAFETY: Architecture resolution says we support at least V4.
            Some(unsafe { Self::new() })
        } else {
            None
        }
    }

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

    v4_features! {
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

impl Architecture for V4 {
    arch::maskdef!();
    arch::typedef!();

    fn level() -> arch::Level {
        arch::Level::v4()
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
        let f: unsafe fn(Self, T0::Of<'_>) -> R = Self::run_function_with_1::<F, _, _>;

        // SAFETY: The present of `self` as an argument attests that it is safe to construct
        // A `V4` architecture. Additionally, since `V4` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compattible with `Hidden`.
        unsafe { arch::hide1(f) }
    }

    fn dispatch2<F, R, T0, T1>(self) -> Dispatched2<R, T0, T1>
    where
        T0: AddLifetime,
        T1: AddLifetime,
        F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>,
    {
        let f: unsafe fn(Self, T0::Of<'_>, T1::Of<'_>) -> R =
            Self::run_function_with_2::<F, _, _, _>;

        // SAFETY: The present of `self` as an argument attests that it is safe to construct
        // A `V4` architecture. Additionally, since `V4` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compattible with `Hidden`.
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
            Self::run_function_with_3::<F, _, _, _, _>;

        // SAFETY: The present of `self` as an argument attests that it is safe to construct
        // A `V4` architecture. Additionally, since `V4` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compattible with `Hidden`.
        unsafe { arch::hide3(f) }
    }
}

// Conversions
impl From<V4> for Scalar {
    fn from(_: V4) -> Scalar {
        Scalar
    }
}

impl From<V4> for V3 {
    fn from(arch: V4) -> V3 {
        arch.retarget()
    }
}
