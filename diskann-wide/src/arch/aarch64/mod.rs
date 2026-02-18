/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */

use crate::{
    Architecture, SIMDVector,
    arch::{
        self, AddLifetime, Dispatched1, Dispatched2, Dispatched3, FTarget1, FTarget2, FTarget3,
        Hidden, Scalar, Target, Target1, Target2, Target3,
    },
};

mod algorithms;

pub mod f16x4_;
pub use f16x4_::f16x4;

pub mod f16x8_;
pub use f16x8_::f16x8;

pub mod f32x2_;
pub use f32x2_::f32x2;

pub mod f32x4_;
pub use f32x4_::f32x4;

// Unsigned
pub mod u8x8_;
pub use u8x8_::u8x8;

pub mod u8x16_;
pub use u8x16_::u8x16;

pub mod u16x8_;
pub use u16x8_::u16x8;

pub mod u32x4_;
pub use u32x4_::u32x4;

pub mod u64x2_;
pub use u64x2_::u64x2;

// Signed
pub mod i8x8_;
pub use i8x8_::i8x8;

pub mod i8x16_;
pub use i8x16_::i8x16;

pub mod i16x8_;
pub use i16x8_::i16x8;

pub mod i32x4_;
pub use i32x4_::i32x4;

pub mod i64x2_;
pub use i64x2_::i64x2;

// Extra wide types.
pub mod double;

pub use double::f16x16;

pub use double::f32x8;
pub use double::f32x16;

pub use double::i8x32;
pub use double::i8x64;

pub use double::i16x16;
pub use double::i16x32;

pub use double::i32x8;
pub use double::i32x16;

pub use double::u8x32;
pub use double::u8x64;

pub use double::u32x8;
pub use double::u32x16;

pub use double::u64x4;

// Internal helpers
mod macros;
mod masks;

// The ordering is `Scalar < Neon`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub(super) enum LevelInner {
    Scalar,
    Neon,
}

/////////////
// Current //
/////////////

cfg_if::cfg_if! {
    if #[cfg(all(target_feature = "neon", target_feature = "dotprod"))] {
        pub type Current = Neon;

        pub const fn current() -> Current {
            // SAFETY: Requirements are checked at compile time.
            unsafe { Neon::new() }
        }
    } else {
        pub type Current = Scalar;

        pub const fn current() -> Current {
            Scalar::new()
        }
    }
}

/////////////////
// Dispatching //
/////////////////

pub fn dispatch<T, R>(f: T) -> R
where
    T: Target<Scalar, R> + Target<Neon, R>,
{
    if let Some(arch) = Neon::new_checked() {
        arch.run(f)
    } else {
        Scalar::new().run(f)
    }
}

pub fn dispatch_no_features<T, R>(f: T) -> R
where
    T: Target<Scalar, R> + Target<Neon, R>,
{
    dispatch(f)
}

pub fn dispatch1<T, T0, R>(f: T, x0: T0) -> R
where
    T: Target1<Scalar, R, T0> + Target1<Neon, R, T0>,
{
    if let Some(arch) = Neon::new_checked() {
        arch.run1(f, x0)
    } else {
        Scalar::new().run1(f, x0)
    }
}

pub fn dispatch1_no_features<T, T0, R>(f: T, x0: T0) -> R
where
    T: Target1<Scalar, R, T0> + Target1<Neon, R, T0>,
{
    dispatch1(f, x0)
}

pub fn dispatch2<T, T0, T1, R>(f: T, x0: T0, x1: T1) -> R
where
    T: Target2<Scalar, R, T0, T1> + Target2<Neon, R, T0, T1>,
{
    if let Some(arch) = Neon::new_checked() {
        arch.run2(f, x0, x1)
    } else {
        Scalar::new().run2(f, x0, x1)
    }
}

pub fn dispatch2_no_features<T, T0, T1, R>(f: T, x0: T0, x1: T1) -> R
where
    T: Target2<Scalar, R, T0, T1> + Target2<Neon, R, T0, T1>,
{
    dispatch2(f, x0, x1)
}

pub fn dispatch3<T, T0, T1, T2, R>(f: T, x0: T0, x1: T1, x2: T2) -> R
where
    T: Target3<Scalar, R, T0, T1, T2> + Target3<Neon, R, T0, T1, T2>,
{
    if let Some(arch) = Neon::new_checked() {
        arch.run3(f, x0, x1, x2)
    } else {
        Scalar::new().run3(f, x0, x1, x2)
    }
}

pub fn dispatch3_no_features<T, T0, T1, T2, R>(f: T, x0: T0, x1: T1, x2: T2) -> R
where
    T: Target3<Scalar, R, T0, T1, T2> + Target3<Neon, R, T0, T1, T2>,
{
    dispatch3(f, x0, x1, x2)
}

//////////////////
// Architecture //
//////////////////

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neon(Hidden);

impl arch::sealed::Sealed for Neon {}

impl Neon {
    /// Construct a new `Neon` architecture struct.
    ///
    /// # Safety
    ///
    /// To avoid undefined behavior, this function must only be called on a machine that
    /// supports following features
    ///
    /// * `neon`
    /// * `dotprod`
    pub const unsafe fn new() -> Self {
        Self(Hidden)
    }

    /// Construct a new `Neon` architecture if it is safe to do so on the current hardware.
    pub fn new_checked() -> Option<Self> {
        // This check here ensures that if we ever switch to dynamically dispatching to
        // `Neon` that we do not forget to update `new_checked`.
        if cfg!(all(target_feature = "neon", target_feature = "dotprod")) {
            // SAFETY: The compile-time feature check above ensures we do not accidentally
            // return an unsafe instance of `Self`.
            Some(unsafe { Self::new() })
        } else {
            None
        }
    }

    /// Retarget the [`Scalar`] architecture.
    pub const fn retarget(self) -> Scalar {
        Scalar::new()
    }

    fn run_function_with_1<F, T0, R>(self, x0: T0::Of<'_>) -> R
    where
        T0: AddLifetime,
        F: for<'a> FTarget1<Self, R, T0::Of<'a>>,
    {
        F::run(self, x0)
    }

    fn run_function_with_2<F, T0, T1, R>(self, x0: T0::Of<'_>, x1: T1::Of<'_>) -> R
    where
        T0: AddLifetime,
        T1: AddLifetime,
        F: for<'a, 'b> FTarget2<Self, R, T0::Of<'a>, T1::Of<'b>>,
    {
        F::run(self, x0, x1)
    }

    fn run_function_with_3<F, T0, T1, T2, R>(
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

impl From<Neon> for Scalar {
    fn from(neon: Neon) -> Self {
        neon.retarget()
    }
}

impl arch::Architecture for Neon {
    arch::maskdef!();
    arch::typedef!();

    fn level() -> arch::Level {
        arch::Level::neon()
    }

    fn run<F, R>(self, f: F) -> R
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

    fn run2<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
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
    fn run_inline<F, R>(self, f: F) -> R
    where
        F: Target<Self, R>,
    {
        f.run(self)
    }

    #[inline(always)]
    fn run1_inline<F, T0, R>(self, f: F, x0: T0) -> R
    where
        F: Target1<Self, R, T0>,
    {
        f.run(self, x0)
    }

    #[inline(always)]
    fn run2_inline<F, T0, T1, R>(self, f: F, x0: T0, x1: T1) -> R
    where
        F: Target2<Self, R, T0, T1>,
    {
        f.run(self, x0, x1)
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
        let f: unsafe fn(Self, T0::Of<'_>) -> R = Self::run_function_with_1::<F, _, _>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // a `Neon` architecture. Additionally, since `Neon` is a `Copy` zero-sized type,
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
            Self::run_function_with_2::<F, _, _, _>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // a `Neon` architecture. Additionally, since `Neon` is a `Copy` zero-sized type,
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
            Self::run_function_with_3::<F, _, _, _, _>;

        // SAFETY: The presence of `self` as an argument attests that it is safe to construct
        // a `Neon` architecture. Additionally, since `Neon` is a `Copy` zero-sized type,
        // it is safe to wink into existence and is ABI compatible with `Hidden`.
        unsafe { arch::hide3(f) }
    }
}

///////////
// Tests //
///////////

/// Return `Some(Neon)` if the Neon architecture should be used for testing.
///
/// If the environment variable `WIDE_TEST_MIN_ARCH` is set, this uses the configured
/// architecture with the following mapping:
///
/// * `all` or `neon`: Run the Neon backend
/// * `scalar`: Skip the Neon backend (returns `None`)
///
/// If the variable is not set, this defaults to [`Neon::new_checked()`].
#[cfg(test)]
pub(super) fn test_neon() -> Option<Neon> {
    match crate::get_test_arch() {
        Some(arch) => {
            if arch == "all" || arch == "neon" {
                match Neon::new_checked() {
                    Some(v) => Some(v),
                    None => panic!(
                        "Neon architecture was requested but is not available on the current target"
                    ),
                }
            } else if arch == "scalar" {
                None
            } else {
                panic!("Unrecognized test architecture: \"{arch}\"");
            }
        }
        None => Neon::new_checked(),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Architecture;

    struct TestOp;

    impl Target<Scalar, &'static str> for TestOp {
        fn run(self, _: Scalar) -> &'static str {
            "scalar"
        }
    }

    impl Target1<Scalar, String, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str) -> String {
            format!("scalar: {}", x0)
        }
    }

    impl Target2<Scalar, String, &str, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str, x1: &str) -> String {
            format!("scalar: {}, {}", x0, x1)
        }
    }

    impl Target3<Scalar, String, &str, &str, &str> for TestOp {
        fn run(self, _: Scalar, x0: &str, x1: &str, x2: &str) -> String {
            format!("scalar: {}, {}, {}", x0, x1, x2)
        }
    }

    impl Target<Neon, &'static str> for TestOp {
        fn run(self, _: Neon) -> &'static str {
            "neon"
        }
    }

    impl Target1<Neon, String, &str> for TestOp {
        fn run(self, _: Neon, x0: &str) -> String {
            format!("neon: {}", x0)
        }
    }

    impl Target2<Neon, String, &str, &str> for TestOp {
        fn run(self, _: Neon, x0: &str, x1: &str) -> String {
            format!("neon: {}, {}", x0, x1)
        }
    }

    impl Target3<Neon, String, &str, &str, &str> for TestOp {
        fn run(self, _: Neon, x0: &str, x1: &str, x2: &str) -> String {
            format!("neon: {}, {}, {}", x0, x1, x2)
        }
    }

    #[test]
    fn test_dispatch() {
        let expected = if Neon::new_checked().is_some() {
            "neon"
        } else {
            "scalar"
        };

        assert_eq!(dispatch(TestOp), expected);
        assert_eq!(dispatch_no_features(TestOp), expected);

        assert_eq!(dispatch1(TestOp, "foo"), format!("{expected}: foo"));
        assert_eq!(
            dispatch1_no_features(TestOp, "foo"),
            format!("{expected}: foo")
        );

        assert_eq!(
            dispatch2(TestOp, "foo", "bar"),
            format!("{expected}: foo, bar")
        );
        assert_eq!(
            dispatch2_no_features(TestOp, "foo", "bar"),
            format!("{expected}: foo, bar")
        );

        assert_eq!(
            dispatch3(TestOp, "foo", "bar", "baz"),
            format!("{expected}: foo, bar, baz")
        );
        assert_eq!(
            dispatch3_no_features(TestOp, "foo", "bar", "baz"),
            format!("{expected}: foo, bar, baz"),
        );
    }

    #[test]
    fn test_run() {
        if let Some(arch) = test_neon() {
            let mut x = 10;
            let y: &str = arch.run(|| {
                x += 10;
                "foo"
            });
            assert_eq!(x, 20);
            assert_eq!(y, "foo");
        }
    }

    #[test]
    fn test_level_ordering() {
        let scalar = Scalar::level();
        let neon = Neon::level();

        // Scalar < Neon
        assert!(scalar < neon);
        assert!(neon > scalar);

        // Equality
        assert_eq!(scalar, Scalar::level());
        assert_eq!(neon, Neon::level());

        // Not equal across levels
        assert_ne!(scalar, neon);
    }
}
