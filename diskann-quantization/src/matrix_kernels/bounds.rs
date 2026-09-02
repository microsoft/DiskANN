/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A debug-assisting wrapper for [`usize`] that only exists under tests or when debug
/// assertions are enabled.
///
/// This is used as a means of tracking sizes and lengths of slices and matrices in memory
/// to help matrix kernel implementations keep accesses in bounds.
///
/// On release builds, this becomes a zero-sized type to remove run-time overhead.
///
/// For purposes of documentation when writing kernels, a [`Bound`] is **always** the
/// correct size or length of an object, whether or not tracking is enabled.
///
/// Unsafe code must **not** rely on [`Bound`]s being checked in order to uphold a safety
/// contract unless such code is:
///
/// * Gated by `cfg(test)`.
/// * The unsafety is exclusively due to relationships between [`Bound`].
/// * The method being called is known to exhaustively check all [`Bound`]-based invariants.
///
/// In these cases, special `checked_*` methods can be introduced to reduce the number of
/// `unsafe` blocks in tests.
///
/// # Checking Bounds
///
/// Users should use the macros [`check_eq`], [`check_lt`], [`check_le`], [`check_gt`] and
/// [`check_ge`] to perform checks with a [`Bound`] on the left-hand side.
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub(super) struct Bound(inner::Bound);

#[cfg(not(any(test, debug_assertions)))]
const _: () = assert!(std::mem::size_of::<Bound>() == 0);

impl Bound {
    /// Construct a new [`Bound`].
    pub(super) const fn new(length: usize) -> Self {
        Self(inner::Bound::new(length))
    }

    /// Construct a new [`Bound`] from the results of `f`.
    ///
    /// This function can be used when the computation of the [`Bound`] cannot be reliably
    /// elided by the compiler when [`Bound`]s are disabled.
    ///
    /// The closure `f` will not be called when [`Bound`]s are disabled.
    pub(super) fn from_fn<F>(f: F) -> Self
    where
        F: FnOnce() -> usize,
    {
        Self(inner::Bound::from_fn(f))
    }

    /// Perform the computation of the value of the [`Bound`] if bounds are tracked.
    ///
    /// If [`Bound`]s are not tracked, the closure `f` is not called.
    pub(super) fn with<F>(self, f: F)
    where
        F: FnOnce(usize),
    {
        self.0.with(f)
    }

    /// Return the tracked value of the [`Bound`].
    #[cfg(test)]
    pub(super) fn value(self) -> usize {
        self.0.value()
    }

    // This should be called through the macros instead of being called directly.
    #[track_caller]
    pub(super) fn __check<T>(self, check: Check, expected: T, msg: Option<std::fmt::Arguments<'_>>)
    where
        T: IntoBound,
    {
        self.0.__check(check.0, expected.into_bound().0, msg)
    }
}

impl std::ops::Mul for Bound {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }
}

impl std::ops::Add for Bound {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0.add(rhs.0))
    }
}

impl std::ops::Sub for Bound {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0.sub(rhs.0))
    }
}

/// The kind of check operation to be performed. This type should not be used directly,
/// instead, use the [`check_eq`] style macros instead.
#[derive(Debug, Clone, Copy)]
pub(super) struct Check(inner::Check);

impl Check {
    pub(super) const fn eq() -> Self {
        Self(inner::Check::eq())
    }

    pub(super) const fn lt() -> Self {
        Self(inner::Check::lt())
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "this completes the comparison API")
    )]
    pub(super) const fn le() -> Self {
        Self(inner::Check::le())
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "this completes the comparison API")
    )]
    pub(super) const fn gt() -> Self {
        Self(inner::Check::gt())
    }

    pub(super) const fn ge() -> Self {
        Self(inner::Check::ge())
    }
}

/// Enable simple types to be converted to [`Bound`]s.
pub(super) trait IntoBound {
    fn into_bound(self) -> Bound;
}

impl IntoBound for usize {
    fn into_bound(self) -> Bound {
        Bound::new(self)
    }
}

impl IntoBound for std::num::NonZeroUsize {
    fn into_bound(self) -> Bound {
        Bound::new(self.get())
    }
}

impl IntoBound for super::num::DimK {
    fn into_bound(self) -> Bound {
        Bound::new(self.value().get())
    }
}

impl IntoBound for Bound {
    fn into_bound(self) -> Bound {
        self
    }
}

macro_rules! check_eq {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(eq, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(eq, $lhs, $rhs, $($arg)+)
    };
}

macro_rules! check_lt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(lt, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(lt, $lhs, $rhs, $($arg)+)
    };
}

#[cfg_attr(
    not(test),
    expect(unused_macros, reason = "this completes the comparison API")
)]
macro_rules! check_le {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(le, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(le, $lhs, $rhs, $($arg)+)
    };
}

#[cfg_attr(
    not(test),
    expect(unused_macros, reason = "this completes the comparison API")
)]
macro_rules! check_gt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(gt, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(gt, $lhs, $rhs, $($arg)+)
    };
}

macro_rules! check_ge {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(ge, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)*) => {
        $crate::matrix_kernels::bounds::__assert!(ge, $lhs, $rhs, $($arg)*)
    };
}

macro_rules! __assert {
    ($op:ident, $lhs:expr, $rhs:expr $(,)?) => {
        if cfg!(any(test, debug_assertions)) {
            ($lhs).__check(
                $crate::matrix_kernels::bounds::Check::$op(),
                $rhs,
                None,
            )
        }
    };
    ($op:ident, $lhs:expr, $rhs:expr, $($arg:tt)+) => {
        if cfg!(any(test, debug_assertions)) {
            ($lhs).__check(
                $crate::matrix_kernels::bounds::Check::$op(),
                $rhs,
                Some(format_args!($($arg)+)),
            )
        }
    };
}

pub(super) use __assert;
pub(super) use check_eq;
pub(super) use check_ge;

#[allow(unused_imports, reason = "this completes the comparison API")]
pub(super) use check_gt;

#[allow(unused_imports, reason = "this completes the comparison API")]
pub(super) use check_le;

#[allow(unused_imports, reason = "this completes the comparison API")]
pub(super) use check_lt;

#[cfg(any(test, debug_assertions))]
mod inner {
    #[derive(Debug, Clone, Copy)]
    pub(super) enum Check {
        Eq,
        Lt,
        Le,
        Gt,
        Ge,
    }

    impl Check {
        pub(super) const fn eq() -> Self {
            Self::Eq
        }

        pub(super) const fn lt() -> Self {
            Self::Lt
        }

        pub(super) const fn le() -> Self {
            Self::Le
        }

        pub(super) const fn gt() -> Self {
            Self::Gt
        }

        pub(super) const fn ge() -> Self {
            Self::Ge
        }

        fn as_str(&self) -> &'static str {
            match self {
                Self::Eq => "equal to",
                Self::Lt => "less than",
                Self::Le => "less than or equal to",
                Self::Gt => "greater than",
                Self::Ge => "greater than or equal to",
            }
        }

        fn check(self, lhs: usize, rhs: usize, message: Option<std::fmt::Arguments<'_>>) {
            let passed = match self {
                Self::Eq => lhs == rhs,
                Self::Lt => lhs < rhs,
                Self::Le => lhs <= rhs,
                Self::Gt => lhs > rhs,
                Self::Ge => lhs >= rhs,
            };

            if !passed {
                if let Some(message) = message {
                    panic!(
                        "expected {} to be {} {} -- {}",
                        lhs,
                        self.as_str(),
                        rhs,
                        message
                    );
                } else {
                    panic!("expected {} to be {} {}", lhs, self.as_str(), rhs);
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct Bound(usize);

    impl Bound {
        pub(super) const fn new(length: usize) -> Self {
            Self(length)
        }

        pub(super) fn from_fn<F>(f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self::new(f())
        }

        pub(super) fn with<F>(self, f: F)
        where
            F: FnOnce(usize),
        {
            f(self.0)
        }

        #[cfg(test)]
        pub(super) fn value(self) -> usize {
            self.0
        }

        #[track_caller]
        pub(super) fn __check(
            self,
            check: Check,
            expected: Self,
            message: Option<std::fmt::Arguments<'_>>,
        ) {
            check.check(self.0, expected.0, message)
        }
    }

    impl std::ops::Mul for Bound {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0.mul(rhs.0))
        }
    }

    impl std::ops::Add for Bound {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self::new(self.0.add(rhs.0))
        }
    }

    impl std::ops::Sub for Bound {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0.sub(rhs.0))
        }
    }
}

#[cfg(not(any(test, debug_assertions)))]
mod inner {
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Check(());

    impl Check {
        pub(super) const fn eq() -> Self {
            Self(())
        }

        pub(super) const fn lt() -> Self {
            Self(())
        }

        pub(super) const fn le() -> Self {
            Self(())
        }

        pub(super) const fn gt() -> Self {
            Self(())
        }

        pub(super) const fn ge() -> Self {
            Self(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct Bound(());

    impl Bound {
        pub(super) const fn new(_length: usize) -> Self {
            Self(())
        }

        pub(super) fn from_fn<F>(_f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self(())
        }

        pub(super) fn with<F>(self, _f: F)
        where
            F: FnOnce(usize),
        {
        }

        pub(super) fn __check(
            self,
            _check: Check,
            _expected: Self,
            _msg: Option<std::fmt::Arguments<'_>>,
        ) {
        }
    }

    impl std::ops::Mul for Bound {
        type Output = Self;

        fn mul(self, _rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Add for Bound {
        type Output = Self;

        fn add(self, _rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Sub for Bound {
        type Output = Self;

        fn sub(self, _rhs: Self) -> Self {
            Self(())
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::matrix_kernels::{num::DimK, test_util::panic_message_for};

    // These tests specifically test the checked flavor since bounds are enabled under
    // `cfg(test)`.
    #[test]
    fn test_bound() {
        for i in 0..10 {
            let b = Bound::new(i);
            assert_eq!(b.value(), i);

            let b = Bound::from_fn(|| i);
            assert_eq!(b.value(), i);

            let mut called = false;
            b.with(|v| {
                assert_eq!(v, i);
                called = true;
            });
            assert!(called);

            assert_eq!((b * b).value(), i * i);
            assert_eq!((b + b).value(), i + i);

            assert_eq!((b - b).value(), 0);
            if i > 1 {
                assert_eq!((b - Bound::new(1)).value(), i - 1);
            }
        }
    }

    #[test]
    fn test_into_bound() {
        for i in 1..10 {
            let v = std::num::NonZeroUsize::new(i).unwrap();

            assert_eq!(v.into_bound().value(), i);
            assert_eq!(DimK::new(v).into_bound().value(), i);

            let b = Bound::new(v.get());
            assert_eq!(b.into_bound().value(), i);
        }
    }

    #[test]
    fn test_macros() {
        let b = Bound::new(10);

        let disp = "hello";

        // check_eq
        check_eq!(b, 10);
        check_eq!(b, 10, "some stuff {disp}: {}", disp);

        let msg = panic_message_for(|| check_eq!(b, 5));
        assert_eq!(msg, "expected 10 to be equal to 5");
        let msg = panic_message_for(|| check_eq!(b, 5, "word"));
        assert_eq!(msg, "expected 10 to be equal to 5 -- word");

        // check_lt
        check_lt!(b, 100);
        check_lt!(b, 100, "some stuff {disp}: {}", disp);

        let msg = panic_message_for(|| check_lt!(b, 10));
        assert_eq!(msg, "expected 10 to be less than 10");
        let msg = panic_message_for(|| check_lt!(b, 5));
        assert_eq!(msg, "expected 10 to be less than 5");
        let msg = panic_message_for(|| check_lt!(b, 10, "word"));
        assert_eq!(msg, "expected 10 to be less than 10 -- word");

        // check_le
        check_le!(b, 100);
        check_le!(b, 100, "some stuff {disp}: {}", disp);
        check_le!(b, 10);
        check_le!(b, 10, "some stuff {disp}: {}", disp);

        let msg = panic_message_for(|| check_le!(b, 5));
        assert_eq!(msg, "expected 10 to be less than or equal to 5");
        let msg = panic_message_for(|| check_le!(b, 5, "word"));
        assert_eq!(msg, "expected 10 to be less than or equal to 5 -- word");

        // check_gt
        check_gt!(b, 5);
        check_gt!(b, 5, "some stuff {disp}: {}", disp);

        let msg = panic_message_for(|| check_gt!(b, 10));
        assert_eq!(msg, "expected 10 to be greater than 10");
        let msg = panic_message_for(|| check_gt!(b, 15));
        assert_eq!(msg, "expected 10 to be greater than 15");
        let msg = panic_message_for(|| check_gt!(b, 10, "word"));
        assert_eq!(msg, "expected 10 to be greater than 10 -- word");

        // check_ge
        check_ge!(b, 5);
        check_ge!(b, 5, "some stuff {disp}: {}", disp);
        check_ge!(b, 10);
        check_ge!(b, 10, "some stuff {disp}: {}", disp);

        let msg = panic_message_for(|| check_ge!(b, 15));
        assert_eq!(msg, "expected 10 to be greater than or equal to 15");

        let msg = panic_message_for(|| check_ge!(b, 15, "word"));
        assert_eq!(msg, "expected 10 to be greater than or equal to 15 -- word");
    }
}
