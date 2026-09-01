/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(super) use inner::{Bound, Check};

pub(super) trait IntoBound {
    #[cfg_attr(
        not(any(test, debug_assertions)),
        expect(dead_code, reason = "this is only used in debug builds")
    )]
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

#[expect(unused, reason = "this completes the API")]
macro_rules! check_lt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(lt, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(lt, $lhs, $rhs, $($arg)+)
    };
}

#[expect(unused, reason = "this completes the API")]
macro_rules! check_le {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::matrix_kernels::bounds::__assert!(le, $lhs, $rhs)
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::matrix_kernels::bounds::__assert!(le, $lhs, $rhs, $($arg)+)
    };
}

#[expect(unused, reason = "this completes the API")]
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
            ($lhs).check_with(
                $crate::matrix_kernels::bounds::Check::$op(),
                || $rhs,
                None,
            )
        }
    };
    ($op:ident, $lhs:expr, $rhs:expr, $($arg:tt)+) => {
        if cfg!(any(test, debug_assertions)) {
            ($lhs).check_with(
                $crate::matrix_kernels::bounds::Check::$op(),
                || $rhs,
                Some(format_args!($($arg)+)),
            )
        }
    };
}

pub(super) use __assert;
pub(super) use check_eq;
pub(super) use check_ge;

#[expect(unused, reason = "this completes the API")]
pub(super) use check_gt;

#[expect(unused, reason = "this completes the API")]
pub(super) use check_le;

#[expect(unused, reason = "this completes the API")]
pub(super) use check_lt;

#[cfg(any(test, debug_assertions))]
mod inner {
    use super::IntoBound;

    #[derive(Debug, Clone, Copy)]
    pub(in crate::matrix_kernels) struct Check(Inner);

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    enum Inner {
        Eq,
        #[expect(unused, reason = "this completes the API")]
        Lt,
        #[expect(unused, reason = "this completes the API")]
        Le,
        #[expect(unused, reason = "this completes the API")]
        Gt,
        Ge,
    }

    impl Check {
        pub(in crate::matrix_kernels) const fn eq() -> Self {
            Self(Inner::Eq)
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn lt() -> Self {
            Self(Inner::Lt)
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn le() -> Self {
            Self(Inner::Le)
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn gt() -> Self {
            Self(Inner::Gt)
        }

        pub(in crate::matrix_kernels) const fn ge() -> Self {
            Self(Inner::Ge)
        }

        fn as_str(&self) -> &'static str {
            match self.0 {
                Inner::Eq => "equal to",
                Inner::Lt => "less than",
                Inner::Le => "less than or equal to",
                Inner::Gt => "greater than",
                Inner::Ge => "greater than or equal to",
            }
        }

        fn check(self, lhs: usize, rhs: usize, message: Option<std::fmt::Arguments<'_>>) {
            let passed = match self.0 {
                Inner::Eq => lhs == rhs,
                Inner::Lt => lhs < rhs,
                Inner::Le => lhs <= rhs,
                Inner::Gt => lhs > rhs,
                Inner::Ge => lhs >= rhs,
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

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub(in crate::matrix_kernels) struct Bound(usize);

    impl Bound {
        pub(in crate::matrix_kernels) fn from_fn<F>(f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self::new(f())
        }

        pub(in crate::matrix_kernels) const fn new(length: usize) -> Self {
            Self(length)
        }

        #[track_caller]
        pub(in crate::matrix_kernels) fn check<T>(
            self,
            check: Check,
            expected: T,
            message: Option<std::fmt::Arguments<'_>>,
        ) where
            T: IntoBound,
        {
            check.check(self.0, expected.into_bound().0, message)
        }

        #[track_caller]
        pub(in crate::matrix_kernels) fn check_with<F, T>(
            self,
            check: Check,
            f: F,
            message: Option<std::fmt::Arguments<'_>>,
        ) where
            F: FnOnce() -> T,
            T: IntoBound,
        {
            self.check(check, f(), message)
        }

        pub(in crate::matrix_kernels) fn with<F>(self, f: F)
        where
            F: FnOnce(usize),
        {
            f(self.0)
        }

        #[cfg(test)]
        pub(in crate::matrix_kernels) fn value(self) -> usize {
            self.0
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
    use super::IntoBound;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub(in crate::matrix_kernels) struct Check(());

    impl Check {
        pub(in crate::matrix_kernels) const fn eq() -> Self {
            Self(())
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn lt() -> Self {
            Self(())
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn le() -> Self {
            Self(())
        }

        #[expect(unused, reason = "this completes the API")]
        pub(in crate::matrix_kernels) const fn gt() -> Self {
            Self(())
        }

        pub(in crate::matrix_kernels) const fn ge() -> Self {
            Self(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(in crate::matrix_kernels) struct Bound(());

    impl Bound {
        pub(in crate::matrix_kernels) fn from_fn<F>(_f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self(())
        }

        pub(in crate::matrix_kernels) const fn new(_length: usize) -> Self {
            Self(())
        }

        #[expect(unused, reason = "this should not be called in release builds")]
        pub(in crate::matrix_kernels) fn check<T>(
            self,
            _check: Check,
            _expected: T,
            _msg: Option<std::fmt::Arguments<'_>>,
        ) where
            T: IntoBound,
        {
        }

        pub(in crate::matrix_kernels) fn check_with<F, T>(
            self,
            _check: Check,
            _f: F,
            _message: Option<std::fmt::Arguments<'_>>,
        ) where
            F: FnOnce() -> T,
            T: IntoBound,
        {
        }

        pub(in crate::matrix_kernels) fn with<F>(self, _f: F)
        where
            F: FnOnce(usize),
        {
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
