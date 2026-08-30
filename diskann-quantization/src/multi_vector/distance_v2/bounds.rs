/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(super) use inner::{Bound, Check};

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
        $crate::multi_vector::distance_v2::bounds::__assert!(eq, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(eq, $lhs, $rhs, $($arg)+);
    };
}

#[expect(unused, reason = "this completes the API")]
macro_rules! check_lt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(lt, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(lt, $lhs, $rhs, $($arg)+);
    };
}

#[expect(unused, reason = "this completes the API")]
macro_rules! check_le {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(le, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(le, $lhs, $rhs, $($arg)+);
    };
}

#[expect(unused, reason = "this completes the API")]
macro_rules! check_gt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(gt, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(gt, $lhs, $rhs, $($arg)+);
    };
}

macro_rules! check_ge {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(ge, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)*) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(ge, $lhs, $rhs, $($arg)*);
    };
}

macro_rules! __assert {
    ($op:ident, $lhs:expr, $rhs:expr $(,)?) => {
        if cfg!(debug_assertions) {
            ($lhs).check(
                $crate::multi_vector::distance_v2::bounds::Check::$op(),
                $rhs,
                None,
            )
        }
    };
    ($op:ident, $lhs:expr, $rhs:expr, $($arg:tt)+) => {
        if cfg!(debug_assertions) {
            ($lhs).check(
                $crate::multi_vector::distance_v2::bounds::Check::$op(),
                $rhs,
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

#[cfg(debug_assertions)]
mod inner {
    use super::IntoBound;

    #[derive(Debug, Clone, Copy)]
    pub(in crate::multi_vector::distance_v2) struct Check(Inner);

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    enum Inner {
        Eq,
        Lt,
        Le,
        Gt,
        Ge,
    }

    impl Check {
        pub(in crate::multi_vector::distance_v2) const fn eq() -> Self {
            Self(Inner::Eq)
        }

        pub(in crate::multi_vector::distance_v2) const fn lt() -> Self {
            Self(Inner::Lt)
        }

        pub(in crate::multi_vector::distance_v2) const fn le() -> Self {
            Self(Inner::Le)
        }

        pub(in crate::multi_vector::distance_v2) const fn gt() -> Self {
            Self(Inner::Gt)
        }

        pub(in crate::multi_vector::distance_v2) const fn ge() -> Self {
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
    pub(in crate::multi_vector::distance_v2) struct Bound(usize);

    impl Bound {
        pub(in crate::multi_vector::distance_v2) fn from_fn<F>(f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self::new(f())
        }

        pub(in crate::multi_vector::distance_v2) const fn new(length: usize) -> Self {
            Self(length)
        }

        #[track_caller]
        pub(in crate::multi_vector::distance_v2) fn check<T>(
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
        pub(in crate::multi_vector::distance_v2) fn check_with<F, T>(
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

        pub(in crate::multi_vector::distance_v2) fn with<F>(self, f: F)
        where
            F: FnOnce(usize),
        {
            f(self.0)
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

#[cfg(not(debug_assertions))]
mod inner {
    use super::IntoBound;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub(in crate::multi_vector::distance_v2) struct Check(());

    impl Check {
        pub(in crate::multi_vector::distance_v2) const fn eq() -> Self {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) const fn lt() -> Self {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) const fn le() -> Self {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) const fn gt() -> Self {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) const fn ge() -> Self {
            Self(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(in crate::multi_vector::distance_v2) struct Bound(());

    impl Bound {
        pub(in crate::multi_vector::distance_v2) fn from_fn<F>(_f: F) -> Self
        where
            F: FnOnce() -> usize,
        {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) const fn new(_length: usize) -> Self {
            Self(())
        }

        pub(in crate::multi_vector::distance_v2) fn check<T>(
            self,
            _check: Check,
            _expected: T,
            _msg: Option<std::fmt::Arguments<'_>>,
        ) where
            T: IntoBound,
        {
        }

        pub(in crate::multi_vector::distance_v2) fn check_with<F, T>(
            self,
            _check: Check,
            _f: F,
            _message: Option<std::fmt::Arguments<'_>>,
        ) where
            F: FnOnce() -> T,
            T: IntoBound,
        {
        }

        pub(in crate::multi_vector::distance_v2) fn with<F>(self, _f: F)
        where
            F: FnOnce(usize),
        {
        }
    }

    impl std::ops::Mul for Bound {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Add for Bound {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Sub for Bound {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self {
            Self(())
        }
    }
}
