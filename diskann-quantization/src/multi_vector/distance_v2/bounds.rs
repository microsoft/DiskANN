/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub(super) use length::{Check, Length};

pub(super) trait IntoLength {
    fn into_length(self) -> Length;
}

impl IntoLength for usize {
    fn into_length(self) -> Length {
        Length::new(self)
    }
}

impl IntoLength for std::num::NonZeroUsize {
    fn into_length(self) -> Length {
        Length::new(self.get())
    }
}

impl IntoLength for Length {
    fn into_length(self) -> Length {
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

macro_rules! check_lt {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(lt, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(lt, $lhs, $rhs, $($arg)+);
    };
}

macro_rules! check_le {
    ($lhs:expr, $rhs:expr $(,)?) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(le, $lhs, $rhs);
    };
    ($lhs:expr, $rhs:expr, $($arg:tt)+) => {
        $crate::multi_vector::distance_v2::bounds::__assert!(le, $lhs, $rhs, $($arg)+);
    };
}

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

#[cfg(test)]
macro_rules! __assert {
    ($op:ident, $lhs:expr, $rhs:expr $(,)?) => {
        ($lhs).check(
            $crate::multi_vector::distance_v2::Check::$op(),
            $rhs,
            None,
        )
    };
    ($op:ident, $lhs:expr, $rhs:expr, $($arg:tt)+) => {
        ($lhs).check(
            $crate::multi_vector::distance_v2::Check::$op(),
            $rhs,
            Some(&::diskann_utils::lazy_format!($($arg)+)),
        )
    };
}

#[cfg(not(test))]
macro_rules! __assert {
    ($op:ident, $lhs:expr, $rhs:expr $(,)?) => {};
    ($op:ident, $lhs:expr, $rhs:expr, $($arg:tt)+) => {};
}

pub(super) use check_eq;
pub(super) use check_ge;
pub(super) use check_gt;
pub(super) use check_le;
pub(super) use check_lt;
pub(super) use __assert;

#[cfg(test)]
mod length {
    use super::IntoLength;

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

        fn check(self, lhs: usize, rhs: usize, message: Option<&dyn std::fmt::Display>) {
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
    pub(in crate::multi_vector::distance_v2) struct Length(usize);

    impl Length {
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
            message: Option<&dyn std::fmt::Display>,
        ) where
            T: IntoLength,
        {
            check.check(self.0, expected.into_length().0, message)
        }

        #[track_caller]
        pub(in crate::multi_vector::distance_v2) fn check_with<F, T>(
            self,
            check: Check,
            f: F,
            message: Option<&dyn std::fmt::Display>,
        ) where
            F: FnOnce() -> T,
            T: IntoLength,
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

    impl std::ops::Mul for Length {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0.mul(rhs.0))
        }
    }

    impl std::ops::Add for Length {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self::new(self.0.add(rhs.0))
        }
    }

    impl std::ops::Sub for Length {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0.sub(rhs.0))
        }
    }
}

#[cfg(not(test))]
mod length {
    use super::IntoLength;

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
    pub(in crate::multi_vector::distance_v2) struct Length(());

    impl Length {
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
            _msg: Option<&dyn std::fmt::Display>,
        ) where
            T: IntoLength,
        {
        }

        pub(in crate::multi_vector::distance_v2) fn check_with<F, T>(
            self,
            _check: Check,
            _f: F,
            _message: Option<&dyn std::fmt::Display>,
        ) where
            F: FnOnce() -> T,
            T: IntoLength,
        {
        }

        pub(in crate::multi_vector::distance_v2) fn with<F>(self, _f: F)
        where
            F: FnOnce(usize),
        {
        }
    }

    impl std::ops::Mul for Length {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Add for Length {
        type Output = Self;

        fn add(self, rhs: Self) -> Self {
            Self(())
        }
    }

    impl std::ops::Sub for Length {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self {
            Self(())
        }
    }
}
