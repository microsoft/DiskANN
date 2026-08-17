/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Local tagged error type for `diskann-disk`.
//!
//! # Historical Context
//!
//! Error representation has changed over the course of the `diskann`'s history. Original
//! versions of error handling used single top-level enum to encode error types and payloads
//! and much of the test code checking errors within `diskann-disk` was written using this
//! paradigm.
//!
//! The decisions around using a tagged [`ErrorKind`] are to keep test code within
//! `diskann-disk` relatively static during refactors of central [`diskann::ANNError`].
//!
//! The internal `diskann_error!` macro should be used to get most of the benefits from
//! [`diskann::ANNError`] by:
//!
//! * Constructing a tagged [`Error`] in an efficient way.
//! * Creating a new [`diskann::ANNError`] in-place, ensuring that the source line tracking
//!   of that type is accurate.
//!
//! A limitation of this approach is that it forces string formatting upon error construction
//! (though in a few cases like `&'static str` literals, we can avoid this allocation).
//! Depending on the context, this formatting can negatively impact generated code even when
//! not used or add overhead on the error path. Direct use of [`diskann::ANNResult`] has
//! less overhead as error/display types are moved directly into that's types allocated
//! storage, costing just a relatively small allocation at construction time rather than
//! running string formatting eagerly.

use std::borrow::Cow;

/// Disk index related errors tagged with a provenance [`ErrorKind`].
///
/// These errors can be retrieved from [`diskann::ANNError`] by using the
/// [`downcast`](diskann::ANNError::downcast_ref) APIs.
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: Cow<'static, str>,
}

impl Error {
    /// Construct a new tagged [`Error`].
    pub(crate) fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Construct a new [`Error`] using `message`'s implementation of [`ToString`].
    pub(crate) fn from_display<D>(kind: ErrorKind, message: D) -> Self
    where
        D: ToString,
    {
        Self::new(kind, message.to_string())
    }

    /// Return the tagged [`ErrorKind`] of this error.
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Construct a new [`Error`] from [`std::fmt::Arguments`].
    ///
    /// This method avoids allocating if [`std::fmt::Arguments::as_str`] returns `Some`.
    #[inline]
    pub(crate) fn from_args(kind: ErrorKind, args: std::fmt::Arguments<'_>) -> Self {
        let message = match args.as_str() {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(args.to_string()),
        };

        Self { kind, message }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl std::error::Error for Error {}

diskann::convert_error!(Error);

/// Classification of error types in [`Error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    IndexError,
    PQError,
    KMeansError,
    IndexConfigError(&'static str),
    DimensionMismatchError,
    SerdeError,
    DiskIOAlignmentError,
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorKind::IndexError => f.write_str("IndexError"),
            ErrorKind::PQError => f.write_str("PQError"),
            ErrorKind::KMeansError => f.write_str("KMeansError"),
            ErrorKind::IndexConfigError(key) => write!(f, "IndexConfigError for \"{}\"", key),
            ErrorKind::DimensionMismatchError => f.write_str("DimensionMismatchError"),
            ErrorKind::SerdeError => f.write_str("SerdeError"),
            ErrorKind::DiskIOAlignmentError => f.write_str("DiskIOAlignmentError"),
        }
    }
}

/// Construct a [`diskann::ANNError`] containing a tagged [`Error`].
///
/// This macro attempts to use the most efficient construction mechanism.
///
/// Since this is an internal macro, see the tests for usage.
macro_rules! diskann_error {
    ($kind:expr, $var:ident) => {
        ::diskann::ANNError::new(
            $crate::error::Error::from_display(
                $kind,
                &$var,
            )
        )
    };
    ($kind:expr, $($args:tt)*) => {
        ::diskann::ANNError::new(
            $crate::error::Error::from_args(
                $kind,
                format_args!($($args)*),
            )
        )
    };
}

pub(crate) use diskann_error;

//----------------//
// Test Utilities //
//----------------//

#[cfg(test)]
pub(crate) fn error_kind(err: &diskann::ANNError) -> ErrorKind {
    match err.downcast_ref::<Error>() {
        Some(e) => e.kind(),
        None => panic!("error payload is not a `$crate::Error`"),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error() {
        // New - borrowed
        let err = Error::new(ErrorKind::IndexError, "a &'static str");
        assert_eq!(err.kind(), ErrorKind::IndexError);
        assert!(matches!(err.message, Cow::Borrowed("a &'static str")));
        assert_eq!(err.to_string(), "IndexError: a &'static str");

        // New - owned
        let err = Error::new(ErrorKind::PQError, String::from("a string"));
        assert_eq!(err.kind(), ErrorKind::PQError);
        assert!(matches!(err.message, Cow::Owned(_)));
        assert_eq!(err.message, "a string");
        assert_eq!(err.to_string(), "PQError: a string");

        // `from_display`
        let err = Error::from_display(ErrorKind::PQError, ErrorKind::PQError);
        assert_eq!(err.kind(), ErrorKind::PQError);
        assert!(matches!(err.message, Cow::Owned(_)));
        assert_eq!(err.message, "PQError");

        // `from_args` - non-allocating.
        let err = Error::from_args(ErrorKind::IndexError, format_args!("a &'static str"));
        assert_eq!(err.kind(), ErrorKind::IndexError);
        assert!(matches!(err.message, Cow::Borrowed("a &'static str")));

        // `from_args` - allocating.
        let err = Error::from_args(
            ErrorKind::IndexError,
            format_args!("a {}", ErrorKind::PQError),
        );
        assert_eq!(err.kind(), ErrorKind::IndexError);
        assert!(matches!(err.message, Cow::Owned(_)));
        assert_eq!(err.message, "a PQError");
    }

    #[test]
    fn test_error_kind() {
        assert_eq!(ErrorKind::IndexError.to_string(), "IndexError");
        assert_eq!(ErrorKind::PQError.to_string(), "PQError");
        assert_eq!(ErrorKind::KMeansError.to_string(), "KMeansError");
        assert_eq!(
            ErrorKind::IndexConfigError("foo").to_string(),
            "IndexConfigError for \"foo\""
        );
        assert_eq!(
            ErrorKind::IndexConfigError("bar").to_string(),
            "IndexConfigError for \"bar\""
        );
        assert_eq!(
            ErrorKind::DimensionMismatchError.to_string(),
            "DimensionMismatchError"
        );
        assert_eq!(ErrorKind::SerdeError.to_string(), "SerdeError");
        assert_eq!(
            ErrorKind::DiskIOAlignmentError.to_string(),
            "DiskIOAlignmentError"
        );
    }

    #[test]
    fn test_macro() {
        // Variable identifiers - `IndexConfigError`
        let var = String::from("oops");
        let err = diskann_error!(ErrorKind::IndexConfigError("some variable"), var);
        assert_eq!(
            error_kind(&err),
            ErrorKind::IndexConfigError("some variable")
        );
        let err = err.downcast::<Error>().unwrap();
        assert_eq!(
            err.to_string(),
            "IndexConfigError for \"some variable\": oops"
        );

        // Variable identifiers - unit error.
        let err = diskann_error!(ErrorKind::IndexError, var);
        assert_eq!(error_kind(&err), ErrorKind::IndexError);
        let err = err.downcast::<Error>().unwrap();
        assert_eq!(err.to_string(), "IndexError: oops");

        // formatting with string literal - non-allocating.
        let err = diskann_error!(ErrorKind::IndexError, "something went wrong");
        assert_eq!(error_kind(&err), ErrorKind::IndexError);
        let err = err.downcast::<Error>().unwrap();
        assert_eq!(err.to_string(), "IndexError: something went wrong");
        assert!(matches!(err.message, Cow::Borrowed(_)));

        let err = diskann_error!(ErrorKind::IndexConfigError("foo"), "something went wrong");
        assert_eq!(error_kind(&err), ErrorKind::IndexConfigError("foo"));
        let err = err.downcast::<Error>().unwrap();
        assert_eq!(
            err.to_string(),
            "IndexConfigError for \"foo\": something went wrong"
        );
        assert!(matches!(err.message, Cow::Borrowed(_)));

        // formatting - allocating.
        let x = 1;
        let y = 2;
        let z = 3;
        let err = diskann_error!(
            ErrorKind::IndexError,
            "something went wrong - {x}, {}, and {}",
            y,
            z,
        );
        assert_eq!(error_kind(&err), ErrorKind::IndexError);
        let err = err.downcast::<Error>().unwrap();
        assert_eq!(
            err.to_string(),
            "IndexError: something went wrong - 1, 2, and 3"
        );
    }
}
