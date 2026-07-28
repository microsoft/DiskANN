/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::borrow::Cow;

use diskann::convert_error;

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

#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    message: Cow<'static, str>,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn from_display<D>(kind: ErrorKind, message: &D) -> Self
    where
        D: std::string::ToString,
    {
        Self::new(kind, message.to_string())
    }

    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

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

convert_error!(Error);

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
