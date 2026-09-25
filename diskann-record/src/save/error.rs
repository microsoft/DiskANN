/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Save-side error type.
//!
//! Mirrors the load-side [`super::super::load::Error`] in shape but does not carry a
//! recoverable / critical distinction: every save failure is terminal because no
//! probing fallback exists on the writer side.

use std::fmt::{Debug, Display};

/// A specialized [`std::result::Result`] for save-side operations.
pub type Result<T> = ::std::result::Result<T, Error>;

/// Save-side error.
///
/// Wraps [`anyhow::Error`] for rich context chains (see [`Error::context`]) and is
/// returned from every fallible save-side operation, including [`super::Save::save`]
/// impls and `save_to_disk`.
#[derive(Debug)]
pub struct Error {
    inner: anyhow::Error,
}

impl Error {
    /// Wrap an underlying source error.
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Error {
            inner: anyhow::Error::new(err),
        }
    }

    /// Construct an error from a display message with no source.
    pub fn message<D>(message: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Error {
            inner: anyhow::Error::msg(message),
        }
    }

    /// Attach additional context describing what was being attempted.
    pub fn context<D>(self, message: D) -> Self
    where
        D: Display + Send + Sync + 'static,
    {
        Error {
            inner: self.inner.context(message),
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Save Error: {:?}", self.inner)
    }
}

impl std::error::Error for Error {}
