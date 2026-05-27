/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Load-side error type and classification.
//!
//! The [`Error`] type wraps [`anyhow::Error`] for rich diagnostics and carries a
//! recoverable / critical bit used by probing call sites. The [`Kind`] enum enumerates
//! the well-known structural failure modes; [`Kind::is_recoverable`] is the canonical
//! source of truth for the recoverable / critical classification.

use std::fmt::{Debug, Display};

/// A specialized [`std::result::Result`] for load-side operations.
pub type Result<T> = ::std::result::Result<T, Error>;

/// Load-side error.
///
/// Carries an inner [`anyhow::Error`] for rich diagnostics (chained context,
/// backtraces) along with a single `recoverable` bit. Recoverable errors are
/// the contract for probing APIs: a caller that tries multiple load strategies
/// (e.g. current version, then legacy) can distinguish "this attempt didn't
/// match, try another" from "the data is broken, stop now".
///
/// Most constructors produce *critical* (non-recoverable) errors. Probing
/// call sites use the explicit `*_recoverable` constructors, or rely on the
/// [`From<Kind>`] impl which classifies each [`Kind`] variant according to
/// [`Kind::is_recoverable`].
#[derive(Debug)]
pub struct Error {
    inner: anyhow::Error,
    recoverable: bool,
}

impl Error {
    /// Construct a critical error from an underlying source error.
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            inner: anyhow::Error::new(err),
            recoverable: false,
        }
    }

    /// Construct a critical error from a display message.
    pub fn message<D>(message: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Self {
            inner: anyhow::Error::msg(message),
            recoverable: false,
        }
    }

    /// Construct a recoverable error from an underlying source. Suitable for
    /// probing APIs that may attempt an alternative load strategy.
    pub fn new_recoverable<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self {
            inner: anyhow::Error::new(err),
            recoverable: true,
        }
    }

    /// Construct a recoverable error from a display message. Suitable for
    /// probing APIs that may attempt an alternative load strategy.
    pub fn message_recoverable<D>(message: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Self {
            inner: anyhow::Error::msg(message),
            recoverable: true,
        }
    }

    /// Attach additional context. The `recoverable` flag is preserved.
    pub fn context<D>(self, message: D) -> Self
    where
        D: Display + Send + Sync + 'static,
    {
        Self {
            inner: self.inner.context(message),
            recoverable: self.recoverable,
        }
    }

    /// Returns `true` if this error is recoverable. Probing call sites should
    /// only fall back to alternative load strategies when this is `true`.
    pub fn is_recoverable(&self) -> bool {
        self.recoverable
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Load Error: {:?}", self.inner)
    }
}

impl std::error::Error for Error {}

/// Well-known classes of load-side failure.
///
/// Used in two roles:
///
/// * As the source of an [`Error`] via `From<Kind>` (and the matching `From<Kind>` for
///   [`Error`] which classifies recoverable / critical according to
///   [`Kind::is_recoverable`]).
/// * As a probe value in error chains — high-level callers can introspect the kind to
///   decide whether to try a fallback loader.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Kind {
    /// The manifest's `$version` does not match the loader's expected
    /// [`Load::VERSION`](crate::load::Load::VERSION).
    VersionMismatch,
    /// A required field is absent from the record.
    MissingField,
    /// The shape of the saved value does not match what the loader expected (e.g. found
    /// an array where an object was needed).
    TypeMismatch,
    /// The manifest's version is recognized as not matching the current schema, and the
    /// type's [`Load::load_legacy`](crate::load::Load::load_legacy) has no upgrade path
    /// for it.
    UnknownVersion,
    /// The variant tag read from the wire format does not match any known
    /// variant of the target enum.
    UnknownVariant,
    /// A numeric value in the manifest does not fit in the requested Rust type
    /// (either out of range or would lose precision).
    NumberOutOfRange,
    /// A `$handle` references a file name that is not registered in the
    /// manifest's `files` set.
    MissingFile,
}

impl Kind {
    /// Stable, human-readable description of this kind. Used as the default error
    /// message when constructing an [`Error`] from a `Kind`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VersionMismatch => "version mismatch",
            Self::MissingField => "missing field",
            Self::TypeMismatch => "type mismatch",
            Self::UnknownVersion => "unknown version",
            Self::UnknownVariant => "unknown variant",
            Self::NumberOutOfRange => "number out of range for target type",
            Self::MissingFile => "handle references a file not present in the manifest",
        }
    }

    /// Whether an error of this kind should be treated as recoverable by
    /// probing APIs (i.e., suitable for triggering a fallback to an alternative
    /// load strategy).
    ///
    /// Recoverable kinds describe "the data did not match what this loader
    /// expected" (a different version or shape might still succeed). Critical
    /// kinds describe structural or integrity problems where retrying would be
    /// pointless or unsafe.
    pub const fn is_recoverable(self) -> bool {
        match self {
            // Shape/version probing signals — another loader might succeed.
            Self::VersionMismatch | Self::MissingField | Self::TypeMismatch => true,
            // Structural / integrity failures — give up.
            Self::UnknownVersion
            | Self::UnknownVariant
            | Self::NumberOutOfRange
            | Self::MissingFile => false,
        }
    }
}

impl std::fmt::Display for Kind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Kind> for Error {
    fn from(kind: Kind) -> Self {
        Self {
            inner: anyhow::Error::msg(kind.as_str()),
            recoverable: kind.is_recoverable(),
        }
    }
}
