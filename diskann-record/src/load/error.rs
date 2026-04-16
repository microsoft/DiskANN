/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{Debug, Display};

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    inner: ErrorInner,
}

impl Error {
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Error {
            inner: ErrorInner::Heavy(anyhow::Error::new(err)),
        }
    }

    pub fn message<D>(message: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Error {
            inner: ErrorInner::Heavy(anyhow::Error::msg(message)),
        }
    }

    pub fn context<D>(self, message: D) -> Self
    where
        D: Display + Send + Sync + 'static,
    {
        // TODO: Should we do something clever with "light" errors to avoid context
        // proliferation?
        match self.inner {
            ErrorInner::Light(kind) => Self {
                inner: ErrorInner::Light(kind),
            },
            ErrorInner::Heavy(kind) => Self {
                inner: ErrorInner::Heavy(kind.context(message)),
            },
        }
    }
}

#[derive(Debug)]
enum ErrorInner {
    Light(Kind),
    Heavy(anyhow::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            ErrorInner::Light(kind) => write!(f, "Load Error: {}", kind),
            ErrorInner::Heavy(error) => write!(f, "Load Error: {:?}", error),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Kind {
    VersionMismatch,
    MissingField,
    TypeMismatch,
    UnknownVersion,
}

impl Kind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VersionMismatch => "version mismatch",
            Self::MissingField => "missing field",
            Self::TypeMismatch => "type mismatch",
            Self::UnknownVersion => "unknown version",
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
        let inner = ErrorInner::Light(kind);
        Self { inner }
    }
}
