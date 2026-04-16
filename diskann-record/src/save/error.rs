/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::{Debug, Display};

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    inner: anyhow::Error,
}

impl Error {
    pub fn new<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Error {
            inner: anyhow::Error::new(err),
        }
    }

    pub fn message<D>(message: D) -> Self
    where
        D: Display + Debug + Send + Sync + 'static,
    {
        Error {
            inner: anyhow::Error::msg(message),
        }
    }

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
