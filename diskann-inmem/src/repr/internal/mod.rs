/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

pub(super) mod intrusive;
pub(super) mod simple;

pub(super) mod macros;

//////////
// Calf //
//////////

// A baby [`std::borrow::Cow`].
#[derive(Debug)]
pub(super) enum Calf<'a, T>
where
    T: ?Sized,
{
    Borrowed(&'a T),
    Owned(Box<T>),
}

impl<T> std::ops::Deref for Calf<'_, T>
where
    T: ?Sized,
{
    type Target = T;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(slice) => slice,
            Self::Owned(boxed) => boxed,
        }
    }
}

///////////////
// Distances //
///////////////

pub(super) trait RawQueryDistance: std::fmt::Debug + Send + Sync {
    type Error: diskann::error::StandardError;

    fn eval(&self, x: &[u8]) -> Result<f32, Self::Error>;
}

pub(super) trait RawDistance: std::fmt::Debug + Send + Sync {
    type Error: diskann::error::StandardError;

    fn eval(&self, x: &[u8], y: &[u8]) -> Result<f32, Self::Error>;
}

////////////
// Errors //
////////////

#[derive(Debug, Error)]
#[error("index {} is out-of-bounds", self.0)]
pub(super) struct OutOfBounds(u32);

impl OutOfBounds {
    pub(super) const fn new(id: u32) -> Self {
        Self(id)
    }
}

diskann::convert_error!(OutOfBounds);
