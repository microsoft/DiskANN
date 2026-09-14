/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

pub(super) mod intrusive;

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
