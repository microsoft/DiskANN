/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

#[derive(Debug, Clone, Copy, Default, Error)]
#[error(transparent)]
#[repr(transparent)]
// A wrapper type to bridge errors from one type to another.
// We need this to convert errors from other crates into ANNError for interop
pub struct Bridge<T>(pub T);

impl<T> Bridge<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
}

pub trait BridgeErr<T, E> {
    fn bridge_err(self) -> Result<T, Bridge<E>>;
}

impl<T, E> BridgeErr<T, E> for Result<T, E> {
    fn bridge_err(self) -> Result<T, Bridge<E>> {
        self.map_err(Bridge)
    }
}

// Bridge error conversions to ANNError for spherical quantization types.
// These live here to satisfy the orphan rule since Bridge is defined in this crate.

impl From<Bridge<diskann_quantization::spherical::CompressionError>> for diskann::ANNError {
    #[track_caller]
    fn from(
        err: Bridge<diskann_quantization::spherical::CompressionError>,
    ) -> Self {
        diskann::ANNError::new(diskann::ANNErrorKind::SQError, err)
    }
}

impl From<Bridge<diskann_quantization::spherical::UnsupportedMetric>> for diskann::ANNError {
    #[track_caller]
    fn from(
        err: Bridge<diskann_quantization::spherical::UnsupportedMetric>,
    ) -> Self {
        diskann::ANNError::new(diskann::ANNErrorKind::SQError, err)
    }
}
