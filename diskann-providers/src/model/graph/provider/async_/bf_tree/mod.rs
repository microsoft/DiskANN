/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod neighbor_provider;
mod provider;
mod quant_vector_provider;
mod vector_provider;

// Accessors
pub use provider::{
    BfTreeProvider, BfTreeProviderParameters, CreateQuantProvider, FullAccessor, Hidden, Index,
    QuantAccessor, QuantIndex, StartPoint,
};

pub use bf_tree::Config;

use diskann::ANNError;

/// Wrapper around [`bf_tree::ConfigError`] that implements [`std::error::Error`].
#[derive(Debug, Clone)]
pub struct ConfigError(pub bf_tree::ConfigError);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BfTree configuration error: {:?}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<ConfigError> for ANNError {
    #[track_caller]
    #[inline(never)]
    fn from(error: ConfigError) -> ANNError {
        ANNError::new(diskann::ANNErrorKind::IndexError, error)
    }
}
