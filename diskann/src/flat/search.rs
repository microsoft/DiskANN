/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Error types for flat search parameter validation.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{ANNError, ANNErrorKind};

/// Errors raised when validating flat search parameters.
#[derive(Debug, Error)]
pub enum KnnFlatError {
    /// `k` was zero.
    #[error("k cannot be zero")]
    KZero,
}

impl From<KnnFlatError> for ANNError {
    #[track_caller]
    fn from(err: KnnFlatError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

/// Validate and wrap a `k` value as [`NonZeroUsize`].
///
/// This is a convenience for callers that want to validate `k` before passing it to
/// [`FlatIndex::knn_search`](crate::flat::FlatIndex::knn_search).
pub fn validate_k(k: usize) -> Result<NonZeroUsize, KnnFlatError> {
    NonZeroUsize::new(k).ok_or(KnnFlatError::KZero)
}
