/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io;
use thiserror::Error;

/// An error encountered while building, loading, or querying an encoded label index.
#[derive(Debug, Error)]
pub enum EncodedLabelIndexError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
}
