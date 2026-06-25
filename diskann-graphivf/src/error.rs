/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Error and result types for the graph-IVF index.

/// Errors produced while building, loading, or searching a graph-IVF index.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GraphIvfError {
    /// An I/O error reading or writing index files.
    #[error("graph-ivf I/O error")]
    Io(#[from] std::io::Error),

    /// An error originating from a `diskann` subsystem (graph build/search,
    /// k-means, aligned disk reads, ...).
    #[error(transparent)]
    Ann(#[from] diskann::ANNError),

    /// A caller-supplied parameter was invalid.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// An on-disk index file was malformed or had an unexpected layout.
    #[error("malformed index file: {0}")]
    Malformed(String),
}

/// Convenience alias for results returned by this crate.
pub type Result<T> = std::result::Result<T, GraphIvfError>;

impl GraphIvfError {
    pub(crate) fn invalid(msg: impl Into<String>) -> Self {
        Self::InvalidParameter(msg.into())
    }

    pub(crate) fn malformed(msg: impl Into<String>) -> Self {
        Self::Malformed(msg.into())
    }
}
