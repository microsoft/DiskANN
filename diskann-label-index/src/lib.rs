/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense, versioned label-index encoding and flat DNF/CNF query evaluation for DiskANN.

mod builder;
mod error;
mod format;
mod index;

pub use builder::encode_label_index_jsonl;
pub use error::EncodedLabelIndexError;
pub use index::{EncodedLabelIndex, EncodedLabelQuery, FilterExpressionType};

#[cfg(test)]
mod tests;
