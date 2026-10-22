/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Versioned label-index encoding and flat DNF/CNF query evaluation for DiskANN.

mod bloom;
mod builder;
mod error;
mod format;
mod index;

pub use bloom::BloomFilterConfig;
pub use builder::encode_unified_label_index_jsonl;
pub use error::EncodedLabelIndexError;
pub use index::{
    BloomQueryConfig, BloomVisitScratch, CnfEvaluator, DnfEvaluator, EncodedLabelIndex,
    EncodedLabelPointEvaluator, EncodedLabelQuery, ExactAnd2PayloadEvaluator,
    ExactAnd2SummaryEvaluator, FilterExpressionType,
};

#[cfg(test)]
mod tests;
