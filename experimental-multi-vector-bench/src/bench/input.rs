// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Input types for multi-vector benchmarks.

use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

use diskann_benchmark_runner::{Any, CheckDeserialization, Checker};

/// Approach to use for multi-vector distance computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Approach {
    /// Naive scalar approach.
    Naive,
    /// SIMD-accelerated approach.
    Simd,
    /// Transposed SIMD approach using block-transposed data layout.
    TransposedSimd,
    /// Transposed SIMD approach with tiling optimization.
    TransposedWithTiling,
    /// Query-transposed SIMD approach with tiling optimization.
    QueryTransposedWithTiling,
    /// SGEMM-based approach using BLAS matrix multiplication (baseline).
    Sgemm,
}

impl std::fmt::Display for Approach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Naive => write!(f, "naive"),
            Self::Simd => write!(f, "simd"),
            Self::TransposedSimd => write!(f, "transposed_simd"),
            Self::TransposedWithTiling => write!(f, "transposed_with_tiling"),
            Self::QueryTransposedWithTiling => write!(f, "query_transposed_with_tiling"),
            Self::Sgemm => write!(f, "sgemm"),
        }
    }
}

/// A single benchmark run configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Dimensionality of each token embedding.
    pub dim: NonZeroUsize,
    /// Number of document multi-vectors to compare against.
    pub num_points: NonZeroUsize,
    /// Number of loops per measurement (for timing stability).
    pub loops_per_measurement: NonZeroUsize,
    /// Number of measurements to collect.
    pub num_measurements: NonZeroUsize,
    /// Number of tokens in the query multi-vector.
    pub num_query_token: NonZeroUsize,
    /// Number of tokens in each document multi-vector.
    pub num_doc_token: NonZeroUsize,
}

/// Input specification for multi-vector benchmarks.
#[derive(Debug, Serialize, Deserialize)]
pub struct MultiVectorOp {
    /// Approach to use: naive or simd.
    pub approach: Approach,
    /// List of benchmark runs to execute.
    pub runs: Vec<Run>,
    /// Whether to compute and output distance checksum for verification.
    #[serde(default)]
    pub verify: bool,
}

impl CheckDeserialization for MultiVectorOp {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>24}: {}", $field, $($expr)*)
    }
}

impl MultiVectorOp {
    pub(crate) const fn tag() -> &'static str {
        "multivec-op"
    }

    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "approach", self.approach)?;
        write_field!(f, "number of runs", self.runs.len())?;
        Ok(())
    }
}

impl std::fmt::Display for MultiVectorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Operation\n")?;
        write_field!(f, "tag", Self::tag())?;
        self.summarize_fields(f)
    }
}

/// Input parser for multi-vector benchmarks.
#[derive(Debug)]
pub struct MultiVectorInput;

impl diskann_benchmark_runner::Input for MultiVectorInput {
    fn tag(&self) -> &'static str {
        MultiVectorOp::tag()
    }

    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(MultiVectorOp::deserialize(serialized)?)
    }

    fn example(&self) -> anyhow::Result<serde_json::Value> {
        const DIM: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const NUM_POINTS: NonZeroUsize = NonZeroUsize::new(100).unwrap();
        const LOOPS_PER_MEASUREMENT: NonZeroUsize = NonZeroUsize::new(5).unwrap();
        const NUM_MEASUREMENTS: NonZeroUsize = NonZeroUsize::new(10).unwrap();
        const NUM_QUERY_TOKEN: NonZeroUsize = NonZeroUsize::new(32).unwrap();
        const NUM_DOC_TOKEN: NonZeroUsize = NonZeroUsize::new(64).unwrap();

        let runs = vec![Run {
            dim: DIM,
            num_points: NUM_POINTS,
            loops_per_measurement: LOOPS_PER_MEASUREMENT,
            num_measurements: NUM_MEASUREMENTS,
            num_query_token: NUM_QUERY_TOKEN,
            num_doc_token: NUM_DOC_TOKEN,
        }];

        Ok(serde_json::to_value(&MultiVectorOp {
            approach: Approach::Simd,
            runs,
            verify: false,
        })?)
    }
}
