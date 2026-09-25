/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::{utils::datatype::DataType, Checker, Input};
use diskann_quantization::multi_vector::MaxSimIsa;
use serde::{Deserialize, Serialize};

////////////////
// Enum types //
////////////////

/// JSON-facing shadow of [`MaxSimIsa`]. The library's enum is deliberately
/// serde-free; this owns the kebab-case JSON shape and converts via `From`.
/// Stays variant-for-variant in sync with `MaxSimIsa` manually.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub(crate) enum BenchIsa {
    #[serde(rename = "x86-64-v4")]
    #[allow(non_camel_case_types)]
    X86_64_V4,
    #[serde(rename = "x86-64-v3")]
    #[allow(non_camel_case_types)]
    X86_64_V3,
    Neon,
    Scalar,
    Reference,
    Auto,
}

impl std::fmt::Display for BenchIsa {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::X86_64_V4 => "x86-64-v4",
            Self::X86_64_V3 => "x86-64-v3",
            Self::Neon => "neon",
            Self::Scalar => "scalar",
            Self::Reference => "reference",
            Self::Auto => "auto",
        };
        write!(f, "{}", st)
    }
}

impl From<BenchIsa> for MaxSimIsa {
    fn from(b: BenchIsa) -> Self {
        match b {
            BenchIsa::X86_64_V4 => MaxSimIsa::X86_64_V4,
            BenchIsa::X86_64_V3 => MaxSimIsa::X86_64_V3,
            BenchIsa::Neon => MaxSimIsa::Neon,
            BenchIsa::Scalar => MaxSimIsa::Scalar,
            BenchIsa::Reference => MaxSimIsa::Reference,
            BenchIsa::Auto => MaxSimIsa::Auto,
        }
    }
}

/// One benchmark configuration: a single shape measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Run {
    pub(crate) num_query_vectors: NonZeroUsize,
    pub(crate) num_doc_vectors: NonZeroUsize,
    pub(crate) dim: NonZeroUsize,
    pub(crate) loops_per_measurement: NonZeroUsize,
    pub(crate) num_measurements: NonZeroUsize,
}

///////////////////////
// Multi-Vector Op   //
///////////////////////

/// A complete multi-vector benchmark job.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MultiVectorOp {
    pub(crate) element_type: DataType,
    pub(crate) isa: BenchIsa,
    pub(crate) runs: Vec<Run>,
}

impl MultiVectorOp {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-op"
    }
}

impl Input for MultiVectorOp {
    type Raw = Self;

    fn tag() -> &'static str {
        Self::tag()
    }

    fn from_raw(raw: Self::Raw, _checker: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self {
        const NUM_DOC_VECTORS: NonZeroUsize = NonZeroUsize::new(64).unwrap();
        const DIM: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const LOOPS_PER_MEASUREMENT: NonZeroUsize = NonZeroUsize::new(200).unwrap();
        const NUM_MEASUREMENTS: NonZeroUsize = NonZeroUsize::new(100).unwrap();

        let runs = vec![
            Run {
                num_query_vectors: NonZeroUsize::new(32).unwrap(),
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
            Run {
                num_query_vectors: NonZeroUsize::new(64).unwrap(),
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
        ];

        Self {
            element_type: DataType::Float32,
            isa: BenchIsa::Auto,
            runs,
        }
    }
}

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>18}: {}", $field, $($expr)*)
    }
}

impl std::fmt::Display for MultiVectorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Operation\n")?;
        write_field!(f, "tag", Self::tag())?;
        write_field!(f, "element type", self.element_type)?;
        write_field!(f, "isa", self.isa)?;
        write_field!(f, "number of runs", self.runs.len())?;
        Ok(())
    }
}
