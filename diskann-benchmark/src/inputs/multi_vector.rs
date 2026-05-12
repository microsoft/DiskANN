/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::{
    utils::{datatype::DataType, num::NonNegativeFinite},
    CheckDeserialization, Checker,
};
use serde::{Deserialize, Serialize};

use crate::inputs::{as_input, Example};

//////////////
// Registry //
//////////////

as_input!(MultiVectorOp);
as_input!(MultiVectorTolerance);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register::<MultiVectorOp>()?;
    registry.register::<MultiVectorTolerance>()?;
    Ok(())
}

////////////////
// Enum types //
////////////////

/// The two distance operations exposed by `QueryComputer`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Operation {
    Chamfer,
    MaxSim,
}

impl std::fmt::Display for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::Chamfer => "chamfer",
            Self::MaxSim => "max_sim",
        };
        write!(f, "{}", st)
    }
}

/// Which implementation tier to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Implementation {
    Optimized,
    Reference,
}

impl std::fmt::Display for Implementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = match self {
            Self::Optimized => "optimized",
            Self::Reference => "reference",
        };
        write!(f, "{}", st)
    }
}

/// One benchmark configuration: a single (operation, shape) measurement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Run {
    pub(crate) operation: Operation,
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
    pub(crate) implementation: Implementation,
    pub(crate) runs: Vec<Run>,
}

impl MultiVectorOp {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-op"
    }
}

impl CheckDeserialization for MultiVectorOp {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl Example for MultiVectorOp {
    fn example() -> Self {
        const NUM_QUERY_VECTORS: NonZeroUsize = NonZeroUsize::new(32).unwrap();
        const NUM_DOC_VECTORS: NonZeroUsize = NonZeroUsize::new(64).unwrap();
        const DIM: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const LOOPS_PER_MEASUREMENT: NonZeroUsize = NonZeroUsize::new(200).unwrap();
        const NUM_MEASUREMENTS: NonZeroUsize = NonZeroUsize::new(100).unwrap();

        let runs = vec![
            Run {
                operation: Operation::Chamfer,
                num_query_vectors: NUM_QUERY_VECTORS,
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
            Run {
                operation: Operation::MaxSim,
                num_query_vectors: NUM_QUERY_VECTORS,
                num_doc_vectors: NUM_DOC_VECTORS,
                dim: DIM,
                loops_per_measurement: LOOPS_PER_MEASUREMENT,
                num_measurements: NUM_MEASUREMENTS,
            },
        ];

        Self {
            element_type: DataType::Float32,
            implementation: Implementation::Optimized,
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
        write_field!(f, "implementation", self.implementation)?;
        write_field!(f, "number of runs", self.runs.len())?;
        Ok(())
    }
}

/////////////////////////////
// Multi-Vector Tolerance  //
/////////////////////////////

/// Tolerance thresholds for multi-vector benchmark regression detection.
///
/// Each field specifies the maximum allowed relative increase in the corresponding metric.
/// For example, a value of `0.05` means a 5% increase is tolerated.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct MultiVectorTolerance {
    pub(crate) min_time_regression: NonNegativeFinite,
}

impl MultiVectorTolerance {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-tolerance"
    }
}

impl CheckDeserialization for MultiVectorTolerance {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl Example for MultiVectorTolerance {
    fn example() -> Self {
        Self {
            min_time_regression: NonNegativeFinite::new(0.05)
                .expect("0.05 is a valid non-negative finite"),
        }
    }
}
