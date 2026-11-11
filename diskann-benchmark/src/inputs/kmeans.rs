/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::Checker;
use serde::{Deserialize, Serialize};

use super::{as_input, Example};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum KmeansImplementation {
    Disk,
    Quantization,
}

impl std::fmt::Display for KmeansImplementation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disk => write!(f, "disk"),
            Self::Quantization => write!(f, "quantization"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum KmeansPhase {
    All,
    Init,
}

impl std::fmt::Display for KmeansPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::All => write!(f, "all"),
            Self::Init => write!(f, "init"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KmeansComparison {
    pub(crate) implementation: KmeansImplementation,
    pub(crate) phase: KmeansPhase,
    pub(crate) num_points: NonZeroUsize,
    pub(crate) dimensions: Vec<NonZeroUsize>,
    pub(crate) center_counts: Vec<NonZeroUsize>,
    pub(crate) max_iterations: NonZeroUsize,
    pub(crate) thread_counts: Vec<NonZeroUsize>,
    pub(crate) measurements: NonZeroUsize,
    pub(crate) seed: u64,
}

impl KmeansComparison {
    pub(crate) const fn tag() -> &'static str {
        "kmeans-comparison"
    }

    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        anyhow::ensure!(!self.dimensions.is_empty(), "dimensions cannot be empty");
        anyhow::ensure!(
            !self.center_counts.is_empty(),
            "center_counts cannot be empty"
        );
        anyhow::ensure!(
            !self.thread_counts.is_empty(),
            "thread_counts cannot be empty"
        );
        anyhow::ensure!(
            self.center_counts
                .iter()
                .all(|count| count.get() <= self.num_points.get()),
            "center counts cannot exceed num_points"
        );
        Ok(())
    }
}

impl Example for KmeansComparison {
    fn example() -> Self {
        Self {
            implementation: KmeansImplementation::Quantization,
            phase: KmeansPhase::All,
            num_points: NonZeroUsize::new(10_000).unwrap(),
            dimensions: vec![NonZeroUsize::new(128).unwrap()],
            center_counts: vec![NonZeroUsize::new(64).unwrap()],
            max_iterations: NonZeroUsize::new(3).unwrap(),
            thread_counts: vec![NonZeroUsize::new(1).unwrap()],
            measurements: NonZeroUsize::new(10).unwrap(),
            seed: 42,
        }
    }
}

as_input!(KmeansComparison);

impl std::fmt::Display for KmeansComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "K-means Implementation Benchmark\n")?;
        writeln!(f, "{:>18}: {}", "implementation", self.implementation)?;
        writeln!(f, "{:>18}: {}", "phase", self.phase)?;
        writeln!(f, "{:>18}: {}", "points", self.num_points)?;
        writeln!(f, "{:>18}: {:?}", "dimensions", self.dimensions)?;
        writeln!(f, "{:>18}: {:?}", "center counts", self.center_counts)?;
        writeln!(f, "{:>18}: {}", "max iterations", self.max_iterations)?;
        writeln!(f, "{:>18}: {:?}", "thread counts", self.thread_counts)?;
        writeln!(f, "{:>18}: {}", "measurements", self.measurements)?;
        writeln!(f, "{:>18}: {}", "seed", self.seed)
    }
}
