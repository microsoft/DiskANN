/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::{ensure, Context};
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, write_field, Example, PRINT_WIDTH},
    utils::SimilarityMeasure,
};

as_input!(IvfBuildSearch);

/// Build an IVF index and optionally search it using an in-memory provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IvfBuildSearch {
    /// None selects the fixed synthetic smoke dataset and its held-out queries.
    pub(crate) data: Option<InputFile>,
    #[serde(default = "default_data_type")]
    pub(crate) data_type: DataType,
    #[serde(default)]
    pub(crate) format: DataFormat,
    #[serde(default)]
    pub(crate) build: Build,
    /// None runs only the build phase.
    pub(crate) search: Option<SearchPhase>,
}

impl IvfBuildSearch {
    pub(crate) const fn tag() -> &'static str {
        "ivf-build-search"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        ensure!(
            matches!(self.data_type, DataType::Float32 | DataType::Float16),
            "IVF benchmark supports float32 and float16 input"
        );
        ensure!(
            self.format != DataFormat::Fvecs || self.data_type == DataType::Float32,
            "fvecs input requires float32 data"
        );
        self.build.validate()?;
        if let Some(search) = &mut self.search {
            ensure!(
                self.data.is_some() == search.queries.is_some(),
                "provide both dataset and query files, or neither for synthetic data"
            );
            search.validate()?;
            if let Some(queries) = &mut search.queries {
                queries.resolve(checker).context("resolving queries file")?;
            }
            if let Some(groundtruth) = &mut search.groundtruth {
                groundtruth
                    .resolve(checker)
                    .context("resolving groundtruth file")?;
            }
        }
        if let Some(data) = &mut self.data {
            data.resolve(checker).context("resolving dataset file")?;
        }
        Ok(())
    }
}

impl std::fmt::Display for IvfBuildSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.data {
            Some(data) => write_field!(f, "Data", data.display())?,
            None => write_field!(f, "Data", "synthetic smoke (4096 base, 128 queries, D=32)")?,
        }
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Format", self.format)?;
        write_field!(f, "Batch Size", self.build.batch_size)?;
        write_field!(f, "Split Threshold", self.build.split_threshold)?;
        write_field!(f, "Iterations", self.build.iterations)?;
        write_field!(f, "Split K", self.build.split_k)?;
        write_field!(f, "Reassignment Neighbors", self.build.reassign_count)?;
        write_field!(
            f,
            "Reassignment Mode",
            format!("{:?}", self.build.reassignment_mode)
        )?;
        if let Some(search) = &self.search {
            write_field!(f, "Search Distance", search.distance)?;
            if let Some(queries) = &search.queries {
                write_field!(f, "Queries", queries.display())?;
            }
            if let Some(groundtruth) = &search.groundtruth {
                write_field!(f, "Groundtruth", groundtruth.display())?;
            }
            write_field!(f, "K", search.k)?;
            write_field!(
                f,
                "Nprobe",
                search
                    .nprobe
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
            if !search.target_scan_ratios.is_empty() {
                write_field!(
                    f,
                    "Scan Targets",
                    search
                        .target_scan_ratios
                        .iter()
                        .map(|ratio| format!("{:.1}%", ratio * 100.0))
                        .collect::<Vec<_>>()
                        .join(", ")
                )?;
            }
            write_field!(f, "Reps", search.reps)?;
        } else {
            write_field!(f, "Search", "disabled")?;
        }
        Ok(())
    }
}

impl Example for IvfBuildSearch {
    fn example() -> Self {
        Self {
            data: Some(InputFile::new("path/to/data.bin")),
            data_type: DataType::Float32,
            format: DataFormat::Bin,
            build: Build::default(),
            search: Some(SearchPhase::example()),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DataFormat {
    #[default]
    Bin,
    Fvecs,
}

impl std::fmt::Display for DataFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Bin => "bin",
            Self::Fvecs => "fvecs",
        })
    }
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReassignmentMode {
    #[default]
    Baseline,
    #[cfg(feature = "ivf-gemm")]
    Gemm,
}

impl From<ReassignmentMode> for diskann::ivf::ReassignmentMode {
    fn from(mode: ReassignmentMode) -> Self {
        match mode {
            ReassignmentMode::Baseline => Self::Baseline,
            #[cfg(feature = "ivf-gemm")]
            ReassignmentMode::Gemm => Self::Gemm,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Build {
    pub(crate) batch_size: NonZeroUsize,
    pub(crate) split_threshold: NonZeroUsize,
    pub(crate) split_k: NonZeroUsize,
    pub(crate) reassign_count: usize,
    pub(crate) iterations: NonZeroUsize,
    /// Record every published layout outside the measured insertion interval.
    pub(crate) record_layout: bool,
    pub(crate) reassignment_mode: ReassignmentMode,
}

impl Build {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.split_k.get() >= 2 && self.split_k.get() - 1 <= self.split_threshold.get(),
            "split_k must be in 2..=split_threshold+1"
        );
        Ok(())
    }
}

impl Default for Build {
    fn default() -> Self {
        Self {
            batch_size: NonZeroUsize::new(256).unwrap(),
            split_threshold: NonZeroUsize::new(256).unwrap(),
            split_k: NonZeroUsize::new(2).unwrap(),
            reassign_count: 8,
            iterations: NonZeroUsize::new(10).unwrap(),
            record_layout: false,
            reassignment_mode: ReassignmentMode::Baseline,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SearchPhase {
    /// Search metric. CosineNormalized requires unit-length base vectors and queries.
    pub(crate) distance: SimilarityMeasure,
    pub(crate) queries: Option<InputFile>,
    /// None computes exact groundtruth from the loaded dataset.
    pub(crate) groundtruth: Option<InputFile>,
    pub(crate) k: NonZeroUsize,
    pub(crate) nprobe: Vec<NonZeroUsize>,
    pub(crate) reps: NonZeroUsize,
    pub(crate) query_limit: Option<NonZeroUsize>,
    /// Restricts only the full-list correctness check; None checks all queries.
    pub(crate) full_probe_queries: Option<NonZeroUsize>,
    /// Add one fixed nprobe per mean vector scan target over all selected queries.
    #[serde(
        alias = "target_scan_ratio",
        deserialize_with = "deserialize_scan_targets"
    )]
    pub(crate) target_scan_ratios: Vec<f64>,
}

impl SearchPhase {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.groundtruth.is_none() || self.queries.is_some(),
            "groundtruth requires a query file"
        );
        ensure!(
            !self.nprobe.is_empty() || !self.target_scan_ratios.is_empty(),
            "nprobe must not be empty without a scan target"
        );
        for &ratio in &self.target_scan_ratios {
            ensure!(
                ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
                "scan target must be in (0, 1]"
            );
        }
        Ok(())
    }
}

impl Default for SearchPhase {
    fn default() -> Self {
        Self {
            distance: SimilarityMeasure::SquaredL2,
            queries: None,
            groundtruth: None,
            k: NonZeroUsize::new(1000).unwrap(),
            nprobe: [1, 2, 4, 8, 16, 32]
                .map(|n| NonZeroUsize::new(n).unwrap())
                .to_vec(),
            reps: NonZeroUsize::new(3).unwrap(),
            query_limit: None,
            full_probe_queries: None,
            target_scan_ratios: Vec::new(),
        }
    }
}

impl Example for SearchPhase {
    fn example() -> Self {
        Self {
            queries: Some(InputFile::new("path/to/queries.bin")),
            groundtruth: Some(InputFile::new("path/to/groundtruth.bin")),
            ..Self::default()
        }
    }
}

fn default_data_type() -> DataType {
    DataType::Float32
}

// Continue accepting the original scalar/null input while serializing a list of targets.
fn deserialize_scan_targets<'de, D>(deserializer: D) -> Result<Vec<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Targets {
        One(f64),
        Many(Vec<f64>),
    }

    Ok(match Option::<Targets>::deserialize(deserializer)? {
        Some(Targets::One(ratio)) => vec![ratio],
        Some(Targets::Many(ratios)) => ratios,
        None => Vec::new(),
    })
}
