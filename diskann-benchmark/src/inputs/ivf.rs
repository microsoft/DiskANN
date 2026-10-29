/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::{ensure, Context};
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};

use crate::inputs::{as_input, write_field, Example, PRINT_WIDTH};

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
            ensure!(
                search.groundtruth.is_none() || search.queries.is_some(),
                "groundtruth requires a query file"
            );
            ensure!(!search.nprobe.is_empty(), "nprobe must not be empty");
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
        write_field!(f, "Seed", self.build.seed)?;
        write_field!(f, "Build Threads", self.build.num_threads)?;
        let split_mode = match self.build.split_mode {
            SplitMode::Baseline => "baseline",
            SplitMode::ProjectionGemm => "projection-gemm",
        };
        write_field!(f, "Split Mode", split_mode)?;
        if let Some(search) = &self.search {
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

/// Named benchmark comparisons; kernel policies remain independent in the library.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum SplitMode {
    #[default]
    Baseline,
    ProjectionGemm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Build {
    pub(crate) batch_size: NonZeroUsize,
    pub(crate) split_threshold: usize,
    /// Zero still performs one Lloyd iteration, matching the index configuration.
    pub(crate) iterations: usize,
    pub(crate) seed: u64,
    pub(crate) num_threads: NonZeroUsize,
    pub(crate) split_mode: SplitMode,
}

impl Build {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.split_threshold >= 2,
            "split threshold must be at least 2"
        );
        ensure!(
            self.num_threads.get() == 1 || cfg!(feature = "ivf-parallel"),
            "multiple IVF build threads require ivf-parallel"
        );
        ensure!(
            self.split_mode == SplitMode::Baseline || cfg!(feature = "ivf-experimental-compute"),
            "projection-gemm requires ivf-experimental-compute"
        );
        Ok(())
    }
}

impl Default for Build {
    fn default() -> Self {
        Self {
            batch_size: NonZeroUsize::new(256).unwrap(),
            split_threshold: 256,
            iterations: 10,
            seed: 7,
            num_threads: NonZeroUsize::new(1).unwrap(),
            split_mode: SplitMode::Baseline,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct SearchPhase {
    pub(crate) queries: Option<InputFile>,
    /// None computes exact groundtruth from the loaded dataset.
    pub(crate) groundtruth: Option<InputFile>,
    pub(crate) k: NonZeroUsize,
    pub(crate) nprobe: Vec<NonZeroUsize>,
    pub(crate) reps: NonZeroUsize,
    pub(crate) query_limit: Option<NonZeroUsize>,
    /// Restricts only the full-list correctness check; None checks all queries.
    pub(crate) full_probe_queries: Option<NonZeroUsize>,
}

impl Default for SearchPhase {
    fn default() -> Self {
        Self {
            queries: None,
            groundtruth: None,
            k: NonZeroUsize::new(1000).unwrap(),
            nprobe: [1, 2, 4, 8, 16, 32]
                .map(|n| NonZeroUsize::new(n).unwrap())
                .to_vec(),
            reps: NonZeroUsize::new(3).unwrap(),
            query_limit: None,
            full_probe_queries: None,
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
