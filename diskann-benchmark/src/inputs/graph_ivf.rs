/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::{files::InputFile, Checker};
use serde::{Deserialize, Serialize};

use super::{as_input, Example};

as_input!(GraphIvfBuild);

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfBuild {
    pub dataset: Dataset,
    pub build: Build,
    pub graph: Graph,
    /// Omit for a build-only measurement.
    pub search: Option<Search>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Dataset {
    Synthetic {
        vectors: NonZeroUsize,
        queries: NonZeroUsize,
        dimensions: NonZeroUsize,
        seed: u64,
    },
    /// Typed .bin vectors; groundtruth uses uint32 row IDs.
    Files {
        data: InputFile,
        queries: Option<InputFile>,
        /// If absent, compute exhaustive groundtruth outside all timers.
        groundtruth: Option<InputFile>,
        #[serde(default)]
        data_type: FileType,
    },
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FileType {
    #[default]
    Float32,
    Float16,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Build {
    pub initial_centroids: NonZeroUsize,
    pub batch_size: NonZeroUsize,
    pub split_threshold: usize,
    pub two_means_iterations: NonZeroUsize,
    pub seed: u64,
    pub repetitions: NonZeroUsize,
    #[cfg(feature = "ivf-two-means")]
    #[serde(default = "default_two_means_strategy")]
    pub two_means_strategy: String,
}

#[cfg(feature = "ivf-two-means")]
fn default_two_means_strategy() -> String {
    "paired_point".into()
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Graph {
    pub max_degree: NonZeroUsize,
    pub l_build: NonZeroUsize,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Search {
    pub k: NonZeroUsize,
    pub nprobe: Vec<NonZeroUsize>,
    pub l_search: NonZeroUsize,
    pub warmup_repetitions: usize,
    pub repetitions: NonZeroUsize,
}

impl GraphIvfBuild {
    pub(crate) const fn tag() -> &'static str {
        "graph-ivf-build"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        #[cfg(feature = "ivf-two-means")]
        anyhow::ensure!(
            diskann::ivf::two_means_benchmark::Strategy::parse(&self.build.two_means_strategy)
                .is_some(),
            "unknown two_means_strategy: {}",
            self.build.two_means_strategy
        );
        anyhow::ensure!(
            self.build.split_threshold >= 2,
            "split_threshold must be >= 2"
        );
        self.graph.config()?;
        if let Some(search) = &self.search {
            anyhow::ensure!(!search.nprobe.is_empty(), "nprobe must not be empty");
            anyhow::ensure!(
                search.nprobe.iter().all(|n| *n <= search.l_search),
                "l_search must be >= every requested nprobe"
            );
        }
        match &mut self.dataset {
            Dataset::Synthetic { vectors, .. } => self.build.validate_rows(vectors.get())?,
            Dataset::Files {
                data,
                queries,
                groundtruth,
                ..
            } => {
                data.resolve(checker)?;
                if self.search.is_some() {
                    queries
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("search requires a queries file"))?
                        .resolve(checker)?;
                    if let Some(path) = groundtruth {
                        path.resolve(checker)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl Build {
    pub(crate) fn validate_rows(&self, rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            rows > 0 && rows < u32::MAX as usize,
            "dataset must contain 1..u32::MAX vectors"
        );
        anyhow::ensure!(
            self.initial_centroids.get() <= rows,
            "initial_centroids exceeds the dataset size"
        );
        Ok(())
    }
}

impl Graph {
    pub(crate) fn config(&self) -> diskann::ANNResult<diskann::graph::Config> {
        Ok(diskann::graph::config::Builder::new(
            self.max_degree.get(),
            diskann::graph::config::MaxDegree::default_slack(),
            self.l_build.get(),
            diskann_vector::distance::Metric::L2.into(),
        )
        .build()?)
    }
}

impl Example for GraphIvfBuild {
    fn example() -> Self {
        Self {
            dataset: Dataset::Synthetic {
                vectors: NonZeroUsize::new(8192).unwrap(),
                queries: NonZeroUsize::new(128).unwrap(),
                dimensions: NonZeroUsize::new(128).unwrap(),
                seed: 42,
            },
            build: Build {
                initial_centroids: NonZeroUsize::new(8).unwrap(),
                batch_size: NonZeroUsize::new(256).unwrap(),
                split_threshold: 512,
                two_means_iterations: NonZeroUsize::new(5).unwrap(),
                seed: 7,
                repetitions: NonZeroUsize::new(3).unwrap(),
                #[cfg(feature = "ivf-two-means")]
                two_means_strategy: default_two_means_strategy(),
            },
            graph: Graph {
                max_degree: NonZeroUsize::new(16).unwrap(),
                l_build: NonZeroUsize::new(64).unwrap(),
            },
            search: Some(Search {
                k: NonZeroUsize::new(10).unwrap(),
                nprobe: [1, 4, 16].map(|n| NonZeroUsize::new(n).unwrap()).to_vec(),
                l_search: NonZeroUsize::new(64).unwrap(),
                warmup_repetitions: 1,
                repetitions: NonZeroUsize::new(5).unwrap(),
            }),
        }
    }
}
