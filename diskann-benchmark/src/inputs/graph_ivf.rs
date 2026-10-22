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
#[serde(deny_unknown_fields)]
pub(crate) struct Dataset {
    /// Float32 .bin vectors; groundtruth uses uint32 row IDs.
    pub data: InputFile,
    pub queries: Option<InputFile>,
    /// If absent, compute exhaustive groundtruth outside all timers.
    pub groundtruth: Option<InputFile>,
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
        self.dataset.data.resolve(checker)?;
        if self.search.is_some() {
            self.dataset
                .queries
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("search requires a queries file"))?
                .resolve(checker)?;
            if let Some(path) = &mut self.dataset.groundtruth {
                path.resolve(checker)?;
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
            dataset: Dataset {
                data: InputFile::new("disk_index_siftsmall_learn_256pts_data.fbin"),
                queries: Some(InputFile::new("disk_index_sample_query_10pts.fbin")),
                groundtruth: Some(InputFile::new(
                    "disk_index_10pts_idx_uint32_exact_ground_truth.bin",
                )),
            },
            build: Build {
                initial_centroids: NonZeroUsize::new(8).unwrap(),
                batch_size: NonZeroUsize::new(64).unwrap(),
                split_threshold: 64,
                two_means_iterations: NonZeroUsize::new(5).unwrap(),
                seed: 7,
                repetitions: NonZeroUsize::new(3).unwrap(),
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
