/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_benchmark_runner::{files::InputFile, Checker};
use serde::{Deserialize, Serialize};

use super::graph_ivf::FileType;
use super::{as_input, Example};

as_input!(GraphIvfTwoMeans);

/// Matrix of complete two-means workloads.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfTwoMeans {
    /// Omit for synthetic rows. A file source is sampled without replacement.
    pub data: Option<InputFile>,
    #[serde(default)]
    pub data_type: FileType,
    pub bucket_sizes: Vec<NonZeroUsize>,
    pub dimensions: Vec<NonZeroUsize>,
    pub iterations: Vec<NonZeroUsize>,
    pub layouts: Vec<Layout>,
    pub seed: u64,
    pub repetitions: NonZeroUsize,
    pub warmup_repetitions: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Layout {
    Contiguous,
    Scattered,
}

impl GraphIvfTwoMeans {
    pub(crate) const fn tag() -> &'static str {
        "graph-ivf-two-means"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        if let Some(file) = &mut self.data {
            file.resolve(checker)?;
        }
        anyhow::ensure!(
            !self.bucket_sizes.is_empty()
                && !self.dimensions.is_empty()
                && !self.iterations.is_empty()
                && !self.layouts.is_empty(),
            "all workload axes must be nonempty"
        );
        anyhow::ensure!(
            self.bucket_sizes.iter().all(|size| size.get() >= 2),
            "two-means bucket sizes must be at least two"
        );
        Ok(())
    }
}

impl Example for GraphIvfTwoMeans {
    fn example() -> Self {
        Self {
            data: None,
            data_type: FileType::Float32,
            bucket_sizes: [32, 128, 512, 1024, 4096]
                .map(|value| NonZeroUsize::new(value).unwrap())
                .to_vec(),
            dimensions: [64, 128, 384, 768, 1536]
                .map(|value| NonZeroUsize::new(value).unwrap())
                .to_vec(),
            iterations: [1, 5, 10]
                .map(|value| NonZeroUsize::new(value).unwrap())
                .to_vec(),
            layouts: vec![Layout::Contiguous, Layout::Scattered],
            seed: 42,
            repetitions: NonZeroUsize::new(7).unwrap(),
            warmup_repetitions: 1,
        }
    }
}
