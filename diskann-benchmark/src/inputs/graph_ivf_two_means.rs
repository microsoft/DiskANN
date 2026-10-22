/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, path::PathBuf};

use diskann::ivf::clustering;
use diskann_benchmark_runner::{files::InputFile, Checker};
use serde::{Deserialize, Serialize};

use super::graph_ivf::FileType;
use super::{as_input, Example};

as_input!(GraphIvfTwoMeans);
as_input!(GraphIvfParentReplay);

/// Independent diagnostic build followed by complete fits on predetermined parents.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfParentReplay {
    pub index: super::graph_ivf::GraphIvfBuild,
    pub directory: PathBuf,
    pub parent_ordinals: Vec<usize>,
    pub seed: u64,
    pub repetitions: NonZeroUsize,
    pub warmup_repetitions: usize,
}

impl GraphIvfParentReplay {
    pub(crate) const fn tag() -> &'static str {
        "graph-ivf-parent-replay"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.index.validate(checker)?;
        anyhow::ensure!(
            self.index.search.is_none() && self.index.build.repetitions.get() == 1,
            "parent export needs one diagnostic build without search"
        );
        anyhow::ensure!(
            self.index.build.two_means_strategy == TwoMeansStrategy::PairedPoint,
            "all replay parents must come from the paired baseline"
        );
        anyhow::ensure!(
            !self.parent_ordinals.is_empty()
                && self.parent_ordinals.windows(2).all(|p| p[0] < p[1]),
            "parent ordinals must be nonempty, sorted and unique"
        );
        anyhow::ensure!(
            !self.directory.exists(),
            "use a new parent export directory"
        );
        Ok(())
    }
}

impl Example for GraphIvfParentReplay {
    fn example() -> Self {
        let mut index = super::graph_ivf::GraphIvfBuild::example();
        index.search = None;
        index.build.repetitions = NonZeroUsize::new(1).unwrap();
        Self {
            index,
            directory: PathBuf::from("parent-replay"),
            parent_ordinals: vec![0, 32, 256],
            seed: 7,
            repetitions: NonZeroUsize::new(7).unwrap(),
            warmup_repetitions: 1,
        }
    }
}

/// CLI spelling and experiment order are tooling concerns, separate from kernels.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TwoMeansStrategy {
    #[default]
    PairedPoint,
    #[serde(rename = "transposed_l2_16")]
    TransposedL2,
    #[serde(rename = "transposed_dot_16")]
    TransposedDot,
}

impl TwoMeansStrategy {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 3] = [Self::PairedPoint, Self::TransposedL2, Self::TransposedDot];

    #[cfg(test)]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::PairedPoint => "paired_point",
            Self::TransposedL2 => "transposed_l2_16",
            Self::TransposedDot => "transposed_dot_16",
        }
    }
}

impl From<TwoMeansStrategy> for clustering::Strategy {
    fn from(value: TwoMeansStrategy) -> Self {
        match value {
            TwoMeansStrategy::PairedPoint => Self::PairedPoint,
            TwoMeansStrategy::TransposedL2 => Self::TransposedL2,
            TwoMeansStrategy::TransposedDot => Self::TransposedDot,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FitStrategy {
    PairedPoint,
    #[serde(rename = "transposed_l2_16")]
    TransposedL2,
    #[serde(rename = "transposed_dot_16")]
    TransposedDot,
    TransposedLloydReference,
}

impl FitStrategy {
    pub(crate) const ALL: [Self; 4] = [
        Self::PairedPoint,
        Self::TransposedL2,
        Self::TransposedDot,
        Self::TransposedLloydReference,
    ];
    pub(crate) fn all() -> Vec<Self> {
        Self::ALL.to_vec()
    }
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::PairedPoint => "paired_point",
            Self::TransposedL2 => "transposed_l2_16",
            Self::TransposedDot => "transposed_dot_16",
            Self::TransposedLloydReference => "transposed_lloyd_reference",
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FitOrder {
    #[default]
    Rotating,
    Abba,
}

/// Matrix of complete two-means workloads.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfTwoMeans {
    #[serde(default = "FitStrategy::all")]
    pub strategies: Vec<FitStrategy>,
    #[serde(default)]
    pub order: FitOrder,
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
        anyhow::ensure!(
            !self.strategies.is_empty()
                && self
                    .strategies
                    .iter()
                    .enumerate()
                    .all(|(i, value)| !self.strategies[..i].contains(value)),
            "fit strategies must be nonempty and unique"
        );
        if self.order == FitOrder::Abba {
            anyhow::ensure!(
                self.strategies.len() == 2 && self.strategies[0] == FitStrategy::PairedPoint,
                "ABBA needs exactly [paired_point, candidate]; repetitions count complete blocks"
            );
        }
        Ok(())
    }
}

impl Example for GraphIvfTwoMeans {
    fn example() -> Self {
        Self {
            strategies: FitStrategy::all(),
            order: FitOrder::Rotating,
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
