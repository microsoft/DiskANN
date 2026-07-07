/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::Context;
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, write_field, Example, PRINT_WIDTH},
    utils::SimilarityMeasure,
};

//////////////
// Registry //
//////////////

as_input!(FlatSearch);

///////////
// Input //
///////////

/// Input specification for a flat-index (brute-force kNN) benchmark.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FlatSearch {
    /// Path to the dataset vectors (`.bin` format).
    pub(crate) data: InputFile,

    /// The on-disk data type of the dataset.
    pub(crate) data_type: DataType,

    /// The distance metric to use.
    pub(crate) distance: SimilarityMeasure,

    /// Search configuration.
    pub(crate) search: SearchPhase,
}

impl FlatSearch {
    pub(crate) const fn tag() -> &'static str {
        "flat-search"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.data.resolve(checker)?;
        self.search.validate(checker)?;
        Ok(())
    }
}

impl std::fmt::Display for FlatSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "Data", self.data.display())?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Distance", self.distance)?;
        write_field!(f, "Queries", self.search.queries.display())?;
        write_field!(f, "Groundtruth", self.search.groundtruth.display())?;
        write_field!(f, "K", self.search.k)?;
        write_field!(
            f,
            "Threads",
            self.search
                .num_threads
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write_field!(f, "Reps", self.search.reps)?;
        Ok(())
    }
}

impl Example for FlatSearch {
    fn example() -> Self {
        Self {
            data: InputFile::new("path/to/data.bin"),
            data_type: DataType::Float32,
            distance: SimilarityMeasure::SquaredL2,
            search: SearchPhase::example(),
        }
    }
}

///////////////////
// Search Phase  //
///////////////////

/// Parameters controlling the search phase of a flat benchmark.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SearchPhase {
    /// Path to the query vectors (`.bin` format).
    pub(crate) queries: InputFile,

    /// Path to the groundtruth file (`.bin` format).
    pub(crate) groundtruth: InputFile,

    /// The number of nearest neighbors to retrieve per query.
    pub(crate) k: NonZeroUsize,

    /// Number of threads to use for parallel query execution.
    pub(crate) num_threads: Vec<NonZeroUsize>,

    /// Number of repetitions per configuration for stable timing.
    pub(crate) reps: NonZeroUsize,
}

impl SearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.queries
            .resolve(checker)
            .context("resolving queries file")?;
        self.groundtruth
            .resolve(checker)
            .context("resolving groundtruth file")?;
        Ok(())
    }
}

impl Example for SearchPhase {
    fn example() -> Self {
        Self {
            queries: InputFile::new("path/to/queries.bin"),
            groundtruth: InputFile::new("path/to/groundtruth.bin"),
            k: NonZeroUsize::new(10).unwrap(),
            num_threads: vec![
                NonZeroUsize::new(1).unwrap(),
                NonZeroUsize::new(4).unwrap(),
                NonZeroUsize::new(8).unwrap(),
            ],
            reps: NonZeroUsize::new(5).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::Example;

    #[test]
    fn example_flat_search_round_trips() {
        let example = FlatSearch::example();
        let json = serde_json::to_value(&example).unwrap();
        let _: FlatSearch = serde_json::from_value(json).unwrap();
    }

    #[test]
    fn display_flat_search() {
        let example = FlatSearch::example();
        let text = format!("{}", example);
        assert!(text.contains("Data"));
        assert!(text.contains("Threads"));
        assert!(text.contains("Reps"));
    }
}
