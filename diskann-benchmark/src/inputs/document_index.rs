/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Input types for document index benchmarks using DocumentInsertStrategy.

use std::num::NonZeroUsize;

use anyhow::Context;
use diskann_benchmark_runner::{
    files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker,
};
use serde::{Deserialize, Serialize};

use super::async_::GraphSearch;
use crate::inputs::{as_input, Example, Input};

//////////////
// Registry //
//////////////

as_input!(DocumentIndexBuild);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<DocumentIndexBuild>::new())?;
    Ok(())
}

/// Build parameters for document index construction.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentBuildParams {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) data_labels: InputFile,
    pub(crate) distance: crate::utils::SimilarityMeasure,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    pub(crate) alpha: f32,
    pub(crate) num_threads: usize,
}

impl CheckDeserialization for DocumentBuildParams {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.check_deserialization(checker)?;
        self.data_labels.check_deserialization(checker)?;

        // checking if the max_degree, l_build and alpha values are valid.
        use diskann::graph::config::{Builder, MaxDegree, PruneKind};
        let mut builder = Builder::new(
            self.max_degree,
            MaxDegree::Value(self.max_degree),
            self.l_build,
            PruneKind::Occluding,
        );
        builder.alpha(self.alpha);
        builder.build()?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentSearchParams {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) beta: f32,
    #[serde(default = "default_reps")]
    pub(crate) reps: NonZeroUsize,
    #[serde(default = "default_thread_counts")]
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

fn default_reps() -> NonZeroUsize {
    NonZeroUsize::new(5).unwrap()
}
fn default_thread_counts() -> Vec<NonZeroUsize> {
    vec![NonZeroUsize::new(1).unwrap()]
}

impl CheckDeserialization for DocumentSearchParams {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.check_deserialization(checker)?;
        self.query_predicates.check_deserialization(checker)?;
        self.groundtruth.check_deserialization(checker)?;
        if self.beta <= 0.0 || self.beta > 1.0 {
            return Err(anyhow::anyhow!(
                "beta must be in range (0, 1], got: {}",
                self.beta
            ));
        }
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentIndexBuild {
    pub(crate) build: DocumentBuildParams,
    pub(crate) search: DocumentSearchParams,
}

impl DocumentIndexBuild {
    pub(crate) const fn tag() -> &'static str {
        "document-index-build"
    }
}

impl CheckDeserialization for DocumentIndexBuild {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.build.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;
        Ok(())
    }
}

impl Example for DocumentIndexBuild {
    fn example() -> Self {
        Self {
            build: DocumentBuildParams {
                data_type: DataType::Float32,
                data: InputFile::new("data.fbin"),
                data_labels: InputFile::new("data.label.jsonl"),
                distance: crate::utils::SimilarityMeasure::SquaredL2,
                max_degree: 32,
                l_build: 50,
                alpha: 1.2,
                num_threads: 1,
            },
            search: DocumentSearchParams {
                queries: InputFile::new("queries.fbin"),
                query_predicates: InputFile::new("query.label.jsonl"),
                groundtruth: InputFile::new("groundtruth.bin"),
                beta: 0.5,
                reps: default_reps(),
                num_threads: default_thread_counts(),
                runs: vec![GraphSearch {
                    search_n: 10,
                    search_l: vec![20, 30, 40, 50],
                    recall_k: 10,
                }],
            },
        }
    }
}

impl std::fmt::Display for DocumentIndexBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Document Index Build with Label Filters\n")?;
        writeln!(f, "tag: \"{}\"", Self::tag())?;
        writeln!(
            f,
            "\nBuild: data={}, labels={}, R={}, L={}, alpha={}",
            self.build.data.display(),
            self.build.data_labels.display(),
            self.build.max_degree,
            self.build.l_build,
            self.build.alpha
        )?;
        writeln!(
            f,
            "Search: queries={}, beta={}",
            self.search.queries.display(),
            self.search.beta
        )?;
        Ok(())
    }
}
