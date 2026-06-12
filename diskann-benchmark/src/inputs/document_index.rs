/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Input types for document index benchmarks using DocumentInsertStrategy.

use std::num::NonZeroUsize;

use diskann::graph::{
    config::{Builder, ConfigError, MaxDegree, PruneKind},
    Config,
};
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};

use super::graph_index::GraphSearch;
use crate::inputs::{as_input, Example};

//////////////
// Registry //
//////////////

as_input!(DocumentIndexBuild);

/// Build parameters for document index construction.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentBuildParams {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) data_labels: InputFile,
    #[serde(default = "default_use_ast_filters")]
    pub(crate) use_ast_filters: bool,
    pub(crate) distance: crate::utils::SimilarityMeasure,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    pub(crate) alpha: f32,
    pub(crate) num_threads: usize,
}

impl DocumentBuildParams {
    pub(crate) fn build_config(&self) -> Result<Config, ConfigError> {
        let metric = self.distance.into();
        let prune_kind = PruneKind::from_metric(metric);
        let mut config_builder = Builder::new(
            self.max_degree,            // pruned_degree
            MaxDegree::default_slack(), // max_degree
            self.l_build,
            prune_kind,
        );
        config_builder.alpha(self.alpha);
        let config = config_builder.build()?;
        Ok(config)
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.data.resolve(checker)?;
        self.data_labels.resolve(checker)?;
        self.build_config()?;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DocumentSearchParams {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) groundtruth: InputFile,
    #[serde(default = "default_use_ast_filters")]
    pub(crate) use_ast_filters: bool,
    pub(crate) beta: f32,
    pub(crate) reps: NonZeroUsize,
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl DocumentSearchParams {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.queries.resolve(checker)?;
        self.query_predicates.resolve(checker)?;
        self.groundtruth.resolve(checker)?;
        Ok(())
    }
}

fn default_use_ast_filters() -> bool {
    true
}

fn default_reps() -> NonZeroUsize {
    NonZeroUsize::new(5).unwrap()
}
fn default_thread_counts() -> Vec<NonZeroUsize> {
    vec![NonZeroUsize::new(1).unwrap()]
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

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.build.validate(checker)?;
        self.search.validate(checker)?;
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
                use_ast_filters: default_use_ast_filters(),
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
                use_ast_filters: default_use_ast_filters(),
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
