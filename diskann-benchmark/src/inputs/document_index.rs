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
    #[serde(default)]
    pub(crate) search_algorithm: DocumentSearchAlgorithm,
    /// Adaptive-L scaling for `ast-label-provider` search. Omit to disable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) adaptive_l: Option<AdaptiveLConfig>,
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
        if let Some(ref al) = self.adaptive_l {
            if al.scale_factor < 1.0 {
                return Err(anyhow::anyhow!(
                    "adaptive_l.scale_factor must be >= 1.0, got {}",
                    al.scale_factor
                ));
            }
            if al.sample_count == 0 {
                return Err(anyhow::anyhow!("adaptive_l.sample_count must be > 0"));
            }
        }
        Ok(())
    }
}

/// Configuration for adaptive-L scaling during inline filtered search.
///
/// After visiting `sample_count` nodes, specificity is estimated from matched
/// results and `l_search` is scaled up by up to `scale_factor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AdaptiveLConfig {
    /// Number of nodes to visit before estimating filter specificity.
    pub(crate) sample_count: usize,
    /// Maximum multiplier applied to `l_search` (must be ≥ 1.0).
    pub(crate) scale_factor: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum DocumentSearchAlgorithm {
    #[default]
    Auto,
    InlineBeta,
    AstLabelProvider,
    Bitmap,
    /// Bitmap-precomputed label provider with InlineFilterSearch hard-rejection traversal.
    /// Combines O(1) per-node bitmap lookup with adaptive-L support.
    BitmapInline,
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
                search_algorithm: DocumentSearchAlgorithm::default(),
                adaptive_l: None,
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
            "Search: queries={}, beta={}, algorithm={:?}",
            self.search.queries.display(),
            self.search.beta,
            self.search.search_algorithm
        )?;
        Ok(())
    }
}
