/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, num::NonZeroUsize, path::Path};

use anyhow::Context;

use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
#[cfg(feature = "disk-index")]
use diskann_disk::QuantizationType;
use diskann_providers::storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, graph_index::AdaptiveL, post_processor::TopkPostProcessor, Example},
    utils::SimilarityMeasure,
};

//////////////
// Registry //
//////////////

as_input!(DiskIndexOperation);

///////////
// Input //
///////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DiskIndexOperation {
    pub(crate) source: DiskIndexSource, // either load or build
    pub(crate) search_phase: DiskSearchPhase,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "disk-index-source")] // Use tagged enums for JSON
pub(crate) enum DiskIndexSource {
    Load(DiskIndexLoad),
    Build(DiskIndexBuild),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct DiskIndexLoad {
    pub(crate) data_type: DataType,
    pub(crate) load_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct DiskIndexBuild {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) dim: usize,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    pub(crate) num_threads: usize,
    pub(crate) build_ram_limit_gb: f64,
    pub(crate) num_pq_chunks: NonZeroUsize,
    #[cfg(feature = "disk-index")]
    pub(crate) quantization_type: QuantizationType,
    pub(crate) save_path: String,
}

/// Disk search mode, modeled after the four backend search strategies.
/// Strategy-specific settings live only on the variants that use them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub(crate) enum DiskSearchMode {
    /// Brute-force flat scan, optionally restricted by a per-query vector filter.
    Flat {
        #[serde(default)]
        vector_filters_file: Option<InputFile>,
    },
    /// Greedy graph search, optionally post-filtered by a per-query vector filter.
    Graph {
        #[serde(default)]
        vector_filters_file: Option<InputFile>,
    },
    /// Graph search that checks a required vector filter during traversal.
    GraphInlineFilter {
        vector_filters_file: InputFile,
        #[serde(default)]
        adaptive_l: Option<AdaptiveL>,
    },
    /// Graph search followed by determinant-diversity selection.
    GraphDiverse {
        #[serde(default)]
        vector_filters_file: Option<InputFile>,
        post_processor: TopkPostProcessor,
    },
}

/// Path-independent search metadata written to benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case")]
pub(crate) enum DiskSearchStrategy {
    Flat {
        uses_vector_filters: bool,
    },
    Graph {
        uses_vector_filters: bool,
    },
    GraphInlineFilter {
        #[serde(default)]
        adaptive_l: Option<AdaptiveL>,
    },
    GraphDiverse {
        uses_vector_filters: bool,
        post_processor: TopkPostProcessor,
    },
}

impl Default for DiskSearchMode {
    fn default() -> Self {
        Self::Graph {
            vector_filters_file: None,
        }
    }
}

impl DiskSearchMode {
    pub(crate) fn vector_filters_file(&self) -> Option<&InputFile> {
        match self {
            Self::Flat {
                vector_filters_file,
            }
            | Self::Graph {
                vector_filters_file,
            }
            | Self::GraphDiverse {
                vector_filters_file,
                ..
            } => vector_filters_file.as_ref(),
            Self::GraphInlineFilter {
                vector_filters_file,
                ..
            } => Some(vector_filters_file),
        }
    }

    pub(crate) fn post_processor(&self) -> Option<&TopkPostProcessor> {
        match self {
            Self::GraphDiverse { post_processor, .. } => Some(post_processor),
            _ => None,
        }
    }

    pub(crate) fn strategy(&self) -> DiskSearchStrategy {
        match self {
            Self::Flat {
                vector_filters_file,
            } => DiskSearchStrategy::Flat {
                uses_vector_filters: vector_filters_file.is_some(),
            },
            Self::Graph {
                vector_filters_file,
            } => DiskSearchStrategy::Graph {
                uses_vector_filters: vector_filters_file.is_some(),
            },
            Self::GraphInlineFilter { adaptive_l, .. } => DiskSearchStrategy::GraphInlineFilter {
                adaptive_l: adaptive_l.clone(),
            },
            Self::GraphDiverse {
                vector_filters_file,
                post_processor,
            } => DiskSearchStrategy::GraphDiverse {
                uses_vector_filters: vector_filters_file.is_some(),
                post_processor: post_processor.clone(),
            },
        }
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            Self::Flat {
                vector_filters_file,
            }
            | Self::Graph {
                vector_filters_file,
            } => {
                if let Some(vf) = vector_filters_file.as_mut() {
                    vf.resolve(checker).context("invalid vector_filters_file")?;
                }
            }
            Self::GraphInlineFilter {
                vector_filters_file,
                adaptive_l,
            } => {
                vector_filters_file
                    .resolve(checker)
                    .context("invalid vector_filters_file")?;
                if let Some(adaptive_l) = adaptive_l {
                    adaptive_l.validate(checker)?;
                }
            }
            Self::GraphDiverse {
                vector_filters_file,
                post_processor,
            } => {
                if let Some(vf) = vector_filters_file.as_mut() {
                    vf.resolve(checker).context("invalid vector_filters_file")?;
                }
                post_processor
                    .validate(checker)
                    .context("invalid disk search post processor")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for DiskSearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.strategy().fmt(f)
    }
}

impl fmt::Display for DiskSearchStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flat {
                uses_vector_filters: false,
            } => write!(f, "flat"),
            Self::Flat {
                uses_vector_filters: true,
            } => write!(f, "flat + vector-filter"),
            Self::Graph {
                uses_vector_filters: false,
            } => write!(f, "graph"),
            Self::Graph {
                uses_vector_filters: true,
            } => write!(f, "graph + vector-filter"),
            Self::GraphInlineFilter { adaptive_l: None } => write!(f, "graph inline-filter"),
            Self::GraphInlineFilter {
                adaptive_l: Some(_),
            } => write!(f, "graph inline-filter + adaptive-l"),
            Self::GraphDiverse {
                uses_vector_filters: false,
                ..
            } => write!(f, "graph diverse"),
            Self::GraphDiverse {
                uses_vector_filters: true,
                ..
            } => write!(f, "graph diverse + vector-filter"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_search_modes_deserialize() {
        let flat: DiskSearchMode =
            serde_json::from_str(r#"{ "mode": "flat" }"#).expect("flat mode must deserialize");
        assert!(matches!(flat, DiskSearchMode::Flat { .. }));

        let graph: DiskSearchMode =
            serde_json::from_str(r#"{ "mode": "graph", "vector_filters_file": "filters.bin" }"#)
                .expect("graph mode must deserialize");
        assert!(matches!(graph, DiskSearchMode::Graph { .. }));

        let inline: DiskSearchMode = serde_json::from_str(
            r#"{
                "mode": "graph-inline-filter",
                "vector_filters_file": "filters.bin",
                "adaptive_l": { "sample_count": 1, "scale_factor": 2.0 }
            }"#,
        )
        .expect("inline-filter mode must deserialize");
        assert!(matches!(
            inline,
            DiskSearchMode::GraphInlineFilter {
                adaptive_l: Some(_),
                ..
            }
        ));

        let diverse: DiskSearchMode = serde_json::from_str(
            r#"{
                "mode": "graph-diverse",
                "post_processor": {
                    "type": "determinant-diversity",
                    "power": 2.0,
                    "eta": 1.0
                }
            }"#,
        )
        .expect("diverse mode must deserialize");
        assert!(matches!(diverse, DiskSearchMode::GraphDiverse { .. }));
    }

    #[test]
    fn strategy_specific_fields_are_required() {
        let inline_error =
            serde_json::from_str::<DiskSearchMode>(r#"{ "mode": "graph-inline-filter" }"#)
                .expect_err("inline-filter mode must require a vector filter");
        assert!(inline_error.to_string().contains("vector_filters_file"));

        let diverse_error =
            serde_json::from_str::<DiskSearchMode>(r#"{ "mode": "graph-diverse" }"#)
                .expect_err("diverse mode must require a post-processor");
        assert!(diverse_error.to_string().contains("post_processor"));
    }

    #[test]
    fn search_strategy_omits_filter_file_paths() {
        let mode: DiskSearchMode = serde_json::from_str(
            r#"{
                "mode": "graph-inline-filter",
                "vector_filters_file": "private/filters.bin",
                "adaptive_l": { "sample_count": 1, "scale_factor": 2.0 }
            }"#,
        )
        .unwrap();

        let value = serde_json::to_value(mode.strategy()).unwrap();
        assert_eq!(value["mode"], "graph-inline-filter");
        assert!(value.get("vector_filters_file").is_none());
        assert_eq!(value["adaptive_l"]["sample_count"], 1);
    }

    #[test]
    fn disk_search_phase_rejects_legacy_phase_level_search_mode_fields() {
        let error = serde_json::from_str::<DiskSearchPhase>(
            r#"{
                "queries": "queries.fbin",
                "groundtruth": "groundtruth.bin",
                "num_threads": 1,
                "beam_width": 1,
                "search_list": [1],
                "recall_at": 1,
                "distance": "squared_l2",
                "is_flat_search": true
            }"#,
        )
        .expect_err("legacy phase-level search settings must be rejected");

        assert!(error.to_string().contains("is_flat_search"));
    }
}

/// Search phase configuration
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DiskSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: usize,
    pub(crate) beam_width: usize,
    pub(crate) search_list: Vec<u32>,
    pub(crate) recall_at: u32,
    #[serde(default)]
    pub(crate) search_mode: DiskSearchMode,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) num_nodes_to_cache: Option<usize>,
    pub(crate) search_io_limit: Option<usize>,
}

/////////
// Tag //
/////////

impl DiskIndexOperation {
    pub(crate) const fn tag() -> &'static str {
        "disk-index"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match &mut self.source {
            DiskIndexSource::Load(load) => load.validate(checker)?,
            DiskIndexSource::Build(build) => build.validate(checker)?,
        }
        self.search_phase.validate(checker)?;
        Ok(())
    }
}

impl DiskIndexLoad {
    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        let files = [
            (get_pq_pivot_file(&self.load_path), "pq pivot file"),
            (
                get_compressed_pq_file(&self.load_path),
                "compressed pq file",
            ),
            (get_disk_index_file(&self.load_path), "disk index file"),
        ];

        for (path_str, label) in files {
            let path = Path::new(&path_str);
            if !path.is_file() {
                anyhow::bail!("{} {} does not exist", label, path.display());
            }
        }

        Ok(())
    }
}

impl DiskIndexBuild {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker).context("invalid data file")?;

        // basic constraints
        if self.dim == 0 {
            anyhow::bail!("dim must be positive");
        }
        if self.max_degree == 0 {
            anyhow::bail!("max_degree must be positive");
        }
        if self.l_build == 0 {
            anyhow::bail!("l_build must be positive");
        }
        if self.num_threads == 0 {
            anyhow::bail!("num_threads must be positive");
        }
        if self.build_ram_limit_gb <= 0.0 {
            anyhow::bail!("build_ram_limit_gb must be strictly positive");
        }

        // Relative save path with respect to output directory is not supported.
        if checker.output_directory().is_some() {
            anyhow::bail!("relative save_path with respect to output_directory is not supported");
        }

        // We allow overwriting of already existing save paths, since users like to do this.
        // Only check if the parent directory exists.
        match Path::new(&self.save_path).parent() {
            Some(parent_dir) => {
                let parent_str = parent_dir.to_string_lossy();

                if !parent_str.is_empty() && !parent_dir.is_dir() {
                    anyhow::bail!(
                        "parent directory - {} of save_path - {} does not exist",
                        parent_str,
                        self.save_path
                    );
                }
            }
            None => {
                anyhow::bail!("invalid save_path - {}", self.save_path);
            }
        };

        Ok(())
    }
}

impl DiskSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries
            .resolve(checker)
            .context("invalid queries file")?;
        self.groundtruth
            .resolve(checker)
            .context("invalid groundtruth file")?;

        self.search_mode
            .validate(checker)
            .context("invalid disk search mode")?;

        // basic numeric sanity checks
        if self.search_list.is_empty() {
            anyhow::bail!("search_list must have at least one value");
        }
        if self
            .search_list
            .iter()
            .any(|&l| l == 0 || l < self.recall_at)
        {
            anyhow::bail!("search_list must contain positive values only");
        }
        if self.beam_width == 0 {
            anyhow::bail!("beam_width must be positive");
        }
        if self.recall_at == 0 {
            anyhow::bail!("recall_at must be positive");
        }
        if self.num_threads == 0 {
            anyhow::bail!("num_threads must be positive");
        }
        if let Some(n) = self.num_nodes_to_cache {
            if n == 0 {
                anyhow::bail!("num_nodes_to_cache must be positive if specified");
            }
        }
        if let Some(lim) = self.search_io_limit {
            if lim == 0 {
                anyhow::bail!("search_io_limit must be positive if specified");
            }
        }

        Ok(())
    }
}

/////////////
// Example //
/////////////

impl Example for DiskIndexOperation {
    fn example() -> Self {
        // a small, realistic example
        let build = DiskIndexBuild {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data.fbin"),
            distance: SimilarityMeasure::SquaredL2,
            dim: 128,
            max_degree: 32,
            l_build: 50,
            num_threads: 8,
            build_ram_limit_gb: 16.0,
            num_pq_chunks: NonZeroUsize::new(16).unwrap(),
            #[cfg(feature = "disk-index")]
            quantization_type: QuantizationType::PQ { num_chunks: 16 },
            save_path: "sample_index_l50_r32".to_string(),
        };

        let search = DiskSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            search_list: vec![64, 128, 256, 512],
            beam_width: 16,
            recall_at: 10,
            num_threads: 8,
            search_mode: DiskSearchMode::Graph {
                vector_filters_file: None,
            },
            distance: SimilarityMeasure::SquaredL2,
            num_nodes_to_cache: None,
            search_io_limit: None,
        };

        Self {
            source: DiskIndexSource::Build(build),
            search_phase: search,
        }
    }
}

/////////////
// Display //
/////////////

// This constant is used to ensure that summaries of disk-index jobs properly have
// their field descriptions aligned.
const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f,"{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

impl fmt::Display for DiskIndexSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiskIndexSource::Load(load) => load.fmt(f),
            DiskIndexSource::Build(build) => build.fmt(f),
        }
    }
}

impl DiskIndexLoad {
    fn summarize_fields(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Load Path", self.load_path)?;
        Ok(())
    }
}

impl fmt::Display for DiskIndexLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Disk Index Load")?;
        self.summarize_fields(f)
    }
}

impl DiskIndexBuild {
    fn summarize_fields(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Data File", self.data.display())?;
        write_field!(f, "Distance", self.distance)?;
        write_field!(f, "Dim", self.dim)?;
        write_field!(f, "Max Degree", self.max_degree)?;
        write_field!(f, "L Build", self.l_build)?;
        write_field!(f, "Build Threads", self.num_threads)?;
        write_field!(f, "Build RAM Limit GB", self.build_ram_limit_gb)?;
        write_field!(f, "PQ Chunks", self.num_pq_chunks)?;
        #[cfg(feature = "disk-index")]
        match &self.quantization_type {
            QuantizationType::FP => write_field!(f, "Quantization", "full precision")?,
            QuantizationType::PQ { num_chunks } => {
                write_field!(f, "Quantization", format!("pq, chunks {num_chunks}"))?
            }
            QuantizationType::SQ {
                nbits,
                standard_deviation,
            } => {
                if let Some(sd) = standard_deviation {
                    write_field!(
                        f,
                        "Quantization",
                        format!("sq, nbits {nbits}, stdev {}", sd.into_inner())
                    )?
                } else {
                    write_field!(f, "Quantization", format!("sq, nbits {nbits}"))?
                }
            }
        }
        write_field!(f, "Save Path", self.save_path)?;
        Ok(())
    }
}

impl fmt::Display for DiskIndexBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Disk Index Build")?;
        self.summarize_fields(f)
    }
}

impl DiskSearchPhase {
    fn summarize_fields(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_field!(f, "Queries", self.queries.display())?;
        write_field!(f, "Groundtruth", self.groundtruth.display())?;
        {
            // join search_list nicely
            let mut first = true;
            write!(f, "        Search List:")?;
            for v in &self.search_list {
                if !first {
                    write!(f, ",")?;
                }
                write!(f, "{}", v)?;
                first = false;
            }
            writeln!(f)?;
        }
        write_field!(f, "Beam Width", self.beam_width)?;
        write_field!(f, "Recall@", self.recall_at)?;
        write_field!(f, "Threads", self.num_threads)?;
        write_field!(f, "Search Mode", self.search_mode)?;
        write_field!(f, "Distance", self.distance)?;
        match self.search_mode.vector_filters_file() {
            Some(vf) => write_field!(f, "Vector Filters File", vf.display())?,
            None => write_field!(f, "Vector Filters File", "none")?,
        }
        match &self.num_nodes_to_cache {
            Some(n) => write_field!(f, "Num Nodes to Cache", n)?,
            None => write_field!(f, "Num Nodes to Cache", "none (defaults to 0)")?,
        }
        match &self.search_io_limit {
            Some(lim) => write_field!(f, "Search IO Limit", format!("{lim}"))?,
            None => write_field!(f, "Search IO Limit", "none (defaults to `usize::MAX`)")?,
        }
        match self.search_mode.post_processor() {
            Some(pp) => write_field!(f, "Post Processor", pp)?,
            None => write_field!(f, "Post Processor", "none")?,
        }
        Ok(())
    }
}

impl fmt::Display for DiskSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Disk Index Search Phase")?;
        self.summarize_fields(f)
    }
}
