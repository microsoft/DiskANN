/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, num::NonZeroUsize, path::Path};

use anyhow::Context;
use diskann_benchmark_runner::{
    files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker,
};
#[cfg(feature = "disk-index")]
use diskann_disk::QuantizationType;
use diskann_providers::storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, Example, Input},
    utils::SimilarityMeasure,
};

//////////////
// Registry //
//////////////

as_input!(DiskIndexOperation);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<DiskIndexOperation>::new())?;
    Ok(())
}

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

/// Search algorithm to use for disk index search.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(tag = "mode")]
pub(crate) enum SearchMode {
    /// Standard beam search (default, current behavior).
    #[default]
    BeamSearch,
    /// PipeANN pipelined search with IO/compute overlap.
    PipeSearch {
        /// Initial beam width before adaptive adjustment (default: 4).
        #[serde(default = "default_initial_beam_width")]
        initial_beam_width: usize,
        /// Optional relaxed monotonicity parameter for early termination.
        relaxed_monotonicity_l: Option<usize>,
        /// Enable kernel-side SQ polling (ms idle timeout). None = disabled.
        #[serde(default)]
        sqpoll_idle_ms: Option<u32>,
    },
    /// Unified pipelined search through the generic search loop (queue-based ExpandBeam).
    UnifiedPipeSearch {
        /// Enable kernel-side SQ polling (ms idle timeout). None = disabled.
        #[serde(default)]
        sqpoll_idle_ms: Option<u32>,
    },
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::BeamSearch => write!(f, "BeamSearch"),
            SearchMode::PipeSearch {
                initial_beam_width,
                relaxed_monotonicity_l,
                sqpoll_idle_ms,
            } => {
                write!(f, "PipeSearch(bw={}", initial_beam_width)?;
                if let Some(rm) = relaxed_monotonicity_l {
                    write!(f, ",rm={}", rm)?;
                }
                if let Some(sq) = sqpoll_idle_ms {
                    write!(f, ",sqpoll={}ms", sq)?;
                }
                write!(f, ")")
            }
            SearchMode::UnifiedPipeSearch { sqpoll_idle_ms } => {
                write!(f, "UnifiedPipeSearch")?;
                if let Some(sq) = sqpoll_idle_ms {
                    write!(f, "(sqpoll={}ms)", sq)?;
                }
                Ok(())
            }
        }
    }
}

fn default_initial_beam_width() -> usize {
    4
}

/// Search phase configuration
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct DiskSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: usize,
    pub(crate) beam_width: usize,
    pub(crate) search_list: Vec<u32>,
    pub(crate) recall_at: u32,
    pub(crate) is_flat_search: bool,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) vector_filters_file: Option<InputFile>,
    pub(crate) num_nodes_to_cache: Option<usize>,
    pub(crate) search_io_limit: Option<usize>,
    /// Search algorithm to use (defaults to BeamSearch).
    #[serde(default)]
    pub(crate) search_mode: SearchMode,
}

/////////
// Tag //
/////////

impl DiskIndexOperation {
    pub(crate) const fn tag() -> &'static str {
        "disk-index"
    }
}

///////////////////////////
// Check Deserialization //
///////////////////////////

impl CheckDeserialization for DiskIndexOperation {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // validate the source
        match &mut self.source {
            DiskIndexSource::Load(load) => load.check_deserialization(checker)?,
            DiskIndexSource::Build(build) => build.check_deserialization(checker)?,
        }

        // validate the search phase
        self.search_phase.check_deserialization(checker)?;

        Ok(())
    }
}

impl CheckDeserialization for DiskIndexLoad {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
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

impl CheckDeserialization for DiskIndexBuild {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // file input
        self.data
            .check_deserialization(checker)
            .context("invalid data file")?;

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

impl CheckDeserialization for DiskSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // inputs
        self.queries
            .check_deserialization(checker)
            .context("invalid queries file")?;
        self.groundtruth
            .check_deserialization(checker)
            .context("invalid groundtruth file")?;
        if let Some(vf) = self.vector_filters_file.as_mut() {
            vf.check_deserialization(checker)
                .context("invalid vector_filters_file")?;
        }

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
        match &self.search_mode {
            SearchMode::BeamSearch => {}
            SearchMode::PipeSearch { initial_beam_width, .. } => {
                if *initial_beam_width == 0 {
                    anyhow::bail!("initial_beam_width must be positive");
                }
            }
            SearchMode::UnifiedPipeSearch { .. } => {}
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
            is_flat_search: false,
            distance: SimilarityMeasure::SquaredL2,
            vector_filters_file: None,
            num_nodes_to_cache: None,
            search_io_limit: None,
            search_mode: SearchMode::default(),
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
        write_field!(f, "Flat Search", self.is_flat_search)?;
        write_field!(f, "Distance", self.distance)?;
        match &self.vector_filters_file {
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
        write_field!(f, "Search Mode", format!("{:?}", self.search_mode))?;
        Ok(())
    }
}

impl fmt::Display for DiskSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Disk Index Search Phase")?;
        self.summarize_fields(f)
    }
}
