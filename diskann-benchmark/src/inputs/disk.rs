/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, num::NonZeroUsize, path::Path};

use anyhow::Context;
#[cfg(feature = "disk-index")]
use std::collections::HashSet;

#[cfg(feature = "disk-index")]
use diskann::graph;
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
#[cfg(feature = "disk-index")]
use diskann_disk::search::search_mode::SearchMode;
#[cfg(feature = "disk-index")]
use diskann_disk::QuantizationType;
use diskann_providers::storage::{get_compressed_pq_file, get_disk_index_file, get_pq_pivot_file};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, post_processor::TopkPostProcessor, Example},
    utils::SimilarityMeasure,
};

#[cfg(feature = "disk-index")]
use crate::inputs::graph_index::AdaptiveL;

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
    #[cfg(feature = "disk-index")]
    #[serde(default)]
    pub(crate) ivf_pq_router_build: Option<IvfPqRouterBuildConfig>,
}

#[cfg(feature = "disk-index")]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct IvfPqRouterBuildConfig {
    pub(crate) save_path: String,
    pub(crate) num_centroids: NonZeroUsize,
    pub(crate) max_iterations: NonZeroUsize,
    pub(crate) seed: u64,
    pub(crate) training_sample_size: Option<NonZeroUsize>,
}

#[cfg(feature = "disk-index")]
#[derive(Debug, Serialize, Deserialize, Default)]
pub(crate) struct DiskSearchMode {
    pub(crate) is_flat_search: bool,
    #[serde(default)]
    pub(crate) adaptive_l: Option<AdaptiveL>,
}

#[cfg(feature = "disk-index")]
impl DiskSearchMode {
    pub(crate) fn search_mode<'a>(
        &'a self,
        has_vector_filters: bool,
        vector_filter: &'a HashSet<u32>,
        post_processor: Option<&TopkPostProcessor>,
    ) -> SearchMode<'a> {
        let adaptive_l = self.adaptive_l.as_ref().map(|adaptive_l| {
            graph::search::AdaptiveL::new(adaptive_l.sample_count.into(), adaptive_l.scale_factor)
                .expect("validated adaptive L must construct")
        });

        match (
            self.is_flat_search,
            has_vector_filters,
            post_processor,
            adaptive_l,
        ) {
            (true, false, _, _) => SearchMode::flat(),
            (true, true, _, _) => {
                SearchMode::flat_filtered(move |vid: &u32| vector_filter.contains(vid))
            }
            (false, false, Some(TopkPostProcessor::DeterminantDiversity(params)), _) => {
                SearchMode::diverse_graph(*params)
            }
            (false, true, Some(TopkPostProcessor::DeterminantDiversity(params)), _) => {
                SearchMode::diverse_graph_filtered(
                    move |vid: &u32| vector_filter.contains(vid),
                    *params,
                )
            }
            (false, false, None, Some(adaptive_l)) => {
                SearchMode::inline_filter(|_| true, Some(adaptive_l))
            }
            (false, true, None, Some(adaptive_l)) => SearchMode::inline_filter(
                move |vid: &u32| vector_filter.contains(vid),
                Some(adaptive_l),
            ),
            (false, false, None, None) => SearchMode::graph(),
            (false, true, None, None) => {
                SearchMode::graph_filtered(move |vid: &u32| vector_filter.contains(vid))
            }
        }
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        if let Some(adaptive_l) = self.adaptive_l.as_mut() {
            adaptive_l.validate(checker)?;
        }
        Ok(())
    }
}

#[cfg(feature = "disk-index")]
impl fmt::Display for DiskSearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let base = if self.is_flat_search { "flat" } else { "graph" };
        if self.adaptive_l.is_some() {
            write!(f, "{} + adaptive-l", base)
        } else {
            write!(f, "{}", base)
        }
    }
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
    #[cfg(feature = "disk-index")]
    #[serde(default)]
    pub(crate) search_mode: DiskSearchMode,
    // Backward compatibility for older benchmark inputs that used
    // `is_flat_search` directly at the search-phase level.
    #[cfg(feature = "disk-index")]
    #[serde(default, skip_serializing)]
    pub(crate) is_flat_search: Option<bool>,
    #[cfg(not(feature = "disk-index"))]
    pub(crate) is_flat_search: bool,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) vector_filters_file: Option<InputFile>,
    pub(crate) num_nodes_to_cache: Option<usize>,
    pub(crate) search_io_limit: Option<usize>,
    pub(crate) post_processor: Option<TopkPostProcessor>,
    #[cfg(feature = "disk-index")]
    #[serde(default)]
    pub(crate) start_point_router: Option<DiskStartPointRouter>,
}

#[cfg(feature = "disk-index")]
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub(crate) enum DiskStartPointRouter {
    #[serde(rename = "ivf_pq")]
    IvfPq(IvfPqStartPointRouterConfig),
}

#[cfg(feature = "disk-index")]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct IvfPqStartPointRouterConfig {
    pub(crate) artifact: InputFile,
    pub(crate) nprobe: NonZeroUsize,
    pub(crate) max_start_points: NonZeroUsize,
    pub(crate) posting_list_samples_per_list: NonZeroUsize,
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

        #[cfg(feature = "disk-index")]
        if let Some(ivf_pq_router_build) = &self.ivf_pq_router_build {
            ivf_pq_router_build.validate()?;
        }

        Ok(())
    }
}

#[cfg(feature = "disk-index")]
impl IvfPqRouterBuildConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        match Path::new(&self.save_path).parent() {
            Some(parent_dir) => {
                let parent_str = parent_dir.to_string_lossy();
                if !parent_str.is_empty() && !parent_dir.is_dir() {
                    anyhow::bail!(
                        "parent directory - {} of ivf_pq_router_build.save_path - {} does not exist",
                        parent_str,
                        self.save_path
                    );
                }
            }
            None => {
                anyhow::bail!("invalid ivf_pq_router_build.save_path - {}", self.save_path);
            }
        }
        if let Some(training_sample_size) = self.training_sample_size {
            if training_sample_size < self.num_centroids {
                anyhow::bail!(
                    "ivf_pq_router_build.training_sample_size must be at least num_centroids"
                );
            }
        }
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
        if let Some(vf) = self.vector_filters_file.as_mut() {
            vf.resolve(checker).context("invalid vector_filters_file")?;
        }

        #[cfg(feature = "disk-index")]
        if let Some(is_flat_search) = self.is_flat_search {
            self.search_mode.is_flat_search = is_flat_search;
        }

        #[cfg(feature = "disk-index")]
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

        if let Some(pp) = self.post_processor.as_mut() {
            pp.validate(checker)
                .context("invalid disk search post processor")?;
        }

        #[cfg(feature = "disk-index")]
        if let Some(router) = self.start_point_router.as_mut() {
            router
                .validate(checker)
                .context("invalid disk start-point router")?;
        }

        Ok(())
    }
}

#[cfg(feature = "disk-index")]
impl DiskStartPointRouter {
    fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        match self {
            DiskStartPointRouter::IvfPq(config) => config.validate(checker),
        }
    }
}

#[cfg(feature = "disk-index")]
impl IvfPqStartPointRouterConfig {
    fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        // Resolve existing artifacts for load-only jobs, but do not fail here:
        // build-and-search jobs can produce this file earlier in the same
        // benchmark operation. The search loader reports a missing artifact if
        // it still cannot be opened at execution time.
        if let Ok(resolved) = checker.find_input_file(&self.artifact) {
            self.artifact = InputFile::new(resolved);
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
            #[cfg(feature = "disk-index")]
            ivf_pq_router_build: None,
        };

        let search = DiskSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            search_list: vec![64, 128, 256, 512],
            beam_width: 16,
            recall_at: 10,
            num_threads: 8,
            #[cfg(feature = "disk-index")]
            search_mode: DiskSearchMode {
                is_flat_search: false,
                adaptive_l: None,
            },
            #[cfg(feature = "disk-index")]
            is_flat_search: None,
            #[cfg(not(feature = "disk-index"))]
            is_flat_search: false,
            distance: SimilarityMeasure::SquaredL2,
            vector_filters_file: None,
            num_nodes_to_cache: None,
            search_io_limit: None,
            post_processor: None,
            #[cfg(feature = "disk-index")]
            start_point_router: None,
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
        #[cfg(feature = "disk-index")]
        match &self.ivf_pq_router_build {
            Some(config) => write_field!(f, "IVF PQ Build", config)?,
            None => write_field!(f, "IVF PQ Build", "none")?,
        }
        Ok(())
    }
}

#[cfg(feature = "disk-index")]
impl fmt::Display for IvfPqRouterBuildConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "save_path={} num_centroids={} max_iterations={} seed={} training_sample_size={}",
            self.save_path,
            self.num_centroids,
            self.max_iterations,
            self.seed,
            self.training_sample_size
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        )
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
        #[cfg(feature = "disk-index")]
        write_field!(f, "Search Mode", self.search_mode)?;
        #[cfg(not(feature = "disk-index"))]
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
        match &self.post_processor {
            Some(pp) => write_field!(f, "Post Processor", pp)?,
            None => write_field!(f, "Post Processor", "none")?,
        }
        #[cfg(feature = "disk-index")]
        match &self.start_point_router {
            Some(router) => write_field!(f, "Start Router", router)?,
            None => write_field!(f, "Start Router", "none")?,
        }
        Ok(())
    }
}

#[cfg(feature = "disk-index")]
impl fmt::Display for DiskStartPointRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IvfPq(config) => write!(
                f,
                "ivf_pq artifact={} nprobe={} max_start_points={} posting_list_samples_per_list={}",
                config.artifact.display(),
                config.nprobe,
                config.max_start_points,
                config.posting_list_samples_per_list
            ),
        }
    }
}

impl fmt::Display for DiskSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Disk Index Search Phase")?;
        self.summarize_fields(f)
    }
}

#[cfg(all(test, feature = "disk-index"))]
mod tests {
    use super::*;

    #[test]
    fn deserializes_ivf_pq_router_build_config() {
        let raw = r#"
        {
          "save_path": "index.ivf_pq_router.bin",
          "num_centroids": 2048,
          "max_iterations": 4,
          "seed": 42,
          "training_sample_size": 100000
        }
        "#;

        let config: IvfPqRouterBuildConfig = serde_json::from_str(raw).unwrap();

        assert_eq!(config.save_path, "index.ivf_pq_router.bin");
        assert_eq!(config.num_centroids.get(), 2048);
        assert_eq!(config.max_iterations.get(), 4);
        assert_eq!(config.seed, 42);
        assert_eq!(config.training_sample_size.unwrap().get(), 100000);
    }

    #[test]
    fn deserializes_ivf_pq_sampled_adc_start_point_router_config() {
        let raw = r#"
        {
          "source": {
            "disk-index-source": "Load",
            "data_type": "float32",
            "load_path": "test-index"
          },
          "search_phase": {
            "queries": "queries.fbin",
            "groundtruth": "truth.bin",
            "search_list": [200],
            "beam_width": 64,
            "recall_at": 100,
            "num_threads": 8,
            "is_flat_search": false,
            "distance": "squared_l2",
            "vector_filters_file": null,
            "num_nodes_to_cache": null,
            "search_io_limit": null,
            "post_processor": null,
            "start_point_router": {
              "type": "ivf_pq",
              "artifact": "router.ivf_pq_router.bin",
              "nprobe": 8,
              "max_start_points": 16,
              "posting_list_samples_per_list": 2048
            }
          }
        }
        "#;

        let input: DiskIndexOperation = serde_json::from_str(raw).unwrap();
        let router = input.search_phase.start_point_router.unwrap();

        match router {
            DiskStartPointRouter::IvfPq(config) => {
                assert_eq!(
                    config.artifact.display().to_string(),
                    "router.ivf_pq_router.bin"
                );
                assert_eq!(config.nprobe.get(), 8);
                assert_eq!(config.max_start_points.get(), 16);
                assert_eq!(config.posting_list_samples_per_list.get(), 2048);
            }
        }
    }
}
