/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, path::Path};

use anyhow::Context;
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{as_input, Example},
    utils::SimilarityMeasure,
};

//////////////
// Registry //
//////////////

as_input!(GraphIvfOperation);

///////////
// Input //
///////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GraphIvfOperation {
    pub(crate) source: GraphIvfSource, // either load or build
    pub(crate) search_phase: GraphIvfSearchPhase,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "graph-ivf-source")] // Use tagged enums for JSON
pub(crate) enum GraphIvfSource {
    Load(GraphIvfLoad),
    Build(GraphIvfBuild),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct GraphIvfLoad {
    pub(crate) data_type: DataType,
    /// Path prefix the index was saved under (without the `.graphivf_*` suffix).
    pub(crate) load_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct GraphIvfBuild {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) dim: usize,
    /// Number of clusters / centroids (`k`).
    pub(crate) num_clusters: usize,
    /// Number of corpus points to sample for k-means training.
    pub(crate) sample_size: usize,
    /// Number of Lloyd's iterations for k-means.
    pub(crate) kmeans_iters: usize,
    /// Search-list size used when assigning corpus points to centroids.
    pub(crate) assign_l: usize,
    /// Pruned out-degree of the centroid graph (`R`).
    pub(crate) graph_degree: usize,
    /// Maximum out-degree as a multiple of `graph_degree` (slack, `>= 1.0`).
    pub(crate) graph_slack: f32,
    /// Search-list size used during centroid-graph construction (`L`).
    pub(crate) graph_l_build: usize,
    /// Pruning alpha (`>= 1.0`).
    pub(crate) graph_alpha: f32,
    pub(crate) num_threads: usize,
    /// RNG seed for sampling and k-means (for reproducibility).
    pub(crate) seed: u64,
    /// Path prefix to save the index under (without the `.graphivf_*` suffix).
    pub(crate) save_path: String,
}

/// Search phase configuration.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct GraphIvfSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: usize,
    /// Numbers of nearest clusters to probe — one search sweep per value.
    pub(crate) nlist: Vec<usize>,
    /// Search-list size for the centroid graph search (`>= nlist`).
    pub(crate) centroid_search_l: usize,
    pub(crate) recall_at: u32,
    pub(crate) distance: SimilarityMeasure,
}

/////////
// Tag //
/////////

impl GraphIvfOperation {
    pub(crate) const fn tag() -> &'static str {
        "graph-ivf"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match &mut self.source {
            GraphIvfSource::Load(load) => load.validate(checker)?,
            GraphIvfSource::Build(build) => build.validate(checker)?,
        }
        self.search_phase.validate(checker)?;
        Ok(())
    }
}

impl GraphIvfLoad {
    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        let meta = format!("{}.graphivf_meta", self.load_path);
        if !Path::new(&meta).is_file() {
            anyhow::bail!("index metadata file {} does not exist", meta);
        }
        Ok(())
    }
}

impl GraphIvfBuild {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker).context("invalid data file")?;

        if self.dim == 0 {
            anyhow::bail!("dim must be positive");
        }
        if self.num_clusters == 0 {
            anyhow::bail!("num_clusters must be positive");
        }
        if self.sample_size < self.num_clusters {
            anyhow::bail!("sample_size must be >= num_clusters");
        }
        if self.assign_l == 0 {
            anyhow::bail!("assign_l must be positive");
        }
        if self.graph_degree == 0 {
            anyhow::bail!("graph_degree must be positive");
        }
        if self.graph_l_build == 0 {
            anyhow::bail!("graph_l_build must be positive");
        }
        if self.num_threads == 0 {
            anyhow::bail!("num_threads must be positive");
        }

        // Relative save path with respect to output directory is not supported.
        if checker.output_directory().is_some() {
            anyhow::bail!("relative save_path with respect to output_directory is not supported");
        }

        // Only check that the parent directory of the save prefix exists; overwriting is allowed.
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
            None => anyhow::bail!("invalid save_path - {}", self.save_path),
        };

        Ok(())
    }
}

impl GraphIvfSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries
            .resolve(checker)
            .context("invalid queries file")?;
        self.groundtruth
            .resolve(checker)
            .context("invalid groundtruth file")?;

        if self.nlist.is_empty() {
            anyhow::bail!("nlist must have at least one value");
        }
        if self.nlist.contains(&0) {
            anyhow::bail!("nlist values must be positive");
        }
        if self.centroid_search_l == 0 {
            anyhow::bail!("centroid_search_l must be positive");
        }
        if self.recall_at == 0 {
            anyhow::bail!("recall_at must be positive");
        }
        if self.num_threads == 0 {
            anyhow::bail!("num_threads must be positive");
        }
        Ok(())
    }
}

/////////////
// Example //
/////////////

impl Example for GraphIvfOperation {
    fn example() -> Self {
        let build = GraphIvfBuild {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data.fbin"),
            distance: SimilarityMeasure::SquaredL2,
            dim: 128,
            num_clusters: 1024,
            sample_size: 65536,
            kmeans_iters: 10,
            assign_l: 32,
            graph_degree: 32,
            graph_slack: 1.2,
            graph_l_build: 64,
            graph_alpha: 1.2,
            num_threads: 8,
            seed: 0,
            save_path: "sample_graphivf_index".to_string(),
        };

        let search = GraphIvfSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            num_threads: 8,
            nlist: vec![8, 16, 32],
            centroid_search_l: 64,
            recall_at: 10,
            distance: SimilarityMeasure::SquaredL2,
        };

        Self {
            source: GraphIvfSource::Build(build),
            search_phase: search,
        }
    }
}

/////////////
// Display //
/////////////

const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f,"{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

impl fmt::Display for GraphIvfSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphIvfSource::Load(load) => load.fmt(f),
            GraphIvfSource::Build(build) => build.fmt(f),
        }
    }
}

impl fmt::Display for GraphIvfLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Load")?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Load Path", self.load_path)?;
        Ok(())
    }
}

impl fmt::Display for GraphIvfBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Build")?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Data File", self.data.display())?;
        write_field!(f, "Distance", self.distance)?;
        write_field!(f, "Dim", self.dim)?;
        write_field!(f, "Num Clusters", self.num_clusters)?;
        write_field!(f, "Sample Size", self.sample_size)?;
        write_field!(f, "KMeans Iters", self.kmeans_iters)?;
        write_field!(f, "Assign L", self.assign_l)?;
        write_field!(f, "Graph Degree", self.graph_degree)?;
        write_field!(f, "Graph Slack", self.graph_slack)?;
        write_field!(f, "Graph L Build", self.graph_l_build)?;
        write_field!(f, "Graph Alpha", self.graph_alpha)?;
        write_field!(f, "Build Threads", self.num_threads)?;
        write_field!(f, "Seed", self.seed)?;
        write_field!(f, "Save Path", self.save_path)?;
        Ok(())
    }
}

impl fmt::Display for GraphIvfSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Search Phase")?;
        write_field!(f, "Queries", self.queries.display())?;
        write_field!(f, "Groundtruth", self.groundtruth.display())?;
        {
            let mut first = true;
            write!(f, "{:>PRINT_WIDTH$}: ", "NList")?;
            for v in &self.nlist {
                if !first {
                    write!(f, ",")?;
                }
                write!(f, "{}", v)?;
                first = false;
            }
            writeln!(f)?;
        }
        write_field!(f, "Centroid L", self.centroid_search_l)?;
        write_field!(f, "Recall@", self.recall_at)?;
        write_field!(f, "Threads", self.num_threads)?;
        write_field!(f, "Distance", self.distance)?;
        Ok(())
    }
}
