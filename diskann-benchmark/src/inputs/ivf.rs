/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    num::{NonZeroU32, NonZeroUsize},
    path::Path,
};

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

as_input!(IvfOperation);

///////////
// Input //
///////////

/// Supported quantization bit widths for MinMax quantization.
///
/// MinMax quantization compresses each vector independently using per-vector min/max
/// scaling, enabling streaming compression without training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum QuantizationBits {
    /// 1-bit quantization (highest compression, lowest fidelity).
    #[serde(rename = "1")]
    One,
    /// 4-bit quantization (good balance of compression and fidelity).
    #[serde(rename = "4")]
    Four,
    /// 8-bit quantization (lowest compression, highest fidelity).
    #[serde(rename = "8")]
    Eight,
}

impl QuantizationBits {
    pub(crate) fn as_usize(self) -> usize {
        match self {
            Self::One => 1,
            Self::Four => 4,
            Self::Eight => 8,
        }
    }

    pub(crate) fn as_u8(self) -> u8 {
        self.as_usize() as u8
    }
}

impl fmt::Display for QuantizationBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_usize())
    }
}

/// MinMax quantization configuration for IVF cluster vectors.
///
/// When enabled, cluster vectors are stored as quantized codes instead of full-precision
/// f32 values, reducing the on-disk size and I/O during search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct QuantizationConfig {
    /// Number of bits per dimension.
    pub(crate) nbits: QuantizationBits,
    /// Scaling parameter for the MinMax quantizer range. Values in `[0.8, 1.0]` work well.
    /// Must be positive.
    pub(crate) grid_scale: f32,
}

/// Reranking configuration for IVF search with quantized vectors.
///
/// When enabled, the search first collects `search_l` candidates using quantized distances,
/// then reranks them using full-precision vectors loaded from `vectors.bin`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct RerankConfig {
    /// Number of candidates to collect before reranking. Must be ≥ `recall_at`.
    pub(crate) search_l: NonZeroU32,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IvfOperation {
    pub(crate) source: IvfSource,
    pub(crate) search_phase: IvfSearchPhase,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "ivf-source")]
pub(crate) enum IvfSource {
    Load(IvfLoad),
    Build(IvfBuild),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct IvfLoad {
    pub(crate) data_type: DataType,
    pub(crate) load_path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct IvfBuild {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) nlist: NonZeroU32,
    pub(crate) num_threads: NonZeroUsize,
    pub(crate) kmeans_iterations: NonZeroU32,
    pub(crate) save_path: String,
    /// Optional MinMax quantization of cluster vectors.
    /// When set, cluster files store quantized codes instead of f32 vectors,
    /// and a `vectors.bin` blob is written for optional reranking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) quantization: Option<QuantizationConfig>,
}

/// Search phase configuration for IVF.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct IvfSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: NonZeroUsize,
    /// List of nprobe values to sweep. Each becomes a row in the output table, mapped to
    /// `search_l` in the result struct for comparability with DiskANN.
    pub(crate) nprobe_list: Vec<NonZeroU32>,
    pub(crate) recall_at: NonZeroU32,
    pub(crate) distance: SimilarityMeasure,
    /// Optional reranking with full-precision vectors.
    /// Requires quantization to be enabled in the build phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) rerank: Option<RerankConfig>,
}

/////////
// Tag //
/////////

impl IvfOperation {
    pub(crate) const fn tag() -> &'static str {
        "ivf"
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match &mut self.source {
            IvfSource::Load(load) => load.validate()?,
            IvfSource::Build(build) => build.validate(checker)?,
        }
        self.search_phase.validate(checker)?;
        Ok(())
    }
}

impl IvfLoad {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        let dir = Path::new(&self.load_path);

        let meta_path = dir.join("ivf_meta.bin");
        let centroids_path = dir.join("ivf_centroids.bin");
        let clusters_dir = dir.join("clusters");

        for (path, label) in [
            (meta_path.as_path(), "IVF meta file"),
            (centroids_path.as_path(), "IVF centroids file"),
        ] {
            if !path.is_file() {
                anyhow::bail!("{} {} does not exist", label, path.display());
            }
        }

        if !clusters_dir.is_dir() {
            anyhow::bail!(
                "IVF clusters directory {} does not exist",
                clusters_dir.display()
            );
        }

        // vectors.bin is optional — only present when index was built with quantization
        Ok(())
    }
}

impl IvfBuild {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker).context("invalid data file")?;

        // Relative save path with respect to output directory is not supported.
        if checker.output_directory().is_some() {
            anyhow::bail!("relative save_path with respect to output_directory is not supported");
        }

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
        }

        if let Some(ref qconfig) = self.quantization {
            anyhow::ensure!(
                qconfig.grid_scale > 0.0,
                "quantization grid_scale must be positive, got {}",
                qconfig.grid_scale
            );
        }

        Ok(())
    }
}

impl IvfSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries
            .resolve(checker)
            .context("invalid queries file")?;
        self.groundtruth
            .resolve(checker)
            .context("invalid groundtruth file")?;

        if self.nprobe_list.is_empty() {
            anyhow::bail!("nprobe_list must have at least one value");
        }

        if let Some(ref rerank) = self.rerank {
            anyhow::ensure!(
                rerank.search_l.get() >= self.recall_at.get(),
                "rerank search_l ({}) must be >= recall_at ({})",
                rerank.search_l.get(),
                self.recall_at.get()
            );
        }

        Ok(())
    }
}

/////////////
// Example //
/////////////

impl Example for IvfOperation {
    fn example() -> Self {
        let build = IvfBuild {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data.fbin"),
            distance: SimilarityMeasure::SquaredL2,
            nlist: NonZeroU32::new(16).unwrap(),
            num_threads: NonZeroUsize::new(1).unwrap(),
            kmeans_iterations: NonZeroU32::new(20).unwrap(),
            save_path: "sample_ivf_index".to_string(),
            quantization: Some(QuantizationConfig {
                nbits: QuantizationBits::Four,
                grid_scale: 0.9,
            }),
        };

        let search = IvfSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            nprobe_list: vec![
                NonZeroU32::new(1).unwrap(),
                NonZeroU32::new(4).unwrap(),
                NonZeroU32::new(8).unwrap(),
                NonZeroU32::new(16).unwrap(),
            ],
            recall_at: NonZeroU32::new(10).unwrap(),
            num_threads: NonZeroUsize::new(1).unwrap(),
            distance: SimilarityMeasure::SquaredL2,
            rerank: Some(RerankConfig {
                search_l: NonZeroU32::new(100).unwrap(),
            }),
        };

        Self {
            source: IvfSource::Build(build),
            search_phase: search,
        }
    }
}

/////////////
// Display //
/////////////

impl fmt::Display for IvfSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IvfSource::Load(load) => load.fmt(f),
            IvfSource::Build(build) => build.fmt(f),
        }
    }
}

impl fmt::Display for IvfLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IVF Index Load")?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Load Path", self.load_path)?;
        Ok(())
    }
}

impl fmt::Display for IvfBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IVF Index Build")?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Data File", self.data.display())?;
        write_field!(f, "Distance", self.distance)?;
        write_field!(f, "Nlist", self.nlist)?;
        write_field!(f, "K-Means Iters", self.kmeans_iterations)?;
        write_field!(f, "Build Threads", self.num_threads)?;
        write_field!(f, "Save Path", self.save_path)?;
        match &self.quantization {
            Some(q) => {
                write_field!(f, "Quantization", format!("MinMax-{}", q.nbits))?;
                write_field!(f, "Grid Scale", q.grid_scale)?;
            }
            None => {
                write_field!(f, "Quantization", "None (full precision)")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for IvfSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "IVF Search Phase")?;
        write_field!(f, "Queries", self.queries.display())?;
        write_field!(f, "Groundtruth", self.groundtruth.display())?;
        {
            let mut first = true;
            write!(f, "       Nprobe List:")?;
            for v in &self.nprobe_list {
                if !first {
                    write!(f, ",")?;
                }
                write!(f, "{}", v)?;
                first = false;
            }
            writeln!(f)?;
        }
        write_field!(f, "Recall@", self.recall_at)?;
        write_field!(f, "Threads", self.num_threads)?;
        write_field!(f, "Distance", self.distance)?;
        match &self.rerank {
            Some(r) => {
                write_field!(f, "Rerank Search L", r.search_l)?;
            }
            None => {
                write_field!(f, "Reranking", "disabled")?;
            }
        }
        Ok(())
    }
}
