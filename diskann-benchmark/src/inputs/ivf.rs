/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, num::{NonZeroU32, NonZeroUsize}, path::Path};

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
        Ok(())
    }
}
