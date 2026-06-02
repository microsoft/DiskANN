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
    pub(crate) nlist: u32,
    pub(crate) num_threads: usize,
    pub(crate) kmeans_iterations: u32,
    pub(crate) save_path: String,
}

/// Search phase configuration for IVF.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct IvfSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: usize,
    /// List of nprobe values to sweep. Each becomes a row in the output table, mapped to
    /// `search_l` in the result struct for comparability with DiskANN.
    pub(crate) nprobe_list: Vec<u32>,
    pub(crate) recall_at: u32,
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
        let meta_path = format!("{}/ivf_meta.bin", self.load_path);
        let centroids_path = format!("{}/ivf_centroids.bin", self.load_path);
        let invlists_path = format!("{}/ivf_invlists.bin", self.load_path);

        for (path_str, label) in [
            (&meta_path, "IVF meta file"),
            (&centroids_path, "IVF centroids file"),
            (&invlists_path, "IVF inverted lists file"),
        ] {
            let path = Path::new(path_str);
            if !path.is_file() {
                anyhow::bail!("{} {} does not exist", label, path.display());
            }
        }

        Ok(())
    }
}

impl IvfBuild {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker).context("invalid data file")?;

        if self.nlist == 0 {
            anyhow::bail!("nlist must be positive");
        }
        if self.num_threads == 0 {
            anyhow::bail!("num_threads must be positive");
        }
        if self.kmeans_iterations == 0 {
            anyhow::bail!("kmeans_iterations must be positive");
        }

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
        if self.nprobe_list.contains(&0) {
            anyhow::bail!("nprobe_list values must be positive");
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

impl Example for IvfOperation {
    fn example() -> Self {
        let build = IvfBuild {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data.fbin"),
            distance: SimilarityMeasure::SquaredL2,
            nlist: 16,
            num_threads: 1,
            kmeans_iterations: 20,
            save_path: "sample_ivf_index".to_string(),
        };

        let search = IvfSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            nprobe_list: vec![1, 4, 8, 16],
            recall_at: 10,
            num_threads: 1,
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

const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f,"{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

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
