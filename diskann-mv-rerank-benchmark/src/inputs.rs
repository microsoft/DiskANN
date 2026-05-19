/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Input types for the multi-vector rerank benchmark.
//!
//! The JSON shape mirrors `graph-index-build` from `diskann-benchmark` so anyone
//! familiar with that benchmark can predict what works here. The additions are
//! [`MultiVectorRerankOperation::doc_mv`] / `query_mv`, which point at the
//! multi-vector `.mvbin` files the reranker scores against — see
//! [`crate::datafiles`] for the format.

use std::num::{NonZero, NonZeroUsize};

use anyhow::{anyhow, Context};
use diskann::graph::{config, StartPointStrategy};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    files::InputFile,
    utils::datatype::{AsDataType, DataType},
    Any, CheckDeserialization, Checker,
};
use diskann_providers::model::{
    configuration::IndexConfiguration,
    graph::provider::async_::inmem::DefaultProviderParameters,
};
use serde::{Deserialize, Serialize};

///////////////////////
// Similarity Metric //
///////////////////////

/// Locally-redeclared (de)serializable analogue of [`diskann_vector::distance::Metric`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SimilarityMeasure {
    SquaredL2,
    InnerProduct,
    Cosine,
    CosineNormalized,
}

impl From<SimilarityMeasure> for diskann_vector::distance::Metric {
    fn from(value: SimilarityMeasure) -> Self {
        match value {
            SimilarityMeasure::SquaredL2 => Self::L2,
            SimilarityMeasure::InnerProduct => Self::InnerProduct,
            SimilarityMeasure::Cosine => Self::Cosine,
            SimilarityMeasure::CosineNormalized => Self::CosineNormalized,
        }
    }
}

impl std::fmt::Display for SimilarityMeasure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SquaredL2 => "squared_l2",
            Self::InnerProduct => "inner_product",
            Self::Cosine => "cosine",
            Self::CosineNormalized => "cosine_normalized",
        };
        f.write_str(s)
    }
}

const DATA_TYPE_MISMATCH: FailureScore = FailureScore(1000);

pub(crate) fn match_data_type<T>(data_type: DataType) -> Result<MatchScore, FailureScore>
where
    T: AsDataType,
{
    if T::is_match(data_type) {
        Ok(MatchScore(0))
    } else {
        Err(DATA_TYPE_MISMATCH)
    }
}

////////////////////////////
// StartPointStrategy ref //
////////////////////////////

/// `#[serde(remote)]` shadow for `diskann::graph::StartPointStrategy`, which does not
/// derive `Serialize`/`Deserialize` upstream. Variants stay in lockstep with the real
/// enum; the shadow is wired up at each use site via `#[serde(with = "StartPointStrategyRef")]`.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(remote = "StartPointStrategy")]
#[serde(rename_all = "snake_case")]
pub(crate) enum StartPointStrategyRef {
    RandomVectors {
        norm: f32,
        nsamples: NonZeroUsize,
        seed: u64,
    },
    RandomSamples {
        nsamples: NonZeroUsize,
        seed: u64,
    },
    Medoid,
    LatinHyperCube {
        nsamples: NonZeroUsize,
        seed: u64,
    },
    FirstVector,
}

///////////////////////
// Search Phase      //
///////////////////////

/// A single search-parameter sweep run (a search_n / search_l grid emitted as one
/// row of the result table).
///
/// `search_n` plays three roles after the variable-k GT change:
///   - the K of the kNN search (how many results the graph walk returns),
///   - the cap on how many results we examine for BEIR-style recall,
///   - the cap on the BEIR denominator (`min(search_n, |relevant_q|)`).
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphSearch {
    pub(crate) search_n: usize,
    pub(crate) search_l: Vec<usize>,
}

impl CheckDeserialization for GraphSearch {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        for (i, l) in self.search_l.iter().enumerate() {
            if *l < self.search_n {
                return Err(anyhow!(
                    "search_l {} at position {} is less than search_n: {}",
                    l,
                    i,
                    self.search_n
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TopkSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl CheckDeserialization for TopkSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.check_deserialization(checker)?;
        self.groundtruth.check_deserialization(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }
        Ok(())
    }
}

//////////////////
// Build / Load //
//////////////////

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildConfig {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    pub(crate) alpha: f32,
    pub(crate) backedge_ratio: f32,
    pub(crate) num_threads: usize,
    #[serde(with = "StartPointStrategyRef")]
    pub(crate) start_point_strategy: StartPointStrategy,
    #[serde(default)]
    pub(crate) save_path: Option<String>,
}

impl BuildConfig {
    fn exact_max_degree(&self) -> usize {
        (self.max_degree as f32 * 1.3) as usize
    }

    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        let metric: diskann_vector::distance::Metric = self.distance.into();
        let exact_max_degree = self.exact_max_degree();
        let builder = config::Builder::new_with(
            self.max_degree,
            config::MaxDegree::new(exact_max_degree),
            self.l_build,
            metric.into(),
            |b| {
                b.alpha(self.alpha).backedge_ratio(self.backedge_ratio);
            },
        );
        Ok(builder)
    }

    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> DefaultProviderParameters {
        DefaultProviderParameters {
            max_points: num_points,
            frozen_points: NonZero::new(self.start_point_strategy.count()).unwrap(),
            metric: self.distance.into(),
            dim,
            max_degree: self.exact_max_degree() as u32,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
        }
    }
}

impl CheckDeserialization for BuildConfig {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.check_deserialization(checker)?;
        if let Some(save_path) = &self.save_path {
            let save_path_buf = std::path::Path::new(save_path).to_path_buf();
            let save_filename = save_path_buf
                .file_name()
                .unwrap_or_else(|| save_path_buf.as_os_str());
            let resolved_path = checker.register_output(save_path_buf.parent())?;
            let full_path = resolved_path.join(save_filename);
            self.save_path = Some(full_path.to_string_lossy().to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadConfig {
    pub(crate) data_type: DataType,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) load_path: String,
}

impl LoadConfig {
    /// Reconstruct an [`IndexConfiguration`] suitable for [`diskann_providers::storage::LoadWith`].
    ///
    /// Mirrors `IndexLoad::to_config` from `diskann-benchmark/src/inputs/graph_index.rs`.
    pub(crate) fn to_config(&self) -> Result<IndexConfiguration, anyhow::Error> {
        use diskann::utils::IntoUsize;
        use diskann_providers::utils::load_metadata_from_file;
        use diskann_providers::storage::FileStorageProvider;

        let storage_provider = FileStorageProvider;
        let num_frozen_pts = crate::backend::saveload::get_graph_num_frozen_points(
            &storage_provider,
            &self.load_path,
        )?;
        let max_observed_degree = crate::backend::saveload::get_graph_max_observed_degree(
            &storage_provider,
            &self.load_path,
        )?;
        let metadata =
            load_metadata_from_file(&storage_provider, &format!("{}.data", &self.load_path))?;

        let distance: diskann_vector::distance::Metric = self.distance.into();
        let config = config::Builder::new(
            max_observed_degree.into_usize(),
            config::MaxDegree::same(),
            1, // No building happening — L-build is irrelevant.
            distance.into(),
        )
        .build()?;

        Ok(IndexConfiguration::new(
            self.distance.into(),
            metadata.ndims(),
            metadata.npoints(),
            num_frozen_pts,
            1,
            config,
        ))
    }
}

impl CheckDeserialization for LoadConfig {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        let path = std::path::Path::new(&self.load_path);
        let p = checker.check_path(path)?;
        self.load_path = p.to_string_lossy().to_string();
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "index-source", deny_unknown_fields)]
pub(crate) enum Source {
    Build(BuildConfig),
    Load(LoadConfig),
}

impl Source {
    pub(crate) fn data_type(&self) -> &DataType {
        match self {
            Source::Build(b) => &b.data_type,
            Source::Load(l) => &l.data_type,
        }
    }
}

impl CheckDeserialization for Source {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            Source::Build(b) => b.check_deserialization(checker),
            Source::Load(l) => l.check_deserialization(checker),
        }
    }
}

///////////////////////////////////////
// MultiVectorRerankOperation        //
///////////////////////////////////////

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MultiVectorRerankOperation {
    pub(crate) source: Source,
    pub(crate) search: TopkSearchPhase,
    pub(crate) doc_mv: InputFile,
    pub(crate) query_mv: InputFile,
}

impl MultiVectorRerankOperation {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-rerank-build"
    }
}

impl CheckDeserialization for MultiVectorRerankOperation {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.source.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;
        self.doc_mv.check_deserialization(checker)?;
        self.query_mv.check_deserialization(checker)?;
        Ok(())
    }
}

impl diskann_benchmark_runner::Input for MultiVectorRerankOperation {
    fn tag() -> &'static str {
        Self::tag()
    }

    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        checker.any(<Self as serde::Deserialize>::deserialize(serialized)?)
    }

    fn example() -> anyhow::Result<serde_json::Value> {
        let example = Self {
            source: Source::Build(BuildConfig {
                data_type: DataType::Float32,
                data: InputFile::new("path/to/docs.fbin"),
                distance: SimilarityMeasure::InnerProduct,
                max_degree: 32,
                l_build: 64,
                alpha: 1.2,
                backedge_ratio: 1.0,
                num_threads: 8,
                start_point_strategy: StartPointStrategy::Medoid,
                save_path: Some("graph_save_dir".into()),
            }),
            search: TopkSearchPhase {
                queries: InputFile::new("path/to/queries.fbin"),
                groundtruth: InputFile::new("path/to/gt.bin"),
                reps: NonZeroUsize::new(3).unwrap(),
                num_threads: vec![NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(8).unwrap()],
                runs: vec![GraphSearch {
                    search_n: 10,
                    search_l: vec![10, 50, 100],
                }],
            },
            doc_mv: InputFile::new("path/to/docs.mvbin"),
            query_mv: InputFile::new("path/to/queries.mvbin"),
        };
        Ok(serde_json::to_value(example)?)
    }
}

impl std::fmt::Display for MultiVectorRerankOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Rerank Operation\n")?;
        writeln!(f, "{:>18}: {}", "tag", Self::tag())?;
        match &self.source {
            Source::Build(b) => {
                writeln!(f, "{:>18}: Build", "source")?;
                writeln!(f, "{:>18}: {}", "data", b.data.display())?;
                writeln!(f, "{:>18}: {}", "data_type", b.data_type)?;
                writeln!(f, "{:>18}: {}", "distance", b.distance)?;
                writeln!(f, "{:>18}: {}", "max_degree", b.max_degree)?;
                writeln!(f, "{:>18}: {}", "l_build", b.l_build)?;
                writeln!(f, "{:>18}: {}", "alpha", b.alpha)?;
                writeln!(f, "{:>18}: {}", "build_threads", b.num_threads)?;
                match &b.save_path {
                    Some(p) => writeln!(f, "{:>18}: {}", "save_path", p)?,
                    None => writeln!(f, "{:>18}: <none>", "save_path")?,
                }
            }
            Source::Load(l) => {
                writeln!(f, "{:>18}: Load", "source")?;
                writeln!(f, "{:>18}: {}", "load_path", l.load_path)?;
                writeln!(f, "{:>18}: {}", "data_type", l.data_type)?;
                writeln!(f, "{:>18}: {}", "distance", l.distance)?;
            }
        }
        writeln!(f, "{:>18}: {}", "doc_mv", self.doc_mv.display())?;
        writeln!(f, "{:>18}: {}", "query_mv", self.query_mv.display())?;
        Ok(())
    }
}
