/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZero;
use std::num::{NonZeroU32, NonZeroUsize};

use anyhow::{anyhow, Context};
use diskann::{
    graph::{self, config, RangeSearchParams, RangeSearchParamsError, StartPointStrategy},
    utils::IntoUsize,
};
use diskann_benchmark_runner::{
    files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker,
};
use diskann_providers::{
    model::{
        configuration::IndexConfiguration,
    },
    utils::load_metadata_from_file,
};
use diskann_inmem::DefaultProviderParameters;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::{self, as_input, save_and_load, Example, Input},
    utils::{
        datafiles::{DynamicRunbook, RunbookFile},
        SimilarityMeasure,
    },
};

//////////////
// Registry //
//////////////

as_input!(IndexOperation);
as_input!(IndexPQOperation);
as_input!(IndexSQOperation);
as_input!(SphericalQuantBuild);
as_input!(DynamicIndexRun);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<IndexOperation>::new())?;
    registry.register(Input::<IndexPQOperation>::new())?;
    registry.register(Input::<IndexSQOperation>::new())?;
    registry.register(Input::<SphericalQuantBuild>::new())?;
    registry.register(Input::<DynamicIndexRun>::new())?;
    Ok(())
}

////////////
// Search //
////////////

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct TargetRecall {
    pub(crate) target: Vec<usize>,
    pub(crate) percentile: Vec<f32>,
    pub(crate) max_search_l: NonZeroUsize,
    pub(crate) calibration_size: NonZeroUsize,
}

impl CheckDeserialization for TargetRecall {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        for p in self.percentile.iter() {
            if *p < 0.0 || *p > 1.0 {
                return Err(anyhow!("percentile {} is not in the range [0.0, 1.0]", p));
            }
        }

        if self.percentile.is_empty() {
            return Err(anyhow!("at least one percentile must be specified"));
        }
        if self.target.is_empty() {
            return Err(anyhow!("at least one target must be specified"));
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct GraphSearch {
    pub(crate) search_n: usize,
    pub(crate) search_l: Option<Vec<usize>>,
    pub(crate) target_recall: Option<Vec<TargetRecall>>,
    pub(crate) recall_k: usize,
    pub(crate) enhanced_metrics: Option<bool>,
}

impl CheckDeserialization for GraphSearch {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        if let Some(search_l) = self.search_l.as_mut() {
            for (i, l) in search_l.iter().enumerate() {
                if *l < self.search_n {
                    return Err(anyhow!(
                        "search_l {} at position {} is less than search_n: {}",
                        l,
                        i,
                        self.search_n
                    ));
                }
            }
        }
        if let Some(target_recall) = self.target_recall.as_mut() {
            for (i, tr) in target_recall.iter_mut().enumerate() {
                for target in tr.target.iter() {
                    if *target > self.search_n {
                        return Err(anyhow!(
                            "target_recall target_n at position {} has value {} which is greater than search_n: {}",
                            i,
                            target,
                            self.search_n
                        ));
                    }
                }
                tr.check_deserialization(checker)?;
            }
        }
        if self.search_l.is_none() && self.target_recall.is_none() {
            return Err(anyhow!(
                "at least one of search_l or target_recall must be specified"
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GraphRangeSearch {
    pub(crate) initial_search_l: Vec<usize>,
    pub(crate) radius: f32,
    pub(crate) inner_radius: Option<f32>,
    pub(crate) max_returned: Option<usize>,
    pub(crate) beam_width: Option<usize>,
    pub(crate) initial_search_slack: f32,
    pub(crate) range_search_slack: f32,
}

impl GraphRangeSearch {
    pub(crate) fn construct_params(
        &self,
    ) -> Result<Vec<RangeSearchParams>, RangeSearchParamsError> {
        self.initial_search_l
            .iter()
            .map(|&l| {
                RangeSearchParams::new(
                    self.max_returned,
                    l,
                    self.beam_width,
                    self.radius,
                    self.inner_radius,
                    self.initial_search_slack,
                    self.range_search_slack,
                )
            })
            .collect()
    }
}

impl CheckDeserialization for GraphRangeSearch {
    // all necessary checks are carried out when RangeSearchParams is initialized
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.construct_params()
            .context("invalid range search params")?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TopkSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl CheckDeserialization for TopkSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.queries.check_deserialization(checker)?;

        self.groundtruth.check_deserialization(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

impl Example for TopkSearchPhase {
    fn example() -> Self {
        const THREAD_COUNTS: [NonZeroUsize; 4] = [
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(4).unwrap(),
            NonZeroUsize::new(8).unwrap(),
        ];

        const REPS: NonZeroUsize = NonZeroUsize::new(5).unwrap();

        let runs = vec![GraphSearch {
            search_n: 10,
            search_l: Some(vec![10, 20, 30, 40]),
            target_recall: Some(vec![TargetRecall {
                target: vec![5, 6, 7],
                percentile: vec![0.9, 0.95],
                max_search_l: NonZeroUsize::new(1000).unwrap(),
                calibration_size: NonZeroUsize::new(1).unwrap(),
            }]),
            recall_k: 10,
            enhanced_metrics: None,
        }];

        Self {
            queries: InputFile::new("path/to/queries"),
            groundtruth: InputFile::new("path/to/groundtruth"),
            reps: REPS,
            num_threads: THREAD_COUNTS.to_vec(),
            runs,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RangeSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphRangeSearch>,
}

impl CheckDeserialization for RangeSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.queries.check_deserialization(checker)?;

        self.groundtruth.check_deserialization(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BetaSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    pub(crate) beta: f32,
    pub(crate) data_labels: InputFile,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl CheckDeserialization for BetaSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.queries.check_deserialization(checker)?;

        self.query_predicates.check_deserialization(checker)?;
        self.data_labels.check_deserialization(checker)?;

        if self.beta <= 0.0 || self.beta > 1.0 {
            return Err(anyhow::anyhow!(
                "beta must be in the range (0, 1], got: {}",
                self.beta
            ));
        }

        self.groundtruth.check_deserialization(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MultiHopSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    pub(crate) data_labels: InputFile,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl CheckDeserialization for MultiHopSearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.queries.check_deserialization(checker)?;

        self.query_predicates.check_deserialization(checker)?;
        self.data_labels.check_deserialization(checker)?;

        self.groundtruth.check_deserialization(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.check_deserialization(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

/// A one-to-one correspondence with [`diskann::index::config::IntraBatchCandidates`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum IntraBatchCandidates {
    /// No intra-batch candidates will be considered.
    None,
    /// An upper bound on the number of candidates. Smaller batches may not hit this max.
    Max(NonZeroU32),
    /// Consider all elements in the batch for intra-batch candidates.
    All,
}

impl std::fmt::Display for IntraBatchCandidates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Max(v) => write!(f, "{}", v),
            Self::All => write!(f, "all"),
        }
    }
}

impl From<IntraBatchCandidates> for config::IntraBatchCandidates {
    fn from(value: IntraBatchCandidates) -> Self {
        use IntraBatchCandidates::{All, Max, None};
        match value {
            None => Self::None,
            Max(v) => Self::Max(v),
            All => Self::All,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MultiInsert {
    pub(crate) batch_size: NonZeroUsize,
    pub(crate) batch_parallelism: NonZeroUsize,
    pub(crate) intra_batch_candidates: IntraBatchCandidates,
}

impl Example for MultiInsert {
    fn example() -> Self {
        const BATCH_SIZE: NonZeroUsize = NonZeroUsize::new(128).unwrap();
        const BATCH_PARALLELISM: NonZeroUsize = NonZeroUsize::new(32).unwrap();

        Self {
            batch_size: BATCH_SIZE,
            batch_parallelism: BATCH_PARALLELISM,
            intra_batch_candidates: IntraBatchCandidates::None,
        }
    }
}

// This constant is used to ensure that summaries of async-index related jobs properly have
// their field descriptions aligned.
const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "search-type", rename_all = "kebab-case")]
pub(crate) enum SearchPhase {
    Topk(TopkSearchPhase),
    Range(RangeSearchPhase),
    TopkBetaFilter(BetaSearchPhase),
    TopkMultihopFilter(MultiHopSearchPhase),
}

impl CheckDeserialization for SearchPhase {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            SearchPhase::Topk(phase) => phase.check_deserialization(checker),
            SearchPhase::Range(phase) => phase.check_deserialization(checker),
            SearchPhase::TopkBetaFilter(phase) => phase.check_deserialization(checker),
            SearchPhase::TopkMultihopFilter(phase) => phase.check_deserialization(checker),
        }
    }
}

////////////////////////////
// Build - Full Precision //
////////////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexLoad {
    pub(crate) data_type: DataType,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) load_path: String,
}

impl IndexLoad {
    pub(crate) const fn tag() -> &'static str {
        "async-index-load"
    }

    pub(crate) fn to_config(&self) -> Result<IndexConfiguration, anyhow::Error> {
        let storage_provider = diskann_providers::storage::FileStorageProvider;
        let num_frozen_pts =
            save_and_load::get_graph_num_frozen_points(&storage_provider, &self.load_path)?;

        let max_observed_degree =
            save_and_load::get_graph_max_observed_degree(&storage_provider, &self.load_path)?;

        let metadata =
            load_metadata_from_file(&storage_provider, &format!("{}.data", &self.load_path))?;

        let distance: diskann_vector::distance::Metric = self.distance.into();
        let config = config::Builder::new(
            max_observed_degree.into_usize(),
            config::MaxDegree::same(),
            1, // No building happening - no need to configure `l_build`.
            distance.into(),
        )
        .build()?;

        let index_config = IndexConfiguration::new(
            self.distance.into(),
            metadata.ndims,
            metadata.npoints,
            num_frozen_pts,
            1,
            config,
        );
        Ok(index_config)
    }

    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "data_type", self.data_type)?;
        write_field!(f, "Load Path", self.load_path)?;
        Ok(())
    }
}

impl CheckDeserialization for IndexLoad {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check if the file exists (allowing for relative paths with respect to the current
        // directory.
        //
        // This isn't a fully complete check since the index may be composed of multiple files,
        // but without encoding the type into the loader, it seems complicated to do better than this
        let path = std::path::Path::new(&self.load_path);
        let p = checker.check_path(path);
        match p {
            Ok(p) => {
                self.load_path = p.to_string_lossy().to_string();
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
}

impl std::fmt::Display for IndexLoad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async Full-Precision Index Load\n")?;

        write_field!(f, "tag", Self::tag())?;

        self.summarize_fields(f)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct InsertRetry {
    num_insert_attempts: NonZeroU32,
    retry_threshold: f32,
    saturate_inserts: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(remote = "StartPointStrategy")]
#[serde(rename_all = "snake_case")]
pub enum StartPointStrategyRef {
    /// Randomly select vector(s) with given norm as starting points with seed provided.
    /// Requires the norm (f32), number of samples (usize), and random seed (u64) to be provided.
    RandomVectors {
        norm: f32,
        nsamples: NonZeroUsize,
        seed: u64,
    },

    /// Sample data from the dataset with seed provided.
    /// Requires number of samples (usize) and random seed (u64) to be provided.
    RandomSamples { nsamples: NonZeroUsize, seed: u64 },

    /// Use the medoid as the starting point. Can select only one starting point.
    Medoid,

    /// Use the Latin Hypercube sampling method to select the starting points.
    /// Requires number of samples (usize) and random seed (u64) to be provided.
    LatinHyperCube { nsamples: NonZeroUsize, seed: u64 },

    /// Use the first vector in the dataset as the starting point. Can select only one starting point.
    FirstVector,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexBuild {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    pub(crate) insert_retry: Option<InsertRetry>,
    #[serde(with = "StartPointStrategyRef")]
    pub(crate) start_point_strategy: StartPointStrategy,
    pub(crate) alpha: f32,
    pub(crate) backedge_ratio: f32,
    pub(crate) num_threads: usize,
    pub(crate) multi_insert: Option<MultiInsert>,
    pub(crate) save_path: Option<String>,
}

impl IndexBuild {
    pub(crate) const fn tag() -> &'static str {
        "async-index-builder"
    }

    fn exact_max_degree(&self) -> usize {
        (self.max_degree as f32 * 1.3) as usize
    }

    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        let metric: diskann_vector::distance::Metric = self.distance.into();
        let exact_max_degree = self.exact_max_degree();
        let mut builder = config::Builder::new_with(
            self.max_degree,
            config::MaxDegree::new(exact_max_degree),
            self.l_build,
            metric.into(),
            |builder| {
                builder
                    .alpha(self.alpha)
                    .backedge_ratio(self.backedge_ratio);

                if let Some(mi) = &self.multi_insert {
                    builder
                        .max_minibatch_par(mi.batch_parallelism.get())
                        .intra_batch_candidates(mi.intra_batch_candidates.into());
                }
            },
        );

        if let Some(insert_retry) = self.insert_retry.as_ref() {
            let threshold =
                NonZeroU32::new((insert_retry.retry_threshold * exact_max_degree as f32) as u32)
                    .ok_or_else(|| {
                        anyhow::Error::msg("retry threshold could not fit in a NonZerou32")
                    })?;
            let retry = diskann::graph::config::experimental::InsertRetry::new(
                insert_retry.num_insert_attempts,
                threshold,
                insert_retry.saturate_inserts,
            );

            builder.insert_retry(retry);
        }

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

    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "file", self.data.display())?;
        write_field!(f, "data_type", self.data_type)?;
        write_field!(f, "max degree", self.max_degree)?;
        write_field!(f, "L-build", self.l_build)?;
        write_field!(f, "alpha", self.alpha)?;
        write_field!(f, "start point strategy", self.start_point_strategy)?;
        write_field!(f, "backedge ratio", self.backedge_ratio)?;
        match &self.multi_insert {
            None => write_field!(f, "Using Multi Insert", "NO")?,
            Some(mi) => {
                write_field!(f, "Insert Batch Size", mi.batch_size)?;
                write_field!(f, "Batch Parallelism", mi.batch_parallelism)?;
                write_field!(f, "Intra Batch Candidates", mi.intra_batch_candidates)?;
            }
        }
        write_field!(f, "start_point_strategy", self.start_point_strategy)?;
        write_field!(f, "build threads", self.num_threads)?;
        match &self.save_path {
            None => write_field!(f, "Save Path", "None")?,
            Some(p) => write_field!(f, "Save Path", p)?,
        }
        Ok(())
    }
}

impl CheckDeserialization for IndexBuild {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.data.check_deserialization(checker)?;

        // We allow overwriting of already existing save paths, since users like to do this
        // The save path must either (1) be an absolute path, in which case we check that its parent directory exists
        // or (2) it must be a file written to the output directory, in which case we concatenate the path and
        // ensure the parent directory exists or (3) if it has no parent, it is written to the output directory
        if let Some(save_path) = &self.save_path {
            let save_path = std::path::Path::new(save_path).to_path_buf();
            let save_filename = save_path
                .file_name()
                .unwrap_or_else(|| save_path.as_os_str());
            let resolved_path = checker.register_output(save_path.parent())?;
            let full_path = resolved_path.join(save_filename);
            self.save_path = Some(full_path.to_string_lossy().to_string());
        }

        Ok(())
    }
}

impl Example for IndexBuild {
    fn example() -> Self {
        Self {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data"),
            distance: SimilarityMeasure::SquaredL2,
            max_degree: 32,
            l_build: 50,
            alpha: 1.2,
            backedge_ratio: 1.0,
            num_threads: 1,
            multi_insert: Some(MultiInsert::example()),
            insert_retry: None,
            start_point_strategy: StartPointStrategy::Medoid,
            save_path: None,
        }
    }
}

impl std::fmt::Display for IndexBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async Full-Precision Index Build\n")?;

        write_field!(f, "tag", Self::tag())?;

        self.summarize_fields(f)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "index-source")] // Use tagged enums for JSON
pub enum IndexSource {
    Load(IndexLoad),
    Build(IndexBuild),
}

impl CheckDeserialization for IndexSource {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            IndexSource::Load(load) => load.check_deserialization(checker),
            IndexSource::Build(build) => build.check_deserialization(checker),
        }
    }
}

impl IndexSource {
    fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexSource::Load(load) => load.summarize_fields(f),
            IndexSource::Build(build) => build.summarize_fields(f),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexOperation {
    pub(crate) source: IndexSource, // either load or build
    pub(crate) search_phase: SearchPhase,
}

impl IndexOperation {
    pub(crate) const fn tag() -> &'static str {
        "async-index-build"
    }
}

impl CheckDeserialization for IndexOperation {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        // Check the validity of the input files.
        self.source.check_deserialization(checker)?;
        self.search_phase.check_deserialization(checker)?;

        Ok(())
    }
}

impl Example for IndexOperation {
    fn example() -> Self {
        Self {
            source: IndexSource::Build(IndexBuild::example()),
            search_phase: SearchPhase::Topk(TopkSearchPhase::example()),
        }
    }
}

impl std::fmt::Display for IndexOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async Full-Precision Index Build\n")?;

        write_field!(f, "tag", Self::tag())?;

        self.source.summarize_fields(f)
    }
}

////////////////////
// Async Build PQ //
////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexPQOperation {
    pub(crate) index_operation: IndexOperation, // either load or build
    pub(crate) num_pq_chunks: usize,
    pub(crate) seed: u64,
    pub(crate) max_fp_vecs_per_prune: Option<usize>,
    pub(crate) use_fp_for_search: bool,
}

impl IndexPQOperation {
    pub(crate) const fn tag() -> &'static str {
        "async-index-build-pq"
    }

    #[cfg(feature = "product-quantization")]
    pub(crate) fn to_config(&self) -> Result<IndexConfiguration, anyhow::Error> {
        match &self.index_operation.source {
            IndexSource::Load(load) => load.to_config(),
            IndexSource::Build(_) => Err(anyhow::anyhow!(
                "This function not supported on Build type, as it is only used during loading."
            )),
        }
    }

    #[cfg(feature = "product-quantization")]
    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        match &self.index_operation.source {
            IndexSource::Load(_) => Err(anyhow::anyhow!(
                "This function not supported on Load type, as it is only used during building."
            )),
            IndexSource::Build(build) => build.try_as_config(),
        }
    }

    #[cfg(feature = "product-quantization")]
    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> Result<DefaultProviderParameters, anyhow::Error> {
        match &self.index_operation.source {
            IndexSource::Load(_) => Err(anyhow::anyhow!(
                "inmem_parameters is only supported for builds, not loads"
            )),
            IndexSource::Build(b) => Ok(b.inmem_parameters(num_points, dim)),
        }
    }
}

impl CheckDeserialization for IndexPQOperation {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.index_operation.check_deserialization(checker)
    }
}

impl Example for IndexPQOperation {
    fn example() -> Self {
        Self {
            index_operation: IndexOperation::example(),
            num_pq_chunks: 16,
            seed: 0xb578b71e688e65e3,
            max_fp_vecs_per_prune: Some(48),
            use_fp_for_search: false,
        }
    }
}

impl std::fmt::Display for IndexPQOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async PQ Index Build")?;
        write_field!(f, "tag", Self::tag())?;
        write_field!(f, "PQ Chunks", self.num_pq_chunks)?;
        const MAX_FP_VECS: &str = "Max FP Vecs";
        match &self.max_fp_vecs_per_prune {
            Some(v) => write_field!(f, MAX_FP_VECS, v)?,
            None => write_field!(f, MAX_FP_VECS, "none")?,
        }
        write_field!(f, "Use Full Precision for Search: ", self.use_fp_for_search)?;
        // New line to separate PQ parameters from index build parameters.
        writeln!(f)?;
        self.index_operation.source.summarize_fields(f)?;

        Ok(())
    }
}

////////////////////
// Async Build SQ //
////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct IndexSQOperation {
    pub(crate) index_operation: IndexOperation,
    pub(crate) num_bits: usize,
    pub(crate) standard_deviations: f64,
    pub(crate) use_fp_for_search: bool,
}

impl IndexSQOperation {
    pub(crate) const fn tag() -> &'static str {
        "async-index-build-sq"
    }

    #[cfg(feature = "scalar-quantization")]
    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        match &self.index_operation.source {
            IndexSource::Load(_) => Err(anyhow::anyhow!(
                "This function not supported on Load type, as it is only used during building."
            )),
            IndexSource::Build(build) => build.try_as_config(),
        }
    }

    #[cfg(feature = "scalar-quantization")]
    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> Result<DefaultProviderParameters, anyhow::Error> {
        match &self.index_operation.source {
            IndexSource::Load(_) => Err(anyhow::anyhow!(
                "inmem_parameters is only supported for builds, not loads"
            )),
            IndexSource::Build(b) => Ok(b.inmem_parameters(num_points, dim)),
        }
    }
}

impl CheckDeserialization for IndexSQOperation {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        if self.standard_deviations <= 0.0 {
            return Err(anyhow::anyhow!(
                "scalar quantization standard deviations ({}) must be strictly positive",
                self.standard_deviations
            ));
        }

        self.index_operation.check_deserialization(checker)
    }
}

impl Example for IndexSQOperation {
    fn example() -> Self {
        // Scalar Quantization does not support multi-insert, so make sure that we explicitly
        // disable multi-insert from the example input.
        let mut index_operation = IndexOperation::example();
        match &mut index_operation.source {
            IndexSource::Load(_) => {}
            IndexSource::Build(b) => b.multi_insert = None,
        }

        Self {
            index_operation,
            num_bits: 4,
            standard_deviations: 2.0,
            use_fp_for_search: false,
        }
    }
}

impl std::fmt::Display for IndexSQOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async SQ Index Build")?;
        write_field!(f, "tag", Self::tag())?;
        write_field!(f, "SQ bits", self.num_bits)?;
        write_field!(f, "StdDev", self.standard_deviations)?;
        write_field!(f, "Use FP Search", self.use_fp_for_search)?;

        // New line to separate SQ parameters from index build parameters.
        writeln!(f)?;
        self.index_operation.source.summarize_fields(f)?;

        Ok(())
    }
}

///////////////////////////
// Async Build Spherical //
///////////////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SphericalQuantBuild {
    pub(crate) build: IndexBuild, // spherical does not support saving and loading
    pub(crate) search_phase: SearchPhase,
    pub(crate) seed: u64,
    pub(crate) transform_kind: inputs::exhaustive::TransformKind,
    pub(crate) query_layouts: Vec<inputs::exhaustive::SphericalQuery>,
    pub(crate) num_bits: NonZeroUsize,
    pub(crate) pre_scale: Option<inputs::exhaustive::PreScale>,
}

impl SphericalQuantBuild {
    pub(crate) const fn tag() -> &'static str {
        "async-index-build-spherical-quantization"
    }

    #[cfg(feature = "spherical-quantization")]
    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        self.build.try_as_config()
    }

    #[cfg(feature = "spherical-quantization")]
    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> DefaultProviderParameters {
        self.build.inmem_parameters(num_points, dim)
    }
}

impl CheckDeserialization for SphericalQuantBuild {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.build.check_deserialization(checker)?;
        self.search_phase.check_deserialization(checker)?;

        if self.build.save_path.is_some() {
            return Err(anyhow::anyhow!(
                "Spherical quantization does not support saving the index"
            ));
        }

        // Check query plan.
        for (i, layout) in self.query_layouts.iter().enumerate() {
            inputs::exhaustive::check_compatibility(self.num_bits.get(), *layout).with_context(
                || {
                    format!(
                        "while validating query layout {} of {}",
                        i + 1,
                        self.query_layouts.len()
                    )
                },
            )?;
        }

        if let Some(pre_scale) = &mut self.pre_scale {
            pre_scale.check_deserialization(checker)?;
        }

        Ok(())
    }
}

impl Example for SphericalQuantBuild {
    fn example() -> Self {
        let mut build = IndexBuild::example();
        build.multi_insert = None;

        const NUM_BITS: NonZeroUsize = NonZeroUsize::new(1).unwrap();

        Self {
            build,
            search_phase: SearchPhase::Topk(TopkSearchPhase::example()),
            seed: 0xc0ffee,
            transform_kind: inputs::exhaustive::TransformKind::PaddingHadamard(
                inputs::exhaustive::TargetDim::Same,
            ),
            query_layouts: vec![
                inputs::exhaustive::SphericalQuery::FourBitTransposed,
                inputs::exhaustive::SphericalQuery::SameAsData,
                inputs::exhaustive::SphericalQuery::ScalarQuantized,
            ],
            num_bits: NUM_BITS,
            pre_scale: None,
        }
    }
}

impl std::fmt::Display for SphericalQuantBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async Spherically Quantized Index Build")?;
        if cfg!(not(feature = "spherical-quantization")) {
            writeln!(f, "Requires the `spherical-quantization` feature")?;
        }

        write_field!(f, "tag", Self::tag())?;

        write_field!(f, "seed", self.seed)?;
        write_field!(f, "Transform kind", self.transform_kind)?;
        write_field!(f, "Num Bits", self.num_bits)?;
        write_field!(
            f,
            "Pre Scale",
            self.pre_scale
                .as_ref()
                .unwrap_or(&inputs::exhaustive::PreScale::None)
        )?;

        // New line to separate SQ parameters from index build parameters.
        writeln!(f)?;
        self.build.summarize_fields(f)?;

        Ok(())
    }
}

////////////////////////////
// Dynamic Runbook Params //
////////////////////////////

#[derive(Copy, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(tag = "method", content = "params")]
pub enum InplaceDeleteMethod {
    #[serde(rename = "visited_and_top_k")]
    VisitedAndTopK { k_value: usize, l_value: usize },
    #[serde(rename = "two_hop_and_one_hop")]
    TwoHopAndOneHop,
    #[serde(rename = "one_hop")]
    OneHop,
}

impl From<InplaceDeleteMethod> for graph::InplaceDeleteMethod {
    fn from(value: InplaceDeleteMethod) -> Self {
        match value {
            InplaceDeleteMethod::VisitedAndTopK { k_value, l_value } => {
                graph::InplaceDeleteMethod::VisitedAndTopK { k_value, l_value }
            }
            InplaceDeleteMethod::TwoHopAndOneHop => graph::InplaceDeleteMethod::TwoHopAndOneHop,
            InplaceDeleteMethod::OneHop => graph::InplaceDeleteMethod::OneHop,
        }
    }
}

/// Runbook loading and phase type definitions are in utils.datafiles
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DynamicRunbookParams {
    pub(crate) runbook_path: InputFile,
    pub(crate) dataset_name: String,
    pub(crate) gt_directory: String,
    pub(crate) ip_delete_method: InplaceDeleteMethod,
    pub(crate) ip_delete_num_to_replace: usize,
    pub(crate) consolidate_threshold: f32,
    #[serde(skip)]
    pub(crate) resolved_gt_directory: Option<std::path::PathBuf>,
}

// Validates:
// 1. The runbook file can be parsed
// 2. The dataset_name exists in the runbook
// 3. All required ground truth files exist in gt_directory
impl CheckDeserialization for DynamicRunbookParams {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.runbook_path.check_deserialization(checker)?;

        // Validate consolidate_threshold is greater than 0
        if self.consolidate_threshold <= 0.0 {
            return Err(anyhow::anyhow!(
                "consolidate_threshold must be greater than 0, but got {}",
                self.consolidate_threshold
            ));
        }

        // Resolve gt_directory using search directories, similar to InputFile
        let mut resolved_gt_directory = None;
        let gt_path = std::path::Path::new(&self.gt_directory);

        // Check if the path is absolute or exists relative to current directory
        if gt_path.is_dir() {
            resolved_gt_directory = Some(gt_path.to_path_buf());
        } else if gt_path.is_absolute() {
            return Err(anyhow::anyhow!(
                "Ground truth directory with absolute path \"{}\" either does not exist or is not a directory",
                self.gt_directory
            ));
        } else {
            // Search in the provided directories
            for dir in checker.search_directories() {
                let absolute = dir.join(gt_path);
                if absolute.is_dir() {
                    resolved_gt_directory = Some(absolute);
                    break;
                }
            }
        }

        let final_gt_directory = resolved_gt_directory.ok_or_else(|| {
            anyhow::anyhow!(
                "Could not find ground truth directory \"{}\" in the search directories: {:?}",
                self.gt_directory,
                checker.search_directories()
            )
        })?;

        // Store the resolved path for later use
        self.resolved_gt_directory = Some(final_gt_directory.clone());

        let runbook_file = RunbookFile(&self.runbook_path);
        let _runbook = DynamicRunbook::new_from_runbook_file(
            runbook_file,
            self.dataset_name.clone(),
            Some(&final_gt_directory.to_string_lossy()),
        )
        .with_context(|| {
            format!(
                "Failed to validate runbook '{}' with dataset '{}' and gt_directory '{}'",
                self.runbook_path.display(),
                self.dataset_name,
                self.gt_directory
            )
        })?;

        Ok(())
    }
}

impl Example for DynamicRunbookParams {
    fn example() -> Self {
        Self {
            runbook_path: InputFile::new("path/to/runbook"),
            dataset_name: "dataset-1M".into(),
            gt_directory: "parent_directory/to/gt".into(),
            ip_delete_method: InplaceDeleteMethod::VisitedAndTopK {
                k_value: 10,
                l_value: 64,
            },
            ip_delete_num_to_replace: 3,
            consolidate_threshold: 0.2,
            resolved_gt_directory: None,
        }
    }
}

impl std::fmt::Display for DynamicRunbookParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dynamic Runbook Parameters")?;
        write_field!(f, "Runbook Path", self.runbook_path.display())?;
        write_field!(f, "Dataset Name", self.dataset_name)?;

        // Show resolved path if available, otherwise show original
        let gt_dir_display = match &self.resolved_gt_directory {
            Some(resolved) => resolved.display().to_string(),
            None => self.gt_directory.clone(),
        };
        write_field!(f, "Ground Truth Directory", gt_dir_display)?;

        match self.ip_delete_method {
            InplaceDeleteMethod::VisitedAndTopK { k_value, l_value } => {
                write_field!(f, "IP Delete Method", "VisitedAndTopK")?;
                write_field!(f, "IP Delete K Value", k_value)?;
                write_field!(f, "IP Delete L Value", l_value)?;
            }
            InplaceDeleteMethod::TwoHopAndOneHop => {
                write_field!(f, "IP Delete Method", "TwoHopAndOneHop")?;
            }
            InplaceDeleteMethod::OneHop => {
                write_field!(f, "IP Delete Method", "OneHop")?;
            }
        }
        write_field!(f, "IP Delete Num to Replace", self.ip_delete_num_to_replace)?;
        write_field!(f, "Consolidate Threshold", self.consolidate_threshold)?;

        Ok(())
    }
}

///////////////////
// Async Dynamic //
///////////////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct DynamicIndexRun {
    pub(crate) build: IndexBuild,
    pub(crate) search_phase: SearchPhase,
    pub(crate) runbook_params: DynamicRunbookParams,
}

impl DynamicIndexRun {
    pub(crate) const fn tag() -> &'static str {
        "async-dynamic-index-run"
    }

    pub(crate) fn try_as_config(&self, insert_l: usize) -> anyhow::Result<config::Builder> {
        let mut builder = self.build.try_as_config()?;
        builder.l_build(insert_l);
        Ok(builder)
    }

    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> DefaultProviderParameters {
        self.build.inmem_parameters(num_points, dim)
    }
}

impl CheckDeserialization for DynamicIndexRun {
    fn check_deserialization(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.build.check_deserialization(checker)?;
        self.runbook_params.check_deserialization(checker)?;
        self.search_phase.check_deserialization(checker)?;
        Ok(())
    }
}

impl Example for DynamicIndexRun {
    fn example() -> Self {
        let build = IndexBuild::example();

        Self {
            build,
            search_phase: SearchPhase::Topk(TopkSearchPhase::example()),
            runbook_params: DynamicRunbookParams::example(),
        }
    }
}

impl std::fmt::Display for DynamicIndexRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Async Dynamic Index Run")?;
        write_field!(f, "tag", Self::tag())?;
        writeln!(f, "Runbook Parameters:")?;
        write!(f, "{}", self.runbook_params)?;

        writeln!(f, "Index Build Parameters:")?;
        self.build.summarize_fields(f)?;

        Ok(())
    }
}
