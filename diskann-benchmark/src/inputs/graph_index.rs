/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::{NonZeroU32, NonZeroUsize};

use anyhow::{anyhow, Context};
use diskann::graph::{self, config, search::Range, RangeSearchError, StartPointStrategy};
use diskann_benchmark_core::streaming::executors::bigann;
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    inputs::{write_field, Example, PRINT_WIDTH},
    utils::SimilarityMeasure,
};

////////////
// Search //
////////////

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct GraphSearch {
    pub(crate) search_n: usize,
    pub(crate) search_l: Vec<usize>,
    pub(crate) recall_k: usize,
}

impl GraphSearch {
    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
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
    pub(crate) fn construct_params(&self) -> Result<Vec<Range>, RangeSearchError> {
        self.initial_search_l
            .iter()
            .map(|&l| {
                Range::builder(l, self.radius)
                    .max_returned(self.max_returned)
                    .beam_width(self.beam_width)
                    .inner_radius(self.inner_radius)
                    .initial_slack(self.initial_search_slack)
                    .range_slack(self.range_search_slack)
                    .build()
            })
            .collect()
    }
}

impl GraphRangeSearch {
    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.construct_params()
            .context("invalid range search params")?;

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TopkSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl TopkSearchPhase {
    pub(crate) fn max_k(&self) -> usize {
        self.runs.iter().map(|run| run.recall_k).max().unwrap_or(0)
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.resolve(checker)?;
        self.groundtruth.resolve(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.validate(checker)
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
            search_l: vec![10, 20, 30, 40],
            recall_k: 10,
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

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FilteredRangeSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) data_labels: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphRangeSearch>,
}

impl FilteredRangeSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.resolve(checker)?;
        self.query_predicates.resolve(checker)?;
        self.data_labels.resolve(checker)?;
        self.groundtruth.resolve(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.validate(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

impl RangeSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.resolve(checker)?;
        self.groundtruth.resolve(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.validate(checker)
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

impl BetaSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.resolve(checker)?;
        self.query_predicates.resolve(checker)?;
        self.data_labels.resolve(checker)?;

        if self.beta <= 0.0 || self.beta > 1.0 {
            return Err(anyhow::anyhow!(
                "beta must be in the range (0, 1], got: {}",
                self.beta
            ));
        }

        self.groundtruth.resolve(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.validate(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct MultihopFilterSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) query_predicates: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) reps: NonZeroUsize,
    pub(crate) data_labels: InputFile,
    // Enable sweeping threads
    pub(crate) num_threads: Vec<NonZeroUsize>,
    pub(crate) runs: Vec<GraphSearch>,
}

impl MultihopFilterSearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries.resolve(checker)?;
        self.query_predicates.resolve(checker)?;
        self.data_labels.resolve(checker)?;
        self.groundtruth.resolve(checker)?;
        for (i, run) in self.runs.iter_mut().enumerate() {
            run.validate(checker)
                .with_context(|| format!("search run {}", i))?;
        }

        Ok(())
    }
}

/// A one-to-one correspondence with [`diskann::graph::config::IntraBatchCandidates`].
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "search-type", rename_all = "kebab-case")]
pub(crate) enum SearchPhase {
    Topk(TopkSearchPhase),
    Range(RangeSearchPhase),
    FilteredRange(FilteredRangeSearchPhase),
    TopkBetaFilter(BetaSearchPhase),
    TopkMultihopFilter(MultihopFilterSearchPhase),
}

#[derive(Debug, Error)]
#[error(
    "INTERNAL ERROR: expected search phase kind \"{}\" - instead got \"{}\"",
    self.expected,
    self.got
)]
pub(crate) struct WrongSearchPhaseKind {
    expected: SearchPhaseKind,
    got: SearchPhaseKind,
}

impl WrongSearchPhaseKind {
    fn new(expected: SearchPhaseKind, got: SearchPhaseKind) -> Self {
        Self { expected, got }
    }
}

impl SearchPhase {
    pub(crate) fn kind(&self) -> SearchPhaseKind {
        match self {
            Self::Topk(_) => SearchPhaseKind::Topk,
            Self::Range(_) => SearchPhaseKind::Range,
            Self::FilteredRange(_) => SearchPhaseKind::FilteredRange,
            Self::TopkBetaFilter(_) => SearchPhaseKind::TopkBetaFilter,
            Self::TopkMultihopFilter(_) => SearchPhaseKind::TopkMultihopFilter,
        }
    }

    pub(crate) fn as_topk(&self) -> Result<&TopkSearchPhase, WrongSearchPhaseKind> {
        match self {
            Self::Topk(phase) => Ok(phase),
            _ => Err(WrongSearchPhaseKind::new(
                SearchPhaseKind::Topk,
                self.kind(),
            )),
        }
    }
}

impl SearchPhase {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            SearchPhase::Topk(phase) => phase.validate(checker),
            SearchPhase::Range(phase) => phase.validate(checker),
            SearchPhase::FilteredRange(phase) => phase.validate(checker),
            SearchPhase::TopkBetaFilter(phase) => phase.validate(checker),
            SearchPhase::TopkMultihopFilter(phase) => phase.validate(checker),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchPhaseKind {
    Topk,
    Range,
    FilteredRange,
    TopkBetaFilter,
    TopkMultihopFilter,
}

impl SearchPhaseKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Topk => "topk",
            Self::Range => "range",
            Self::FilteredRange => "filtered-range",
            Self::TopkBetaFilter => "topk-beta-filter",
            Self::TopkMultihopFilter => "topk-multihop-filter",
        }
    }
}

impl std::fmt::Display for SearchPhaseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

////////////////////////////
// Build - Full Precision //
////////////////////////////

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
    data_type: DataType,
    data: InputFile,
    distance: SimilarityMeasure,
    max_degree: usize,
    l_build: usize,
    insert_retry: Option<InsertRetry>,
    #[serde(with = "StartPointStrategyRef")]
    start_point_strategy: StartPointStrategy,
    alpha: f32,
    backedge_ratio: f32,
    num_threads: usize,
    multi_insert: Option<MultiInsert>,
    save_path: Option<String>,
}

impl IndexBuild {
    pub(crate) const fn tag() -> &'static str {
        "graph-index-builder"
    }

    pub(crate) fn exact_max_degree(&self) -> usize {
        (self.max_degree as f32 * 1.3) as usize
    }

    #[cfg(feature = "bftree")]
    pub(crate) fn max_degree(&self) -> u32 {
        self.max_degree as u32
    }

    #[cfg(feature = "bftree")]
    pub(crate) fn graph_slack_factor(&self) -> f32 {
        1.3
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

    pub(crate) fn summarize_fields(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker)?;

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

    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn num_threads(&self) -> usize {
        self.num_threads
    }

    #[cfg(any(feature = "spherical-quantization", feature = "bftree"))]
    pub(crate) fn distance(&self) -> SimilarityMeasure {
        self.distance
    }

    pub(crate) fn data(&self) -> &InputFile {
        &self.data
    }

    pub(crate) fn start_point_strategy(&self) -> &StartPointStrategy {
        &self.start_point_strategy
    }

    pub(crate) fn multi_insert(&self) -> Option<&MultiInsert> {
        self.multi_insert.as_ref()
    }

    pub(crate) fn save_path(&self) -> Option<&str> {
        self.save_path.as_deref()
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
        writeln!(f, "Graph Index Full-Precision Build\n")?;

        write_field!(f, "tag", Self::tag())?;

        self.summarize_fields(f)
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
    /// Threshold for deferred consolidation. Required for soft-delete providers (inmem).
    /// Hard-delete providers (bf-tree) ignore this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) consolidate_threshold: Option<f32>,
    #[serde(skip)]
    pub(crate) resolved_gt_directory: Option<std::path::PathBuf>,
}

// Validates:
// 1. The runbook file can be parsed
// 2. The dataset_name exists in the runbook
// 3. All required ground truth files exist in gt_directory
impl DynamicRunbookParams {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.runbook_path.resolve(checker)?;

        // Validate consolidate_threshold if provided
        if let Some(threshold) = self.consolidate_threshold {
            if threshold <= 0.0 {
                return Err(anyhow::anyhow!(
                    "consolidate_threshold must be greater than 0, but got {}",
                    threshold
                ));
            }
        }

        let final_gt_directory = checker.find_input_dir(self.gt_directory.as_ref())?;
        self.resolved_gt_directory = Some(final_gt_directory.clone());

        // Run pre-flight checks for the runbook.
        let _runbook = bigann::RunBook::load(
            &self.runbook_path,
            &self.dataset_name,
            &mut bigann::ScanDirectory::new(final_gt_directory)?,
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
            consolidate_threshold: Some(0.2),
            resolved_gt_directory: None,
        }
    }
}

impl DynamicRunbookParams {
    /// Example for hard-delete providers that don't use consolidation.
    #[cfg(feature = "bftree")]
    pub(crate) fn example_immediate() -> Self {
        Self {
            consolidate_threshold: None,
            ..Self::example()
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
        if let Some(threshold) = self.consolidate_threshold {
            write_field!(f, "Consolidate Threshold", threshold)?;
        }

        Ok(())
    }
}
