/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, path::Path};

use anyhow::Context;
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, Checker};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    inputs::{as_input, Example},
    utils::SimilarityMeasure,
};

//////////////
// Registry //
//////////////

as_input!(GraphIvfOperation);

/////////////
// Recall@ //
/////////////

/// The `k` values recall is reported at.
///
/// Accepts either `50` or `[50, 1000]`. A sweep searches once to the largest
/// value and scores every value from that one result set, so a second `k` costs
/// a set intersection rather than another pass over the queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecallAt(Vec<u32>);

/// Wire form of [`RecallAt`]. Untagged so a bare number and a list both parse,
/// which keeps every config written against the single-valued field working.
#[derive(Deserialize)]
#[serde(untagged)]
enum RecallAtRepr {
    One(u32),
    Many(Vec<u32>),
}

impl RecallAt {
    pub(crate) fn new(values: Vec<u32>) -> Self {
        Self(values)
    }

    /// The depth every search must run to for all configured `k` to be scorable.
    pub(crate) fn max(&self) -> u32 {
        self.0.iter().copied().max().unwrap_or(0)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().copied()
    }

    /// Sorted ascending and deduplicated, so reported columns are ordered by
    /// depth however the config listed them.
    fn validate(&mut self) -> Result<(), anyhow::Error> {
        if self.0.is_empty() {
            anyhow::bail!("recall_at must have at least one value");
        }
        if self.0.contains(&0) {
            anyhow::bail!("recall_at values must be positive");
        }
        self.0.sort_unstable();
        self.0.dedup();
        Ok(())
    }
}

impl<'de> Deserialize<'de> for RecallAt {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match RecallAtRepr::deserialize(deserializer)? {
            RecallAtRepr::One(k) => Self(vec![k]),
            RecallAtRepr::Many(ks) => Self(ks),
        })
    }
}

impl Serialize for RecallAt {
    /// Round-trips the form it was written in: a lone value stays a number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.as_slice() {
            [k] => k.serialize(serializer),
            ks => ks.serialize(serializer),
        }
    }
}

impl fmt::Display for RecallAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        CommaList(&self.0).fmt(f)
    }
}

///////////
// Input //
///////////

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct GraphIvfOperation {
    pub(crate) source: GraphIvfSource,
    /// Omit to build (or load) the index without searching it.
    #[serde(default)]
    pub(crate) search_phase: Option<GraphIvfSearchPhase>,
}

/// How the index under test comes into being.
///
/// The tag values match the `build_kind` reported in the results, so a config and the
/// output it produced name the same builder.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "graph-ivf-source")] // Use tagged enums for JSON
pub(crate) enum GraphIvfSource {
    Load(GraphIvfLoad),
    Static(GraphIvfStaticBuild),
    Online(GraphIvfOnlineBuild),
    OnlineRunbook(GraphIvfOnlineRunbook),
}

#[cfg(feature = "graph-ivf")]
impl GraphIvfSource {
    /// The stored element type this job operates on, whichever way the index is sourced.
    pub(crate) fn data_type(&self) -> DataType {
        match self {
            Self::Load(load) => load.data_type,
            Self::Static(build) => build.data_type,
            Self::Online(online) => online.data_type,
            Self::OnlineRunbook(runbook) => runbook.build.data_type,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfLoad {
    pub(crate) data_type: DataType,
    /// Path prefix the index was saved under (without the `.graphivf_*` suffix).
    pub(crate) load_path: String,
}

/// A static (batch) build: `k` centroids are fit by k-means over a sample of the
/// corpus, then every point is assigned to one of them.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfStaticBuild {
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
    /// Nearest-centroid assignment strategy used during k-means refinement.
    /// Defaults to exact brute-force assignment when omitted.
    #[serde(default)]
    pub(crate) assign_method: AssignMethodConfig,
    /// Policy for clusters that become empty during k-means refinement.
    #[serde(default)]
    pub(crate) empty_clusters: EmptyClusterConfig,
    /// Path prefix to save the index under (without the `.graphivf_*` suffix).
    pub(crate) save_path: String,
}

/// An online (incremental) build: points are streamed in corpus order and clusters
/// split whenever they overflow, so the partition emerges from the data.
///
/// Deliberately disjoint from [`GraphIvfStaticBuild`]: an online build takes no target
/// cluster count, no k-means sampling, and no empty-cluster policy, while
/// `split_threshold` and the reassignment knobs have no meaning for a batch build.
/// Keeping them in separate structs (with `deny_unknown_fields`) makes a config that
/// mixes the two a hard error rather than a set of silently ignored keys.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfOnlineBuild {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) distance: SimilarityMeasure,
    pub(crate) dim: usize,
    /// A cluster is split once it holds strictly more than this many points.
    pub(crate) split_threshold: usize,
    /// Points inserted per batch; a single insert is a batch of one, so `1` is
    /// the reference semantics. Larger values route each batch in parallel and
    /// split every cluster that overflowed with one joint k-means rather than
    /// one bisection at a time; a few thousand matches how a real writer
    /// arrives. That joint split changes the partition, and therefore recall.
    #[serde(default = "default_batch_size")]
    pub(crate) batch_size: usize,
    /// Hard cap on live clusters. Omit (or `null`) for uncapped, data-driven growth.
    #[serde(default)]
    pub(crate) max_clusters: Option<usize>,
    /// Initial centroids produced by a light k-means over the corpus prefix.
    #[serde(default = "default_warmup_centroids")]
    pub(crate) warmup_centroids: usize,
    /// Leading corpus points used for the warmup clustering.
    #[serde(default = "default_warmup_points")]
    pub(crate) warmup_points: usize,
    /// Lloyd iterations for the warmup clustering.
    #[serde(default = "default_warmup_iters")]
    pub(crate) warmup_iters: usize,
    /// Centroid-graph search-list size used to route each inserted point.
    #[serde(default = "default_assign_l")]
    pub(crate) assign_l: usize,
    /// Lloyd iterations for the 2-means run at each split.
    #[serde(default = "default_two_means_iters")]
    pub(crate) two_means_iters: usize,
    /// Nearest clusters pooled as reassignment candidates when a cluster splits (`s`).
    #[serde(default = "default_reassign_neighbors")]
    pub(crate) reassign_neighbors: usize,
    /// A cluster is retired once deletes drop it below this many points: it
    /// leaves the centroid graph and its members are scattered onto their
    /// nearest surviving clusters. `0` disables merging, which is the only
    /// sensible setting for a run that never deletes. Must leave hysteresis
    /// below `split_threshold`: `2 * merge_threshold <= split_threshold`, so a
    /// freshly split cluster is not immediately a merge candidate.
    #[serde(default)]
    pub(crate) merge_threshold: usize,
    /// Floor on live clusters that merging will not go below.
    #[serde(default = "default_min_clusters")]
    pub(crate) min_clusters: usize,
    /// Search-list size selecting those neighbors. Resolved during validation to
    /// `max(reassign_neighbors, assign_l)` when omitted, so the effective value is
    /// always recorded in the job's serialized input.
    #[serde(default)]
    pub(crate) reassign_l: Option<usize>,
    /// Headroom multiplier for the centroid id budget, which is derived from the
    /// corpus size at build time (every split permanently retires one id).
    #[serde(default = "default_capacity_mult")]
    pub(crate) capacity_mult: usize,
    /// L2-normalize the two child centroids after a split (for unit-norm corpora).
    #[serde(default)]
    pub(crate) normalize: bool,
    /// Pruned out-degree of the centroid graph (`R`).
    pub(crate) graph_degree: usize,
    /// Maximum out-degree as a multiple of `graph_degree` (slack, `>= 1.0`).
    pub(crate) graph_slack: f32,
    /// Search-list size used during centroid-graph construction (`L`).
    pub(crate) graph_l_build: usize,
    /// Pruning alpha (`>= 1.0`).
    pub(crate) graph_alpha: f32,
    pub(crate) num_threads: usize,
    /// RNG seed for warmup sampling and split seeding (for reproducibility).
    pub(crate) seed: u64,
    /// Path prefix to save the index under (without the `.graphivf_*` suffix).
    pub(crate) save_path: String,
    /// Optional path for the per-split telemetry CSV. Omit to skip writing it.
    #[serde(default)]
    pub(crate) telemetry_csv: Option<String>,
}

const fn default_warmup_centroids() -> usize {
    100
}
const fn default_warmup_points() -> usize {
    10_000
}
const fn default_warmup_iters() -> usize {
    15
}
const fn default_assign_l() -> usize {
    64
}
const fn default_two_means_iters() -> usize {
    12
}
const fn default_reassign_neighbors() -> usize {
    8
}
const fn default_capacity_mult() -> usize {
    3
}
const fn default_batch_size() -> usize {
    1
}
const fn default_min_clusters() -> usize {
    1
}

/// An online build driven by a BigANN streaming runbook rather than by a single
/// pass over the corpus.
///
/// The runbook's stages are replayed against a live
/// `diskann_graphivf::OnlineClusterer`: insert stages feed corpus rows in,
/// delete stages take them back out, and search stages measure recall against
/// the stage's own groundtruth. The index is flushed once the runbook ends, so a
/// job may still attach a `search_phase` to measure the on-disk result.
///
/// The clustering knobs live in [`build`](Self::build), unchanged from a plain
/// online build; only `merge_threshold` there has any effect in a run that never
/// deletes.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfOnlineRunbook {
    /// Corpus, clustering, and save-path configuration. Its `batch_size` sets
    /// how a stage's range is sub-batched, not the stage boundaries themselves.
    pub(crate) build: GraphIvfOnlineBuild,
    pub(crate) runbook: GraphIvfRunbookConfig,
    pub(crate) search: GraphIvfRunbookSearch,
}

/// Which runbook to replay, and where its per-stage groundtruth lives.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfRunbookConfig {
    pub(crate) runbook_path: InputFile,
    /// Key of the dataset section within the runbook YAML.
    pub(crate) dataset_name: String,
    /// Directory holding the groundtruth file each search stage names.
    pub(crate) gt_directory: String,
    /// Resolved during validation; never part of the serialized config.
    #[serde(skip)]
    pub(crate) resolved_gt_directory: Option<std::path::PathBuf>,
}

/// How every search stage of the runbook is measured.
///
/// Searches run single-threaded against the in-memory clusterer, so the reported
/// latency is a clean per-query cost rather than a throughput figure; the point
/// of a runbook run is recall as the index churns.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfRunbookSearch {
    pub(crate) queries: InputFile,
    /// Numbers of nearest clusters to probe — one sweep per value, at every
    /// search stage.
    pub(crate) nlist: Vec<usize>,
    /// Search-list size for the centroid graph search (`>= nlist`).
    pub(crate) centroid_search_l: usize,
    pub(crate) recall_at: RecallAt,
}

/// Serializable mirror of `diskann_graphivf::AssignMethod` (the benchmark's
/// `inputs` layer is compiled without the optional `graph-ivf` dependency, so
/// it cannot name that type directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum AssignMethodConfig {
    /// Exact brute-force nearest-centroid assignment.
    #[default]
    Exact,
    /// Graph-accelerated approximate assignment.
    Graph {
        /// Rebuild the centroid graph every this many iterations.
        rebuild_every: usize,
        /// Number of graph candidates to re-rank exactly per point.
        rerank: usize,
    },
}

/// Serializable mirror of `diskann_graphivf::EmptyClusterPolicy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) enum EmptyClusterConfig {
    /// Zero the centroid (legacy behavior).
    Zero,
    /// Keep the centroid at its previous position.
    #[default]
    PreserveOld,
    /// Move the centroid onto the farthest assigned point.
    ReseedFarthest,
}

/// Search phase configuration.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GraphIvfSearchPhase {
    pub(crate) queries: InputFile,
    pub(crate) groundtruth: InputFile,
    pub(crate) num_threads: usize,
    /// Numbers of nearest clusters to probe — one search sweep per value.
    pub(crate) nlist: Vec<usize>,
    /// Search-list size for the centroid graph search (`>= nlist`).
    pub(crate) centroid_search_l: usize,
    pub(crate) recall_at: RecallAt,
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
            GraphIvfSource::Static(build) => build.validate(checker)?,
            GraphIvfSource::Online(online) => online.validate(checker)?,
            GraphIvfSource::OnlineRunbook(runbook) => runbook.validate(checker)?,
        }
        if let Some(search_phase) = &mut self.search_phase {
            search_phase.validate(checker)?;
        }
        Ok(())
    }
}

/// Check that an index `save_path` prefix is usable: absolute (the runner's
/// `output_directory` redirection is not supported for index prefixes) and with an
/// existing parent directory. Overwriting an existing index is allowed.
fn validate_save_path(save_path: &str, checker: &Checker) -> anyhow::Result<()> {
    // Relative save path with respect to output directory is not supported.
    if checker.output_directory().is_some() {
        anyhow::bail!("relative save_path with respect to output_directory is not supported");
    }

    match Path::new(save_path).parent() {
        Some(parent_dir) => {
            let parent_str = parent_dir.to_string_lossy();
            if !parent_str.is_empty() && !parent_dir.is_dir() {
                anyhow::bail!(
                    "parent directory - {} of save_path - {} does not exist",
                    parent_str,
                    save_path
                );
            }
        }
        None => anyhow::bail!("invalid save_path - {}", save_path),
    }
    Ok(())
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

impl GraphIvfStaticBuild {
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

        validate_save_path(&self.save_path, checker)?;

        Ok(())
    }
}

impl GraphIvfOnlineBuild {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.resolve(checker).context("invalid data file")?;

        if self.dim == 0 {
            anyhow::bail!("dim must be positive");
        }
        // A cluster of 1 point cannot be split into two non-empty children.
        if self.split_threshold < 2 {
            anyhow::bail!("split_threshold must be >= 2");
        }
        if self.batch_size == 0 {
            anyhow::bail!("batch_size must be positive");
        }
        if self.warmup_centroids == 0 {
            anyhow::bail!("warmup_centroids must be positive");
        }
        if self.warmup_points < self.warmup_centroids {
            anyhow::bail!("warmup_points must be >= warmup_centroids");
        }
        if self.assign_l == 0 {
            anyhow::bail!("assign_l must be positive");
        }
        if self.two_means_iters == 0 {
            anyhow::bail!("two_means_iters must be positive");
        }
        if self.reassign_neighbors == 0 {
            anyhow::bail!("reassign_neighbors must be positive");
        }
        if self.capacity_mult == 0 {
            anyhow::bail!("capacity_mult must be positive");
        }
        if self.min_clusters == 0 {
            anyhow::bail!("min_clusters must be positive");
        }
        // Without hysteresis a cluster that just split is already small enough
        // to be merged back, so a single delete could undo the split.
        if self.merge_threshold > 0 && 2 * self.merge_threshold > self.split_threshold {
            anyhow::bail!(
                "merge_threshold ({}) leaves no hysteresis below split_threshold ({}); \
                 require 2 * merge_threshold <= split_threshold",
                self.merge_threshold,
                self.split_threshold
            );
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

        // The seed already produces `warmup_centroids` live clusters, so a cap at or
        // below that would be violated before the first insert.
        if let Some(max_clusters) = self.max_clusters {
            if max_clusters <= self.warmup_centroids {
                anyhow::bail!(
                    "max_clusters ({}) must exceed warmup_centroids ({})",
                    max_clusters,
                    self.warmup_centroids
                );
            }
        }

        // Stored rows are written verbatim, so there is no opportunity to normalize the
        // corpus during an online build; it must already be unit-norm.
        if self.distance == SimilarityMeasure::Cosine {
            anyhow::bail!(
                "online builds store corpus rows verbatim and cannot normalize them; \
                 pre-normalize the corpus and use `cosine_normalized`"
            );
        }

        // Resolve the documented default now so the effective value is recorded in the
        // job's serialized input rather than being implied by the backend.
        let reassign_l = self
            .reassign_l
            .unwrap_or_else(|| self.reassign_neighbors.max(self.assign_l));
        if reassign_l == 0 {
            anyhow::bail!("reassign_l must be positive");
        }
        self.reassign_l = Some(reassign_l);

        validate_save_path(&self.save_path, checker)?;

        Ok(())
    }

    /// The reassignment search-list size, falling back to the documented default so an
    /// unvalidated struct still reports the value the backend would use.
    pub(crate) fn effective_reassign_l(&self) -> usize {
        self.reassign_l
            .unwrap_or_else(|| self.reassign_neighbors.max(self.assign_l))
    }
}

/// Shared validation for all search configs that have `nlist`, `centroid_search_l`,
/// and `recall_at`. Called by both [`GraphIvfRunbookSearch`] and [`GraphIvfSearchPhase`].
fn validate_nlist_and_recall(
    nlist: &[usize],
    centroid_search_l: usize,
    recall_at: &mut RecallAt,
) -> Result<(), anyhow::Error> {
    if nlist.is_empty() {
        anyhow::bail!("nlist must have at least one value");
    }
    if nlist.contains(&0) {
        anyhow::bail!("nlist values must be positive");
    }
    if centroid_search_l == 0 {
        anyhow::bail!("centroid_search_l must be positive");
    }
    recall_at.validate()
}

impl GraphIvfOnlineRunbook {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.build.validate(checker)?;
        self.runbook.validate(checker)?;
        self.search.validate(checker)?;
        Ok(())
    }
}

impl GraphIvfRunbookConfig {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.runbook_path
            .resolve(checker)
            .context("invalid runbook file")?;

        // Mirrors `InputFile::resolve`: an absolute path must exist as given, a
        // relative one is looked up under each search directory in turn.
        let gt_path = Path::new(&self.gt_directory);
        let resolved = if gt_path.is_dir() {
            Some(gt_path.to_path_buf())
        } else if gt_path.is_absolute() {
            None
        } else {
            checker
                .search_directories()
                .iter()
                .map(|dir| dir.join(gt_path))
                .find(|candidate| candidate.is_dir())
        };

        self.resolved_gt_directory = Some(resolved.ok_or_else(|| {
            anyhow::anyhow!(
                "could not find groundtruth directory \"{}\" in the search directories: {:?}",
                self.gt_directory,
                checker.search_directories()
            )
        })?);

        Ok(())
    }
}

impl GraphIvfRunbookSearch {
    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.queries
            .resolve(checker)
            .context("invalid queries file")?;
        validate_nlist_and_recall(&self.nlist, self.centroid_search_l, &mut self.recall_at)
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
        validate_nlist_and_recall(&self.nlist, self.centroid_search_l, &mut self.recall_at)?;
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
        let build = GraphIvfStaticBuild {
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
            assign_method: AssignMethodConfig::Exact,
            empty_clusters: EmptyClusterConfig::PreserveOld,
            save_path: "sample_graphivf_index".to_string(),
        };

        let search = GraphIvfSearchPhase {
            queries: InputFile::new("path/to/queries.fbin"),
            groundtruth: InputFile::new("path/to/groundtruth.ibin"),
            num_threads: 8,
            nlist: vec![8, 16, 32],
            centroid_search_l: 64,
            recall_at: RecallAt::new(vec![10]),
            distance: SimilarityMeasure::SquaredL2,
        };

        Self {
            source: GraphIvfSource::Static(build),
            search_phase: Some(search),
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
            GraphIvfSource::Static(build) => build.fmt(f),
            GraphIvfSource::Online(online) => online.fmt(f),
            GraphIvfSource::OnlineRunbook(runbook) => runbook.fmt(f),
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

impl fmt::Display for GraphIvfStaticBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Static Build")?;
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

impl fmt::Display for GraphIvfOnlineBuild {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Online Build")?;
        write_field!(f, "Data Type", self.data_type)?;
        write_field!(f, "Data File", self.data.display())?;
        write_field!(f, "Distance", self.distance)?;
        write_field!(f, "Dim", self.dim)?;
        write_field!(f, "Split Threshold", self.split_threshold)?;
        match self.max_clusters {
            Some(m) => write_field!(f, "Max Clusters", m)?,
            None => write_field!(f, "Max Clusters", "uncapped")?,
        }
        write_field!(f, "Warmup Centroids", self.warmup_centroids)?;
        write_field!(f, "Warmup Points", self.warmup_points)?;
        write_field!(f, "Warmup Iters", self.warmup_iters)?;
        write_field!(f, "Assign L", self.assign_l)?;
        write_field!(f, "Two Means Iters", self.two_means_iters)?;
        write_field!(f, "Reassign S", self.reassign_neighbors)?;
        write_field!(f, "Reassign L", self.effective_reassign_l())?;
        if self.merge_threshold > 0 {
            write_field!(f, "Merge Threshold", self.merge_threshold)?;
            write_field!(f, "Min Clusters", self.min_clusters)?;
        } else {
            write_field!(f, "Merge Threshold", "disabled")?;
        }
        write_field!(f, "Capacity Mult", self.capacity_mult)?;
        write_field!(f, "Normalize", self.normalize)?;
        write_field!(f, "Graph Degree", self.graph_degree)?;
        write_field!(f, "Graph Slack", self.graph_slack)?;
        write_field!(f, "Graph L Build", self.graph_l_build)?;
        write_field!(f, "Graph Alpha", self.graph_alpha)?;
        write_field!(f, "Build Threads", self.num_threads)?;
        write_field!(f, "Seed", self.seed)?;
        write_field!(f, "Save Path", self.save_path)?;
        if let Some(csv) = &self.telemetry_csv {
            write_field!(f, "Telemetry CSV", csv)?;
        }
        Ok(())
    }
}

/// Comma-separated sweep values, rendered like any other labelled field.
struct CommaList<'a, T>(&'a [T]);

impl<T: fmt::Display> fmt::Display for CommaList<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{v}")?;
        }
        Ok(())
    }
}

impl fmt::Display for GraphIvfOnlineRunbook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.build.fmt(f)?;
        self.runbook.fmt(f)?;
        self.search.fmt(f)
    }
}

impl fmt::Display for GraphIvfRunbookConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Runbook")?;
        write_field!(f, "Runbook", self.runbook_path.display())?;
        write_field!(f, "Dataset", self.dataset_name)?;
        write_field!(f, "GT Directory", self.gt_directory)?;
        Ok(())
    }
}

impl fmt::Display for GraphIvfRunbookSearch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Runbook Search")?;
        write_field!(f, "Queries", self.queries.display())?;
        write_field!(f, "NList", CommaList(&self.nlist))?;
        write_field!(f, "Centroid L", self.centroid_search_l)?;
        write_field!(f, "Recall@", self.recall_at)?;
        Ok(())
    }
}

impl fmt::Display for GraphIvfSearchPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Graph-IVF Search Phase")?;
        write_field!(f, "Queries", self.queries.display())?;
        write_field!(f, "Groundtruth", self.groundtruth.display())?;
        write_field!(f, "NList", CommaList(&self.nlist))?;
        write_field!(f, "Centroid L", self.centroid_search_l)?;
        write_field!(f, "Recall@", self.recall_at)?;
        write_field!(f, "Threads", self.num_threads)?;
        write_field!(f, "Distance", self.distance)?;
        Ok(())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// A checker rooted at `dir`, which every test uses as both the search directory and
    /// the parent of `save_path`. No output directory: index prefixes must be absolute.
    fn checker(dir: &Path) -> Checker {
        Checker::new(vec![dir.to_path_buf()], None)
    }

    /// Create an empty `corpus.bin` inside `dir` so `InputFile::resolve` finds it.
    fn touch_corpus(dir: &Path) {
        std::fs::write(dir.join("corpus.bin"), []).unwrap();
    }

    /// A minimal-but-complete online source, as JSON. `extra` is spliced in verbatim so
    /// tests can add or override keys.
    fn online_json(dir: &Path, extra: &str) -> serde_json::Value {
        let save_path = dir.join("index").to_string_lossy().replace('\\', "/");
        serde_json::from_str(&format!(
            r#"{{
                "graph-ivf-source": "Online",
                "data_type": "minmax8",
                "data": "corpus.bin",
                "distance": "squared_l2",
                "dim": 384,
                "split_threshold": 759,
                "graph_degree": 32,
                "graph_slack": 1.2,
                "graph_l_build": 64,
                "graph_alpha": 1.2,
                "num_threads": 16,
                "seed": 0,
                "save_path": "{save_path}"
                {extra}
            }}"#
        ))
        .unwrap()
    }

    fn parse_online(value: serde_json::Value) -> Result<GraphIvfSource, serde_json::Error> {
        serde_json::from_value(value)
    }

    /// Parse and validate, returning the online build on success.
    fn validated(dir: &Path, extra: &str) -> anyhow::Result<GraphIvfOnlineBuild> {
        touch_corpus(dir);
        let mut source = parse_online(online_json(dir, extra)).unwrap();
        let mut checker = checker(dir);
        match &mut source {
            GraphIvfSource::Online(online) => {
                online.validate(&mut checker)?;
                Ok(online.clone())
            }
            other => panic!("expected an online source, got {other:?}"),
        }
    }

    #[test]
    fn online_source_tag_selects_online_variant() {
        let dir = tempfile::tempdir().unwrap();
        let source = parse_online(online_json(dir.path(), "")).unwrap();
        assert!(
            matches!(source, GraphIvfSource::Online(_)),
            "the `Online` tag must not fall through to another variant"
        );
    }

    #[test]
    fn online_defaults_match_documented_values() {
        let dir = tempfile::tempdir().unwrap();
        let online = validated(dir.path(), "").unwrap();
        assert_eq!(online.warmup_centroids, 100);
        assert_eq!(online.warmup_points, 10_000);
        assert_eq!(online.warmup_iters, 15);
        assert_eq!(online.assign_l, 64);
        assert_eq!(online.two_means_iters, 12);
        assert_eq!(online.reassign_neighbors, 8);
        assert_eq!(online.capacity_mult, 3);
        assert!(!online.normalize);
        assert_eq!(online.max_clusters, None, "omitted means uncapped");
        assert_eq!(online.telemetry_csv, None);
    }

    #[test]
    fn validation_resolves_reassign_l_to_the_documented_default() {
        let dir = tempfile::tempdir().unwrap();

        // Default is max(reassign_neighbors, assign_l); here assign_l wins.
        let online = validated(dir.path(), r#", "reassign_neighbors": 32"#).unwrap();
        assert_eq!(online.effective_reassign_l(), 64);
        assert_eq!(
            online.reassign_l,
            Some(64),
            "validation must write the effective value back so it is recorded in the \
             serialized input"
        );

        // ...and here reassign_neighbors wins.
        let online =
            validated(dir.path(), r#", "reassign_neighbors": 128, "assign_l": 16"#).unwrap();
        assert_eq!(online.reassign_l, Some(128));

        // An explicit value is never overwritten.
        let online = validated(
            dir.path(),
            r#", "reassign_neighbors": 32, "reassign_l": 256"#,
        )
        .unwrap();
        assert_eq!(online.reassign_l, Some(256));
    }

    #[test]
    fn online_rejects_static_only_keys() {
        let dir = tempfile::tempdir().unwrap();
        // `num_clusters` belongs to a batch build and has no online meaning. Without
        // `deny_unknown_fields` this would be silently dropped.
        for key in [
            r#", "num_clusters": 1024"#,
            r#", "sample_size": 100000"#,
            r#", "kmeans_iters": 5"#,
            r#", "empty_clusters": "PreserveOld""#,
        ] {
            let err = parse_online(online_json(dir.path(), key)).unwrap_err();
            assert!(
                err.to_string().contains("unknown field"),
                "expected an unknown-field error for {key}, got: {err}"
            );
        }
    }

    #[test]
    fn static_rejects_online_only_keys() {
        let dir = tempfile::tempdir().unwrap();
        let save_path = dir
            .path()
            .join("index")
            .to_string_lossy()
            .replace('\\', "/");
        let json = format!(
            r#"{{
                "graph-ivf-source": "Static",
                "data_type": "float32",
                "data": "corpus.bin",
                "distance": "squared_l2",
                "dim": 384,
                "num_clusters": 1024,
                "sample_size": 100000,
                "kmeans_iters": 5,
                "assign_l": 32,
                "graph_degree": 32,
                "graph_slack": 1.2,
                "graph_l_build": 64,
                "graph_alpha": 1.2,
                "num_threads": 8,
                "seed": 0,
                "save_path": "{save_path}",
                "split_threshold": 759
            }}"#
        );
        let err = serde_json::from_str::<GraphIvfSource>(&json).unwrap_err();
        assert!(
            err.to_string().contains("unknown field"),
            "a batch build must reject `split_threshold`, got: {err}"
        );
    }

    #[test]
    fn misspelled_key_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let err =
            parse_online(online_json(dir.path(), r#", "reasign_neighbors": 32"#)).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }

    #[test]
    fn split_threshold_must_allow_two_children() {
        let dir = tempfile::tempdir().unwrap();
        touch_corpus(dir.path());
        for threshold in [0, 1] {
            let mut value = online_json(dir.path(), "");
            value["split_threshold"] = serde_json::json!(threshold);
            let mut source = parse_online(value).unwrap();
            let GraphIvfSource::Online(online) = &mut source else {
                panic!("expected online source");
            };
            let err = online.validate(&mut checker(dir.path())).unwrap_err();
            assert!(
                err.to_string().contains("split_threshold must be >= 2"),
                "a threshold of {threshold} cannot be split into two clusters, got: {err}"
            );
        }
    }

    #[test]
    fn max_clusters_must_exceed_warmup_centroids() {
        let dir = tempfile::tempdir().unwrap();
        // The seed already creates `warmup_centroids` clusters, so a cap at or below
        // that is unsatisfiable before the first insert.
        let err = validated(
            dir.path(),
            r#", "warmup_centroids": 100, "max_clusters": 100"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("must exceed warmup_centroids"),
            "got: {err}"
        );

        // One above the warmup is the smallest cap that can ever split.
        let online = validated(
            dir.path(),
            r#", "warmup_centroids": 100, "max_clusters": 101"#,
        )
        .unwrap();
        assert_eq!(online.max_clusters, Some(101));
    }

    #[test]
    fn warmup_points_must_cover_warmup_centroids() {
        let dir = tempfile::tempdir().unwrap();
        let err = validated(
            dir.path(),
            r#", "warmup_centroids": 500, "warmup_points": 100"#,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("warmup_points must be >= warmup_centroids"),
            "got: {err}"
        );
    }

    #[test]
    fn online_rejects_unnormalized_cosine() {
        let dir = tempfile::tempdir().unwrap();
        let mut value = online_json(dir.path(), "");
        value["distance"] = serde_json::json!("cosine");
        touch_corpus(dir.path());
        let mut source = parse_online(value).unwrap();
        let GraphIvfSource::Online(online) = &mut source else {
            panic!("expected online source");
        };
        let err = online.validate(&mut checker(dir.path())).unwrap_err();
        assert!(
            err.to_string().contains("cosine_normalized"),
            "the error should point at the supported alternative, got: {err}"
        );

        // `cosine_normalized` is the supported way to express a cosine index.
        let mut value = online_json(dir.path(), "");
        value["distance"] = serde_json::json!("cosine_normalized");
        let mut source = parse_online(value).unwrap();
        let GraphIvfSource::Online(online) = &mut source else {
            panic!("expected online source");
        };
        online.validate(&mut checker(dir.path())).unwrap();
    }

    #[test]
    fn zero_valued_knobs_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        for (key, needle) in [
            ("dim", "dim must be positive"),
            ("warmup_centroids", "warmup_centroids must be positive"),
            ("assign_l", "assign_l must be positive"),
            ("two_means_iters", "two_means_iters must be positive"),
            ("reassign_neighbors", "reassign_neighbors must be positive"),
            ("capacity_mult", "capacity_mult must be positive"),
            ("graph_degree", "graph_degree must be positive"),
            ("graph_l_build", "graph_l_build must be positive"),
            ("num_threads", "num_threads must be positive"),
        ] {
            let mut value = online_json(dir.path(), "");
            value[key] = serde_json::json!(0);
            touch_corpus(dir.path());
            let mut source = parse_online(value).unwrap();
            let GraphIvfSource::Online(online) = &mut source else {
                panic!("expected online source");
            };
            let err = online.validate(&mut checker(dir.path())).unwrap_err();
            assert!(
                err.to_string().contains(needle),
                "setting {key} to 0 should report {needle:?}, got: {err}"
            );
        }
    }

    #[test]
    fn save_path_parent_must_exist() {
        let dir = tempfile::tempdir().unwrap();
        let mut value = online_json(dir.path(), "");
        value["save_path"] = serde_json::json!(dir
            .path()
            .join("no_such_dir/index")
            .to_string_lossy()
            .replace('\\', "/"));
        touch_corpus(dir.path());
        let mut source = parse_online(value).unwrap();
        let GraphIvfSource::Online(online) = &mut source else {
            panic!("expected online source");
        };
        let err = online.validate(&mut checker(dir.path())).unwrap_err();
        assert!(err.to_string().contains("does not exist"), "got: {err}");
    }

    #[test]
    fn missing_corpus_is_reported_as_such() {
        let dir = tempfile::tempdir().unwrap();
        // Deliberately do not create `corpus.bin`.
        let mut source = parse_online(online_json(dir.path(), "")).unwrap();
        let GraphIvfSource::Online(online) = &mut source else {
            panic!("expected online source");
        };
        let err = online.validate(&mut checker(dir.path())).unwrap_err();
        assert!(err.to_string().contains("invalid data file"), "got: {err}");
    }

    #[test]
    fn validated_online_source_round_trips_through_json() {
        let dir = tempfile::tempdir().unwrap();
        let online = validated(
            dir.path(),
            r#", "reassign_neighbors": 32, "normalize": true"#,
        )
        .unwrap();

        // The recorded input must replay identically: this is what makes a run
        // reproducible from its own output.
        let source = GraphIvfSource::Online(online.clone());
        let encoded = serde_json::to_value(&source).unwrap();
        let decoded: GraphIvfSource = serde_json::from_value(encoded).unwrap();
        let GraphIvfSource::Online(decoded) = decoded else {
            panic!("round trip changed the source variant");
        };
        assert_eq!(decoded.reassign_neighbors, 32);
        assert_eq!(decoded.reassign_l, online.reassign_l);
        assert!(decoded.normalize);
        assert_eq!(decoded.split_threshold, online.split_threshold);
        assert_eq!(decoded.seed, online.seed);
    }

    /// Parse a `recall_at` payload on its own, then validate it.
    fn recall_at(json: &str) -> anyhow::Result<RecallAt> {
        let mut parsed: RecallAt = serde_json::from_str(json)?;
        parsed.validate()?;
        Ok(parsed)
    }

    #[test]
    fn recall_at_accepts_a_scalar_or_a_list() {
        assert_eq!(recall_at("50").unwrap().iter().collect::<Vec<_>>(), [50]);
        assert_eq!(
            recall_at("[50, 1000]").unwrap().iter().collect::<Vec<_>>(),
            [50, 1000]
        );
    }

    #[test]
    fn recall_at_is_sorted_and_deduplicated() {
        // The deepest value sets the search depth, and ordering the rest by depth
        // keeps the reported columns stable however the config listed them.
        let parsed = recall_at("[1000, 50, 1000]").unwrap();
        assert_eq!(parsed.iter().collect::<Vec<_>>(), [50, 1000]);
        assert_eq!(parsed.max(), 1000);
    }

    #[test]
    fn recall_at_rejects_empty_and_zero() {
        assert!(recall_at("[]").is_err(), "no k means nothing to measure");
        assert!(recall_at("0").is_err());
        assert!(recall_at("[50, 0]").is_err());
    }

    #[test]
    fn recall_at_round_trips_in_the_form_it_was_written() {
        // A single value must stay a bare number so configs written against the
        // scalar field are reproduced unchanged from a run's own output.
        let one: RecallAt = serde_json::from_str("50").unwrap();
        assert_eq!(serde_json::to_string(&one).unwrap(), "50");

        let many: RecallAt = serde_json::from_str("[50,1000]").unwrap();
        assert_eq!(serde_json::to_string(&many).unwrap(), "[50,1000]");
    }
}
