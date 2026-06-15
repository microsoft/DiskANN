/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Benchmark for DocumentInsertStrategy which allows inserting Documents
//! (vector + attributes) into a DiskANN index built with DocumentProvider.
//! Also benchmarks filtered search using InlineBetaStrategy.

use std::io::Write;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use diskann::{
    graph::{
        glue, index::QueryLabelProvider, search::{AdaptiveL, Knn}, search_output_buffer, DiskANNIndex,
        SearchOutputBuffer,
        StartPointStrategy,
    },
    provider::DefaultContext,
    utils::VectorRepr,
    ANNError, ANNErrorKind,
};
use diskann_benchmark_core::{
    build,
    build::{Build, Parallelism},
    recall, search as search_api, tokio,
};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    registry::Registry,
    utils::{datatype::AsDataType, fmt, percentiles, MicroSeconds},
    Benchmark,
};
use diskann_label_filter::{
    attribute::{Attribute, AttributeValue},
    document::Document,
    encoded_attribute_provider::encoded_filter_expr::EncodedFilterExpr,
    encoded_attribute_provider::{
        document_insert_strategy::DocumentInsertStrategy, document_provider::DocumentProvider,
        roaring_attribute_store::RoaringAttributeStore,
    },
    filtered_query_label_provider::FilteredQueryLabelProvider,
    inline_beta_search::inline_beta_filter::InlineBetaStrategy,
    inline_beta_search::inverted_bitmap_evaluator::InvertedBitmapEvaluator,
    parser::format::Document as LabelDocument,
    query::FilteredQuery,
    read_and_parse_queries, read_baselabels,
    traits::attribute_store::AttributeStore,
    ASTExpr,
};

use diskann_providers::{
    model::graph::provider::{
        async_::{
            common::{self, NoStore, TableBasedDeletes},
            inmem::{
                CreateFullPrecision, DefaultProvider, DefaultProviderParameters, SetStartPoints,
            },
        },
        layers::BetaFilter,
    },
    utils::Timer,
};
use diskann_utils::views::Matrix;
use diskann_utils::views::MatrixView;
use diskann_utils::{future::AsyncFriendly, sampling::medoid::ComputeMedoid};
use diskann_vector::distance::SquaredL2;
use diskann_vector::PureDistanceFunction;
#[cfg(feature = "precompute-query-bitmaps")]
use rayon::prelude::*;
#[cfg(feature = "precompute-query-bitmaps")]
use rayon::ThreadPoolBuilder;
use roaring::RoaringTreemap;
use serde::Serialize;

use crate::{
    backend::index::build::ProgressMeter,
    inputs::{
        document_index::{
            AdaptiveLConfig, DocumentBuildParams, DocumentIndexBuild, DocumentSearchAlgorithm,
            DocumentSearchParams,
        },
        graph_index::GraphSearch,
    },
    utils::{
        self,
        datafiles::{self, BinFile},
        recall::RecallMetrics,
    },
};

/// Register the document index benchmarks.
pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register::<DocumentIndexJob<f32>>(
        "document-index-build-f32",
        DocumentIndexJob::<f32>::new(),
    )?;
    registry.register::<DocumentIndexJob<u8>>(
        "document-index-build-u8",
        DocumentIndexJob::<u8>::new(),
    )?;
    Ok(())
}

/// Document index benchmark job.
pub(super) struct DocumentIndexJob<T> {
    _type: std::marker::PhantomData<T>,
}

impl<T> DocumentIndexJob<T> {
    pub(crate) fn new() -> Self {
        Self {
            _type: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for DocumentIndexJob<T>
where
    T: VectorRepr
        + diskann::graph::SampleableForStart
        + diskann_utils::sampling::WithApproximateNorm
        + AsDataType
        + 'static,
    for<'b> diskann_vector::distance::SquaredL2: PureDistanceFunction<&'b [T], &'b [T]>,
{
    type Input = DocumentIndexBuild;
    type Output = DocumentIndexStats;

    fn try_match(&self, input: &Self::Input) -> std::result::Result<MatchScore, FailureScore> {
        utils::match_data_type::<T>(input.build.data_type)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Self::Input>,
    ) -> std::fmt::Result {
        match input {
            Some(arg) => write!(f, "{}", T::describe(arg.build.data_type)),
            None => write!(f, "{}", T::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: diskann_benchmark_runner::Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        DocumentIndexJob::<T>::run(input, output)
    }
}

/// Convert a HashMap<String, AttributeValue> to Vec<Attribute>
fn hashmap_to_attributes(map: std::collections::HashMap<String, AttributeValue>) -> Vec<Attribute> {
    map.into_iter()
        .map(|(k, v)| Attribute::from_value(k, v))
        .collect()
}

fn find_medoid_index<T>(x: MatrixView<'_, T>, y: &[T]) -> Option<usize>
where
    for<'a> diskann_vector::distance::SquaredL2: PureDistanceFunction<&'a [T], &'a [T], f32>,
{
    let mut min_dist = f32::INFINITY;
    let mut min_ind = x.nrows();
    for (i, row) in x.row_iter().enumerate() {
        let dist = SquaredL2::evaluate(row, y);
        if dist < min_dist {
            min_dist = dist;
            min_ind = i;
        }
    }

    // No closest neighbor found.
    if min_ind == x.nrows() {
        None
    } else {
        Some(min_ind)
    }
}

/// Compute the index of the row closest to the medoid (centroid) of the data.
fn compute_medoid_index<T>(data: &Matrix<T>) -> anyhow::Result<usize>
where
    T: bytemuck::Pod + Copy + 'static + ComputeMedoid,
    for<'a> diskann_vector::distance::SquaredL2: PureDistanceFunction<&'a [T], &'a [T], f32>,
{
    let dim = data.ncols();
    if dim == 0 || data.nrows() == 0 {
        return Ok(0);
    }

    // returns row closes to centroid.
    let medoid = T::compute_medoid(data.as_view());

    find_medoid_index(data.as_view(), medoid.as_slice())
        .ok_or_else(|| anyhow::anyhow!("Failed to find medoid index: no closest row found"))
}

#[derive(Debug, Clone)]
struct BitmapQueryLabelProvider {
    #[cfg(not(feature = "precompute-query-bitmaps"))]
    matches: RoaringTreemap,

    #[cfg(feature = "precompute-query-bitmaps")]
    query_index: usize,
    #[cfg(feature = "precompute-query-bitmaps")]
    precomputed_bitmaps: Arc<[RoaringTreemap]>,
}

#[cfg(not(feature = "precompute-query-bitmaps"))]
impl Default for BitmapQueryLabelProvider {
    fn default() -> Self {
        Self {
            matches: RoaringTreemap::new(),
        }
    }
}

#[cfg(feature = "precompute-query-bitmaps")]
impl Default for BitmapQueryLabelProvider {
    fn default() -> Self {
        Self {
            query_index: 0,
            precomputed_bitmaps: Arc::from([]),
        }
    }
}

impl BitmapQueryLabelProvider {
    #[cfg(not(feature = "precompute-query-bitmaps"))]
    fn from_ast(
        expr: &ASTExpr,
        attribute_store: &RoaringAttributeStore<u32>,
        universe: &RoaringTreemap,
    ) -> anyhow::Result<Self> {
        let encoded_filter = EncodedFilterExpr::from_attribute_store(expr, attribute_store)
            .context("failed to encode filter expression")?;

        let evaluator = InvertedBitmapEvaluator::new(attribute_store, universe);
        let matches = evaluator
            .evaluate(&encoded_filter)
            .context("failed to evaluate encoded expression with inverted index")?;

        Ok(Self { matches })
    }
}

fn build_label_attribute_store(
    labels: &[LabelDocument],
) -> anyhow::Result<(Arc<RoaringAttributeStore<u32>>, Vec<u32>)> {
    let store = Arc::new(RoaringAttributeStore::<u32>::new());
    let mut label_ids = Vec::with_capacity(labels.len());

    for label in labels {
        let id = u32::try_from(label.doc_id)
            .with_context(|| format!("label doc_id {} does not fit in u32", label.doc_id))?;
        let attrs = hashmap_to_attributes(label.flatten_metadata_with_separator(""));
        store
            .set_element(&id, &attrs)
            .with_context(|| format!("failed to set attributes for label id {}", id))?;
        label_ids.push(id);
    }

    Ok((store, label_ids))
}

impl QueryLabelProvider<u32> for BitmapQueryLabelProvider {
    fn is_match(&self, vec_id: u32) -> bool {
        #[cfg(not(feature = "precompute-query-bitmaps"))]
        {
            self.matches.contains(u64::from(vec_id))
        }

        #[cfg(feature = "precompute-query-bitmaps")]
        {
            self.precomputed_bitmaps[self.query_index].contains(u64::from(vec_id))
        }
    }
}

enum InlineLabelProviderSource {
    Ast {
        predicates: Arc<Vec<(usize, ASTExpr)>>,
        attribute_store: Arc<RoaringAttributeStore<u32>>,
    },
    Bitmap {
        #[cfg(not(feature = "precompute-query-bitmaps"))]
        predicates: Arc<Vec<(usize, ASTExpr)>>,
        #[cfg(not(feature = "precompute-query-bitmaps"))]
        attribute_store: Arc<RoaringAttributeStore<u32>>,
        #[cfg(not(feature = "precompute-query-bitmaps"))]
        label_universe: Arc<RoaringTreemap>,
        #[cfg(feature = "precompute-query-bitmaps")]
        precomputed_bitmaps: Arc<[RoaringTreemap]>,
    },
}

fn load_search_inputs<T>(
    search_input: &DocumentSearchParams,
    mut output: &mut dyn Output,
) -> anyhow::Result<(Matrix<T>, Vec<(usize, ASTExpr)>, Vec<Vec<u32>>)> 
where
    T: bytemuck::Pod + Copy + 'static,
{
    writeln!(output, "\nLoading query vectors...")?;
    let query_path: &Path = search_input.queries.as_ref();
    let queries: Matrix<T> = datafiles::load_dataset(BinFile(query_path))?;
    let num_queries = queries.nrows();
    writeln!(output, "  Loaded {} queries", num_queries)?;

    writeln!(output, "Loading query predicates...")?;
    let predicate_path: &Path = search_input.query_predicates.as_ref();
    let parsed_predicates = read_and_parse_queries(predicate_path)?;
    writeln!(output, "  Loaded {} predicates", parsed_predicates.len())?;

    if num_queries != parsed_predicates.len() {
        return Err(anyhow::anyhow!(
            "Mismatch: {} queries but {} predicates",
            num_queries,
            parsed_predicates.len()
        ));
    }

    writeln!(output, "Loading groundtruth...")?;
    let gt_path: &Path = search_input.groundtruth.as_ref();
    let groundtruth: Vec<Vec<u32>> = datafiles::load_range_groundtruth(BinFile(gt_path))?;
    writeln!(
        output,
        "  Loaded groundtruth with {} rows",
        groundtruth.len()
    )?;

    Ok((queries, parsed_predicates, groundtruth))
}

fn execute_search_runs(
    mut output: &mut dyn Output,
    search_input: &DocumentSearchParams,
    mut run_one: impl FnMut(NonZeroUsize, &GraphSearch, usize) -> anyhow::Result<SearchRunStats>,
) -> anyhow::Result<Vec<SearchRunStats>> {
    writeln!(
        output,
        "\nRunning filtered searches (beta={})...",
        search_input.beta
    )?;
    let mut search_results = Vec::new();

    for num_threads in &search_input.num_threads {
        for run in &search_input.runs {
            for &search_l in &run.search_l {
                writeln!(
                    output,
                    "  threads={}, search_n={}, search_l={}...",
                    num_threads, run.search_n, search_l
                )?;

                let search_run_result = run_one(*num_threads, run, search_l)?;

                writeln!(
                    output,
                    "    recall={:.4}, mean_qps={:.1}",
                    search_run_result.recall.average,
                    if search_run_result.qps.is_empty() {
                        0.0
                    } else {
                        search_run_result.qps.iter().sum::<f64>()
                            / search_run_result.qps.len() as f64
                    }
                )?;

                search_results.push(search_run_result);
            }
        }
    }

    Ok(search_results)
}

fn default_provider_parameters(
    build: &DocumentBuildParams,
    num_vectors: usize,
    dim: usize,
) -> DefaultProviderParameters {
    DefaultProviderParameters {
        max_points: num_vectors,
        frozen_points: diskann::utils::ONE,
        metric: build.distance.into(),
        dim,
        prefetch_lookahead: None,
        prefetch_cache_line_level: None,
        max_degree: build.max_degree as u32,
    }
}

impl<T> DocumentIndexJob<T> {
    fn run(input: &DocumentIndexBuild, mut output: &mut dyn Output) -> Result<DocumentIndexStats, anyhow::Error>
    where
        T: diskann::utils::VectorRepr
            + diskann::graph::SampleableForStart
            + diskann_utils::sampling::WithApproximateNorm
            + 'static,
        for<'b> diskann_vector::distance::SquaredL2: PureDistanceFunction<&'b [T], &'b [T]>,
    {
        let build = &input.build;

        // 1. Load vectors from data file in the original data type
        writeln!(output, "Loading vectors ({})...", build.data_type)?;
        let timer = std::time::Instant::now();
        let data_path: &Path = build.data.as_ref();
        writeln!(output, "Data path is: {}", data_path.to_string_lossy())?;
        let data: Matrix<T> = datafiles::load_dataset(BinFile(data_path))?;
        let data_load_time: MicroSeconds = timer.elapsed().into();
        let num_vectors = data.nrows();
        let dim = data.ncols();
        writeln!(
            output,
            "  Loaded {} vectors of dimension {}",
            num_vectors, dim
        )?;

        // 2. Load and parse labels from the data_labels file
        writeln!(output, "Loading labels...")?;
        let timer = std::time::Instant::now();
        let label_path: &Path = build.data_labels.as_ref();
        let labels = read_baselabels(label_path)?;
        let label_load_time: MicroSeconds = timer.elapsed().into();
        let label_count = labels.len();
        writeln!(output, "  Loaded {} label documents", label_count)?;

        if num_vectors != label_count {
            return Err(anyhow::anyhow!(
                "Mismatch: {} vectors but {} label documents",
                num_vectors,
                label_count
            ));
        }

        // Convert labels to attribute vectors
        let attributes: Vec<Vec<Attribute>> = labels
            .iter()
            .map(|doc| hashmap_to_attributes(doc.flatten_metadata_with_separator("")))
            .collect();

        let search_input = &input.search;
        let (build_time, insert_percentiles, search_results) = match search_input.search_algorithm {
            // DocumentProvider path: attributes embedded in the index, InlineBeta strategy.
            DocumentSearchAlgorithm::Auto | DocumentSearchAlgorithm::InlineBeta => {
                writeln!(output, "Creating index...")?;
                let params = default_provider_parameters(build, num_vectors, dim);
                let fp_precursor = CreateFullPrecision::<T>::new(dim, None);
                let inner_provider =
                    DefaultProvider::new_empty(params, fp_precursor, NoStore, TableBasedDeletes)?;

                let start_points = StartPointStrategy::Medoid
                    .compute(data.as_view())
                    .map_err(|e| anyhow::anyhow!("Failed to compute start points: {}", e))?;
                inner_provider.set_start_points(start_points.row_iter())?;

                let attribute_store = RoaringAttributeStore::<u32>::new();
                let medoid_idx = compute_medoid_index(&data)?;
                let start_point_id = num_vectors as u32;
                let default_attrs = vec![];
                let medoid_attrs = attributes.get(medoid_idx).unwrap_or(&default_attrs);
                attribute_store.set_element(&start_point_id, medoid_attrs)?;

                let doc_provider = DocumentProvider::new(inner_provider, attribute_store);
                let doc_index =
                    Arc::new(DiskANNIndex::new(build.build_config()?, doc_provider, None));

                writeln!(
                    output,
                    "Building index with {} vectors using {} threads...",
                    num_vectors, build.num_threads
                )?;
                let timer = Timer::new();

                let rt = tokio::runtime(build.num_threads)?;
                let data_arc = Arc::new(data);
                let attributes_arc = Arc::new(attributes);

                let builder = DocumentIndexBuilder::new(
                    doc_index.clone(),
                    data_arc,
                    attributes_arc,
                    DocumentInsertStrategy::new(common::FullPrecision),
                );
                let num_tasks = NonZeroUsize::new(build.num_threads).unwrap_or(diskann::utils::ONE);
                let parallelism = Parallelism::dynamic(diskann::utils::ONE, num_tasks);
                let build_results = build::build_tracked(
                    builder,
                    parallelism,
                    &rt,
                    Some(&ProgressMeter::new(output)),
                )?;
                let mut insert_latencies: Vec<MicroSeconds> = build_results
                    .take_output()
                    .into_iter()
                    .map(|r| r.latency)
                    .collect();

                let build_time: MicroSeconds = timer.elapsed().into();
                writeln!(output, "  Index built in {} s", build_time.as_seconds())?;
                writeln!(
                    output,
                    "  Peak memory usage: {:.3} GB",
                    timer.get_peak_memory_usage()
                )?;

                let insert_percentiles = percentiles::compute_percentiles(&mut insert_latencies)?;
                let (queries, parsed_predicates, groundtruth) =
                    load_search_inputs(search_input, output)?;
                let search_results =
                    execute_search_runs(output, search_input, |num_threads, run, search_l| {
                        run_filtered_search(
                            &doc_index,
                            &queries,
                            &parsed_predicates,
                            &groundtruth,
                            search_input.beta,
                            num_threads,
                            run.search_n,
                            search_l,
                            run.recall_k,
                            search_input.reps,
                        )
                    })?;

                (build_time, insert_percentiles, search_results)
            }

            // Plain provider path: attributes loaded from labels file at search time.
            algo @ (DocumentSearchAlgorithm::AstLabelProvider
                | DocumentSearchAlgorithm::Bitmap
                | DocumentSearchAlgorithm::BitmapInline) => {
                writeln!(output, "Creating index...")?;
                let params = default_provider_parameters(build, num_vectors, dim);
                let fp_precursor = CreateFullPrecision::<T>::new(dim, None);
                let inner_provider =
                    DefaultProvider::new_empty(params, fp_precursor, NoStore, TableBasedDeletes)?;

                let start_points = StartPointStrategy::Medoid
                    .compute(data.as_view())
                    .map_err(|e| anyhow::anyhow!("Failed to compute start points: {}", e))?;
                inner_provider.set_start_points(start_points.row_iter())?;

                let index = Arc::new(DiskANNIndex::new(build.build_config()?, inner_provider, None));

                writeln!(
                    output,
                    "Building index with {} vectors using {} threads...",
                    num_vectors, build.num_threads
                )?;
                let timer = Timer::new();

                let rt = tokio::runtime(build.num_threads)?;
                let data_arc = Arc::new(data);
                let builder =
                    DefaultIndexBuilder::new(index.clone(), data_arc, common::FullPrecision);
                let num_tasks = NonZeroUsize::new(build.num_threads).unwrap_or(diskann::utils::ONE);
                let parallelism = Parallelism::dynamic(diskann::utils::ONE, num_tasks);
                let build_results = build::build_tracked(
                    builder,
                    parallelism,
                    &rt,
                    Some(&ProgressMeter::new(output)),
                )?;
                let mut insert_latencies: Vec<MicroSeconds> = build_results
                    .take_output()
                    .into_iter()
                    .map(|r| r.latency)
                    .collect();

                let build_time: MicroSeconds = timer.elapsed().into();
                writeln!(output, "  Index built in {} s", build_time.as_seconds())?;
                writeln!(
                    output,
                    "  Peak memory usage: {:.3} GB",
                    timer.get_peak_memory_usage()
                )?;

                let insert_percentiles = percentiles::compute_percentiles(&mut insert_latencies)?;
                let (queries, parsed_predicates, groundtruth) =
                    load_search_inputs(search_input, output)?;
                let search_results =
                    execute_search_runs(output, search_input, |num_threads, run, search_l| {
                        match algo {
                            DocumentSearchAlgorithm::AstLabelProvider => {
                                run_filtered_search_with_label_provider(
                                    &index,
                                    &queries,
                                    &labels,
                                    &parsed_predicates,
                                    &groundtruth,
                                    search_input.adaptive_l.as_ref(),
                                    num_threads,
                                    run.search_n,
                                    search_l,
                                    run.recall_k,
                                    search_input.reps,
                                )
                            }
                            DocumentSearchAlgorithm::BitmapInline => {
                                run_bitmap_inline_search(
                                    &index,
                                    &queries,
                                    &labels,
                                    &groundtruth,
                                    &parsed_predicates,
                                    search_input.adaptive_l.as_ref(),
                                    num_threads,
                                    run.search_n,
                                    search_l,
                                    run.recall_k,
                                    search_input.reps,
                                )
                            }
                            _ => run_bitmap_filtered_search(
                                &index,
                                &queries,
                                &labels,
                                &groundtruth,
                                &parsed_predicates,
                                search_input.beta,
                                num_threads,
                                run.search_n,
                                search_l,
                                run.recall_k,
                                search_input.reps,
                            ),
                        }
                    })?;

                (build_time, insert_percentiles, search_results)
            }
        };

        let stats = DocumentIndexStats {
            num_vectors,
            dim,
            label_count,
            data_load_time,
            label_load_time,
            build_time,
            insert_latencies: insert_percentiles,
            search: search_results,
        };

        writeln!(output, "\n{}", stats)?;
        Ok(stats)
    }
}

/// Per-query output from [`FilteredSearcher::search`].
struct FilteredSearchOutput {
    distances: Vec<f32>,
    comparisons: u32,
    hops: u32,
}

/// Implements [`search_api::Search`] for parallelized inline-beta filtered search.
///
/// Each query is paired with a predicate at the same index in `predicates`. The
/// [`InlineBetaStrategy`] is used with a [`FilteredQuery`] containing the raw vector
/// and the predicate's [`ASTExpr`].
struct FilteredSearcher<DP, T>
where
    DP: diskann::provider::DataProvider,
{
    index: Arc<DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    predicates: Arc<Vec<(usize, ASTExpr)>>,
    beta: f32,
}

/// Implements [`search_api::Search`] using any [`QueryLabelProvider`] + `InlineFilterSearch`.
///
/// Label providers are pre-built (one per query) by the caller — use [`FilteredQueryLabelProvider`]
/// for per-node AST evaluation or [`BitmapProvider`] for O(1) bitmap lookup.
struct InlineLabelProviderSearcher<DP, T>
where
    DP: diskann::provider::DataProvider,
{
    index: Arc<DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    provider_source: InlineLabelProviderSource,
    adaptive_l: Option<AdaptiveL>,
}

/// Implements [`search_api::Search`] using [`BitmapQueryLabelProvider`] + `InlineFilterSearch`.
///
/// The bitmap is computed once per-query for O(1) per-node lookup, and
/// [`diskann::graph::search::InlineFilterSearch`] applies hard rejection (no beta biasing).
/// Optionally scales `l_search` via adaptive L for low-selectivity filters.
/// Implements [`search_api::Search`] for bitmap-backed beta filtering on a plain provider.
struct BitmapFilteredSearcher<DP, T>
where
    DP: diskann::provider::DataProvider,
{
    index: Arc<DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    beta: f32,

    // Dynamic path: compute bitmaps per-query
    #[cfg(not(feature = "precompute-query-bitmaps"))]
    attribute_store: Arc<RoaringAttributeStore<u32>>,
    #[cfg(not(feature = "precompute-query-bitmaps"))]
    label_universe: Arc<RoaringTreemap>,
    #[cfg(not(feature = "precompute-query-bitmaps"))]
    predicates: Arc<Vec<(usize, ASTExpr)>>,

    // Precomputed path: bitmaps precomputed at searcher creation
    #[cfg(feature = "precompute-query-bitmaps")]
    precomputed_bitmaps: Arc<[RoaringTreemap]>,
}

impl<DP, T> search_api::Search for BitmapFilteredSearcher<DP, T>
where
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> BetaFilter<common::FullPrecision, u32>:
        glue::SearchStrategy<'a, DP, &'a [T]>
        + glue::DefaultPostProcessor<'a, DP, &'a [T], u32>,
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
{
    type Id = DP::ExternalId;
    type Parameters = Knn;
    type Output = FilteredSearchOutput;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Knn) -> search_api::IdCount {
        search_api::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Knn,
        buffer: &mut O,
        index: usize,
    ) -> diskann::ANNResult<FilteredSearchOutput>
    where
        O: diskann::graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let ctx = DefaultContext;
        let query_vec = self.queries.row(index);

        #[cfg(not(feature = "precompute-query-bitmaps"))]
        let label_provider: Arc<dyn QueryLabelProvider<u32>> = {
            // let start = std::time::Instant::now();
            let (_, ref expr) = self.predicates[index];
            let label_provider = BitmapQueryLabelProvider::from_ast(
                expr,
                self.attribute_store.as_ref(),
                self.label_universe.as_ref(),
            ).map_err(ANNError::log_async_error)?;
            // eprintln!(
            //     "  Computed bitmap for query {} in {} ms",
            //     index,
            //     start.elapsed().as_millis()
            // );
            Arc::new(label_provider)
        };

        #[cfg(feature = "precompute-query-bitmaps")]
        let label_provider: Arc<dyn QueryLabelProvider<u32>> = {
            Arc::new(BitmapQueryLabelProvider {
                query_index: index,
                precomputed_bitmaps: self.precomputed_bitmaps.clone(),
            })
        };

        let strategy = BetaFilter::new(common::FullPrecision, label_provider, self.beta);

        let k = parameters.k_value().get();
        let mut ids = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        let mut scratch = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let stats = &self
            .index
            .search(*parameters, &strategy, &ctx, query_vec, &mut scratch)
            .await?;

        let count = scratch.current_len();
        for (&id, &dist) in std::iter::zip(&ids[..count], &distances[..count]) {
            if buffer.push(id, dist).is_full() {
                break;
            }
        }

        Ok(FilteredSearchOutput {
            distances: distances[..count].to_vec(),
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

impl<DP, T> search_api::Search for FilteredSearcher<DP, T>
where
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> InlineBetaStrategy<common::FullPrecision>:
        glue::SearchStrategy<'a, DP, &'a FilteredQuery<'a, &'a [T]>>
        + glue::DefaultPostProcessor<'a, DP, &'a FilteredQuery<'a, &'a [T]>, u32>,
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
{
    type Id = DP::ExternalId;
    type Parameters = Knn;
    type Output = FilteredSearchOutput;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Knn) -> search_api::IdCount {
        search_api::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Knn,
        buffer: &mut O,
        index: usize,
    ) -> diskann::ANNResult<FilteredSearchOutput>
    where
        O: diskann::graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let ctx = DefaultContext;
        let query_vec = self.queries.row(index);
        let (_, ref ast_expr) = self.predicates[index];
        let strategy = InlineBetaStrategy::new(self.beta, common::FullPrecision);
        let filtered_query = FilteredQuery::new(query_vec, ast_expr);

        // Use a concrete IdDistance scratch buffer so that both the IDs and distances
        // are captured. Afterwards, the valid IDs are forwarded into the framework buffer.
        let k = parameters.k_value().get();
        let mut ids = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        let mut scratch = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let stats = &self
            .index
            .search(*parameters, &strategy, &ctx, &filtered_query, &mut scratch)
            .await?;

        let count = scratch.current_len();
        for (&id, &dist) in std::iter::zip(&ids[..count], &distances[..count]) {
            if buffer.push(id, dist).is_full() {
                break;
            }
        }

        Ok(FilteredSearchOutput {
            distances: distances[..count].to_vec(),
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

impl<DP, T> search_api::Search for InlineLabelProviderSearcher<DP, T>
where
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> common::FullPrecision:
        glue::SearchStrategy<'a, DP, &'a [T]>
        + glue::DefaultPostProcessor<'a, DP, &'a [T], u32>,
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
{
    type Id = u32;
    type Parameters = Knn;
    type Output = FilteredSearchOutput;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Knn) -> search_api::IdCount {
        search_api::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Knn,
        buffer: &mut O,
        index: usize,
    ) -> diskann::ANNResult<FilteredSearchOutput>
    where
        O: diskann::graph::SearchOutputBuffer<Self::Id> + Send,
    {
        let ctx = DefaultContext;
        let query_vec = self.queries.row(index);
        let label_provider: Arc<dyn QueryLabelProvider<u32>> = match &self.provider_source {
            InlineLabelProviderSource::Ast {
                predicates,
                attribute_store,
            } => {
                let (_, ref ast_expr) = predicates[index];
                Arc::new(
                    FilteredQueryLabelProvider::new(ast_expr.clone(), attribute_store.clone())
                        .map_err(ANNError::log_async_error)?,
                )
            }
            InlineLabelProviderSource::Bitmap {
                #[cfg(not(feature = "precompute-query-bitmaps"))]
                predicates,
                #[cfg(not(feature = "precompute-query-bitmaps"))]
                attribute_store,
                #[cfg(not(feature = "precompute-query-bitmaps"))]
                label_universe,
                #[cfg(feature = "precompute-query-bitmaps")]
                precomputed_bitmaps,
            } => {
                #[cfg(not(feature = "precompute-query-bitmaps"))]
                {
                    let (_, ref expr) = predicates[index];
                    Arc::new(
                        BitmapQueryLabelProvider::from_ast(
                            expr,
                            attribute_store.as_ref(),
                            label_universe.as_ref(),
                        )
                        .map_err(ANNError::log_async_error)?,
                    )
                }
                #[cfg(feature = "precompute-query-bitmaps")]
                {
                    Arc::new(BitmapQueryLabelProvider {
                        query_index: index,
                        precomputed_bitmaps: precomputed_bitmaps.clone(),
                    })
                }
            }
        };

        let k = parameters.k_value().get();
        let mut ids = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        let mut scratch = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let inline = diskann::graph::search::InlineFilterSearch::new(
            *parameters,
            &*label_provider,
            self.adaptive_l.clone(),
        );

        let stats = self
            .index
            .search(inline, &common::FullPrecision, &ctx, query_vec, &mut scratch)
            .await?;

        let count = scratch.current_len();
        for (&id, &dist) in std::iter::zip(&ids[..count], &distances[..count]) {
            if buffer.push(id, dist).is_full() {
                break;
            }
        }

        Ok(FilteredSearchOutput {
            distances: distances[..count].to_vec(),
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

/// Aggregates per-rep [`search_api::SearchResults`] into a [`SearchRunStats`].
struct FilteredSearchAggregator<'a> {
    groundtruth: &'a Vec<Vec<u32>>,
    predicates: &'a [(usize, ASTExpr)],
    recall_k: usize,
}

impl search_api::Aggregate<Knn, u32, FilteredSearchOutput> for FilteredSearchAggregator<'_> {
    type Output = SearchRunStats;

    fn aggregate(
        &mut self,
        run: search_api::Run<Knn>,
        results: Vec<search_api::SearchResults<u32, FilteredSearchOutput>>,
    ) -> anyhow::Result<SearchRunStats> {
        let parameters = run.parameters();
        let search_n = parameters.k_value().get();
        let num_queries = results.first().map(|r| r.len()).unwrap_or(0);

        // Recall from first rep only.
        let recall_metrics: RecallMetrics = match results.first() {
            Some(first) => (&recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                search_n,
                recall::GroundTruthMode::Flexible,
            )?)
                .into(),
            None => anyhow::bail!("no search results"),
        };

        // Per-query details from first rep (only queries with recall < 1).
        let first = results.first().unwrap();
        let per_query_details: Vec<PerQueryDetails> = (0..num_queries)
            .filter_map(|query_idx| {
                let result_ids: Vec<u32> = first.ids().as_rows().row(query_idx).to_vec();
                let result_distances: Vec<f32> = first
                    .output()
                    .get(query_idx)
                    .map(|o| o.distances.clone())
                    .unwrap_or_default();
                let gt_ids: Vec<u32> = self
                    .groundtruth
                    .get(query_idx)
                    .map(|gt| gt.iter().take(20).copied().collect())
                    .unwrap_or_default();

                let result_set: std::collections::HashSet<u32> =
                    result_ids.iter().copied().collect();
                let gt_set: std::collections::HashSet<u32> =
                    gt_ids.iter().take(self.recall_k).copied().collect();
                let intersection = result_set.intersection(&gt_set).count();
                let per_query_recall = if gt_set.is_empty() {
                    1.0
                } else {
                    intersection as f64 / gt_set.len() as f64
                };

                if per_query_recall >= 1.0 {
                    return None;
                }

                let (_, ref ast_expr) = self.predicates[query_idx];
                Some(PerQueryDetails {
                    query_id: query_idx,
                    filter: format!("{:?}", ast_expr),
                    recall: per_query_recall,
                    result_ids,
                    result_distances,
                    groundtruth_ids: gt_ids,
                })
            })
            .collect();

        // Wall-clock latency and QPS per rep.
        let rep_latencies: Vec<MicroSeconds> =
            results.iter().map(|r| r.end_to_end_latency()).collect();
        let qps: Vec<f64> = rep_latencies
            .iter()
            .map(|l| num_queries as f64 / l.as_seconds())
            .collect();

        // Per-query latencies, comparisons, and hops aggregated across all reps.
        let mut all_latencies: Vec<MicroSeconds> = Vec::new();
        let mut all_cmps: Vec<u32> = Vec::new();
        let mut all_hops: Vec<u32> = Vec::new();
        for r in &results {
            all_latencies.extend_from_slice(r.latencies());
            for o in r.output() {
                all_cmps.push(o.comparisons);
                all_hops.push(o.hops);
            }
        }

        let percentiles::Percentiles { mean, p90, p99, .. } =
            percentiles::compute_percentiles(&mut all_latencies)?;

        let mean_cmps = if all_cmps.is_empty() {
            0.0
        } else {
            all_cmps.iter().map(|&x| x as f32).sum::<f32>() / all_cmps.len() as f32
        };
        let mean_hops = if all_hops.is_empty() {
            0.0
        } else {
            all_hops.iter().map(|&x| x as f32).sum::<f32>() / all_hops.len() as f32
        };

        Ok(SearchRunStats {
            num_threads: run.setup().threads.get(),
            num_queries,
            search_n,
            search_l: parameters.l_value().get(),
            recall: recall_metrics,
            qps,
            wall_clock_time: rep_latencies,
            mean_latency: mean,
            p90_latency: p90,
            p99_latency: p99,
            mean_cmps,
            mean_hops,
            per_query_details: Some(per_query_details),
        })
    }
}

/// Run filtered search with the given parameters.
#[allow(clippy::too_many_arguments)]
fn run_filtered_search<DP, T>(
    index: &Arc<DiskANNIndex<DP>>,
    queries: &Matrix<T>,
    predicates: &[(usize, ASTExpr)],
    groundtruth: &Vec<Vec<u32>>,
    beta: f32,
    num_threads: NonZeroUsize,
    search_n: usize,
    search_l: usize,
    recall_k: usize,
    reps: NonZeroUsize,
) -> anyhow::Result<SearchRunStats>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> InlineBetaStrategy<common::FullPrecision>:
        glue::SearchStrategy<'a, DP, &'a FilteredQuery<'a, &'a [T]>>
        + glue::DefaultPostProcessor<'a, DP, &'a FilteredQuery<'a, &'a [T]>, u32>,
{
    let searcher = Arc::new(FilteredSearcher {
        index: index.clone(),
        queries: Arc::new(queries.clone()),
        predicates: Arc::new(predicates.to_vec()),
        beta,
    });

    let parameters = Knn::new_default(search_n, search_l)?;
    let setup = search_api::Setup {
        threads: num_threads,
        tasks: num_threads,
        reps,
    };

    let mut results = search_api::search_all(
        searcher,
        [search_api::Run::new(parameters, setup)],
        FilteredSearchAggregator {
            groundtruth,
            predicates,
            recall_k,
        },
    )?;

    results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no search results"))
}

#[allow(clippy::too_many_arguments)]
fn run_filtered_search_with_label_provider<DP, T>(
    index: &Arc<DiskANNIndex<DP>>,
    queries: &Matrix<T>,
    labels: &[LabelDocument],
    predicates: &[(usize, ASTExpr)],
    groundtruth: &Vec<Vec<u32>>,
    adaptive_l_config: Option<&AdaptiveLConfig>,
    num_threads: NonZeroUsize,
    search_n: usize,
    search_l: usize,
    recall_k: usize,
    reps: NonZeroUsize,
) -> anyhow::Result<SearchRunStats>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> common::FullPrecision:
        glue::SearchStrategy<'a, DP, &'a [T]>
        + glue::DefaultPostProcessor<'a, DP, &'a [T], u32>,
{
    let (attribute_store, _) = build_label_attribute_store(labels)?;

    let adaptive_l = adaptive_l_config
        .map(|cfg| AdaptiveL::new(cfg.sample_count, cfg.scale_factor))
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid adaptive_l config: {}", e))?;

    let searcher = Arc::new(InlineLabelProviderSearcher {
        index: index.clone(),
        queries: Arc::new(queries.clone()),
        provider_source: InlineLabelProviderSource::Ast {
            predicates: Arc::new(predicates.to_vec()),
            attribute_store,
        },
        adaptive_l,
    });

    let parameters = Knn::new_default(search_n, search_l)?;
    let setup = search_api::Setup {
        threads: num_threads,
        tasks: num_threads,
        reps,
    };

    let mut results = search_api::search_all(
        searcher,
        [search_api::Run::new(parameters, setup)],
        FilteredSearchAggregator {
            groundtruth,
            predicates,
            recall_k,
        },
    )?;

    results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no search results"))
}

#[allow(clippy::too_many_arguments)]
fn run_bitmap_inline_search<DP, T>(
    index: &Arc<DiskANNIndex<DP>>,
    queries: &Matrix<T>,
    labels: &[LabelDocument],
    groundtruth: &Vec<Vec<u32>>,
    predicates: &[(usize, ASTExpr)],
    adaptive_l_config: Option<&AdaptiveLConfig>,
    num_threads: NonZeroUsize,
    search_n: usize,
    search_l: usize,
    recall_k: usize,
    reps: NonZeroUsize,
) -> anyhow::Result<SearchRunStats>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> common::FullPrecision:
        glue::SearchStrategy<'a, DP, &'a [T]>
        + glue::DefaultPostProcessor<'a, DP, &'a [T], u32>,
{
    let adaptive_l = adaptive_l_config
        .map(|cfg| AdaptiveL::new(cfg.sample_count, cfg.scale_factor))
        .transpose()
        .map_err(|e| anyhow::anyhow!("invalid adaptive_l config: {}", e))?;

    let (attribute_store, label_ids) = build_label_attribute_store(labels)?;
    let mut universe = RoaringTreemap::new();
    universe.extend(label_ids.iter().copied().map(u64::from));
    let universe = Arc::new(universe);

    #[cfg(feature = "precompute-query-bitmaps")]
    let precomputed_bitmaps: Arc<[RoaringTreemap]> = {
        let evaluator = InvertedBitmapEvaluator::new(&attribute_store, universe.as_ref());
        let bitmaps = predicates
            .iter()
            .map(|(_, expr)| -> anyhow::Result<RoaringTreemap> {
                let encoded = EncodedFilterExpr::from_attribute_store(expr, &attribute_store)
                    .context("failed to encode filter expression")?;
                evaluator
                    .evaluate(&encoded)
                    .context("failed to evaluate encoded expression")
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        Arc::from(bitmaps.into_boxed_slice())
    };

    let searcher = Arc::new(InlineLabelProviderSearcher {
        index: index.clone(),
        queries: Arc::new(queries.to_owned()),
        provider_source: InlineLabelProviderSource::Bitmap {
            #[cfg(not(feature = "precompute-query-bitmaps"))]
            predicates: Arc::new(predicates.to_vec()),
            #[cfg(not(feature = "precompute-query-bitmaps"))]
            attribute_store,
            #[cfg(not(feature = "precompute-query-bitmaps"))]
            label_universe: universe,
            #[cfg(feature = "precompute-query-bitmaps")]
            precomputed_bitmaps,
        },
        adaptive_l,
    });

    let parameters = Knn::new_default(search_n, search_l)?;
    let setup = search_api::Setup {
        threads: num_threads,
        tasks: num_threads,
        reps,
    };

    let mut results = search_api::search_all(
        searcher,
        [search_api::Run::new(parameters, setup)],
        FilteredSearchAggregator {
            groundtruth,
            predicates,
            recall_k,
        },
    )?;

    results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no search results"))
}

#[allow(clippy::too_many_arguments)]
fn run_bitmap_filtered_search<DP, T>(
    index: &Arc<DiskANNIndex<DP>>,
    queries: &Matrix<T>,
    labels: &[LabelDocument],
    groundtruth: &Vec<Vec<u32>>,
    predicates: &[(usize, ASTExpr)],
    beta: f32,
    num_threads: NonZeroUsize,
    search_n: usize,
    search_l: usize,
    recall_k: usize,
    reps: NonZeroUsize,
) -> anyhow::Result<SearchRunStats>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync
        + 'static,
    for<'a> BetaFilter<common::FullPrecision, u32>:
        glue::SearchStrategy<'a, DP, &'a [T]>
        + glue::DefaultPostProcessor<'a, DP, &'a [T], u32>,
{
    let (attribute_store, label_ids) = build_label_attribute_store(labels)?;
    let mut universe = RoaringTreemap::new();
    universe.extend(label_ids.iter().copied().map(u64::from));
    let universe = Arc::new(universe);

    #[cfg(not(feature = "precompute-query-bitmaps"))]
    let searcher: Arc<BitmapFilteredSearcher<DP, T>> = Arc::new(BitmapFilteredSearcher {
        index: index.clone(),
        queries: Arc::new(queries.to_owned()),
        attribute_store,
        label_universe: universe.clone(),
        predicates: Arc::new(predicates.to_vec()),
        beta,
    });

    #[cfg(feature = "precompute-query-bitmaps")]
    let searcher: Arc<BitmapFilteredSearcher<DP, T>> = {
        // Precompute all query bitmaps at searcher creation time
        let precompute_start = std::time::Instant::now();
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.get())
            .build()
            .context("Failed to create Rayon thread pool for query bitmap precomputation")?;

        let evaluator = InvertedBitmapEvaluator::new(&attribute_store, universe.as_ref());
        let precomputed_bitmaps: Vec<RoaringTreemap> = thread_pool.install(|| {
            predicates
                .par_iter()
                .enumerate()
                .map(|(query_index, (_, expr))| -> anyhow::Result<RoaringTreemap> {
                    let encoded_filter =
                        EncodedFilterExpr::from_attribute_store(expr, &attribute_store).with_context(
                            || {
                                format!(
                                    "Failed to encode filter expression for precomputation at query index {}",
                                    query_index
                                )
                            },
                        )?;

                    evaluator.evaluate(&encoded_filter).with_context(|| {
                        format!(
                            "Failed to evaluate encoded expression during precomputation at query index {}",
                            query_index
                        )
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        let precompute_elapsed = precompute_start.elapsed();
        eprintln!(
            "Precomputed {} query bitmaps in {:.3}s using {} Rayon threads",
            precomputed_bitmaps.len(),
            precompute_elapsed.as_secs_f64(),
            num_threads.get()
        );

        Arc::new(BitmapFilteredSearcher {
            index: index.clone(),
            queries: Arc::new(queries.to_owned()),
            precomputed_bitmaps: Arc::from(precomputed_bitmaps.into_boxed_slice()),
            beta,
        })
    };

    let parameters = Knn::new_default(search_n, search_l)?;
    let setup = search_api::Setup {
        threads: num_threads,
        tasks: num_threads,
        reps,
    };

    let mut results = search_api::search_all(
        searcher,
        [search_api::Run::new(parameters, setup)],
        FilteredSearchAggregator {
            groundtruth,
            predicates,
            recall_k,
        },
    )?;

    results
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no search results"))
}

/// Per-query detailed results for debugging/analysis
#[derive(Debug, Serialize)]
pub struct PerQueryDetails {
    pub query_id: usize,
    pub filter: String,
    pub recall: f64,
    pub result_ids: Vec<u32>,
    pub result_distances: Vec<f32>,
    pub groundtruth_ids: Vec<u32>,
}

/// Results from a single search configuration (one search_l value).
#[derive(Debug, Serialize)]
pub struct SearchRunStats {
    pub num_threads: usize,
    pub num_queries: usize,
    pub search_n: usize,
    pub search_l: usize,
    pub recall: RecallMetrics,
    pub qps: Vec<f64>,
    pub wall_clock_time: Vec<MicroSeconds>,
    pub mean_latency: f64,
    pub p90_latency: MicroSeconds,
    pub p99_latency: MicroSeconds,
    pub mean_cmps: f32,
    pub mean_hops: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_query_details: Option<Vec<PerQueryDetails>>,
}

#[derive(Debug, Serialize)]
pub struct DocumentIndexStats {
    pub num_vectors: usize,
    pub dim: usize,
    pub label_count: usize,
    pub data_load_time: MicroSeconds,
    pub label_load_time: MicroSeconds,
    pub build_time: MicroSeconds,
    pub insert_latencies: percentiles::Percentiles<MicroSeconds>,
    pub search: Vec<SearchRunStats>,
}

impl std::fmt::Display for DocumentIndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Document Index Build Stats:")?;
        writeln!(f, "  Vectors: {} x {}", self.num_vectors, self.dim)?;
        writeln!(f, "  Label Count: {}", self.label_count)?;
        writeln!(
            f,
            "  Data Load Time: {} s",
            self.data_load_time.as_seconds()
        )?;
        writeln!(
            f,
            "  Label Load Time: {} s",
            self.label_load_time.as_seconds()
        )?;
        writeln!(f, "  Total Build Time: {} s", self.build_time.as_seconds())?;
        writeln!(f, "  Insert Latencies:")?;
        writeln!(f, "    Mean: {} us", self.insert_latencies.mean)?;
        writeln!(f, "    P50: {} us", self.insert_latencies.median)?;
        writeln!(f, "    P90: {} us", self.insert_latencies.p90)?;
        writeln!(f, "    P99: {} us", self.insert_latencies.p99)?;

        if !self.search.is_empty() {
            let header = [
                "L",
                "KNN",
                "Avg Cmps",
                "Avg Hops",
                "QPS -mean(max)",
                "Avg Latency",
                "p99 Latency",
                "Recall",
                "Threads",
                "Queries",
                "WallClock(s)",
            ];
            writeln!(f, "\nFiltered Search Results:")?;
            let mut table = fmt::Table::new(header, self.search.len());
            self.search.iter().enumerate().for_each(|(row_idx, s)| {
                let mut row = table.row(row_idx);
                let mean_qps = percentiles::mean(&s.qps).unwrap_or(0.0);
                let max_qps = s.qps.iter().cloned().fold(0.0_f64, f64::max);
                let mean_wall_clock = percentiles::mean(
                    &s.wall_clock_time
                        .iter()
                        .map(|l| l.as_seconds())
                        .collect::<Vec<_>>(),
                )
                .unwrap_or(0.0);
                row.insert(s.search_l, 0);
                row.insert(s.search_n, 1);
                row.insert(format!("{:.1}", s.mean_cmps), 2);
                row.insert(format!("{:.1}", s.mean_hops), 3);
                row.insert(format!("{:.1}({:.1})", mean_qps, max_qps), 4);
                row.insert(format!("{:.1} s", s.mean_latency), 5);
                row.insert(format!("{:.1} s", s.p99_latency), 6);
                row.insert(format!("{:.4}", s.recall.average), 7);
                row.insert(s.num_threads, 8);
                row.insert(s.num_queries, 9);
                row.insert(format!("{:.3} s", mean_wall_clock), 10);
            });
            write!(f, "{}", table)?;
        }
        Ok(())
    }
}

// ================================
// Parallel Build Support
// ================================

/// Implements [`Build`] for parallel document insertion into a [`DiskANNIndex`]
/// backed by a [`DocumentProvider`]. Each call to [`Build::build`] inserts a
/// contiguous range of vectors and their associated attributes.
struct DocumentIndexBuilder<DP: diskann::provider::DataProvider, T> {
    index: Arc<DiskANNIndex<DP>>,
    data: Arc<Matrix<T>>,
    attributes: Arc<Vec<Vec<Attribute>>>,
    strategy: DocumentInsertStrategy<common::FullPrecision>,
}

impl<DP: diskann::provider::DataProvider, T> DocumentIndexBuilder<DP, T> {
    fn new(
        index: Arc<DiskANNIndex<DP>>,
        data: Arc<Matrix<T>>,
        attributes: Arc<Vec<Vec<Attribute>>>,
        strategy: DocumentInsertStrategy<common::FullPrecision>,
    ) -> Arc<Self> {
        Arc::new(Self {
            index,
            data,
            attributes,
            strategy,
        })
    }
}

struct DefaultIndexBuilder<DP: diskann::provider::DataProvider, T> {
    index: Arc<DiskANNIndex<DP>>,
    data: Arc<Matrix<T>>,
    strategy: common::FullPrecision,
}

impl<DP: diskann::provider::DataProvider, T> DefaultIndexBuilder<DP, T> {
    fn new(
        index: Arc<DiskANNIndex<DP>>,
        data: Arc<Matrix<T>>,
        strategy: common::FullPrecision,
    ) -> Arc<Self> {
        Arc::new(Self {
            index,
            data,
            strategy,
        })
    }
}

impl<DP, T> Build for DefaultIndexBuilder<DP, T>
where
    DP: diskann::provider::DataProvider<Context = DefaultContext, ExternalId = u32>
        + for<'b> diskann::provider::SetElement<&'b [T]>
        + AsyncFriendly,
    for<'b> common::FullPrecision: diskann::graph::glue::InsertStrategy<'b, DP, &'b [T]>,
    common::FullPrecision: AsyncFriendly,
    T: AsyncFriendly,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: std::ops::Range<usize>) -> diskann::ANNResult<Self::Output> {
        let ctx = DefaultContext;
        for i in range {
            self.index
                .insert(&self.strategy, &ctx, &(i as u32), self.data.row(i))
                .await?;
        }
        Ok(())
    }
}

impl<DP, T> Build for DocumentIndexBuilder<DP, T>
where
    DP: diskann::provider::DataProvider<Context = DefaultContext, ExternalId = u32>
        + for<'a> diskann::provider::SetElement<&'a Document<'a, [T]>>
        + AsyncFriendly,
    for<'a> DocumentInsertStrategy<common::FullPrecision>:
        diskann::graph::glue::InsertStrategy<'a, DP, &'a Document<'a, [T]>>,
    DocumentInsertStrategy<common::FullPrecision>: AsyncFriendly,
    T: AsyncFriendly,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: std::ops::Range<usize>) -> diskann::ANNResult<Self::Output> {
        let ctx = DefaultContext;
        for i in range {
            let attrs = self.attributes.get(i).ok_or_else(|| {
                ANNError::message(
                    ANNErrorKind::Opaque,
                    format!("Failed to get attributes at index {}", i),
                )
            })?;
            let doc = Document::new(self.data.row(i), attrs);
            self.index
                .insert(&self.strategy, &ctx, &(i as u32), &doc)
                .await?;
        }
        Ok(())
    }
}
