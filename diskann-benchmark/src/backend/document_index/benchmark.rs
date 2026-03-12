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

use anyhow::Result;
use diskann::{
    graph::{
        config::Builder as ConfigBuilder, config::MaxDegree, config::PruneKind,
        search_output_buffer, DiskANNIndex, SearchOutputBuffer, SearchParams, StartPointStrategy,
    },
    provider::DefaultContext,
};
use diskann_benchmark_core::{
    build::{self, AsProgress, Build, Parallelism, Progress},
    recall,
    search as search_api,
    tokio,
};
use diskann_benchmark_runner::{
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    registry::Benchmarks,
    utils::{datatype, percentiles, MicroSeconds},
    Any,
};
use diskann_label_filter::{
    attribute::{Attribute, AttributeValue},
    document::Document,
    encoded_attribute_provider::{
        document_insert_strategy::DocumentInsertStrategy, document_provider::DocumentProvider,
        roaring_attribute_store::RoaringAttributeStore,
    },
    inline_beta_search::inline_beta_filter::InlineBetaStrategy,
    query::FilteredQuery,
    read_and_parse_queries, read_baselabels, ASTExpr,
};
use diskann_providers::model::graph::provider::async_::{
    common::{self, NoStore, TableBasedDeletes},
    inmem::{CreateFullPrecision, DefaultProvider, DefaultProviderParameters, SetStartPoints},
};
use diskann_utils::{future::AsyncFriendly, sampling::medoid::ComputeMedoid};
use diskann_utils::views::MatrixView;
use diskann_utils::views::Matrix;
use diskann_vector::PureDistanceFunction;
use diskann_vector::distance::SquaredL2;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::{
    inputs::document_index::DocumentIndexBuild,
    utils::{
        datafiles::{self, BinFile},
        recall::SerializableRecallMetrics,
    },
};

/// Register the document index benchmarks.
pub(crate) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    benchmarks.register::<DocumentIndexJob<'static, f32>>(
        "document-index-build-f32",
        |job, _checkpoint, out| {
            let stats = job.run(out)?;
            Ok(serde_json::to_value(stats)?)
        },
    );
}

/// Document index benchmark job.
pub(super) struct DocumentIndexJob<'a, T> {
    input: &'a DocumentIndexBuild,
    _type: std::marker::PhantomData<T>,
}

impl<'a, T> DocumentIndexJob<'a, T> {
    fn new(input: &'a DocumentIndexBuild) -> Self {
        Self {
            input,
            _type: std::marker::PhantomData,
        }
    }
}

impl<T: 'static> diskann_benchmark_runner::dispatcher::Map for DocumentIndexJob<'static, T> {
    type Type<'a> = DocumentIndexJob<'a, T>;
}

// Dispatch from the concrete input type
impl<'a, T> DispatchRule<&'a DocumentIndexBuild> for DocumentIndexJob<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a DocumentIndexBuild) -> Result<MatchScore, FailureScore> {
        match _from.build.data_type {
            datatype::DataType::Float32 => Ok(MatchScore(0)),
            datatype::DataType::UInt8 => Ok(MatchScore(0)),
            datatype::DataType::Int8 => Ok(MatchScore(0)),
            _ => Err(datatype::MATCH_FAIL),
        }
    }

    fn convert(from: &'a DocumentIndexBuild) -> Result<Self, Self::Error> {
        Ok(DocumentIndexJob::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        _from: Option<&&'a DocumentIndexBuild>,
    ) -> std::fmt::Result {
        writeln!(f, "tag: \"{}\"", DocumentIndexBuild::tag())
    }
}

// Central dispatch mapping from Any
impl<'a, T> DispatchRule<&'a Any> for DocumentIndexJob<'a, T>
where
    datatype::Type<T>: DispatchRule<datatype::DataType>,
{
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<DocumentIndexBuild, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<DocumentIndexBuild, Self>()
    }

    fn description(f: &mut std::fmt::Formatter, from: Option<&&'a Any>) -> std::fmt::Result {
        Any::description::<DocumentIndexBuild, Self>(f, from, DocumentIndexBuild::tag())
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

impl<'a, T> DocumentIndexJob<'a, T> {
    fn run(self, mut output: &mut dyn Output) -> Result<DocumentIndexStats, anyhow::Error>
    where
        T: diskann::utils::VectorRepr
            + diskann::graph::SampleableForStart
            + diskann_utils::sampling::WithApproximateNorm
            + 'static,
        for<'b> diskann_vector::distance::SquaredL2: PureDistanceFunction<&'b [T], &'b [T]>
    {
        let build = &self.input.build;

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
            .into_iter()
            .map(|doc| hashmap_to_attributes(doc.flatten_metadata_with_separator("")))
            .collect();

        // 3. Create the index configuration
        let metric = build.distance.into();
        let prune_kind = PruneKind::from_metric(metric);
        let mut config_builder = ConfigBuilder::new(
            build.max_degree, // pruned_degree
            MaxDegree::Same,  // max_degree
            build.l_build,    // l_build
            prune_kind,       // prune_kind
        );
        config_builder.alpha(build.alpha);
        let config = config_builder.build()?;

        // 4. Create the data provider directly
        writeln!(output, "Creating index...")?;
        let params = DefaultProviderParameters {
            max_points: num_vectors,
            frozen_points: diskann::utils::ONE,
            metric,
            dim,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
            max_degree: build.max_degree as u32,
        };

        // Create the underlying provider
        let fp_precursor = CreateFullPrecision::<T>::new(dim, None);
        let inner_provider =
            DefaultProvider::new_empty(params, fp_precursor, NoStore, TableBasedDeletes)?;

        // Set start points using medoid strategy
        let start_points = StartPointStrategy::Medoid
            .compute(data.as_view())
            .map_err(|e| anyhow::anyhow!("Failed to compute start points: {}", e))?;
        inner_provider.set_start_points(start_points.row_iter())?;

        // 5. Create DocumentProvider wrapping the inner provider
        let attribute_store = RoaringAttributeStore::<u32>::new();

        // Store attributes for the start point (medoid)
        // Start points are stored at indices num_vectors..num_vectors+frozen_points
        let medoid_idx = compute_medoid_index(&data)?;
        let start_point_id = num_vectors as u32; // Start points begin at max_points
        let medoid_attrs = attributes.get(medoid_idx).cloned().unwrap_or_default();
        use diskann_label_filter::traits::attribute_store::AttributeStore;
        attribute_store.set_element(&start_point_id, &medoid_attrs)?;

        let doc_provider = DocumentProvider::new(inner_provider, attribute_store);

        // Create a new DiskANNIndex with DocumentProvider
        let doc_index = Arc::new(DiskANNIndex::new(config, doc_provider, None));

        // 6. Build index by inserting vectors and attributes (parallel)
        writeln!(
            output,
            "Building index with {} vectors using {} threads...",
            num_vectors, build.num_threads
        )?;
        let timer = std::time::Instant::now();

        let rt = tokio::runtime(build.num_threads)?;
        let data_arc = Arc::new(data);
        let attributes_arc = Arc::new(attributes);

        let builder = DocumentIndexBuilder::new(
            doc_index.clone(),
            data_arc.clone(),
            attributes_arc.clone(),
            DocumentInsertStrategy::new(common::FullPrecision),
        );
        let num_tasks = NonZeroUsize::new(build.num_threads).unwrap_or(diskann::utils::ONE);
        let parallelism = Parallelism::dynamic(diskann::utils::ONE, num_tasks);
        let progress = IndicatifAsProgress({
            let bar = ProgressBar::with_draw_target(Some(num_vectors as u64), output.draw_target());
            bar.set_style(
                ProgressStyle::with_template("Building [{elapsed_precise}] {wide_bar} {percent}")
                    .expect("valid template"),
            );
            bar
        });
        let build_results =
            build::build_tracked(builder, parallelism, &rt, Some(&progress))?;
        let insert_latencies: Vec<MicroSeconds> = build_results
            .take_output()
            .into_iter()
            .map(|r| r.latency)
            .collect();

        let build_time: MicroSeconds = timer.elapsed().into();
        writeln!(output, "  Index built in {} s", build_time.as_seconds())?;

        let insert_percentiles = percentiles::compute_percentiles(&mut insert_latencies.clone())?;
        // =====================
        // Search Phase
        // =====================
        let search_input = &self.input.search;

        // Load query vectors (same type as data for compatible distance computation)
        writeln!(output, "\nLoading query vectors...")?;
        let query_path: &Path = search_input.queries.as_ref();
        let queries: Matrix<T> = datafiles::load_dataset(BinFile(query_path))?;
        let num_queries = queries.nrows();
        writeln!(output, "  Loaded {} queries", num_queries)?;

        // Load and parse query predicates
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

        // Load groundtruth
        writeln!(output, "Loading groundtruth...")?;
        let gt_path: &Path = search_input.groundtruth.as_ref();
        let groundtruth: Vec<Vec<u32>> = datafiles::load_range_groundtruth(BinFile(gt_path))?;
        writeln!(
            output,
            "  Loaded groundtruth with {} rows",
            groundtruth.len()
        )?;

        // Run filtered searches
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

                    let search_run_result = run_filtered_search(
                        &doc_index,
                        &queries,
                        &parsed_predicates,
                        &groundtruth,
                        search_input.beta,
                        *num_threads,
                        run.search_n,
                        search_l,
                        run.recall_k,
                        search_input.reps,
                    )?;

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

        let stats = DocumentIndexStats {
            num_vectors,
            dim,
            label_count,
            data_load_time,
            label_load_time,
            build_time,
            insert_latencies: insert_percentiles,
            build_params: BuildParamsStats {
                max_degree: build.max_degree,
                l_build: build.l_build,
                alpha: build.alpha,
            },
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

impl<DP, T> search_api::Search for FilteredSearcher<DP, T>
where
    DP: diskann::provider::DataProvider<Context = DefaultContext, ExternalId = u32, InternalId = u32>
        + Send
        + Sync
        + 'static,
    InlineBetaStrategy<common::FullPrecision>: diskann::graph::glue::SearchStrategy<DP, FilteredQuery<Vec<T>>, u32>,
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
{
    type Id = DP::ExternalId;
    type Parameters = SearchParams;
    type Output = FilteredSearchOutput;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &SearchParams) -> search_api::IdCount {
        search_api::IdCount::Fixed(
            NonZeroUsize::new(parameters.k_value).unwrap_or(diskann::utils::ONE),
        )
    }

    async fn search<O>(
        &self,
        parameters: &SearchParams,
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
        let filtered_query = FilteredQuery::new(query_vec, ast_expr.clone());

        // Use a concrete IdDistance scratch buffer so that both the IDs and distances
        // are captured. Afterwards, the valid IDs are forwarded into the framework buffer.
        let k = parameters.k_value;
        let mut ids = vec![0u32; k];
        let mut distances = vec![0.0f32; k];
        let mut scratch = search_output_buffer::IdDistance::new(&mut ids, &mut distances);

        let stats = self
            .index
            .search(&strategy, &ctx, &filtered_query, parameters, &mut scratch)
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

impl search_api::Aggregate<SearchParams, u32, FilteredSearchOutput>
    for FilteredSearchAggregator<'_>
{
    type Output = SearchRunStats;

    fn aggregate(
        &mut self,
        run: search_api::Run<SearchParams>,
        results: Vec<search_api::SearchResults<u32, FilteredSearchOutput>>,
    ) -> anyhow::Result<SearchRunStats> {
        let parameters = run.parameters();
        let search_n = parameters.k_value;
        let num_queries = results.first().map(|r| r.len()).unwrap_or(0);

        // Recall from first rep only.
        let recall_metrics: SerializableRecallMetrics = match results.first() {
            Some(first) => (&recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                search_n,
                true,
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
            search_l: parameters.l_value,
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
    InlineBetaStrategy<common::FullPrecision>:
        diskann::graph::glue::SearchStrategy<DP, FilteredQuery<Vec<T>>>,
{
    let searcher = Arc::new(FilteredSearcher {
        index: index.clone(),
        queries: Arc::new(queries.clone()),
        predicates: Arc::new(predicates.to_vec()),
        beta,
    });

    let parameters = SearchParams::new_default(search_n, search_l)?;
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

    results.pop().ok_or_else(|| anyhow::anyhow!("no search results"))
}
#[derive(Debug, Serialize)]
pub struct BuildParamsStats {
    pub max_degree: usize,
    pub l_build: usize,
    pub alpha: f32,
}

/// Helper module for serializing arrays as compact single-line JSON strings
mod compact_array {
    use serde::Serializer;

    pub fn serialize_u32_vec<S>(vec: &Vec<u32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as a string containing the compact JSON array
        let compact = serde_json::to_string(vec).unwrap_or_default();
        serializer.serialize_str(&compact)
    }

    pub fn serialize_f32_vec<S>(vec: &Vec<f32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as a string containing the compact JSON array
        let compact = serde_json::to_string(vec).unwrap_or_default();
        serializer.serialize_str(&compact)
    }
}

/// Per-query detailed results for debugging/analysis
#[derive(Debug, Serialize)]
pub struct PerQueryDetails {
    pub query_id: usize,
    pub filter: String,
    pub recall: f64,
    #[serde(serialize_with = "compact_array::serialize_u32_vec")]
    pub result_ids: Vec<u32>,
    #[serde(serialize_with = "compact_array::serialize_f32_vec")]
    pub result_distances: Vec<f32>,
    #[serde(serialize_with = "compact_array::serialize_u32_vec")]
    pub groundtruth_ids: Vec<u32>,
}

/// Results from a single search configuration (one search_l value).
#[derive(Debug, Serialize)]
pub struct SearchRunStats {
    pub num_threads: usize,
    pub num_queries: usize,
    pub search_n: usize,
    pub search_l: usize,
    pub recall: SerializableRecallMetrics,
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
    pub build_params: BuildParamsStats,
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
        writeln!(f, "  Build Parameters:")?;
        writeln!(f, "    max_degree (R): {}", self.build_params.max_degree)?;
        writeln!(f, "    l_build (L): {}", self.build_params.l_build)?;
        writeln!(f, "    alpha: {}", self.build_params.alpha)?;

        if !self.search.is_empty() {
            writeln!(f, "\nFiltered Search Results:")?;
            writeln!(
                f,
                "  {:>8} {:>8} {:>10} {:>10} {:>15} {:>12} {:>12} {:>10} {:>8} {:>10} {:>12}",
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
                "WallClock(s)"
            )?;
            for s in &self.search {
                let mean_qps = if s.qps.is_empty() {
                    0.0
                } else {
                    s.qps.iter().sum::<f64>() / s.qps.len() as f64
                };
                let max_qps = s.qps.iter().cloned().fold(0.0_f64, f64::max);
                let mean_wall_clock = if s.wall_clock_time.is_empty() {
                    0.0
                } else {
                    s.wall_clock_time
                        .iter()
                        .map(|t| t.as_seconds())
                        .sum::<f64>()
                        / s.wall_clock_time.len() as f64
                };
                writeln!(
                    f,
                    "  {:>8} {:>8} {:>10.1} {:>10.1} {:>7.1}({:>5.1}) {:>12.1} {:>12} {:>10.4} {:>8} {:>10} {:>12.3}",
                    s.search_l,
                    s.search_n,
                    s.mean_cmps,
                    s.mean_hops,
                    mean_qps,
                    max_qps,
                    s.mean_latency,
                    s.p99_latency,
                    s.recall.average,
                    s.num_threads,
                    s.num_queries,
                    mean_wall_clock
                )?;
            }
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
    strategy: DocumentInsertStrategy<common::FullPrecision, [T]>,
}

impl<DP: diskann::provider::DataProvider, T> DocumentIndexBuilder<DP, T> {
    fn new(
        index: Arc<DiskANNIndex<DP>>,
        data: Arc<Matrix<T>>,
        attributes: Arc<Vec<Vec<Attribute>>>,
        strategy: DocumentInsertStrategy<common::FullPrecision, [T]>,
    ) -> Arc<Self> {
        Arc::new(Self {
            index,
            data,
            attributes,
            strategy,
        })
    }
}

impl<DP, T> Build for DocumentIndexBuilder<DP, T>
where
    DP: diskann::provider::DataProvider<Context = DefaultContext, ExternalId = u32>
        + for<'doc> diskann::provider::SetElement<Document<'doc, [T]>>
        + AsyncFriendly,
    for<'doc> DocumentInsertStrategy<common::FullPrecision, [T]>:
        diskann::graph::glue::InsertStrategy<DP, Document<'doc, [T]>>,
    DocumentInsertStrategy<common::FullPrecision, [T]>: AsyncFriendly,
    T: AsyncFriendly,
{
    type Output = ();

    fn num_data(&self) -> usize {
        self.data.nrows()
    }

    async fn build(&self, range: std::ops::Range<usize>) -> diskann::ANNResult<Self::Output> {
        let ctx = DefaultContext;
        for i in range {
            let attrs = self.attributes.get(i).cloned().unwrap_or_default();
            let doc = Document::new(self.data.row(i), attrs);
            self.index
                .insert(self.strategy, &ctx, &(i as u32), &doc)
                .await?;
        }
        Ok(())
    }
}

/// Adapts an already-constructed [`ProgressBar`] into the [`AsProgress`] / [`Progress`]
/// traits expected by [`build_tracked`].
struct IndicatifAsProgress(ProgressBar);

struct IndicatifProgress(ProgressBar);

impl Progress for IndicatifProgress {
    fn progress(&self, handled: usize) {
        self.0.inc(handled as u64);
    }

    fn finish(&self) {
        self.0.finish();
    }
}

impl AsProgress for IndicatifAsProgress {
    fn as_progress(&self, _max: usize) -> Arc<dyn Progress> {
        Arc::new(IndicatifProgress(self.0.clone()))
    }
}
