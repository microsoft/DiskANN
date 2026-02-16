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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use diskann::{
    graph::{
        config::Builder as ConfigBuilder, config::MaxDegree, config::PruneKind,
        search_output_buffer, DiskANNIndex, SearchParams, StartPointStrategy,
    },
    provider::DefaultContext,
    utils::{async_tools, IntoUsize},
};
use diskann_benchmark_runner::{
    dispatcher::{DispatchRule, FailureScore, MatchScore},
    output::Output,
    registry::Benchmarks,
    utils::{datatype::DataType, percentiles, MicroSeconds},
    Any, Checkpoint,
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
use diskann_utils::views::Matrix;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::{
    inputs::document_index::DocumentIndexBuild,
    utils::{
        self,
        datafiles::{self, BinFile},
        recall,
    },
};

/// Register the document index benchmarks.
pub(crate) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    benchmarks.register::<DocumentIndexJob<'static>>(
        "document-index-build",
        |job, checkpoint, out| {
            let stats = job.run(checkpoint, out)?;
            Ok(serde_json::to_value(stats)?)
        },
    );
}

/// Document index benchmark job.
pub(super) struct DocumentIndexJob<'a> {
    input: &'a DocumentIndexBuild,
}

impl<'a> DocumentIndexJob<'a> {
    fn new(input: &'a DocumentIndexBuild) -> Self {
        Self { input }
    }
}

impl diskann_benchmark_runner::dispatcher::Map for DocumentIndexJob<'static> {
    type Type<'a> = DocumentIndexJob<'a>;
}

// Dispatch from the concrete input type
impl<'a> DispatchRule<&'a DocumentIndexBuild> for DocumentIndexJob<'a> {
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a DocumentIndexBuild) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(1))
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
impl<'a> DispatchRule<&'a Any> for DocumentIndexJob<'a> {
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

/// Compute the index of the row closest to the medoid (centroid) of the data.
fn compute_medoid_index<T>(data: &Matrix<T>) -> usize
where
    T: bytemuck::Pod + Copy + 'static,
{
    use diskann_vector::{distance::SquaredL2, PureDistanceFunction};

    let dim = data.ncols();
    if dim == 0 || data.nrows() == 0 {
        return 0;
    }

    // Compute the centroid (mean of all rows) as f64 for precision
    let mut sum = vec![0.0f64; dim];
    for i in 0..data.nrows() {
        let row = data.row(i);
        for (j, &v) in row.iter().enumerate() {
            // Convert T to f64 for summation using bytemuck
            let f64_val: f64 = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                let f32_val: f32 = bytemuck::cast(v);
                f32_val as f64
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
                let u8_val: u8 = bytemuck::cast(v);
                u8_val as f64
            } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
                let i8_val: i8 = bytemuck::cast(v);
                i8_val as f64
            } else {
                0.0
            };
            sum[j] += f64_val;
        }
    }

    // Convert centroid to f32 and compute distances
    let centroid_f32: Vec<f32> = sum
        .iter()
        .map(|s| (s / data.nrows() as f64) as f32)
        .collect();

    // Find the row closest to the centroid
    let mut min_dist = f32::MAX;
    let mut medoid_idx = 0;
    for i in 0..data.nrows() {
        let row = data.row(i);
        let row_f32: Vec<f32> = row
            .iter()
            .map(|&v| {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    bytemuck::cast(v)
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
                    let u8_val: u8 = bytemuck::cast(v);
                    u8_val as f32
                } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i8>() {
                    let i8_val: i8 = bytemuck::cast(v);
                    i8_val as f32
                } else {
                    0.0
                }
            })
            .collect();
        let d = SquaredL2::evaluate(centroid_f32.as_slice(), row_f32.as_slice());
        if d < min_dist {
            min_dist = d;
            medoid_idx = i;
        }
    }

    medoid_idx
}

impl<'a> DocumentIndexJob<'a> {
    fn run(
        self,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> Result<DocumentIndexStats, anyhow::Error> {
        // Print the input description
        writeln!(output, "{}", self.input)?;

        let build = &self.input.build;

        // Dispatch based on data type - retain original type without conversion
        match build.data_type {
            DataType::Float32 => self.run_typed::<f32>(output),
            DataType::UInt8 => self.run_typed::<u8>(output),
            DataType::Int8 => self.run_typed::<i8>(output),
            _ => Err(anyhow::anyhow!(
                "Unsupported data type: {:?}. Supported types: float32, uint8, int8.",
                build.data_type
            )),
        }
    }

    fn run_typed<T>(self, mut output: &mut dyn Output) -> Result<DocumentIndexStats, anyhow::Error>
    where
        T: bytemuck::Pod + Copy + Send + Sync + 'static + std::fmt::Debug,
        T: diskann::graph::SampleableForStart + diskann_utils::future::AsyncFriendly,
        T: diskann::utils::VectorRepr + diskann_utils::sampling::WithApproximateNorm,
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
        let medoid_idx = compute_medoid_index(&data);
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

        let insert_strategy: DocumentInsertStrategy<_, [T]> =
            DocumentInsertStrategy::new(common::FullPrecision);
        let rt = utils::tokio::runtime(build.num_threads)?;

        // Create control block for parallel work distribution
        let data_arc = Arc::new(data);
        let attributes_arc = Arc::new(attributes);
        let control_block = DocumentControlBlock::new(
            data_arc.clone(),
            attributes_arc.clone(),
            output.draw_target(),
        )?;

        let num_tasks = build.num_threads;
        let insert_latencies = rt.block_on(async {
            let tasks: Vec<_> = (0..num_tasks)
                .map(|_| {
                    let block = control_block.clone();
                    let index = doc_index.clone();
                    let strategy = insert_strategy;
                    tokio::spawn(async move {
                        let mut latencies = Vec::<MicroSeconds>::new();
                        let ctx = DefaultContext;
                        loop {
                            match block.next() {
                                Some((id, vector, attrs)) => {
                                    let doc = Document::new(vector, attrs);
                                    let start = std::time::Instant::now();
                                    let result =
                                        index.insert(strategy, &ctx, &(id as u32), &doc).await;
                                    latencies.push(MicroSeconds::from(start.elapsed()));

                                    if let Err(e) = result {
                                        block.cancel();
                                        return Err(e);
                                    }
                                }
                                None => return Ok(latencies),
                            }
                        }
                    })
                })
                .collect();

            // Collect results from all tasks
            let mut all_latencies = Vec::with_capacity(num_vectors);
            for task in tasks {
                let task_latencies = task.await??;
                all_latencies.extend(task_latencies);
            }
            Ok::<_, anyhow::Error>(all_latencies)
        })?;

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
/// Local results from a partition of queries.
struct SearchLocalResults {
    ids: Matrix<u32>,
    distances: Vec<Vec<f32>>,
    latencies: Vec<MicroSeconds>,
    comparisons: Vec<u32>,
    hops: Vec<u32>,
}

impl SearchLocalResults {
    fn merge(all: &[SearchLocalResults]) -> anyhow::Result<Self> {
        let first = all
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty results"))?;
        let num_ids = first.ids.ncols();
        let total_rows: usize = all.iter().map(|r| r.ids.nrows()).sum();

        let mut ids = Matrix::new(0, total_rows, num_ids);
        let mut output_row = 0;
        for r in all {
            for input_row in r.ids.row_iter() {
                ids.row_mut(output_row).copy_from_slice(input_row);
                output_row += 1;
            }
        }

        let mut distances = Vec::new();
        let mut latencies = Vec::new();
        let mut comparisons = Vec::new();
        let mut hops = Vec::new();
        for r in all {
            distances.extend_from_slice(&r.distances);
            latencies.extend_from_slice(&r.latencies);
            comparisons.extend_from_slice(&r.comparisons);
            hops.extend_from_slice(&r.hops);
        }

        Ok(Self {
            ids,
            distances,
            latencies,
            comparisons,
            hops,
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
    let rt = utils::tokio::runtime(num_threads.get())?;
    let num_queries = queries.nrows();

    let mut all_rep_results = Vec::with_capacity(reps.get());
    let mut rep_latencies = Vec::with_capacity(reps.get());

    for _ in 0..reps.get() {
        let start = std::time::Instant::now();
        let results = rt.block_on(run_search_parallel(
            index.clone(),
            queries,
            predicates,
            beta,
            num_threads,
            search_n,
            search_l,
        ))?;
        rep_latencies.push(MicroSeconds::from(start.elapsed()));
        all_rep_results.push(results);
    }

    // Merge results from first rep for recall calculation
    let merged = SearchLocalResults::merge(&all_rep_results[0])?;

    // Compute recall
    let recall_metrics: recall::RecallMetrics =
        (&recall::knn(groundtruth, None, &merged.ids, recall_k, search_n, false)?).into();

    // Compute per-query details (only for queries with recall < 1)
    let per_query_details: Vec<PerQueryDetails> = (0..num_queries)
        .filter_map(|query_idx| {
            let result_ids: Vec<u32> = merged
                .ids
                .row(query_idx)
                .iter()
                .copied()
                .filter(|&id| id != u32::MAX)
                .collect();
            let result_distances: Vec<f32> = merged
                .distances
                .get(query_idx)
                .map(|d| d.iter().copied().filter(|&dist| dist != f32::MAX).collect())
                .unwrap_or_default();
            // Only keep top 20 from ground truth
            let gt_ids: Vec<u32> = groundtruth
                .get(query_idx)
                .map(|gt| gt.iter().take(20).copied().collect())
                .unwrap_or_default();

            // Compute per-query recall: intersection of result_ids with gt_ids / recall_k
            let result_set: std::collections::HashSet<u32> = result_ids.iter().copied().collect();
            let gt_set: std::collections::HashSet<u32> =
                gt_ids.iter().take(recall_k).copied().collect();
            let intersection = result_set.intersection(&gt_set).count();
            let per_query_recall = if gt_set.is_empty() {
                1.0
            } else {
                intersection as f64 / gt_set.len() as f64
            };

            // Only include queries with imperfect recall
            if per_query_recall >= 1.0 {
                return None;
            }

            let (_, ref ast_expr) = predicates[query_idx];
            let filter_str = format!("{:?}", ast_expr);

            Some(PerQueryDetails {
                query_id: query_idx,
                filter: filter_str,
                recall: per_query_recall,
                result_ids,
                result_distances,
                groundtruth_ids: gt_ids,
            })
        })
        .collect();

    // Compute QPS from rep latencies
    let qps: Vec<f64> = rep_latencies
        .iter()
        .map(|l| num_queries as f64 / l.as_seconds())
        .collect();

    // Aggregate per-query latencies across all reps
    let (all_latencies, all_cmps, all_hops): (Vec<_>, Vec<_>, Vec<_>) = all_rep_results
        .iter()
        .map(|results| {
            let mut lat = Vec::new();
            let mut cmp = Vec::new();
            let mut hop = Vec::new();
            for r in results {
                lat.extend_from_slice(&r.latencies);
                cmp.extend_from_slice(&r.comparisons);
                hop.extend_from_slice(&r.hops);
            }
            (lat, cmp, hop)
        })
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut a, mut b, mut c): (Vec<MicroSeconds>, Vec<u32>, Vec<u32>), (x, y, z)| {
                a.extend(x);
                b.extend(y);
                c.extend(z);
                (a, b, c)
            },
        );

    let mut query_latencies = all_latencies;
    let percentiles::Percentiles { mean, p90, p99, .. } =
        percentiles::compute_percentiles(&mut query_latencies)?;

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
        num_threads: num_threads.get(),
        num_queries,
        search_n,
        search_l,
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
async fn run_search_parallel<DP, T>(
    index: Arc<DiskANNIndex<DP>>,
    queries: &Matrix<T>,
    predicates: &[(usize, ASTExpr)],
    beta: f32,
    num_tasks: NonZeroUsize,
    search_n: usize,
    search_l: usize,
) -> anyhow::Result<Vec<SearchLocalResults>>
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
    let num_queries = queries.nrows();

    // Plan query partitions
    let partitions: Result<Vec<_>, _> = (0..num_tasks.get())
        .map(|task_id| async_tools::partition(num_queries, num_tasks, task_id))
        .collect();
    let partitions = partitions?;

    // We need to clone data for each task
    let queries_arc = Arc::new(queries.clone());
    let predicates_arc = Arc::new(predicates.to_vec());

    let handles: Vec<_> = partitions
        .into_iter()
        .map(|range| {
            let index = index.clone();
            let queries = queries_arc.clone();
            let predicates = predicates_arc.clone();
            tokio::spawn(async move {
                run_search_local(index, queries, predicates, beta, range, search_n, search_l).await
            })
        })
        .collect();

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await??);
    }

    Ok(results)
}

async fn run_search_local<DP, T>(
    index: Arc<DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    predicates: Arc<Vec<(usize, ASTExpr)>>,
    beta: f32,
    range: std::ops::Range<usize>,
    search_n: usize,
    search_l: usize,
) -> anyhow::Result<SearchLocalResults>
where
    T: bytemuck::Pod + Copy + Send + Sync + 'static,
    DP: diskann::provider::DataProvider<
            Context = DefaultContext,
            ExternalId = u32,
            InternalId = u32,
        > + Send
        + Sync,
    InlineBetaStrategy<common::FullPrecision>:
        diskann::graph::glue::SearchStrategy<DP, FilteredQuery<Vec<T>>>,
{
    let mut ids = Matrix::new(0, range.len(), search_n);
    let mut all_distances: Vec<Vec<f32>> = Vec::with_capacity(range.len());
    let mut latencies = Vec::with_capacity(range.len());
    let mut comparisons = Vec::with_capacity(range.len());
    let mut hops = Vec::with_capacity(range.len());

    let ctx = DefaultContext;
    let search_params = SearchParams::new_default(search_n, search_l)?;

    for (output_idx, query_idx) in range.enumerate() {
        let query_vec = queries.row(query_idx);
        let (_, ref ast_expr) = predicates[query_idx];

        let strategy = InlineBetaStrategy::new(beta, common::FullPrecision);
        let query_vec_owned = query_vec.to_vec();
        let filtered_query: FilteredQuery<Vec<T>> =
            FilteredQuery::new(query_vec_owned, ast_expr.clone());

        let start = std::time::Instant::now();

        let mut distances = vec![0.0f32; search_n];
        let result_ids = ids.row_mut(output_idx);
        let mut result_buffer = search_output_buffer::IdDistance::new(result_ids, &mut distances);

        let stats = index
            .search(
                &strategy,
                &ctx,
                &filtered_query,
                &search_params,
                &mut result_buffer,
            )
            .await?;

        let result_count = stats.result_count.into_usize();
        result_ids[result_count..].fill(u32::MAX);
        distances[result_count..].fill(f32::MAX);

        latencies.push(MicroSeconds::from(start.elapsed()));
        comparisons.push(stats.cmps);
        hops.push(stats.hops);
        all_distances.push(distances);
    }

    Ok(SearchLocalResults {
        ids,
        distances: all_distances,
        latencies,
        comparisons,
        hops,
    })
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
    pub recall: recall::RecallMetrics,
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
                "L", "KNN", "Avg Cmps", "Avg Hops", "QPS -mean(max)", "Avg Latency", "p99 Latency", "Recall", "Threads", "Queries", "WallClock(s)"
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
                    s.wall_clock_time.iter().map(|t| t.as_seconds()).sum::<f64>() / s.wall_clock_time.len() as f64
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

fn make_progress_bar(
    nrows: usize,
    draw_target: indicatif::ProgressDrawTarget,
) -> anyhow::Result<ProgressBar> {
    let progress = ProgressBar::with_draw_target(Some(nrows as u64), draw_target);
    progress.set_style(ProgressStyle::with_template(
        "Building [{elapsed_precise}] {wide_bar} {percent}",
    )?);
    Ok(progress)
}

/// Control block for parallel document insertion.
/// Manages work distribution and progress tracking across multiple tasks.
struct DocumentControlBlock<T> {
    data: Arc<Matrix<T>>,
    attributes: Arc<Vec<Vec<Attribute>>>,
    position: AtomicUsize,
    cancel: AtomicBool,
    progress: ProgressBar,
}

impl<T> DocumentControlBlock<T> {
    fn new(
        data: Arc<Matrix<T>>,
        attributes: Arc<Vec<Vec<Attribute>>>,
        draw_target: indicatif::ProgressDrawTarget,
    ) -> anyhow::Result<Arc<Self>> {
        let nrows = data.nrows();
        Ok(Arc::new(Self {
            data,
            attributes,
            position: AtomicUsize::new(0),
            cancel: AtomicBool::new(false),
            progress: make_progress_bar(nrows, draw_target)?,
        }))
    }

    /// Return the next document data to insert: (id, vector_slice, attributes).
    fn next(&self) -> Option<(usize, &[T], Vec<Attribute>)> {
        let cancel = self.cancel.load(Ordering::Relaxed);
        if cancel {
            None
        } else {
            let i = self.position.fetch_add(1, Ordering::Relaxed);
            match self.data.get_row(i) {
                Some(row) => {
                    let attrs = self.attributes.get(i).cloned().unwrap_or_default();
                    self.progress.inc(1);
                    Some((i, row, attrs))
                }
                None => None,
            }
        }
    }

    /// Tell all users of the control block to cancel and return early.
    fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl<T> Drop for DocumentControlBlock<T> {
    fn drop(&mut self) {
        self.progress.finish();
    }
}
