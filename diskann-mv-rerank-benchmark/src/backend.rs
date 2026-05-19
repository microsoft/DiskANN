/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! End-to-end multi-vector rerank benchmark (f32 element type).
//!
//! Pipeline:
//! 1. Build (or load) a full-precision DiskANN graph on **single-vector** embeddings.
//! 2. Run a top-N graph-walk search per query.
//! 3. Rerank the top-N candidates using **multi-vector** Chamfer, with both the
//!    doc multi-vectors and the per-query multi-vector loaded from `.mvbin` files
//!    independent of the index.
//!
//! Adopted from `origin/users/mhildebr/multi-vector`:
//! - `.mvbin` data format (see [`crate::datafiles`]).
//! - Side-loaded reranker shape (this crate's [`MultiVectorReranker`]).
//! - [`TimedPostProcessor`] wrapper for splitting first-stage and rerank latency.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use diskann::{
    graph::{
        glue::{FilterStartPoints, Pipeline, SearchPostProcess},
        search::Knn,
        DiskANNIndex, SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DefaultContext, HasId},
    ANNResult,
};
use diskann_benchmark_core::{self as benchmark_core, build as build_core, search as core_search};
use diskann_benchmark_runner::{
    output::Output,
    utils::{fmt::Table, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Registry,
};
use diskann_providers::{
    index::diskann_async,
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::{
            common::{FullPrecision as FullPrecisionStrategy, NoDeletes, NoStore},
            inmem::{DefaultProvider, FullPrecisionStore, SetStartPoints},
        },
    },
    storage::{AsyncIndexMetadata, FileStorageProvider, LoadWith, SaveWith},
};
use diskann_quantization::multi_vector::{Mat, MatRef, QueryComputer, Standard};
use diskann_utils::views::Matrix;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::{
    datafiles::{self, BinFile},
    inputs::{
        match_data_type, BuildConfig, MultiVectorRerankOperation, Source, TopkSearchPhase,
    },
};

//////////////
// Saveload //
//////////////

pub(crate) mod saveload {
    use std::{io::Read, mem::size_of, num::NonZeroUsize};

    use diskann::{ANNError, ANNResult};
    use diskann_providers::storage::StorageReadProvider;

    pub(crate) fn get_graph_num_frozen_points(
        storage_provider: &impl StorageReadProvider,
        graph_file: &str,
    ) -> ANNResult<NonZeroUsize> {
        let mut file = storage_provider.open_reader(graph_file)?;
        let mut usize_buffer = [0; size_of::<usize>()];
        let mut u32_buffer = [0; size_of::<u32>()];

        file.read_exact(&mut usize_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        file.read_exact(&mut usize_buffer)?;
        let file_frozen_pts = usize::from_le_bytes(usize_buffer);

        NonZeroUsize::new(file_frozen_pts).ok_or_else(|| {
            ANNError::log_index_config_error(
                "num_frozen_pts".to_string(),
                "num_frozen_pts is zero in saved file".to_string(),
            )
        })
    }

    pub(crate) fn get_graph_max_observed_degree(
        storage_provider: &impl StorageReadProvider,
        graph_file: &str,
    ) -> ANNResult<u32> {
        let mut file = storage_provider.open_reader(graph_file)?;
        let mut usize_buffer = [0; size_of::<usize>()];
        let mut u32_buffer = [0; size_of::<u32>()];

        file.read_exact(&mut usize_buffer)?;
        file.read_exact(&mut u32_buffer)?;
        let max_observed_degree = u32::from_le_bytes(u32_buffer);

        Ok(max_observed_degree)
    }
}

///////////////////
// Registration //
///////////////////

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("multi-vector-rerank-f32", MultiVectorRerankF32)?;
    Ok(())
}

////////////////////////
// Benchmark struct   //
////////////////////////

pub(crate) struct MultiVectorRerankF32;

impl Benchmark for MultiVectorRerankF32 {
    type Input = MultiVectorRerankOperation;
    type Output = BuildResult;

    fn try_match(
        &self,
        input: &MultiVectorRerankOperation,
    ) -> Result<
        diskann_benchmark_runner::benchmark::MatchScore,
        diskann_benchmark_runner::benchmark::FailureScore,
    > {
        match_data_type::<f32>(*input.source.data_type())
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&MultiVectorRerankOperation>,
    ) -> std::fmt::Result {
        use diskann_benchmark_runner::utils::datatype::AsDataType;
        match input {
            Some(i) => {
                let desc = <f32 as AsDataType>::describe(*i.source.data_type());
                if !desc.is_match() {
                    writeln!(f, "Data type: {}", desc)?;
                }
                Ok(())
            }
            None => writeln!(f, "Data type: {}", <f32 as AsDataType>::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &MultiVectorRerankOperation,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<BuildResult> {
        use std::io::Write;
        writeln!(output, "{}", input)?;
        // One io-only runtime shared by save and load. Cheaper than building two.
        let io_rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let (index, build_stats, doc_count) = match &input.source {
            Source::Build(b) => {
                let (idx, stats, n) = build_index_f32(b, output)?;
                if let Some(save_path) = &b.save_path {
                    io_rt.block_on(save_index_async(idx.clone(), save_path))?;
                }
                (idx, Some(stats), n)
            }
            Source::Load(l) => {
                let config = l.to_config()?;
                // max_points includes frozen (medoid) points reserved by the provider;
                // subtract to get the corpus size, i.e. the range of ids the search returns.
                let doc_count = config
                    .max_points
                    .checked_sub(config.num_frozen_pts.get())
                    .ok_or_else(|| anyhow::anyhow!("loaded index reports more frozen points than total"))?;
                let idx = io_rt.block_on(load_index_async(&l.load_path, &config))?;
                (Arc::new(idx), None, doc_count)
            }
        };

        checkpoint.checkpoint(&build_stats)?;

        let summaries = run_search(index.clone(), &input.search, input, doc_count)?;
        let search_results: Vec<SearchResults> =
            summaries.into_iter().map(SearchResults::new).collect();

        let result = BuildResult::new(build_stats, search_results);
        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}

type ProviderF32 = DefaultProvider<FullPrecisionStore<f32>, NoStore, NoDeletes, DefaultContext>;
type MemoryIndexF32 = Arc<DiskANNIndex<ProviderF32>>;

///////////
// Build //
///////////

fn build_index_f32(
    build: &BuildConfig,
    output: &mut dyn Output,
) -> anyhow::Result<(MemoryIndexF32, BuildStats, usize)> {
    let data: Arc<Matrix<f32>> =
        Arc::new(datafiles::load_dataset::<f32>(BinFile(build.data.as_ref()))?);
    let dim = data.ncols();
    let npoints = data.nrows();

    let config = build.try_as_config()?.build()?;
    let params = build.inmem_parameters(npoints, dim);

    let index: MemoryIndexF32 = diskann_async::new_index::<f32, _>(config, params, NoDeletes)?;

    let start_points = build
        .start_point_strategy
        .compute(data.as_view())
        .map_err(|e| anyhow::anyhow!("failed to compute start points: {}", e))?;
    index.provider().set_start_points(start_points.row_iter())?;

    let runtime = benchmark_core::tokio::runtime(build.num_threads)?;
    let runner = build_core::graph::SingleInsert::new(
        index.clone(),
        data,
        FullPrecisionStrategy,
        build_core::ids::Identity::<u32>::new(),
    );
    let results = build_core::build_tracked(
        runner,
        build_core::Parallelism::dynamic(
            diskann::utils::ONE,
            NonZeroUsize::new(build.num_threads).unwrap(),
        ),
        &runtime,
        Some(&ProgressMeter::new(output)),
    )?;

    let stats = BuildStats::new(results)?;
    Ok((index, stats, npoints))
}

async fn save_index_async(index: MemoryIndexF32, save_path: &str) -> anyhow::Result<()> {
    index
        .save_with(&FileStorageProvider, &AsyncIndexMetadata::new(save_path))
        .await?;
    Ok(())
}

async fn load_index_async(
    load_path: &str,
    index_config: &IndexConfiguration,
) -> anyhow::Result<DiskANNIndex<ProviderF32>> {
    let index = DiskANNIndex::<ProviderF32>::load_with(
        &FileStorageProvider,
        &(load_path, index_config.clone()),
    )
    .await?;
    Ok(index)
}

////////////////////////
// Reranker + Search  //
////////////////////////

/// Side-loaded multi-vector reranker. For each query we construct one instance with the
/// per-query [`MatRef<Standard<f32>>`] borrowed from a preallocated `Arc<[Mat<...>]>` of
/// query multi-vectors, and a shared `Arc<[Mat<...>]>` of document multi-vectors.
///
/// In `post_process` we score every candidate the graph walk produced via
/// [`QueryComputer::chamfer`] (architecture-dispatched MaxSim), sort ascending
/// (`QueryComputer::chamfer` returns negated max IP so lower-is-better), and forward
/// the resulting `(id, score)` pairs to the output buffer.
pub(crate) struct MultiVectorReranker<'a> {
    pub(crate) doc_mv: Arc<[Mat<Standard<f32>>]>,
    pub(crate) query_mv: MatRef<'a, Standard<f32>>,
}

impl<A, Q> SearchPostProcess<A, Q> for MultiVectorReranker<'_>
where
    A: BuildQueryComputer<Q> + HasId<Id = u32>,
    Q: Send,
{
    type Error = std::convert::Infallible;

    async fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: Q,
        _computer: &<A as BuildQueryComputer<Q>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> Result<usize, Self::Error>
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        // Score every candidate the graph walk produced (the iter yields up to L items
        // from the size-L priority queue at scratch.best — see diskann::graph::knn_search).
        // Architecture-dispatched fast path: detect CPU arch + capture block-transposed
        // query once here, then every `qc.chamfer(doc)` call goes through
        // `Architecture::run3` with target-feature propagation — no per-candidate dispatch.
        // See `diskann-quantization/src/multi_vector/distance/query_computer/mod.rs`.
        let qc = QueryComputer::<f32>::new(self.query_mv);
        let mut scored: Vec<(u32, f32)> = candidates
            .map(|n| {
                let doc = self.doc_mv[n.id as usize].as_view();
                let score = qc.chamfer(doc);
                (n.id, score)
            })
            .collect();

        // Ascending — Chamfer already negates MaxSim sum so lower-is-better.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let written = output.extend(scored.into_iter());
        Ok(written)
    }
}

/// Wrap a [`SearchPostProcess`] with elapsed-time tracking. Each call writes the elapsed
/// micros into a shared `Arc<AtomicU64>` so the outer search loop can report rerank
/// latency separately from end-to-end. Ported from
/// `diskann-benchmark/src/backend/index/multi.rs:230-280` on `origin/users/mhildebr/multi-vector`.
pub(crate) struct TimedPostProcessor<P> {
    inner: P,
    elapsed_micros: Arc<AtomicU64>,
}

impl<P> TimedPostProcessor<P> {
    fn new(inner: P, elapsed_micros: Arc<AtomicU64>) -> Self {
        Self {
            inner,
            elapsed_micros,
        }
    }
}

impl<A, Q, O, P> SearchPostProcess<A, Q, O> for TimedPostProcessor<P>
where
    A: BuildQueryComputer<Q>,
    O: Send,
    P: SearchPostProcess<A, Q, O> + Send + Sync,
{
    type Error = P::Error;

    fn post_process<I, B>(
        &self,
        accessor: &mut A,
        query: Q,
        computer: &<A as BuildQueryComputer<Q>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<A::Id>> + Send,
        B: SearchOutputBuffer<O> + Send + ?Sized,
    {
        let elapsed_micros = Arc::clone(&self.elapsed_micros);
        let future = self
            .inner
            .post_process(accessor, query, computer, candidates, output);
        async move {
            let start = std::time::Instant::now();
            let result = future.await;
            elapsed_micros.store(start.elapsed().as_micros() as u64, Ordering::Relaxed);
            result
        }
    }
}

////////////////////////
// MultiVectorKNN    //
////////////////////////

/// `core_search::Search` impl that drives one query at a time through
/// `DiskANNIndex::search_with`, constructing a per-query [`MultiVectorReranker`] and
/// wrapping it with [`FilterStartPoints`] + [`TimedPostProcessor`].
pub(crate) struct MultiVectorKNN {
    index: MemoryIndexF32,
    queries: Arc<Matrix<f32>>,
    doc_mv: Arc<[Mat<Standard<f32>>]>,
    query_mv: Arc<[Mat<Standard<f32>>]>,
}

impl MultiVectorKNN {
    fn new(
        index: MemoryIndexF32,
        queries: Arc<Matrix<f32>>,
        doc_mv: Arc<[Mat<Standard<f32>>]>,
        query_mv: Arc<[Mat<Standard<f32>>]>,
    ) -> anyhow::Result<Arc<Self>> {
        if query_mv.len() != queries.nrows() {
            anyhow::bail!(
                "query .mvbin has {} records but single-vec queries .fbin has {} rows",
                query_mv.len(),
                queries.nrows()
            );
        }
        Ok(Arc::new(Self {
            index,
            queries,
            doc_mv,
            query_mv,
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MvMetrics {
    pub(crate) comparisons: u32,
    pub(crate) hops: u32,
    pub(crate) rerank_micros: u64,
}

impl core_search::Search for MultiVectorKNN {
    type Id = u32;
    type Parameters = Knn;
    type Output = MvMetrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> core_search::IdCount {
        core_search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: SearchOutputBuffer<u32> + Send,
    {
        let context = DefaultContext;
        let rerank_micros = Arc::new(AtomicU64::new(0));
        let reranker = MultiVectorReranker {
            doc_mv: self.doc_mv.clone(),
            query_mv: self.query_mv[index].as_view(),
        };
        let pipeline = Pipeline::new(FilterStartPoints, reranker);
        let processor = TimedPostProcessor::new(pipeline, Arc::clone(&rerank_micros));

        let stats = self
            .index
            .search_with(
                *parameters,
                &FullPrecisionStrategy,
                processor,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(MvMetrics {
            comparisons: stats.cmps,
            hops: stats.hops,
            rerank_micros: rerank_micros.load(Ordering::Relaxed),
        })
    }
}

/// BEIR-style recall@N where N = `search_n`. For each query:
///   - count `hits` = |top_N_results ∩ relevant_set|
///   - per_query_recall = hits / min(N, |relevant|)
///   - average across queries with at least one positive judgment
///
/// Matches the user's eval-script `recall_at_k(...)` semantics
/// (`hits / min(k, len(relevant))`). Queries with no relevant docs are skipped.
fn beir_recall_at_n(
    gt: &[Vec<u32>],
    results: &core_search::ResultIds<u32>,
    n: usize,
) -> RecallSummary {
    let rows = results.as_rows();
    let nqueries = rows.nrows().min(gt.len());

    let mut sum = 0.0;
    let mut min_r = f64::INFINITY;
    let mut max_r = 0.0f64;
    let mut scored = 0usize;

    for (i, relevant) in gt.iter().take(nqueries).enumerate() {
        if relevant.is_empty() {
            continue;
        }
        let rel_set: std::collections::HashSet<u32> = relevant.iter().copied().collect();
        let ranked = rows.row(i);
        let hits = ranked
            .iter()
            .take(n)
            .filter(|id| rel_set.contains(id))
            .count();
        let denom = n.min(relevant.len()) as f64;
        let r = hits as f64 / denom;
        sum += r;
        if r < min_r {
            min_r = r;
        }
        if r > max_r {
            max_r = r;
        }
        scored += 1;
    }

    if scored == 0 {
        RecallSummary {
            average: 0.0,
            minimum: 0.0,
            maximum: 0.0,
            num_queries: 0,
        }
    } else {
        RecallSummary {
            average: sum / scored as f64,
            minimum: min_r,
            maximum: max_r,
            num_queries: scored,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RecallSummary {
    pub(crate) average: f64,
    pub(crate) minimum: f64,
    pub(crate) maximum: f64,
    pub(crate) num_queries: usize,
}

/// Folds the per-rep [`core_search::SearchResults`] for one (search_l, num_threads, run)
/// triple into a single [`MvSummary`]: BEIR recall@N (N = `search_n`), per-query latency
/// percentiles, mean rerank-time slice, and mean cmps/hops.
pub(crate) struct Aggregator<'a> {
    groundtruth: &'a [Vec<u32>],
}

impl<'a> Aggregator<'a> {
    fn new(groundtruth: &'a [Vec<u32>]) -> Self {
        Self { groundtruth }
    }
}

impl core_search::Aggregate<Knn, u32, MvMetrics> for Aggregator<'_> {
    type Output = MvSummary;

    fn aggregate(
        &mut self,
        run: core_search::Run<Knn>,
        mut results: Vec<core_search::SearchResults<u32, MvMetrics>>,
    ) -> anyhow::Result<MvSummary> {
        let search_n = run.parameters().k_value().get();
        let recall = match results.first() {
            Some(first) => beir_recall_at_n(self.groundtruth, first.ids(), search_n),
            None => anyhow::bail!("Results must be non-empty"),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());
        let mut mean_rerank_latencies = Vec::with_capacity(results.len());

        for r in results.iter_mut() {
            let (sum_rerank, n) = r
                .output()
                .iter()
                .fold((0u64, 0usize), |(s, n), o| (s + o.rerank_micros, n + 1));
            let mean_rerank = if n > 0 {
                sum_rerank as f64 / n as f64
            } else {
                0.0
            };

            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(p) => {
                    mean_latencies.push(p.mean);
                    p90_latencies.push(p.p90);
                    p99_latencies.push(p.p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
            mean_rerank_latencies.push(mean_rerank);
        }

        let (sum_cmps, sum_hops, n) = results.iter().flat_map(|r| r.output().iter()).fold(
            (0u64, 0u64, 0usize),
            |(c, h, n), o| (c + o.comparisons as u64, h + o.hops as u64, n + 1),
        );
        let mean_cmps = if n > 0 { sum_cmps as f64 / n as f64 } else { 0.0 };
        let mean_hops = if n > 0 { sum_hops as f64 / n as f64 } else { 0.0 };

        Ok(MvSummary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_rerank_latencies,
            recall_avg: recall.average,
            recall_min: recall.minimum,
            recall_max: recall.maximum,
            num_queries: recall.num_queries,
            mean_cmps,
            mean_hops,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MvSummary {
    pub(crate) setup: core_search::Setup,
    pub(crate) parameters: Knn,
    pub(crate) end_to_end_latencies: Vec<MicroSeconds>,
    pub(crate) mean_latencies: Vec<f64>,
    pub(crate) p90_latencies: Vec<MicroSeconds>,
    pub(crate) p99_latencies: Vec<MicroSeconds>,
    pub(crate) mean_rerank_latencies: Vec<f64>,
    pub(crate) recall_avg: f64,
    pub(crate) recall_min: f64,
    pub(crate) recall_max: f64,
    pub(crate) num_queries: usize,
    pub(crate) mean_cmps: f64,
    pub(crate) mean_hops: f64,
}

fn run_search(
    index: MemoryIndexF32,
    search: &TopkSearchPhase,
    op: &MultiVectorRerankOperation,
    doc_count: usize,
) -> anyhow::Result<Vec<MvSummary>> {
    let queries: Arc<Matrix<f32>> =
        Arc::new(datafiles::load_dataset::<f32>(BinFile(search.queries.as_ref()))?);
    let groundtruth = datafiles::load_groundtruth(BinFile(search.groundtruth.as_ref()))?;
    let doc_mv: Arc<[Mat<Standard<f32>>]> =
        Arc::from(datafiles::load_multi_vectors::<f32>(&op.doc_mv)?);
    let query_mv: Arc<[Mat<Standard<f32>>]> =
        Arc::from(datafiles::load_multi_vectors::<f32>(&op.query_mv)?);

    if doc_mv.is_empty() {
        anyhow::bail!("doc .mvbin file produced 0 records");
    }
    if doc_mv.len() != doc_count {
        anyhow::bail!(
            "doc .mvbin has {} records but the single-vector index holds {} docs; \
             these must match because the reranker indexes doc_mv[id] by the graph's internal id",
            doc_mv.len(),
            doc_count,
        );
    }
    if query_mv.len() != queries.nrows() {
        anyhow::bail!(
            "query .mvbin has {} records but query .fbin has {} rows",
            query_mv.len(),
            queries.nrows()
        );
    }
    let doc_mv_dim = doc_mv[0].vector_dim();
    let query_mv_dim = query_mv[0].vector_dim();
    if doc_mv_dim != query_mv_dim {
        anyhow::bail!(
            "doc .mvbin per-vector dim ({}) != query .mvbin per-vector dim ({}); \
             Chamfer scoring requires matching D",
            doc_mv_dim,
            query_mv_dim,
        );
    }

    let runner = MultiVectorKNN::new(index, queries, doc_mv, query_mv)?;

    let mut all: Vec<MvSummary> = Vec::new();
    for threads in &search.num_threads {
        for run in &search.runs {
            let setup = core_search::Setup {
                threads: *threads,
                tasks: *threads,
                reps: search.reps,
            };
            let parameters: Vec<_> = run
                .search_l
                .iter()
                .map(|search_l| {
                    let search_params = Knn::new(run.search_n, *search_l, None).unwrap();
                    core_search::Run::new(search_params, setup.clone())
                })
                .collect();

            let agg = Aggregator::new(&groundtruth);
            let results =
                core_search::search_all(runner.clone(), parameters.into_iter(), agg)?;
            Extend::extend(&mut all, results);
        }
    }
    Ok(all)
}

///////////////////////
// Build stats       //
///////////////////////

#[derive(Debug, Serialize)]
pub(crate) struct BuildStats {
    pub(crate) total_time: MicroSeconds,
    pub(crate) vectors_inserted: usize,
    pub(crate) insert_latencies: percentiles::Percentiles<MicroSeconds>,
}

impl BuildStats {
    fn new(results: build_core::BuildResults<()>) -> anyhow::Result<Self> {
        let total_time = results.end_to_end_latency();
        let mut latencies = Vec::new();
        let mut vectors_inserted = 0;
        results.take_output().into_iter().for_each(|r| {
            vectors_inserted += r.batchsize();
            latencies.push(r.latency);
        });
        Ok(Self {
            total_time,
            vectors_inserted,
            insert_latencies: percentiles::compute_percentiles(&mut latencies)?,
        })
    }
}

impl std::fmt::Display for BuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index Build Time: {}s", self.total_time.as_seconds())?;
        writeln!(f, "Vectors Inserted: {}", self.vectors_inserted)?;
        write!(
            f,
            "Insert Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
            self.insert_latencies.mean, self.insert_latencies.p90, self.insert_latencies.p99,
        )
    }
}

struct ProgressMeter<'a> {
    output: &'a mut dyn Output,
}

impl<'a> ProgressMeter<'a> {
    fn new(output: &'a mut dyn Output) -> Self {
        Self { output }
    }
}

impl build_core::AsProgress for ProgressMeter<'_> {
    fn as_progress(&self, max: usize) -> Arc<dyn build_core::Progress> {
        let target = self.output.draw_target();
        let meter = ProgressBar::with_draw_target(Some(max as u64), target);
        meter.set_style(
            ProgressStyle::with_template("Building [{elapsed_precise}] {wide_bar} {percent}")
                .expect("static template is valid"),
        );
        Arc::new(Meter { meter })
    }
}

#[derive(Debug)]
struct Meter {
    meter: ProgressBar,
}

impl build_core::Progress for Meter {
    fn progress(&self, handled: usize) {
        self.meter.inc(handled as u64);
    }
    fn finish(&self) {
        self.meter.finish();
    }
}

///////////////////////
// Result types      //
///////////////////////

#[derive(Debug, Serialize)]
pub(crate) struct BuildResult {
    pub(crate) build: Option<BuildStats>,
    pub(crate) search: Vec<SearchResults>,
}

impl BuildResult {
    fn new(build: Option<BuildStats>, search: Vec<SearchResults>) -> Self {
        Self { build, search }
    }
}

impl std::fmt::Display for BuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(b) = &self.build {
            write!(f, "{}", b)?;
        }
        format_search_table(f, &self.search)
    }
}

fn format_search_table(
    f: &mut std::fmt::Formatter<'_>,
    results: &[SearchResults],
) -> std::fmt::Result {
    if results.is_empty() {
        return Ok(());
    }
    let header = [
        "L",
        "N",
        "Thr",
        "Cmps",
        "Hops",
        "QPS",
        "Mean(us)",
        "p99(us)",
        "Rerank(us)",
        "Recall@N",
    ];
    let mut table = Table::new(header, results.len());
    for (i, r) in results.iter().enumerate() {
        let qps = percentiles::mean(&r.qps).unwrap_or(0.0);
        let mean_lat = percentiles::mean(&r.mean_latencies).unwrap_or(0.0);
        let rerank = percentiles::mean(&r.mean_rerank_latencies).unwrap_or(0.0);
        let p99 = r
            .p99_latencies
            .iter()
            .max()
            .copied()
            .unwrap_or(MicroSeconds::new(0));

        let mut row = table.row(i);
        row.insert(r.search_l, 0);
        row.insert(r.search_n, 1);
        row.insert(r.num_tasks, 2);
        row.insert(format!("{:.0}", r.mean_cmps), 3);
        row.insert(format!("{:.0}", r.mean_hops), 4);
        row.insert(format!("{:.1}", qps), 5);
        row.insert(format!("{:.1}", mean_lat), 6);
        row.insert(format!("{}", p99), 7);
        row.insert(format!("{:.1}", rerank), 8);
        row.insert(format!("{:.4}", r.recall_avg), 9);
    }
    write!(f, "{}", table)
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResults {
    pub(crate) num_tasks: usize,
    pub(crate) search_n: usize,
    pub(crate) search_l: usize,
    pub(crate) qps: Vec<f64>,
    pub(crate) search_latencies: Vec<MicroSeconds>,
    pub(crate) mean_latencies: Vec<f64>,
    pub(crate) p90_latencies: Vec<MicroSeconds>,
    pub(crate) p99_latencies: Vec<MicroSeconds>,
    pub(crate) mean_rerank_latencies: Vec<f64>,
    pub(crate) recall_avg: f64,
    pub(crate) recall_min: f64,
    pub(crate) recall_max: f64,
    /// The N in BEIR recall@N (equal to `search_n`).
    pub(crate) recall_at: usize,
    pub(crate) mean_cmps: f32,
    pub(crate) mean_hops: f32,
}

impl SearchResults {
    fn new(summary: MvSummary) -> Self {
        let MvSummary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_rerank_latencies,
            recall_avg,
            recall_min,
            recall_max,
            num_queries,
            mean_cmps,
            mean_hops,
        } = summary;
        let recall_at = parameters.k_value().get();

        let qps = end_to_end_latencies
            .iter()
            .map(|l| num_queries as f64 / l.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            search_n: parameters.k_value().get(),
            search_l: parameters.l_value().get(),
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_rerank_latencies,
            recall_avg,
            recall_min,
            recall_max,
            recall_at,
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
        }
    }
}


