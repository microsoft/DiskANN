/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Backend for flat-index (brute-force kNN) benchmarks.
//!
//! This exercises [`diskann::flat::FlatIndex::knn_search`] over an in-memory
//! provider, measuring recall and latency.

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann::{
    flat::{DistancesUnordered, FlatIndex, SearchStrategy},
    graph::SearchOutputBuffer,
    provider::{DataProvider, DefaultContext, HasId, NoopGuard},
    utils::VectorRepr,
    ANNResult,
};
use diskann_benchmark_core::{self as benchmark_core, recall::GroundTruthMode, search};
use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    output::Output,
    utils::{datatype::AsDataType, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Registry,
};
use diskann_utils::{future::SendFuture, views::Matrix};
use diskann_vector::{distance::Metric, PreprocessedDistanceFunction};
use half::f16;
use serde::Serialize;

use crate::{
    inputs::flat::FlatSearch,
    utils::{self, datafiles, recall::RecallMetrics},
};

////////////////////////////
// Benchmark Registration //
////////////////////////////

const NAME: &str = "flat-index";

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(NAME, FlatBenchmark::<f32>::new())?;
    registry.register(NAME, FlatBenchmark::<f16>::new())?;
    registry.register(NAME, FlatBenchmark::<u8>::new())?;
    registry.register(NAME, FlatBenchmark::<i8>::new())?;
    Ok(())
}

/////////////////
// FlatSearch  //
/////////////////

/// A minimal in-memory provider for flat search benchmarks.
///
/// Wraps a loaded [`Matrix<T>`] and implements [`DataProvider`] with identity
/// ID mapping.
struct InMemProvider<T> {
    data: Arc<Matrix<T>>,
}

impl<T: VectorRepr> DataProvider for InMemProvider<T> {
    type Context = DefaultContext;
    type InternalId = u32;
    type ExternalId = u32;
    type Error = diskann::ANNError;
    type Guard = NoopGuard<u32>;

    fn to_internal_id(&self, _ctx: &DefaultContext, gid: &u32) -> Result<u32, Self::Error> {
        Ok(*gid)
    }

    fn to_external_id(&self, _ctx: &DefaultContext, id: u32) -> Result<u32, Self::Error> {
        Ok(id)
    }
}

struct FlatBenchmark<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> FlatBenchmark<T> {
    fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> Benchmark for FlatBenchmark<T>
where
    T: VectorRepr + AsDataType,
{
    type Input = FlatSearch;
    type Output = FlatResult;

    fn try_match(&self, input: &FlatSearch) -> Result<MatchScore, FailureScore> {
        utils::match_data_type::<T>(input.data_type)
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&FlatSearch>,
    ) -> std::fmt::Result {
        match input {
            Some(i) => {
                let desc = T::describe(i.data_type);
                if !desc.is_match() {
                    writeln!(f, "Data Type: {}", desc)?;
                }
                Ok(())
            }
            None => writeln!(f, "Data Type: {}", T::DATA_TYPE),
        }
    }

    fn run(
        &self,
        input: &FlatSearch,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<FlatResult> {
        writeln!(output, "{}", input)?;

        let metric: Metric = input.distance.into();

        // Load dataset
        writeln!(output, "Loading dataset...")?;
        let data: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(&input.data))?;
        let nrows = data.nrows();
        let ncols = data.ncols();
        writeln!(output, "  Loaded {} vectors of dimension {}", nrows, ncols)?;

        // Build the provider and wrap in FlatIndex
        let data = Arc::new(data);
        let provider = InMemProvider { data: data.clone() };
        let index = FlatIndex::new(provider);
        let index = Arc::new(index);

        // Load queries and groundtruth
        let queries: Matrix<T> =
            datafiles::load_dataset(datafiles::BinFile(&input.search.queries))?;
        let groundtruth: Matrix<u32> =
            datafiles::load_dataset(datafiles::BinFile(&input.search.groundtruth))?;

        writeln!(
            output,
            "  Queries: {}, Groundtruth: {}x{}",
            queries.nrows(),
            groundtruth.nrows(),
            groundtruth.ncols(),
        )?;

        // Run searches for each thread count
        let k = input.search.k;
        let reps = input.search.reps;

        let mut results = Vec::new();

        for &threads in &input.search.num_threads {
            let searcher = Arc::new(FlatSearcher {
                index: index.clone(),
                queries: Arc::new(queries.clone()),
                metric,
            });

            let setup = search::Setup {
                threads,
                tasks: threads,
                reps,
            };

            let run = search::Run::new(FlatSearchParameters { k }, setup);
            let aggregated = search::search_all(
                searcher,
                std::iter::once(run),
                FlatAggregator::new(&groundtruth, k.get()),
            )?;

            for item in aggregated {
                results.push(item);
            }
        }

        let result = FlatResult { results };
        writeln!(output, "\n\n{}", result)?;
        Ok(result)
    }
}

///////////////////////
// Flat SearchStrategy //
///////////////////////

/// A [`SearchStrategy`] implementation for [`InMemProvider`] that drives
/// a full sequential scan over all vectors.
struct FlatScanStrategy<T: VectorRepr> {
    metric: Metric,
    num_vectors: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: VectorRepr> FlatScanStrategy<T> {
    fn new(metric: Metric, num_vectors: usize) -> Self {
        Self {
            metric,
            num_vectors,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// The visitor that iterates over all vectors in the provider.
struct FlatVisitor<'a, T> {
    data: &'a Matrix<T>,
    num_vectors: usize,
}

impl<T: VectorRepr> HasId for FlatVisitor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> DistancesUnordered<T::QueryDistance> for FlatVisitor<'_, T> {
    type ElementRef<'a> = &'a [T];
    type Id = u32;
    type Error = diskann::error::Infallible;

    fn distances_unordered<F>(
        &mut self,
        computer: &T::QueryDistance,
        mut f: F,
    ) -> impl SendFuture<Result<(), Self::Error>>
    where
        F: Send + FnMut(Self::Id, f32),
    {
        async move {
            for i in 0..self.num_vectors {
                let vector = self.data.row(i);
                let dist = computer.evaluate_similarity(vector);
                f(i as u32, dist);
            }
            Ok(())
        }
    }
}

impl<T: VectorRepr> SearchStrategy<InMemProvider<T>, &[T]> for FlatScanStrategy<T> {
    type ElementRef<'a> = &'a [T];
    type Id = u32;
    type QueryComputer = T::QueryDistance;
    type QueryComputerError = diskann::error::Infallible;
    type Visitor<'a>
        = FlatVisitor<'a, T>
    where
        Self: 'a,
        InMemProvider<T>: 'a;
    type Error = diskann::error::Infallible;

    fn create_visitor<'a>(
        &'a self,
        provider: &'a InMemProvider<T>,
        _context: &'a DefaultContext,
    ) -> Result<Self::Visitor<'a>, Self::Error> {
        Ok(FlatVisitor {
            data: &provider.data,
            num_vectors: self.num_vectors,
        })
    }

    fn build_query_computer(
        &self,
        query: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        Ok(T::query_distance(query, self.metric))
    }
}

//////////////////////////////////////////
// benchmark_core::search::Search impl  //
//////////////////////////////////////////

/// Wraps a [`FlatIndex`] and queries to implement the [`Search`] trait from benchmark_core.
struct FlatSearcher<T: VectorRepr> {
    index: Arc<FlatIndex<InMemProvider<T>>>,
    queries: Arc<Matrix<T>>,
    metric: Metric,
}

/// Search parameters for flat-index benchmarks.
#[derive(Debug, Clone, Copy)]
struct FlatSearchParameters {
    k: NonZeroUsize,
}

/// Additional metrics collected during flat search.
#[derive(Debug, Clone, Copy)]
struct FlatMetrics {
    /// The number of distance comparisons performed.
    pub comparisons: u32,
}

impl<T> search::Search for FlatSearcher<T>
where
    T: VectorRepr,
{
    type Id = u32;
    type Parameters = FlatSearchParameters;
    type Output = FlatMetrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k)
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
        let strategy = FlatScanStrategy::<T>::new(self.metric, self.index.provider().data.nrows());
        let context = DefaultContext;
        let query = self.queries.row(index);

        let stats = self
            .index
            .knn_search(parameters.k, &strategy, &context, query, buffer)
            .await?;

        Ok(FlatMetrics {
            comparisons: stats.cmps,
        })
    }
}

//////////////////
// Aggregation  //
//////////////////

/// Aggregates results from multiple flat search runs, computing recall metrics.
struct FlatAggregator<'a> {
    groundtruth: &'a Matrix<u32>,
    recall_k: usize,
}

impl<'a> FlatAggregator<'a> {
    fn new(groundtruth: &'a Matrix<u32>, recall_k: usize) -> Self {
        Self {
            groundtruth,
            recall_k,
        }
    }
}

/// Results of a single flat search run.
#[derive(Debug, Clone, Serialize)]
struct FlatSearchResults {
    num_tasks: usize,
    k: usize,
    qps: Vec<f64>,
    search_latencies: Vec<MicroSeconds>,
    mean_latencies: Vec<f64>,
    p90_latencies: Vec<MicroSeconds>,
    p99_latencies: Vec<MicroSeconds>,
    recall: RecallMetrics,
    mean_cmps: f32,
}

impl search::Aggregate<FlatSearchParameters, u32, FlatMetrics> for FlatAggregator<'_> {
    type Output = FlatSearchResults;

    fn aggregate(
        &mut self,
        run: search::Run<FlatSearchParameters>,
        mut results: Vec<search::SearchResults<u32, FlatMetrics>>,
    ) -> anyhow::Result<FlatSearchResults> {
        // Compute recall using the first repetition's results.
        let recall = match results.first() {
            Some(first) => benchmark_core::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                run.parameters().k.get(),
                GroundTruthMode::Fixed,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    mean_latencies.push(0.0);
                    p90_latencies.push(MicroSeconds::new(0));
                    p99_latencies.push(MicroSeconds::new(0));
                }
            }
        });

        let qps: Vec<f64> = results
            .iter()
            .map(|r| recall.num_queries as f64 / r.end_to_end_latency().as_seconds())
            .collect();

        let mean_cmps = {
            let (sum, count) = results
                .iter()
                .flat_map(|r| r.output().iter().map(|o| o.comparisons as f64))
                .fold((0.0f64, 0usize), |(s, c), v| (s + v, c + 1));
            if count == 0 {
                0.0
            } else {
                sum / count as f64
            }
        } as f32;

        Ok(FlatSearchResults {
            num_tasks: run.setup().tasks.into(),
            k: run.parameters().k.get(),
            qps,
            search_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps,
        })
    }
}

//////////////
// Results  //
//////////////

#[derive(Debug, Serialize)]
struct FlatResult {
    results: Vec<FlatSearchResults>,
}

impl std::fmt::Display for FlatResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.results.is_empty() {
            return Ok(());
        }

        let headers: &[&str] = &[
            "K",
            "Avg cmps",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ];

        let mut table =
            diskann_benchmark_runner::utils::fmt::Table::new(headers, self.results.len());
        for (i, r) in self.results.iter().enumerate() {
            let mut row = table.row(i);
            row.insert(r.k, 0);
            row.insert(r.mean_cmps, 1);
            row.insert(
                format!(
                    "{:.1} ({:.1})",
                    utils::MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                    utils::MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
                ),
                2,
            );
            row.insert(
                format!(
                    "{:.1}us ({:.1}us)",
                    utils::MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                    utils::MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
                ),
                3,
            );
            row.insert(
                format!(
                    "{:.1}us ({:.1})",
                    utils::MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                    utils::MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
                ),
                4,
            );
            row.insert(format!("{:3}", r.recall.average), 5);
            row.insert(r.num_tasks, 6);
        }

        write!(f, "{}", table)
    }
}
