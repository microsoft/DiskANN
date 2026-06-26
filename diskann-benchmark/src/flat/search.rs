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
    graph::{glue::CopyIds, SearchOutputBuffer},
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
use diskann_providers::model::graph::provider::async_::FastMemoryVectorProviderAsync;
use diskann_providers::storage::FileStorageProvider;
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
/// Wraps a [`FastMemoryVectorProviderAsync<T>`] and implements [`DataProvider`]
/// with identity ID mapping.
struct InMemProvider<T: VectorRepr> {
    data: FastMemoryVectorProviderAsync<T>,
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
        let provider = {
            let fmvp = FastMemoryVectorProviderAsync::<T>::load_from_bin(
                &FileStorageProvider,
                &input.data.to_string_lossy(),
                metric,
                None,
                None,
            )?;
            InMemProvider { data: fmvp }
        };
        let nrows = provider.data.total();
        let ncols = provider.data.dim();
        anyhow::ensure!(
            nrows <= u32::MAX as usize,
            "flat-index benchmark requires <= {} vectors (got {}) to fit in u32 ids",
            u32::MAX,
            nrows,
        );
        writeln!(output, "  Loaded {} vectors of dimension {}", nrows, ncols)?;

        // Build the FlatIndex
        let index = FlatIndex::new(provider);

        // Load queries and groundtruth
        let queries: Matrix<T> =
            datafiles::load_dataset(datafiles::BinFile(&input.search.queries))?;
        let groundtruth = datafiles::load_groundtruth(
            datafiles::BinFile(&input.search.groundtruth),
            Some(input.search.k.get()),
        )?;
        anyhow::ensure!(
            ncols == queries.ncols(),
            "dataset dimension ({}) does not match query dimension ({})",
            ncols,
            queries.ncols(),
        );

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
        anyhow::ensure!(
            k.get() <= nrows,
            "k ({}) must be <= number of dataset vectors ({})",
            k,
            nrows,
        );

        let mut results = Vec::new();

        let searcher = Arc::new(FlatSearcher {
            index,
            queries,
            strategy: FlatScanStrategy::new(metric),
        });

        for &threads in &input.search.num_threads {
            let setup = search::Setup {
                threads,
                tasks: threads,
                reps,
            };

            let run = search::Run::new(FlatSearchParameters { k }, setup);
            let aggregated = search::search_all(
                searcher.clone(),
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
    _phantom: std::marker::PhantomData<T>,
}

impl<T: VectorRepr> FlatScanStrategy<T> {
    fn new(metric: Metric) -> Self {
        Self {
            metric,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// The visitor that iterates over all vectors in the provider.
struct FlatVisitor<'a, T: VectorRepr> {
    data: &'a FastMemoryVectorProviderAsync<T>,
}

impl<T: VectorRepr> HasId for FlatVisitor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> DistancesUnordered<T::QueryDistance> for FlatVisitor<'_, T> {
    type ElementRef<'a> = &'a [T];
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
            let total = self.data.total();
            for i in 0..total {
                // SAFETY: single-writer load completed before search; no concurrent mutation.
                let vector = unsafe { self.data.get_vector_sync(i) };
                let dist = computer.evaluate_similarity(vector);
                f(i as u32, dist);
            }
            Ok(())
        }
    }
}

impl<T: VectorRepr> SearchStrategy<InMemProvider<T>, &[T]> for FlatScanStrategy<T> {
    type ElementRef<'a> = &'a [T];
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
    index: FlatIndex<InMemProvider<T>>,
    queries: Matrix<T>,
    strategy: FlatScanStrategy<T>,
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
        let context = DefaultContext;
        let query = self.queries.row(index);

        let stats = self
            .index
            .knn_search(
                parameters.k,
                &self.strategy,
                CopyIds,
                &context,
                query,
                buffer,
            )
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

        for r in results.iter_mut() {
            let percentiles::Percentiles { mean, p90, p99, .. } =
                percentiles::compute_percentiles(r.latencies_mut())?;
            mean_latencies.push(mean);
            p90_latencies.push(p90);
            p99_latencies.push(p99);
        }

        let qps: Vec<f64> = results
            .iter()
            .map(|r| recall.num_queries as f64 / r.end_to_end_latency().as_seconds())
            .collect();

        let mean_cmps = benchmark_core::utils::average_all(
            results
                .iter()
                .flat_map(|r| r.output().iter().map(|o| o.comparisons)),
        ) as f32;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::Example;
    use diskann_benchmark_runner::utils::MicroSeconds;

    fn make_dummy_results(num_results: usize) -> FlatResult {
        let results = (0..num_results)
            .map(|i| FlatSearchResults {
                num_tasks: i + 1,
                k: 10,
                qps: vec![100.0],
                search_latencies: vec![MicroSeconds::new(1000)],
                mean_latencies: vec![10.0],
                p90_latencies: vec![MicroSeconds::new(900)],
                p99_latencies: vec![MicroSeconds::new(990)],
                recall: RecallMetrics {
                    recall_k: 10,
                    recall_n: 10,
                    num_queries: 100,
                    average: 0.95,
                },
                mean_cmps: 256.0,
            })
            .collect();
        FlatResult { results }
    }

    #[test]
    fn display_empty_flat_result() {
        let result = FlatResult {
            results: Vec::new(),
        };
        let text = format!("{}", result);
        assert!(text.is_empty());
    }

    #[test]
    fn display_flat_result_with_data() {
        let result = make_dummy_results(1);
        let text = format!("{}", result);
        assert!(text.contains("K"));
        assert!(text.contains("Recall"));
    }

    #[test]
    fn description_with_matching_type() {
        let benchmark = FlatBenchmark::<f32>::new();
        let input = crate::inputs::flat::FlatSearch::example();
        let text = format!("{}", DescriptionHelper(&benchmark, Some(&input)));
        // When the type matches, description writes nothing (is_match() == true)
        assert!(!text.contains("Data Type:"));
    }

    #[test]
    fn description_without_input() {
        let benchmark = FlatBenchmark::<f32>::new();
        let text = format!("{}", DescriptionHelper::<f32>(&benchmark, None));
        assert!(text.contains("Data Type: float32"));
    }

    #[test]
    fn description_with_mismatched_type() {
        use diskann_benchmark_runner::utils::datatype::DataType;
        let benchmark = FlatBenchmark::<f32>::new();
        let mut input = crate::inputs::flat::FlatSearch::example();
        input.data_type = DataType::UInt8;
        let text = format!("{}", DescriptionHelper(&benchmark, Some(&input)));
        assert!(text.contains("Data Type: expected \"float32\" but found \"uint8\""));
    }

    /// Helper to call `description()` via `Display`.
    struct DescriptionHelper<'a, T: VectorRepr + AsDataType>(
        &'a FlatBenchmark<T>,
        Option<&'a crate::inputs::flat::FlatSearch>,
    );

    impl<T: VectorRepr + AsDataType> std::fmt::Display for DescriptionHelper<'_, T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.description(f, self.1)
        }
    }
}
