/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A built-in helper for benchmarking K-nearest neighbors.

use std::{num::NonZeroUsize, sync::Arc};
use thiserror::Error;

use diskann::{
    ANNError, ANNResult,
    error::IntoANNResult,
    graph::{self, compat, glue},
};
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_utils::{
    future::AsyncFriendly,
    views::rowmajor::{self, Matrix},
};

use crate::{
    recall,
    recall::GroundTruthMode,
    search::{self, Search, graph::Strategy},
    utils,
};

/// A built-in helper for benchmarking the K-nearest neighbors method
/// [`graph::compat::DiskANNIndex::search`] with optional post-processing support.
///
/// This is intended to be used in conjunction with [`search::search`] or
/// [`search::search_all`] and provides some basic additional metrics for
/// the latter. Result aggregation for [`search::search_all`] is provided
/// by the [`Aggregator`] type.
///
/// The provided implementation of [`Search`] accepts [`KnnParams`]
/// and returns [`Metrics`] as additional output.
///
/// # Type Parameters
///
/// - `DP`: The data provider type
/// - `T`: The query element type
/// - `S`: The search strategy type
/// - `PP`: Post-processor selector. Defaults to [`Defaulted`], which uses the
///   accessor's default post-processor. Use [`KNN::with_postprocessor`] to
///   supply an explicit post-processor.
#[derive(Debug)]
pub struct KNN<DP, T, S, PP = Defaulted>
where
    DP: compat::DataProvider,
{
    index: Arc<graph::compat::DiskANNIndex<DP>>,
    queries: Arc<rowmajor::Owned<T>>,
    strategy: Strategy<S>,
    post_processor: PP,
}

impl<DP, T, S> KNN<DP, T, S, Defaulted>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`KNN`] searcher using the accessor's default post-processor.
    ///
    /// If `strategy` is one of the container variants of [`Strategy`], its length
    /// must match the number of rows in `queries`. If this is the case, then the
    /// strategies will have a querywise correspondence (see [`search::SearchResults`])
    /// with the query matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of rows in `queries`.
    pub fn new(
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
            post_processor: Defaulted,
        }))
    }
}

impl<DP, T, S, PP> KNN<DP, T, S, Forwarded<PP>>
where
    DP: compat::DataProvider,
{
    /// Construct a new [`KNN`] searcher with an explicit post-processor.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of elements in `strategy` is not compatible with
    /// the number of rows in `queries`.
    pub fn with_postprocessor(
        index: Arc<graph::compat::DiskANNIndex<DP>>,
        queries: Arc<rowmajor::Owned<T>>,
        strategy: Strategy<S>,
        post_processor: PP,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        Ok(Arc::new(Self {
            index,
            queries,
            strategy,
            post_processor: Forwarded(post_processor),
        }))
    }
}

impl<DP, T, S, PP> KNN<DP, T, S, PP>
where
    DP: compat::DataProvider,
{
    /// Access the index.
    pub fn index(&self) -> &Arc<graph::compat::DiskANNIndex<DP>> {
        &self.index
    }
}

/// Resolves a post-processor for [`KNN`] given a search accessor.
///
/// This trait lets [`KNN`] support both "use the accessor's default post-processor"
/// ([`Defaulted`]) and "use this explicit post-processor" ([`Forwarded`]) without
/// duplicating the search loop.
pub trait AsPostProcessor<'a, S, DP, T>
where
    DP: compat::DataProvider,
    S: compat::SearchStrategy<'a, DP, T>,
{
    /// The concrete post-processor used for a single search.
    type Processor: glue::SearchPostProcess<S::SearchAccessor, DP::ExternalId> + Send + Sync;

    /// Construct the post-processor to use for a single search.
    fn as_post_processor(&'a self, accessor: &S::SearchAccessor) -> Self::Processor;
}

/// Marker indicating that [`KNN`] should use the accessor's default post-processor.
#[derive(Debug, Clone, Copy)]
pub struct Defaulted;

impl<'a, S, DP, T> AsPostProcessor<'a, S, DP, T> for Defaulted
where
    DP: compat::DataProvider,
    S: compat::SearchStrategy<'a, DP, T>,
    S::SearchAccessor: glue::DefaultPostProcessor<DP::ExternalId>,
{
    type Processor = <S::SearchAccessor as glue::DefaultPostProcessor<DP::ExternalId>>::Processor;

    fn as_post_processor(&'a self, accessor: &S::SearchAccessor) -> Self::Processor {
        glue::DefaultPostProcessor::default_post_processor(accessor)
    }
}

/// Wraps an explicit post-processor for use with [`KNN::with_postprocessor`].
#[derive(Debug, Clone, Copy)]
pub struct Forwarded<PP>(PP);

impl<'a, S, DP, T, PP> AsPostProcessor<'a, S, DP, T> for Forwarded<PP>
where
    DP: compat::DataProvider,
    S: compat::SearchStrategy<'a, DP, T>,
    PP: glue::SearchPostProcess<S::SearchAccessor, DP::ExternalId> + Clone + AsyncFriendly,
{
    type Processor = PP;

    fn as_post_processor(&'a self, _accessor: &S::SearchAccessor) -> Self::Processor {
        self.0.clone()
    }
}

/// Additional metrics collected during [`KNN`] search.
///
/// # Note
///
/// This struct is marked as non-exhaustive to allow for future additions.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Metrics {
    /// The number of distance comparisons performed during search.
    pub comparisons: u32,
    /// The number of candidates expanded during search.
    pub hops: u32,
}

impl Metrics {
    pub fn new(comparisons: u32, hops: u32) -> Self {
        Self { comparisons, hops }
    }
}

impl<DP, T, S, PP> Search for KNN<DP, T, S, PP>
where
    DP: compat::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> compat::SearchStrategy<'a, DP, &'a [T], SearchAccessor: glue::SearchAccessor>
        + Clone
        + AsyncFriendly,
    PP: for<'a> AsPostProcessor<'a, S, DP, &'a [T]> + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = KnnParams;
    type Output = Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let knn_search = parameters.knn;
        let strategy = self.strategy.get(index)?;
        let mut accessor = strategy
            .search_accessor(self.index.provider(), &context, self.queries.row(index))
            .into_ann_result()?;
        let processor = self.post_processor.as_post_processor(&accessor);
        let stats = graph::Search::search(knn_search, &mut accessor, processor, buffer).await?;

        Ok(Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
        })
    }
}

#[derive(Debug, Error)]
pub enum KnnParamsError {
    #[error("k_value cannot be zero")]
    KZero,
    #[error("l_value ({l_value}) must be at least k_value ({k_value})")]
    LLessThanK { l_value: usize, k_value: usize },
    #[error("invalid KNN parameters")]
    InvalidKnnParameters,
}

impl From<KnnParamsError> for ANNError {
    #[track_caller]
    fn from(err: KnnParamsError) -> Self {
        ANNError::new(err)
    }
}

/// A wrapper for the [`graph::search::Knn`] struct that also includes the `k` value.
#[derive(Debug, Copy, Clone)]
pub struct KnnParams {
    k_value: NonZeroUsize,
    pub knn: graph::search::Knn,
}

impl KnnParams {
    /// Construct a new [`KnnParams`].
    pub fn new(k_value: usize, l_value: usize) -> Result<Self, KnnParamsError> {
        let k_value = NonZeroUsize::new(k_value).ok_or(KnnParamsError::KZero)?;
        if l_value < k_value.get() {
            return Err(KnnParamsError::LLessThanK {
                l_value,
                k_value: k_value.get(),
            });
        }

        let knn = graph::search::Knn::new(l_value, None)
            .map_err(|_| KnnParamsError::InvalidKnnParameters)?;
        Ok(Self { k_value, knn })
    }

    pub fn k_value(&self) -> NonZeroUsize {
        self.k_value
    }
}

/// An [`search::Aggregate`]d summary of multiple [`KNN`] search runs
/// returned by the provided [`Aggregator`].
///
/// This struct is marked as non-exhaustive to allow for future additions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Summary {
    /// The [`search::Setup`] used for the batch of runs.
    pub setup: search::Setup,

    /// The [`Search::Parameters`] used for the batch of runs.
    pub parameters: KnnParams,

    /// The end-to-end latency for each repetition in the batch.
    pub end_to_end_latencies: Vec<MicroSeconds>,

    /// The average latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub mean_latencies: Vec<f64>,

    /// The 90th percentile latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub p90_latencies: Vec<MicroSeconds>,

    /// The 99th percentile latency for individual queries.
    ///
    /// This contains one entry per repetition in the batch.
    pub p99_latencies: Vec<MicroSeconds>,

    /// The recall metrics for search.
    ///
    /// This implementation assumes that search is deterministic and only
    /// uses the first repetition's results to compute recall.
    pub recall: recall::RecallMetrics,

    /// The average number of distance comparisons per query.
    pub mean_cmps: f64,

    /// The average number of neighbor hops per query.
    pub mean_hops: f64,
}

/// A [`search::Aggregate`] for collecting the results of multiple [`KNN`] search runs.
///
/// In addition to collecting latencies and other metrics, this aggregator computes
/// recall using a provided groundtruth.
///
/// The aggregated results are available as a [`Summary`].
pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
    groundtruth_mode: GroundTruthMode,
}

impl<'a, I> Aggregator<'a, I> {
    /// Construct a new [`Aggregator`] using `groundtruth` for recall computation.
    ///
    /// Recall will be computed as `recall_k`-NN recall over the top `recall_n` neighbors.
    ///
    /// This implementation allows fewer than `recall_n` neighbors to be returned
    /// per query without error.
    pub fn new(
        groundtruth: &'a dyn crate::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
        groundtruth_mode: GroundTruthMode,
    ) -> Self {
        Self {
            groundtruth,
            recall_k,
            recall_n,
            groundtruth_mode,
        }
    }
}

impl<I> search::Aggregate<KnnParams, I, Metrics> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
{
    type Output = Summary;

    fn aggregate(
        &mut self,
        run: search::Run<KnnParams>,
        mut results: Vec<search::SearchResults<I, Metrics>>,
    ) -> anyhow::Result<Summary> {
        // Compute the recall using just the first result.
        let recall = match results.first() {
            Some(first) => crate::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                self.recall_n,
                self.groundtruth_mode,
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
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
        });

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            recall,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_cmps: utils::average_all(
                results
                    .iter()
                    .flat_map(|r| r.output().iter().map(|o| o.comparisons)),
            ),
            mean_hops: utils::average_all(
                results
                    .iter()
                    .flat_map(|r| r.output().iter().map(|o| o.hops)),
            ),
        })
    }
}
