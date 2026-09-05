/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Reusable execution harness for [`crate::flat::knn_search`] tests.
//!
//! Use [`KnnOracleRun::run`] to drive `knn_search` under a chosen [`OracleProcessor`]
//! and pair the result with the oracle's expected post-processed output.

use std::{convert::Infallible, num::NonZeroUsize};

use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};
use diskann_utils::views::rowmajor::Matrix;

use crate::{
    ANNResult,
    error::IntoANNResult,
    flat::{
        SearchStats, knn_search,
        test::provider::{Provider, Visitor},
    },
    graph::{
        SearchOutputBuffer,
        glue::{CopyIds, SearchPostProcess},
    },
    neighbor::{self, BackInserter, Neighbor},
    provider::HasId,
    test::tokio::current_thread_runtime,
    utils::VectorRepr,
};

/// Result of running [`knn_search`] under the harness alongside the
/// oracle's expected post-processed output.
#[derive(Debug, Clone)]
pub(crate) struct KnnOracleRun {
    /// Post-processed `(id, distance)` pairs returned by the search.
    /// Re-sorted from the output buffer so equality checks are deterministic on ties.
    pub top_k: Vec<(u32, f32)>,
    /// `top_k.iter().map(|(_, d)| d).collect()`.
    pub top_k_distances: Vec<f32>,
    /// Statistics returned by `knn_search` (cmps, result_count).
    pub stats: SearchStats,
    /// Oracle-derived expected output: the brute-force top-`k` after applying the
    /// [`OracleProcessor`]'s [`expected`](OracleProcessor::expected) transform, in
    /// `(distance asc, id asc)` order.
    pub ground_truth: Vec<(u32, f32)>,
}

impl KnnOracleRun {
    /// Run [`knn_search`] once under `oracle`, blocking on a fresh
    /// single-threaded runtime, and pair the result with the oracle's expected output.
    pub fn run_sync<O: OracleProcessor>(
        provider: &Provider,
        oracle: &O,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        current_thread_runtime().block_on(Self::run(provider, oracle, query, k))
    }

    /// Run [`knn_search`] with an already initialized visitor.
    pub fn run_sync_with_visitor<O: OracleProcessor>(
        provider: &Provider,
        mut visitor: Visitor<'_>,
        oracle: &O,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        current_thread_runtime().block_on(Self::run_with_visitor(
            provider,
            &mut visitor,
            oracle,
            query,
            k,
        ))
    }

    /// Async variant of [`KnnOracleRun::run_sync`]. Use this from tests that already
    /// have a Tokio runtime (e.g. `#[tokio::test]`) or that need to drive
    /// `knn_search` concurrently across tasks.
    pub async fn run<O: OracleProcessor>(
        provider: &Provider,
        oracle: &O,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        let mut visitor = Visitor::new(provider, query).into_ann_result()?;
        Self::run_with_visitor(provider, &mut visitor, oracle, query, k).await
    }

    async fn run_with_visitor<O: OracleProcessor>(
        provider: &Provider,
        visitor: &mut Visitor<'_>,
        oracle: &O,
        query: &[f32],
        k: usize,
    ) -> ANNResult<Self> {
        let mut buf = vec![Neighbor::<u32>::default(); k];

        let stats = knn_search(
            visitor,
            NonZeroUsize::new(k).expect("flat::test::harness requires k > 0"),
            oracle.processor(),
            query,
            &mut BackInserter::new(buf.as_mut_slice()),
        )
        .await?;

        let mut top_k: Vec<Neighbor<u32>> = buf
            .iter()
            .copied()
            .take(stats.result_count as usize)
            .collect();
        sort_neighbors(&mut top_k);
        let top_k_distances = top_k.iter().map(|n| *n.distance()).collect();

        let ground_truth = oracle.expected(brute_force_topk(provider, Metric::L2, query, k));

        Ok(Self {
            top_k: top_k.into_iter().map(Neighbor::as_tuple).collect(),
            top_k_distances,
            stats,
            ground_truth: ground_truth.into_iter().map(Neighbor::as_tuple).collect(),
        })
    }
}

///////////////////
// Post-process //
//////////////////

/// Wrapper for producing a [`SearchPostProcess`] and its reference impl
pub(crate) trait OracleProcessor {
    /// The post-processor exercised by the search.
    type Processor: for<'a, 'q> SearchPostProcess<Visitor<'a>, &'q [f32], u32> + Send + Sync;

    /// Construct the processor instance fed to [`knn_search`].
    fn processor(&self) -> Self::Processor;

    /// Transform the brute-force top-`k` neighbors into the output the processor is
    /// expected to produce.
    fn expected(&self, gt: Vec<Neighbor<u32>>) -> Vec<Neighbor<u32>>;
}

#[derive(Clone, Copy)]
pub(crate) struct CopyIdsOracle;

impl OracleProcessor for CopyIdsOracle {
    type Processor = CopyIds;

    fn processor(&self) -> Self::Processor {
        CopyIds
    }

    fn expected(&self, gt: Vec<Neighbor<u32>>) -> Vec<Neighbor<u32>> {
        gt
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct EvenIdsOnly;

impl<A, T> SearchPostProcess<A, T> for EvenIdsOnly
where
    A: HasId<Id = u32>,
{
    type Error = Infallible;

    fn post_process<I, B>(
        &self,
        _accessor: &mut A,
        _query: T,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = Neighbor<u32>> + Send,
        B: SearchOutputBuffer<u32> + Send + ?Sized,
    {
        let count = output.extend(candidates.filter(|n| *n.id() % 2 == 0));
        std::future::ready(Ok(count))
    }
}

/// Oracle for [`EvenIdsOnly`]: the brute-force top-`k` with odd ids dropped.
#[derive(Clone, Copy)]
pub(crate) struct EvenIdsOnlyOracle;

impl OracleProcessor for EvenIdsOnlyOracle {
    type Processor = EvenIdsOnly;

    fn processor(&self) -> Self::Processor {
        EvenIdsOnly
    }

    fn expected(&self, gt: Vec<Neighbor<u32>>) -> Vec<Neighbor<u32>> {
        gt.into_iter().filter(|n| *n.id() % 2 == 0).collect()
    }
}

/// Compute the brute-force top-`k` neighbors over every element of `provider` under
/// `metric`.
pub(crate) fn brute_force_topk(
    provider: &Provider,
    metric: Metric,
    query: &[f32],
    k: usize,
) -> Vec<Neighbor<u32>> {
    let computer = f32::query_distance(query, metric);

    let mut neighbors: Vec<Neighbor<u32>> = provider
        .items()
        .row_iter()
        .enumerate()
        .map(|(id, element)| Neighbor::new(id as u32, computer.evaluate_similarity(element)))
        .collect();

    sort_neighbors(&mut neighbors);
    neighbors.truncate(k);
    neighbors
}

/// Sort a slice of [`Neighbor<u32>`] by `(distance asc, id asc)`.
fn sort_neighbors(neighbors: &mut [Neighbor<u32>]) {
    neighbors.sort_by(neighbor::ord::fast_distance_total);
}
