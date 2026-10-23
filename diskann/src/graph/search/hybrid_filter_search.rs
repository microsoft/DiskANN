/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Route selective label filters to exhaustive search and broader filters to multihop.

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    marker::PhantomData,
    num::{NonZeroU64, NonZeroUsize},
    time::{Duration, Instant},
};

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Knn, MultihopFilterSearch, Search};
use crate::{
    ANNError, ANNResult, convert_error,
    error::IntoANNResult,
    graph::{
        ext::labeled::CandidateLabelProvider,
        glue::{FilteredAccessor, RandomAccessQueryDistance, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
    utils::VectorId,
};

/// Invalid parameters for [`HybridFilterSearch`].
#[derive(Debug, Error)]
pub enum HybridFilterSearchError {
    /// A top-k search must request at least one result.
    #[error("hybrid search k cannot be zero")]
    KZero,
    /// Exhaustive search cannot return more than the graph search list size.
    #[error("hybrid search k {k} exceeds search L {l}")]
    KExceedsL { k: usize, l: usize },
    /// The routing threshold must be positive.
    #[error("hybrid match-count threshold cannot be zero")]
    ThresholdZero,
}

convert_error!(HybridFilterSearchError);

#[derive(Debug, Clone, Copy)]
struct Ranked<I> {
    id: I,
    distance: f32,
}

impl<I: Ord> PartialEq for Ranked<I> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl<I: Ord> Eq for Ranked<I> {}

impl<I: Ord> PartialOrd for Ranked<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Ord> Ord for Ranked<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn keep_nearest<I: Ord>(nearest: &mut BinaryHeap<Ranked<I>>, k: usize, candidate: Ranked<I>) {
    if nearest.len() < k {
        nearest.push(candidate);
    } else if nearest.peek().is_some_and(|worst| candidate < *worst) {
        nearest.pop();
        nearest.push(candidate);
    }
}

/// Per-query timings for the exhaustive branch of hybrid filtered search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HybridPhaseTimings {
    /// Materialize candidate IDs from the filter, including allocation.
    pub candidate_scan: Duration,
    /// Prepare the query accessor and evaluate every candidate distance.
    pub distance_evaluation: Duration,
    /// Select and sort the best k scored candidates.
    pub topk_selection: Duration,
    /// Run the configured search post-processor.
    pub post_processing: Duration,
    /// Number of Bloom-positive candidate IDs scored.
    pub candidate_count: u32,
}

/// Hybrid search statistics, with phase timings for exhaustive queries only.
#[derive(Debug, Clone, Copy)]
pub struct HybridSearchStats {
    /// Standard graph search statistics.
    pub stats: SearchStats,
    /// `None` for multihop queries; zero durations for a filter with no matches.
    pub timings: Option<HybridPhaseTimings>,
}

/// A filtered search that uses a conservative match bound to choose its search path.
///
/// The caller must use the same label provider in the filtered search strategy and in
/// this parameter object. For a positive bound below `threshold`, every enumerated
/// candidate is scored against the query; Bloom-backed providers can return false
/// positives. Larger bounds retain the usual multihop graph traversal.
#[derive(Debug)]
pub struct HybridFilterSearch<'f, I, F: ?Sized>
where
    I: VectorId,
    F: CandidateLabelProvider<I>,
{
    inner: Knn,
    k: NonZeroUsize,
    threshold: NonZeroU64,
    filter: &'f F,
    _id: PhantomData<fn(I)>,
}

impl<'f, I, F> HybridFilterSearch<'f, I, F>
where
    I: VectorId,
    F: CandidateLabelProvider<I> + ?Sized,
{
    /// Construct a hybrid search from graph parameters and a label provider.
    ///
    /// # Errors
    ///
    /// Fails when `k` or `threshold` is zero, or `k` exceeds graph search L.
    pub fn new(
        inner: Knn,
        k: usize,
        threshold: u64,
        filter: &'f F,
    ) -> Result<Self, HybridFilterSearchError> {
        let k = NonZeroUsize::new(k).ok_or(HybridFilterSearchError::KZero)?;
        if k > inner.l_value() {
            return Err(HybridFilterSearchError::KExceedsL {
                k: k.get(),
                l: inner.l_value().get(),
            });
        }
        let threshold = NonZeroU64::new(threshold).ok_or(HybridFilterSearchError::ThresholdZero)?;
        Ok(Self {
            inner,
            k,
            threshold,
            filter,
            _id: PhantomData,
        })
    }
}

impl<'a, DP, S, T, F> Search<'a, DP, S, T> for HybridFilterSearch<'_, DP::InternalId, F>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor + RandomAccessQueryDistance>,
    T: Copy + Send + Sync,
    F: CandidateLabelProvider<DP::InternalId> + ?Sized,
{
    type Output = HybridSearchStats;

    fn search<O, PP, OB>(
        self,
        index: &'a DiskANNIndex<DP>,
        strategy: &'a S,
        processor: PP,
        context: &'a DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let bound = self.filter.match_upper_bound().ok_or_else(|| {
                ANNError::message("hybrid search requires exact per-label match counts")
            })?;
            if bound == 0 {
                return Ok(HybridSearchStats {
                    stats: SearchStats {
                        cmps: 0,
                        hops: 0,
                        result_count: 0,
                        range_search_second_round: false,
                    },
                    timings: Some(HybridPhaseTimings::default()),
                });
            }
            if bound >= self.threshold.get() {
                let stats = MultihopFilterSearch::new(self.inner)
                    .search(index, strategy, processor, context, query, output)
                    .await?;
                return Ok(HybridSearchStats {
                    stats,
                    timings: None,
                });
            }

            let scan_start = Instant::now();
            let capacity = usize::try_from(bound)
                .map_err(|_| ANNError::message("hybrid match bound exceeds usize"))?;
            let mut candidates = Vec::new();
            candidates
                .try_reserve_exact(capacity)
                .map_err(|_| ANNError::message("cannot reserve hybrid candidate IDs"))?;
            let mut failure = None;
            self.filter.visit_candidates(|id| {
                if failure.is_some() {
                    return;
                }
                if candidates.len() == candidates.capacity() && candidates.try_reserve(1).is_err() {
                    failure = Some(ANNError::message("cannot grow hybrid candidate IDs"));
                    return;
                }
                candidates.push(id);
            });
            if let Some(error) = failure {
                return Err(error);
            }
            let candidate_scan = scan_start.elapsed();
            let comparisons = u32::try_from(candidates.len()).map_err(|_| {
                ANNError::message("hybrid candidate count exceeds search statistics capacity")
            })?;

            let distance_start = Instant::now();
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;
            let mut distances = Vec::new();
            distances
                .try_reserve_exact(candidates.len())
                .map_err(|_| ANNError::message("cannot reserve hybrid candidate distances"))?;
            for &id in &candidates {
                distances.push(accessor.distance_to_id(id)?);
            }
            let distance_evaluation = distance_start.elapsed();

            let topk_start = Instant::now();
            let mut nearest = BinaryHeap::new();
            nearest
                .try_reserve(self.k.get())
                .map_err(|_| ANNError::message("cannot reserve hybrid top-k search results"))?;
            for (&id, &distance) in candidates.iter().zip(&distances) {
                keep_nearest(&mut nearest, self.k.get(), Ranked { id, distance });
            }
            let ranked = nearest.into_sorted_vec();
            let topk_selection = topk_start.elapsed();

            let post_start = Instant::now();
            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    ranked
                        .into_iter()
                        .map(|candidate| Neighbor::new(candidate.id, candidate.distance)),
                    output,
                )
                .await
                .into_ann_result()?;
            let result_count = u32::try_from(result_count)
                .map_err(|_| ANNError::message("hybrid search result count exceeds u32"))?;
            Ok(HybridSearchStats {
                stats: SearchStats {
                    cmps: comparisons,
                    hops: 0,
                    result_count,
                    range_search_second_round: false,
                },
                timings: Some(HybridPhaseTimings {
                    candidate_scan,
                    distance_evaluation,
                    topk_selection,
                    post_processing: post_start.elapsed(),
                    candidate_count: comparisons,
                }),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{config, ext::labeled, search_output_buffer::IdDistance, test};

    #[derive(Debug)]
    struct StubFilter {
        bound: Option<u64>,
        candidates: Vec<u32>,
        true_matches: Vec<u32>,
    }

    impl crate::graph::ext::labeled::QueryLabelProvider<u32> for StubFilter {
        fn is_match(&self, id: u32) -> bool {
            self.true_matches.contains(&id)
        }
    }

    impl CandidateLabelProvider<u32> for StubFilter {
        fn match_upper_bound(&self) -> Option<u64> {
            self.bound
        }

        fn visit_candidates(&self, mut visit: impl FnMut(u32)) {
            for &id in &self.candidates {
                visit(id);
            }
        }
    }

    #[test]
    fn validates_hybrid_parameters_and_distance_id_ties() {
        let knn = Knn::new_default(10).unwrap();
        let filter = StubFilter {
            bound: Some(1),
            candidates: vec![0],
            true_matches: vec![0],
        };
        assert!(HybridFilterSearch::new(knn, 0, 200, &filter).is_err());
        assert!(HybridFilterSearch::new(knn, 11, 200, &filter).is_err());
        assert!(HybridFilterSearch::new(knn, 2, 0, &filter).is_err());
        assert!(HybridFilterSearch::new(knn, 2, 200, &filter).is_ok());

        let mut heap = BinaryHeap::new();
        for (id, distance) in [(6, 8.0), (2, 3.0), (4, 3.0), (1, 1.0), (3, 3.0)] {
            keep_nearest(&mut heap, 3, Ranked { id, distance });
        }
        assert_eq!(
            heap.into_sorted_vec()
                .into_iter()
                .map(|entry| entry.id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[tokio::test]
    async fn hybrid_routes_exhaustive_multihop_empty_and_missing_counts() -> ANNResult<()> {
        let provider = test::provider::Provider::grid(test::synthetic::Grid::One, 4)?;
        let config = config::Builder::new(
            provider.max_degree(),
            config::MaxDegree::same(),
            10,
            provider.distance_metric().into(),
        )
        .build()?;
        let index = DiskANNIndex::new(config, provider, None);
        let context = test::provider::Context::default();
        let query = [0.0f32];
        let knn = Knn::new_default(10)?;

        let sparse = StubFilter {
            bound: Some(1),
            candidates: vec![1, 2],
            true_matches: vec![1],
        };
        let strategy = labeled::Filtered::new(test::provider::Strategy::new(), &sparse);
        let params = HybridFilterSearch::new(knn, 2, 3, &sparse)?;
        let mut ids = [u32::MAX; 2];
        let mut distances = [f32::INFINITY; 2];
        let mut output = IdDistance::new(&mut ids, &mut distances);
        let stats = index
            .search(params, &strategy, &context, &query[..], &mut output)
            .await?;
        assert_eq!(ids, [1, 2]);
        assert_eq!(distances, [1.0, 4.0]);
        assert_eq!(stats.stats.cmps, 2);
        assert_eq!(stats.stats.hops, 0);
        assert_eq!(stats.timings.map(|times| times.candidate_count), Some(2));

        let broad = StubFilter {
            bound: Some(3),
            candidates: vec![0, 1, 2, 3],
            true_matches: vec![0, 1, 2],
        };
        let strategy = labeled::Filtered::new(test::provider::Strategy::new(), &broad);
        let params = HybridFilterSearch::new(knn, 2, 3, &broad)?;
        let mut ids = [u32::MAX; 2];
        let mut distances = [f32::INFINITY; 2];
        let mut output = IdDistance::new(&mut ids, &mut distances);
        let stats = index
            .search(params, &strategy, &context, &query[..], &mut output)
            .await?;
        assert_eq!(ids, [0, 1]);
        assert!(stats.stats.hops > 0);
        assert!(stats.timings.is_none());

        let empty = StubFilter {
            bound: Some(0),
            candidates: vec![],
            true_matches: vec![],
        };
        let strategy = labeled::Filtered::new(test::provider::Strategy::new(), &empty);
        let params = HybridFilterSearch::new(knn, 2, 3, &empty)?;
        let mut ids = [u32::MAX; 2];
        let mut distances = [f32::INFINITY; 2];
        let mut output = IdDistance::new(&mut ids, &mut distances);
        let stats = index
            .search(params, &strategy, &context, &query[..], &mut output)
            .await?;
        assert_eq!(output.current_len(), 0);
        assert_eq!(stats.stats.cmps, 0);
        assert_eq!(stats.stats.hops, 0);
        assert_eq!(stats.timings, Some(HybridPhaseTimings::default()));

        let missing_counts = StubFilter {
            bound: None,
            candidates: vec![0],
            true_matches: vec![0],
        };
        let strategy = labeled::Filtered::new(test::provider::Strategy::new(), &missing_counts);
        let params = HybridFilterSearch::new(knn, 2, 3, &missing_counts)?;
        let mut ids = [u32::MAX; 2];
        let mut distances = [f32::INFINITY; 2];
        let mut output = IdDistance::new(&mut ids, &mut distances);
        assert!(
            index
                .search(params, &strategy, &context, &query[..], &mut output)
                .await
                .is_err()
        );
        Ok(())
    }
}
