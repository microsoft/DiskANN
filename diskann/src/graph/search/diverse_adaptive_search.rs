/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Attribute-diversity search with adaptive-L over-fetch (Design B).
//!
//! This is a plain greedy graph search whose only addition over [`Knn`] is a
//! traversal-time sampler that estimates *bucket concentration* — how many
//! distinct attribute buckets appear among the nodes visited so far — and grows
//! the search list `L` accordingly. The actual bucket-diverse top-`k` selection
//! is still performed by a downstream post-processor over the enlarged pool.
//!
//! # Motivation
//!
//! Post-processing-only diversity (Design A) relies on the caller picking an `L`
//! large enough that the top-`L` pool contains enough distinct buckets to fill a
//! diverse top-`k`. When a query lands in a region dominated by a few buckets,
//! a fixed `L` under-fetches and recall suffers. Design B sizes `L` per query
//! from the observed bucket concentration, reusing the same piecewise scaling as
//! inline-filter [`AdaptiveL`].
//!
//! # Yield metric
//!
//! Analogous to filter specificity, the diversity *yield* of a sample is
//!
//! ```text
//! yield = (Σ_b min(count_b, k)) / visited
//! ```
//!
//! where `count_b` is the number of sampled nodes in bucket `b` and `k` is
//! `diverse_results_k`. For `k = 1` this is simply `distinct_buckets / visited`.
//! A low yield (many duplicate buckets) triggers a larger `L`; a high yield
//! (mostly distinct buckets) leaves `L` unchanged.

use std::sync::Arc;

use diskann_utils::future::SendFuture;
use hashbrown::HashMap;

use super::{
    AdaptiveL, Knn, Search, inline_filter_search::compute_adaptive_l, scratch::SearchScratch,
};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, SearchAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::{AttributeValueProvider, Neighbor},
    provider::DataProvider,
    utils::VectorId,
};

/// Attribute-diversity search with adaptive-L over-fetch.
///
/// Runs a plain greedy beam search, sampling attribute buckets during traversal
/// to grow `L` when bucket concentration is low. The final bucket-diverse
/// selection is delegated to the post-processor supplied at search time.
pub struct DiverseAdaptiveSearch<P>
where
    P: AttributeValueProvider + ?Sized,
{
    inner: Knn,
    provider: Arc<P>,
    diverse_results_k: usize,
    adaptive_l: AdaptiveL,
}

impl<P> DiverseAdaptiveSearch<P>
where
    P: AttributeValueProvider + ?Sized,
{
    /// Create new adaptive diverse search parameters.
    pub fn new(
        inner: Knn,
        provider: Arc<P>,
        diverse_results_k: usize,
        adaptive_l: AdaptiveL,
    ) -> Self {
        Self {
            inner,
            provider,
            diverse_results_k,
            adaptive_l,
        }
    }
}

impl<'a, DP, S, T, P> Search<'a, DP, S, T> for DiverseAdaptiveSearch<P>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: SearchAccessor>,
    T: Copy + Send + Sync,
    P: AttributeValueProvider<Id = DP::InternalId> + ?Sized,
{
    type Output = SearchStats;

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
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;

            let num_starting_points = accessor.num_starting_points().await?;
            let mut scratch = index.search_scratch(self.inner.l_value().get(), num_starting_points);

            let stats = diverse_adaptive_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                self.provider.as_ref(),
                self.diverse_results_k,
                &self.adaptive_l,
                &mut accessor,
                &mut scratch,
            )
            .await?;

            let result_count = processor
                .post_process(&mut accessor, query, scratch.best.iter(), output)
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

async fn diverse_adaptive_search_internal<I, A, P>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    provider: &P,
    diverse_results_k: usize,
    adaptive_l: &AdaptiveL,
    accessor: &mut A,
    scratch: &mut SearchScratch<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: SearchAccessor<Id = I>,
    P: AttributeValueProvider<Id = I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();
    let l_search = search_params.l_value().get();

    // Bucket-concentration sampling state. Sampling stops once L is (re)sized.
    let mut bucket_counts: HashMap<P::Value, usize> = HashMap::new();
    let mut sample_visited: usize = 0;
    let mut l_adjusted = false;

    accessor
        .start_point_distances(|id, distance| {
            scratch.visited.insert(id);
            scratch.best.insert(Neighbor::new(id, distance));
            scratch.cmps += 1;
        })
        .await?;

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();

        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            scratch.beam_nodes.push(closest_node.id);
        }

        if scratch.beam_nodes.is_empty() {
            break;
        }

        neighbors.clear();
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        for neighbor in neighbors.iter() {
            scratch.best.insert(*neighbor);

            // Sample the bucket of each newly visited node until L is resized.
            if !l_adjusted {
                if let Some(bucket) = provider.get(neighbor.id) {
                    *bucket_counts.entry(bucket).or_insert(0) += 1;
                }
                sample_visited += 1;
            }
        }

        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // Once enough nodes are sampled, estimate bucket-concentration yield and
        // grow L. `matched` is the number of "useful" diverse slots observed:
        // Σ_b min(count_b, k). A low yield => few distinct buckets => larger L.
        if !l_adjusted && sample_visited >= adaptive_l.sample_count() {
            l_adjusted = true;
            let matched = diverse_matched_slots(bucket_counts.values().copied(), diverse_results_k);
            let new_l =
                compute_adaptive_l(l_search, sample_visited, matched, adaptive_l.scale_factor());
            if new_l > l_search {
                scratch.resize(new_l);
            }
        }
    }

    Ok(InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    })
}

/// Count the "useful" diverse slots observed in a bucket-count sample.
///
/// This is `Σ_b min(count_b, k)`: each distinct bucket contributes at most `k`
/// slots, mirroring the per-bucket cap applied by the downstream bucket
/// selector. Feeding this as `matched` into [`compute_adaptive_l`] makes the
/// adaptive-L yield equal to the fraction of visited nodes that would survive
/// bucket selection — high yield (mostly distinct buckets) leaves `L`
/// unchanged, low yield (heavy bucket concentration) grows it.
fn diverse_matched_slots<I>(bucket_counts: I, diverse_results_k: usize) -> usize
where
    I: IntoIterator<Item = usize>,
{
    bucket_counts
        .into_iter()
        .map(|count| count.min(diverse_results_k))
        .sum()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_slots_all_distinct_k1() {
        // Every node in its own bucket, k=1: every node is a useful slot.
        let counts = [1, 1, 1, 1];
        assert_eq!(diverse_matched_slots(counts, 1), 4);
    }

    #[test]
    fn matched_slots_single_bucket_k1() {
        // All nodes in one bucket, k=1: only one useful slot.
        let counts = [10];
        assert_eq!(diverse_matched_slots(counts, 1), 1);
    }

    #[test]
    fn matched_slots_caps_per_bucket() {
        // k=2: each bucket contributes at most 2.
        let counts = [5, 1, 3];
        assert_eq!(diverse_matched_slots(counts, 2), 2 + 1 + 2);
    }

    #[test]
    fn matched_slots_empty_sample() {
        let counts: [usize; 0] = [];
        assert_eq!(diverse_matched_slots(counts, 1), 0);
    }

    #[test]
    fn matched_slots_drives_adaptive_l_growth() {
        // 100 visited nodes, k=1. High concentration (few distinct buckets)
        // must grow L; full diversity (all distinct) must leave it unchanged.
        let base_l = 100;
        let max_multiplier = 8.0;

        // 100 nodes across 100 distinct buckets => yield 1.0 => no growth.
        let all_distinct = std::iter::repeat_n(1usize, 100);
        let matched = diverse_matched_slots(all_distinct, 1);
        assert_eq!(
            compute_adaptive_l(base_l, 100, matched, max_multiplier),
            base_l
        );

        // 100 nodes across 10 distinct buckets => yield 0.1 => 2x growth.
        let ten_buckets = std::iter::repeat_n(10usize, 10);
        let matched = diverse_matched_slots(ten_buckets, 1);
        assert_eq!(
            compute_adaptive_l(base_l, 100, matched, max_multiplier),
            2 * base_l
        );

        // 100 nodes in a single bucket => yield 0.01 => 4x growth.
        let one_bucket = std::iter::once(100usize);
        let matched = diverse_matched_slots(one_bucket, 1);
        assert_eq!(
            compute_adaptive_l(base_l, 100, matched, max_multiplier),
            4 * base_l
        );
    }
}
