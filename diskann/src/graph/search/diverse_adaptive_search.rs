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
//! from the observed bucket concentration.
//!
//! # Yield metric
//!
//! The diversity *yield* of a sample measures how close the sampled nodes are
//! to a maximally diverse set, normalized to `[0, 1]`:
//!
//! ```text
//! yield = (Σ_b min(count_b, k)) / min(sample_visited, num_buckets · k)
//! ```
//!
//! where `count_b` is the number of sampled nodes in bucket `b`, `k` is
//! `diverse_results_k` (the per-bucket cap applied downstream), `num_buckets`
//! is the total number of distinct attribute buckets in the dataset (reported
//! by [`AttributeValueProvider::num_buckets`]), and `sample_visited` is the
//! number of nodes sampled. The denominator is the tight upper bound on the
//! numerator, coupling the yield to all three quantities so it spans the full
//! `[0, 1]` range regardless of how few buckets the dataset has or how small
//! the sample is:
//!
//! * few buckets, large sample → denominator is `num_buckets · k`, so covering
//!   every bucket reaches `1.0`;
//! * many buckets, small sample → denominator is `sample_visited`, so a fully
//!   distinct sample reaches `1.0` and diffuse data leaves `L` unchanged.
//!
//! Unlike inline-filter specificity — where a value of `0.5` over a large
//! sample is already good enough to stop enlarging `L` — a diversity yield of
//! `0.5` means only ~50% of the achievable diversity has been observed so far,
//! so `L` should still grow substantially. `L` is therefore scaled from the
//! *deficit* `1 − yield`: a yield near `1.0` leaves `L` unchanged, while a low
//! yield grows it toward the configured maximum multiplier. If the provider
//! cannot report `num_buckets`, adaptive growth is skipped and `L` stays at its
//! base value.

use std::sync::Arc;

use diskann_utils::future::SendFuture;
use hashbrown::HashMap;

use super::{AdaptiveL, Knn, Search, scratch::SearchScratch};
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

    // Total distinct buckets in the dataset. When unknown, adaptive growth is
    // disabled: the yield cannot be normalized, so `L` stays at its base value.
    let num_buckets = provider.num_buckets();

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

        // Once enough nodes are sampled, estimate the bucket-diversity yield
        // and grow L from its deficit. `matched` is the number of "useful"
        // diverse slots observed (Σ_b min(count_b, k)); dividing by the maximum
        // achievable slots normalizes the yield to [0, 1]. A low yield => far
        // from a full diverse set => larger L.
        if !l_adjusted && sample_visited >= adaptive_l.sample_count() {
            l_adjusted = true;
            if let Some(num_buckets) = num_buckets {
                let matched =
                    diverse_matched_slots(bucket_counts.values().copied(), diverse_results_k);
                let diversity_yield =
                    diverse_yield(matched, num_buckets, diverse_results_k, sample_visited);
                let new_l = compute_diverse_adaptive_l(
                    l_search,
                    diversity_yield,
                    adaptive_l.scale_factor(),
                );
                if new_l > l_search {
                    scratch.resize(new_l);
                }
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
/// selector.
fn diverse_matched_slots<I>(bucket_counts: I, diverse_results_k: usize) -> usize
where
    I: IntoIterator<Item = usize>,
{
    bucket_counts
        .into_iter()
        .map(|count| count.min(diverse_results_k))
        .sum()
}

/// Normalize observed diverse slots into a `[0, 1]` yield.
///
/// `matched` is `Σ_b min(count_b, k)` (see [`diverse_matched_slots`]). Its tight
/// upper bound is `min(sample_visited, num_buckets · k)`: `matched` can never
/// exceed the number of nodes sampled, nor the total capped slots across every
/// bucket. Dividing by that bound couples the yield to all three quantities that
/// bound the achievable diversity — `diverse_results_k`, `num_buckets`, and the
/// sample size — so the yield spans the full `[0, 1]` range in both regimes:
///
/// * few buckets, large sample (`num_buckets · k ≤ sample_visited`): the
///   denominator is `num_buckets · k`, so seeing every bucket up to its cap
///   reaches `1.0`.
/// * many buckets, small sample (`sample_visited < num_buckets · k`): the
///   denominator is `sample_visited`, so a fully distinct sample reaches `1.0`
///   and diffuse data keeps `L` unchanged (Design-A behavior).
///
/// The result is clamped to `[0, 1]`; a zero denominator yields `1.0` (nothing
/// to diversify over, so no over-fetch is warranted).
fn diverse_yield(
    matched: usize,
    num_buckets: usize,
    diverse_results_k: usize,
    sample_visited: usize,
) -> f64 {
    let achievable = num_buckets.saturating_mul(diverse_results_k).min(sample_visited);
    if achievable == 0 {
        return 1.0;
    }
    (matched as f64 / achievable as f64).clamp(0.0, 1.0)
}

/// Compute adaptive L from the observed diversity yield.
///
/// `L` is scaled linearly from the yield *deficit* `1 − yield`, so higher yields
/// (closer to a full diverse set) grow `L` less:
///   yield = 1.0 → 1× L (no change; the sample already covers the buckets)
///   yield = 0.5 → midway between 1× and `max_multiplier`× L
///   yield = 0.0 → `max_multiplier`× L (maximum expansion)
///
/// This differs from inline-filter [`compute_adaptive_l`](super::inline_filter_search)
/// on purpose: filter specificity of `0.5` is already good, whereas a diversity
/// yield of `0.5` means only half of the achievable diversity has been seen and
/// warrants substantial further search.
///
/// The multiplier is clamped to `[1, max_multiplier]`.
pub(crate) fn compute_diverse_adaptive_l(
    base_l: usize,
    diversity_yield: f64,
    max_multiplier: f64,
) -> usize {
    let deficit = (1.0 - diversity_yield).clamp(0.0, 1.0);
    let multiplier = (1.0 + (max_multiplier - 1.0) * deficit).clamp(1.0, max_multiplier);
    (base_l as f64 * multiplier) as usize
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
    fn yield_is_normalized_to_unit_range() {
        // 60 buckets, k=1, large sample (1000): denominator is num_buckets·k=60.
        // Seeing all 60 distinct buckets is full diversity, independent of how
        // many nodes were visited to get there.
        let all_distinct = std::iter::repeat_n(1usize, 60);
        let matched = diverse_matched_slots(all_distinct, 1);
        assert_eq!(diverse_yield(matched, 60, 1, 1000), 1.0);

        // Half the buckets covered => yield 0.5.
        let half = std::iter::repeat_n(1usize, 30);
        let matched = diverse_matched_slots(half, 1);
        assert_eq!(diverse_yield(matched, 60, 1, 1000), 0.5);

        // Everything in a single bucket => near-zero yield.
        let one_bucket = std::iter::once(1000usize);
        let matched = diverse_matched_slots(one_bucket, 1);
        assert!((diverse_yield(matched, 60, 1, 1000) - 1.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn yield_many_buckets_small_sample_reaches_one() {
        // Diffuse dataset: far more buckets than the sample size. The
        // denominator is capped at sample_visited, so a fully distinct sample
        // yields 1.0 (no over-fetch) rather than being stuck near zero.
        let all_distinct = std::iter::repeat_n(1usize, 100);
        let matched = diverse_matched_slots(all_distinct, 1);
        assert_eq!(diverse_yield(matched, 12_849, 1, 100), 1.0);

        // Same dataset, but the sample piles into a few buckets => low yield.
        let concentrated = [40usize, 30, 30];
        let matched = diverse_matched_slots(concentrated, 1);
        assert!((diverse_yield(matched, 12_849, 1, 100) - 3.0 / 100.0).abs() < 1e-9);
    }

    #[test]
    fn yield_caps_per_bucket_with_k() {
        // k=2, 10 buckets, sample 20: denominator is num_buckets·k=20. One
        // saturated bucket (>=2) plus one singleton contributes 2 + 1 = 3
        // useful slots => yield 3/20.
        let counts = [5usize, 1];
        let matched = diverse_matched_slots(counts, 2);
        assert_eq!(matched, 3);
        assert!((diverse_yield(matched, 10, 2, 20) - 3.0 / 20.0).abs() < 1e-9);
    }

    #[test]
    fn yield_zero_buckets_is_full() {
        // No buckets to diversify over => treat as fully satisfied, no growth.
        assert_eq!(diverse_yield(0, 0, 1, 100), 1.0);
    }

    #[test]
    fn diverse_adaptive_l_scales_from_deficit() {
        let base_l = 100;
        let max_multiplier = 8.0;

        // Full diversity => no growth.
        assert_eq!(
            compute_diverse_adaptive_l(base_l, 1.0, max_multiplier),
            base_l
        );

        // Zero yield => maximum expansion.
        assert_eq!(
            compute_diverse_adaptive_l(base_l, 0.0, max_multiplier),
            base_l * 8
        );

        // A yield of 0.5 is "bad" for diversity and must grow L a lot: halfway
        // between 1x and 8x => 4.5x.
        assert_eq!(
            compute_diverse_adaptive_l(base_l, 0.5, max_multiplier),
            (base_l as f64 * 4.5) as usize
        );

        // Out-of-range yields are clamped.
        assert_eq!(
            compute_diverse_adaptive_l(base_l, 1.5, max_multiplier),
            base_l
        );
        assert_eq!(
            compute_diverse_adaptive_l(base_l, -0.5, max_multiplier),
            base_l * 8
        );
    }
}
