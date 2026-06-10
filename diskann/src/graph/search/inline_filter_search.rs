/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Inline label-filtered search with optional adaptive-L sizing.

use diskann_utils::future::SendFuture;
use thiserror::Error;

use super::{Knn, Search, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNError, ANNErrorKind, ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, SearchAccessor, SearchPostProcess, SearchStrategy},
        index::{
            DiskANNIndex, InternalSearchStats, QueryLabelProvider, QueryVisitDecision, SearchStats,
        },
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
    utils::VectorId,
};
/// Error type for [`Knn`] parameter validation.
#[derive(Debug, Error)]
pub enum AdaptiveLSearchError {
    #[error("adaptive L scale factor must be >= 1.0")]
    ScaleFactorLessThanOne,
    #[error("sample count cannot be zero")]
    SampleCountZero,
}

impl From<AdaptiveLSearchError> for ANNError {
    #[track_caller]
    fn from(err: AdaptiveLSearchError) -> Self {
        Self::new(ANNErrorKind::IndexError, err)
    }
}

/// Adaptive L for inline filtered search.
#[derive(Debug, Clone)]
pub struct AdaptiveL {
    sample_count: usize,
    scale_factor: f64,
}

impl AdaptiveL {
    /// Create a new adaptive L.
    pub fn new(sample_count: usize, scale_factor: f64) -> Result<Self, AdaptiveLSearchError> {
        if scale_factor < 1.0 {
            return Err(AdaptiveLSearchError::ScaleFactorLessThanOne);
        }
        if sample_count == 0 {
            return Err(AdaptiveLSearchError::SampleCountZero);
        }
        Ok(Self {
            sample_count,
            scale_factor,
        })
    }
}

/// Inline filtered search: a standard k-NN search with an additional step
/// of keeping track of all results satisfying the query predicate, and
/// returning only those that meet the criteria.
///
/// An additional option for better performance on low specificity scenarios
/// is the use of the adaptive L algorithm. After visiting a set number of nodes,
/// and estimating the specificity of the filter from that sample, `l_search` is
/// scaled up in the following manner:
///   specificity ≥ 50%    → 1× L (no change, most nodes match)
///   10% ≤ specificity < 50% → 2× L
///   specificity < 10%    → log-scale: 2^(-log10(specificity))
///     specificity = 10%  (100/1000) → 2× L
///     specificity = 1%   (10/1000)  → 4× L
///     specificity = 0.1% (1/1000)   → 8× L
///   and so on up to a pre-set maximum multiplier
#[derive(Debug)]
pub struct InlineFilterSearch<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining node matches and early termination.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
    /// Adaptive L for the search.
    pub adaptive_l: Option<AdaptiveL>,
}

impl<'q, InternalId> InlineFilterSearch<'q, InternalId> {
    /// Create new inline filter search parameters.
    pub fn new(
        inner: Knn,
        label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
        adaptive_l: Option<AdaptiveL>,
    ) -> Self {
        Self {
            inner,
            label_evaluator,
            adaptive_l,
        }
    }
}

impl<'a, 'q, DP, S, T> Search<'a, DP, S, T> for InlineFilterSearch<'q, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T>,
    T: Copy + Send + Sync,
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

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), start_ids.len());

            let (stats, matched_results) = inline_filter_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                self.label_evaluator,
                self.adaptive_l,
            )
            .await?;

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    matched_results.into_iter().take(self.inner.l_value().get()),
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn inline_filter_search_internal<I, A, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    query_label_evaluator: &dyn QueryLabelProvider<I>,
    adaptive_l: Option<AdaptiveL>,
) -> ANNResult<(InternalSearchStats, Vec<Neighbor<I>>)>
where
    I: VectorId,
    A: SearchAccessor<Id = I>,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();
    let l_search = search_params.l_value().get();

    // Helper to build the final stats from scratch state.
    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Matched results tracked separately — scratch.best contains all nodes
    // for greedy navigation, matched_results contains only filter-matching nodes.
    let mut matched_results = Vec::new();

    accessor
        .start_point_distances(|id, distance| {
            scratch.visited.insert(id);
            scratch.best.insert(Neighbor::new(id, distance));
            // Check if the start point matches the filter
            // Note that we don't allow termination on start points. This is mostly a moot point
            // as we're planning to get rid of the termination option for `on_visit` anyway
            if query_label_evaluator.on_visit(Neighbor::new(id, distance))
                == QueryVisitDecision::Accept(Neighbor::new(id, distance))
            {
                matched_results.push(Neighbor::new(id, distance));
            }
        })
        .await?;

    // Pre-allocate with good capacity to avoid repeated allocations
    let mut one_hop_neighbors = Vec::with_capacity(max_degree_with_slack);

    let mut sample_visited: usize = 0;
    let mut sample_matched: usize = 0;
    let mut l_adjusted = false;

    loop {
        // Check termination conditions
        if accessor.terminate_early() {
            break;
        }

        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();

        // Fill beam from scratch.best (all nodes participate in navigation)
        while scratch.beam_nodes.len() < beam_width {
            let Some(closest_node) = scratch.best.closest_notvisited() else {
                break;
            };
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        // Exit if no nodes to process
        if scratch.beam_nodes.is_empty() {
            break;
        }

        // compute distances from query to one-hop neighbors, and mark them visited
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| one_hop_neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for neighbor in one_hop_neighbors.iter().copied() {
            let decision = query_label_evaluator.on_visit(neighbor);

            match decision {
                QueryVisitDecision::Accept(accepted) => {
                    // All nodes go into scratch.best for navigation,
                    // matched nodes also go into matched_results for final output.
                    scratch.best.insert(accepted);
                    matched_results.push(accepted);
                    sample_matched += 1;
                }
                QueryVisitDecision::Reject => {
                    // Unmatched nodes still guide navigation
                    scratch.best.insert(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    matched_results.sort_unstable_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    return Ok((make_stats(scratch), matched_results));
                }
            }
            if adaptive_l.is_some() {
                sample_visited += 1;
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        // Adaptive L: after enough samples, estimate specificity and scale L.
        if let Some(adaptive_l) = adaptive_l.as_ref()
            && !l_adjusted
            && sample_visited >= adaptive_l.sample_count
        {
            l_adjusted = true;
            let new_l = compute_adaptive_l(
                l_search,
                sample_visited,
                sample_matched,
                adaptive_l.scale_factor,
            );
            if new_l > l_search {
                scratch.resize(new_l);
            }
        }
    }

    matched_results.sort_unstable_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok((make_stats(scratch), matched_results))
}

/// Compute adaptive L based on observed specificity.
///
/// Piecewise scaling:
/// Piecewise scaling:
///   specificity ≥ 50%   → 1× L (no change, most nodes match)
///   10% ≤ specificity < 50%  → 2× L
///   specificity < 10%   → log-scale: 2^(-log10(specificity))
///     specificity = 0.01 (1%)    → 4× L
///     specificity = 0.001 (0.1%) → 8× L
///   0 matches in sample → `max_multiplier`× L (maximum expansion)
///
/// Clamped to [1×, max_multiplier] range.
fn compute_adaptive_l(base_l: usize, visited: usize, matched: usize, max_multiplier: f64) -> usize {
    let specificity = if visited > 0 && matched > 0 { matched as f64 / visited as f64 } else { 0.0 };
    let branch = if matched == 0 || visited == 0 { "zero" } else if specificity >= 0.5 { ">=50%" } else if specificity >= 0.1 { "10-50%" } else { "<10% log" };
    eprintln!("[adaptive_l] base_l={base_l} visited={visited} matched={matched} specificity={specificity:.4} branch={branch}");
    if matched == 0 || visited == 0 {
        // No matches at all — use maximum multiplier
        return (base_l as f64 * max_multiplier) as usize;
    }

    let specificity = matched as f64 / visited as f64;

    let multiplier = if specificity >= 0.5 {
        // ≥50% specificity: no scaling needed
        1.0
    } else if specificity >= 0.1 {
        // 10%-50%: use 2× L
        2.0
    } else {
        // Below 10%: log-scale from 2× upward
        // specificity=0.1 → 2×, specificity=0.01 → 4×, specificity=0.001 → 8×
        let neg_log10 = -specificity.log10(); // 0.1→1, 0.01→2, 0.001→3
        2.0_f64.powf(neg_log10) // 2^1=2, 2^2=4, 2^3=8
    };

    let multiplier = multiplier.clamp(1.0, max_multiplier);
    (base_l as f64 * multiplier) as usize
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_l_validation() {
        // Valid
        assert!(AdaptiveL::new(1000, 1.0).is_ok());
        assert!(AdaptiveL::new(1000, 2.0).is_ok());

        // Invalid: scale factor < 1.0
        assert!(matches!(
            AdaptiveL::new(1000, 0.99),
            Err(AdaptiveLSearchError::ScaleFactorLessThanOne)
        ));

        // Invalid: sample count = 0
        assert!(matches!(
            AdaptiveL::new(0, 1.5),
            Err(AdaptiveLSearchError::SampleCountZero)
        ));
    }

    #[test]
    fn test_compute_adaptive_l_piecewise_regions() {
        let base_l = 100;
        let max_multiplier = 16.0;

        // >= 50% specificity => 1x
        assert_eq!(compute_adaptive_l(base_l, 1000, 500, max_multiplier), 100);
        assert_eq!(compute_adaptive_l(base_l, 1000, 900, max_multiplier), 100);

        // 10% to <50% specificity => 2x
        assert_eq!(compute_adaptive_l(base_l, 1000, 100, max_multiplier), 200);
        assert_eq!(compute_adaptive_l(base_l, 1000, 499, max_multiplier), 200);

        // <10% specificity => log scaling (0.01 => 4x, 0.001 => 8x)
        assert_eq!(compute_adaptive_l(base_l, 1000, 10, max_multiplier), 400);
        assert_eq!(compute_adaptive_l(base_l, 1000, 1, max_multiplier), 800);
    }

    #[test]
    fn test_compute_adaptive_l_zero_samples_or_matches() {
        let base_l = 100;
        let max_multiplier = 16.0;

        assert_eq!(compute_adaptive_l(base_l, 1000, 0, max_multiplier), 1600);
        assert_eq!(compute_adaptive_l(base_l, 0, 0, max_multiplier), 1600);
    }

    #[test]
    fn test_compute_adaptive_l_respects_max_multiplier() {
        let base_l = 100;

        // 0.1% would be 8x, but clamp to 4x.
        assert_eq!(compute_adaptive_l(base_l, 1000, 1, 4.0), 400);

        // 1% would be 4x, but clamp to 1.5x.
        assert_eq!(compute_adaptive_l(base_l, 1000, 10, 1.5), 150);
    }
}

#[cfg(test)]
mod probe_tests {
    use super::*;
    use crate::graph::{self, search::Knn, search_output_buffer, test::provider as test_provider, test::synthetic::Grid, index::QueryLabelProvider};
    use diskann_vector::distance::Metric;
    use crate::test::tokio::current_thread_runtime;
    use std::collections::HashSet;

    #[derive(Debug)]
    struct EvenlyDistributedFilter { ids: HashSet<u32> }
    impl EvenlyDistributedFilter {
        fn new(total: usize, matching: usize) -> Self {
            let mut ids = HashSet::with_capacity(matching);
            for i in 0..matching {
                let id = ((2 * i + 1) * total / (2 * matching)) as u32;
                ids.insert(id.min((total - 1) as u32));
            }
            Self { ids }
        }
    }
    impl QueryLabelProvider<u32> for EvenlyDistributedFilter {
        fn is_match(&self, id: u32) -> bool { self.ids.contains(&id) }
    }

    fn run(matching: usize, sc: usize, l: usize, k: usize, adaptive: bool) -> usize {
        let grid_size = 10;
        let provider = test_provider::Provider::grid(Grid::Three, grid_size).unwrap();
        let cfg = graph::config::Builder::new(provider.max_degree(), graph::config::MaxDegree::same(), 100, Metric::L2.into()).build().unwrap();
        let index = std::sync::Arc::new(graph::DiskANNIndex::new(cfg, provider, None));
        let filter = EvenlyDistributedFilter::new(1000, matching);
        let adapt = if adaptive { Some(AdaptiveL::new(sc, 16.0).unwrap()) } else { None };
        let inline = super::super::InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, adapt);
        let mut ids = vec![0u32; k];
        let mut dists = vec![0.0f32; k];
        let mut buf = search_output_buffer::IdDistance::new(&mut ids, &mut dists);
        let rt = current_thread_runtime();
        let stats = rt.block_on(index.search(inline, &test_provider::Strategy::new(), &test_provider::Context::new(), [5.0f32, 5.0, 5.0].as_slice(), &mut buf)).unwrap();
        stats.result_count as usize
    }

    #[test]
    fn probe_branches() {
        for (matching, k, l, sc) in [(50, 5, 20, 50), (100, 5, 20, 50), (100, 5, 20, 100), (50, 5, 50, 50), (100, 5, 50, 50), (100, 5, 50, 100), (50, 5, 100, 100)] {
            eprintln!("\n=== m={matching} k={k} l={l} sc={sc} ===");
            eprintln!("-- non-adaptive --");
            let rc_no = run(matching, sc, l, k, false);
            eprintln!("-- adaptive --");
            let rc_ad = run(matching, sc, l, k, true);
            eprintln!("result: no_adapt={rc_no} adaptive={rc_ad}");
        }
    }
}
#[cfg(test)]
mod probe_tests2 {
    use super::*;
    use crate::graph::{self, search::Knn, search_output_buffer, test::provider as test_provider, test::synthetic::Grid, index::QueryLabelProvider};
    use diskann_vector::distance::Metric;
    use crate::test::tokio::current_thread_runtime;
    use std::collections::HashSet;

    /// High-ID filter plus a few extra scattered IDs to control specificity.
    #[derive(Debug)]
    struct HybridFilter { ids: HashSet<u32> }
    impl HybridFilter {
        fn new(total: usize, high_id_count: usize, extra_scattered: usize) -> Self {
            let mut ids = HashSet::new();
            // High-ID matches near query region
            for i in 0..high_id_count {
                ids.insert((total - 1 - i) as u32);
            }
            // Scattered matches throughout
            for i in 0..extra_scattered {
                let id = ((2 * i + 1) * total / (2 * extra_scattered.max(1))) as u32;
                ids.insert(id.min((total - 1) as u32));
            }
            Self { ids }
        }
    }
    impl QueryLabelProvider<u32> for HybridFilter {
        fn is_match(&self, id: u32) -> bool { self.ids.contains(&id) }
    }

    fn run(high: usize, scattered: usize, sc: usize, l: usize, k: usize, adaptive: bool) -> usize {
        let grid_size = 10;
        let provider = test_provider::Provider::grid(Grid::Three, grid_size).unwrap();
        let cfg = graph::config::Builder::new(provider.max_degree(), graph::config::MaxDegree::same(), 100, Metric::L2.into()).build().unwrap();
        let index = std::sync::Arc::new(graph::DiskANNIndex::new(cfg, provider, None));
        let filter = HybridFilter::new(1000, high, scattered);
        let adapt = if adaptive { Some(AdaptiveL::new(sc, 16.0).unwrap()) } else { None };
        let inline = super::super::InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, adapt);
        let mut ids = vec![0u32; k];
        let mut dists = vec![0.0f32; k];
        let mut buf = search_output_buffer::IdDistance::new(&mut ids, &mut dists);
        let rt = current_thread_runtime();
        let stats = rt.block_on(index.search(inline, &test_provider::Strategy::new(), &test_provider::Context::new(), [5.0f32, 5.0, 5.0].as_slice(), &mut buf)).unwrap();
        stats.result_count as usize
    }

    #[test]
    fn probe_hybrid() {
        // Target <10%: need matched/visited < 0.1. E.g., 3 matches in 50 visits = 6%
        // Target 10-50%: e.g., 10 matches in 50 visits = 20%  
        // Target >=50%: e.g., 30 matches in 50 visits = 60%
        for (high, scattered, k, l, sc, label) in [
            (3, 0, 5, 20, 50, "<10% target"),
            (5, 0, 5, 20, 50, "<10% target v2"),
            (2, 0, 5, 20, 50, "<10% target v3"),
            (10, 0, 5, 20, 50, "10-50% target"),
            (30, 0, 5, 20, 50, ">=50% target"),
            (3, 10, 5, 20, 50, "<10% hybrid"),
            (10, 20, 5, 20, 50, "10-50% hybrid"),
        ] {
            eprintln!("\n=== {label}: high={high} scattered={scattered} k={k} l={l} sc={sc} ===");
            let rc_no = run(high, scattered, sc, l, k, false);
            let rc_ad = run(high, scattered, sc, l, k, true);
            eprintln!("result: no_adapt={rc_no} adaptive={rc_ad} improved={}", rc_ad > rc_no);
        }
    }
}
#[cfg(test)]
mod probe_tests3 {
    use super::*;
    use crate::graph::{self, search::Knn, search_output_buffer, test::provider as test_provider, test::synthetic::Grid, index::QueryLabelProvider};
    use diskann_vector::distance::Metric;
    use crate::test::tokio::current_thread_runtime;
    use std::collections::HashSet;

    #[derive(Debug)]
    struct HybridFilter { ids: HashSet<u32> }
    impl HybridFilter {
        fn new(total: usize, high_id_count: usize, extra_scattered: usize) -> Self {
            let mut ids = HashSet::new();
            for i in 0..high_id_count {
                ids.insert((total - 1 - i) as u32);
            }
            for i in 0..extra_scattered {
                let id = ((2 * i + 1) * total / (2 * extra_scattered.max(1))) as u32;
                ids.insert(id.min((total - 1) as u32));
            }
            Self { ids }
        }
    }
    impl QueryLabelProvider<u32> for HybridFilter {
        fn is_match(&self, id: u32) -> bool { self.ids.contains(&id) }
    }

    fn run(high: usize, scattered: usize, sc: usize, l: usize, k: usize, adaptive: bool) -> (usize, Vec<u32>) {
        let grid_size = 10;
        let provider = test_provider::Provider::grid(Grid::Three, grid_size).unwrap();
        let cfg = graph::config::Builder::new(provider.max_degree(), graph::config::MaxDegree::same(), 100, Metric::L2.into()).build().unwrap();
        let index = std::sync::Arc::new(graph::DiskANNIndex::new(cfg, provider, None));
        let filter = HybridFilter::new(1000, high, scattered);
        let adapt = if adaptive { Some(AdaptiveL::new(sc, 16.0).unwrap()) } else { None };
        let inline = super::super::InlineFilterSearch::new(Knn::new_default(k, l).unwrap(), &filter, adapt);
        let mut ids = vec![0u32; k];
        let mut dists = vec![0.0f32; k];
        let mut buf = search_output_buffer::IdDistance::new(&mut ids, &mut dists);
        let rt = current_thread_runtime();
        let stats = rt.block_on(index.search(inline, &test_provider::Strategy::new(), &test_provider::Context::new(), [5.0f32, 5.0, 5.0].as_slice(), &mut buf)).unwrap();
        let rc = stats.result_count as usize;
        (rc, ids[..rc].to_vec())
    }

    #[test]
    fn probe_hybrid_improvement() {
        // Idea: high-ID matches seed the sample so we hit non-zero branches,
        // scattered matches give adaptive L extra results to find
        for (high, scattered, k, l, sc, label) in [
            // <10% branch: 3 high-ID seeds + scattered for adaptive to find
            (3, 20, 5, 20, 50, "<10% + scattered 20"),
            (3, 30, 5, 20, 50, "<10% + scattered 30"),
            (3, 50, 5, 20, 50, "<10% + scattered 50"),
            (3, 50, 10, 20, 50, "<10% + scattered 50, k=10"),
            (3, 50, 10, 30, 50, "<10% + scattered 50, k=10, l=30"),
            (3, 100, 10, 30, 50, "<10% + scattered 100, k=10, l=30"),
            // 10-50% branch with scattered
            (30, 50, 10, 20, 50, "10-50% + scattered 50, k=10"),
            (30, 100, 10, 30, 50, "10-50% + scattered 100, k=10, l=30"),
        ] {
            eprintln!("\n=== {label}: high={high} scat={scattered} ===");
            let (rc_no, ids_no) = run(high, scattered, sc, l, k, false);
            let (rc_ad, ids_ad) = run(high, scattered, sc, l, k, true);
            eprintln!("no_adapt={rc_no} adaptive={rc_ad} improved={} ids_diff={}", rc_ad > rc_no, rc_ad as i32 - rc_no as i32);
        }
    }
}
