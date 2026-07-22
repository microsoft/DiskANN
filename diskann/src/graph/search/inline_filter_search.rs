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
        glue::{self, FilteredAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
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

    /// Number of nodes to sample before estimating specificity and resizing L.
    #[inline]
    pub(crate) fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Maximum multiplier applied to the base L.
    #[inline]
    pub(crate) fn scale_factor(&self) -> f64 {
        self.scale_factor
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
pub struct InlineFilterSearch {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Adaptive L for the search.
    pub adaptive_l: Option<AdaptiveL>,
}

impl InlineFilterSearch {
    /// Create new inline filter search parameters.
    pub fn new(inner: Knn, adaptive_l: Option<AdaptiveL>) -> Self {
        Self { inner, adaptive_l }
    }
}

impl<'a, DP, S, T> Search<'a, DP, S, T> for InlineFilterSearch
where
    DP: DataProvider,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor>,
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

            let num_starting_points = accessor.num_starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), num_starting_points);

            let Ret {
                cmps,
                hops,
                matched_results,
            } = inline_filter_search_internal(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &mut scratch,
                &mut NoopSearchRecord::new(),
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

            let stats = SearchStats {
                cmps,
                hops,
                range_search_second_round: false,
                result_count: result_count as u32,
            };

            Ok(stats)
        }
    }
}

#[derive(Debug)]
struct Ret<I>
where
    I: Eq,
{
    cmps: u32,
    hops: u32,
    matched_results: Vec<Neighbor<I>>,
}

async fn inline_filter_search_internal<I, A, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    adaptive_l: Option<AdaptiveL>,
) -> ANNResult<Ret<I>>
where
    I: VectorId,
    A: FilteredAccessor<Id = I>,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();
    let l_search = search_params.l_value().get();

    // Matched results tracked separately — scratch.best contains all nodes
    // for greedy navigation, matched_results contains only filter-matching nodes.
    let mut matched_results = Vec::new();

    accessor
        .start_point_distances(|id, distance| {
            scratch.visited.insert(id.into_inner());
            scratch
                .best
                .insert(Neighbor::new(id.into_inner(), distance));

            if let glue::Decision::Accept(id) = id {
                matched_results.push(Neighbor::new(id.into_inner(), distance));
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
            .expand_beam_filtered(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| one_hop_neighbors.push((id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for (decision, distance) in one_hop_neighbors.iter().copied() {
            if let glue::Decision::Accept(id) = decision {
                // matched nodes also go into matched_results for final output.
                matched_results.push(Neighbor::new(id.into_inner(), distance));
                sample_matched += 1;
            }

            // All nodes go into scratch.best for navigation,
            scratch
                .best
                .insert(Neighbor::new(decision.into_inner(), distance));
            sample_visited += 1;
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

    Ok(Ret {
        cmps: scratch.cmps,
        hops: scratch.hops,
        matched_results,
    })
}

/// Compute adaptive L based on observed specificity.
///
/// Piecewise scaling:
///   specificity ≥ 50%   → 1× L (no change, most nodes match)
///   10% ≤ specificity < 50%  → 2× L
///   specificity < 10%   → log-scale: 2^(-log10(specificity))
///     specificity = 0.01 (1%)    → 4× L
///     specificity = 0.001 (0.1%) → 8× L
///   0 matches in sample → `max_multiplier`× L (maximum expansion)
///
/// Clamped to [1×, max_multiplier] range.
pub(crate) fn compute_adaptive_l(
    base_l: usize,
    visited: usize,
    matched: usize,
    max_multiplier: f64,
) -> usize {
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
