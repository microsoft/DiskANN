/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Filtered range-based search with a distance radius

use diskann_utils::future::SendFuture;

use super::{Knn, Search, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        glue::{self, FilteredAccessor, SearchStrategy},
        index::{DiskANNIndex, InternalSearchStats, SearchStats},
        search::inline_filter_search::{Ret, inline_filter_search_internal},
        search::{Range, RangeSearchError, range_search::RangeBuilder, record::NoopSearchRecord},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::DataProvider,
};

use super::range_search::DistanceFiltered;

/// Parameters for range-based search.
///
/// Finds all points within a specified distance radius from the query.
#[derive(Debug, Clone, Copy)]
pub struct FilteredRange {
    range_params: Range,
}

impl FilteredRange {
    /// Create range search with default slack values.
    pub fn new(starting_l: usize, radius: f32) -> Result<Self, RangeSearchError> {
        Self::builder(starting_l, radius).build_filtered()
    }

    /// Create a builder for filtered range search parameters.
    ///
    /// The builder starts with the same defaults as [`Self::new`].
    pub fn builder(starting_l: usize, radius: f32) -> RangeBuilder {
        Range::builder(starting_l, radius)
    }

    /// Returns the maximum number of results to return.
    #[inline]
    pub fn max_returned(&self) -> Option<usize> {
        self.range_params.max_returned()
    }

    /// Returns the initial search list size.
    #[inline]
    pub fn starting_l(&self) -> usize {
        self.range_params.starting_l()
    }

    /// Returns the optional beam width.
    #[inline]
    pub fn beam_width(&self) -> Option<usize> {
        self.range_params.beam_width()
    }

    /// Returns the outer radius.
    #[inline]
    pub fn radius(&self) -> f32 {
        self.range_params.radius()
    }

    /// Returns the inner radius (points closer are excluded).
    #[inline]
    pub fn inner_radius(&self) -> Option<f32> {
        self.range_params.inner_radius()
    }

    /// Returns the initial search slack factor.
    #[inline]
    pub fn initial_slack(&self) -> f32 {
        self.range_params.initial_slack()
    }

    /// Returns the range search slack factor.
    #[inline]
    pub fn range_slack(&self) -> f32 {
        self.range_params.range_slack()
    }

    /// Returns the underlying range search parameters.
    #[inline]
    pub fn range(&self) -> Range {
        self.range_params
    }
}

impl RangeBuilder {
    /// Build validated [`FilteredRange`] parameters.
    pub fn build_filtered(self) -> Result<FilteredRange, RangeSearchError> {
        let range_params = self.build()?;
        Ok(FilteredRange { range_params })
    }
}

impl<'a, DP, S, T> Search<'a, DP, S, T> for FilteredRange
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
        PP: glue::SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context, query)
                .into_ann_result()?;
            let num_start_ids = accessor.num_starting_points().await?;
            let mut scratch = index.search_scratch(self.starting_l(), num_start_ids);

            // Perform an initial inline filtered search, store both filtered and unfiltered results

            let search_knn = Knn::new(self.starting_l(), self.starting_l(), self.beam_width())?;

            let Ret {
                cmps,
                hops,
                matched_results,
            } = inline_filter_search_internal(
                index.max_degree_with_slack(),
                &search_knn,
                &mut accessor,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                None,
            )
            .await?;

            let max_returned = self.max_returned().unwrap_or(usize::MAX);

            // merge matched_results with the best results from the first round, filtering by radius

            let mut in_range: Vec<_> = scratch
                .best
                .iter()
                .take(self.starting_l())
                .chain(matched_results.iter().copied())
                .filter(|neighbor| neighbor.distance <= self.radius())
                .collect();

            in_range.sort_unstable_by(|left, right| {
                left.id
                    .cmp(&right.id)
                    .then_with(|| left.distance.total_cmp(&right.distance))
            });
            in_range.dedup_by_key(|neighbor| neighbor.id);

            in_range.sort_unstable_by(|left, right| {
                left.distance
                    .total_cmp(&right.distance)
                    .then_with(|| left.id.cmp(&right.id))
            });

            let mut matched_within_radius = Vec::with_capacity(matched_results.len());
            for neighbor in matched_results.iter().copied() {
                if neighbor.distance <= self.radius() {
                    matched_within_radius.push(neighbor);
                }
            }

            // clear the visited set and repopulate it with all in-range points found so far, filtered and unfiltered
            scratch.visited.clear();
            scratch.range_frontier.clear();
            for neighbor in in_range.iter() {
                scratch.visited.insert(neighbor.id);
                scratch.range_frontier.push_back(neighbor.id);
            }

            let stats = if in_range.len()
                >= ((self.starting_l() as f32) * self.initial_slack()) as usize
                && matched_within_radius.len() < max_returned
            {
                // Move to filtered range search
                let range_stats = filtered_range_search_internal(
                    index.max_degree_with_slack(),
                    &self,
                    &mut accessor,
                    &mut scratch,
                    &mut matched_within_radius,
                )
                .await?;

                InternalSearchStats {
                    cmps,
                    hops: hops + range_stats.hops,
                    range_search_second_round: true,
                }
            } else {
                InternalSearchStats {
                    cmps,
                    hops,
                    range_search_second_round: false,
                }
            };

            // Post-process results directly into the output buffer, filtering by radius.
            let radius = self.radius();
            let inner_radius = self.inner_radius();

            let mut filtered = DistanceFiltered::new(output, |dist| {
                if let Some(ir) = inner_radius
                    && dist <= ir
                {
                    return false;
                }
                dist <= radius
            });

            let truncated_matched = matched_within_radius.iter().copied().take(max_returned);

            let result_count = processor
                .post_process(&mut accessor, query, truncated_matched, &mut filtered)
                .await
                .into_ann_result()?;

            Ok(SearchStats {
                cmps: stats.cmps,
                hops: stats.hops,
                result_count: result_count as u32,
                range_search_second_round: stats.range_search_second_round,
            })
        }
    }
}

/////////////////////////////
// Internal Implementation //
/////////////////////////////

/// Internal range search implementation.
///
/// Expands the search frontier to find all points within the specified radius.
/// Called after the initial graph search has identified starting candidates.
pub(crate) async fn filtered_range_search_internal<A>(
    max_degree_with_slack: usize,
    search_params: &FilteredRange,
    accessor: &mut A,
    scratch: &mut SearchScratch<A::Id>,
    matched_in_range: &mut Vec<Neighbor<A::Id>>,
) -> ANNResult<InternalSearchStats>
where
    A: FilteredAccessor,
{
    let beam_width = search_params.beam_width().unwrap_or(1);

    let mut neighbors = Vec::with_capacity(max_degree_with_slack);

    let max_returned = search_params.max_returned().unwrap_or(usize::MAX);

    while !scratch.range_frontier.is_empty() && matched_in_range.len() < max_returned {
        scratch.beam_nodes.clear();

        // In this loop we are going to find the beam_width number of remaining nodes within the radius
        // Each of these nodes will be a frontier node.
        while !scratch.range_frontier.is_empty() && scratch.beam_nodes.len() < beam_width {
            let next = scratch.range_frontier.pop_front();
            if let Some(next_node) = next {
                scratch.beam_nodes.push(next_node);
            }
        }

        neighbors.clear();
        accessor
            .expand_beam_filtered(
                scratch.beam_nodes.iter().copied(),
                glue::NotInMut::new(&mut scratch.visited),
                |id, distance| neighbors.push((id, distance)),
            )
            .await?;

        // The predicate ensures that the contents of `neighbors` are unique.
        // We still traverse both accepted and rejected IDs via frontier expansion,
        // but only accepted IDs are added to in-range results.
        for (decision, distance) in neighbors.iter().copied() {
            if distance <= search_params.radius() * search_params.range_slack() {
                let is_accept = decision.is_accept();
                let id = decision.into_inner();

                scratch.range_frontier.push_back(id);

                if is_accept && matched_in_range.len() < max_returned {
                    matched_in_range.push(Neighbor::new(id, distance));
                }
            }
        }
        scratch.cmps += neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;
    }

    Ok(InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: true,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_search_validation() {
        // Valid
        assert!(FilteredRange::new(100, 0.5).is_ok());

        // Invalid: zero l
        assert!(FilteredRange::new(0, 0.5).is_err());

        // Invalid slack values
        assert!(
            FilteredRange::builder(100, 0.5)
                .initial_slack(1.5)
                .build_filtered()
                .is_err()
        );
        assert!(
            FilteredRange::builder(100, 0.5)
                .range_slack(0.5)
                .build_filtered()
                .is_err()
        );

        // Invalid inner radius > radius
        assert!(
            FilteredRange::builder(100, 0.5)
                .inner_radius(Some(1.0))
                .build_filtered()
                .is_err()
        );
    }
}
