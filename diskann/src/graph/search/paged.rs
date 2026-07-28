/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::future::SendFuture;

use crate::{
    ANNError, ANNResult,
    graph::{
        DiskANNIndex,
        glue::SearchAccessor,
        search::{record::NoopSearchRecord, scratch::SearchScratch},
    },
    neighbor::{Neighbor, NeighborPriorityQueue},
    provider::DataProvider,
    utils::VectorId,
};

/// Intermediate state for paged search.
///
/// Each call to [`next_page`](Self::next_page) resumes the graph search and returns the
/// next page of nearest-neighbor results. Returns an empty `Vec` when the search is exhausted.
///
/// See also: [`DiskANNIndex::paged_search`], [`DiskANNIndex::paged_search_with_init_ids`].
#[derive(Debug)]
pub struct PagedSearch<'a, DP, A>
where
    DP: DataProvider,
    A: SearchAccessor<Id = DP::InternalId> + 'a,
{
    pub(in crate::graph) index: &'a DiskANNIndex<DP>,
    pub(in crate::graph) scratch: SearchScratch<DP::InternalId>,
    pub(in crate::graph) computed_result: Vec<Neighbor<DP::InternalId>>,
    pub(in crate::graph) next_result_index: usize,
    pub(in crate::graph) search_param_l: usize,
    pub(in crate::graph) accessor: A,
}

impl<'a, DP, A> PagedSearch<'a, DP, A>
where
    DP: DataProvider,
    A: SearchAccessor<Id = DP::InternalId> + 'a,
{
    /// Returns the next page of at most `k` nearest-neighbor results.
    ///
    /// Results across pages are non-overlapping but not guaranteed to be monotonic with
    /// respect to distance.
    ///
    /// Within a page, results ordered by non-decreasing distance.
    ///
    /// When the search is exhausted, returns an empty `Vec`.
    pub fn next_page(
        &mut self,
        k: usize,
    ) -> impl SendFuture<ANNResult<Vec<Neighbor<DP::InternalId>>>> {
        async move {
            if k > self.search_param_l {
                return ANNResult::Err(ANNError::log_paged_search_error(
                    "k should be less than or equal to search_param_l".to_string(),
                ));
            }
            if k == 0 {
                return ANNResult::Err(ANNError::log_paged_search_error(
                    "k should be greater than 0".to_string(),
                ));
            }

            let mut result = Vec::with_capacity(k);

            // Drain any already-computed results first.
            let available = self
                .computed_result
                .len()
                .saturating_sub(self.next_result_index);
            let from_cache = std::cmp::min(k, available);
            if from_cache > 0 {
                result.extend_from_slice(
                    &self.computed_result
                        [self.next_result_index..self.next_result_index + from_cache],
                );
                self.next_result_index += from_cache;

                if result.len() == k {
                    return ANNResult::Ok(result);
                }
            }

            // Resume graph search to fill the next batch.
            self.index
                .search_internal(
                    None, // beam_width
                    &mut self.accessor,
                    &mut self.scratch,
                    &mut NoopSearchRecord::new(),
                )
                .await?;

            let start_points = self.accessor.starting_points().await?;
            let (mut candidates, total_considered) =
                filter_search_candidates(&start_points, k, &mut self.scratch.best);
            self.scratch.best.drain_best(total_considered);

            let computed_result_count = candidates.len();
            self.computed_result.clear();
            self.computed_result.append(&mut candidates);
            self.next_result_index = 0;

            let remaining_need = k - result.len();
            let leftover = std::cmp::min(remaining_need, computed_result_count);
            if leftover > 0 {
                result.extend_from_slice(
                    &self.computed_result
                        [self.next_result_index..self.next_result_index + leftover],
                );
                self.next_result_index += leftover;
            }

            ANNResult::Ok(result)
        }
    }
}

// FIXME: Wire proper post-processing support into paged search.
fn filter_search_candidates<I>(
    start_points: &[I],
    page_size: usize,
    best: &mut NeighborPriorityQueue<I>,
) -> (Vec<Neighbor<I>>, usize)
where
    I: VectorId,
{
    let mut total = 0usize;
    let mut candidates = Vec::with_capacity(page_size);
    for n in best.iter() {
        total += 1;
        if !start_points.contains(n.id()) {
            candidates.push(n);
            if candidates.len() >= page_size {
                break;
            }
        }
    }

    debug_assert!(
        page_size.min(best.size().saturating_sub(start_points.len())) <= candidates.len(),
        "Not enough candidates after filtering starting points",
    );

    (candidates, total)
}
