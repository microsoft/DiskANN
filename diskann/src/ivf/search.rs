/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Minimal IVF-Flat: exact centroid selection followed by exact L2 list scans.

use std::cmp::Ordering;

use diskann_utils::future::SendFuture;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use super::{IVFIndex, traits::Provider, workingset::WorkingSet};
use crate::{ANNError, ANNResult, neighbor::Neighbor};

/// Read-only search boundary. No build transaction or publication is performed.
pub trait SearchProvider: Provider {
    fn dim(&self) -> usize;

    /// Return the nearest `min(nprobe, live_list_count)` live lists, selected by
    /// squared L2 to their centroids; break centroid ties by list ID. Borrow the
    /// complete, finite, `dim()`-dimensional vectors in those lists. Each internal
    /// point ID must occur in exactly one live list. Selection and membership
    /// reads must use a consistent snapshot. Return an error if uninitialized.
    fn search_lists<'a>(
        &'a self,
        context: &'a Self::Context,
        query: &'a [f32],
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<WorkingSet<'a, Self::InternalId, Self::ListId>>>;
}

/// Neighbors use internal IDs. Convert through `Provider::to_external_id` if needed.
#[derive(Debug)]
pub struct SearchResult<I> {
    /// At most k hits, ordered by (squared L2, internal ID).
    pub neighbors: Vec<Neighbor<I>>,
    /// Includes selected empty lists.
    pub lists_scanned: usize,
    /// Number of point-distance evaluations, not centroid-distance evaluations.
    pub candidates_scanned: usize,
}

impl<P: SearchProvider> IVFIndex<P> {
    /// Basic read-only IVF-Flat search, optimized for small k rather than large k.
    /// A small sorted top-k buffer avoids materializing all candidate distances.
    ///
    /// `k == 0` returns no hits and does not read lists. Fewer than k candidates
    /// produces fewer hits. An oversized nprobe is clamped by the provider.
    ///
    /// # Errors
    /// Rejects zero nprobe, a wrong-dimensional/non-finite query, an uninitialized
    /// provider, and malformed point dimensions; propagates provider read errors.
    pub fn search<'a>(
        &'a self,
        context: &'a P::Context,
        query: &'a [f32],
        k: usize,
        nprobe: usize,
    ) -> impl SendFuture<ANNResult<SearchResult<P::InternalId>>> {
        async move {
            let provider = self.provider();
            if nprobe == 0
                || query.is_empty()
                || query.len() != provider.dim()
                || !query.iter().all(|value| value.is_finite())
            {
                return Err(ANNError::message("invalid IVF search query or nprobe"));
            }
            let mut result = SearchResult {
                neighbors: Vec::new(),
                lists_scanned: 0,
                candidates_scanned: 0,
            };
            if k == 0 {
                return Ok(result);
            }
            let lists = provider.search_lists(context, query, nprobe).await?;
            if lists.is_empty() {
                return Err(ANNError::message("IVF search returned no live list"));
            }
            result.lists_scanned = lists.len();
            for point in lists.iter().flat_map(|list| &list.points) {
                if point.vector.len() != query.len() {
                    return Err(ANNError::message("invalid IVF stored point dimension"));
                }
                let hit = Neighbor::new(point.id, SquaredL2::evaluate(query, point.vector));
                result.candidates_scanned += 1;
                if result.neighbors.len() == k && !compare(&hit, &result.neighbors[k - 1]).is_lt() {
                    continue;
                }
                let pos = result
                    .neighbors
                    .partition_point(|old| compare(old, &hit).is_lt());
                result.neighbors.insert(pos, hit);
                result.neighbors.truncate(k);
            }
            Ok(result)
        }
    }
}

fn compare<I: Ord>(a: &Neighbor<I>, b: &Neighbor<I>) -> Ordering {
    a.distance()
        .total_cmp(b.distance())
        .then_with(|| a.id().cmp(b.id()))
}
