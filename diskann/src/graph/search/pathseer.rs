/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PathSeer search for in-memory graphs.

use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use diskann_utils::future::SendFuture;
use hashbrown::HashSet;

use super::{Knn, Search};
use crate::{
    ANNResult,
    error::IntoANNResult,
    graph::{
        AdjacencyList,
        glue::{self, FilteredAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::PathSeerGraph,
    utils::VectorId,
};

/// Two-phase filtered graph search for an in-memory index.
///
/// Until `L` matching results have been found, all direct neighbors can be
/// scored and up to one graph degree of second-order adjacency entries are
/// examined for matching points. Once the result heap is full, second-order
/// expansion stops. Direct neighbors of a rejected candidate are prefiltered;
/// direct neighbors of an accepted candidate remain distance-first.
///
/// Use with a filtered search strategy, for example `labeled::Filtered`, and
/// call `index.search(PathSeer::new(Knn::new_default(l)?), ...)`.
#[derive(Debug, Clone, Copy)]
pub struct PathSeer {
    /// Search list size and beam parameters. PathSeer expands one candidate at a time.
    pub inner: Knn,
}

impl PathSeer {
    /// Construct PathSeer search parameters.
    pub fn new(inner: Knn) -> Self {
        Self { inner }
    }
}

#[derive(Debug, Clone, Copy)]
struct Scored<I> {
    id: I,
    distance: f32,
    accepted: bool,
}

impl<I: Ord> Ord for Scored<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
            .then_with(|| self.accepted.cmp(&other.accepted))
    }
}

impl<I: Ord> PartialOrd for Scored<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Ord> PartialEq for Scored<I> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<I: Ord> Eq for Scored<I> {}

fn admit<I: VectorId>(
    point: Scored<I>,
    l: usize,
    candidates: &mut BinaryHeap<Reverse<Scored<I>>>,
    results: &mut BinaryHeap<Scored<I>>,
) {
    if results.len() == l && point.distance >= results.peek().unwrap().distance {
        return;
    }
    candidates.push(Reverse(point));
    if point.accepted {
        if results.len() == l {
            results.pop();
        }
        results.push(point);
    }
}

/// Allow only selected second-order IDs that have not yet been scored. The
/// filter accessor checks label membership before calling `eval_mut`.
struct SelectedNotVisited<'a, I> {
    selected: &'a HashSet<I>,
    visited: &'a mut HashSet<I>,
}

impl<I: VectorId> glue::Predicate<I> for SelectedNotVisited<'_, I> {
    fn eval(&self, id: &I) -> bool {
        self.selected.contains(id) && !self.visited.contains(id)
    }
}

impl<I: VectorId> glue::PredicateMut<glue::Accept<I>> for SelectedNotVisited<'_, I> {
    fn eval_mut(&mut self, id: &glue::Accept<I>) -> bool {
        self.selected.contains(id.get()) && self.visited.insert(*id.get())
    }
}

impl<'a, DP, S, T> Search<'a, DP, S, T> for PathSeer
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor<Id = DP::InternalId>>,
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
            let l = self.inner.l_value().get();
            let mut scratch = index.search_scratch(l, accessor.num_starting_points().await?);
            let mut candidates = BinaryHeap::new();
            let mut results = BinaryHeap::new();
            let mut starts = Vec::new();
            accessor
                .start_point_distances(|decision, distance| {
                    starts.push(Scored {
                        id: decision.into_inner(),
                        distance,
                        accepted: decision.is_accept(),
                    });
                })
                .await?;
            for point in starts {
                scratch.visited.insert(point.id);
                admit(point, l, &mut candidates, &mut results);
            }
            let budget = index.max_degree_with_slack();
            let mut direct = AdjacencyList::with_capacity(budget);
            let mut second = AdjacencyList::with_capacity(budget);
            let mut scored = Vec::with_capacity(budget * 2);

            while let Some(Reverse(candidate)) = candidates.pop() {
                if accessor.terminate_early() {
                    break;
                }
                if results.len() == l && candidate.distance > results.peek().unwrap().distance {
                    break;
                }
                let phase_two = results.len() == l;
                direct.clear();
                if !phase_two {
                    index
                        .data_provider
                        .pathseer_neighbors(candidate.id, &mut direct)?;
                }
                scored.clear();
                if phase_two && !candidate.accepted {
                    accessor
                        .expand_beam_accept_only(
                            std::iter::once(candidate.id),
                            glue::NotInMut::new(&mut scratch.visited),
                            |id, distance| {
                                scored.push(Scored {
                                    id: id.into_inner(),
                                    distance,
                                    accepted: true,
                                })
                            },
                        )
                        .await?;
                } else {
                    accessor
                        .expand_beam_filtered(
                            std::iter::once(candidate.id),
                            glue::NotInMut::new(&mut scratch.visited),
                            |decision, distance| {
                                scored.push(Scored {
                                    id: decision.into_inner(),
                                    distance,
                                    accepted: decision.is_accept(),
                                })
                            },
                        )
                        .await?;
                }
                scratch.cmps += scored.len() as u32;
                scratch.hops += 1;
                for point in scored.drain(..) {
                    admit(point, l, &mut candidates, &mut results);
                }

                if !phase_two {
                    let mut remaining = budget;
                    for &parent in direct.iter() {
                        if remaining == 0 || accessor.terminate_early() {
                            break;
                        }
                        index
                            .data_provider
                            .pathseer_neighbors(parent, &mut second)?;
                        scratch.hops += 1;
                        let count = remaining.min(second.len());
                        remaining -= count;
                        let selected: HashSet<_> = second.iter().take(count).copied().collect();
                        scored.clear();
                        accessor
                            .expand_beam_accept_only(
                                std::iter::once(parent),
                                SelectedNotVisited {
                                    selected: &selected,
                                    visited: &mut scratch.visited,
                                },
                                |id, distance| {
                                    scored.push(Scored {
                                        id: id.into_inner(),
                                        distance,
                                        accepted: true,
                                    })
                                },
                            )
                            .await?;
                        scratch.cmps += scored.len() as u32;
                        for point in scored.drain(..) {
                            admit(point, l, &mut candidates, &mut results);
                        }
                    }
                }
            }

            let matched: Vec<_> = results
                .into_sorted_vec()
                .into_iter()
                .map(|point| Neighbor::new(point.id, point.distance))
                .collect();
            let result_count = processor
                .post_process(&mut accessor, query, matched.into_iter(), output)
                .await
                .into_ann_result()?;
            Ok(SearchStats {
                cmps: scratch.cmps,
                hops: scratch.hops,
                result_count: result_count as u32,
                range_search_second_round: false,
            })
        }
    }
}
