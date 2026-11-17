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
        glue::{self, FilterClassifier, FilteredAccessor, SearchPostProcess, SearchStrategy},
        index::{DiskANNIndex, SearchStats},
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::PathSeerGraph,
    utils::VectorId,
};

/// Select one independent PathSeer implementation for ablation experiments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PathSeerMethod {
    /// Original PathSeer.
    #[default]
    Original,
    /// Limit total and per-parent matching second-order checks; visit all classified IDs.
    WhatIf1,
    /// Single-hop search with unbounded candidates and a bounded matching-result heap.
    WhatIf2,
    /// Original candidate admission and second-hop expansion, without parent-fail.
    WhatIf3,
    /// Single-hop PathSeer with parent-fail once the matching-result heap is full.
    WhatIf4,
}

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
    method: PathSeerMethod,
}

impl PathSeer {
    /// Construct PathSeer search parameters.
    pub fn new(inner: Knn) -> Self {
        Self {
            inner,
            method: PathSeerMethod::Original,
        }
    }

    /// Select the search implementation used for this request.
    pub fn with_method(mut self, method: PathSeerMethod) -> Self {
        self.method = method;
        self
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
    if results.len() == l
        && results
            .peek()
            .is_some_and(|worst| point.distance >= worst.distance)
    {
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

// Selected IDs are already visited. Remove each after admitting it for scoring.
struct SelectedForScoring<'a, I>(&'a mut HashSet<I>);

impl<I: VectorId> glue::Predicate<I> for SelectedForScoring<'_, I> {
    fn eval(&self, id: &I) -> bool {
        self.0.contains(id)
    }
}

impl<I: VectorId> glue::PredicateMut<I> for SelectedForScoring<'_, I> {
    fn eval_mut(&mut self, id: &I) -> bool {
        self.0.remove(id)
    }
}

impl<I: VectorId> glue::HybridPredicate<I> for SelectedForScoring<'_, I> {}

async fn search_original<'a, DP, S, T, O, PP, OB>(
    inner: Knn,
    index: &'a DiskANNIndex<DP>,
    strategy: &'a S,
    processor: PP,
    context: &'a DP::Context,
    query: T,
    output: &mut OB,
) -> ANNResult<SearchStats>
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor<Id = DP::InternalId>>,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    let mut accessor = strategy
        .search_accessor(&index.data_provider, context, query)
        .into_ann_result()?;
    let l = inner.l_value().get();
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
        if results.len() == l
            && results
                .peek()
                .is_some_and(|worst| candidate.distance > worst.distance)
        {
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

async fn search_what_if_1<'a, DP, S, T, O, PP, OB>(
    inner: Knn,
    index: &'a DiskANNIndex<DP>,
    strategy: &'a S,
    processor: PP,
    context: &'a DP::Context,
    query: T,
    output: &mut OB,
) -> ANNResult<SearchStats>
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilterClassifier<Id = DP::InternalId>>,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    let mut accessor = strategy
        .search_accessor(&index.data_provider, context, query)
        .into_ann_result()?;
    let l = inner.l_value().get();
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
    // DiskANN's adjacency bound plays the role of HNSW's layer-zero 2M bound.
    let max_second_order = ((budget as f64).ln().round() as usize).max(1);
    let mut direct = AdjacencyList::with_capacity(budget);
    let mut second = AdjacencyList::with_capacity(budget);
    let mut scored = Vec::with_capacity(budget * 2);
    let mut selected = HashSet::with_capacity(max_second_order);
    let mut expanded_second = HashSet::new();

    while let Some(Reverse(candidate)) = candidates.pop() {
        if accessor.terminate_early() {
            break;
        }
        if results.len() == l
            && results
                .peek()
                .is_some_and(|worst| candidate.distance > worst.distance)
        {
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
            // All parents share one raw classification budget per candidate.
            let mut remaining = budget;
            for &parent in direct.iter() {
                if remaining == 0 || accessor.terminate_early() {
                    break;
                }
                // With a fixed query filter, a parent's matching prefix is fixed.
                // Its entire scanned prefix was visited on the first expansion, so
                // another expansion cannot discover a new point to score.
                if expanded_second.contains(&parent) {
                    continue;
                }
                second.clear();
                index
                    .data_provider
                    .pathseer_neighbors(parent, &mut second)?;
                scratch.hops += 1;
                let mut satisfactory_count = 0;
                selected.clear();
                let mut checked = 0;
                for &id in second.iter() {
                    if remaining == 0 || satisfactory_count == max_second_order {
                        break;
                    }
                    // Classify even visited IDs: matching IDs consume this parent's
                    // quota regardless of whether they need a distance computation.
                    remaining -= 1;
                    checked += 1;
                    let accepted = accessor.classify(id).is_accept();
                    let unvisited = scratch.visited.insert(id);
                    if accepted {
                        satisfactory_count += 1;
                        if unvisited {
                            selected.insert(id);
                        }
                    }
                }
                // A budget-truncated prefix may still have useful entries later.
                // Cache only parents scanned to the end or to their matching quota.
                if checked == second.len() || satisfactory_count == max_second_order {
                    expanded_second.insert(parent);
                }
                // Do not reread an adjacency list with no work left.
                if selected.is_empty() {
                    continue;
                }
                scored.clear();
                accessor
                    .expand_beam_preclassified(
                        std::iter::once(parent),
                        SelectedForScoring(&mut selected),
                        |id, distance| {
                            scored.push(Scored {
                                id,
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

async fn search_what_if_2<'a, DP, S, T, O, PP, OB>(
    inner: Knn,
    index: &'a DiskANNIndex<DP>,
    strategy: &'a S,
    processor: PP,
    context: &'a DP::Context,
    query: T,
    output: &mut OB,
) -> ANNResult<SearchStats>
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor<Id = DP::InternalId>>,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    let mut accessor = strategy
        .search_accessor(&index.data_provider, context, query)
        .into_ann_result()?;
    let l = inner.l_value().get();
    let mut scratch = index.search_scratch(l, accessor.num_starting_points().await?);
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::with_capacity(l);
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
        if !scratch.visited.insert(point.id) {
            continue;
        }
        // Candidate admission is unconditional, including when results is full.
        candidates.push(Reverse(point));
        if point.accepted
            && (results.len() < l
                || results
                    .peek()
                    .is_some_and(|worst: &Scored<DP::InternalId>| point.distance < worst.distance))
        {
            if results.len() == l {
                results.pop();
            }
            results.push(point);
        }
    }
    let mut scored = Vec::with_capacity(index.max_degree_with_slack());

    // Exhaustion is one normal stopping condition. Otherwise, use the Lth
    // matching result as the distance gate when popping the nearest candidate.
    while let Some(Reverse(candidate)) = candidates.pop() {
        if accessor.terminate_early() {
            break;
        }
        if results.len() == l
            && results
                .peek()
                .is_some_and(|worst| candidate.distance > worst.distance)
        {
            break;
        }
        scored.clear();
        // Always score and classify every unseen one-hop neighbor, even for a
        // rejected parent after the result heap becomes full. No two-hop scan.
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
        scratch.cmps += scored.len() as u32;
        scratch.hops += 1;
        for point in scored.drain(..) {
            candidates.push(Reverse(point));
            if point.accepted
                && (results.len() < l
                    || results
                        .peek()
                        .is_some_and(|worst| point.distance < worst.distance))
            {
                if results.len() == l {
                    results.pop();
                }
                results.push(point);
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

async fn search_what_if_3<'a, DP, S, T, O, PP, OB>(
    inner: Knn,
    index: &'a DiskANNIndex<DP>,
    strategy: &'a S,
    processor: PP,
    context: &'a DP::Context,
    query: T,
    output: &mut OB,
) -> ANNResult<SearchStats>
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor<Id = DP::InternalId>>,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    let mut accessor = strategy
        .search_accessor(&index.data_provider, context, query)
        .into_ann_result()?;
    let l = inner.l_value().get();
    let mut scratch = index.search_scratch(l, accessor.num_starting_points().await?);
    let mut candidates = BinaryHeap::new();
    let mut results = BinaryHeap::with_capacity(l);
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
        if !scratch.visited.insert(point.id) {
            continue;
        }
        // Use the same result-distance admission gate as original PathSeer.
        admit(point, l, &mut candidates, &mut results);
    }
    let budget = index.max_degree_with_slack();
    let mut direct = AdjacencyList::with_capacity(budget);
    let mut second = AdjacencyList::with_capacity(budget);
    let mut scored = Vec::with_capacity(budget * 2);

    // Exhaustion is one normal stopping condition. Otherwise, use the Lth
    // matching result as the distance gate when popping the nearest candidate.
    while let Some(Reverse(candidate)) = candidates.pop() {
        if accessor.terminate_early() {
            break;
        }
        if results.len() == l
            && results
                .peek()
                .is_some_and(|worst| candidate.distance > worst.distance)
        {
            break;
        }
        // As in original PathSeer, snapshot the phase at the start of expansion.
        // Direct neighbors may fill results during this expansion; its already
        // selected phase-one second-hop work still runs to its budget.
        let phase_one = results.len() < l;
        direct.clear();
        if phase_one {
            index
                .data_provider
                .pathseer_neighbors(candidate.id, &mut direct)?;
        }
        scored.clear();
        // Every unseen direct neighbor is scored, regardless of parent acceptance
        // and result count. This variant has no parent-fail prefilter.
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
        scratch.cmps += scored.len() as u32;
        scratch.hops += 1;
        for point in scored.drain(..) {
            admit(point, l, &mut candidates, &mut results);
        }

        if phase_one {
            // Share the original raw second-hop budget across all direct parents.
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
                // Only accepted second-hop IDs pass eval_mut and become visited.
                // Rejected IDs remain reachable later as direct neighbors.
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

async fn search_what_if_4<'a, DP, S, T, O, PP, OB>(
    inner: Knn,
    index: &'a DiskANNIndex<DP>,
    strategy: &'a S,
    processor: PP,
    context: &'a DP::Context,
    query: T,
    output: &mut OB,
) -> ANNResult<SearchStats>
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilteredAccessor<Id = DP::InternalId>>,
    T: Copy + Send + Sync,
    O: Send,
    PP: SearchPostProcess<S::SearchAccessor, T, O> + Send + Sync,
    OB: SearchOutputBuffer<O> + Send + ?Sized,
{
    let mut accessor = strategy
        .search_accessor(&index.data_provider, context, query)
        .into_ann_result()?;
    let l = inner.l_value().get();
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
    let mut scored = Vec::with_capacity(index.max_degree_with_slack());

    while let Some(Reverse(candidate)) = candidates.pop() {
        if accessor.terminate_early() {
            break;
        }
        if results.len() == l
            && results
                .peek()
                .is_some_and(|worst| candidate.distance > worst.distance)
        {
            break;
        }
        // Phase 1 scores every unseen direct neighbor. Once L matches exist,
        // Phase 2 prefilters neighbors only when the parent is rejected.
        let phase_two = results.len() == l;
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

impl<'a, DP, S, T> Search<'a, DP, S, T> for PathSeer
where
    DP: PathSeerGraph,
    S: SearchStrategy<'a, DP, T, SearchAccessor: FilterClassifier<Id = DP::InternalId>>,
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
            match self.method {
                PathSeerMethod::Original => {
                    search_original(
                        self.inner, index, strategy, processor, context, query, output,
                    )
                    .await
                }
                PathSeerMethod::WhatIf1 => {
                    search_what_if_1(
                        self.inner, index, strategy, processor, context, query, output,
                    )
                    .await
                }
                PathSeerMethod::WhatIf2 => {
                    search_what_if_2(
                        self.inner, index, strategy, processor, context, query, output,
                    )
                    .await
                }
                PathSeerMethod::WhatIf3 => {
                    search_what_if_3(
                        self.inner, index, strategy, processor, context, query, output,
                    )
                    .await
                }
                PathSeerMethod::WhatIf4 => {
                    search_what_if_4(
                        self.inner, index, strategy, processor, context, query, output,
                    )
                    .await
                }
            }
        }
    }
}
