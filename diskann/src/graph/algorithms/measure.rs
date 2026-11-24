/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Algorithms measuring statistics about a graph.

use std::collections::VecDeque;

use hashbrown::HashSet;

use crate::{ANNResult, graph::AdjacencyList, provider::NeighborAccessor};

/// Count the number of distinct nodes reachable from the start points.
///
/// Let `N` be the number of reachable nodes. This algorithm has a memory consumption roughly
/// on the order of `(2 * N).next_power_of(2)` and will make `N` calls to
/// [`NeighborAccessor::get_neighbors`].
pub async fn reachable_from<A>(accessor: &mut A, start: &[A::Id]) -> ANNResult<usize>
where
    A: NeighborAccessor,
{
    // Filter out redundant start points and use the deduplicated set as the start seed.
    let mut seen: HashSet<_> = start.iter().copied().collect();
    let mut queue: VecDeque<_> = seen.iter().copied().collect();

    let mut neighbors = AdjacencyList::new();
    while let Some(node) = queue.pop_front() {
        accessor.get_neighbors(node, neighbors.as_empty()).await?;
        for &neighbor in neighbors.iter() {
            if seen.insert(neighbor) {
                queue.push_back(neighbor)
            }
        }
    }

    Ok(seen.len())
}
