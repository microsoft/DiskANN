/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::VecDeque};

use hashbrown::HashSet;

use crate::{
    ANNResult,
    graph::{
        AdjacencyList,
    },
    provider::NeighborAccessor,
};

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
