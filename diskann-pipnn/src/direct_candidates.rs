/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Concurrent, deduplicated leaf-candidate accumulation without HashPrune.

use diskann::{graph::AdjacencyList, ANNError, ANNResult};
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::rayon_util::ParIterInstalled;

pub(crate) struct DirectCandidates {
    rows: Vec<Mutex<AdjacencyList<u32>>>,
}

impl DirectCandidates {
    pub(crate) fn new(points: usize) -> ANNResult<Self> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(points).map_err(ANNError::opaque)?;
        rows.resize_with(points, || Mutex::new(AdjacencyList::new()));
        Ok(Self { rows })
    }

    pub(crate) fn add_leaf_edges(
        &self,
        point_ids: &[u32],
        edge_offsets: &[u32],
        edges: &[(u32, f32)],
    ) {
        // ponytail: candidate rows are small in the measured one-shot workload
        // (55 IDs on average at BigANN10M), so reuse AdjacencyList's linear SIMD
        // membership check; switch to per-row hash sets only if profiles regress.
        debug_assert_eq!(edge_offsets.len(), point_ids.len() + 1);
        for (local_source, offsets) in edge_offsets.windows(2).enumerate() {
            let source = point_ids[local_source] as usize;
            let mut row = self.rows[source].lock();
            for &(local_target, _) in &edges[offsets[0] as usize..offsets[1] as usize] {
                row.push(point_ids[local_target as usize]);
            }
        }
    }

    pub(crate) fn into_rows(self) -> ANNResult<Vec<Vec<u32>>> {
        self.rows
            .par_iter()
            .for_each_installed(|row| row.lock().sort());

        let mut rows = Vec::new();
        rows.try_reserve_exact(self.rows.len())
            .map_err(ANNError::opaque)?;
        rows.extend(self.rows.into_iter().map(|row| Vec::from(row.into_inner())));
        Ok(rows)
    }
}

#[cfg(test)]
mod tests;
