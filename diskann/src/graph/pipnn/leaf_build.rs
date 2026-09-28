/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local graph construction and candidate accumulation.
//!
//! Partitioning supplies sorted, unique global point IDs for each leaf. One leaf
//! job does these steps:
//!
//! 1. Gather each ID and convert its vector to reusable `f32` storage.
//! 2. Call the leaf kernel for ranking-distance construction and local ranking.
//! 3. Convert local positions to global point IDs.
//! 4. Add both edge directions to global candidate lists.
//!
//! Overlapping leaves run concurrently. A worker locks one destination list only
//! while it adds one leaf's IDs. Reusable buffers keep their largest allocation.
//! Each operation uses an explicit active prefix.

use parking_lot::Mutex;

use crate::{ANNResult, graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    conversion::gather_as_f32,
    leaf_kernel::{LeafKernelWorkspace, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    simd::Simd,
    topk::Candidate,
};

/// Reusable buffers for one Rayon leaf job.
///
/// The numerical vectors keep the largest leaf shape that this job observed.
/// The job creates local adjacency lists only when the effective `k` is not zero.
#[derive(Default)]
struct LeafBuffers {
    point_values: Vec<f32>,
    neighbors: Vec<Candidate>,
    local_adjacency: Vec<Vec<u32>>,
    kernel_workspace: LeafKernelWorkspace,
}

impl LeafBuffers {
    /// Grow the buffers for one leaf and return its effective `k`.
    fn prepare(&mut self, point_count: usize, dimension_count: usize, requested_k: usize) -> usize {
        // A point has at most `point_count - 1` other points in its leaf. Wider rows
        // would hold only empty slots.
        let leaf_k = requested_k.min(point_count.saturating_sub(1));
        grow(&mut self.point_values, point_count * dimension_count, 0.0);
        grow(
            &mut self.neighbors,
            point_count * leaf_k,
            Candidate::default(),
        );
        leaf_k
    }

    fn prepare_local_adjacency(&mut self, point_count: usize) {
        if self.local_adjacency.len() < point_count {
            self.local_adjacency.resize_with(point_count, Vec::new);
        }
        self.local_adjacency[..point_count]
            .iter_mut()
            .for_each(Vec::clear);
    }
}

/// Build direct graph candidates from all overlapping leaves.
///
/// Each selected leaf pair contributes both edge directions. Candidate lists use
/// global dataset IDs and contain no duplicate IDs.
pub(super) fn build_leaf_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    let candidates: Vec<_> = (0..data.nrows())
        .map(|_| Mutex::new(AdjacencyList::new()))
        .collect();
    leaves
        .par_iter()
        .try_for_each_init(LeafBuffers::default, |buffers, point_ids| {
            add_direct_leaf_candidates::<A, M, T>(
                arch,
                data,
                point_ids,
                requested_k,
                buffers,
                &candidates,
            )
        })?;
    Ok(candidates
        .into_iter()
        .map(Mutex::into_inner)
        .map(|mut neighbors| {
            neighbors.sort();
            neighbors
        })
        .collect())
}

/// Add one leaf's symmetric neighbors to the direct candidate lists.
///
/// Reusable buffers can be longer than this leaf, so all accesses use the current
/// leaf shape. Leaf IDs are distinct dataset rows, so `points x dimensions` is no
/// larger than the dataset.
fn add_direct_leaf_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
    candidates: &[Mutex<AdjacencyList<u32>>],
) -> ANNResult<()>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let leaf_k = buffers.prepare(point_count, data.ncols(), requested_k);
    if leaf_k == 0 {
        return Ok(());
    }

    let point_values = &mut buffers.point_values[..point_count * data.ncols()];
    gather_as_f32(data, point_ids, point_values)?;
    let points = MatrixView::try_from(&*point_values, point_count, data.ncols())?;
    let neighbor_count = point_count * leaf_k;
    let output = MutMatrixView::try_from(
        &mut buffers.neighbors[..neighbor_count],
        point_count,
        leaf_k,
    )?;
    select_leaf_neighbors::<A, M>(arch, points, output, &mut buffers.kernel_workspace)?;

    buffers.prepare_local_adjacency(point_count);
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..neighbor_count],
        &mut buffers.local_adjacency[..point_count],
    );
    for (&point_id, additions) in point_ids.iter().zip(&buffers.local_adjacency) {
        candidates[point_id as usize]
            .lock()
            .extend_from_slice(additions);
    }
    Ok(())
}

fn add_symmetric_neighbors(
    point_ids: &[u32],
    leaf_k: usize,
    neighbors: &[Candidate],
    local_adjacency: &mut [Vec<u32>],
) {
    for (source, source_neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in source_neighbors {
            if !neighbor.is_assigned() {
                continue;
            }
            let target = neighbor.local_idx as usize;
            let source_id = point_ids[source];
            let target_id = point_ids[target];
            // Leaves contain unique IDs, and the kernel excludes each point itself.
            local_adjacency[source].push(target_id);
            local_adjacency[target].push(source_id);
        }
    }
}

fn grow<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    if values.len() < len {
        values.resize(len, value);
    }
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{dense_points, sorted_members_per_row, thread_pool},
    };
    use diskann_wide::ARCH;
    use std::collections::BTreeSet;

    #[expect(clippy::unwrap_used, reason = "an invalid fixture fails the test")]
    fn build(values: &[f32], leaves: Vec<Vec<u32>>, k: usize, workers: usize) -> Vec<Vec<u32>> {
        let data = MatrixView::try_from(values, values.len(), 1).unwrap();
        let candidates = thread_pool(workers)
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, k))
            .unwrap();
        sorted_members_per_row(&candidates.into_iter().map(Vec::from).collect::<Vec<_>>())
    }

    #[test]
    fn selected_neighbors_use_global_ids_and_contribute_both_edge_directions() {
        // The leaf lists IDs [5, 1, 3], at coordinates [9, 0, 2].
        // The directed choices are 1 -> 3, 3 -> 1 and 5 -> 3.
        let values = [100.0_f32, 0.0, -100.0, 2.0, 200.0, 9.0];

        let actual = build(&values, vec![vec![5, 1, 3]], 1, 1);

        assert_eq!(
            actual,
            [vec![], vec![3], vec![], vec![1, 5], vec![], vec![3]]
        );
    }

    #[test]
    fn overlapping_leaves_merge_each_neighbor_once() {
        let values = [0.0_f32, 1.0, 4.0, 9.0, 16.0];
        let leaves = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4], vec![0, 1, 2]];

        for workers in [1, 3] {
            let actual = build(&values, leaves.clone(), 1, workers);

            assert_eq!(
                actual,
                [vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]],
                "{workers} workers"
            );
        }
    }

    #[test]
    fn leaves_without_selected_pairs_produce_empty_adjacency() {
        let values = [0.0_f32, 1.0, 4.0];

        for (leaves, k) in [
            (vec![], 2),
            (vec![vec![]], 2),
            (vec![vec![2]], 2),
            (vec![vec![0, 1, 2]], 0),
        ] {
            let case = format!("leaves {leaves:?}, k={k}");

            let actual = build(&values, leaves, k, 1);

            assert_eq!(actual, [Vec::<u32>::new(), vec![], vec![]], "{case}");
        }
    }

    #[test]
    fn requesting_more_neighbors_than_a_leaf_has_selects_all_other_points() {
        let values = [100.0_f32, 0.0, -100.0, 2.0, 200.0, 9.0];

        let actual = build(&values, vec![vec![1, 3, 5]], 99, 1);

        assert_eq!(
            actual,
            [vec![], vec![3, 5], vec![], vec![1, 5], vec![], vec![1, 3]]
        );
    }

    #[test]
    fn unrankable_pairs_do_not_add_unassigned_ids_to_the_graph() {
        let values = [0.0_f32, 3.0, f32::NAN];

        let actual = build(&values, vec![vec![0, 1, 2]], 2, 1);

        assert_eq!(actual, [vec![1], vec![0], vec![]]);
    }

    #[test]
    fn candidates_at_production_shape_are_the_symmetric_leaf_kernel_results() {
        // Sixty overlapping leaves of 17 to 40 points run on four workers with
        // reused buffers. They fill full SIMD groups of the leaf kernel. k = 2
        // uses a fixed top-k width, and k = 5 uses the runtime width. The kernel
        // on each leaf alone is the reference, so tie order cannot differ.
        let (point_count, dimensions) = (500, 24);
        let values = dense_points(point_count, dimensions, 1290);
        let data = MatrixView::try_from(values.as_slice(), point_count, dimensions).unwrap();
        let leaves: Vec<Vec<u32>> = (0..60u32)
            .map(|leaf| {
                let ids: BTreeSet<_> = (0..17 + leaf % 24)
                    .map(|i| (leaf * 7 + i * 13) % point_count as u32)
                    .collect();
                ids.into_iter().collect()
            })
            .collect();

        for k in [2, 5] {
            let mut expected = vec![BTreeSet::new(); point_count];
            for leaf in &leaves {
                let rows: Vec<f32> = leaf
                    .iter()
                    .flat_map(|&id| &values[id as usize * dimensions..][..dimensions])
                    .copied()
                    .collect();
                let width = k.min(leaf.len() - 1);
                let mut output = vec![Candidate::default(); leaf.len() * width];
                select_leaf_neighbors::<_, L2>(
                    ARCH,
                    MatrixView::try_from(rows.as_slice(), leaf.len(), dimensions).unwrap(),
                    MutMatrixView::try_from(output.as_mut_slice(), leaf.len(), width).unwrap(),
                    &mut LeafKernelWorkspace::default(),
                )
                .unwrap();
                for (source, row) in output.chunks_exact(width).enumerate() {
                    for neighbor in row.iter().filter(|neighbor| neighbor.is_assigned()) {
                        let (source, target) = (leaf[source], leaf[neighbor.local_idx as usize]);
                        expected[source as usize].insert(target);
                        expected[target as usize].insert(source);
                    }
                }
            }
            let expected: Vec<Vec<u32>> = expected
                .into_iter()
                .map(|ids| ids.into_iter().collect())
                .collect();

            let actual = thread_pool(4)
                .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves.clone(), k))
                .unwrap();

            let actual: Vec<Vec<u32>> = actual.into_iter().map(Vec::from).collect();
            assert_eq!(actual, expected, "k={k}");
        }
    }

    #[test]
    fn reused_leaf_buffers_do_not_carry_edges_between_different_leaf_shapes() {
        let values = [0.0_f32, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let data = MatrixView::try_from(&values[..], 7, 1).unwrap();
        let mut buffers = LeafBuffers::default();

        // The same worker processes a leaf, a smaller leaf, then more neighbors
        // per point. Every call must use only its current IDs and active rows.
        for (ids, requested_k, expected) in [
            (
                vec![0, 1, 2, 3],
                1,
                vec![
                    vec![1],
                    vec![0, 2],
                    vec![1, 3],
                    vec![2],
                    vec![],
                    vec![],
                    vec![],
                ],
            ),
            (
                vec![4, 6],
                99,
                vec![vec![], vec![], vec![], vec![], vec![6], vec![], vec![4]],
            ),
            (
                vec![1, 3, 5],
                2,
                vec![
                    vec![],
                    vec![3, 5],
                    vec![],
                    vec![1, 5],
                    vec![],
                    vec![1, 3],
                    vec![],
                ],
            ),
        ] {
            let candidates: Vec<_> = (0..7).map(|_| Mutex::new(AdjacencyList::new())).collect();

            add_direct_leaf_candidates::<_, L2, _>(
                ARCH,
                data,
                &ids,
                requested_k,
                &mut buffers,
                &candidates,
            )
            .unwrap();

            let actual: Vec<_> = candidates
                .into_iter()
                .map(|row| Vec::from(row.into_inner()))
                .collect();
            assert_eq!(
                sorted_members_per_row(&actual),
                expected,
                "leaf {ids:?}, k={requested_k}"
            );
        }
    }
}
