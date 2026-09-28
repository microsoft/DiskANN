/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf construction: select the nearest neighbors inside each leaf and merge
//! them into one candidate list per data point.
//!
//! A leaf job gathers the vectors of its points as `f32`. The leaf kernel then
//! selects the `k` nearest leaf points of each point, and the job adds each
//! selected pair in both directions. Leaves overlap and run in parallel, so each
//! data point has a locked candidate list. A job groups its edges by point first
//! and then locks each list once.
//!
//! The HashPrune merge groups the edges of a leaf the same way. Each edge also
//! keeps its distance. The job adds the edges of each source point to the locked
//! reservoir of that point.

use parking_lot::Mutex;

use crate::{ANNError, ANNResult, graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    conversion::gather_as_f32,
    hash_prune::HashPrune,
    leaf_kernel::{LeafKernelWorkspace, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    simd::Simd,
    topk::Candidate,
};

/// Scratch for one Rayon leaf job.
///
/// Each buffer keeps the largest size that the job needed. A leaf uses a prefix.
/// The direct merge uses `local_adjacency`. The HashPrune merge uses the edge
/// buffers and `sketch_scratch`.
#[derive(Default)]
struct LeafBuffers {
    point_values: Vec<f32>,
    neighbors: Vec<Candidate>,
    local_adjacency: Vec<Vec<u32>>,
    kernel_workspace: LeafKernelWorkspace,
    seen_pairs: Vec<bool>,
    edge_offsets: Vec<u32>,
    edges: Vec<(u32, f32)>,
    edge_cursor: Vec<u32>,
    sketch_scratch: Vec<f32>,
}

impl LeafBuffers {
    /// Grow the buffers for a leaf of `point_count` points and return the
    /// effective `k` of the leaf.
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

    /// Clear the edge lists of the first `point_count` leaf points.
    fn prepare_local_adjacency(&mut self, point_count: usize) {
        if self.local_adjacency.len() < point_count {
            self.local_adjacency.resize_with(point_count, Vec::new);
        }
        self.local_adjacency[..point_count]
            .iter_mut()
            .for_each(Vec::clear);
    }

    /// Grow the `source x target` flags of a leaf of `point_count` points.
    ///
    /// A leaf has at most `u32::MAX` distinct points, so the product fits in a
    /// 64-bit `usize`.
    fn prepare_seen_pairs(&mut self, point_count: usize) {
        grow(&mut self.seen_pairs, point_count * point_count, false);
    }
}

/// Build one candidate list per data point from the nearest neighbors in all
/// leaves.
///
/// Each selected pair adds both directions. A list holds global IDs without
/// duplicates. The lists are sorted, so the result does not depend on the order
/// in which the parallel jobs finish.
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
            add_leaf_candidates::<A, M, T>(arch, data, point_ids, requested_k, buffers, &candidates)
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

/// Add the selected pairs of all leaves, with their distances, to the HashPrune
/// reservoirs.
///
/// Each selected pair adds both directions.
pub(super) fn add_hash_prune_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
    reservoirs: &HashPrune,
) -> ANNResult<()>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    leaves
        .par_iter()
        .try_for_each_init(LeafBuffers::default, |buffers, point_ids| {
            let leaf_k =
                gather_leaf_neighbors::<A, M, T>(arch, data, point_ids, requested_k, buffers)?;
            if leaf_k == 0 {
                return Ok(());
            }
            let point_count = point_ids.len();
            buffers.prepare_seen_pairs(point_count);
            let edge_count = build_symmetric_edge_csr(
                leaf_k,
                &buffers.neighbors[..point_count * leaf_k],
                EdgeBuffers {
                    seen: &mut buffers.seen_pairs[..point_count * point_count],
                    offsets: &mut buffers.edge_offsets,
                    edges: &mut buffers.edges,
                    cursor: &mut buffers.edge_cursor,
                },
            )?;
            reservoirs.add_leaf_edges(
                point_ids,
                &buffers.edge_offsets[..point_count + 1],
                &buffers.edges[..edge_count],
                &mut buffers.sketch_scratch,
            );
            Ok(())
        })
}

/// Add the selected pairs of one leaf to the candidate lists.
fn add_leaf_candidates<A, M, T>(
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
    let leaf_k = gather_leaf_neighbors::<A, M, T>(arch, data, point_ids, requested_k, buffers)?;
    if leaf_k == 0 {
        return Ok(());
    }

    let point_count = point_ids.len();
    buffers.prepare_local_adjacency(point_count);
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..point_count * leaf_k],
        &mut buffers.local_adjacency[..point_count],
    );
    for (&point_id, additions) in point_ids.iter().zip(&buffers.local_adjacency) {
        candidates[point_id as usize]
            .lock()
            .extend_from_slice(additions);
    }
    Ok(())
}

/// Select the `k` nearest leaf points of each point of one leaf into
/// `buffers.neighbors`, and return the effective `k` of the leaf.
///
/// The IDs of a leaf are distinct dataset rows, so `points x dimensions` is not
/// larger than the dataset.
fn gather_leaf_neighbors<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
) -> ANNResult<usize>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    let point_count = point_ids.len();
    let leaf_k = buffers.prepare(point_count, data.ncols(), requested_k);
    if leaf_k == 0 {
        return Ok(0);
    }

    let point_values = &mut buffers.point_values[..point_count * data.ncols()];
    gather_as_f32(data, point_ids, point_values)?;
    let points = MatrixView::try_from(&*point_values, point_count, data.ncols())?;
    let output = MutMatrixView::try_from(
        &mut buffers.neighbors[..point_count * leaf_k],
        point_count,
        leaf_k,
    )?;
    select_leaf_neighbors::<A, M>(arch, points, output, &mut buffers.kernel_workspace)?;
    Ok(leaf_k)
}

/// Convert each selected pair from leaf positions to global IDs and add it to
/// the edge lists of both points.
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
            // The kernel never selects a point for itself, and the IDs of a leaf
            // are unique, so no list gets a self edge.
            local_adjacency[source].push(point_ids[target]);
            local_adjacency[target].push(point_ids[source]);
        }
    }
}

/// Reusable storage of [`build_symmetric_edge_csr`].
struct EdgeBuffers<'a> {
    /// `source x target` flags of the leaf. All flags are clear between calls.
    seen: &'a mut [bool],
    offsets: &'a mut Vec<u32>,
    edges: &'a mut Vec<(u32, f32)>,
    cursor: &'a mut Vec<u32>,
}

/// Write the directed edges of one leaf as a CSR matrix and return the edge count.
///
/// Each selected pair adds both directions, and each direction appears once.
/// Row `source` lists `(target, distance)` in leaf positions, in the order of
/// the first selection. `offsets[..=point_count]` holds the row bounds.
fn build_symmetric_edge_csr(
    leaf_k: usize,
    neighbors: &[Candidate],
    buffers: EdgeBuffers<'_>,
) -> ANNResult<usize> {
    let EdgeBuffers {
        seen,
        offsets,
        edges,
        cursor,
    } = buffers;
    let point_count = neighbors.len() / leaf_k;
    grow(offsets, point_count + 1, 0);
    offsets[..point_count + 1].fill(0);

    // The count pass sets the flag of each new direction and counts it in
    // `offsets[source + 1]`. A source has at most `point_count - 1` targets, and
    // a leaf has at most `u32::MAX` points, so a count fits in `u32`.
    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors.iter().filter(|neighbor| neighbor.is_assigned()) {
            let target = neighbor.local_idx as usize;
            count_directed_edge(point_count, source, target, seen, offsets);
            count_directed_edge(point_count, target, source, seen, offsets);
        }
    }
    for point in 1..=point_count {
        offsets[point] = offsets[point]
            .checked_add(offsets[point - 1])
            .ok_or_else(|| ANNError::message("a leaf has more than u32::MAX directed edges"))?;
    }

    let edge_count = offsets[point_count] as usize;
    grow(edges, edge_count, (0, 0.0));
    grow(cursor, point_count, 0);
    cursor[..point_count].copy_from_slice(&offsets[..point_count]);
    let edges = &mut edges[..edge_count];
    let cursor = &mut cursor[..point_count];

    // The write pass visits the directions in the same order. The first visit
    // of a direction writes its edge and clears its flag for the next leaf.
    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors.iter().filter(|neighbor| neighbor.is_assigned()) {
            let target = neighbor.local_idx as usize;
            let distance = neighbor.distance;
            write_counted_directed_edge(point_count, source, target, distance, seen, edges, cursor);
            write_counted_directed_edge(point_count, target, source, distance, seen, edges, cursor);
        }
    }
    Ok(edge_count)
}

fn count_directed_edge(
    point_count: usize,
    source: usize,
    target: usize,
    seen: &mut [bool],
    offsets: &mut [u32],
) {
    let seen_entry = &mut seen[source * point_count + target];
    if !*seen_entry {
        *seen_entry = true;
        offsets[source + 1] += 1;
    }
}

fn write_counted_directed_edge(
    point_count: usize,
    source: usize,
    target: usize,
    distance: f32,
    seen: &mut [bool],
    edges: &mut [(u32, f32)],
    cursor: &mut [u32],
) {
    let seen_entry = &mut seen[source * point_count + target];
    if *seen_entry {
        *seen_entry = false;
        edges[cursor[source] as usize] = (target as u32, distance);
        cursor[source] += 1;
    }
}

fn grow<T: Clone>(values: &mut Vec<T>, len: usize, value: T) {
    if values.len() < len {
        values.resize(len, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{dense_points, sorted_members_per_row, thread_pool},
    };
    use diskann_wide::ARCH;
    use std::collections::BTreeSet;

    fn build(values: &[f32], leaves: Vec<Vec<u32>>, k: usize, workers: usize) -> Vec<Vec<u32>> {
        let data = MatrixView::try_from(values, values.len(), 1).unwrap();
        let candidates = thread_pool(workers)
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, k))
            .unwrap();
        sorted_members_per_row(&candidates.into_iter().map(Vec::from).collect::<Vec<_>>())
    }

    #[test]
    fn selected_pairs_add_global_ids_in_both_directions() {
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
    fn leaves_without_pairs_add_no_candidates() {
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
    fn k_above_the_leaf_size_selects_every_other_point() {
        let values = [100.0_f32, 0.0, -100.0, 2.0, 200.0, 9.0];

        let actual = build(&values, vec![vec![1, 3, 5]], 99, 1);

        assert_eq!(
            actual,
            [vec![], vec![3, 5], vec![], vec![1, 5], vec![], vec![1, 3]]
        );
    }

    #[test]
    fn unrankable_pairs_add_no_candidates() {
        let values = [0.0_f32, 3.0, f32::NAN];

        let actual = build(&values, vec![vec![0, 1, 2]], 2, 1);

        assert_eq!(actual, [vec![1], vec![0], vec![]]);
    }

    #[test]
    fn candidates_match_the_leaf_kernel_on_each_leaf() {
        // The reference runs the leaf kernel on each leaf alone, so the tie order
        // is the same. Sixty overlapping leaves of 17 to 40 points run on four
        // workers with reused buffers and fill whole SIMD groups. k = 2 uses a
        // fixed top-k width, and k = 5 uses the runtime width.
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
    fn reused_buffers_do_not_carry_edges_between_leaves() {
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

            add_leaf_candidates::<_, L2, _>(
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

    #[test]
    fn edge_csr_lists_each_direction_once_in_first_selection_order() {
        // Each case gives the kernel output of a leaf with `k` columns.
        // `Candidate::default()` is an empty slot.
        for (case, k, neighbors, expected_offsets, expected_edges) in [
            (
                // The pair 1-2 is selected by both points. Its first selection,
                // from point 1, sets the distance of both directions.
                "three points",
                1,
                vec![
                    Candidate::new(1, 1.0),
                    Candidate::new(2, 2.0),
                    Candidate::new(1, 1.5),
                ],
                vec![0, 1, 3, 4],
                vec![(1, 1.0), (0, 1.0), (2, 2.0), (1, 2.0)],
            ),
            (
                "empty slot",
                1,
                vec![Candidate::new(1, 1.0), Candidate::default()],
                vec![0, 1, 2],
                vec![(1, 1.0), (0, 1.0)],
            ),
            (
                "pair selected from both ends",
                1,
                vec![Candidate::new(1, 1.0), Candidate::new(0, 1.0)],
                vec![0, 1, 2],
                vec![(1, 1.0), (0, 1.0)],
            ),
            (
                "two neighbors per point",
                2,
                vec![
                    Candidate::new(1, 1.0),
                    Candidate::new(2, 4.0),
                    Candidate::new(0, 1.0),
                    Candidate::new(2, 1.0),
                    Candidate::new(1, 1.0),
                    Candidate::new(0, 4.0),
                ],
                vec![0, 2, 4, 6],
                vec![(1, 1.0), (2, 4.0), (0, 1.0), (2, 1.0), (0, 4.0), (1, 1.0)],
            ),
        ] {
            let point_count = neighbors.len() / k;
            let mut seen = vec![false; point_count * point_count];
            // Stale values from a larger earlier leaf must not leak into this one.
            let mut offsets = vec![9; point_count + 3];
            let mut edges = vec![(99, 99.0); 8];
            let mut cursor = vec![9; point_count + 3];

            let edge_count = build_symmetric_edge_csr(
                k,
                &neighbors,
                EdgeBuffers {
                    seen: &mut seen,
                    offsets: &mut offsets,
                    edges: &mut edges,
                    cursor: &mut cursor,
                },
            )
            .unwrap();

            assert_eq!(edge_count, expected_edges.len(), "{case}");
            assert_eq!(offsets[..=point_count], expected_offsets, "{case}");
            assert_eq!(edges[..edge_count], expected_edges, "{case}");
            assert!(seen.iter().all(|&flag| !flag), "{case}: flags stay set");
        }
    }
}
