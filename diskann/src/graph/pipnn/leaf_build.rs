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

use parking_lot::Mutex;

use crate::{ANNResult, graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    conversion::{f32_dimensions, gather_as_f32},
    leaf_kernel::{LeafKernelWorkspace, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    simd::Simd,
    topk::Candidate,
};

/// Scratch for one Rayon leaf job.
///
/// Each buffer keeps the largest size that the job needed. A leaf uses a prefix.
#[derive(Default)]
struct LeafBuffers {
    point_values: Vec<f32>,
    neighbors: Vec<Candidate>,
    local_adjacency: Vec<Vec<u32>>,
    kernel_workspace: LeafKernelWorkspace,
}

impl LeafBuffers {
    /// Grow the buffers for a leaf of `point_count` points and return the
    /// effective `k` of the leaf.
    ///
    /// The buffers never shrink. A smaller leaf uses a prefix, so a later large
    /// leaf does not fill the same slots again.
    fn prepare(&mut self, point_count: usize, dimension_count: usize, requested_k: usize) -> usize {
        // A point has at most `point_count - 1` other points in its leaf. Wider rows
        // would hold only empty slots.
        let leaf_k = requested_k.min(point_count.saturating_sub(1));
        let value_count = point_count * dimension_count;
        if self.point_values.len() < value_count {
            self.point_values.resize(value_count, 0.0);
        }
        let neighbor_count = point_count * leaf_k;
        if self.neighbors.len() < neighbor_count {
            self.neighbors.resize(neighbor_count, Candidate::default());
        }
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
}

/// Build one candidate list per data point from the nearest neighbors in all
/// leaves.
///
/// Each selected pair adds both directions. A list holds global IDs without
/// duplicates, in the order in which the parallel jobs add them. The lists stay
/// unsorted: final pruning ranks candidates by distance, and a graph search reads
/// all neighbors of a point, so the order only breaks distance ties.
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
    let dimensions = f32_dimensions(data)?;
    leaves
        .par_iter()
        .try_for_each_init(LeafBuffers::default, |buffers, point_ids| {
            add_leaf_candidates::<A, M, T>(
                arch,
                data,
                dimensions,
                point_ids,
                requested_k,
                buffers,
                &candidates,
            )
        })?;
    Ok(candidates.into_iter().map(Mutex::into_inner).collect())
}

/// Add the selected pairs of one leaf to the candidate lists.
///
/// The IDs of a leaf are distinct dataset rows, so `points x dimensions` is not
/// larger than the dataset.
fn add_leaf_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    dimensions: usize,
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
    let leaf_k = buffers.prepare(point_count, dimensions, requested_k);
    if leaf_k == 0 {
        return Ok(());
    }

    buffers.prepare_local_adjacency(point_count);
    let mut points = MutMatrixView::try_from(
        &mut buffers.point_values[..point_count * dimensions],
        point_count,
        dimensions,
    )?;
    gather_as_f32(data, point_ids, points.as_mut_view())?;
    let mut output = MutMatrixView::try_from(
        &mut buffers.neighbors[..point_count * leaf_k],
        point_count,
        leaf_k,
    )?;
    select_leaf_neighbors::<A, M>(
        arch,
        points.as_view(),
        output.as_mut_view(),
        &mut buffers.kernel_workspace,
    )?;

    add_symmetric_neighbors(
        point_ids,
        output.as_view(),
        &mut buffers.local_adjacency[..point_count],
    );
    for (&point_id, additions) in point_ids.iter().zip(&buffers.local_adjacency) {
        candidates[point_id as usize]
            .lock()
            .extend_from_slice(additions);
    }
    Ok(())
}

/// Convert each selected pair from leaf positions to global IDs and add it to
/// the edge lists of both points.
fn add_symmetric_neighbors(
    point_ids: &[u32],
    neighbors: MatrixView<'_, Candidate>,
    local_adjacency: &mut [Vec<u32>],
) {
    for (source, source_neighbors) in neighbors.row_iter().enumerate() {
        for neighbor in source_neighbors {
            // Skip the empty slots that the kernel leaves for NaN and infinite
            // distances.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::pipnn::{
        L2,
        test_support::{both_directions, dense_points, edges, thread_pool},
    };
    use diskann_wide::ARCH;
    use half::f16;
    use std::collections::BTreeSet;

    fn build_1d(x: &[f32], leaves: Vec<Vec<u32>>, k: usize, workers: usize) -> Vec<Vec<u32>> {
        let data = MatrixView::try_from(x, x.len(), 1).unwrap();
        let candidates = thread_pool(workers)
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, k))
            .unwrap();
        candidates.into_iter().map(Vec::from).collect()
    }

    #[test]
    fn selected_pairs_add_global_ids_in_both_directions() {
        // Only points 1, 3 and 5 form the leaf; point 1 is at 0.0.
        let x = [0.0, 0.0, 0.0, 2.0, 0.0, 9.0];
        // k = 1: 1 -> 3, 3 -> 1, 5 -> 3
        let expected = both_directions([(1, 3), (3, 5)]);

        let actual = build_1d(&x, vec![vec![5, 1, 3]], 1, 1);

        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn overlapping_leaves_merge_each_neighbor_once() {
        // Point i is at x = i², so its nearest neighbors are its index neighbors.
        let x = [0.0, 1.0, 4.0, 9.0, 16.0];
        let leaves = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4], vec![0, 1, 2]];
        let expected = both_directions([(0, 1), (1, 2), (2, 3), (3, 4)]);

        for workers in [1, 3] {
            let actual = build_1d(&x, leaves.clone(), 1, workers);

            // `edges` panics on duplicate candidates.
            assert_eq!(edges(&actual), expected, "{workers} workers");
        }
    }

    #[test]
    fn k_above_the_leaf_size_selects_every_other_point() {
        // Only points 1, 3 and 5 form the leaf; point 1 is at 0.0.
        let x = [0.0, 0.0, 0.0, 2.0, 0.0, 9.0];
        let expected = both_directions([(1, 3), (1, 5), (3, 5)]);

        let actual = build_1d(&x, vec![vec![1, 3, 5]], 99, 1);

        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn unrankable_pairs_add_no_candidates() {
        let x = [0.0, 3.0, f32::NAN];
        // Point 2 is NaN, so its pairs are not rankable.
        let expected = both_directions([(0, 1)]);

        let actual = build_1d(&x, vec![vec![0, 1, 2]], 2, 1);

        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn candidates_match_the_leaf_kernel_on_each_leaf() {
        // The kernel tests cover metrics, dimensions, widths and architectures.
        // This test covers what leaf build adds: the gather of each element type,
        // k cut to the leaf size, buffers reused across leaf shapes, and the merge
        // of overlapping leaves on four workers.
        let dimensions = 24;
        let values = dense_points(500, dimensions, 1290);
        let half: Vec<_> = values.iter().map(|&x| f16::from_f32(x)).collect();
        // Dense points are multiples of 1/8 in [-2, 14], so eight times them fits
        // `i8`, and an offset of 16 makes them fit `u8`.
        let signed: Vec<_> = values.iter().map(|&x| (8.0 * x) as i8).collect();
        let unsigned: Vec<_> = values.iter().map(|&x| (8.0 * x + 16.0) as u8).collect();
        // Leaves of 1 to 40 points put each k above, equal to and below the other
        // points of a leaf.
        for k in 1..=10 {
            check_against_kernel(&values, dimensions, k);
            check_against_kernel(&half, dimensions, k);
            check_against_kernel(&signed, dimensions, k);
            check_against_kernel(&unsigned, dimensions, k);
        }
    }

    /// Compare `build_leaf_candidates` with the leaf kernel run on each leaf
    /// alone. The kernel sees the same rows in the same order, so ties break the
    /// same way and the check is exact.
    fn check_against_kernel<T: VectorRepr>(values: &[T], dimensions: usize, k: usize) {
        let point_count = values.len() / dimensions;
        let data = MatrixView::try_from(values, point_count, dimensions).unwrap();
        // Sixty overlapping leaves of 1 to 40 points.
        let leaves: Vec<Vec<u32>> = (0..60u32)
            .map(|leaf| {
                let ids: BTreeSet<_> = (0..1 + leaf % 40)
                    .map(|i| (leaf * 7 + i * 13) % point_count as u32)
                    .collect();
                ids.into_iter().collect()
            })
            .collect();

        let mut expected = BTreeSet::new();
        for leaf in &leaves {
            let width = k.min(leaf.len() - 1);
            if width == 0 {
                continue;
            }
            // Convert with `VectorRepr`, not with the gather under test.
            let mut rows = vec![0.0; leaf.len() * dimensions];
            for (&id, row) in leaf.iter().zip(rows.chunks_exact_mut(dimensions)) {
                T::as_f32_into(data.row(id as usize), row).unwrap();
            }
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
                    expected.insert((source, target));
                    expected.insert((target, source));
                }
            }
        }

        let actual = thread_pool(4)
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, k))
            .unwrap();

        let actual: Vec<Vec<u32>> = actual.into_iter().map(Vec::from).collect();
        assert_eq!(
            edges(&actual),
            expected,
            "{}, k={k}",
            std::any::type_name::<T>()
        );
    }

    /// Run `add_leaf_candidates` for one leaf of one-dimensional points `x` with
    /// the given buffers, and return the candidate list of each point.
    fn add_one_leaf(x: &[f32], leaf: &[u32], k: usize, buffers: &mut LeafBuffers) -> Vec<Vec<u32>> {
        let data = MatrixView::try_from(x, x.len(), 1).unwrap();
        let candidates: Vec<_> = (0..x.len())
            .map(|_| Mutex::new(AdjacencyList::new()))
            .collect();

        add_leaf_candidates::<_, L2, _>(ARCH, data, 1, leaf, k, buffers, &candidates).unwrap();

        candidates
            .into_iter()
            .map(|row| Vec::from(row.into_inner()))
            .collect()
    }

    #[test]
    fn reused_buffers_do_not_carry_edges_between_leaves() {
        // Point i is at x = i², so its nearest neighbors are its index neighbors.
        let x = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let mut buffers = LeafBuffers::default();

        // One worker runs a leaf, a smaller leaf with k above its size, then a
        // third leaf with a larger k. Each call must see only its own points.
        for (leaf, k, expected) in [
            (
                vec![0, 1, 2, 3],
                1,
                both_directions([(0, 1), (1, 2), (2, 3)]),
            ),
            (vec![4, 6], 99, both_directions([(4, 6)])),
            (vec![1, 3, 5], 2, both_directions([(1, 3), (1, 5), (3, 5)])),
        ] {
            let actual = add_one_leaf(&x, &leaf, k, &mut buffers);

            assert_eq!(edges(&actual), expected, "leaf {leaf:?}, k={k}");
        }
    }
}
