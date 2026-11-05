/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf construction: select the nearest neighbors inside each leaf and merge
//! them into one candidate list per data point.
//!
//! A leaf job gathers the vectors of its points as `f32`. The leaf kernel then
//! selects the `k` nearest leaf points of each point, and the job adds each
//! selected pair in both directions. Leaves overlap and their jobs can run in
//! parallel, so each data point has a locked candidate list. A job groups its
//! edges by point first and then locks each list once.

use std::sync::{Mutex, PoisonError};

use crate::{ANNResult, graph::AdjacencyList};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rayon::prelude::*;

use super::{
    dataset::Dataset, kernels::Kernels, leaf_kernel::LeafKernelWorkspace, topk::Candidate,
};

/// Scratch for one leaf job.
///
/// Each buffer keeps the largest allocation that the job needed, and takes
/// the shape of the current leaf.
#[derive(Default)]
struct LeafBuffers {
    points: rowmajor::Buffer<f32>,
    neighbors: rowmajor::Buffer<Candidate>,
    local_adjacency: Vec<Vec<u32>>,
    kernel_workspace: LeafKernelWorkspace,
}

impl LeafBuffers {
    /// Shape the buffers for a leaf of `point_count` points that selects
    /// `leaf_k` neighbors each, and clear the edge lists of those points.
    fn prepare(
        &mut self,
        point_count: usize,
        dimension_count: usize,
        leaf_k: usize,
    ) -> ANNResult<()> {
        self.points.resize(point_count, dimension_count)?;
        self.neighbors.resize(point_count, leaf_k)?;
        if self.local_adjacency.len() < point_count {
            self.local_adjacency.resize_with(point_count, Vec::new);
        }
        self.local_adjacency[..point_count]
            .iter_mut()
            .for_each(Vec::clear);
        Ok(())
    }
}

/// Build one candidate list per data point from the nearest neighbors in all
/// leaves.
///
/// Each selected pair adds both directions. A list holds global IDs without
/// duplicates, in the order in which the leaf jobs add them. The lists stay
/// unsorted: final pruning ranks candidates by distance, and a graph search reads
/// all neighbors of a point, so the order only breaks distance ties.
pub(super) fn build_leaf_candidates(
    data: &dyn Dataset,
    kernels: &dyn Kernels,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
) -> ANNResult<Vec<AdjacencyList<u32>>> {
    let candidates: Vec<_> = (0..data.nrows())
        .map(|_| Mutex::new(AdjacencyList::new()))
        .collect();
    leaves
        .into_par_iter()
        .try_for_each_init(LeafBuffers::default, |buffers, point_ids| {
            add_leaf_candidates(data, kernels, &point_ids, requested_k, buffers, &candidates)
        })?;
    Ok(candidates
        .into_iter()
        .map(|list| list.into_inner().unwrap_or_else(PoisonError::into_inner))
        .collect())
}

/// Add the selected pairs of one leaf to the candidate lists.
///
/// The IDs of a leaf are distinct dataset rows, so `points x dimensions` is not
/// larger than the dataset.
fn add_leaf_candidates(
    data: &dyn Dataset,
    kernels: &dyn Kernels,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
    candidates: &[Mutex<AdjacencyList<u32>>],
) -> ANNResult<()> {
    let point_count = point_ids.len();
    // A point has at most `point_count - 1` other points in its leaf. Wider rows
    // would hold only empty slots.
    let leaf_k = requested_k.min(point_count.saturating_sub(1));
    if leaf_k == 0 {
        return Ok(());
    }

    buffers.prepare(point_count, data.dimensions(), leaf_k)?;
    data.gather(point_ids, buffers.points.as_view_mut())?;
    kernels.select_leaf_neighbors(
        buffers.points.as_view(),
        buffers.neighbors.as_view_mut(),
        &mut buffers.kernel_workspace,
    )?;

    add_symmetric_neighbors(
        point_ids,
        buffers.neighbors.as_view(),
        &mut buffers.local_adjacency[..point_count],
    );
    for (&point_id, additions) in point_ids.iter().zip(&buffers.local_adjacency) {
        candidates[point_id as usize]
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .extend_from_slice(additions);
    }
    Ok(())
}

/// Convert each selected pair from leaf positions to global IDs and add it to
/// the edge lists of both points.
fn add_symmetric_neighbors(
    point_ids: &[u32],
    neighbors: rowmajor::Ref<'_, Candidate>,
    local_adjacency: &mut [Vec<u32>],
) {
    for (source, source_neighbors) in neighbors.rows().enumerate() {
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
        dataset::Rows,
        kernels,
        test_support::{both_directions, edges},
    };
    use diskann_vector::distance::Metric;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::collections::BTreeSet;

    /// Build L2 candidates from one-dimensional points.
    fn build_1d(x: &[f32], leaves: Vec<Vec<u32>>, k: usize) -> Vec<Vec<u32>> {
        let dimensions = 1;
        let data = rowmajor::Ref::try_from_data(x, x.len(), dimensions).unwrap();
        let candidates = build_leaf_candidates(
            &Rows::new(data, Metric::L2).unwrap(),
            &*kernels::select(Metric::L2),
            leaves,
            k,
        )
        .unwrap();
        candidates.into_iter().map(Vec::from).collect()
    }

    /// Build the candidates of one leaf, given as `(point ID, value)` pairs in
    /// leaf order. Points outside the leaf are NaN, so a read of a wrong row
    /// drops edges and fails the test.
    fn build_one_leaf(leaf: &[(u32, f32)], k: usize) -> Vec<Vec<u32>> {
        let len = leaf
            .iter()
            .map(|&(id, _)| id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut x = vec![f32::NAN; len];
        for &(id, value) in leaf {
            x[id as usize] = value;
        }
        let ids = leaf.iter().map(|&(id, _)| id).collect();
        build_1d(&x, vec![ids], k)
    }

    #[test]
    fn selected_pairs_add_global_ids_in_both_directions() {
        // (point ID, one-dimensional vector) pairs, in leaf order.
        let leaf = [(7, 0.0), (2, 1.0), (5, 3.0), (0, 7.0)];
        // Nearest pairs: 7 picks 2 and 5; 2 picks 7 and 5;
        // 5 picks 2 and 7; 0 picks 5 and 2.
        let k = 2;
        let expected = both_directions([(2, 7), (5, 7), (2, 5), (0, 5), (0, 2)]);

        let actual = build_one_leaf(&leaf, k);

        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn should_build_leaf_candidates_successfully() {
        // The main path of leaf build: multi-dimensional rows, leaves of 1 to 64
        // points that overlap, k cut to the leaf size, and one thread that reuses
        // its buffers across leaf sizes.
        let point_count = 300;
        let dimensions = 16;
        let seed = 1290;
        let leaf_count = 40;
        let max_leaf_size = 64;
        let k = 2;
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = (0..point_count * dimensions)
            .map(|_| rng.random())
            .collect();
        let data =
            rowmajor::Ref::try_from_data(values.as_slice(), point_count, dimensions).unwrap();
        // The stride 11 is coprime to 300, so no leaf repeats a point.
        let leaves: Vec<Vec<u32>> = (0..leaf_count)
            .map(|j| {
                let size = 1 + (j * 13) % max_leaf_size;
                (0..size)
                    .map(|i| ((j * 7 + i * 11) % point_count) as u32)
                    .collect()
            })
            .collect();
        assert!(
            leaves.iter().any(|leaf| leaf.len() == 1),
            "fixture must contain a single-point leaf"
        );
        assert!(
            leaves.iter().any(|leaf| leaf.len() == 2),
            "fixture must contain a two-point leaf below k + 1"
        );

        let squared_l2 = |source: u32, target: u32| {
            data.row(source as usize)
                .iter()
                .zip(data.row(target as usize))
                .map(|(&a, &b)| {
                    let delta = f64::from(a) - f64::from(b);
                    delta * delta
                })
                .sum::<f64>()
        };
        let mut expected = BTreeSet::new();
        let mut selected_pairs = BTreeSet::new();
        let mut pair_selected_in_different_leaves = false;
        for leaf in &leaves {
            let mut leaf_pairs = BTreeSet::new();
            for &source in leaf {
                let mut nearest: Vec<_> = leaf
                    .iter()
                    .copied()
                    .filter(|&target| target != source)
                    .map(|target| (target, squared_l2(source, target)))
                    .collect();
                nearest.sort_unstable_by(|left, right| left.1.total_cmp(&right.1));
                for (target, _) in nearest.into_iter().take(k.min(leaf.len() - 1)) {
                    expected.extend([(source, target), (target, source)]);
                    leaf_pairs.insert((source.min(target), source.max(target)));
                }
            }
            // Count a pair once per leaf; selection from both ends is not overlap.
            for pair in leaf_pairs {
                pair_selected_in_different_leaves |= !selected_pairs.insert(pair);
            }
        }
        assert!(
            pair_selected_in_different_leaves,
            "the oracle must select a pair in two different leaves"
        );

        let actual = build_leaf_candidates(
            &Rows::new(data, Metric::L2).unwrap(),
            &*kernels::select(Metric::L2),
            leaves,
            k,
        )
        .unwrap();

        let actual: Vec<Vec<u32>> = actual.into_iter().map(Vec::from).collect();
        assert_eq!(edges(&actual), expected, "seed {seed}");
    }

    #[test]
    fn a_pair_found_in_several_leaves_is_added_only_once() {
        // x[i] = i² is the one-dimensional vector of point i, so each point's
        // nearest point is an index neighbor.
        let x = [0.0, 1.0, 4.0, 9.0, 16.0];
        // The leaves overlap, and the last one repeats the first.
        let leaves = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4], vec![0, 1, 2]];
        let k = 1;
        // Three leaves find pair 1-2, and two leaves each find 0-1 and 2-3.
        let expected = both_directions([(0, 1), (1, 2), (2, 3), (3, 4)]);

        let actual = build_1d(&x, leaves, k);

        // `edges` panics on a repeated edge.
        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn k_above_the_leaf_size_selects_every_other_point() {
        // (point ID, one-dimensional vector) pairs, in leaf order.
        let leaf = [(1, 0.0), (3, 2.0), (5, 9.0)];
        let k = leaf.len() + 1;
        let expected = both_directions([(1, 3), (1, 5), (3, 5)]);

        let actual = build_one_leaf(&leaf, k);

        assert_eq!(edges(&actual), expected);
    }

    #[test]
    fn a_single_point_leaf_adds_no_candidates() {
        // (point ID, one-dimensional vector) pairs, in leaf order.
        let leaf = [(0, 3.0)];
        // No other point exists, so the leaf returns without adding candidates.
        let k = 2;

        let actual = build_one_leaf(&leaf, k);

        assert_eq!(actual, [Vec::<u32>::new()]);
    }

    #[test]
    fn unrankable_pairs_add_no_candidates() {
        // x[i] is the one-dimensional vector of point i.
        let x = [0.0, 3.0, f32::NAN];
        let leaf = vec![0, 1, 2];
        let k = leaf.len() - 1;
        // Point 2 is NaN, so its pairs are not rankable.
        let expected = both_directions([(0, 1)]);

        let actual = build_1d(&x, vec![leaf], k);

        assert_eq!(edges(&actual), expected);
    }

    /// Run `add_leaf_candidates` for one leaf of one-dimensional points `x` with
    /// the given buffers, and return the candidate list of each point.
    fn add_one_leaf(x: &[f32], leaf: &[u32], k: usize, buffers: &mut LeafBuffers) -> Vec<Vec<u32>> {
        let dimensions = 1;
        let data = rowmajor::Ref::try_from_data(x, x.len(), dimensions).unwrap();
        let candidates: Vec<_> = (0..x.len())
            .map(|_| Mutex::new(AdjacencyList::new()))
            .collect();

        add_leaf_candidates(
            &Rows::new(data, Metric::L2).unwrap(),
            &*kernels::select(Metric::L2),
            leaf,
            k,
            buffers,
            &candidates,
        )
        .unwrap();

        candidates
            .into_iter()
            .map(|row| Vec::from(row.into_inner().unwrap_or_else(PoisonError::into_inner)))
            .collect()
    }

    #[test]
    fn reused_buffers_do_not_carry_edges_between_leaves() {
        // x[i] = i² is the one-dimensional vector of point i, so its nearest
        // neighbors are its index neighbors.
        let x = [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
        let mut buffers = LeafBuffers::default();

        // One thread runs a leaf, a smaller leaf with k above its size, then a
        // third leaf with a larger k. Each call must see only its own points.
        for (leaf, k, expected) in [
            // k = 1 selects one of the three other points of this leaf.
            (
                vec![0, 1, 2, 3],
                1,
                both_directions([(0, 1), (1, 2), (2, 3)]),
            ),
            // k = 2 is above the one other point of this leaf.
            (vec![4, 6], 2, both_directions([(4, 6)])),
            // k = 2 selects both other points of this leaf.
            (vec![1, 3, 5], 2, both_directions([(1, 3), (1, 5), (3, 5)])),
        ] {
            let actual = add_one_leaf(&x, &leaf, k, &mut buffers);

            assert_eq!(edges(&actual), expected, "leaf {leaf:?}, k={k}");
        }
    }
}
