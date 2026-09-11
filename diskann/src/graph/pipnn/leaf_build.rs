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

use crate::{graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    leaf_kernel::{LeafKernelWorkspace, leaf_neighbor_count, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    simd::PiPNNSIMDSchema,
    topk::Candidate,
};

/// Failure while converting leaves into direct graph candidates.
#[derive(Debug, thiserror::Error)]
pub(crate) enum LeafBuildError {
    #[error("leaf {leaf} shape {rows} x {columns} overflows usize")]
    ShapeOverflow {
        leaf: usize,
        rows: usize,
        columns: usize,
    },
    #[error("failed to convert point {point} in leaf {leaf}")]
    Conversion {
        leaf: usize,
        point: u32,
        #[source]
        source: crate::ANNError,
    },
    #[error("nearest-neighbor selection failed for leaf {leaf}")]
    Kernel {
        leaf: usize,
        #[source]
        source: crate::ANNError,
    },
}

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
    fn prepare(
        &mut self,
        leaf: usize,
        point_count: usize,
        dimension_count: usize,
        requested_k: usize,
    ) -> Result<(usize, usize), LeafBuildError> {
        let point_value_count =
            point_count
                .checked_mul(dimension_count)
                .ok_or(LeafBuildError::ShapeOverflow {
                    leaf,
                    rows: point_count,
                    columns: dimension_count,
                })?;
        let leaf_k = leaf_neighbor_count(point_count, requested_k);
        let neighbor_count =
            point_count
                .checked_mul(leaf_k)
                .ok_or(LeafBuildError::ShapeOverflow {
                    leaf,
                    rows: point_count,
                    columns: leaf_k,
                })?;

        grow(&mut self.point_values, point_value_count, 0.0);
        grow(&mut self.neighbors, neighbor_count, Candidate::default());
        Ok((leaf_k, neighbor_count))
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
#[allow(clippy::disallowed_methods)] // The supplied pool owns this terminal operation.
pub(super) fn build_leaf_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
) -> Result<Vec<AdjacencyList<u32>>, LeafBuildError>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
    T: VectorRepr,
{
    let candidates: Vec<_> = (0..data.nrows())
        .map(|_| Mutex::new(AdjacencyList::new()))
        .collect();
    leaves.par_iter().enumerate().try_for_each_init(
        LeafBuffers::default,
        |buffers, (leaf, point_ids)| {
            add_direct_leaf_candidates::<A, M, T>(
                arch,
                data,
                leaf,
                point_ids,
                requested_k,
                buffers,
                &candidates,
            )
        },
    )?;
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
/// leaf shape.
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::expect_used,
    reason = "buffer prefixes have the checked leaf shape"
)]
fn add_direct_leaf_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
    candidates: &[Mutex<AdjacencyList<u32>>],
) -> Result<(), LeafBuildError>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
    T: VectorRepr,
{
    let (leaf_k, neighbor_value_count) =
        buffers.prepare(leaf, point_ids.len(), data.ncols(), requested_k)?;
    if leaf_k == 0 {
        return Ok(());
    }

    let point_value_count = point_ids.len() * data.ncols();
    let point_values = &mut buffers.point_values[..point_value_count];

    for (&point, point_output) in point_ids
        .iter()
        .zip(point_values.chunks_exact_mut(data.ncols()))
    {
        let source_values = data.row(point as usize);
        super::conversion::as_f32_into(source_values, point_output).map_err(|source| {
            LeafBuildError::Conversion {
                leaf,
                point,
                source: source.into(),
            }
        })?;
    }

    let points = MatrixView::try_from(&*point_values, point_ids.len(), data.ncols())
        .expect("point buffer prefix has the checked leaf shape");
    let output = MutMatrixView::try_from(
        &mut buffers.neighbors[..neighbor_value_count],
        point_ids.len(),
        leaf_k,
    )
    .expect("neighbor buffer prefix has the checked leaf shape");
    select_leaf_neighbors::<A, M>(arch, points, output, &mut buffers.kernel_workspace)
        .map_err(|source| LeafBuildError::Kernel { leaf, source })?;

    buffers.prepare_local_adjacency(point_ids.len());
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..neighbor_value_count],
        &mut buffers.local_adjacency[..point_ids.len()],
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

#[cfg(test)]
mod build_leaf_candidates_tests {
    use diskann_wide::arch::Scalar;
    use half::f16;
    use rstest::rstest;

    use super::*;
    use crate::graph::pipnn::{InnerProduct, L2};

    fn build_candidates<T: VectorRepr, M: LeafMetric>(
        data: MatrixView<'_, T>,
        leaves: Vec<Vec<u32>>,
        k: usize,
        threads: usize,
    ) -> Vec<Vec<u32>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| build_leaf_candidates::<_, M, _>(Scalar, data, leaves, k))
            .unwrap()
            .into_iter()
            .map(Vec::from)
            .collect()
    }

    #[rstest]
    #[case::f32([5.5_f32, 100.0, 1.25, 50.0, 0.25])]
    #[case::f16([5.5, 100.0, 1.25, 50.0, 0.25].map(f16::from_f32))]
    #[case::u8([5_u8, 100, 1, 50, 0])]
    #[case::i8([-6_i8, 100, -10, 50, -11])]
    fn selected_local_positions_become_global_ids<T: VectorRepr>(#[case] values: [T; 5]) {
        // Given: only IDs 0, 2, 4 belong to the leaf. Their coordinates decrease,
        // so 0 selects 2, while 2 and 4 select each other. Reverse edges add 2 -> 0.
        let data = MatrixView::column_vector(&values[..]);
        let leaves = vec![vec![0, 2, 4]];
        let expected = [vec![2], vec![], vec![0, 4], vec![], vec![2]];

        // When
        let actual = build_candidates::<_, L2>(data, leaves, 1, 1);

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn unassigned_kernel_results_add_no_edges() {
        // Given: the NaN point has no rankable distances. The two finite points
        // select each other, leaving their second output slot unassigned.
        let values = [0.0_f32, 1.0, f32::NAN];
        let data = MatrixView::column_vector(&values[..]);
        let expected = [vec![1], vec![0], vec![]];

        // When
        let actual = build_candidates::<_, InnerProduct>(data, vec![vec![0, 1, 2]], 2, 1);

        // Then
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::four_workers(4)]
    fn overlapping_leaves_merge_into_sorted_unique_neighbors(#[case] threads: usize) {
        // Given: k = 2 connects every pair in each three-point leaf. The union
        // contains all pairs except 1 <-> 3, regardless of repeated leaf jobs.
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::column_vector(&values[..]);
        let leaves = [vec![0, 1, 2], vec![0, 2, 3], vec![0, 1, 2]]
            .into_iter()
            .cycle()
            .take(48)
            .collect();
        let expected = [vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];

        // When
        let actual = build_candidates::<_, L2>(data, leaves, 2, threads);

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn reverse_edges_can_give_a_point_more_than_twice_k_neighbors() {
        // Given: each outer point is distance 1 from the origin and at least
        // sqrt(2) from another outer point. Their three reverse edges exceed 2k.
        let points = [[0.0_f32, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]];
        let data = MatrixView::try_from(points.as_flattened(), 4, 2).unwrap();
        let expected = [vec![1, 2, 3], vec![0], vec![0], vec![0]];

        // When
        let actual = build_candidates::<_, L2>(data, vec![vec![0, 1, 2, 3]], 1, 1);

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn singleton_leaves_add_no_edges() {
        // Given
        let values = [0.0_f32, 1.0, 2.0];
        let data = MatrixView::column_vector(&values[..]);
        let leaves = vec![vec![0], vec![1], vec![2]];
        let expected: [Vec<u32>; 3] = [vec![], vec![], vec![]];

        // When
        let actual = build_candidates::<_, L2>(data, leaves, 1, 1);

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn reused_leaf_buffers_do_not_add_edges_from_previous_leaves() {
        // Given: use the same buffers for four points, two points, then three.
        // The final leaf excludes ID 1 and connects all pairs among 0, 2, 3.
        let values = [0.0_f32, 1.0, 2.0, 3.0];
        let data = MatrixView::column_vector(&values[..]);
        let mut buffers = LeafBuffers::default();
        let mut build_leaf = |point_ids: &[u32]| {
            let candidates: Vec<_> = (0..data.nrows())
                .map(|_| Mutex::new(AdjacencyList::new()))
                .collect();
            add_direct_leaf_candidates::<_, L2, _>(
                Scalar,
                data,
                0,
                point_ids,
                2,
                &mut buffers,
                &candidates,
            )
            .unwrap();
            candidates
                .into_iter()
                .map(|list| {
                    let mut ids = Vec::from(list.into_inner());
                    ids.sort_unstable();
                    ids
                })
                .collect::<Vec<_>>()
        };
        build_leaf(&[0, 1, 2, 3]);
        build_leaf(&[1, 3]);
        let expected = [vec![2, 3], vec![], vec![0, 3], vec![0, 2]];

        // When
        let actual = build_leaf(&[0, 2, 3]);

        // Then
        assert_eq!(actual, expected);
    }

    #[test]
    fn overflowing_leaf_shape_is_rejected() {
        // Given
        let mut buffers = LeafBuffers::default();

        // When
        let result = buffers.prepare(7, usize::MAX, 2, 1);

        // Then
        assert!(matches!(
            result,
            Err(LeafBuildError::ShapeOverflow { leaf: 7, .. })
        ));
    }
}
