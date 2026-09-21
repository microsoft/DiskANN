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

    super::conversion::gather_as_f32(data, point_ids, point_values).map_err(
        |(point, source)| LeafBuildError::Conversion {
            leaf,
            point,
            source: source.into(),
        },
    )?;

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

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::graph::pipnn::{L2, test_support};
    use diskann_wide::ARCH;
    use rstest::rstest;

    #[test]
    fn selected_neighbors_use_global_ids_and_contribute_both_edge_directions() {
        // The leaf lists IDs [5, 1, 3], at coordinates [9, 0, 2].
        // The directed choices are 1 -> 3, 3 -> 1 and 5 -> 3.
        let values = [100.0_f32, 0.0, -100.0, 2.0, 200.0, 9.0];
        let data = MatrixView::try_from(&values[..], 6, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, vec![vec![5, 1, 3]], 1))
            .unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            [vec![], vec![3], vec![], vec![1, 5], vec![], vec![3]]
        );
    }

    #[rstest]
    #[case::one_worker(1)]
    #[case::several_workers(3)]
    fn overlapping_leaves_merge_each_neighbor_once(#[case] workers: usize) {
        let values = [0.0_f32, 1.0, 4.0, 9.0, 16.0];
        let data = MatrixView::try_from(&values[..], 5, 1).unwrap();
        let leaves = vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4], vec![0, 1, 2]];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();

        let actual = pool
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, 1))
            .unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            [vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]]
        );
    }

    #[rstest]
    #[case::no_leaves(vec![], 2)]
    #[case::empty_leaf(vec![vec![]], 2)]
    #[case::singleton(vec![vec![2]], 2)]
    #[case::zero_neighbors(vec![vec![0, 1, 2]], 0)]
    fn leaves_without_selected_pairs_produce_empty_adjacency(
        #[case] leaves: Vec<Vec<u32>>,
        #[case] requested_k: usize,
    ) {
        let values = [0.0_f32, 1.0, 4.0];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, leaves, requested_k))
            .unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [Vec::<u32>::new(), vec![], vec![]]
        );
    }

    #[test]
    fn requesting_more_neighbors_than_a_leaf_has_selects_all_other_points() {
        let values = [100.0_f32, 0.0, -100.0, 2.0, 200.0, 9.0];
        let data = MatrixView::try_from(&values[..], 6, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, vec![vec![1, 3, 5]], 99))
            .unwrap();
        let actual: Vec<_> = actual.into_iter().map(Vec::from).collect();

        assert_eq!(
            test_support::sorted_members_per_row(&actual),
            [vec![], vec![3, 5], vec![], vec![1, 5], vec![], vec![1, 3]]
        );
    }

    #[test]
    fn unrankable_pairs_do_not_add_unassigned_ids_to_the_graph() {
        let values = [0.0_f32, 3.0, f32::NAN];
        let data = MatrixView::try_from(&values[..], 3, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let actual = pool
            .install(|| build_leaf_candidates::<_, L2, _>(ARCH, data, vec![vec![0, 1, 2]], 2))
            .unwrap();

        assert_eq!(
            actual.into_iter().map(Vec::from).collect::<Vec<_>>(),
            [vec![1], vec![0], vec![]]
        );
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
                0,
                &ids,
                requested_k,
                &mut buffers,
                &candidates,
            )
            .unwrap();

            let actual: Vec<_> = candidates
                .into_iter()
                .map(|row| {
                    let mut ids = Vec::from(row.into_inner());
                    ids.sort_unstable();
                    ids
                })
                .collect();
            assert_eq!(actual, expected, "leaf {ids:?}, k={requested_k}");
        }
    }

    #[rstest]
    #[case::point_values(2, 1)]
    #[case::neighbor_slots(1, 2)]
    fn an_overflowing_leaf_shape_is_rejected_before_buffers_change(
        #[case] dimensions: usize,
        #[case] requested_k: usize,
    ) {
        let old_neighbor = Candidate::new(1, 7.0);
        let mut buffers = LeafBuffers {
            point_values: vec![3.0, 4.0],
            neighbors: vec![old_neighbor],
            ..LeafBuffers::default()
        };

        let error = buffers
            .prepare(7, usize::MAX, dimensions, requested_k)
            .unwrap_err();

        assert!(matches!(
            error,
            LeafBuildError::ShapeOverflow {
                leaf: 7,
                rows: usize::MAX,
                columns: 2,
            }
        ));
        assert_eq!(buffers.point_values, [3.0, 4.0]);
        assert_eq!(buffers.neighbors, [old_neighbor]);
    }

    #[test]
    fn a_kernel_error_retains_the_failed_leaf_and_original_cause() {
        #[derive(Debug, thiserror::Error)]
        #[error("distance calculation unavailable")]
        struct DistanceFailure;

        // This stub supplies an otherwise hard-to-trigger dependency failure.
        // Gathering, leaf indexing and error wrapping remain real.
        struct UnavailableMetric;
        impl LeafMetric for UnavailableMetric {
            fn compute_distances(_: MatrixView<'_, f32>, _: &mut [f32]) -> crate::ANNResult<()> {
                Err(crate::ANNError::new(DistanceFailure))
            }
        }

        let values = [0.0_f32, 1.0, 4.0, 9.0];
        let data = MatrixView::try_from(&values[..], 4, 1).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let error = pool
            .install(|| {
                build_leaf_candidates::<_, UnavailableMetric, _>(
                    ARCH,
                    data,
                    vec![vec![], vec![1, 3]],
                    1,
                )
            })
            .unwrap_err();

        let LeafBuildError::Kernel { leaf, source } = error else {
            panic!("expected a leaf kernel error, got {error:?}");
        };
        assert_eq!(leaf, 1);
        assert!(source.downcast_ref::<DistanceFailure>().is_some());
    }
}
