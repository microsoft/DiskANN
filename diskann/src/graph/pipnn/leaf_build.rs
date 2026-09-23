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
//! 4. Add both edge directions to direct candidates or HashPrune reservoirs.
//!
//! Overlapping leaves run concurrently. The direct path locks one destination
//! list while it adds IDs. The HashPrune path locks one source reservoir while it
//! adds weighted edges. Reusable buffers keep their largest allocation. Each
//! operation uses an explicit active prefix.

use parking_lot::Mutex;

use crate::{graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    leaf_kernel::{LeafKernelWorkspace, select_leaf_neighbors},
    leaf_metric::LeafMetric,
    simd::Simd,
    topk::Candidate,
};

/// Failure while converting leaves into graph candidates.
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
    #[error("leaf {leaf} produced too many directed edges")]
    TooManyEdges { leaf: usize },
}

/// Reusable buffers for one Rayon leaf job.
///
/// The buffers keep the largest leaf shape that this job observed. The direct
/// path uses `local_adjacency`. The HashPrune path uses the CSR and sketch
/// buffers.
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
        point_count
            .checked_mul(point_count)
            .ok_or(LeafBuildError::ShapeOverflow {
                leaf,
                rows: point_count,
                columns: point_count,
            })?;
        // A point has at most `point_count - 1` other points in its leaf. Wider rows
        // would hold only empty slots.
        let leaf_k = requested_k.min(point_count.saturating_sub(1));
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

    fn prepare_seen_pairs(&mut self, point_count: usize) {
        // `prepare` checked this product for the same leaf shape.
        grow(&mut self.seen_pairs, point_count * point_count, false);
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
    A: Simd,
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

/// Add weighted symmetric leaf edges to HashPrune reservoirs.
#[allow(clippy::disallowed_methods)] // The supplied pool owns this terminal operation.
pub(super) fn add_hash_prune_candidates<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
    hash_prune: &super::hash_prune::HashPrune,
) -> Result<(), LeafBuildError>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    leaves.par_iter().enumerate().try_for_each_init(
        LeafBuffers::default,
        |buffers, (leaf, point_ids)| {
            let leaf_k = gather_leaf_neighbors::<A, M, T>(
                arch,
                data,
                leaf,
                point_ids,
                requested_k,
                buffers,
            )?;
            let point_count = point_ids.len();
            buffers.prepare_seen_pairs(point_count);
            let edge_count = build_symmetric_edge_csr(
                leaf,
                point_ids,
                leaf_k,
                &buffers.neighbors[..point_count * leaf_k],
                EdgeBuffers {
                    seen: &mut buffers.seen_pairs[..point_count * point_count],
                    offsets: &mut buffers.edge_offsets,
                    edges: &mut buffers.edges,
                    cursor: &mut buffers.edge_cursor,
                },
            )?;
            hash_prune.add_leaf_edges(
                point_ids,
                &buffers.edge_offsets[..point_count + 1],
                &buffers.edges[..edge_count],
                &mut buffers.sketch_scratch,
            );
            Ok(())
        },
    )
}

/// Add one leaf's symmetric neighbors to the direct candidate lists.
///
/// Reusable buffers can be longer than this leaf, so all accesses use the current
/// leaf shape.
#[allow(clippy::too_many_arguments)]
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
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    let leaf_k =
        gather_leaf_neighbors::<A, M, T>(arch, data, leaf, point_ids, requested_k, buffers)?;
    if leaf_k == 0 {
        return Ok(());
    }
    buffers.prepare_local_adjacency(point_ids.len());
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..point_ids.len() * leaf_k],
        &mut buffers.local_adjacency[..point_ids.len()],
    );
    for (&point_id, additions) in point_ids.iter().zip(&buffers.local_adjacency) {
        candidates[point_id as usize]
            .lock()
            .extend_from_slice(additions);
    }
    Ok(())
}

/// Select local nearest neighbors for one leaf.
///
/// The function gathers leaf IDs into a packed `f32` matrix. The leaf kernel
/// owns Gram construction, norm preparation, and local ranking. This function
/// returns the effective neighbor count for graph-edge mapping.
#[expect(
    clippy::expect_used,
    reason = "buffer prefixes have the checked leaf shape"
)]
fn gather_leaf_neighbors<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
) -> Result<usize, LeafBuildError>
where
    A: Simd,
    M: LeafMetric,
    T: VectorRepr,
{
    let (leaf_k, neighbor_value_count) =
        buffers.prepare(leaf, point_ids.len(), data.ncols(), requested_k)?;
    if leaf_k == 0 {
        return Ok(0);
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
    Ok(leaf_k)
}

/// Add symmetric dataset IDs from one leaf-kernel result.
///
/// The leaf kernel returns only leaf-local positions in `point_ids`.
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

struct EdgeBuffers<'a> {
    seen: &'a mut [bool],
    offsets: &'a mut Vec<u32>,
    edges: &'a mut Vec<(u32, f32)>,
    cursor: &'a mut Vec<u32>,
}

/// Create directed leaf edges for HashPrune ingestion.
///
/// Each selected neighbor pair contributes both directions. Duplicate directions
/// appear once. Each target is a position in `point_ids`.
/// Build weighted CSR edges from one leaf-kernel result.
///
/// The leaf kernel returns only leaf-local positions in `point_ids`.
fn build_symmetric_edge_csr(
    leaf: usize,
    point_ids: &[u32],
    leaf_k: usize,
    neighbors: &[Candidate],
    buffers: EdgeBuffers<'_>,
) -> Result<usize, LeafBuildError> {
    let EdgeBuffers {
        seen,
        offsets,
        edges,
        cursor,
    } = buffers;
    let point_count = point_ids.len();
    grow(offsets, point_count + 1, 0);
    offsets[..point_count + 1].fill(0);
    if leaf_k == 0 {
        return Ok(0);
    }

    // The prior successful write pass left the active `seen` area clear. This
    // count pass marks each unique directed edge.
    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors {
            if !neighbor.is_assigned() {
                continue;
            }
            let target = neighbor.local_idx as usize;
            count_directed_edge(leaf, point_count, source, target, seen, offsets)?;
            count_directed_edge(leaf, point_count, target, source, seen, offsets)?;
        }
    }
    for point in 1..=point_count {
        offsets[point] = offsets[point]
            .checked_add(offsets[point - 1])
            .ok_or(LeafBuildError::TooManyEdges { leaf })?;
    }

    let edge_count = offsets[point_count] as usize;
    grow(edges, edge_count, (0, 0.0));
    grow(cursor, point_count, 0);
    cursor[..point_count].copy_from_slice(&offsets[..point_count]);
    let edges = &mut edges[..edge_count];
    let cursor = &mut cursor[..point_count];

    // This pass visits the same directions as the count pass. The first
    // occurrence writes its edge and clears its mark for the next leaf.
    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors {
            if !neighbor.is_assigned() {
                continue;
            }
            let target = neighbor.local_idx as usize;
            write_counted_directed_edge(
                point_count,
                source,
                target,
                neighbor.distance,
                seen,
                edges,
                cursor,
            );
            write_counted_directed_edge(
                point_count,
                target,
                source,
                neighbor.distance,
                seen,
                edges,
                cursor,
            );
        }
    }
    Ok(edge_count)
}

fn count_directed_edge(
    leaf: usize,
    point_count: usize,
    source: usize,
    target: usize,
    seen: &mut [bool],
    offsets: &mut [u32],
) -> Result<(), LeafBuildError> {
    let seen_entry = &mut seen[source * point_count + target];
    if !*seen_entry {
        *seen_entry = true;
        offsets[source + 1] = offsets[source + 1]
            .checked_add(1)
            .ok_or(LeafBuildError::TooManyEdges { leaf })?;
    }
    Ok(())
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
        let edge_slot = cursor[source] as usize;
        edges[edge_slot] = (target as u32, distance);
        cursor[source] += 1;
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
    #[case::point_values(2, 1, 2)]
    #[case::pair_matrix(1, 2, usize::MAX)]
    fn an_overflowing_leaf_shape_is_rejected_before_buffers_change(
        #[case] dimensions: usize,
        #[case] requested_k: usize,
        #[case] expected_columns: usize,
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
                columns,
            } if columns == expected_columns
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
            fn compute_distances(
                _: MatrixView<'_, f32>,
                _: MutMatrixView<'_, f32>,
            ) -> crate::ANNResult<()> {
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

#[cfg(test)]
mod build_symmetric_edge_csr_tests {
    use super::*;

    #[test]
    fn symmetric_edge_csr_contains_both_directions_in_source_order() {
        // Given
        let point_ids = [10, 20, 30];
        let neighbors = [
            Candidate::new(1, 1.0),
            Candidate::new(2, 2.0),
            Candidate::new(1, 1.5),
        ];
        let expected_edge_count = 4;
        let expected_offsets = [0, 1, 3, 4];
        let expected_edges = [(1, 1.0), (0, 1.0), (2, 2.0), (1, 2.0)];
        let mut seen = vec![false; 9];
        let mut offsets = Vec::new();
        let mut edges = Vec::new();
        let mut cursor = Vec::new();

        // When
        let actual_edge_count = build_symmetric_edge_csr(
            0,
            &point_ids,
            1,
            &neighbors,
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap();

        // Then
        assert_eq!(actual_edge_count, expected_edge_count);
        assert_eq!(offsets, expected_offsets);
        assert_eq!(edges, expected_edges);
    }

    #[test]
    fn symmetric_edge_csr_omits_unassigned_neighbors() {
        let point_ids = [10, 20];
        let neighbors = [Candidate::new(1, 1.0), Candidate::default()];
        let mut seen = vec![false; 4];
        let mut offsets = Vec::new();
        let mut edges = Vec::new();
        let mut cursor = Vec::new();

        let count = build_symmetric_edge_csr(
            0,
            &point_ids,
            1,
            &neighbors,
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap();

        assert_eq!(count, 2);
        assert_eq!(offsets, [0, 1, 2]);
        assert_eq!(edges, [(1, 1.0), (0, 1.0)]);
    }

    #[test]
    fn symmetric_edge_csr_deduplicates_edges_seen_from_both_endpoints() {
        let point_ids = [10, 20];
        let neighbors = [Candidate::new(1, 1.0), Candidate::new(0, 1.0)];
        let mut seen = vec![false; 4];
        let mut offsets = Vec::new();
        let mut edges = Vec::new();
        let mut cursor = Vec::new();

        let count = build_symmetric_edge_csr(
            0,
            &point_ids,
            1,
            &neighbors,
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap();

        assert_eq!(count, 2);
        assert_eq!(offsets, [0, 1, 2]);
        assert_eq!(edges, [(1, 1.0), (0, 1.0)]);
        assert!(seen.iter().all(|&entry| !entry));
    }

    #[test]
    fn singleton_leaf_produces_empty_edge_csr() {
        // Given
        let leaf = 0;
        let point_ids = [10];
        let effective_neighbor_count = 0;
        let no_neighbors = [];
        let stale_edge = (99, 99.0);
        let expected_edge_count = 0;
        let expected_offsets = [0, 0];
        let expected_edges = [stale_edge];
        let expected_seen = [false];
        let mut seen = vec![false; 1];
        let mut offsets = Vec::new();
        let mut edges = vec![stale_edge];
        let mut cursor = Vec::new();

        // When
        let actual_edge_count = build_symmetric_edge_csr(
            leaf,
            &point_ids,
            effective_neighbor_count,
            &no_neighbors,
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap();

        // Then
        assert_eq!(actual_edge_count, expected_edge_count);
        assert_eq!(offsets, expected_offsets);
        assert_eq!(edges, expected_edges);
        assert_eq!(seen, expected_seen);
    }
}
