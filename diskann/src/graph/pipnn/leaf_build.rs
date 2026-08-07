/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local graph construction and candidate accumulation.
//!
//! Partitioning supplies sorted, unique global point IDs for each leaf. One leaf
//! job does these steps:
//!
//! 1. Check each ID and convert its vector to reusable `f32` storage.
//! 2. Compute the lower triangle of `A · Aᵀ`.
//! 3. Select local neighbors for both points of each pair.
//! 4. Convert local positions to global point IDs.
//! 5. Add both edge directions to direct candidates or HashPrune reservoirs.
//!
//! Overlapping leaves run concurrently. The direct path locks one destination
//! list while it adds IDs. The HashPrune path locks one source reservoir while it
//! adds weighted edges. Reusable buffers keep their largest allocation. Each
//! operation uses an explicit active prefix.

use std::{collections::TryReserveError, sync::Mutex};

use crate::{graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use rayon::prelude::*;

use super::{
    kernel_metric::{LeafMetric, NormPreparation},
    leaf_kernel::{
        LeafKernelError, LeafKernelWorkspace, LeafNeighbor, leaf_neighbor_count, nearest_neighbors,
    },
    simd::PiPNNSIMDSchema,
};

/// Failure while converting leaves into graph candidates.
#[derive(Debug, thiserror::Error)]
pub(crate) enum LeafBuildError {
    #[error("leaf build requires at least one dimension")]
    EmptyDimensions,
    #[error("dataset point count {0} exceeds the u32 ID limit")]
    TooManyPoints(usize),
    #[error("leaf {leaf} is empty")]
    EmptyLeaf { leaf: usize },
    #[error("point ID {point} in leaf {leaf} is outside a {points}-point dataset")]
    InvalidPointId {
        leaf: usize,
        point: u32,
        points: usize,
    },
    #[error("point ID {point} appears more than once in leaf {leaf}")]
    DuplicatePointId { leaf: usize, point: u32 },
    #[error("point IDs in leaf {leaf} are not strictly increasing")]
    UnsortedPointIds { leaf: usize },
    #[error("leaf {leaf} shape {rows} x {columns} overflows usize")]
    ShapeOverflow {
        leaf: usize,
        rows: usize,
        columns: usize,
    },
    #[error("failed to form {buffer} view for leaf {leaf}")]
    InvalidView { leaf: usize, buffer: &'static str },
    #[error("failed to reserve {additional} values for {buffer}")]
    Allocation {
        buffer: &'static str,
        additional: usize,
        #[source]
        source: TryReserveError,
    },
    #[error("failed to convert point {point} in leaf {leaf}")]
    Conversion {
        leaf: usize,
        point: u32,
        #[source]
        source: crate::ANNError,
    },
    #[error("lower-AAT failed for leaf {leaf}")]
    LowerAat {
        leaf: usize,
        #[source]
        source: diskann_linalg::SgemmError,
    },
    #[error("nearest-neighbor selection failed for leaf {leaf}")]
    Kernel {
        leaf: usize,
        #[source]
        source: LeafKernelError,
    },
    #[error("leaf kernel returned local target {target} for a {points}-point leaf")]
    InvalidLocalTarget { target: u32, points: usize },
    #[error("candidate list for point {point} is poisoned")]
    PoisonedCandidateList { point: u32 },
    #[error("leaf {leaf} produced too many directed edges")]
    TooManyEdges { leaf: usize },
    #[error("HashPrune rejected the edge data for leaf {leaf}")]
    HashPrune {
        leaf: usize,
        #[source]
        source: crate::ANNError,
    },
}

/// Reusable buffers for one Rayon leaf job.
///
/// The buffers keep the largest leaf shape that this job observed. The direct
/// path uses `local_adjacency`. The HashPrune path uses the CSR and sketch
/// buffers.
#[derive(Default)]
struct LeafBuffers {
    point_values: Vec<f32>,
    dots: Vec<f32>,
    norms: Vec<f32>,
    neighbors: Vec<LeafNeighbor>,
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
        let dot_count =
            point_count
                .checked_mul(point_count)
                .ok_or(LeafBuildError::ShapeOverflow {
                    leaf,
                    rows: point_count,
                    columns: point_count,
                })?;
        let leaf_k = leaf_neighbor_count(point_count, requested_k)
            .map_err(|source| LeafBuildError::Kernel { leaf, source })?;
        let neighbor_count =
            point_count
                .checked_mul(leaf_k)
                .ok_or(LeafBuildError::ShapeOverflow {
                    leaf,
                    rows: point_count,
                    columns: leaf_k,
                })?;

        grow(
            "leaf point values",
            &mut self.point_values,
            point_value_count,
            0.0,
        )?;
        grow("leaf dot products", &mut self.dots, dot_count, 0.0)?;
        grow(
            "leaf neighbors",
            &mut self.neighbors,
            neighbor_count,
            LeafNeighbor::default(),
        )?;
        grow("leaf seen pairs", &mut self.seen_pairs, dot_count, false)?;
        Ok((leaf_k, neighbor_count))
    }

    fn prepare_local_adjacency(&mut self, point_count: usize) -> Result<(), LeafBuildError> {
        let additional = point_count.saturating_sub(self.local_adjacency.len());
        self.local_adjacency
            .try_reserve(additional)
            .map_err(|source| allocation_error("leaf adjacency lists", additional, source))?;
        self.local_adjacency.resize_with(point_count, Vec::new);
        self.local_adjacency[..point_count]
            .iter_mut()
            .for_each(Vec::clear);
        Ok(())
    }
}

/// Concurrent candidate lists indexed by global point ID.
///
/// A point can occur in several overlapping leaves. A worker locks one point's
/// list and adds all IDs from one leaf. `AdjacencyList` removes duplicates during
/// this append. The function sorts each list after all leaf jobs finish.
struct DirectCandidates {
    lists: Vec<Mutex<AdjacencyList<u32>>>,
}

impl DirectCandidates {
    fn new(point_count: usize) -> Result<Self, LeafBuildError> {
        let mut lists = Vec::new();
        lists
            .try_reserve_exact(point_count)
            .map_err(|source| allocation_error("candidate lists", point_count, source))?;
        lists.resize_with(point_count, || Mutex::new(AdjacencyList::new()));
        Ok(Self { lists })
    }

    fn add_leaf(
        &self,
        point_ids: &[u32],
        local_adjacency: &[Vec<u32>],
    ) -> Result<(), LeafBuildError> {
        for (&source, additions) in point_ids.iter().zip(local_adjacency) {
            // `add_direct_leaf_candidates` checks every point ID before this append.
            let candidates = &self.lists[source as usize];
            let mut candidates = candidates
                .lock()
                .map_err(|_| LeafBuildError::PoisonedCandidateList { point: source })?;
            candidates.extend_from_slice(additions);
        }
        Ok(())
    }

    fn into_lists(self) -> Result<Vec<AdjacencyList<u32>>, LeafBuildError> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.lists.len())
            .map_err(|source| allocation_error("candidate output", self.lists.len(), source))?;
        for (point, candidates) in self.lists.into_iter().enumerate() {
            let mut candidates =
                candidates
                    .into_inner()
                    .map_err(|_| LeafBuildError::PoisonedCandidateList {
                        point: point as u32,
                    })?;
            candidates.sort();
            output.push(candidates);
        }
        Ok(output)
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
    T: VectorRepr + 'static,
{
    if data.ncols() == 0 {
        return Err(LeafBuildError::EmptyDimensions);
    }
    if data.nrows() > u32::MAX as usize {
        return Err(LeafBuildError::TooManyPoints(data.nrows()));
    }

    let candidates = DirectCandidates::new(data.nrows())?;
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
    candidates.into_lists()
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
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + 'static,
{
    if data.ncols() == 0 {
        return Err(LeafBuildError::EmptyDimensions);
    }
    if data.nrows() > u32::MAX as usize {
        return Err(LeafBuildError::TooManyPoints(data.nrows()));
    }

    leaves.par_iter().enumerate().try_for_each_init(
        LeafBuffers::default,
        |buffers, (leaf, point_ids)| {
            let leaf_k =
                compute_leaf::<A, M, T>(arch, data, leaf, point_ids, requested_k, buffers)?;
            let point_count = point_ids.len();
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
            hash_prune
                .add_leaf_edges(
                    point_ids,
                    &buffers.edge_offsets,
                    &buffers.edges[..edge_count],
                    &mut buffers.sketch_scratch,
                )
                .map_err(|source| LeafBuildError::HashPrune { leaf, source })
        },
    )
}

/// Add one leaf's symmetric neighbors to the direct candidate lists.
///
/// The function rejects empty, duplicate, unsorted, or out-of-range point IDs.
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
    candidates: &DirectCandidates,
) -> Result<(), LeafBuildError>
where
    A: PiPNNSIMDSchema,
    M: LeafMetric,
    T: VectorRepr + 'static,
{
    let leaf_k = compute_leaf::<A, M, T>(arch, data, leaf, point_ids, requested_k, buffers)?;
    if leaf_k == 0 {
        return Ok(());
    }
    buffers.prepare_local_adjacency(point_ids.len())?;
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..point_ids.len() * leaf_k],
        &mut buffers.local_adjacency[..point_ids.len()],
    )?;
    candidates.add_leaf(point_ids, &buffers.local_adjacency[..point_ids.len()])
}

fn compute_leaf<A, M, T>(
    arch: A,
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    requested_k: usize,
    buffers: &mut LeafBuffers,
) -> Result<usize, LeafBuildError>
where
    A: Architecture,
    A::f32x16: std::ops::Div<Output = A::f32x16>,
    <A::f32x16 as SIMDVector>::Mask: SIMDSelect<A::f32x16>,
    u64: From<<<<A::f32x16 as SIMDVector>::Mask as SIMDMask>::BitMask as SIMDMask>::Underlying>,
    M: KernelMetric,
    T: VectorRepr + 'static,
{
    if point_ids.is_empty() {
        return Err(LeafBuildError::EmptyLeaf { leaf });
    }
    for &point in point_ids {
        if point as usize >= data.nrows() {
            return Err(LeafBuildError::InvalidPointId {
                leaf,
                point,
                points: data.nrows(),
            });
        }
    }
    if let Some(pair) = point_ids.windows(2).find(|pair| pair[0] >= pair[1]) {
        if pair[0] == pair[1] {
            return Err(LeafBuildError::DuplicatePointId {
                leaf,
                point: pair[0],
            });
        }
        return Err(LeafBuildError::UnsortedPointIds { leaf });
    }
    let (leaf_k, neighbor_value_count) =
        buffers.prepare(leaf, point_ids.len(), data.ncols(), requested_k)?;
    if leaf_k == 0 {
        return Ok(0);
    }

    let point_value_count = point_ids.len() * data.ncols();
    let dot_count = point_ids.len() * point_ids.len();

    for (&point, point_output) in point_ids
        .iter()
        .zip(buffers.point_values[..point_value_count].chunks_exact_mut(data.ncols()))
    {
        let source_values = data.row(point as usize);
        T::as_f32_into(source_values, point_output).map_err(|source| {
            LeafBuildError::Conversion {
                leaf,
                point,
                source: source.into(),
            }
        })?;
    }

    diskann_linalg::sgemm_aat_lower(
        point_ids.len(),
        data.ncols(),
        &buffers.point_values[..point_value_count],
        &mut buffers.dots[..dot_count],
    )
    .map_err(|source| LeafBuildError::LowerAat { leaf, source })?;
    let dots = MatrixView::try_from(&buffers.dots[..dot_count], point_ids.len(), point_ids.len())
        .map_err(|_| LeafBuildError::InvalidView {
        leaf,
        buffer: "leaf dot-product matrix",
    })?;
    M::prepare_leaf_norms(NormPreparation {
        values: dots,
        norms: &mut buffers.norms,
    })
    .map_err(|source| allocation_error("leaf norms", point_ids.len(), source))?;
    let norms = &*buffers.norms;
    let output = MutMatrixView::try_from(
        &mut buffers.neighbors[..neighbor_value_count],
        point_ids.len(),
        leaf_k,
    )
    .map_err(|_| LeafBuildError::InvalidView {
        leaf,
        buffer: "leaf output",
    })?;
    nearest_neighbors::<A, M>(arch, dots, norms, output, &mut buffers.kernel_workspace)
        .map_err(|source| LeafBuildError::Kernel { leaf, source })?;
    Ok(leaf_k)
}

fn add_symmetric_neighbors(
    point_ids: &[u32],
    leaf_k: usize,
    neighbors: &[LeafNeighbor],
    local_adjacency: &mut [Vec<u32>],
) -> Result<(), LeafBuildError> {
    for (source, source_neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in source_neighbors {
            if !neighbor.is_assigned() {
                continue;
            }
            let target = neighbor.target as usize;
            let Some(&target_id) = point_ids.get(target) else {
                return Err(LeafBuildError::InvalidLocalTarget {
                    target: neighbor.target,
                    points: point_ids.len(),
                });
            };
            let source_id = point_ids[source];
            if source_id != target_id {
                local_adjacency[source].push(target_id);
                local_adjacency[target].push(source_id);
            }
        }
    }
    Ok(())
}

struct EdgeBuffers<'a> {
    seen: &'a mut [bool],
    offsets: &'a mut Vec<u32>,
    edges: &'a mut Vec<(u32, f32)>,
    cursor: &'a mut Vec<u32>,
}

fn build_symmetric_edge_csr(
    leaf: usize,
    point_ids: &[u32],
    leaf_k: usize,
    neighbors: &[LeafNeighbor],
    buffers: EdgeBuffers<'_>,
) -> Result<usize, LeafBuildError> {
    let EdgeBuffers {
        seen,
        offsets,
        edges,
        cursor,
    } = buffers;
    let point_count = point_ids.len();
    if leaf_k == 0 {
        resize("leaf edge offsets", offsets, point_count + 1, 0)?;
        offsets.fill(0);
        edges.clear();
        return Ok(0);
    }
    seen.fill(false);
    resize("leaf edge offsets", offsets, point_count + 1, 0)?;
    offsets.fill(0);

    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors {
            let target = neighbor.target as usize;
            if target >= point_count {
                return Err(LeafBuildError::InvalidLocalTarget {
                    target: neighbor.target,
                    points: point_count,
                });
            }
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
    resize("leaf edges", edges, edge_count, (0, 0.0))?;
    resize("leaf edge cursor", cursor, point_count, 0)?;
    cursor.copy_from_slice(&offsets[..point_count]);
    seen.fill(false);

    for (source, neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in neighbors {
            let target = neighbor.target as usize;
            write_directed_edge(
                point_count,
                source,
                target,
                neighbor.distance,
                seen,
                edges,
                cursor,
            );
            write_directed_edge(
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

fn write_directed_edge(
    point_count: usize,
    source: usize,
    target: usize,
    distance: f32,
    seen: &mut [bool],
    edges: &mut [(u32, f32)],
    cursor: &mut [u32],
) {
    let seen_entry = &mut seen[source * point_count + target];
    if !*seen_entry {
        *seen_entry = true;
        let edge_slot = cursor[source] as usize;
        edges[edge_slot] = (target as u32, distance);
        cursor[source] += 1;
    }
}

fn grow<T: Clone>(
    buffer: &'static str,
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), LeafBuildError> {
    if values.len() < len {
        resize(buffer, values, len, value)?;
    }
    Ok(())
}

fn resize<T: Clone>(
    buffer: &'static str,
    values: &mut Vec<T>,
    len: usize,
    value: T,
) -> Result<(), LeafBuildError> {
    let additional = len.saturating_sub(values.len());
    values
        .try_reserve(additional)
        .map_err(|source| allocation_error(buffer, additional, source))?;
    values.resize(len, value);
    Ok(())
}

fn allocation_error(
    buffer: &'static str,
    additional: usize,
    source: TryReserveError,
) -> LeafBuildError {
    LeafBuildError::Allocation {
        buffer,
        additional,
        source,
    }
}

#[cfg(test)]
mod tests {
    use diskann_utils::views::MatrixView;
    use diskann_vector::distance::Metric;
    use diskann_wide::arch::{self, Target1};
    use half::f16;
    use rstest::rstest;
    use std::collections::BTreeSet;

    use super::super::{leaf_kernel::LeafNeighbor, simd::PiPNNSIMDSchema};
    use super::{
        DirectCandidates, EdgeBuffers, LeafBuffers, LeafBuildError, add_symmetric_neighbors,
        allocation_error, build_leaf_candidates, build_symmetric_edge_csr,
    };

    fn matrix_view<T>(data: &[T], rows: usize, columns: usize) -> MatrixView<'_, T> {
        MatrixView::try_from(data, rows, columns).unwrap()
    }

    fn leaf_build_pool() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
    }

    struct LeafBuildCall<'a, T> {
        data: MatrixView<'a, T>,
        leaves: Vec<Vec<u32>>,
        k: usize,
    }

    struct DispatchLeafBuild(Metric);

    impl<A, T>
        Target1<
            A,
            Result<Vec<crate::graph::AdjacencyList<u32>>, LeafBuildError>,
            LeafBuildCall<'_, T>,
        > for DispatchLeafBuild
    where
        A: PiPNNSIMDSchema,
        T: crate::utils::VectorRepr + 'static,
    {
        fn run(
            self,
            arch: A,
            call: LeafBuildCall<'_, T>,
        ) -> Result<Vec<crate::graph::AdjacencyList<u32>>, LeafBuildError> {
            use super::super::kernel_metric::{Cosine, CosineNormalized, InnerProduct, L2};

            match self.0 {
                Metric::L2 => {
                    build_leaf_candidates::<A, L2, T>(arch, call.data, call.leaves, call.k)
                }
                Metric::Cosine => {
                    build_leaf_candidates::<A, Cosine, T>(arch, call.data, call.leaves, call.k)
                }
                Metric::CosineNormalized => build_leaf_candidates::<A, CosineNormalized, T>(
                    arch,
                    call.data,
                    call.leaves,
                    call.k,
                ),
                Metric::InnerProduct => build_leaf_candidates::<A, InnerProduct, T>(
                    arch,
                    call.data,
                    call.leaves,
                    call.k,
                ),
            }
        }
    }

    fn build_candidate_graph<T>(
        data: MatrixView<'_, T>,
        leaves: &[Vec<u32>],
        k: usize,
        metric: Metric,
    ) -> Result<Vec<crate::graph::AdjacencyList<u32>>, LeafBuildError>
    where
        T: crate::utils::VectorRepr + 'static,
    {
        leaf_build_pool().install(|| {
            arch::dispatch1_no_features(
                DispatchLeafBuild(metric),
                LeafBuildCall {
                    data,
                    leaves: leaves.to_vec(),
                    k,
                },
            )
        })
    }

    fn adjacency_lists(graph: Vec<crate::graph::AdjacencyList<u32>>) -> Vec<Vec<u32>> {
        graph.into_iter().map(Vec::from).collect()
    }

    fn circular_overlapping_leaves(
        point_count: usize,
        leaf_count: usize,
        leaf_size: usize,
    ) -> Vec<Vec<u32>> {
        (0..leaf_count)
            .map(|offset| {
                (0..leaf_size)
                    .map(|point| ((point + offset) % point_count) as u32)
                    .collect()
            })
            .collect()
    }

    fn brute_force_symmetric_l2(data: &[[f32; 2]], k: usize) -> Vec<Vec<u32>> {
        let mut graph = vec![BTreeSet::new(); data.len()];
        for (source, left) in data.iter().enumerate() {
            let mut nearest: Vec<_> = data
                .iter()
                .enumerate()
                .filter(|(target, _)| *target != source)
                .map(|(target, right)| {
                    let distance = left
                        .iter()
                        .zip(right)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>();
                    (target, distance)
                })
                .collect();
            nearest.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for &(target, _) in nearest.iter().take(k) {
                graph[source].insert(target as u32);
                graph[target].insert(source as u32);
            }
        }
        graph
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect())
            .collect()
    }

    #[test]
    fn leaf_adjacency_matches_an_independent_all_pairs_reference() {
        // Given
        let points = [
            [0.0_f32, 0.0],
            [1.0, 0.2],
            [3.1, 0.5],
            [7.8, 1.4],
            [-2.3, 4.1],
            [6.7, -3.2],
        ];
        let flat: Vec<_> = points.into_iter().flatten().collect();
        let expected_adjacency = brute_force_symmetric_l2(&points, 2);

        // When
        let actual_adjacency = adjacency_lists(
            build_candidate_graph(
                matrix_view(&flat, points.len(), 2),
                &[(0..points.len() as u32).collect()],
                2,
                Metric::L2,
            )
            .unwrap(),
        );

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn non_rankable_neighbors_are_omitted() {
        // Given
        let data = [0.0_f32, 1.0, f32::NAN];
        let expected_adjacency = [vec![1], vec![0], vec![]];

        // When
        let graph = build_candidate_graph(
            matrix_view(&data, 3, 1),
            &[vec![0, 1, 2]],
            2,
            Metric::InnerProduct,
        )
        .unwrap();
        let actual_adjacency = adjacency_lists(graph);

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn overlapping_leaves_contribute_each_candidate_once() {
        // Given
        let data = [0.0_f32, 1.0, 2.0, 3.0];
        let leaves = vec![vec![0, 1, 2], vec![0, 2, 3], vec![0, 1, 2]];
        let expected_adjacency = [vec![1, 2, 3], vec![0, 2], vec![0, 1, 3], vec![0, 2]];

        // When
        let graph =
            build_candidate_graph(matrix_view(&data, 4, 1), &leaves, 2, Metric::L2).unwrap();
        let actual_adjacency = adjacency_lists(graph);

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn symmetric_edges_can_give_a_center_more_than_two_k_neighbors() {
        let dimensions = 9;
        let mut data = vec![0.0_f32; 10 * dimensions];
        for source in 1..10 {
            data[source * dimensions + source - 1] = 1.0;
        }

        let graph = build_candidate_graph(
            matrix_view(&data, 10, dimensions),
            &[(0..10).collect()],
            1,
            Metric::L2,
        )
        .unwrap();

        assert_eq!(&*graph[0], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(graph.iter().enumerate().all(|(source, neighbors)| {
            neighbors.iter().all(|&target| target as usize != source)
                && neighbors
                    .iter()
                    .all(|&target| graph[target as usize].contains(source as u32))
        }));
    }

    fn build_l2_candidate_graph<T>(data: &[T], points: usize, dimensions: usize) -> Vec<Vec<u32>>
    where
        T: crate::utils::VectorRepr + 'static,
    {
        let leaves = vec![(0..points as u32).collect()];
        adjacency_lists(
            build_candidate_graph(
                matrix_view(data, points, dimensions),
                &leaves,
                2,
                Metric::L2,
            )
            .unwrap(),
        )
    }

    fn assert_source_conversion_matches_f32<T>(label: &str, convert: impl Fn(u8) -> T)
    where
        T: crate::utils::VectorRepr + 'static,
    {
        let points = 8;
        // Source dimension controls VectorRepr conversion chunking. Cover tails on
        // both sides of 4-, 8-, and 16-element boundaries, then a second 16-lane
        // chunk. Input integers remain exact in every tested representation.
        for dimensions in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33] {
            let raw: Vec<u8> = (0..points * dimensions)
                .map(|index| {
                    let source = index / dimensions;
                    let dimension = index % dimensions;
                    (source + dimension) as u8
                })
                .collect();
            let f32_data: Vec<f32> = raw.iter().map(|&value| value as f32).collect();
            let converted: Vec<T> = raw.iter().copied().map(&convert).collect();
            assert_eq!(
                build_l2_candidate_graph(&converted, points, dimensions),
                build_l2_candidate_graph(&f32_data, points, dimensions),
                "{label} dimensions={dimensions}"
            );
        }
    }

    #[test]
    fn f16_conversion_matches_f32_across_dimension_boundaries() {
        assert_source_conversion_matches_f32("f16", |value| f16::from_f32(value as f32));
    }

    #[test]
    fn u8_conversion_matches_f32_across_dimension_boundaries() {
        assert_source_conversion_matches_f32("u8", |value| value);
    }

    #[test]
    fn i8_conversion_matches_f32_across_dimension_boundaries() {
        // Applying the same translation to every coordinate preserves L2 pair
        // ordering while exercising signed conversion.
        assert_source_conversion_matches_f32("i8", |value| value as i8 - 11);
    }

    #[rstest]
    fn candidate_graph_is_symmetric_unique_and_non_self(
        #[values(
            Metric::L2,
            Metric::Cosine,
            Metric::CosineNormalized,
            Metric::InnerProduct
        )]
        metric: Metric,
    ) {
        // Given
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let unit_vectors = [1.0_f32, 0.0, diagonal, diagonal, 0.0, 1.0, -1.0, 0.0];
        let leaves = vec![vec![0, 1, 2, 3], vec![0, 1, 2, 3]];

        // When
        let graph =
            build_candidate_graph(matrix_view(&unit_vectors, 4, 2), &leaves, 2, metric).unwrap();

        // Then
        for (source, neighbors) in graph.iter().enumerate() {
            assert!(neighbors.iter().all(|&target| target as usize != source));
            assert!(
                neighbors
                    .iter()
                    .all(|&target| graph[target as usize].contains(source as u32))
            );
            assert!(neighbors.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }

    #[test]
    fn parallel_leaf_schedule_does_not_change_candidate_order() {
        let data: Vec<f32> = (0..64).map(|value| value as f32).collect();
        let leaves = circular_overlapping_leaves(64, 32, 16);
        let expected_candidate_order =
            build_candidate_graph(matrix_view(&data, 64, 1), &leaves, 2, Metric::L2).unwrap();
        for _ in 0..8 {
            let actual_candidate_order =
                build_candidate_graph(matrix_view(&data, 64, 1), &leaves, 2, Metric::L2).unwrap();
            assert_eq!(actual_candidate_order, expected_candidate_order);
        }
    }

    #[test]
    fn rejects_invalid_dimensions_and_leaf_membership() {
        let data = [0.0_f32, 1.0];
        let no_dimensions = MatrixView::try_from(&data[..0], 2, 0).unwrap();
        assert!(matches!(
            build_candidate_graph(no_dimensions, &[], 1, Metric::L2),
            Err(LeafBuildError::EmptyDimensions)
        ));
        assert!(matches!(
            build_candidate_graph(matrix_view(&data, 2, 1), &[vec![]], 1, Metric::L2),
            Err(LeafBuildError::EmptyLeaf { leaf: 0 })
        ));
        assert!(matches!(
            build_candidate_graph(
                matrix_view(&data, 2, 1),
                &[vec![0, 2]],
                1,
                Metric::L2,
            ),
            Err(LeafBuildError::InvalidPointId {
                leaf: 0,
                point: 2,
                points: 2
            })
        ));
        assert!(matches!(
            build_candidate_graph(matrix_view(&data, 2, 1), &[vec![2]], 1, Metric::L2),
            Err(LeafBuildError::InvalidPointId { point: 2, .. })
        ));
        assert!(matches!(
            build_candidate_graph(
                matrix_view(&data, 2, 1),
                &[vec![0, 2]],
                0,
                Metric::L2,
            ),
            Err(LeafBuildError::InvalidPointId { point: 2, .. })
        ));
        assert!(matches!(
            build_candidate_graph(
                matrix_view(&data, 2, 1),
                &[vec![0, 0]],
                1,
                Metric::L2,
            ),
            Err(LeafBuildError::DuplicatePointId { leaf: 0, point: 0 })
        ));
        assert!(matches!(
            build_candidate_graph(
                matrix_view(&data, 2, 1),
                &[vec![1, 0]],
                1,
                Metric::L2,
            ),
            Err(LeafBuildError::UnsortedPointIds { leaf: 0 })
        ));
    }

    #[test]
    fn singleton_leaves_add_no_candidates() {
        // Given
        let point_values = [0.0_f32, 1.0, 2.0];
        let singleton_leaves = [vec![0], vec![1], vec![2]];
        let expected_adjacency: [Vec<u32>; 3] = [vec![], vec![], vec![]];

        // When
        let actual_adjacency = adjacency_lists(
            build_candidate_graph(
                matrix_view(&point_values, 3, 1),
                &singleton_leaves,
                1,
                Metric::L2,
            )
            .unwrap(),
        );

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn zero_k_adds_no_candidates() {
        // Given
        let point_values = [0.0_f32, 1.0, 2.0];
        let point_count = 3;
        let dimensions = 1;
        let zero_k = 0;
        let leaves = [vec![0, 1, 2]];
        let expected_adjacency: [Vec<u32>; 3] = [vec![], vec![], vec![]];

        // When
        let actual_adjacency = adjacency_lists(
            build_candidate_graph(
                matrix_view(&point_values, point_count, dimensions),
                &leaves,
                zero_k,
                Metric::L2,
            )
            .unwrap(),
        );

        // Then
        assert_eq!(actual_adjacency, expected_adjacency);
    }

    #[test]
    fn leaf_buffer_preparation_reports_shape_overflow_before_allocating() {
        let mut buffers = LeafBuffers::default();
        assert!(matches!(
            buffers.prepare(7, usize::MAX, 2, 1),
            Err(LeafBuildError::ShapeOverflow { leaf: 7, .. })
        ));
    }

    #[test]
    fn rejects_an_invalid_kernel_target() {
        let mut graph = vec![Vec::new(); 2];
        let error = add_symmetric_neighbors(
            &[10, 20],
            1,
            &[
                super::super::leaf_kernel::LeafNeighbor::new(9, 1.0),
                super::super::leaf_kernel::LeafNeighbor::new(0, 1.0),
            ],
            &mut graph,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LeafBuildError::InvalidLocalTarget {
                target: 9,
                points: 2
            }
        ));
    }

    #[test]
    fn symmetric_edge_mapping_skips_duplicate_ids_instead_of_adding_self_edges() {
        let mut graph = vec![Vec::new(); 2];
        add_symmetric_neighbors(
            &[7, 7],
            1,
            &[
                super::super::leaf_kernel::LeafNeighbor::new(1, 0.0),
                super::super::leaf_kernel::LeafNeighbor::new(0, 0.0),
            ],
            &mut graph,
        )
        .unwrap();
        assert!(graph.iter().all(|neighbors| neighbors.is_empty()));
    }

    #[test]
    fn poisoned_candidate_lists_return_errors() {
        let candidates = DirectCandidates::new(1).unwrap();
        let _ = std::panic::catch_unwind(|| {
            let _guard = candidates.lists[0].lock().unwrap();
            panic!("poison candidate list");
        });
        assert!(matches!(
            candidates.add_leaf(&[0], &[Vec::new()]),
            Err(LeafBuildError::PoisonedCandidateList { point: 0 })
        ));
        assert!(matches!(
            candidates.into_lists(),
            Err(LeafBuildError::PoisonedCandidateList { point: 0 })
        ));
    }

    #[test]
    fn allocation_errors_preserve_buffer_context() {
        let mut values = Vec::<u8>::new();
        let source = values.try_reserve(usize::MAX).unwrap_err();
        let error = allocation_error("test", 1, source);
        assert!(matches!(
            error,
            LeafBuildError::Allocation {
                buffer: "test",
                additional: 1,
                ..
            }
        ));
    }

    #[test]
    fn direct_candidate_accumulator_keeps_unique_sorted_lists() {
        let candidates = DirectCandidates::new(2).unwrap();
        candidates
            .add_leaf(&[0, 1], &[vec![1, 1], vec![0]])
            .unwrap();
        assert_eq!(
            adjacency_lists(candidates.into_lists().unwrap()),
            [vec![1], vec![0]]
        );
    }

    #[test]
    fn symmetric_edge_csr_matches_expected_adjacency() {
        let point_ids = [10, 20, 30];
        let neighbors = [
            LeafNeighbor::new(1, 1.0),
            LeafNeighbor::new(2, 2.0),
            LeafNeighbor::new(1, 1.5),
        ];
        let mut seen = vec![false; 9];
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

        assert_eq!(count, 4);
        assert_eq!(offsets, [0, 1, 3, 4]);
        assert_eq!(edges, [(1, 1.0), (0, 1.0), (2, 2.0), (1, 2.0)]);
    }
    #[test]
    fn symmetric_edge_csr_deduplicates_edges_seen_from_both_endpoints() {
        let point_ids = [10, 20];
        let neighbors = [LeafNeighbor::new(1, 1.0), LeafNeighbor::new(0, 1.0)];
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
    fn symmetric_edge_csr_rejects_out_of_range_local_targets() {
        let mut seen = vec![false; 4];
        let mut offsets = Vec::new();
        let mut edges = Vec::new();
        let mut cursor = Vec::new();
        let error = build_symmetric_edge_csr(
            7,
            &[10, 20],
            1,
            &[LeafNeighbor::new(2, 1.0), LeafNeighbor::new(0, 1.0)],
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            LeafBuildError::InvalidLocalTarget {
                target: 2,
                points: 2
            }
        ));
    }
    #[test]
    fn zero_k_edge_csr_has_empty_adjacency() {
        let point_ids = [10, 20, 30];
        let mut seen = vec![false; 9];
        let mut offsets = Vec::new();
        let mut edges = vec![(99, 99.0)];
        let mut cursor = Vec::new();

        let count = build_symmetric_edge_csr(
            0,
            &point_ids,
            0,
            &[],
            EdgeBuffers {
                seen: &mut seen,
                offsets: &mut offsets,
                edges: &mut edges,
                cursor: &mut cursor,
            },
        )
        .unwrap();

        assert_eq!(count, 0);
        assert_eq!(offsets, [0, 0, 0, 0]);
        assert!(edges.is_empty());
    }
}
