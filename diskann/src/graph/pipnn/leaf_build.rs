/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf-local graph construction and candidate accumulation.
//!
//! Partitioning supplies leaves as global point IDs. For each leaf this module:
//!
//! 1. validates IDs and converts only those point vectors to reusable `f32` scratch;
//! 2. computes the lower triangle of `A · Aᵀ`;
//! 3. runs the dual-endpoint leaf top-k kernel; and
//! 4. translates leaf-local positions back to dataset IDs.
//!
//! The final step merges symmetric adjacency lists under per-point locks because
//! overlapping leaves are processed concurrently. Numeric buffers retain their
//! high-water length; every consumer therefore receives an explicit active
//! prefix rather than treating `Vec::len()` as the current leaf shape.

use std::{
    collections::{HashSet, TryReserveError},
    sync::Mutex,
};

use crate::{graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use rayon::prelude::*;

use crate::leaf_kernel::{
    leaf_neighbor_count, leaf_output_len, LeafInput, LeafKernel, LeafKernelError,
    LeafKernelWorkspace, LeafNeighbor,
};

/// Failure while converting leaves into direct graph candidates.
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
    #[error("leaf {leaf} shape {rows} x {columns} overflows usize")]
    ShapeOverflow {
        leaf: usize,
        rows: usize,
        columns: usize,
    },
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
        source: diskann::ANNError,
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
}

/// Scratch leased to one Rayon job and reused for successive leaves.
///
/// The three numerical vectors retain their largest observed leaf shape. The
/// adjacency lists are prepared separately because zero-k/singleton leaves never
/// write them, and because later candidate-merging modes do not necessarily use
/// this representation.
#[derive(Default)]
struct LeafBuffers {
    point_values: Vec<f32>,
    dots: Vec<f32>,
    neighbors: Vec<LeafNeighbor>,
    local_adjacency: Vec<AdjacencyList<u32>>,
    kernel_workspace: LeafKernelWorkspace,
    seen_ids: HashSet<u32>,
}

impl LeafBuffers {
    fn prepare(
        &mut self,
        leaf: usize,
        point_count: usize,
        dimension_count: usize,
        requested_k: usize,
    ) -> Result<usize, LeafBuildError> {
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
        let neighbor_count = leaf_output_len(point_count, requested_k)
            .map_err(|source| LeafBuildError::Kernel { leaf, source })?;

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
        Ok(leaf_k)
    }

    fn prepare_local_adjacency(&mut self, point_count: usize) -> Result<(), LeafBuildError> {
        let additional = point_count.saturating_sub(self.local_adjacency.len());
        self.local_adjacency
            .try_reserve(additional)
            .map_err(|source| allocation_error("leaf adjacency lists", additional, source))?;
        self.local_adjacency
            .resize_with(point_count, AdjacencyList::new);
        self.local_adjacency[..point_count]
            .iter_mut()
            .for_each(AdjacencyList::clear);
        Ok(())
    }
}

/// Concurrent accumulator indexed by global dataset ID.
///
/// A point may appear in several overlapping leaves, so workers lock only the
/// destination list long enough to append one leaf's additions. Sorting and
/// duplicate removal are deferred until all leaves finish; doing either under
/// the lock would lengthen the contended section for no semantic benefit.
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
        local_adjacency: &[AdjacencyList<u32>],
    ) -> Result<(), LeafBuildError> {
        for (&source, additions) in point_ids.iter().zip(local_adjacency) {
            // Every point ID is validated before leaf-local work begins.
            let candidates = &self.lists[source as usize];
            let mut candidates = candidates
                .lock()
                .map_err(|_| poisoned_candidate_list(source))?;
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
            let mut candidates = candidates
                .into_inner()
                .map_err(|_| poisoned_candidate_list(point as u32))?;
            candidates.sort();
            output.push(candidates);
        }
        Ok(output)
    }
}

/// Build symmetric leaf-local k-NN graphs and retain every unique global candidate.
#[allow(clippy::disallowed_methods)] // The supplied pool owns this terminal operation.
pub(crate) fn build_leaf_candidates<T>(
    data: MatrixView<'_, T>,
    leaves: Vec<Vec<u32>>,
    requested_k: usize,
    metric: Metric,
) -> Result<Vec<AdjacencyList<u32>>, LeafBuildError>
where
    T: VectorRepr + 'static,
{
    if data.ncols() == 0 {
        return Err(LeafBuildError::EmptyDimensions);
    }
    if data.nrows() > u32::MAX as usize {
        return Err(LeafBuildError::TooManyPoints(data.nrows()));
    }

    let candidates = DirectCandidates::new(data.nrows())?;
    // Metric and ISA are selected before Rayon workers start. Workers share
    // this Copy handle; each output view supplies its leaf-specific width.
    let kernel = LeafKernel::new(metric);
    leaves.par_iter().enumerate().try_for_each_init(
        LeafBuffers::default,
        |buffers, (leaf, point_ids)| {
            build_leaf(
                data,
                leaf,
                point_ids,
                requested_k,
                &kernel,
                buffers,
                &candidates,
            )
        },
    )?;
    candidates.into_lists()
}

/// Build and publish one leaf's symmetric neighbor lists.
///
/// Validation precedes all dataset indexing. Sorted partition output takes the
/// adjacent-duplicate path, while arbitrary-order callers use `seen_ids`. The
/// active lengths computed after `prepare` must be used for every later slice,
/// because the reusable vectors may still be longer than this leaf.
fn build_leaf<T>(
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    requested_k: usize,
    kernel: &LeafKernel,
    buffers: &mut LeafBuffers,
    candidates: &DirectCandidates,
) -> Result<(), LeafBuildError>
where
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
    if point_ids.is_sorted() {
        if let Some(pair) = point_ids.windows(2).find(|pair| pair[0] == pair[1]) {
            return Err(LeafBuildError::DuplicatePointId {
                leaf,
                point: pair[0],
            });
        }
    } else {
        buffers.seen_ids.clear();
        buffers
            .seen_ids
            .try_reserve(point_ids.len())
            .map_err(|source| allocation_error("leaf ID set", point_ids.len(), source))?;
        for &point in point_ids {
            if !buffers.seen_ids.insert(point) {
                return Err(LeafBuildError::DuplicatePointId { leaf, point });
            }
        }
    }
    let leaf_k = buffers.prepare(leaf, point_ids.len(), data.ncols(), requested_k)?;
    if leaf_k == 0 {
        return Ok(());
    }

    let point_value_count = point_ids.len() * data.ncols();
    let dot_count = point_ids.len() * point_ids.len();
    let neighbor_value_count = leaf_output_len(point_ids.len(), requested_k)
        .map_err(|source| LeafBuildError::Kernel { leaf, source })?;

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
        .map_err(|error| LeafBuildError::Kernel {
        leaf,
        source: LeafKernelError::InvalidBufferLength {
            buffer: "leaf dot-product matrix",
            expected: dot_count,
            actual: error.into_inner().len(),
        },
    })?;
    let output = MutMatrixView::try_from(
        &mut buffers.neighbors[..neighbor_value_count],
        point_ids.len(),
        leaf_k,
    )
    .map_err(|error| LeafBuildError::Kernel {
        leaf,
        source: LeafKernelError::InvalidBufferLength {
            buffer: "output",
            expected: neighbor_value_count,
            actual: error.into_inner().len(),
        },
    })?;
    kernel
        .nearest_neighbors(LeafInput { dots }, output, &mut buffers.kernel_workspace)
        .map_err(|source| LeafBuildError::Kernel { leaf, source })?;

    buffers.prepare_local_adjacency(point_ids.len())?;
    add_symmetric_neighbors(
        point_ids,
        leaf_k,
        &buffers.neighbors[..neighbor_value_count],
        &mut buffers.local_adjacency[..point_ids.len()],
    )?;
    candidates.add_leaf(point_ids, &buffers.local_adjacency[..point_ids.len()])
}

fn add_symmetric_neighbors(
    point_ids: &[u32],
    leaf_k: usize,
    neighbors: &[LeafNeighbor],
    local_adjacency: &mut [AdjacencyList<u32>],
) -> Result<(), LeafBuildError> {
    for (source, source_neighbors) in neighbors.chunks_exact(leaf_k).enumerate() {
        for neighbor in source_neighbors {
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

fn poisoned_candidate_list(point: u32) -> LeafBuildError {
    LeafBuildError::PoisonedCandidateList { point }
}

#[cfg(test)]
mod tests;
