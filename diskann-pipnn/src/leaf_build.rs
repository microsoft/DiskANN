/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Leaf construction and direct candidate accumulation.

use std::{
    collections::{HashSet, TryReserveError},
    sync::Mutex,
};

use diskann::{graph::AdjacencyList, utils::VectorRepr};
use diskann_utils::views::MatrixView;
use diskann_vector::distance::Metric;
use rayon::prelude::*;

use crate::leaf_kernel::{
    nearest_leaf_neighbors, LeafKernelError, LeafNeighbor, LeafTopK, LeafTopKWorkspace,
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
    #[error("leaf kernel returned local position {position} for a {points}-point leaf")]
    InvalidLocalPosition { position: u32, points: usize },
    #[error("candidate row {point} is poisoned")]
    PoisonedCandidateRow { point: u32 },
    #[error("leaf {leaf} produced too many directed edges")]
    TooManyEdges { leaf: usize },
}

#[derive(Default)]
struct LeafBuffers {
    points: Vec<f32>,
    dots: Vec<f32>,
    nearest: Vec<LeafNeighbor>,
    local_graph: Vec<AdjacencyList<u32>>,
    top_k: LeafTopKWorkspace,
    seen_ids: HashSet<u32>,
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
        points: usize,
        dimensions: usize,
        k: usize,
    ) -> Result<usize, LeafBuildError> {
        let point_values = points
            .checked_mul(dimensions)
            .ok_or(LeafBuildError::ShapeOverflow {
                leaf,
                rows: points,
                columns: dimensions,
            })?;
        let dot_values = points
            .checked_mul(points)
            .ok_or(LeafBuildError::ShapeOverflow {
                leaf,
                rows: points,
                columns: points,
            })?;
        let actual_k = k.min(points.saturating_sub(1));
        let nearest_values = points
            .checked_mul(actual_k)
            .ok_or(LeafBuildError::ShapeOverflow {
                leaf,
                rows: points,
                columns: actual_k,
            })?;

        resize("leaf points", &mut self.points, point_values, 0.0)?;
        resize("leaf dot products", &mut self.dots, dot_values, 0.0)?;
        resize(
            "leaf nearest neighbors",
            &mut self.nearest,
            nearest_values,
            LeafNeighbor::default(),
        )?;
        resize("leaf seen pairs", &mut self.seen_pairs, dot_values, false)?;
        let additional = points.saturating_sub(self.local_graph.len());
        self.local_graph
            .try_reserve(additional)
            .map_err(|source| allocation_error("leaf adjacency rows", additional, source))?;
        self.local_graph.resize_with(points, AdjacencyList::new);
        self.local_graph[..points]
            .iter_mut()
            .for_each(AdjacencyList::clear);
        Ok(actual_k)
    }
}

struct DirectCandidates {
    rows: Vec<Mutex<AdjacencyList<u32>>>,
}

impl DirectCandidates {
    fn new(points: usize) -> Result<Self, LeafBuildError> {
        let mut rows = Vec::new();
        rows.try_reserve_exact(points)
            .map_err(|source| allocation_error("candidate rows", points, source))?;
        rows.resize_with(points, || Mutex::new(AdjacencyList::new()));
        Ok(Self { rows })
    }

    fn add_leaf(
        &self,
        point_ids: &[u32],
        local_graph: &[AdjacencyList<u32>],
    ) -> Result<(), LeafBuildError> {
        for (&source, additions) in point_ids.iter().zip(local_graph) {
            // Every point ID is validated before leaf-local work begins.
            let row = &self.rows[source as usize];
            let mut row = row.lock().map_err(|_| poisoned_row(source))?;
            row.extend_from_slice(additions);
        }
        Ok(())
    }

    fn into_rows(self) -> Result<Vec<AdjacencyList<u32>>, LeafBuildError> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(self.rows.len())
            .map_err(|source| allocation_error("candidate output", self.rows.len(), source))?;
        for (point, row) in self.rows.into_iter().enumerate() {
            let mut row = row.into_inner().map_err(|_| poisoned_row(point as u32))?;
            row.sort();
            output.push(row);
        }
        Ok(output)
    }
}

/// Build symmetric leaf-local k-NN graphs and retain every unique global candidate.
#[allow(clippy::disallowed_methods)] // The supplied pool owns this terminal operation.
pub(crate) fn build_leaf_candidates<T>(
    data: MatrixView<'_, T>,
    leaves: &[Vec<u32>],
    k: usize,
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
    leaves.par_iter().enumerate().try_for_each_init(
        LeafBuffers::default,
        |buffers, (leaf, point_ids)| {
            build_leaf(data, leaf, point_ids, k, metric, buffers, &candidates)
        },
    )?;
    candidates.into_rows()
}

/// Build leaf-local edges and stream them into HashPrune reservoirs.
#[allow(clippy::disallowed_methods)] // The supplied pool owns this terminal operation.
pub(crate) fn add_hash_prune_candidates<T>(
    data: MatrixView<'_, T>,
    leaves: &[Vec<u32>],
    k: usize,
    metric: Metric,
    hash_prune: &crate::hash_prune::HashPrune,
) -> Result<(), LeafBuildError>
where
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
            let actual_k = compute_leaf(data, leaf, point_ids, k, metric, buffers)?;
            let edge_count = build_symmetric_edge_csr(
                leaf,
                point_ids,
                actual_k,
                &buffers.nearest,
                EdgeBuffers {
                    seen: &mut buffers.seen_pairs,
                    offsets: &mut buffers.edge_offsets,
                    edges: &mut buffers.edges,
                    cursor: &mut buffers.edge_cursor,
                },
            )?;
            hash_prune.add_leaf_edges(
                point_ids,
                &buffers.edge_offsets,
                &buffers.edges[..edge_count],
                &mut buffers.sketch_scratch,
            );
            Ok(())
        },
    )
}

fn build_leaf<T>(
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    k: usize,
    metric: Metric,
    buffers: &mut LeafBuffers,
    candidates: &DirectCandidates,
) -> Result<(), LeafBuildError>
where
    T: VectorRepr + 'static,
{
    let actual_k = compute_leaf(data, leaf, point_ids, k, metric, buffers)?;
    if actual_k == 0 {
        return Ok(());
    }
    add_symmetric_edges(
        point_ids,
        actual_k,
        &buffers.nearest,
        &mut buffers.local_graph[..point_ids.len()],
    )?;
    candidates.add_leaf(point_ids, &buffers.local_graph[..point_ids.len()])
}

fn compute_leaf<T>(
    data: MatrixView<'_, T>,
    leaf: usize,
    point_ids: &[u32],
    k: usize,
    metric: Metric,
    buffers: &mut LeafBuffers,
) -> Result<usize, LeafBuildError>
where
    T: VectorRepr + 'static,
{
    if point_ids.is_empty() {
        return Err(LeafBuildError::EmptyLeaf { leaf });
    }
    buffers.seen_ids.clear();
    buffers
        .seen_ids
        .try_reserve(point_ids.len())
        .map_err(|source| allocation_error("leaf ID set", point_ids.len(), source))?;
    for &point in point_ids {
        if point as usize >= data.nrows() {
            return Err(LeafBuildError::InvalidPointId {
                leaf,
                point,
                points: data.nrows(),
            });
        }
        if !buffers.seen_ids.insert(point) {
            return Err(LeafBuildError::DuplicatePointId { leaf, point });
        }
    }
    let actual_k = buffers.prepare(leaf, point_ids.len(), data.ncols(), k)?;
    if actual_k == 0 {
        return Ok(0);
    }

    for (&point, output) in point_ids
        .iter()
        .zip(buffers.points.chunks_exact_mut(data.ncols()))
    {
        let row = data.row(point as usize);
        T::as_f32_into(row, output).map_err(|source| LeafBuildError::Conversion {
            leaf,
            point,
            source: source.into(),
        })?;
    }

    diskann_linalg::sgemm_aat_lower(
        &buffers.points,
        point_ids.len(),
        data.ncols(),
        &mut buffers.dots,
    )
    .map_err(|source| LeafBuildError::LowerAat { leaf, source })?;
    nearest_leaf_neighbors(
        LeafTopK {
            dots: &buffers.dots,
            points: point_ids.len(),
            metric,
        },
        k,
        &mut buffers.nearest,
        &mut buffers.top_k,
    )
    .map_err(|source| LeafBuildError::Kernel { leaf, source })?;
    Ok(actual_k)
}

fn add_symmetric_edges(
    point_ids: &[u32],
    k: usize,
    nearest: &[LeafNeighbor],
    local_graph: &mut [AdjacencyList<u32>],
) -> Result<(), LeafBuildError> {
    for (source, nearest) in nearest.chunks_exact(k).enumerate() {
        for neighbor in nearest {
            let target = neighbor.position as usize;
            let Some(&target_id) = point_ids.get(target) else {
                return Err(LeafBuildError::InvalidLocalPosition {
                    position: neighbor.position,
                    points: point_ids.len(),
                });
            };
            let source_id = point_ids[source];
            if source_id != target_id {
                local_graph[source].push(target_id);
                local_graph[target].push(source_id);
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
    k: usize,
    nearest: &[LeafNeighbor],
    buffers: EdgeBuffers<'_>,
) -> Result<usize, LeafBuildError> {
    let EdgeBuffers {
        seen,
        offsets,
        edges,
        cursor,
    } = buffers;
    let points = point_ids.len();
    if k == 0 {
        resize("leaf edge offsets", offsets, points + 1, 0)?;
        offsets.fill(0);
        edges.clear();
        return Ok(0);
    }
    seen.fill(false);
    resize("leaf edge offsets", offsets, points + 1, 0)?;
    offsets.fill(0);

    for (source, neighbors) in nearest.chunks_exact(k).enumerate() {
        for neighbor in neighbors {
            let target = neighbor.position as usize;
            if target >= points {
                return Err(LeafBuildError::InvalidLocalPosition {
                    position: neighbor.position,
                    points,
                });
            }
            count_directed_edge(leaf, points, source, target, seen, offsets)?;
            count_directed_edge(leaf, points, target, source, seen, offsets)?;
        }
    }
    for index in 1..=points {
        offsets[index] = offsets[index]
            .checked_add(offsets[index - 1])
            .ok_or(LeafBuildError::TooManyEdges { leaf })?;
    }

    let edge_count = offsets[points] as usize;
    resize("leaf edges", edges, edge_count, (0, 0.0))?;
    resize("leaf edge cursor", cursor, points, 0)?;
    cursor.copy_from_slice(&offsets[..points]);
    seen.fill(false);

    for (source, neighbors) in nearest.chunks_exact(k).enumerate() {
        for neighbor in neighbors {
            let target = neighbor.position as usize;
            write_directed_edge(
                points,
                source,
                target,
                neighbor.distance,
                seen,
                edges,
                cursor,
            );
            write_directed_edge(
                points,
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
    points: usize,
    source: usize,
    target: usize,
    seen: &mut [bool],
    offsets: &mut [u32],
) -> Result<(), LeafBuildError> {
    let seen = &mut seen[source * points + target];
    if !*seen {
        *seen = true;
        offsets[source + 1] = offsets[source + 1]
            .checked_add(1)
            .ok_or(LeafBuildError::TooManyEdges { leaf })?;
    }
    Ok(())
}

fn write_directed_edge(
    points: usize,
    source: usize,
    target: usize,
    distance: f32,
    seen: &mut [bool],
    edges: &mut [(u32, f32)],
    cursor: &mut [u32],
) {
    let seen = &mut seen[source * points + target];
    if !*seen {
        *seen = true;
        let position = cursor[source] as usize;
        edges[position] = (target as u32, distance);
        cursor[source] += 1;
    }
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

fn poisoned_row(point: u32) -> LeafBuildError {
    LeafBuildError::PoisonedCandidateRow { point }
}

#[cfg(test)]
mod tests;
