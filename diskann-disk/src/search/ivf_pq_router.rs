/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! IVF posting-list router data for query-time PQ start-point selection.
//!
//! The router owns only IVF centroids and posting IDs. Query-time ADC scoring
//! is performed by the disk search provider with the already-loaded global PQ
//! compressed vectors.

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    io::{Read, Write},
    mem::size_of,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{ANNError, ANNResult};
use diskann_providers::utils::{ParallelIteratorInPool, RayonThreadPoolRef};
use rand::{rngs::StdRng, seq::index::sample, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::utils::k_means_clustering;

const BINARY_MAGIC: &[u8; 16] = b"DISKANNIVFPQ0001";
const BINARY_VERSION: u32 = 1;
const NO_FALLBACK_MEDOID: u64 = u64::MAX;

/// Serialized IVF posting-list router data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IvfPqRouterData {
    pub dim: usize,
    pub centroids: Vec<f32>,
    pub offsets: Vec<usize>,
    pub posting_ids: Vec<u32>,
    pub fallback_medoid: Option<u32>,
}

/// Build parameters for an IVF posting-list router artifact.
#[derive(Debug, Clone, Copy)]
pub struct IvfPqRouterBuildParams {
    pub num_centroids: usize,
    pub max_iterations: usize,
    pub seed: u64,
    pub fallback_medoid: Option<u32>,
    pub training_sample_size: Option<usize>,
}

/// In-memory IVF posting-list router.
#[derive(Debug, Clone)]
pub struct IvfPqRouter {
    dim: usize,
    centroids: Vec<f32>,
    offsets: Vec<usize>,
    posting_ids: Vec<u32>,
    fallback_medoid: Option<u32>,
}

/// IVF cells selected for one query plus the number of scored centroids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbedCells {
    pub cells: Vec<usize>,
    pub centroid_scores: usize,
}

impl IvfPqRouterData {
    /// Validate the artifact shape.
    pub fn validate(&self) -> ANNResult<()> {
        validate_layout(
            self.dim,
            &self.centroids,
            &self.offsets,
            self.posting_ids.len(),
        )
    }
}

impl IvfPqRouter {
    /// Construct a router from serialized fields.
    pub fn new(
        dim: usize,
        centroids: Vec<f32>,
        offsets: Vec<usize>,
        posting_ids: Vec<u32>,
        fallback_medoid: Option<u32>,
    ) -> ANNResult<Self> {
        validate_layout(dim, &centroids, &offsets, posting_ids.len())?;
        Ok(Self {
            dim,
            centroids,
            offsets,
            posting_ids,
            fallback_medoid,
        })
    }

    /// Construct a router from serialized data.
    pub fn from_data(data: IvfPqRouterData) -> ANNResult<Self> {
        Self::new(
            data.dim,
            data.centroids,
            data.offsets,
            data.posting_ids,
            data.fallback_medoid,
        )
    }

    /// Export the router as serializable data.
    pub fn to_data(&self) -> IvfPqRouterData {
        IvfPqRouterData {
            dim: self.dim,
            centroids: self.centroids.clone(),
            offsets: self.offsets.clone(),
            posting_ids: self.posting_ids.clone(),
            fallback_medoid: self.fallback_medoid,
        }
    }

    /// Return the vector dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Return the number of IVF centroids.
    pub fn num_centroids(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Return the number of indexed postings.
    pub fn num_points(&self) -> usize {
        self.posting_ids.len()
    }

    /// Return the fallback medoid embedded in the artifact, if present.
    pub fn fallback_medoid(&self) -> Option<u32> {
        self.fallback_medoid
    }

    /// Return posting IDs for one IVF cell.
    pub fn posting_ids(&self, cell: usize) -> &[u32] {
        let start = self.offsets[cell];
        let end = self.offsets[cell + 1];
        &self.posting_ids[start..end]
    }

    /// Return the centroid vector for one IVF cell.
    pub fn centroid(&self, cell: usize) -> &[f32] {
        &self.centroids[cell * self.dim..(cell + 1) * self.dim]
    }

    /// Return probed cells for a query.
    pub fn probe_cells_with_stats(&self, query: &[f32], nprobe: usize) -> ANNResult<ProbedCells> {
        if query.len() != self.dim {
            return Err(index_error(format!(
                "query dimension {} does not match IVF+PQ router dimension {}",
                query.len(),
                self.dim
            )));
        }
        if nprobe == 0 {
            return Err(index_error("nprobe must be positive"));
        }

        let cells = select_nearest_centroids(
            (0..self.num_centroids()).map(|cell| (cell, squared_l2(query, self.centroid(cell)))),
            nprobe,
        );

        Ok(ProbedCells {
            cells,
            centroid_scores: self.num_centroids(),
        })
    }

    /// Return resident heap bytes, excluding allocator overhead.
    pub fn memory_bytes(&self) -> usize {
        self.centroids.len() * size_of::<f32>()
            + self.offsets.len() * size_of::<usize>()
            + self.posting_ids.len() * size_of::<u32>()
    }
}

/// Query-time routing parameters around an IVF posting-list router.
#[derive(Debug, Clone)]
pub struct IvfPqStartPointRouter {
    router: IvfPqRouter,
    nprobe: usize,
    max_start_points: usize,
    posting_list_samples_per_list: usize,
}

impl IvfPqStartPointRouter {
    /// Construct query-time routing parameters.
    pub fn new(
        router: IvfPqRouter,
        nprobe: usize,
        max_start_points: usize,
        posting_list_samples_per_list: usize,
    ) -> ANNResult<Self> {
        if nprobe == 0 {
            return Err(index_error("nprobe must be positive"));
        }
        if max_start_points == 0 {
            return Err(index_error("max_start_points must be positive"));
        }
        if posting_list_samples_per_list == 0 {
            return Err(index_error(
                "posting_list_samples_per_list must be positive",
            ));
        }
        Ok(Self {
            router,
            nprobe,
            max_start_points,
            posting_list_samples_per_list,
        })
    }

    /// Return probed cells and centroid-score count for a query.
    pub fn probe_cells_with_stats(&self, query: &[f32]) -> ANNResult<ProbedCells> {
        self.router.probe_cells_with_stats(query, self.nprobe)
    }

    /// Return posting IDs for one cell.
    pub fn posting_ids(&self, cell: usize) -> &[u32] {
        self.router.posting_ids(cell)
    }

    /// Return configured start-point cap.
    pub fn max_start_points(&self) -> usize {
        self.max_start_points
    }

    /// Return configured sample count per probed posting list.
    pub fn posting_list_samples_per_list(&self) -> usize {
        self.posting_list_samples_per_list
    }

    /// Return number of indexed postings.
    pub fn num_points(&self) -> usize {
        self.router.num_points()
    }

    /// Return fallback medoid from the artifact, if present.
    pub fn fallback_medoid(&self) -> Option<u32> {
        self.router.fallback_medoid()
    }

    /// Return resident heap bytes, excluding allocator overhead.
    pub fn memory_bytes(&self) -> usize {
        self.router.memory_bytes()
    }
}

/// Build an IVF posting-list router artifact from full-precision f32 data.
pub fn build_ivf_pq_router_data(
    data: &[f32],
    num_points: usize,
    dim: usize,
    params: &IvfPqRouterBuildParams,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<IvfPqRouterData> {
    validate_build_inputs(data, num_points, dim, params)?;

    let centroid_values = params
        .num_centroids
        .checked_mul(dim)
        .ok_or_else(|| index_error("num_centroids multiplied by dim overflowed"))?;
    let mut centroids = vec![0.0; centroid_values];
    let mut rng = StdRng::seed_from_u64(params.seed);
    let mut cancellation_token = false;
    let training_sample = sample_training_data(data, num_points, dim, params, &mut rng)?;
    let training_data = training_sample.as_deref().unwrap_or(data);
    let training_points = params
        .training_sample_size
        .unwrap_or(num_points)
        .min(num_points);

    let (_closest_docs, _closest_center, _residual) = k_means_clustering(
        training_data,
        training_points,
        dim,
        &mut centroids,
        params.num_centroids,
        params.max_iterations,
        &mut rng,
        &mut cancellation_token,
        pool,
    )?;

    let closest_docs = assign_points_to_centroids(data, num_points, dim, &centroids, pool);
    let (offsets, posting_ids) = posting_lists_to_layout(closest_docs, num_points)?;

    Ok(IvfPqRouterData {
        dim,
        centroids,
        offsets,
        posting_ids,
        fallback_medoid: params.fallback_medoid,
    })
}

/// Write a binary IVF+PQ router artifact.
pub fn write_ivf_pq_router_binary<W: Write>(
    mut writer: W,
    data: &IvfPqRouterData,
) -> ANNResult<()> {
    data.validate()?;
    writer.write_all(BINARY_MAGIC)?;
    writer.write_u32::<LittleEndian>(BINARY_VERSION)?;
    writer.write_u64::<LittleEndian>(data.dim as u64)?;
    writer.write_u64::<LittleEndian>((data.centroids.len() / data.dim) as u64)?;
    writer.write_u64::<LittleEndian>(data.posting_ids.len() as u64)?;
    writer.write_u64::<LittleEndian>(data.fallback_medoid.map_or(NO_FALLBACK_MEDOID, u64::from))?;

    for &value in &data.centroids {
        writer.write_f32::<LittleEndian>(value)?;
    }
    for &offset in &data.offsets {
        writer.write_u64::<LittleEndian>(offset as u64)?;
    }
    for &id in &data.posting_ids {
        writer.write_u32::<LittleEndian>(id)?;
    }
    Ok(())
}

/// Read a binary IVF+PQ router artifact.
pub fn read_ivf_pq_router_binary<R: Read>(mut reader: R) -> ANNResult<IvfPqRouterData> {
    let mut magic = [0u8; 16];
    reader.read_exact(&mut magic)?;
    if magic != *BINARY_MAGIC {
        return Err(index_error("invalid IVF+PQ router binary magic"));
    }
    let version = reader.read_u32::<LittleEndian>()?;
    if version != BINARY_VERSION {
        return Err(index_error(format!(
            "unsupported IVF+PQ router binary version {version}"
        )));
    }
    let dim = usize::try_from(reader.read_u64::<LittleEndian>()?)?;
    let num_centroids = usize::try_from(reader.read_u64::<LittleEndian>()?)?;
    let num_postings = usize::try_from(reader.read_u64::<LittleEndian>()?)?;
    let fallback_raw = reader.read_u64::<LittleEndian>()?;
    let fallback_medoid = if fallback_raw == NO_FALLBACK_MEDOID {
        None
    } else {
        Some(u32::try_from(fallback_raw)?)
    };

    let mut centroids = vec![0.0; num_centroids * dim];
    for value in &mut centroids {
        *value = reader.read_f32::<LittleEndian>()?;
    }
    let mut offsets = vec![0usize; num_centroids + 1];
    for offset in &mut offsets {
        *offset = usize::try_from(reader.read_u64::<LittleEndian>()?)?;
    }
    let mut posting_ids = vec![0u32; num_postings];
    for id in &mut posting_ids {
        *id = reader.read_u32::<LittleEndian>()?;
    }

    let data = IvfPqRouterData {
        dim,
        centroids,
        offsets,
        posting_ids,
        fallback_medoid,
    };
    data.validate()?;
    Ok(data)
}

fn validate_build_inputs(
    data: &[f32],
    num_points: usize,
    dim: usize,
    params: &IvfPqRouterBuildParams,
) -> ANNResult<()> {
    if dim == 0 {
        return Err(index_error("dim must be positive"));
    }
    if num_points == 0 {
        return Err(index_error("num_points must be positive"));
    }
    if data.len() != num_points * dim {
        return Err(index_error("data length must equal num_points * dim"));
    }
    if params.num_centroids == 0 {
        return Err(index_error("num_centroids must be positive"));
    }
    if params.num_centroids > num_points {
        return Err(index_error("num_centroids must not exceed num_points"));
    }
    if params.max_iterations == 0 {
        return Err(index_error("max_iterations must be positive"));
    }
    if let Some(sample_size) = params.training_sample_size {
        if sample_size == 0 {
            return Err(index_error("training_sample_size must be positive"));
        }
        if sample_size < params.num_centroids {
            return Err(index_error(
                "training_sample_size must be at least num_centroids",
            ));
        }
        if sample_size > num_points {
            return Err(index_error(
                "training_sample_size must not exceed num_points",
            ));
        }
    }
    Ok(())
}

fn validate_layout(
    dim: usize,
    centroids: &[f32],
    offsets: &[usize],
    posting_count: usize,
) -> ANNResult<()> {
    if dim == 0 {
        return Err(index_error("IVF+PQ router dim must be positive"));
    }
    if centroids.is_empty() || !centroids.len().is_multiple_of(dim) {
        return Err(index_error(
            "IVF+PQ router centroids length must be a positive multiple of dim",
        ));
    }
    let num_centroids = centroids.len() / dim;
    if offsets.len() != num_centroids + 1 {
        return Err(index_error(format!(
            "IVF+PQ router offsets length {} must equal num_centroids + 1 ({})",
            offsets.len(),
            num_centroids + 1
        )));
    }
    if offsets.first().copied() != Some(0) {
        return Err(index_error("IVF+PQ router first offset must be zero"));
    }
    if offsets.last().copied() != Some(posting_count) {
        return Err(index_error(
            "IVF+PQ router last offset must equal posting count",
        ));
    }
    if !offsets.windows(2).all(|pair| pair[0] <= pair[1]) {
        return Err(index_error("IVF+PQ router offsets must be sorted"));
    }
    Ok(())
}

fn sample_training_data(
    data: &[f32],
    num_points: usize,
    dim: usize,
    params: &IvfPqRouterBuildParams,
    rng: &mut StdRng,
) -> ANNResult<Option<Vec<f32>>> {
    let Some(sample_size) = params.training_sample_size else {
        return Ok(None);
    };
    if sample_size == num_points {
        return Ok(None);
    }

    let mut sample_indices = sample(rng, num_points, sample_size).into_vec();
    sample_indices.sort_unstable();

    let sample_values = sample_size
        .checked_mul(dim)
        .ok_or_else(|| index_error("training_sample_size multiplied by dim overflowed"))?;
    let mut sampled = Vec::with_capacity(sample_values);
    for point in sample_indices {
        sampled.extend_from_slice(&data[point * dim..(point + 1) * dim]);
    }
    Ok(Some(sampled))
}

fn assign_points_to_centroids(
    data: &[f32],
    num_points: usize,
    dim: usize,
    centroids: &[f32],
    pool: RayonThreadPoolRef<'_>,
) -> Vec<Vec<usize>> {
    let num_centroids = centroids.len() / dim;
    let closest_center: Vec<usize> = (0..num_points)
        .into_par_iter()
        .map(|point| nearest_centroid(&data[point * dim..(point + 1) * dim], centroids, dim))
        .collect_in_pool(pool);

    let mut closest_docs = vec![Vec::new(); num_centroids];
    for (doc, center) in closest_center.into_iter().enumerate() {
        closest_docs[center].push(doc);
    }
    closest_docs
}

fn posting_lists_to_layout(
    closest_docs: Vec<Vec<usize>>,
    num_points: usize,
) -> ANNResult<(Vec<usize>, Vec<u32>)> {
    let mut offsets = Vec::with_capacity(closest_docs.len() + 1);
    let mut posting_ids = Vec::with_capacity(num_points);
    offsets.push(0);
    for docs in closest_docs {
        for doc in docs {
            posting_ids
                .push(u32::try_from(doc).map_err(|_| index_error("posting ID must fit into u32"))?);
        }
        offsets.push(posting_ids.len());
    }
    Ok((offsets, posting_ids))
}

fn nearest_centroid(point: &[f32], centroids: &[f32], dim: usize) -> usize {
    let num_centroids = centroids.len() / dim;
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for cell in 0..num_centroids {
        let distance = squared_l2(point, &centroids[cell * dim..(cell + 1) * dim]);
        if distance < best_distance {
            best_distance = distance;
            best = cell;
        }
    }
    best
}

fn squared_l2(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = a - b;
            delta * delta
        })
        .sum()
}

fn select_nearest_centroids<I>(scores: I, limit: usize) -> Vec<usize>
where
    I: IntoIterator<Item = (usize, f32)>,
{
    let mut heap = BinaryHeap::with_capacity(limit);
    for (cell, distance) in scores {
        let candidate = CentroidProbe { distance, cell };
        if heap.len() < limit {
            heap.push(candidate);
        } else if let Some(worst) = heap.peek() {
            if candidate < *worst {
                heap.pop();
                heap.push(candidate);
            }
        }
    }

    let mut selected: Vec<_> = heap.into_iter().collect();
    selected.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance).then(a.cell.cmp(&b.cell)));
    selected.into_iter().map(|probe| probe.cell).collect()
}

#[derive(Debug, Clone, Copy)]
struct CentroidProbe {
    distance: f32,
    cell: usize,
}

impl PartialEq for CentroidProbe {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for CentroidProbe {}

impl PartialOrd for CentroidProbe {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CentroidProbe {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then(self.cell.cmp(&other.cell))
    }
}

fn index_error(message: impl std::fmt::Display) -> ANNError {
    ANNError::log_index_error(message.to_string())
}

#[cfg(test)]
mod tests {
    use diskann_providers::utils::create_thread_pool;

    use super::*;

    #[test]
    fn probes_nearest_centroids() {
        let router = IvfPqRouter::new(
            2,
            vec![0.0, 0.0, 10.0, 10.0, 2.0, 2.0],
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            Some(0),
        )
        .unwrap();

        let probed = router.probe_cells_with_stats(&[1.9, 2.1], 2).unwrap();

        assert_eq!(probed.cells, vec![2, 0]);
        assert_eq!(probed.centroid_scores, 3);
    }

    #[test]
    fn binary_round_trip_preserves_router_data() {
        let data = IvfPqRouterData {
            dim: 2,
            centroids: vec![0.0, 0.0, 1.0, 1.0],
            offsets: vec![0, 2, 3],
            posting_ids: vec![10, 11, 12],
            fallback_medoid: Some(10),
        };

        let mut bytes = Vec::new();
        write_ivf_pq_router_binary(&mut bytes, &data).unwrap();
        let decoded = read_ivf_pq_router_binary(bytes.as_slice()).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn build_ivf_pq_router_assigns_all_points() {
        let data = [0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
        let params = IvfPqRouterBuildParams {
            num_centroids: 2,
            max_iterations: 2,
            seed: 7,
            fallback_medoid: None,
            training_sample_size: None,
        };
        let pool = create_thread_pool(1).unwrap();

        let artifact = build_ivf_pq_router_data(&data, 4, 2, &params, pool.as_ref()).unwrap();

        assert_eq!(artifact.posting_ids.len(), 4);
        assert_eq!(artifact.offsets.first(), Some(&0));
        assert_eq!(artifact.offsets.last(), Some(&4));
    }
}
