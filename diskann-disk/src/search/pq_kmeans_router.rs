/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    path::Path,
};

use diskann::{ANNError, ANNResult};
use diskann_providers::model::{pq::pq_dist_lookup_single, FixedChunkPQTable};
use serde::{Deserialize, Serialize};

use crate::storage::quant::pq::PQData;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PqKmeansRouterData {
    pub num_points: usize,
    pub num_pq_chunks: usize,
    pub representative_ids: Vec<u32>,
    pub representative_codes: Vec<u8>,
    pub fallback_medoid: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct PqKmeansRouterBuildParams {
    pub num_representatives: Option<usize>,
    pub training_sample_size: Option<usize>,
    pub max_iterations: usize,
}

impl Default for PqKmeansRouterBuildParams {
    fn default() -> Self {
        Self {
            num_representatives: None,
            training_sample_size: None,
            max_iterations: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PqKmeansRouteResult {
    pub start_points: Vec<u32>,
    pub scanned_codes: usize,
}

#[derive(Debug, Clone)]
pub struct PqKmeansStartPointRouter {
    data: PqKmeansRouterData,
    max_start_points: usize,
}

impl PqKmeansRouterData {
    pub fn default_num_representatives(num_points: usize) -> usize {
        if num_points == 0 {
            return 0;
        }
        let root = (num_points as f64).sqrt() as usize;
        if root * root == num_points {
            root
        } else {
            root + 1
        }
    }

    pub fn build_from_pq_data(
        pq_data: &PQData,
        params: PqKmeansRouterBuildParams,
        fallback_medoid: Option<u32>,
    ) -> ANNResult<Self> {
        let compressed = pq_data.pq_compressed_data();
        let num_points = compressed.nrows();
        let num_pq_chunks = pq_data.get_num_chunks();
        if num_points == 0 {
            return Err(ANNError::log_index_error(
                "cannot build PQ-kmeans start-point router for an empty PQ dataset",
            ));
        }
        if num_pq_chunks == 0 {
            return Err(ANNError::log_index_error(
                "cannot build PQ-kmeans start-point router with zero PQ chunks",
            ));
        }

        let requested_k = params
            .num_representatives
            .unwrap_or_else(|| Self::default_num_representatives(num_points));
        let k = requested_k.clamp(1, num_points);
        let default_sample_size = default_training_sample_size(k);
        let sample_size = params
            .training_sample_size
            .unwrap_or(default_sample_size)
            .clamp(k, num_points);
        let sample_ids = evenly_spaced_sample_ids(num_points, sample_size);
        let pq_table = pq_data.pq_geometry_table();
        let dim = pq_data.get_dim();

        let mut centroids = Vec::with_capacity(k * dim);
        for centroid_idx in 0..k {
            let sample_id = sample_ids[centroid_idx * sample_ids.len() / k];
            let mut centroid = vec![0.0; dim];
            pq_table.inflate_vector_into(pq_data.get_compressed_vector(sample_id)?, &mut centroid);
            centroids.extend_from_slice(&centroid);
        }

        let mut counts = vec![0usize; k];
        let mut sums = vec![0.0f32; k * dim];
        let mut reconstructed = vec![0.0f32; dim];
        for _ in 0..params.max_iterations {
            counts.fill(0);
            sums.fill(0.0);

            for sample_id in &sample_ids {
                let code = pq_data.get_compressed_vector(*sample_id)?;
                let centroid = nearest_reconstructed_centroid(code, pq_table, &centroids, dim);
                counts[centroid] += 1;
                pq_table.inflate_vector_into(code, &mut reconstructed);
                let sum_base = centroid * dim;
                for (dimension, value) in reconstructed.iter().enumerate() {
                    sums[sum_base + dimension] += *value;
                }
            }

            for (centroid, count) in counts.iter().copied().enumerate().take(k) {
                if count == 0 {
                    continue;
                }
                let base = centroid * dim;
                let count = count as f32;
                for dimension in 0..dim {
                    centroids[base + dimension] = sums[base + dimension] / count;
                }
            }
        }

        let mut representative_ids = Vec::with_capacity(k);
        let mut representative_codes = Vec::with_capacity(k * num_pq_chunks);
        let mut used = HashSet::with_capacity(k);
        for centroid in 0..k {
            if let Some(sample_id) = nearest_sample_to_centroid(
                pq_data,
                &sample_ids,
                &centroids[centroid * dim..(centroid + 1) * dim],
                &used,
            )? {
                used.insert(sample_id);
                representative_ids.push(sample_id as u32);
                representative_codes.extend_from_slice(pq_data.get_compressed_vector(sample_id)?);
            }
        }

        if representative_ids.len() < k {
            for sample_id in sample_ids {
                if representative_ids.len() >= k {
                    break;
                }
                if used.insert(sample_id) {
                    representative_ids.push(sample_id as u32);
                    representative_codes
                        .extend_from_slice(pq_data.get_compressed_vector(sample_id)?);
                }
            }
        }

        let data = Self {
            num_points,
            num_pq_chunks,
            representative_ids,
            representative_codes,
            fallback_medoid,
        };
        data.validate()?;
        Ok(data)
    }

    pub fn validate(&self) -> ANNResult<()> {
        if self.num_points == 0 {
            return Err(ANNError::log_index_error(
                "PQ-kmeans router data must contain at least one point",
            ));
        }
        if self.num_pq_chunks == 0 {
            return Err(ANNError::log_index_error(
                "PQ-kmeans router data must contain at least one PQ chunk",
            ));
        }
        if self.representative_codes.len() != self.representative_ids.len() * self.num_pq_chunks {
            return Err(ANNError::log_index_error(format!(
                "PQ-kmeans router representative code length {} does not match ids {} × chunks {}",
                self.representative_codes.len(),
                self.representative_ids.len(),
                self.num_pq_chunks
            )));
        }
        if self.representative_ids.is_empty() && self.fallback_medoid.is_none() {
            return Err(ANNError::log_index_error(
                "PQ-kmeans router data needs representatives or a fallback medoid",
            ));
        }
        for id in &self.representative_ids {
            if *id as usize >= self.num_points {
                return Err(ANNError::log_index_error(format!(
                    "PQ-kmeans router representative id {} is out of bounds for {} points",
                    id, self.num_points
                )));
            }
        }
        if let Some(fallback_medoid) = self.fallback_medoid {
            if fallback_medoid as usize >= self.num_points {
                return Err(ANNError::log_index_error(format!(
                    "PQ-kmeans router fallback medoid {} is out of bounds for {} points",
                    fallback_medoid, self.num_points
                )));
            }
        }
        Ok(())
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> ANNResult<()> {
        self.validate()?;
        let bytes = bincode::serialize(self).map_err(ANNError::log_index_error)?;
        std::fs::write(path, bytes).map_err(ANNError::log_index_error)
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> ANNResult<Self> {
        let bytes = std::fs::read(path).map_err(ANNError::log_index_error)?;
        let data: Self = bincode::deserialize(&bytes).map_err(ANNError::log_index_error)?;
        data.validate()?;
        Ok(data)
    }
}

impl PqKmeansStartPointRouter {
    pub fn new(data: PqKmeansRouterData, max_start_points: usize) -> ANNResult<Self> {
        if max_start_points == 0 {
            return Err(ANNError::log_index_error(
                "max_start_points must be greater than 0 for PQ-kmeans start-point router",
            ));
        }
        data.validate()?;
        Ok(Self {
            data,
            max_start_points,
        })
    }

    pub fn data(&self) -> &PqKmeansRouterData {
        &self.data
    }

    pub fn max_start_points(&self) -> usize {
        self.max_start_points
    }

    pub fn route(&self, query: &[f32], pq_data: &PQData) -> ANNResult<PqKmeansRouteResult> {
        if pq_data.pq_compressed_data().nrows() != self.data.num_points {
            return Err(ANNError::log_index_error(format!(
                "PQ-kmeans router artifact has {} points but PQ data contains {} points",
                self.data.num_points,
                pq_data.pq_compressed_data().nrows()
            )));
        }
        if pq_data.get_num_chunks() != self.data.num_pq_chunks {
            return Err(ANNError::log_index_error(format!(
                "PQ-kmeans router chunk count {} does not match PQ data chunk count {}",
                self.data.num_pq_chunks,
                pq_data.get_num_chunks()
            )));
        }
        if query.len() != pq_data.get_dim() {
            return Err(ANNError::log_pq_error(format!(
                "PQ-kmeans router query has dimension {} but PQ table expects {}",
                query.len(),
                pq_data.get_dim()
            )));
        }

        let pq_table = pq_data.pq_geometry_table();
        let mut query_distances =
            vec![0.0f32; pq_table.get_num_chunks() * pq_table.get_num_centers()];
        pq_table.populate_chunk_distances(query, query_distances.as_mut_slice())?;

        let mut scored = BinaryHeap::with_capacity(self.max_start_points);
        for (idx, representative_id) in self.data.representative_ids.iter().enumerate() {
            let code = self.representative_code(idx);
            let candidate = ScoredRepresentative::new(
                pq_dist_lookup_single(code, query_distances.as_slice(), pq_table.get_num_centers()),
                *representative_id,
            );
            if scored.len() < self.max_start_points {
                scored.push(candidate);
            } else if scored.peek().is_some_and(|worst| candidate < *worst) {
                scored.pop();
                scored.push(candidate);
            }
        }
        let mut scored = scored.into_vec();
        scored.sort_unstable();

        let mut seen = HashSet::with_capacity(self.max_start_points);
        let mut start_points = Vec::with_capacity(self.max_start_points);
        for candidate in scored {
            if seen.insert(candidate.id) {
                start_points.push(candidate.id);
                if start_points.len() >= self.max_start_points {
                    break;
                }
            }
        }

        if start_points.len() < self.max_start_points {
            if let Some(fallback_medoid) = self.data.fallback_medoid {
                if seen.insert(fallback_medoid) {
                    start_points.push(fallback_medoid);
                }
            }
        }

        Ok(PqKmeansRouteResult {
            start_points,
            scanned_codes: self.data.representative_ids.len(),
        })
    }

    fn representative_code(&self, idx: usize) -> &[u8] {
        let base = idx * self.data.num_pq_chunks;
        &self.data.representative_codes[base..base + self.data.num_pq_chunks]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScoredRepresentative {
    distance: f32,
    id: u32,
}

impl ScoredRepresentative {
    fn new(distance: f32, id: u32) -> Self {
        Self { distance, id }
    }
}

impl Eq for ScoredRepresentative {}

impl Ord for ScoredRepresentative {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for ScoredRepresentative {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn evenly_spaced_sample_ids(num_points: usize, sample_size: usize) -> Vec<usize> {
    if sample_size >= num_points {
        return (0..num_points).collect();
    }
    (0..sample_size)
        .map(|i| i * num_points / sample_size)
        .collect()
}

fn default_training_sample_size(k: usize) -> usize {
    if k >= 16_384 {
        k
    } else {
        k.saturating_mul(2).min(16_384)
    }
}

fn nearest_reconstructed_centroid(
    code: &[u8],
    pq_table: &FixedChunkPQTable,
    centroids: &[f32],
    dim: usize,
) -> usize {
    centroids
        .chunks_exact(dim)
        .enumerate()
        .min_by(|(left_idx, left), (right_idx, right)| {
            let left_distance = pq_table.l2_distance(left, code);
            let right_distance = pq_table.l2_distance(right, code);
            left_distance
                .total_cmp(&right_distance)
                .then_with(|| left_idx.cmp(right_idx))
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn nearest_sample_to_centroid(
    pq_data: &PQData,
    sample_ids: &[usize],
    centroid: &[f32],
    used: &HashSet<usize>,
) -> ANNResult<Option<usize>> {
    let pq_table = pq_data.pq_geometry_table();
    let mut best_unused = None;
    let mut best_any = None;
    for sample_id in sample_ids {
        let code = pq_data.get_compressed_vector(*sample_id)?;
        let distance = pq_table.l2_distance(centroid, code);
        let candidate = (distance, *sample_id);
        if best_any.is_none_or(|best| pq_distance_candidate_lt(candidate, best)) {
            best_any = Some(candidate);
        }
        if !used.contains(sample_id)
            && best_unused.is_none_or(|best| pq_distance_candidate_lt(candidate, best))
        {
            best_unused = Some(candidate);
        }
    }
    Ok(best_unused.or(best_any).map(|(_, sample_id)| sample_id))
}

fn pq_distance_candidate_lt(lhs: (f32, usize), rhs: (f32, usize)) -> bool {
    lhs.0
        .total_cmp(&rhs.0)
        .then_with(|| lhs.1.cmp(&rhs.1))
        .is_lt()
}

#[cfg(test)]
mod tests {
    use diskann_providers::model::FixedChunkPQTable;
    use diskann_utils::views::Matrix;

    use crate::{search::pq_kmeans_router::*, storage::quant::pq::PQData};

    fn one_chunk_pq_data() -> PQData {
        let table =
            FixedChunkPQTable::new(1, Box::new([0.0, 10.0, 20.0, 30.0]), Box::new([0, 1])).unwrap();
        let codes = Matrix::try_from(Box::new([0u8, 1, 2, 3]) as Box<[u8]>, 4, 1).unwrap();
        PQData::new(table, codes).unwrap()
    }

    fn non_ordinal_one_chunk_pq_data(codes: Box<[u8]>) -> PQData {
        let table =
            FixedChunkPQTable::new(1, Box::new([0.0, 100.0, 101.0]), Box::new([0, 1])).unwrap();
        let num_points = codes.len();
        let codes = Matrix::try_from(codes, num_points, 1).unwrap();
        PQData::new(table, codes).unwrap()
    }

    #[test]
    fn route_scans_representative_pq_codes_and_returns_nearest_start_points() {
        let pq_data = one_chunk_pq_data();
        let router = PqKmeansStartPointRouter::new(
            PqKmeansRouterData {
                num_points: 4,
                num_pq_chunks: 1,
                representative_ids: vec![0, 2, 3],
                representative_codes: vec![0, 2, 3],
                fallback_medoid: Some(1),
            },
            2,
        )
        .unwrap();

        let result = router.route(&[18.0], &pq_data).unwrap();

        assert_eq!(result.start_points, vec![2, 3]);
        assert_eq!(result.scanned_codes, 3);
    }

    #[test]
    fn route_scores_representatives_with_adc_geometry_not_label_ids() {
        let pq_data = non_ordinal_one_chunk_pq_data(Box::new([0u8, 1, 2]));
        let router = PqKmeansStartPointRouter::new(
            PqKmeansRouterData {
                num_points: 3,
                num_pq_chunks: 1,
                representative_ids: vec![0, 2],
                representative_codes: vec![0, 2],
                fallback_medoid: Some(1),
            },
            1,
        )
        .unwrap();

        let result = router.route(&[99.0], &pq_data).unwrap();

        assert_eq!(result.start_points, vec![2]);
        assert_eq!(result.scanned_codes, 2);
    }

    #[test]
    fn build_clusters_pq_reconstructed_geometry_not_label_ids() {
        let pq_data = non_ordinal_one_chunk_pq_data(Box::new([0u8, 2, 2]));

        let data = PqKmeansRouterData::build_from_pq_data(
            &pq_data,
            PqKmeansRouterBuildParams {
                num_representatives: Some(1),
                training_sample_size: Some(3),
                max_iterations: 1,
            },
            Some(0),
        )
        .unwrap();

        assert_eq!(data.representative_ids, vec![1]);
        assert_eq!(data.representative_codes, vec![2]);
    }

    #[test]
    fn default_training_sample_size_handles_k_larger_than_cap() {
        assert_eq!(default_training_sample_size(16_385), 16_385);
    }

    #[test]
    fn route_rejects_router_artifact_for_different_num_points() {
        let pq_data = one_chunk_pq_data();
        let router = PqKmeansStartPointRouter::new(
            PqKmeansRouterData {
                num_points: 5,
                num_pq_chunks: 1,
                representative_ids: vec![4],
                representative_codes: vec![0],
                fallback_medoid: None,
            },
            1,
        )
        .unwrap();

        assert!(router.route(&[0.0], &pq_data).is_err());
    }

    #[test]
    fn build_from_pq_data_defaults_to_ceil_sqrt_num_points_representatives() {
        let pq_data = one_chunk_pq_data();
        let data = PqKmeansRouterData::build_from_pq_data(
            &pq_data,
            PqKmeansRouterBuildParams {
                max_iterations: 2,
                training_sample_size: Some(4),
                ..Default::default()
            },
            Some(1),
        )
        .unwrap();

        assert_eq!(data.num_points, 4);
        assert_eq!(data.num_pq_chunks, 1);
        assert_eq!(data.representative_ids.len(), 2);
        assert_eq!(data.representative_codes.len(), 2);
        assert_eq!(data.fallback_medoid, Some(1));
    }
}
