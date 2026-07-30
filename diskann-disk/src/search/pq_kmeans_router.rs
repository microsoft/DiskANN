/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::{BinaryHeap, HashSet},
    path::Path,
};

use diskann::{ANNError, ANNResult};
use diskann_quantization::CompressInto;
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
        let default_sample_size = (k.saturating_mul(2)).clamp(k, 16_384);
        let sample_size = params
            .training_sample_size
            .unwrap_or(default_sample_size)
            .clamp(k, num_points);
        let sample_ids = evenly_spaced_sample_ids(num_points, sample_size);

        let mut centroids = Vec::with_capacity(k * num_pq_chunks);
        for centroid_idx in 0..k {
            let sample_id = sample_ids[centroid_idx * sample_ids.len() / k];
            centroids.extend_from_slice(pq_data.get_compressed_vector(sample_id)?);
        }

        let mut counts = vec![0usize; k];
        let mut sums = vec![0u64; k * num_pq_chunks];
        for _ in 0..params.max_iterations {
            counts.fill(0);
            sums.fill(0);

            for sample_id in &sample_ids {
                let code = pq_data.get_compressed_vector(*sample_id)?;
                let centroid = nearest_code(code, &centroids, num_pq_chunks);
                counts[centroid] += 1;
                let sum_base = centroid * num_pq_chunks;
                for (chunk, value) in code.iter().enumerate() {
                    sums[sum_base + chunk] += u64::from(*value);
                }
            }

            for (centroid, count) in counts.iter().copied().enumerate().take(k) {
                if count == 0 {
                    continue;
                }
                let base = centroid * num_pq_chunks;
                for chunk in 0..num_pq_chunks {
                    centroids[base + chunk] =
                        ((sums[base + chunk] + (count as u64 / 2)) / count as u64) as u8;
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
                &centroids[centroid * num_pq_chunks..(centroid + 1) * num_pq_chunks],
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
        if pq_data.get_num_chunks() != self.data.num_pq_chunks {
            return Err(ANNError::log_index_error(format!(
                "PQ-kmeans router chunk count {} does not match PQ data chunk count {}",
                self.data.num_pq_chunks,
                pq_data.get_num_chunks()
            )));
        }

        let mut query_code = vec![0u8; self.data.num_pq_chunks];
        pq_data
            .pq_table()
            .compress_into(query, query_code.as_mut_slice())
            .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;

        let mut scored = BinaryHeap::with_capacity(self.max_start_points);
        for (idx, representative_id) in self.data.representative_ids.iter().enumerate() {
            let code = self.representative_code(idx);
            let candidate = (pq_code_distance(&query_code, code), *representative_id);
            if scored.len() < self.max_start_points {
                scored.push(candidate);
            } else if scored.peek().is_some_and(|worst| candidate < *worst) {
                scored.pop();
                scored.push(candidate);
            }
        }
        let mut scored = scored.into_vec();
        scored.sort_unstable_by_key(|(distance, id)| (*distance, *id));

        let mut seen = HashSet::with_capacity(self.max_start_points);
        let mut start_points = Vec::with_capacity(self.max_start_points);
        for (_, id) in scored {
            if seen.insert(id) {
                start_points.push(id);
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

fn evenly_spaced_sample_ids(num_points: usize, sample_size: usize) -> Vec<usize> {
    if sample_size >= num_points {
        return (0..num_points).collect();
    }
    (0..sample_size)
        .map(|i| i * num_points / sample_size)
        .collect()
}

fn nearest_code(query_code: &[u8], codes: &[u8], num_pq_chunks: usize) -> usize {
    codes
        .chunks_exact(num_pq_chunks)
        .enumerate()
        .min_by_key(|(_, code)| pq_code_distance(query_code, code))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn nearest_sample_to_centroid(
    pq_data: &PQData,
    sample_ids: &[usize],
    centroid_code: &[u8],
    used: &HashSet<usize>,
) -> ANNResult<Option<usize>> {
    let mut best_unused = None;
    let mut best_any = None;
    for sample_id in sample_ids {
        let code = pq_data.get_compressed_vector(*sample_id)?;
        let distance = pq_code_distance(centroid_code, code);
        let candidate = (distance, *sample_id);
        if best_any.is_none_or(|best| candidate < best) {
            best_any = Some(candidate);
        }
        if !used.contains(sample_id) && best_unused.is_none_or(|best| candidate < best) {
            best_unused = Some(candidate);
        }
    }
    Ok(best_unused.or(best_any).map(|(_, sample_id)| sample_id))
}

fn pq_code_distance(lhs: &[u8], rhs: &[u8]) -> u32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs, rhs)| {
            let diff = i32::from(*lhs) - i32::from(*rhs);
            (diff * diff) as u32
        })
        .sum()
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
