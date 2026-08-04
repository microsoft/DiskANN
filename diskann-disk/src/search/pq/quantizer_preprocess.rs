/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::distance::Metric;

use diskann_providers::model::{compute_pq_distance, FixedChunkPQTable};
use diskann_providers::utils::BridgeErr;

use super::{PQData, PQScratch};

/// Preprocesses the query vector for PQ distance calculations.
/// This function rotates the query vector and prepares the PQ table distances
/// for efficient computation during search operations.
pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    let table = pq_data.pq_table();
    let expected_len = table.ncenters() * table.nchunks();
    let dst = diskann_utils::views::MutMatrixView::try_from(
        &mut (*pq_scratch.aligned_pqtable_dist_scratch)[..expected_len],
        table.nchunks(),
        table.ncenters(),
    )
    .bridge_err()?;

    match metric {
        Metric::L2 => {
            table.process_into::<diskann_quantization::distances::SquaredL2>(
                &pq_scratch.query_scratch,
                dst,
            );
        }
        Metric::InnerProduct => {
            table.process_into::<diskann_quantization::distances::InnerProduct>(
                &pq_scratch.query_scratch,
                dst,
            );
        }
        Metric::Cosine | Metric::CosineNormalized => {}
    }

    compute_pq_distances_for_metric(pq_scratch, pq_data, metric, id_to_calculate_pq_distance)?;

    Ok(())
}

pub(crate) fn compute_pq_distances_for_metric(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    ids: &[u32],
) -> ANNResult<()> {
    match metric {
        Metric::L2 | Metric::InnerProduct => compute_pq_distance(
            ids,
            pq_data.get_num_chunks(),
            &pq_scratch.aligned_pqtable_dist_scratch,
            pq_data.pq_compressed_data().as_slice(),
            &mut pq_scratch.aligned_pq_coord_scratch,
            &mut pq_scratch.aligned_dist_scratch,
        ),
        Metric::Cosine => compute_direct_pq_distances(
            pq_scratch,
            pq_data,
            FixedChunkPQTable::cosine_distance,
            ids,
        ),
        Metric::CosineNormalized => compute_direct_pq_distances(
            pq_scratch,
            pq_data,
            FixedChunkPQTable::cosine_normalized_distance,
            ids,
        ),
    }
}

fn compute_direct_pq_distances(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    distance: fn(&FixedChunkPQTable, &[f32], &[u8]) -> f32,
    ids: &[u32],
) -> ANNResult<()> {
    let scratch_len = pq_scratch.aligned_dist_scratch.len();
    let dists_out = pq_scratch
        .aligned_dist_scratch
        .get_mut(..ids.len())
        .ok_or_else(|| {
            diskann::ANNError::log_pq_error(format!(
                "ERROR: dists_out length: {} is less than n_pts: {}",
                scratch_len,
                ids.len()
            ))
        })?;
    let pq_table = pq_data.pq_geometry_table();
    for (out, id) in dists_out.iter_mut().zip(ids) {
        let code = pq_data.get_compressed_vector(*id as usize)?;
        *out = distance(pq_table, &pq_scratch.query_scratch, code);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use diskann_providers::model::FixedChunkPQTable;
    use diskann_utils::views::Matrix;
    use diskann_vector::distance::Metric;

    use super::*;

    fn two_dim_cosine_pq_data() -> PQData {
        let mut pivots = vec![0.0f32; 256 * 2];
        pivots[0] = 10.0;
        pivots[1] = 0.0;
        pivots[2] = 0.9;
        pivots[3] = 0.1;
        for center in 2..256 {
            pivots[center * 2] = 0.0;
            pivots[center * 2 + 1] = 1.0;
        }

        let table = FixedChunkPQTable::new(2, pivots.into_boxed_slice(), Box::new([0, 2])).unwrap();
        let codes = Matrix::try_from(Box::new([0u8, 1]) as Box<[u8]>, 2, 1).unwrap();
        PQData::new(table, codes).unwrap()
    }

    #[test]
    fn quantizer_preprocess_uses_cosine_metric_for_pq_distances() {
        let pq_data = two_dim_cosine_pq_data();
        let mut scratch =
            PQScratch::new(2, pq_data.get_dim(), pq_data.get_num_chunks(), 256).unwrap();
        scratch.set(&[1.0, 0.0]).unwrap();

        quantizer_preprocess(&mut scratch, &pq_data, Metric::Cosine, &[0, 1]).unwrap();

        assert!(scratch.aligned_dist_scratch[0].abs() < 1.0e-6);
        assert!((scratch.aligned_dist_scratch[1] - 0.006_116_271).abs() < 1.0e-6);
    }
}
