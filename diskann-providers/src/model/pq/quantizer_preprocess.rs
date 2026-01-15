/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::distance::Metric;

use super::fixed_chunk_pq_table::compute_pq_distance;
use crate::{
    model::{PQData, PQScratch, pq::pq_dataset::PQTable},
    utils::BridgeErr,
};

/// Preprocesses the query vector for PQ distance calculations.
/// This function rotates the query vector and prepares the PQ table distances
/// for efficient computation during search operations.
pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    match &pq_data.pq_table() {
        PQTable::Transposed(table) => {
            let dim = table.dim();
            let expected_len = table.ncenters() * table.nchunks();
            let dst = diskann_utils::views::MutMatrixView::try_from(
                &mut pq_scratch.aligned_pqtable_dist_scratch.as_mut_slice()[..expected_len],
                table.nchunks(),
                table.ncenters(),
            )
            .bridge_err()?;

            match metric {
                // Prior to the introduction of the `quantizer_preprocess` method, the
                // disk index was hard-coded to use L2 distance for comparisons.
                //
                // We're keeping that behavior here - treating `Cosine` and `CosineNormalized`
                // as L2 until a more thorough evaluation can be made.
                Metric::L2 | Metric::Cosine | Metric::CosineNormalized => {
                    table.process_into::<diskann_quantization::distances::SquaredL2>(
                        &pq_scratch.rotated_query[..dim],
                        dst,
                    );
                }
                Metric::InnerProduct => {
                    table.process_into::<diskann_quantization::distances::InnerProduct>(
                        &pq_scratch.rotated_query[..dim],
                        dst,
                    );
                }
            }
        }
        PQTable::Fixed(table) => {
            match metric {
                // Prior to the introduction of the `quantizer_preprocess` method, the
                // disk index was hard-coded to use L2 distance for comparisons.
                //
                // We're keeping that behavior here - treating `Cosine` and `CosineNormalized`
                // as L2 until a more thorough evaluation can be made.
                Metric::L2 | Metric::Cosine | Metric::CosineNormalized => {
                    // The scratch only stores the aligned dimension. However, preprocessing
                    // wants the actual dimension used, so we have to shrink the rotated query
                    // accordingly.
                    let dim = table.get_dim();
                    table.preprocess_query(&mut pq_scratch.rotated_query[..dim]);

                    // Compute the distance between each chunk of the query to each pq centroids.
                    table.populate_chunk_distances(
                        pq_scratch.rotated_query.as_slice(),
                        pq_scratch.aligned_pqtable_dist_scratch.as_mut_slice(),
                    )?;
                }
                Metric::InnerProduct => {
                    table.populate_chunk_inner_products(
                        pq_scratch.rotated_query.as_slice(),
                        pq_scratch.aligned_pqtable_dist_scratch.as_mut_slice(),
                    )?;
                }
            }
        }
    }

    // Compute the pq distance between query vector to all the vertex in the pq
    // calculation id scratch.
    compute_pq_distance(
        id_to_calculate_pq_distance,
        pq_data.get_num_chunks(),
        &pq_scratch.aligned_pqtable_dist_scratch,
        pq_data.pq_compressed_data().get_data(),
        &mut pq_scratch.aligned_pq_coord_scratch,
        &mut pq_scratch.aligned_dist_scratch,
    )?;

    Ok(())
}
