/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::distance::Metric;

use diskann_providers::model::compute_pq_distance;
use diskann_providers::utils::BridgeErr;

use super::{PQData, PQQueryComputer, PQScratch};

/// Preprocesses the query vector for PQ distance calculations.
/// This function rotates the query vector and prepares the PQ table distances
/// for efficient computation during search operations.
fn preprocess_query(
    query: &[f32],
    lookup_table: &mut [f32],
    pq_data: &PQData,
    metric: Metric,
) -> ANNResult<()> {
    let table = pq_data.pq_table();
    let expected_len = table.ncenters() * table.nchunks();
    let dst = diskann_utils::views::MutMatrixView::try_from(
        &mut lookup_table[..expected_len],
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
            table.process_into::<diskann_quantization::distances::SquaredL2>(query, dst);
        }
        Metric::InnerProduct => {
            table.process_into::<diskann_quantization::distances::InnerProduct>(query, dst);
        }
    }

    Ok(())
}

pub(crate) fn prepare_query(
    computer: &mut PQQueryComputer,
    pq_data: &PQData,
    metric: Metric,
    query: &[f32],
) -> ANNResult<()> {
    computer.set(query)?;
    let (query, lookup_table) = computer.preprocessing_buffers();
    preprocess_query(query, lookup_table, pq_data, metric)
}

pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    preprocess_query(
        &pq_scratch.query_scratch,
        &mut pq_scratch.aligned_pqtable_dist_scratch,
        pq_data,
        metric,
    )?;

    compute_pq_distance(
        id_to_calculate_pq_distance,
        pq_data.get_num_chunks(),
        &pq_scratch.aligned_pqtable_dist_scratch,
        pq_data.pq_compressed_data().as_slice(),
        &mut pq_scratch.aligned_pq_coord_scratch,
        &mut pq_scratch.aligned_dist_scratch,
    )?;

    Ok(())
}
