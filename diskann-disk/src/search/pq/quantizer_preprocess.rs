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
    computer: &mut PQQueryComputer,
    pq_data: &PQData,
    metric: Metric,
) -> ANNResult<()> {
    let table = pq_data.pq_table();
    let expected_len = table.ncenters() * table.nchunks();
    let (query, lookup_table) = computer.preprocessing_buffers();
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
    pq_data: &PQData,
    metric: Metric,
    query: &[f32],
) -> ANNResult<PQQueryComputer> {
    let mut computer = PQQueryComputer::new(
        pq_data.get_dim(),
        pq_data.get_num_chunks(),
        pq_data.get_num_centers(),
    )?;
    computer.set(query)?;
    preprocess_query(&mut computer, pq_data, metric)?;
    Ok(computer)
}

pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    preprocess_query(&mut pq_scratch.query_computer, pq_data, metric)?;

    compute_pq_distance(
        id_to_calculate_pq_distance,
        pq_data.get_num_chunks(),
        pq_scratch.query_computer.lookup_table(),
        pq_data.pq_compressed_data().as_slice(),
        &mut pq_scratch.aligned_pq_coord_scratch,
        &mut pq_scratch.aligned_dist_scratch,
    )?;

    Ok(())
}
