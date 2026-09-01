/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::distance::Metric;

use diskann_providers::model::compute_pq_distance;
use diskann_providers::utils::BridgeErr;

use super::{PQData, PQScratch};

impl PQScratch {
    pub(crate) fn prepare_query(
        &mut self,
        pq_data: &PQData,
        metric: Metric,
        query: &[f32],
    ) -> ANNResult<()> {
        self.set(query)?;
        self.preprocess_query(pq_data, metric)
    }

    fn preprocess_query(&mut self, pq_data: &PQData, metric: Metric) -> ANNResult<()> {
        let table = pq_data.pq_table();
        let expected_len = table.ncenters() * table.nchunks();
        let dst = diskann_utils::views::MutMatrixView::try_from(
            &mut self.aligned_pqtable_dist_scratch[..expected_len],
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
                    &self.query_scratch,
                    dst,
                );
            }
            Metric::InnerProduct => {
                table.process_into::<diskann_quantization::distances::InnerProduct>(
                    &self.query_scratch,
                    dst,
                );
            }
        }

        Ok(())
    }
}

/// Preprocesses the query stored in `pq_scratch` and computes PQ distances
/// for `id_to_calculate_pq_distance`.
///
/// The resulting distances are written to `pq_scratch.aligned_dist_scratch`.
pub fn quantizer_preprocess(
    pq_scratch: &mut PQScratch,
    pq_data: &PQData,
    metric: Metric,
    id_to_calculate_pq_distance: &[u32],
) -> ANNResult<()> {
    pq_scratch.preprocess_query(pq_data, metric)?;

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
