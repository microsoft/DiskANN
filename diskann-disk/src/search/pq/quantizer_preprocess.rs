/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;
use diskann_vector::distance::Metric;

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
            // Prior to moving query preprocessing onto `PQScratch`, the disk index
            // was hard-coded to use L2 distance for comparisons.
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
