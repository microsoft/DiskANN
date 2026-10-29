/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};
use diskann_providers::model::compute_pq_distance;
use diskann_quantization::{
    alloc::{GlobalAllocator, ScopedAllocator},
    spherical::iface,
};
use diskann_utils::views::rowmajor::Matrix;
use diskann_vector::{distance::Metric, PreprocessedDistanceFunction};

use super::pq::PQScratch;
use crate::storage::quant::{SearchData, SearchDataInner};

pub(crate) enum QuantizedScratch {
    PQ(PQScratch),
    SphericalQ3 {
        computer: Option<iface::QueryComputer>,
        batch_size: usize,
    },
}

impl QuantizedScratch {
    pub(crate) fn new(data: &SearchData, graph_degree: usize) -> ANNResult<Self> {
        match &data.0 {
            SearchDataInner::PQ(data) => Ok(Self::PQ(PQScratch::new(
                graph_degree,
                data.get_dim(),
                data.get_num_chunks(),
                data.get_num_centers(),
            )?)),
            SearchDataInner::SphericalQ3 { .. } => Ok(Self::SphericalQ3 {
                computer: None,
                batch_size: graph_degree.max(1),
            }),
        }
    }

    pub(crate) fn max_vectors(&self) -> usize {
        match self {
            Self::PQ(scratch) => scratch.max_vectors(),
            Self::SphericalQ3 { batch_size, .. } => *batch_size,
        }
    }

    pub(crate) fn prepare_query(
        &mut self,
        data: &SearchData,
        metric: Metric,
        query: &[f32],
    ) -> ANNResult<()> {
        match (self, &data.0) {
            (Self::PQ(scratch), SearchDataInner::PQ(data)) => {
                scratch.prepare_query(data, metric, query)
            }
            (Self::SphericalQ3 { computer, .. }, SearchDataInner::SphericalQ3 { plan, .. }) => {
                *computer = None;
                *computer = Some(
                    plan.fused_query_computer(
                        query,
                        iface::QueryLayout::FullPrecision,
                        true,
                        GlobalAllocator,
                        ScopedAllocator::global(),
                    )
                    .map_err(ANNError::new)?,
                );
                Ok(())
            }
            _ => Err(ANNError::message("Mismatched search quantization scratch")),
        }
    }

    pub(crate) fn distances<F>(
        &mut self,
        data: &SearchData,
        ids: &[u32],
        mut callback: F,
    ) -> ANNResult<()>
    where
        F: FnMut(f32, u32),
    {
        match (self, &data.0) {
            (Self::PQ(scratch), SearchDataInner::PQ(data)) => {
                compute_pq_distance(
                    ids,
                    data.get_num_chunks(),
                    &scratch.aligned_pqtable_dist_scratch,
                    data.pq_compressed_data().as_slice(),
                    &mut scratch.aligned_pq_coord_scratch,
                    &mut scratch.aligned_dist_scratch,
                )?;
                for (index, id) in ids.iter().enumerate() {
                    callback(scratch.aligned_dist_scratch[index], *id);
                }
            }
            (Self::SphericalQ3 { computer, .. }, SearchDataInner::SphericalQ3 { codes, .. }) => {
                let computer = computer
                    .as_ref()
                    .ok_or_else(|| ANNError::message("Spherical query is not prepared"))?;
                for id in ids {
                    let code = codes
                        .get_row(*id as usize)
                        .ok_or_else(|| ANNError::message("Spherical vector ID is out of bounds"))?;
                    let distance = computer
                        .evaluate_similarity(iface::Opaque::new(code))
                        .map_err(ANNError::new)?;
                    callback(distance, *id);
                }
            }
            _ => return Err(ANNError::message("Mismatched search quantization scratch")),
        }
        Ok(())
    }
}
