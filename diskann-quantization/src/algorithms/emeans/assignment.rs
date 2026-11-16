/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::EmeansError;
use crate::{algorithms::kmeans::lloyds::distances_in_place, multi_vector::BlockTransposed};
use diskann_utils::{strided::Strided, views::rowmajor};

/// Batch nearest-center assignment, independent of e-means' centroid updates.
///
/// The trainer supplies a contiguous batch even when its source data is strided.
/// Write every label with an index below centers.nrows() without moving centers.
/// Centers change between calls: update any backend index before assigning the batch.
/// The intended objective is squared L2; approximate assignment changes training quality.
/// Backends may retain scratch buffers, but must support a smaller final batch.
pub trait Assignment {
    type Error: std::error::Error;

    fn assign(
        &mut self,
        batch: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Ref<'_, f32>,
        labels: &mut [u32],
    ) -> Result<(), Self::Error>;
}

pub(super) struct AssignmentBatch {
    transpose: BlockTransposed<f32, 16>,
    norms: Vec<f32>,
}

impl AssignmentBatch {
    pub(super) fn new(n: usize, d: usize) -> Result<Self, EmeansError> {
        Ok(Self {
            transpose: BlockTransposed::try_new(n, d).map_err(|_| EmeansError::SizeOverflow)?,
            norms: vec![0.0; n],
        })
    }

    pub(super) fn assign(
        &mut self,
        data: Strided<'_, f32>,
        indices: &[usize],
        norms: &[f32],
        centers: rowmajor::Ref<'_, f32>,
        center_norms: &[f32],
        labels: &mut [u32],
    ) {
        // Gather directly into reusable SIMD blocks; no shuffled n*d copy or B*k matrix.
        let d = data.ncols();
        for (block, rows) in self
            .transpose
            .as_mut_slice()
            .chunks_mut(16 * d)
            .zip(indices.chunks(16))
        {
            for (lane, &index) in rows.iter().enumerate() {
                for (coordinate, &value) in data.row(index).iter().enumerate() {
                    block[16 * coordinate + lane] = value;
                }
            }
        }
        for (dst, &index) in self.norms.iter_mut().zip(indices) {
            *dst = norms[index];
        }
        // The kernel also computes an f32 residual; it is deliberately not reported because
        // it refers to moving centers and its reduction can overflow on very large datasets.
        distances_in_place(
            self.transpose.as_view(),
            &self.norms,
            centers,
            center_norms,
            labels,
        );
    }
}
