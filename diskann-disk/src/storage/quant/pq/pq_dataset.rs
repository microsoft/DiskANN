/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use core::fmt::Debug;

use diskann::{ANNError, ANNResult};
use diskann_quantization::product::TransposedTable;

use diskann_providers::model::{FixedChunkPQTable, PQCompressedData};

/// Behind the scenes, we can use either the [`FixedChunkPQTable`] or a
/// [`diskann_quantization::product::TransposedTable`]. The [`TrasposedTable`] is much faster
/// for preprocessing, but does not support:
///
/// 1. Removal of the dataset centroid (if that was used).
/// 2. OPQ style transformations.
///
/// So, we can only use the [`TransposedTable`] when OPQ is not used and the dataset centroid
/// is all zero.
#[derive(Debug)]
pub enum PQTable {
    Transposed(TransposedTable),
    Fixed(FixedChunkPQTable),
}

#[derive(Debug)]
pub struct PQData {
    // pq pivot table.
    pq_pivot_table: PQTable,

    // pq compressed vectors.
    pq_compressed_data: PQCompressedData,
}

impl PQData {
    pub fn new(
        pq_pivot_table: FixedChunkPQTable,
        pq_compressed_data: PQCompressedData,
    ) -> ANNResult<Self> {
        // Check if we can use the transposed table. If so, go for it.
        let centroid_is_zero = pq_pivot_table.get_centroids().iter().all(|i| *i == 0.0);
        let pq_pivot_table = if centroid_is_zero {
            let transposed = TransposedTable::from_parts(
                pq_pivot_table.view_pivots(),
                pq_pivot_table.view_offsets().to_owned(),
            )
            .map_err(|err| ANNError::log_pq_error(diskann_quantization::error::format(&err)))?;
            PQTable::Transposed(transposed)
        } else {
            PQTable::Fixed(pq_pivot_table)
        };

        Ok(Self {
            pq_pivot_table,
            pq_compressed_data,
        })
    }

    /// Get pq_table
    pub fn pq_table(&self) -> &PQTable {
        &self.pq_pivot_table
    }

    /// Return the number of chunks in the underlying PQ schema.
    pub fn get_num_chunks(&self) -> usize {
        match &self.pq_pivot_table {
            PQTable::Transposed(table) => table.nchunks(),
            PQTable::Fixed(table) => table.get_num_chunks(),
        }
    }

    /// Return the number of centers in the underlying PQ schema.
    pub fn get_num_centers(&self) -> usize {
        match &self.pq_pivot_table {
            PQTable::Transposed(table) => table.ncenters(),
            PQTable::Fixed(table) => table.get_num_centers(),
        }
    }

    /// Get pq_compressed_data
    pub fn pq_compressed_data(&self) -> &PQCompressedData {
        &self.pq_compressed_data
    }

    // Get compressed vector with the given vector id from the pq_compressed_data.
    pub fn get_compressed_vector(&self, vector_id: usize) -> ANNResult<&[u8]> {
        self.pq_compressed_data.get_compressed_vector(vector_id)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn create_pq_data() -> ANNResult<PQData> {
        let dim = 2;

        let pq_pivot_table = FixedChunkPQTable::new(
            dim,
            Box::new([0.0, 0.0, 1.0, 1.0]),
            Box::new([0.0, 0.0]),
            Box::new([0, 2]),
        )
        .unwrap();
        let mut pq_compressed_data = PQCompressedData::new(3, 1).unwrap();

        let compressed_vector = [123, 111, 255];
        pq_compressed_data
            .into_dto()
            .data
            .copy_from_slice(&compressed_vector);

        PQData::new(pq_pivot_table, pq_compressed_data)
    }

    #[test]
    fn test_get_compressed_vector() {
        let dataset = create_pq_data().unwrap();

        let vector_id = 0;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[123]);

        let vector_id = 1;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[111]);

        let vector_id = 2;
        let result = dataset.get_compressed_vector(vector_id).unwrap();
        assert_eq!(result, &[255]);
    }

    #[test]
    fn test_get_num_chunks() {
        let dataset = create_pq_data().unwrap();
        assert_eq!(dataset.get_num_chunks(), 1);
    }

    #[test]
    fn test_get_num_centers() {
        let dataset = create_pq_data().unwrap();
        assert_eq!(dataset.get_num_centers(), 2);
    }
}
