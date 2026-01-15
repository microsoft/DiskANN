/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};

use crate::{common::AlignedBoxWithSlice, utils::DatasetDto};

#[allow(dead_code)]
#[derive(Debug)]
pub struct PQCompressedData {
    // Each vector is represented as num_pq_chunks of centroid ids.
    pub num_pq_chunks: usize,

    // Number of vectors.
    pub num_points: usize,

    // Compressed data.
    data: AlignedBoxWithSlice<u8>,

    // Capacity of the data set.
    capacity: usize,
}

impl PQCompressedData {
    /// Creates a new `PQCompressedData` with the given number of pq chunks and data.
    pub fn new(num_points: usize, num_pq_chunks: usize) -> ANNResult<Self> {
        let capacity = num_points * num_pq_chunks;

        Ok(Self {
            num_pq_chunks,
            data: AlignedBoxWithSlice::new(capacity, std::mem::size_of::<u8>())?,
            num_points,
            capacity,
        })
    }

    /// Returns a slice of the compressed data for the given vertex id.
    pub fn get_compressed_vector(&self, vector_id: usize) -> ANNResult<&[u8]> {
        // Ensure the indices are within the bounds of the data.
        if vector_id >= self.num_points {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the compressed dataset.",
            ));
        }

        let start_index = self.num_pq_chunks * vector_id;
        let end_index = start_index + self.num_pq_chunks;

        Ok(&self.data.as_slice()[start_index..end_index])
    }

    pub fn get_compressed_vector_mut(&mut self, vector_id: usize) -> ANNResult<&mut [u8]> {
        // Ensure the indices are within the bounds of the data.
        if vector_id >= self.num_points {
            return Err(ANNError::log_index_error(
                "Vector id is out of boundary in the compressed dataset.",
            ));
        }

        let start_index = self.num_pq_chunks * vector_id;
        let end_index = start_index + self.num_pq_chunks;

        Ok(&mut self.data[start_index..end_index])
    }

    /// get immutable data slice
    pub fn get_data(&self) -> &[u8] {
        &self.data
    }

    /// Convert into dto object
    pub fn into_dto(&mut self) -> DatasetDto<'_, u8> {
        DatasetDto {
            data: &mut self.data,
            rounded_dim: self.num_pq_chunks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_compressed_vector() {
        let num_points = 3;
        let num_pq_chunks = 2;
        let mut pq_compressed_data = PQCompressedData::new(num_points, num_pq_chunks).unwrap();

        // Set some data
        pq_compressed_data
            .data
            .as_mut_slice()
            .clone_from_slice(vec![0, 1, 2, 3, 4, 5].as_slice());

        // Test getting a compressed vector
        let compressed_vector = pq_compressed_data.get_compressed_vector(1).unwrap();
        assert_eq!(compressed_vector, &[2, 3]);

        let compressed_vector = pq_compressed_data.get_compressed_vector(9);
        assert!(compressed_vector.is_err());
    }
}
