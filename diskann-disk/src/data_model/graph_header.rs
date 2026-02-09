/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Cursor;

use byteorder::{LittleEndian, ReadBytesExt};
use diskann::{ANNError, ANNResult};
use thiserror::Error;

use super::{GraphLayoutVersion, GraphMetadata};

/// GraphHeader. The header is stored in the first sector of the disk index file, or the first segment of the JET stream.
#[derive(Clone)]
pub struct GraphHeader {
    // Graph metadata.
    metadata: GraphMetadata,

    // Block size.
    block_size: u64,

    // Graph layout version.
    layout_version: GraphLayoutVersion,
}

#[derive(Error, Debug, PartialEq)]
pub enum GraphHeaderError {
    #[error("Overflow occurred during max_degree calculation.")]
    MaxDegreeOverflow,
    #[error("Unsupported graph layout version {0} for max_degree calculation.")]
    MaxDegreeUnsupportedLayoutVersion(GraphLayoutVersion),
}

impl From<GraphHeaderError> for ANNError {
    #[track_caller]
    fn from(value: GraphHeaderError) -> Self {
        ANNError::log_index_error(value)
    }
}

impl GraphHeader {
    /// Update the layout version when the [GraphHeader] layout is modified.
    pub const CURRENT_LAYOUT_VERSION: GraphLayoutVersion = GraphLayoutVersion::new(1, 0);

    pub fn new(
        metadata: GraphMetadata,
        block_size: u64,
        layout_version: GraphLayoutVersion,
    ) -> Self {
        Self {
            metadata,
            block_size,
            layout_version,
        }
    }
    /// Serialize the `GraphHeader` object to a byte vector.
    /// Layout:
    /// | GraphMetadata (80 bytes) | BlockSize (8 bytes) | GraphLayoutVersion (8 bytes) |
    pub fn to_bytes(&self) -> ANNResult<Vec<u8>> {
        let mut buffer = vec![];
        buffer.extend_from_slice(self.metadata.to_bytes()?.as_ref());
        buffer.extend_from_slice(self.block_size.to_le_bytes().as_ref());
        buffer.extend_from_slice(self.layout_version.to_bytes().as_ref());

        Ok(buffer)
    }

    /// Get the size of the header after serialization.
    #[inline]
    pub fn get_size() -> usize {
        GraphMetadata::get_size()
            + std::mem::size_of::<u64>()
            + std::mem::size_of::<GraphLayoutVersion>()
    }

    pub fn metadata(&self) -> &GraphMetadata {
        &self.metadata
    }

    pub fn block_size(&self) -> u64 {
        self.block_size
    }

    pub fn layout_version(&self) -> &GraphLayoutVersion {
        &self.layout_version
    }

    /// Returns the maximum degree of the graph
    ///
    /// # Type Parameters
    /// * `DataType` - The type of vector data stored in the graph nodes
    pub fn max_degree<DataType>(&self) -> Result<usize, GraphHeaderError> {
        let supported_versions = [GraphLayoutVersion::new(0, 0), GraphLayoutVersion::new(1, 0)];

        if supported_versions.contains(&self.layout_version) {
            // Calculates max degree based on the node layout:
            // - Each node contains: vector data + neighbor list + associated data
            // - Neighbors are stored as u32 indices
            // - The -1 accounts for the first u32 which stores number of neighbors
            let vector_len = std::mem::size_of::<DataType>() * self.metadata.dims;
            let max_degree = (self.metadata.node_len as usize)
                .checked_sub(vector_len)
                .and_then(|len| len.checked_sub(self.metadata.associated_data_length))
                .and_then(|len| len.checked_div(std::mem::size_of::<u32>()))
                .and_then(|len| len.checked_sub(1));

            match max_degree {
                Some(degree) => Ok(degree),
                None => Err(GraphHeaderError::MaxDegreeOverflow),
            }
        } else {
            Err(GraphHeaderError::MaxDegreeUnsupportedLayoutVersion(
                self.layout_version.clone(),
            ))
        }
    }
}

impl<'a> TryFrom<&'a [u8]> for GraphHeader {
    type Error = ANNError;
    /// Try creating a new `GraphHeader` object from a byte slice. The try_from syntax is used here instead of from because this operation can fail.
    ///
    /// Layout:
    /// | GraphMetadata (80 bytes) | BlockSize (8 bytes) | GraphLayoutVersion (8 bytes) |
    fn try_from(value: &'a [u8]) -> ANNResult<Self> {
        if value.len() < Self::get_size() {
            Err(ANNError::log_parse_slice_error(
                "&[u8]".to_string(),
                "GraphHeader".to_string(),
                "The given bytes are not long enough to create a valid graph header.".to_string(),
            ))
        } else {
            // Parse metadata.
            let metadata_len = GraphMetadata::get_size();
            let metadata = GraphMetadata::try_from(&value[0..metadata_len])?;

            // Parse block size.
            let block_size = Cursor::new(&value[metadata_len..]).read_u64::<LittleEndian>()?;

            // Parse layout version.
            let layout_version = GraphLayoutVersion::try_from(&value[metadata_len + 8..])?;

            Ok(Self::new(metadata, block_size, layout_version))
        }
    }
}

#[cfg(test)]
mod tests {
    use diskann::ANNErrorKind;
    use rstest::rstest;

    use super::*;
    use crate::data_model::{GraphHeader, GraphLayoutVersion, GraphMetadata};

    #[test]
    fn test_graph_header_to_bytes_and_try_from() {
        let layout_version = GraphLayoutVersion::new(1, 0);
        let block_size = 128;
        let num_pts = 1000;
        let dims = 32;
        let medoid = 500;
        let node_len = 64;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;
        let data_size = 256;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );

        let header = GraphHeader::new(metadata.clone(), block_size, layout_version.clone());
        let bytes = header.to_bytes().unwrap();
        assert_eq!(bytes.len(), GraphHeader::get_size());

        let deserialized_header = GraphHeader::try_from(bytes.as_slice()).unwrap();
        assert_eq!(metadata.num_pts, deserialized_header.metadata.num_pts);
        assert_eq!(metadata.dims, deserialized_header.metadata.dims);
        assert_eq!(metadata.medoid, deserialized_header.metadata.medoid);
        assert_eq!(metadata.node_len, deserialized_header.metadata.node_len);
        assert_eq!(
            metadata.num_nodes_per_block,
            deserialized_header.metadata.num_nodes_per_block
        );
        assert_eq!(
            metadata.vamana_frozen_num,
            deserialized_header.metadata.vamana_frozen_num
        );
        assert_eq!(
            metadata.vamana_frozen_loc,
            deserialized_header.metadata.vamana_frozen_loc
        );
        assert_eq!(
            metadata.disk_index_file_size,
            deserialized_header.metadata.disk_index_file_size
        );
        assert_eq!(
            metadata.associated_data_length,
            deserialized_header.metadata.associated_data_length
        );

        assert_eq!(block_size, deserialized_header.block_size);
        assert_eq!(layout_version, deserialized_header.layout_version);
    }

    #[test]
    fn test_graph_header_try_from_invalid_bytes() {
        let invalid_bytes = vec![1; GraphHeader::get_size() - 1];
        let result = GraphHeader::try_from(&invalid_bytes[..]);
        assert!(result.is_err());
    }

    #[rstest]
    #[case(384, 1008, 0, 59, GraphLayoutVersion::new(0, 0))]
    #[case(384, 1008, 0, 59, GraphLayoutVersion::new(1, 0))]
    #[case(3072, 6384, 0, 59, GraphLayoutVersion::new(0, 0))]
    #[case(3072, 6384, 0, 59, GraphLayoutVersion::new(1, 0))]
    // Current layout version should support max degree calculation
    #[case(384, 1008, 0, 59, GraphHeader::CURRENT_LAYOUT_VERSION)]
    fn test_graph_header_max_degree(
        #[case] dims: usize,
        #[case] node_len: u64,
        #[case] data_size: usize,
        #[case] expected_max_degree: usize,
        #[case] layout_version: GraphLayoutVersion,
    ) {
        let num_pts = 1000;
        let medoid = 500;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );
        let block_size = 128;

        let header = GraphHeader::new(metadata, block_size, layout_version);

        let max_degree = header.max_degree::<diskann_vector::Half>();
        assert!(max_degree.is_ok());
        assert_eq!(max_degree.unwrap(), expected_max_degree);
    }

    #[rstest]
    #[case(1, 1)]
    #[case(2, 0)]
    fn test_graph_header_max_degree_unsupported_layout_version(
        #[case] major_version: u32,
        #[case] minor_version: u32,
    ) {
        let num_pts = 1000;
        let dims = 32;
        let medoid = 500;
        let node_len = 64;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;
        let data_size = 256;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );
        let layout_version = GraphLayoutVersion::new(major_version, minor_version);
        let block_size = 128;

        let header = GraphHeader::new(metadata, block_size, layout_version.clone());

        let max_degree = header.max_degree::<diskann_vector::Half>();
        assert!(max_degree.is_err());
        assert_eq!(
            max_degree.unwrap_err(),
            GraphHeaderError::MaxDegreeUnsupportedLayoutVersion(layout_version.clone())
        );
    }

    #[test]
    fn test_graph_header_max_degree_overflow() {
        let dims = 384;
        let node_len = 384;
        let data_size = 0;

        let num_pts = 1000;
        let medoid = 500;
        let num_nodes_per_sector = 4;
        let vamana_frozen_num = 20;
        let vamana_frozen_loc = 50;
        let disk_index_file_size = 1024;

        let metadata = GraphMetadata::new(
            num_pts,
            dims,
            medoid,
            node_len,
            num_nodes_per_sector,
            vamana_frozen_num,
            vamana_frozen_loc,
            disk_index_file_size,
            data_size,
        );
        let layout_version = GraphLayoutVersion::new(1, 0);
        let block_size = 128;

        let header = GraphHeader::new(metadata, block_size, layout_version);

        let max_degree = header.max_degree::<diskann_vector::Half>();
        assert!(max_degree.is_err());
        assert_eq!(max_degree.unwrap_err(), GraphHeaderError::MaxDegreeOverflow);
    }

    // test cases for GraphHeaderError conversion to ANNError
    #[test]
    fn test_graph_header_error_conversion() {
        let error = GraphHeaderError::MaxDegreeOverflow;
        let ann_error: ANNError = error.into();
        assert_eq!(ann_error.kind(), ANNErrorKind::IndexError);

        let layout_version = GraphLayoutVersion::new(1, 0);
        let error = GraphHeaderError::MaxDegreeUnsupportedLayoutVersion(layout_version);
        let ann_error: ANNError = error.into();
        assert_eq!(ann_error.kind(), ANNErrorKind::IndexError);
    }
}
