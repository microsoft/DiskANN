/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_docs)]

//! Sector graph
use std::ops::Deref;

use diskann::{ANNError, ANNResult};
use diskann_providers::common::AlignedBoxWithSlice;

use crate::{
    data_model::GraphHeader,
    utils::aligned_file_reader::{traits::AlignedFileReader, AlignedRead},
};

const DEFAULT_DISK_SECTOR_LEN: usize = 4096;

/// Sector graph read from disk index
pub struct DiskSectorGraph<AlignedReaderType: AlignedFileReader> {
    /// Ensure `sector_reader` is dropped before `sectors_data` by placing it before `sectors_data`.
    /// Graph storage to read sectors
    sector_reader: AlignedReaderType,
    /// Sector bytes from disk
    /// One sector has num_nodes_per_sector nodes
    /// Each node's layout: {full precision vector:[T; DIM]}{num_nbrs: u32}{neighbors: [u32; num_nbrs]}
    /// The fp vector is not aligned
    ///
    /// index info for multi-node sectors
    /// node `i` is in sector: [i / num_nodes_per_sector]
    /// offset in sector: [(i % num_nodes_per_sector) * node_len]
    ///
    /// index info for multi-sector nodes
    /// node `i` is in sector: [i * max_node_len.div_ceil(block_size)]
    /// offset in sector: [0]
    sectors_data: AlignedBoxWithSlice<u8>,
    /// Current sector index into which the next read reads data
    cur_sector_idx: u64,

    /// 0 for multi-sector nodes, >0 for multi-node sectors
    num_nodes_per_sector: u64,

    node_len: u64,

    max_n_batch_sector_read: usize,

    num_sectors_per_node: usize,

    block_size: usize,
}

impl<AlignedReaderType: AlignedFileReader> DiskSectorGraph<AlignedReaderType> {
    /// Create SectorGraph instance
    pub fn new(
        sector_reader: AlignedReaderType,
        header: &GraphHeader,
        max_n_batch_sector_read: usize,
    ) -> ANNResult<Self> {
        let mut block_size = header.block_size() as usize;
        let version = header.layout_version();
        if (version.major_version() == 0 && version.minor_version() == 0) || block_size == 0 {
            block_size = DEFAULT_DISK_SECTOR_LEN;
        }

        let num_nodes_per_sector = header.metadata().num_nodes_per_block;
        let node_len = header.metadata().node_len;
        let num_sectors_per_node = if num_nodes_per_sector > 0 {
            1
        } else {
            (node_len as usize).div_ceil(block_size)
        };

        Ok(Self {
            sector_reader,
            sectors_data: AlignedBoxWithSlice::new(
                max_n_batch_sector_read * num_sectors_per_node * block_size,
                block_size,
            )?,
            cur_sector_idx: 0,
            num_nodes_per_sector,
            node_len,
            max_n_batch_sector_read,
            num_sectors_per_node,
            block_size,
        })
    }

    /// Reconfigure SectorGraph if the max number of sectors to read is larger than the current one
    pub fn reconfigure(&mut self, max_n_batch_sector_read: usize) -> ANNResult<()> {
        if max_n_batch_sector_read > self.max_n_batch_sector_read {
            self.max_n_batch_sector_read = max_n_batch_sector_read;
            self.sectors_data = AlignedBoxWithSlice::new(
                max_n_batch_sector_read * self.num_sectors_per_node * self.block_size,
                self.block_size,
            )?;
        }
        Ok(())
    }

    /// Reset SectorGraph
    pub fn reset(&mut self) {
        self.cur_sector_idx = 0;
    }

    /// Read sectors into sectors_data
    /// They are in the same order as sectors_to_fetch
    pub fn read_graph(&mut self, sectors_to_fetch: &[u64]) -> ANNResult<()> {
        let cur_sector_idx_usize: usize = self.cur_sector_idx.try_into()?;
        if sectors_to_fetch.len() > self.max_n_batch_sector_read - cur_sector_idx_usize {
            return Err(ANNError::log_index_error(format_args!(
                "Trying to read too many sectors. number of sectors to read: {}, max number of sectors can read: {}",
                sectors_to_fetch.len(),
                self.max_n_batch_sector_read - cur_sector_idx_usize,
            )));
        }

        let len_per_node = self.num_sectors_per_node * self.block_size;
        let mut sector_slices = self.sectors_data.split_into_nonoverlapping_mut_slices(
            cur_sector_idx_usize * len_per_node
                ..(cur_sector_idx_usize + sectors_to_fetch.len()) * len_per_node,
            len_per_node,
        )?;
        let mut read_requests = Vec::with_capacity(sector_slices.len());
        for (local_sector_idx, slice) in sector_slices.iter_mut().enumerate() {
            let sector_id = sectors_to_fetch[local_sector_idx];
            read_requests.push(AlignedRead::new(sector_id * self.block_size as u64, slice)?);
        }

        self.sector_reader.read(&mut read_requests)?;
        self.cur_sector_idx += sectors_to_fetch.len() as u64;

        Ok(())
    }

    #[inline]
    /// Get node data by local index.
    pub fn node_disk_buf(&self, node_index_local: usize, vertex_id: u32) -> &[u8] {
        // get sector_buf where this node is located
        let sector_buf = self.get_sector_buf(node_index_local);
        let node_offset = self.get_node_offset(vertex_id);
        &sector_buf[node_offset..node_offset + self.node_len as usize]
    }

    /// Get sector data by local index
    #[inline]
    fn get_sector_buf(&self, local_sector_idx: usize) -> &[u8] {
        let len_per_node = self.num_sectors_per_node * self.block_size;
        &self.sectors_data[local_sector_idx * len_per_node..(local_sector_idx + 1) * len_per_node]
    }

    /// Get offset of node in sectors_data
    #[inline]
    fn get_node_offset(&self, vertex_id: u32) -> usize {
        if self.num_nodes_per_sector == 0 {
            // multi-sector node
            0
        } else {
            // multi node in a sector
            (vertex_id as u64 % self.num_nodes_per_sector * self.node_len) as usize
        }
    }

    #[inline]
    /// Gets the index for the sector that contains the node with the given vertex_id
    pub fn node_sector_index(&self, vertex_id: u32) -> u64 {
        1 + if self.num_nodes_per_sector > 0 {
            vertex_id as u64 / self.num_nodes_per_sector
        } else {
            vertex_id as u64 * self.num_sectors_per_node as u64
        }
    }
}

impl<AlignedReaderType: AlignedFileReader> Deref for DiskSectorGraph<AlignedReaderType> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.sectors_data
    }
}

#[cfg(test)]
mod disk_sector_graph_test {
    use crate::utils::aligned_file_reader::{
        traits::AlignedReaderFactory, AlignedFileReaderFactory,
    };
    use diskann_utils::test_data_root;

    use super::*;
    use crate::data_model::{GraphLayoutVersion, GraphMetadata};

    fn test_index_path() -> String {
        test_data_root()
            .join("disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index")
            .to_string_lossy()
            .to_string()
    }

    fn test_initialize_disk_sector_graph(
        num_nodes_per_sector: u64,
        num_sectors_per_node: usize,
        sector_reader: <AlignedFileReaderFactory as AlignedReaderFactory>::AlignedReaderType,
    ) -> DiskSectorGraph<<AlignedFileReaderFactory as AlignedReaderFactory>::AlignedReaderType>
    {
        DiskSectorGraph {
            sectors_data: AlignedBoxWithSlice::new(512, 512).unwrap(),
            sector_reader,
            cur_sector_idx: 0,
            num_nodes_per_sector,
            node_len: 32,
            max_n_batch_sector_read: 4,
            num_sectors_per_node,
            block_size: 64,
        }
    }

    #[test]
    fn test_new_disk_sector_graph_multi_node_per_sector() {
        let metadata = GraphMetadata::new(1000, 32, 500, 32, 2, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 64, GraphLayoutVersion::new(1, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.sectors_data.len(), 128);
        assert_eq!(graph.num_sectors_per_node, 1);
        assert_eq!(graph.num_nodes_per_sector, 2);
    }

    #[test]
    fn test_new_disk_sector_graph_multi_sector_per_node() {
        let metadata = GraphMetadata::new(1000, 32, 500, 128, 0, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 64, GraphLayoutVersion::new(1, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.sectors_data.len(), 256);
        assert_eq!(graph.num_sectors_per_node, 2);
        assert_eq!(graph.num_nodes_per_sector, 0);
    }

    #[test]
    fn test_new_disk_sector_graph_old_version_data() {
        let metadata = GraphMetadata::new(1000, 32, 500, 128, 0, 20, 50, 1024, 256);
        let header = GraphHeader::new(metadata, 9999, GraphLayoutVersion::new(0, 0));
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = DiskSectorGraph::new(reader, &header, 2).unwrap();
        assert_eq!(graph.block_size, DEFAULT_DISK_SECTOR_LEN);
    }

    #[test]
    fn get_sector_buf_test() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(2, 1, reader);
        let sector_buf = graph.get_sector_buf(0);
        assert_eq!(sector_buf.len(), 64);
    }

    #[test]
    fn get_node_offset_test_multi_node_per_sector() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(4, 1, reader);

        assert_eq!(graph.get_node_offset(0), 0);
        assert_eq!(graph.get_node_offset(1), 32);
        assert_eq!(graph.get_node_offset(2), 64);
        assert_eq!(graph.get_node_offset(3), 96);
        assert_eq!(graph.get_node_offset(4), 0);
        assert_eq!(graph.get_node_offset(5), 32);
        assert_eq!(graph.get_node_offset(6), 64);
        assert_eq!(graph.get_node_offset(7), 96);
    }

    #[test]
    fn get_node_offset_test_multi_sector_per_node() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(0, 2, reader);

        assert_eq!(graph.get_node_offset(0), 0);
        assert_eq!(graph.get_node_offset(1), 0);
        assert_eq!(graph.get_node_offset(2), 0);
        assert_eq!(graph.get_node_offset(3), 0);
        assert_eq!(graph.get_node_offset(4), 0);
        assert_eq!(graph.get_node_offset(5), 0);
    }

    #[test]
    fn node_sector_index_test_multi_node_per_sector() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(4, 1, reader);

        assert_eq!(graph.node_sector_index(0), 1);
        assert_eq!(graph.node_sector_index(3), 1);
        assert_eq!(graph.node_sector_index(4), 2);
        assert_eq!(graph.node_sector_index(5), 2);
        assert_eq!(graph.node_sector_index(7), 2);
        assert_eq!(graph.node_sector_index(8), 3);
        assert_eq!(graph.node_sector_index(1023), 256);
        assert_eq!(graph.node_sector_index(1024), 257);
        assert_eq!(graph.node_sector_index(2047), 512);
        assert_eq!(graph.node_sector_index(2048), 513);
    }

    #[test]
    fn node_sector_index_test_multi_sector_per_node() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(0, 2, reader);

        assert_eq!(graph.node_sector_index(0), 1);
        assert_eq!(graph.node_sector_index(3), 7);
        assert_eq!(graph.node_sector_index(4), 9);
        assert_eq!(graph.node_sector_index(5), 11);
        assert_eq!(graph.node_sector_index(7), 15);
        assert_eq!(graph.node_sector_index(8), 17);
        assert_eq!(graph.node_sector_index(1023), 2047);
        assert_eq!(graph.node_sector_index(1024), 2049);
        assert_eq!(graph.node_sector_index(2047), 4095);
        assert_eq!(graph.node_sector_index(2048), 4097);
    }

    #[test]
    fn test_read_graph_max_sectors() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let mut disk_sector_graph = test_initialize_disk_sector_graph(0, 2, reader);

        // Try to read more sectors than the maximum allowed
        let sectors_to_fetch = vec![1, 2, 3, 4, 5, 6];
        let result = disk_sector_graph.read_graph(&sectors_to_fetch);

        // Check that an error is returned
        // Trying to read too many sectors. number of sectors to read: {}, max number of sectors can read: {}",
        assert!(result.is_err());
    }

    #[test]
    fn test_disk_sector_graph_deref() {
        let reader = AlignedFileReaderFactory::new(test_index_path())
            .build()
            .unwrap();
        let graph = test_initialize_disk_sector_graph(1, 1, reader);
        let data = &graph;
        assert_eq!(data.len(), 512);
    }
}
