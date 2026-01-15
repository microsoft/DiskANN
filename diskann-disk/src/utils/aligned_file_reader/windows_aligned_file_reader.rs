/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{ptr, thread, time::Duration};

use diskann::{ANNError, ANNResult};
use diskann_platform::{
    get_queued_completion_status, read_file_to_slice, ssd_io_context::IOContext, AccessMode,
    FileHandle, IOCompletionPort, ShareMode, DWORD, OVERLAPPED, ULONG_PTR,
};

use super::traits::AlignedFileReader;
use crate::utils::aligned_file_reader::AlignedRead;

pub const MAX_IO_CONCURRENCY: usize = 128;
pub const IO_COMPLETION_TIMEOUT: DWORD = u32::MAX; // Infinite timeout.
pub const ASYNC_IO_COMPLETION_CHECK_INTERVAL: Duration = Duration::from_micros(5);

/// AlignedFileReader for Windows.  When you modify this class run the benchmarks to make sure
/// we don't regress on runtime.
///
/// # Run this before making your code change
/// cargo bench --bench bench_main -p diskann -- --save-baseline prior_to_change
///
/// # Run this after making your code change to generate comparison metrics
/// cargo bench --bench bench_main -p diskann -- --baseline prior_to_change
pub struct WindowsAlignedFileReader {
    io_context: IOContext,
}

impl WindowsAlignedFileReader {
    pub fn new(fname: &str) -> ANNResult<Self> {
        let mut io_context = IOContext::new();
        tracing::debug!("Creating file handle for {}", fname);
        match unsafe { FileHandle::new(fname, AccessMode::Read, ShareMode::Read) } {
            Ok(file_handle) => io_context.file_handle = file_handle,
            Err(err) => {
                return Err(ANNError::log_io_error(err));
            }
        }

        // Create a io completion port for the file handle, later it will be used to get the completion status.
        match IOCompletionPort::new(&io_context.file_handle, None, 0, 0) {
            Ok(io_completion_port) => io_context.io_completion_port = io_completion_port,
            Err(err) => {
                return Err(ANNError::log_io_error(err));
            }
        }

        Ok(WindowsAlignedFileReader { io_context })
    }
}

impl AlignedFileReader for WindowsAlignedFileReader {
    // Read the data from the file by sending concurrent io requests in batches.
    fn read(&mut self, read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()> {
        let n_requests = read_requests.len();
        let n_batches = n_requests.div_ceil(MAX_IO_CONCURRENCY);
        let ctx = &self.io_context;
        let mut overlapped_in_out =
            vec![unsafe { std::mem::zeroed::<OVERLAPPED>() }; MAX_IO_CONCURRENCY];

        for batch_idx in 0..n_batches {
            let batch_start = MAX_IO_CONCURRENCY * batch_idx;
            let batch_size = std::cmp::min(n_requests - batch_start, MAX_IO_CONCURRENCY);

            for j in 0..batch_size {
                let req = &mut read_requests[batch_start + j];
                let offset = req.offset();
                let os = &mut overlapped_in_out[j];

                match unsafe {
                    read_file_to_slice(&ctx.file_handle, req.aligned_buf_mut(), os, offset)
                } {
                    Ok(_) => {}
                    Err(error) => {
                        return Err(ANNError::log_io_error(error));
                    }
                }
            }

            let mut n_read: DWORD = 0;
            let mut n_complete: u64 = 0;
            let mut completion_key: ULONG_PTR = 0;
            let mut lp_os: *mut OVERLAPPED = ptr::null_mut();
            while n_complete < batch_size as u64 {
                match unsafe {
                    get_queued_completion_status(
                        &ctx.io_completion_port,
                        &mut n_read,
                        &mut completion_key,
                        &mut lp_os,
                        IO_COMPLETION_TIMEOUT,
                    )
                } {
                    // An IO request completed.
                    Ok(true) => n_complete += 1,
                    // No IO request completed, continue to wait.
                    Ok(false) => {
                        thread::sleep(ASYNC_IO_COMPLETION_CHECK_INTERVAL);
                    }
                    // An error ocurred.
                    Err(error) => return Err(ANNError::log_io_error(error)),
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{BufReader, Read, Seek, SeekFrom},
    };

    use bincode::deserialize_from;
    use diskann_utils::test_data_root;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::utils::aligned_file_reader::AlignedRead;
    use diskann_providers::common::AlignedBoxWithSlice;

    fn test_index_path() -> String {
        test_data_root()
            .join("disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index")
            .to_string_lossy()
            .to_string()
    }

    fn truth_node_data_path() -> String {
        test_data_root()
            .join("disk_index_misc/disk_index_node_data_aligned_reader_truth.bin")
            .to_string_lossy()
            .to_string()
    }

    const DEFAULT_DISK_SECTOR_LEN: usize = 4096;

    #[derive(Debug, Serialize, Deserialize)]
    struct NodeData {
        num_neighbors: u32,
        coordinates: Vec<f32>,
        neighbors: Vec<u32>,
    }

    impl PartialEq for NodeData {
        fn eq(&self, other: &Self) -> bool {
            self.num_neighbors == other.num_neighbors
                && self.coordinates == other.coordinates
                && self.neighbors == other.neighbors
        }
    }

    #[test]
    fn test_new_aligned_file_reader() {
        // Replace "test_file_path" with actual file path
        let result = WindowsAlignedFileReader::new(&test_index_path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_read() {
        let mut reader = WindowsAlignedFileReader::new(&test_index_path()).unwrap();

        let read_length = 512; // adjust according to your logic
        let num_read = 10;
        let mut aligned_mem = AlignedBoxWithSlice::<u8>::new(read_length * num_read, 512).unwrap();

        // create and add AlignedReads to the vector
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .enumerate()
            .map(|(i, slice)| {
                let offset = (i * read_length) as u64;
                AlignedRead::new(offset, slice).unwrap()
            })
            .collect();

        let result = reader.read(&mut aligned_reads);
        assert!(result.is_ok());

        // Assert that the actual data is correct.
        let mut file = File::open(test_index_path()).unwrap();
        for current_read in aligned_reads {
            let mut expected = vec![0; current_read.aligned_buf().len()];
            file.seek(SeekFrom::Start(current_read.offset())).unwrap();
            file.read_exact(&mut expected).unwrap();

            assert_eq!(
                expected,
                current_read.aligned_buf(),
                "aligned_buf did not contain the expected data"
            );
        }
    }

    #[test]
    fn test_read_disk_index_by_sector() {
        let mut reader = WindowsAlignedFileReader::new(&test_index_path()).unwrap();

        let read_length = DEFAULT_DISK_SECTOR_LEN; // adjust according to your logic
        let num_sector = 10;
        let mut aligned_mem =
            AlignedBoxWithSlice::<u8>::new(read_length * num_sector, 512).unwrap();

        // Each slice will be used as the buffer for a read request of a sector.
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .enumerate()
            .map(|(sector_id, slice)| {
                let offset = (sector_id * read_length) as u64;
                AlignedRead::new(offset, slice).unwrap()
            })
            .collect();

        let result = reader.read(&mut aligned_reads);
        assert!(result.is_ok());

        aligned_reads.iter().for_each(|read| {
            assert_eq!(read.aligned_buf().len(), DEFAULT_DISK_SECTOR_LEN);
        });

        let disk_layout_meta = reconstruct_disk_meta(aligned_reads[0].aligned_buf_mut());
        assert!(disk_layout_meta.len() > 9);

        let dims = disk_layout_meta[1];
        let num_pts = disk_layout_meta[0];
        let node_len = disk_layout_meta[3];
        let max_num_nodes_per_sector = disk_layout_meta[4];

        assert!(node_len * max_num_nodes_per_sector < DEFAULT_DISK_SECTOR_LEN as u64);

        let num_nbrs_start = (dims as usize) * std::mem::size_of::<f32>();
        let nbrs_buf_start = num_nbrs_start + std::mem::size_of::<u32>();

        let mut node_data_array = Vec::with_capacity(max_num_nodes_per_sector as usize * 9);

        // Only validate the first 9 sectors with graph nodes.
        (1..9).for_each(|sector_id| {
            let sector_data = &mem_slices[sector_id];
            for node_data in sector_data.chunks_exact(node_len as usize) {
                // Extract coordinates data from the start of the node_data
                let coordinates_end = (dims as usize) * std::mem::size_of::<f32>();
                let coordinates = node_data[0..coordinates_end]
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                // Extract number of neighbors from the node_data
                let neighbors_num = u32::from_le_bytes(
                    node_data[num_nbrs_start..nbrs_buf_start]
                        .try_into()
                        .unwrap(),
                );

                let nbors_buf_end =
                    nbrs_buf_start + (neighbors_num as usize) * std::mem::size_of::<u32>();

                // Extract neighbors from the node data.
                let mut neighbors = Vec::new();
                for nbors_data in node_data[nbrs_buf_start..nbors_buf_end]
                    .chunks_exact(std::mem::size_of::<u32>())
                {
                    let nbors_id = u32::from_le_bytes(nbors_data.try_into().unwrap());
                    assert!(nbors_id < num_pts as u32);
                    neighbors.push(nbors_id);
                }

                // Create NodeData struct and push it to the node_data_array
                node_data_array.push(NodeData {
                    num_neighbors: neighbors_num,
                    coordinates,
                    neighbors,
                });
            }
        });

        // Compare that each node read from the disk index are expected.
        let node_data_truth_file = File::open(truth_node_data_path()).unwrap();
        let reader = BufReader::new(node_data_truth_file);

        let node_data_vec: Vec<NodeData> = deserialize_from(reader).unwrap();
        for (node_from_node_data_file, node_from_disk_index) in
            node_data_vec.iter().zip(node_data_array.iter())
        {
            // Verify that the NodeData from the file is equal to the NodeData in node_data_array
            assert_eq!(node_from_node_data_file, node_from_disk_index);
        }
    }

    #[test]
    fn test_read_fail_invalid_file() {
        let reader = WindowsAlignedFileReader::new("/invalid_path");
        assert!(reader.is_err());
    }

    #[test]
    #[allow(clippy::read_zero_byte_vec)]
    fn test_read_no_requests() {
        let mut reader = WindowsAlignedFileReader::new(&test_index_path()).unwrap();

        let mut read_requests = Vec::<AlignedRead<u8>>::new();
        let result = reader.read(&mut read_requests);
        assert!(result.is_ok());
    }

    fn reconstruct_disk_meta(buffer: &[u8]) -> Vec<u64> {
        let size_of_u64 = std::mem::size_of::<u64>();

        let num_values = buffer.len() / size_of_u64;
        let mut disk_layout_meta = Vec::with_capacity(num_values);
        let meta_data = &buffer[8..];

        for chunk in meta_data.chunks_exact(size_of_u64) {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            disk_layout_meta.push(value);
        }

        disk_layout_meta
    }
}
