/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    fs::OpenOptions,
    os::{fd::AsRawFd, unix::fs::OpenOptionsExt},
};

use diskann::{ANNError, ANNResult};
use diskann_platform::ssd_io_context::IOContext;
use io_uring::IoUring;
use libc;

use crate::utils::aligned_file_reader::{traits::AlignedFileReader, AlignedRead};

pub const MAX_IO_CONCURRENCY: usize = 128;

pub struct LinuxAlignedFileReader {
    io_context: IOContext,
}

/// AlignedFileReader for Linux.  When you modify this class run the benchmarks to make sure
/// we don't regress on runtime.
///
/// # Run this before making code your change
/// cargo bench --bench bench_main -p diskann -- --save-baseline prior_to_change
///
/// # Run this after making your code change to generate comparison metrics
/// cargo bench --bench bench_main -p diskann -- --baseline prior_to_change
impl LinuxAlignedFileReader {
    pub fn new(fname: &str) -> ANNResult<Self> {
        // Open file as read-only
        // Apply the `O_DIRECT` flag to bypass the kernel page cache.
        // See: https://man7.org/linux/man-pages/man2/open.2.html
        let open_result = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(fname);

        let file = match open_result {
            Ok(file_handle) => file_handle,
            Err(err) => {
                return Err(ANNError::log_io_error(err));
            }
        };

        let ring = IoUring::new(MAX_IO_CONCURRENCY as u32)?;
        let fd = file.as_raw_fd();
        ring.submitter().register_files(std::slice::from_ref(&fd))?;
        let io_context = IOContext::new(file, ring);

        Ok(LinuxAlignedFileReader { io_context })
    }

    fn submit_aligned_read(
        aligned_read: &mut AlignedRead<u8>,
        ring: &mut IoUring,
        identifier: u64,
    ) -> Result<(), ANNError> {
        let fixed_buffer = libc::iovec {
            iov_base: aligned_read.aligned_buf_mut().as_mut_ptr() as _,
            iov_len: aligned_read.aligned_buf_mut().len() as _,
        };

        let read = io_uring::opcode::Read::new(
            // 0 represents the file descriptor that was registered with the ring via `register_files()` method.
            io_uring::types::Fixed(0),
            fixed_buffer.iov_base.cast::<u8>(),
            fixed_buffer.iov_len as _,
        )
        .offset(aligned_read.offset())
        .build()
        .user_data(identifier);

        // Submission should not fail because the batch_size should always be less
        // than MAX_IO_CONCURRENCY and the ring was initialized with MAX_IO_CONCURRENCY
        // spaces in the processing queue
        unsafe {
            ring.submission()
                .push(&read)
                .map_err(ANNError::log_push_error)?
        };
        Ok(())
    }
}

impl AlignedFileReader for LinuxAlignedFileReader {
    // Read the data from the file by sending concurrent io requests in batches.
    fn read(&mut self, read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()> {
        let n_requests = read_requests.len();
        let n_batches = n_requests.div_ceil(MAX_IO_CONCURRENCY);

        let ring = &mut self.io_context.ring;

        for batch_idx in 0..n_batches {
            // batch_size is the number of requests to submit, not the size of the request.
            let batch_start = MAX_IO_CONCURRENCY * batch_idx;
            let batch_size = std::cmp::min(n_requests - batch_start, MAX_IO_CONCURRENCY);

            for j in 0..batch_size {
                let read_id = j + batch_start;
                let aligned_read = &mut read_requests[read_id];
                Self::submit_aligned_read(aligned_read, ring, read_id as u64)?;
            }

            // Wait for the batch to complete.
            ring.submit_and_wait(batch_size)?;

            // N.B.: Flushing the completion queue appears to be important for proper
            // operation.
            // Flush the completion queue.
            for cqe in ring.completion() {
                if cqe.result() < 0 {
                    return Err(ANNError::log_io_error(std::io::Error::from_raw_os_error(
                        cqe.result(),
                    )));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cmp::max,
        fs::File,
        io::{BufReader, Read, Seek, SeekFrom},
    };

    use bincode::deserialize_from;
    use serde::{Deserialize, Serialize};

    use super::*;
    use diskann_providers::common::AlignedBoxWithSlice;
    pub const TEST_INDEX_PATH: &str =
        "../test_data/disk_index_misc/disk_index_siftsmall_learn_256pts_R4_L50_A1.2_aligned_reader_test.index";
    pub const TRUTH_NODE_DATA_PATH: &str =
        "../test_data/disk_index_misc/disk_index_node_data_aligned_reader_truth.bin";
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
        let result = LinuxAlignedFileReader::new(TEST_INDEX_PATH);
        assert!(result.is_ok());
    }

    #[test]
    fn test_read() {
        let mut reader = LinuxAlignedFileReader::new(TEST_INDEX_PATH).unwrap();

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
        let mut file = File::open(TEST_INDEX_PATH).unwrap();
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

    /// BUG: io-uring submit_and_wait waits for a cumulative number of items to be completed, not
    /// just the current batch.  This causes the LinuxAlignedFileReader.read method to return
    /// before the final batches have been completely read.  The purpose of this test is to
    /// force many batches to be queued for read and ensure that all are read when the
    /// LinuxAlignedFileReader.read method returns.
    #[test]
    fn many_batches_all_should_have_data() {
        let mut reader = LinuxAlignedFileReader::new(TEST_INDEX_PATH).unwrap();

        let read_length = 512;
        let num_read = MAX_IO_CONCURRENCY * 100; // The LinuxAlignedFileReader batches reads according to MAX_IO_CONCURRENCY.  Make sure we have many batches to handle.
        let mut aligned_mem = AlignedBoxWithSlice::<u8>::new(read_length * num_read, 512).unwrap();

        // create and add AlignedReads to the vector
        let mut mem_slices = aligned_mem
            .split_into_nonoverlapping_mut_slices(0..aligned_mem.len(), read_length)
            .unwrap();

        // Read the same data from disk over and over again.  We guarantee that it is not all zeros.
        let mut aligned_reads: Vec<AlignedRead<'_, u8>> = mem_slices
            .iter_mut()
            .map(|slice| AlignedRead::new(0, slice).unwrap())
            .collect();

        let result = reader.read(&mut aligned_reads);

        // Make sure read completed successfully
        assert!(result.is_ok());

        // If we find any AlignedRead objects that are empty then the reader never read them
        // from disk.
        assert!(
            !aligned_reads.iter().any(aligned_read_buffer_is_empty),
            "Found uninitialized data that should have been read from disk"
        );
    }

    /// Return True if the AlignedRead value is empty or False if the AlignedRead value is not empty.
    fn aligned_read_buffer_is_empty(read: &AlignedRead<'_, u8>) -> bool {
        let max_value = read.aligned_buf().iter().fold(0, |acc, &x| max(acc, x));

        // If max_value is zero then this aligned read was not completed.  Data was not
        // read from disk because all values in memory are zero.
        max_value == 0
    }

    #[test]
    fn test_read_disk_index_by_sector() {
        let mut reader = LinuxAlignedFileReader::new(TEST_INDEX_PATH).unwrap();

        let read_length = 512; // adjust according to your logic
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
            assert_eq!(read.aligned_buf().len(), 512);
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
        let node_data_truth_file = File::open(TRUTH_NODE_DATA_PATH).unwrap();
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
        let reader = LinuxAlignedFileReader::new("/invalid_path");
        assert!(reader.is_err());
    }

    #[test]
    #[allow(clippy::read_zero_byte_vec)]
    fn test_read_no_requests() {
        let mut reader = LinuxAlignedFileReader::new(TEST_INDEX_PATH).unwrap();

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
