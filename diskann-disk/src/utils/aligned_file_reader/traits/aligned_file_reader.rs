/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use crate::utils::aligned_file_reader::AlignedRead;

pub trait AlignedFileReader: Send + Sync {
    /// Read the data from the file by sending concurrent io requests in batches.
    fn read(&mut self, read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing
    struct MockAlignedFileReader;

    impl AlignedFileReader for MockAlignedFileReader {
        fn read(&mut self, _read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_aligned_file_reader_trait() {
        let mut reader = MockAlignedFileReader;
        let mut buffer = vec![0u8; 512];
        let read_request = AlignedRead::new(0, &mut buffer).unwrap();
        let mut requests = [read_request];
        
        assert!(reader.read(&mut requests).is_ok());
    }

    #[test]
    fn test_aligned_file_reader_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockAlignedFileReader>();
    }
}
