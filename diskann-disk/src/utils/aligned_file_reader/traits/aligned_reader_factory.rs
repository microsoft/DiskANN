/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNResult;

use super::AlignedFileReader;

pub trait AlignedReaderFactory: Send + Sync {
    type AlignedReaderType: AlignedFileReader;

    fn build(&self) -> ANNResult<Self::AlignedReaderType>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::aligned_file_reader::AlignedRead;

    // Mock implementation for testing
    struct MockAlignedFileReader;

    impl AlignedFileReader for MockAlignedFileReader {
        fn read(&mut self, _read_requests: &mut [AlignedRead<u8>]) -> ANNResult<()> {
            Ok(())
        }
    }

    struct MockAlignedReaderFactory;

    impl AlignedReaderFactory for MockAlignedReaderFactory {
        type AlignedReaderType = MockAlignedFileReader;

        fn build(&self) -> ANNResult<Self::AlignedReaderType> {
            Ok(MockAlignedFileReader)
        }
    }

    #[test]
    fn test_aligned_reader_factory_trait() {
        let factory = MockAlignedReaderFactory;
        let reader = factory.build();
        
        assert!(reader.is_ok());
    }

    #[test]
    fn test_aligned_reader_factory_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockAlignedReaderFactory>();
    }
}
