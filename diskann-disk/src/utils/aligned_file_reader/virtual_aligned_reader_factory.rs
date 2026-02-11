/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::ANNResult;
use diskann_providers::storage::VirtualStorageProvider;
use vfs::{FileSystem, MemoryFS};

use super::{traits::AlignedReaderFactory, StorageProviderAlignedFileReader};

pub struct VirtualAlignedReaderFactory<P: FileSystem = MemoryFS> {
    pub file_path: String,
    // Use Arc instead of reference because async searcher interfaces require 'static bounds
    // for proper lifetime management in async futures
    pub storage_provider: Arc<VirtualStorageProvider<P>>,
}

impl<P: FileSystem> AlignedReaderFactory for VirtualAlignedReaderFactory<P> {
    type AlignedReaderType = StorageProviderAlignedFileReader;

    fn build(&self) -> ANNResult<Self::AlignedReaderType> {
        StorageProviderAlignedFileReader::new(&*self.storage_provider, self.file_path.as_str())
    }
}

impl<P: FileSystem> VirtualAlignedReaderFactory<P> {
    pub fn new(file_path: String, storage_provider: Arc<VirtualStorageProvider<P>>) -> Self {
        Self {
            file_path,
            storage_provider,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_virtual_aligned_reader_factory_new() {
        let fs = Arc::new(VirtualStorageProvider::new(MemoryFS::new()));
        let path = "/test.bin".to_string();
        let factory = VirtualAlignedReaderFactory::new(path.clone(), fs.clone());
        
        assert_eq!(factory.file_path, path);
    }

    #[test]
    fn test_virtual_aligned_reader_factory_implements_trait() {
        // Verify that VirtualAlignedReaderFactory implements AlignedReaderFactory
        fn check_impl<T: AlignedReaderFactory>() {}
        check_impl::<VirtualAlignedReaderFactory<MemoryFS>>();
    }

    #[test]
    fn test_virtual_aligned_reader_factory_field_access() {
        let fs = Arc::new(VirtualStorageProvider::new(MemoryFS::new()));
        let factory = VirtualAlignedReaderFactory::new("/path/to/file".to_string(), fs);
        
        assert_eq!(factory.file_path, "/path/to/file");
    }
}
