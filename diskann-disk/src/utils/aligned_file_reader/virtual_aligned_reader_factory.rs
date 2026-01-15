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
