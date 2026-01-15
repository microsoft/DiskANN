/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Result;

use super::{StorageReadProvider, StorageWriteProvider};
use diskann_quantization::scalar::ScalarQuantizer;

use super::protos;
use crate::model::graph::provider::async_::inmem::SQError;

/// The suffix for the compressed SQ vectors file.
const COMPRESSED_DATA_FILE_NAME_SUFFIX: &str = "sq_compressed.bin";
/// The suffix for the Scalar quantizer metadata file.
const QUANTIZER_DATA_FILE_NAME_SUFFIX: &str = "scalar_quantizer_proto.bin";

#[derive(Debug)]
pub struct SQStorage {
    /// Path to the scalar compressed data file
    compressed_data_path: String,

    /// Path to the quantizer file
    quantizer_path: String,
}

impl SQStorage {
    /// Constructs a new `SQStorage` from a prefix.
    pub fn new(prefix: &str) -> Self {
        Self {
            compressed_data_path: format!("{}_{}", prefix, COMPRESSED_DATA_FILE_NAME_SUFFIX),
            quantizer_path: format!("{}_{}", prefix, QUANTIZER_DATA_FILE_NAME_SUFFIX),
        }
    }

    /// Returns the path to the scalar compressed data file.
    pub fn compressed_data_path(&self) -> &str {
        &self.compressed_data_path
    }

    /// Returns the path to the quantizer file.
    pub fn quantizer_path(&self) -> &str {
        &self.quantizer_path
    }

    pub fn save_quantizer<Storage>(
        &self,
        quantizer: &ScalarQuantizer,
        write_provider: &Storage,
    ) -> Result<usize>
    where
        Storage: StorageWriteProvider,
    {
        let quantizer_proto =
            protos::ScalarQuantizer::from(quantizer, self.compressed_data_path().to_string());
        protos::save(quantizer_proto, write_provider, self.quantizer_path())
    }

    pub fn load_quantizer<Storage>(
        &self,
        read_provider: &Storage,
    ) -> std::result::Result<ScalarQuantizer, SQError>
    where
        Storage: StorageReadProvider,
    {
        let quantizer_proto: protos::ScalarQuantizer =
            protos::load(read_provider, self.quantizer_path())?;
        Ok(ScalarQuantizer::try_from(quantizer_proto)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::VirtualStorageProvider;
    use vfs::MemoryFS;

    use super::*;

    #[test]
    fn new_constructs_correct_paths() {
        let storage = SQStorage::new("my_prefix");
        assert_eq!(
            storage.compressed_data_path(),
            "my_prefix_sq_compressed.bin"
        );
        assert_eq!(
            storage.quantizer_path(),
            "my_prefix_scalar_quantizer_proto.bin"
        );
    }

    #[test]
    fn getters_return_references_to_internal_strings() {
        let storage = SQStorage::new("foo");
        let comp = storage.compressed_data_path();
        let quant = storage.quantizer_path();

        assert_eq!(comp, storage.compressed_data_path());
        assert_eq!(quant, storage.quantizer_path());
    }

    #[test]
    fn empty_prefix_still_constructs_paths_with_leading_underscore() {
        let storage = SQStorage::new("");
        assert_eq!(storage.compressed_data_path(), "_sq_compressed.bin");
        assert_eq!(storage.quantizer_path(), "_scalar_quantizer_proto.bin");
    }

    #[test]
    fn prefix_with_slashes_is_handled_as_literal() {
        let storage = SQStorage::new("dir/subdir/file");
        assert_eq!(
            storage.compressed_data_path(),
            "dir/subdir/file_sq_compressed.bin"
        );
        assert_eq!(
            storage.quantizer_path(),
            "dir/subdir/file_scalar_quantizer_proto.bin"
        );
    }

    #[test]
    fn save_and_load_quantizer_roundtrip() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        let sq_storage = SQStorage::new("/roundtrip");
        let quantizer = ScalarQuantizer::new(1.0, vec![0.0, 1.0, 2.0], None);

        // Save should succeed and return number of bytes written
        let bytes_written = sq_storage
            .save_quantizer(&quantizer, &storage_provider)
            .expect("save_quantizer should succeed");
        assert!(bytes_written > 0);

        // Load should succeed and return a ScalarQuantizer
        let loaded_quantizer = sq_storage
            .load_quantizer(&storage_provider)
            .expect("load_quantizer should succeed");

        assert_eq!(quantizer.compare(&loaded_quantizer), Ok(()));
    }
}
