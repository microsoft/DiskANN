/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A small utility library for reading and writing Protocol Buffers in binary format.
//!
//! Provides functions to load and save `prost::Message` types using generic
//! storage providers.

use std::io::{Read, Result, Write};

use super::super::{StorageReadProvider, StorageWriteProvider};
use thiserror::Error;

/// Loads a protobuf message of type `S` from a binary file at the given `path`.
///
/// # Parameters
///
/// - `read_provider`: A reference to a type that implements `StorageReadProvider`.
/// - `path`: The file path to read from.
///
/// # Returns
///
/// Returns the decoded protobuf message of type `S`, or a `ProtoStorageError` if
/// an I/O error or decode error occurs.
///
/// # Errors
///
/// - `ProtoStorageError::IoError` if reading the file fails.
/// - `ProtoStorageError::DecodeError` if decoding the bytes into the protobuf struct fails.
pub fn load<P, S>(read_provider: &P, path: &str) -> std::result::Result<S, ProtoStorageError>
where
    P: StorageReadProvider,
    S: prost::Message + Default,
{
    let mut reader = read_provider.open_reader(path)?;
    let mut raw_buffer = Vec::new();
    reader.read_to_end(&mut raw_buffer)?;
    Ok(S::decode(&*raw_buffer)?)
}

/// Saves the given protobuf message `proto_struct` to a binary file at `path`.
///
/// # Parameters
///
/// - `proto_struct`: The protobuf message to encode and save.
/// - `write_provider`: A reference to a type that implements `StorageWriteProvider`.
/// - `path`: The file path to write to.
///
/// # Returns
///
/// Returns the number of bytes written on success, or an I/O `std::io::Error` on failure.
///
/// # Errors
///
/// - Propagates any I/O errors encountered during creation, writing, or flushing of the file.
pub fn save<S, P>(proto_struct: S, write_provider: &P, path: &str) -> Result<usize>
where
    P: StorageWriteProvider,
    S: prost::Message,
{
    let mut writer = write_provider.create_for_write(path)?;
    let encoded_proto = proto_struct.encode_to_vec();
    writer.write_all(&encoded_proto)?;
    writer.flush()?;
    Ok(encoded_proto.len())
}

/// Errors that can occur when reading from or writing to protobuf storage.
#[derive(Debug, Error)]
pub enum ProtoStorageError {
    #[error("Error while creating/opening file {0:?}")]
    IoError(#[from] std::io::Error),
    #[error("Error while decoding bytes to proto struct: {0:?}")]
    DecodeError(#[from] prost::DecodeError),
}

#[cfg(test)]
mod tests {
    use crate::storage::VirtualStorageProvider;
    use prost::Message;

    use super::*;
    use crate::storage::protos::ScalarQuantizer;
    use crate::storage::protos::scalar_quantization::Version;

    #[test]
    fn test_save_and_load_success() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let original = ScalarQuantizer {
            version: Some(Version {
                major: 0,
                minor: 1,
                patch: 0,
            }),
            scale: 1.0,
            shift: vec![0.1, 1.2, 2.3],
            mean_norm: Some(0.5),
            compressed_data_file_name: "compressed_data.bin".to_string(),
        };

        // Save the message
        let bytes_written = save(original.clone(), &storage_provider, "/test.bin").unwrap();
        assert_eq!(bytes_written, original.encode_to_vec().len());

        // Load the message back
        let loaded: ScalarQuantizer = load(&storage_provider, "/test.bin").unwrap();
        assert_eq!(loaded, original);
    }

    #[test]
    fn test_load_invalid_data_returns_decode_error() {
        let storage_provider = VirtualStorageProvider::new_memory();
        // Insert invalid data
        {
            let mut writer = storage_provider.create_for_write("/bad.bin").unwrap();
            writer
                .write_all(&[
                    13, 0, 0, 128, 63, 16, 4, 29, 0, 0, 128, 63, 34, 12, 205, 204,
                ])
                .unwrap();
            writer.flush().unwrap();
        }

        // Loading should result in a DecodeError
        let err = load::<_, ScalarQuantizer>(&storage_provider, "/bad.bin").unwrap_err();
        assert!(matches!(err, ProtoStorageError::DecodeError(_)));
    }

    #[test]
    fn test_load_io_error_when_missing_file() {
        let storage_provider = VirtualStorageProvider::new_memory();

        // Attempt to load a non-existent file should yield IoError
        let err = load::<_, ScalarQuantizer>(&storage_provider, "/missing.bin").unwrap_err();
        assert!(
            matches!(err, ProtoStorageError::IoError(_)),
            "Expected IoError, got: {:?}",
            err
        );
    }
}
