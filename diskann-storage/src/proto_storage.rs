/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Generic protobuf I/O backed by [`StorageReadProvider`] / [`StorageWriteProvider`].
//!
//! Provides [`load_proto`] and [`save_proto`] for reading and writing any
//! [`prost::Message`] type through the storage abstraction layer.

use std::io::{Read, Result, Write};

use crate::{StorageReadProvider, StorageWriteProvider};
use thiserror::Error;

/// Load a protobuf message of type `S` from the item at `path`.
///
/// The item is read in its entirety and then decoded via [`prost::Message::decode`].
///
/// # Errors
///
/// Returns [`ProtoStorageError::IoError`] if reading fails, or
/// [`ProtoStorageError::DecodeError`] if the bytes cannot be decoded.
pub fn load_proto<P, S>(read_provider: &P, path: &str) -> std::result::Result<S, ProtoStorageError>
where
    P: StorageReadProvider,
    S: prost::Message + Default,
{
    let mut reader = read_provider.open_reader(path)?;
    let mut raw_buffer = Vec::new();
    reader.read_to_end(&mut raw_buffer)?;
    Ok(S::decode(&*raw_buffer)?)
}

/// Save a protobuf message to the item at `path`.
///
/// The message is encoded via [`prost::Message::encode_to_vec`] and written
/// through the provider. Returns the number of bytes written on success.
///
/// # Errors
///
/// Propagates any I/O errors encountered during creation, writing, or flushing.
pub fn save_proto<S, P>(proto_struct: S, write_provider: &P, path: &str) -> Result<usize>
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
    /// An I/O error occurred while accessing the underlying storage.
    #[error("I/O error while accessing protobuf storage: {0}")]
    IoError(#[from] std::io::Error),

    /// The bytes read from storage could not be decoded as the expected
    /// protobuf message type.
    #[error("failed to decode protobuf message: {0}")]
    DecodeError(#[from] prost::DecodeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualStorageProvider;
    use prost::Message;

    /// A minimal protobuf message for testing round-trips.
    ///
    /// Generated-style struct (in production these come from `prost-build`).
    #[derive(Clone, PartialEq, Message)]
    struct TestMessage {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(uint32, tag = "2")]
        value: u32,
    }

    #[test]
    fn save_and_load_round_trip() {
        let storage = VirtualStorageProvider::new_memory();
        let original = TestMessage {
            name: "hello".to_string(),
            value: 42,
        };

        let bytes_written = save_proto(original.clone(), &storage, "/test.bin").unwrap();
        assert_eq!(bytes_written, original.encode_to_vec().len());

        let loaded: TestMessage = load_proto(&storage, "/test.bin").unwrap();
        assert_eq!(loaded, original);
    }

    #[test]
    fn load_invalid_data_returns_decode_error() {
        let storage = VirtualStorageProvider::new_memory();
        {
            let mut writer = storage.create_for_write("/bad.bin").unwrap();
            // Write garbage bytes that don't form a valid protobuf.
            writer.write_all(&[0xFF, 0xFF, 0xFF, 0xFF]).unwrap();
            writer.flush().unwrap();
        }

        let err = load_proto::<_, TestMessage>(&storage, "/bad.bin").unwrap_err();
        assert!(
            matches!(err, ProtoStorageError::DecodeError(_)),
            "expected DecodeError, got: {err:?}"
        );
    }

    #[test]
    fn load_missing_file_returns_io_error() {
        let storage = VirtualStorageProvider::new_memory();
        let err = load_proto::<_, TestMessage>(&storage, "/missing.bin").unwrap_err();
        assert!(
            matches!(err, ProtoStorageError::IoError(_)),
            "expected IoError, got: {err:?}"
        );
    }

    #[test]
    fn save_returns_correct_byte_count() {
        let storage = VirtualStorageProvider::new_memory();
        let msg = TestMessage {
            name: "count_test".to_string(),
            value: 99,
        };
        let expected_len = msg.encode_to_vec().len();
        let actual = save_proto(msg, &storage, "/count.bin").unwrap();
        assert_eq!(actual, expected_len);
    }

    #[test]
    fn save_overwrites_existing_item() {
        let storage = VirtualStorageProvider::new_memory();

        let msg1 = TestMessage {
            name: "first".to_string(),
            value: 1,
        };
        save_proto(msg1, &storage, "/overwrite.bin").unwrap();

        let msg2 = TestMessage {
            name: "second".to_string(),
            value: 2,
        };
        save_proto(msg2.clone(), &storage, "/overwrite.bin").unwrap();

        let loaded: TestMessage = load_proto(&storage, "/overwrite.bin").unwrap();
        assert_eq!(loaded, msg2);
    }

    #[test]
    fn empty_message_round_trip() {
        let storage = VirtualStorageProvider::new_memory();
        let empty = TestMessage {
            name: String::new(),
            value: 0,
        };

        save_proto(empty.clone(), &storage, "/empty.bin").unwrap();
        let loaded: TestMessage = load_proto(&storage, "/empty.bin").unwrap();
        assert_eq!(loaded, empty);
    }
}
