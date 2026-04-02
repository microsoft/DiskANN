/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Filesystem-backed storage provider.
//!
//! [`FileStorageProvider`] implements both [`StorageReadProvider`] and
//! [`StorageWriteProvider`] using the local filesystem via [`std::fs`].

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Result},
};

use crate::{StorageReadProvider, StorageWriteProvider};

/// Storage provider backed by the local filesystem.
///
/// Each `item_identifier` is interpreted as a filesystem path. Readers are
/// buffered via [`BufReader`] and writers via [`BufWriter`].
#[derive(Default)]
pub struct FileStorageProvider;

impl StorageReadProvider for FileStorageProvider {
    type Reader = BufReader<File>;

    fn open_reader(&self, item_identifier: &str) -> Result<Self::Reader> {
        let f = File::open(item_identifier)?;
        Ok(BufReader::new(f))
    }

    fn get_length(&self, item_identifier: &str) -> Result<u64> {
        let metadata = fs::metadata(item_identifier)?;
        Ok(metadata.len())
    }

    fn exists(&self, item_identifier: &str) -> bool {
        fs::metadata(item_identifier).is_ok()
    }
}

impl StorageWriteProvider for FileStorageProvider {
    type Writer = BufWriter<File>;

    fn open_writer(&self, item_identifier: &str) -> Result<Self::Writer> {
        let f = OpenOptions::new().write(true).open(item_identifier)?;
        Ok(BufWriter::new(f))
    }

    fn create_for_write(&self, item_identifier: &str) -> Result<Self::Writer> {
        let f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(item_identifier)?;
        Ok(BufWriter::new(f))
    }

    fn delete(&self, item_identifier: &str) -> Result<()> {
        fs::remove_file(item_identifier)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use tempfile::TempDir;

    use super::*;

    #[test]
    fn read_after_write() {
        let tmp_dir =
            TempDir::with_prefix("test_file_reader").expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("test_file_reader.txt");
        let file_name = file_path.to_str().unwrap();

        let mut file = File::create(file_name).unwrap();
        file.write_all(b"Hello, world!").unwrap();

        let mut reader = FileStorageProvider.open_reader(file_name).unwrap();
        let mut buffer = [0; 5];

        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello");

        reader.seek(SeekFrom::Start(5)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b", wor");
    }

    #[test]
    fn create_write_and_append() {
        let storage_provider = FileStorageProvider;

        let tmp_dir = TempDir::with_prefix("test_file_create_write")
            .expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("test_file_create_write.txt");
        let file_name = file_path.to_str().unwrap();

        assert!(!storage_provider.exists(file_name));
        {
            let mut file = storage_provider.create_for_write(file_name).unwrap();
            file.write_all(b"Hello, world! ").unwrap();
            file.flush().unwrap();
        }

        assert!(storage_provider.exists(file_name));

        {
            let mut file = storage_provider.open_writer(file_name).unwrap();
            file.seek(SeekFrom::End(0)).unwrap();
            file.write_all(b"This is the second write! ").unwrap();
            file.write_all(b"This is the third write!").unwrap();
            file.flush().unwrap();
        }

        let expected = b"Hello, world! This is the second write! This is the third write!";

        let mut reader = FileStorageProvider.open_reader(file_name).unwrap();
        let mut file_data: Vec<u8> = Vec::new();
        let read_size = reader.read_to_end(&mut file_data).unwrap();

        assert_eq!(
            expected.len(),
            read_size,
            "Did not read the expected number of bytes"
        );
        assert_eq!(expected, file_data.as_slice());
    }

    #[test]
    fn exists_check() {
        let tmp_dir = TempDir::with_prefix("test_file_storage_exists")
            .expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("test_file_storage_exists.txt");
        let file_name = file_path.to_str().unwrap();

        assert!(!FileStorageProvider.exists(file_name));
        File::create(file_name).unwrap();
        assert!(FileStorageProvider.exists(file_name));
    }

    #[test]
    fn get_length() {
        let tmp_dir = TempDir::with_prefix("test_file_storage_get_length")
            .expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("test_file_storage_get_length.txt");
        let file_name = file_path.to_str().unwrap();

        let mut file = File::create(file_name).unwrap();
        file.write_all(b"Hello, world!").unwrap();

        assert_eq!(FileStorageProvider.get_length(file_name).unwrap(), 13);
    }

    #[test]
    fn delete_removes_file() {
        let tmp_dir = TempDir::with_prefix("test_file_storage_delete")
            .expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("to_delete.txt");
        let file_name = file_path.to_str().unwrap();

        File::create(file_name).unwrap();
        assert!(FileStorageProvider.exists(file_name));

        FileStorageProvider.delete(file_name).unwrap();
        assert!(!FileStorageProvider.exists(file_name));
    }

    #[test]
    fn create_for_write_truncates_existing() {
        let tmp_dir =
            TempDir::with_prefix("test_truncate").expect("Failed to create temporary directory");
        let file_path = tmp_dir.path().join("truncate.txt");
        let file_name = file_path.to_str().unwrap();

        // Write long content.
        {
            let mut w = FileStorageProvider.create_for_write(file_name).unwrap();
            w.write_all(b"long initial content that should be truncated")
                .unwrap();
            w.flush().unwrap();
        }

        // Overwrite with short content — must truncate.
        {
            let mut w = FileStorageProvider.create_for_write(file_name).unwrap();
            w.write_all(b"short").unwrap();
            w.flush().unwrap();
        }

        let len = FileStorageProvider.get_length(file_name).unwrap();
        assert_eq!(len, 5, "create_for_write should truncate the file");
    }

    #[test]
    fn open_reader_nonexistent_returns_error() {
        let err = FileStorageProvider.open_reader("/nonexistent/path/file.bin");
        assert!(err.is_err());
    }

    #[test]
    fn open_writer_nonexistent_returns_error() {
        let err = FileStorageProvider.open_writer("/nonexistent/path/file.bin");
        assert!(err.is_err());
    }

    #[test]
    fn delete_nonexistent_returns_error() {
        let err = FileStorageProvider.delete("/nonexistent/path/file.bin");
        assert!(err.is_err());
    }
}
