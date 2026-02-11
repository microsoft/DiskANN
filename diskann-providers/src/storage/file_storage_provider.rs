/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Result},
};

use super::{StorageReadProvider, StorageWriteProvider};

/// FileStorage implements both StorageReadProvider and StorageWriteProvider.
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
    use vfs::FileSystem;

    use super::*;
    use crate::storage::VirtualStorageProvider;

    #[test]
    fn test_file_reader() {
        let file_name = "/test_file_reader.txt";
        let storage_provider = VirtualStorageProvider::new_memory();

        {
            let mut file = storage_provider
                .filesystem()
                .create_file(file_name)
                .expect("Could not create file");
            file.write_all(b"Hello, world!").unwrap();
        }

        let mut reader = storage_provider.open_reader(file_name).unwrap();
        let mut buffer = [0; 5];

        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello");

        reader.seek(SeekFrom::Start(5)).unwrap();
        reader.read(&mut buffer).unwrap();
        assert_eq!(&buffer, b", wor");
    }

    #[test]
    fn test_file_create_write() {
        let file_name = "/test_file_create_write.txt";
        let storage_provider = VirtualStorageProvider::new_memory();

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

        let mut reader = storage_provider.open_reader(file_name).unwrap();
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
    fn test_file_storage_exists() {
        let file_name = "/test_file_storage_exists.txt";
        let storage_provider = VirtualStorageProvider::new_memory();

        assert!(!storage_provider.exists(file_name));
        storage_provider
            .filesystem()
            .create_file(file_name)
            .unwrap();
        assert!(storage_provider.exists(file_name));
    }

    #[test]
    fn test_file_storage_get_length() {
        let file_name = "/test_file_storage_get_length.txt";
        let storage_provider = VirtualStorageProvider::new_memory();

        {
            let mut file = storage_provider
                .filesystem()
                .create_file(file_name)
                .unwrap();
            file.write_all(b"Hello, world!").unwrap();
        }

        assert_eq!(storage_provider.get_length(file_name).unwrap(), 13);
    }
}
