/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Virtual storage providers for testing.
//!
//! This module provides test utilities that allow using in-memory or overlay filesystems
//! instead of the real filesystem. This is useful for keeping test data in-memory and
//! avoiding filesystem side-effects during testing.

use std::{io, io::Result};

use vfs::{MemoryFS, OverlayFS, PhysicalFS, SeekAndRead, SeekAndWrite, filesystem::FileSystem};

use super::{StorageReadProvider, StorageWriteProvider};

/// VirtualStorageProvider implements both StorageReadProvider and StorageWriteProvider.
/// This is a test utility and is not intended for use in production code.  This allows us to
/// specify alternate filesystems to make testing easier.
///
/// # Examples:
///
/// Use an in-memory filesystem instead of the normal filesystem to keep all test data in-memory
/// ```
/// use vfs::FileSystem;
/// use diskann_providers::storage::VirtualStorageProvider;
/// let storage_provider = VirtualStorageProvider::new_memory();
///
/// // Create the root directory
/// storage_provider.filesystem().create_dir("/test_root").expect("Could not create test directory");
///
/// {
///     // Write test data to the in-memory filesystem inside a scope block so that the writer
///     // is flushed and disposed before using the storage_provider.
///     let mut file = storage_provider.filesystem().create_file("/test_root/input_data.bin").expect("Could not create test file");
///     file.write_all(b"This is test data").expect("Unable to write test data");
/// }
/// ```
///
/// Use the physical filesystem with custom root instead of the normal filesystem.  This prevents
/// the test from writing outside of the expected sandbox
/// ```
/// use vfs::FileSystem;
/// use diskann_providers::storage::VirtualStorageProvider;
///
/// // Use your own path, not the target directory
/// let storage_provider = VirtualStorageProvider::new_physical("../target");
///
/// {
///     // Write test data to the filesystem inside a scope block so that the writer
///     // is flushed and disposed before using the storage_provider. On the local filesystem
///     // input_data.bin will be written to ../target/input_data.bin.
///     let mut file = storage_provider.filesystem().create_file("/input_data.bin").expect("Could not create test file");
///     file.write_all(b"This is test data").expect("Unable to write test data");
/// }
/// ```
///
/// Use the overlay filesystem to read from the local filesystem and write to the in-memory filesystem
/// ```
/// use vfs::FileSystem;
/// use diskann_providers::storage::VirtualStorageProvider;
///
/// // Create a storage provider with an overlay filesystem
/// // This will read data from the ../target/my_data_location path and write to memory.
/// let storage_provider = VirtualStorageProvider::new_overlay("../target/my_data_location");
///
/// storage_provider.filesystem().create_dir("/test_data").expect("Could not create test directory");
///
/// {
///     // Write test data to the in-memory filesystem inside a scope block so that the writer
///     // is flushed and disposed before using the storage_provider.
///     let mut file = storage_provider.filesystem().create_file("/test_data/input_data.bin").expect("Could not create test file");
///     file.write_all(b"This is test data").expect("Unable to write test data");
/// }
///
/// // Storage provider will read from memory & local filesystem but will only write to memory.
/// ```
pub struct VirtualStorageProvider<FileSystemType: FileSystem> {
    filesystem: FileSystemType,
}

impl<FileSystemType: FileSystem> VirtualStorageProvider<FileSystemType> {
    pub(crate) fn new(filesystem: FileSystemType) -> VirtualStorageProvider<FileSystemType> {
        VirtualStorageProvider { filesystem }
    }

    pub fn exists(&self, item_identifier: &str) -> bool {
        self.filesystem.metadata(item_identifier).is_ok()
    }

    /// Return a reference to the underlying filesystem.
    pub fn filesystem(&self) -> &FileSystemType {
        &self.filesystem
    }

    /// Consume the storage provider, returning the underlying filesystem.
    pub fn take(self) -> FileSystemType {
        self.filesystem
    }
}

impl<FileSystemType: FileSystem> StorageReadProvider for VirtualStorageProvider<FileSystemType> {
    type Reader = Box<dyn SeekAndRead + Send>;

    fn open_reader(&self, item_identifier: &str) -> Result<Self::Reader> {
        self.filesystem
            .open_file(item_identifier)
            .map_err(io::Error::other)
    }

    fn get_length(&self, item_identifier: &str) -> Result<u64> {
        self.filesystem
            .metadata(item_identifier)
            .map_err(io::Error::other)
            .map(|metadata| metadata.len)
    }

    fn exists(&self, item_identifier: &str) -> bool {
        self.filesystem.metadata(item_identifier).is_ok()
    }
}

impl<FileSystemType: FileSystem> StorageWriteProvider for VirtualStorageProvider<FileSystemType> {
    type Writer = Box<dyn SeekAndWrite + Send>;

    fn open_writer(&self, item_identifier: &str) -> Result<Self::Writer> {
        self.filesystem
            .append_file(item_identifier)
            .map_err(io::Error::other)
    }

    fn create_for_write(&self, item_identifier: &str) -> Result<Self::Writer> {
        if self
            .filesystem
            .exists(item_identifier)
            .map_err(io::Error::other)?
        {
            self.filesystem
                .remove_file(item_identifier)
                .map_err(io::Error::other)?;
        }

        self.filesystem
            .create_file(item_identifier)
            .map_err(io::Error::other)
    }

    fn delete(&self, item_identifier: &str) -> Result<()> {
        self.filesystem
            .remove_file(item_identifier)
            .map_err(io::Error::other)
    }
}

impl VirtualStorageProvider<OverlayFS> {
    /// Create a two-layer overlay filesystem with an in-memory filesystem for writes
    /// on top of the physical filesystem for reads.
    pub fn new_overlay<P: AsRef<std::path::Path>>(path: P) -> Self {
        #[allow(clippy::disallowed_methods)]
        let base_filesystem = PhysicalFS::new(path);
        let memory_filesystem = MemoryFS::new();
        let overlay_filesystem =
            OverlayFS::new(&[memory_filesystem.into(), base_filesystem.into()]);

        VirtualStorageProvider::new(overlay_filesystem)
    }
}

impl VirtualStorageProvider<MemoryFS> {
    /// Create a storage provider that uses an in-memory filesystem.
    pub fn new_memory() -> Self {
        let memory_filesystem = MemoryFS::new();

        VirtualStorageProvider::new(memory_filesystem)
    }
}

impl VirtualStorageProvider<PhysicalFS> {
    /// Create a storage provider that uses the physical filesystem with a custom root path.
    /// This prevents operations from writing outside of the specified sandbox.
    pub fn new_physical<P: AsRef<std::path::Path>>(path: P) -> Self {
        #[allow(clippy::disallowed_methods)]
        let physical_filesystem = PhysicalFS::new(path);

        VirtualStorageProvider::new(physical_filesystem)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use vfs::MemoryFS;

    use super::*;

    #[test]
    fn test_file_reader() {
        let file_name = "/test_file_reader.txt";
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());

        {
            let mut file = storage_provider
                .filesystem()
                .create_file(file_name)
                .expect("Could not create file");
            write!(file, "Hello, world!").unwrap();
        }

        let mut reader = storage_provider.open_reader(file_name).unwrap();
        let mut buffer = [0; 5];

        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello");

        reader.seek(SeekFrom::Start(5)).unwrap();
        reader.read(&mut buffer).unwrap();
        assert_eq!(&buffer, b", wor");
        storage_provider.take().remove_file(file_name).unwrap();
    }

    #[test]
    fn test_file_storage_exists() {
        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());

        let file_name = "/test_file_storage_exists.txt";
        assert!(
            !storage_provider.exists(file_name),
            "File should not exist yet"
        );

        storage_provider
            .filesystem()
            .create_file(file_name)
            .expect("Could not create file")
            .write_all(b"This is the text")
            .expect("Write did not succeed");

        // Make sure the file exists
        assert!(
            storage_provider.exists(file_name),
            "New file does not exist"
        );

        storage_provider
            .filesystem()
            .remove_file(file_name)
            .unwrap();
        assert!(
            !storage_provider.exists(file_name),
            "New file was not deleted"
        );
    }

    #[test]
    fn test_file_storage_get_length() {
        let file_name = "/test_file_storage_get_length.txt";

        let storage_provider = VirtualStorageProvider::new(MemoryFS::default());
        storage_provider
            .filesystem()
            .create_file(file_name)
            .expect("Could not create new file")
            .write_all(b"Hello, world!")
            .expect("Write did not succeed");

        assert_eq!(storage_provider.get_length(file_name).unwrap(), 13);
        storage_provider.take().remove_file(file_name).unwrap();
    }
}
