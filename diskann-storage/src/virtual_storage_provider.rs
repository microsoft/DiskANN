/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Virtual storage providers for testing.
//!
//! [`VirtualStorageProvider`] wraps a [`vfs::FileSystem`] implementation so
//! that tests can run entirely in memory or against an overlay filesystem
//! without touching the real disk.

use std::{io, io::Result};

use vfs::{MemoryFS, OverlayFS, PhysicalFS, SeekAndRead, SeekAndWrite, filesystem::FileSystem};

use crate::{StorageReadProvider, StorageWriteProvider};

/// Storage provider backed by an arbitrary [`vfs::FileSystem`].
///
/// Use the factory methods [`new_memory`](VirtualStorageProvider::new_memory),
/// [`new_overlay`](VirtualStorageProvider::new_overlay), or
/// [`new_physical`](VirtualStorageProvider::new_physical) to construct
/// instances for different backends.
///
/// # Examples
///
/// ```
/// use std::io::Write;
/// use vfs::FileSystem;
/// use diskann_storage::VirtualStorageProvider;
///
/// let storage_provider = VirtualStorageProvider::new_memory();
///
/// // Create the root directory.
/// storage_provider.filesystem().create_dir("/test_root").expect("Could not create test directory");
///
/// {
///     let mut file = storage_provider
///         .filesystem()
///         .create_file("/test_root/input_data.bin")
///         .expect("Could not create test file");
///     file.write_all(b"This is test data").expect("Unable to write test data");
/// }
/// ```
pub struct VirtualStorageProvider<FileSystemType: FileSystem> {
    filesystem: FileSystemType,
}

impl<FileSystemType: FileSystem> VirtualStorageProvider<FileSystemType> {
    fn new(filesystem: FileSystemType) -> VirtualStorageProvider<FileSystemType> {
        VirtualStorageProvider { filesystem }
    }

    /// Return `true` if the item identified by `item_identifier` exists.
    pub fn exists(&self, item_identifier: &str) -> bool {
        self.filesystem.metadata(item_identifier).is_ok()
    }

    /// Return a reference to the underlying filesystem.
    pub fn filesystem(&self) -> &FileSystemType {
        &self.filesystem
    }

    /// Consume the provider, returning the underlying filesystem.
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
    /// Create a two-layer overlay filesystem with an in-memory layer for writes
    /// on top of a physical filesystem for reads.
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
    /// Create a storage provider backed by a pure in-memory filesystem.
    pub fn new_memory() -> Self {
        let memory_filesystem = MemoryFS::new();
        VirtualStorageProvider::new(memory_filesystem)
    }
}

impl VirtualStorageProvider<PhysicalFS> {
    /// Create a storage provider that sandboxes physical filesystem access
    /// to the given root path.
    pub fn new_physical<P: AsRef<std::path::Path>>(path: P) -> Self {
        #[allow(clippy::disallowed_methods)]
        let physical_filesystem = PhysicalFS::new(path);
        VirtualStorageProvider::new(physical_filesystem)
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use super::*;

    #[test]
    fn read_after_write_in_memory() {
        let file_name = "/test_file_reader.txt";
        let storage_provider = VirtualStorageProvider::new_memory();

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
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b"Hello");

        reader.seek(SeekFrom::Start(5)).unwrap();
        reader.read_exact(&mut buffer).unwrap();
        assert_eq!(&buffer, b", wor");
    }

    #[test]
    fn exists_reports_correctly() {
        let storage_provider = VirtualStorageProvider::new_memory();

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
    fn get_length_returns_byte_count() {
        let file_name = "/test_file_storage_get_length.txt";

        let storage_provider = VirtualStorageProvider::new_memory();
        storage_provider
            .filesystem()
            .create_file(file_name)
            .expect("Could not create new file")
            .write_all(b"Hello, world!")
            .expect("Write did not succeed");

        assert_eq!(storage_provider.get_length(file_name).unwrap(), 13);
    }

    #[test]
    fn create_for_write_replaces_existing() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/replace_me.bin";

        {
            let mut w = storage_provider.create_for_write(path).unwrap();
            w.write_all(b"original content").unwrap();
            w.flush().unwrap();
        }

        {
            let mut w = storage_provider.create_for_write(path).unwrap();
            w.write_all(b"new").unwrap();
            w.flush().unwrap();
        }

        assert_eq!(storage_provider.get_length(path).unwrap(), 3);
    }

    #[test]
    fn delete_removes_item() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let path = "/to_delete.bin";

        {
            let mut w = storage_provider.create_for_write(path).unwrap();
            w.write_all(b"data").unwrap();
            w.flush().unwrap();
        }
        assert!(storage_provider.exists(path));

        storage_provider.delete(path).unwrap();
        assert!(!storage_provider.exists(path));
    }

    #[test]
    fn open_reader_missing_file_returns_error() {
        let storage_provider = VirtualStorageProvider::new_memory();
        let result = storage_provider.open_reader("/does_not_exist.bin");
        assert!(result.is_err());
    }

    #[test]
    fn trait_based_read_provider() {
        let vsp = VirtualStorageProvider::new_memory();
        {
            let mut w = vsp.create_for_write("/trait_test.bin").unwrap();
            w.write_all(b"via trait").unwrap();
            w.flush().unwrap();
        }

        fn read_via_trait(provider: &dyn StorageReadProvider<Reader = impl Read + Seek>) {
            assert!(provider.exists("/trait_test.bin"));
            assert_eq!(provider.get_length("/trait_test.bin").unwrap(), 9);
        }

        // We can't use the dyn version with associated types directly, but we
        // can verify the impl works through a generic function.
        read_via_trait(&vsp);
    }
}
