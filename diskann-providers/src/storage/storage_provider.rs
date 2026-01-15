/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::io::{Read, Result, Seek, Write};

/// This module provides traits and implementations to access a storage system, it could be
/// a file system or an exchange store.
///
/// The `StorageReadProvider` trait is an abstraction for accessing a storage system in a read only manner. It could be
/// implemented with a file system or an exchange store. It provides methods to open a storage
/// for reading, get the length of the storage and check if a storage with a given
/// identifier exists.
///
/// The method is defined in a way that it can work with both file system and BigStorageShim.
///
/// Implementations of this trait will be used by DiskIndexReader to access the storage system.
///
/// The internal Reader associated type of the `StorageReadProvider` trait is defined by the `StorageReader` trait.
pub trait StorageReadProvider: Sync {
    type Reader: Read + Seek;

    /// Open a storage with the given identifier for read.
    fn open_reader(&self, item_identifier: &str) -> Result<Self::Reader>;

    /// Get the length of the storage with the given identifier.
    fn get_length(&self, item_identifier: &str) -> Result<u64>;

    /// Check if the storage with the given identifier exists.
    fn exists(&self, item_identifier: &str) -> bool;
}

/// `StorageWriteProvider` is a trait that abstracts over the ability to write to a storage. Since the ANN algorithm only writes into file system,
/// currently we only have one implementation for this trait based on file system.
pub trait StorageWriteProvider: Sync {
    type Writer: WriteSeek;

    /// Open a storage with the given identifier for write.
    fn open_writer(&self, item_identifier: &str) -> Result<Self::Writer>;

    /// Create a storage with the given identifier for write.
    fn create_for_write(&self, item_identifier: &str) -> Result<Self::Writer>;

    // Deletes a storage item with the given identifier.
    fn delete(&self, item_identifier: &str) -> Result<()>;
}

/// Trait alias for types that implement both `Write` and `Seek`.
///
/// Use this when an API needs a writer that can also move the cursor.
/// Implemented for any type that implements `Write` and `Seek`.
pub trait WriteSeek: Write + Seek {}
impl<T> WriteSeek for T where T: Write + Seek {}

/// Object safe interface for opening and creating writers without exposing a concrete provider type.
///
/// This is useful when passing a writer provider through trait objects or other dynamic
/// boundaries. Methods return boxed writers so callers can use a single uniform interface.
pub trait DynWriteProvider: Sync {
    /// Open an existing item for writing.
    ///
    /// Returns a boxed writer positioned by the provider. Fails if the item does not exist.
    fn open_writer(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>>;

    /// Create a new item for writing.
    ///
    /// Returns a boxed writer for a new item. Behavior if the item already exists depends on the provider.
    fn create_for_write(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>>;

    /// Delete an item identified by `item_identifier`.
    fn delete(&self, item_identifier: &str) -> std::io::Result<()>;
}

impl<T> DynWriteProvider for T
where
    T: StorageWriteProvider,
{
    fn open_writer(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>> {
        self.open_writer(item_identifier)
            .map(|w| Box::new(w) as Box<dyn WriteSeek>)
    }

    fn create_for_write(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>> {
        self.create_for_write(item_identifier)
            .map(|w| Box::new(w) as Box<dyn WriteSeek>)
    }

    fn delete(&self, item_identifier: &str) -> std::io::Result<()> {
        self.delete(item_identifier)
    }
}

/// Adapter that exposes a `&dyn DynWriteProvider` as a `StorageWriteProvider`.
///
/// Useful when an API is generic over `StorageWriteProvider` but the caller only has a dynamic
/// provider. The wrapper forwards all calls to the inner provider and returns boxed writers
/// tied to the wrapper lifetime `'a`.
pub struct WriteProviderWrapper<'a> {
    inner: &'a dyn DynWriteProvider,
}

impl<'a> WriteProviderWrapper<'a> {
    /// Construct a new wrapper around the given dynamic provider reference.
    pub const fn new(inner: &'a dyn DynWriteProvider) -> Self {
        Self { inner }
    }
}

impl<'a> StorageWriteProvider for WriteProviderWrapper<'a> {
    type Writer = Box<dyn WriteSeek + 'a>;

    fn open_writer(&self, item_identifier: &str) -> std::io::Result<Self::Writer> {
        self.inner.open_writer(item_identifier)
    }

    fn create_for_write(&self, item_identifier: &str) -> std::io::Result<Self::Writer> {
        self.inner.create_for_write(item_identifier)
    }

    fn delete(&self, item_identifier: &str) -> std::io::Result<()> {
        self.inner.delete(item_identifier)
    }
}
