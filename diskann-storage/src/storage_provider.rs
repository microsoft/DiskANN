/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Core storage provider traits.
//!
//! These traits abstract over concrete I/O backends so that the rest of the
//! codebase can read and write index data without knowing whether the
//! underlying storage is a local filesystem, an in-memory virtual filesystem,
//! or a remote object store.

use std::io::{Read, Result, Seek, Write};

/// Abstraction for read-only access to a storage backend.
///
/// Implementations may target a local filesystem, an in-memory filesystem, or a
/// remote object store. The associated [`Reader`](Self::Reader) type must
/// support both sequential reads and random seeks.
///
/// # Thread Safety
///
/// Providers are required to be [`Sync`] so that they can be shared across
/// threads. Individual readers are *not* required to be thread-safe.
pub trait StorageReadProvider: Sync {
    /// The reader type returned when opening an item.
    type Reader: Read + Seek;

    /// Open a storage item identified by `item_identifier` for reading.
    fn open_reader(&self, item_identifier: &str) -> Result<Self::Reader>;

    /// Return the size, in bytes, of the item identified by `item_identifier`.
    fn get_length(&self, item_identifier: &str) -> Result<u64>;

    /// Return `true` if an item with the given identifier exists in storage.
    fn exists(&self, item_identifier: &str) -> bool;
}

/// Abstraction for write access to a storage backend.
///
/// The associated [`Writer`](Self::Writer) type supports both sequential writes
/// and random seeks.
///
/// # Thread Safety
///
/// Providers are required to be [`Sync`]; individual writers are not.
pub trait StorageWriteProvider: Sync {
    /// The writer type returned when opening or creating an item.
    type Writer: WriteSeek;

    /// Open an existing storage item for writing.
    fn open_writer(&self, item_identifier: &str) -> Result<Self::Writer>;

    /// Create a new storage item (or truncate an existing one) for writing.
    fn create_for_write(&self, item_identifier: &str) -> Result<Self::Writer>;

    /// Delete the storage item identified by `item_identifier`.
    fn delete(&self, item_identifier: &str) -> Result<()>;
}

/// Trait alias for types that implement both [`Write`] and [`Seek`].
///
/// Automatically implemented for every type satisfying both bounds.
pub trait WriteSeek: Write + Seek {}
impl<T> WriteSeek for T where T: Write + Seek {}

/// Object-safe interface for opening and creating writers without exposing a
/// concrete provider type.
///
/// This is useful when passing a writer provider through trait objects or other
/// dynamic boundaries. Methods return boxed writers so callers can use a single
/// uniform interface.
pub trait DynWriteProvider: Sync {
    /// Open an existing item for writing, returning a boxed writer.
    fn open_writer(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>>;

    /// Create a new item for writing, returning a boxed writer.
    fn create_for_write(&self, item_identifier: &str) -> std::io::Result<Box<dyn WriteSeek + '_>>;

    /// Delete the item identified by `item_identifier`.
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

/// Adapter that exposes a `&dyn DynWriteProvider` as a [`StorageWriteProvider`].
///
/// Useful when an API is generic over [`StorageWriteProvider`] but the caller
/// only has a dynamic provider. The wrapper forwards all calls to the inner
/// provider and returns boxed writers tied to the wrapper lifetime `'a`.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that `WriteProviderWrapper` delegates correctly to the wrapped
    /// `DynWriteProvider` by exercising a round-trip through a concrete
    /// provider.
    #[test]
    fn write_provider_wrapper_round_trip() {
        use crate::VirtualStorageProvider;
        use std::io::{Read, Seek, SeekFrom, Write};

        let vsp = VirtualStorageProvider::new_memory();
        let dyn_provider: &dyn DynWriteProvider = &vsp;
        let wrapper = WriteProviderWrapper::new(dyn_provider);

        // Write via the wrapper using the StorageWriteProvider trait.
        {
            let mut writer =
                StorageWriteProvider::create_for_write(&wrapper, "/round_trip.bin").unwrap();
            writer.write_all(b"hello wrapper").unwrap();
            writer.flush().unwrap();
        }

        // Read back via the original provider.
        let mut reader = vsp.open_reader("/round_trip.bin").unwrap();
        let mut buf = String::new();
        reader.seek(SeekFrom::Start(0)).unwrap();
        reader.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "hello wrapper");
    }

    /// Verify that `DynWriteProvider::delete` propagates correctly.
    #[test]
    fn dyn_write_provider_delete() {
        use crate::VirtualStorageProvider;
        use std::io::Write;

        let vsp = VirtualStorageProvider::new_memory();
        let dyn_provider: &dyn DynWriteProvider = &vsp;

        // Create and then delete.
        {
            let mut w = dyn_provider.create_for_write("/to_delete.bin").unwrap();
            w.write_all(b"tmp").unwrap();
            w.flush().unwrap();
        }
        assert!(vsp.exists("/to_delete.bin"));
        dyn_provider.delete("/to_delete.bin").unwrap();
        assert!(!vsp.exists("/to_delete.bin"));
    }
}
