/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Adapters that expose a single in-hand [`Read`]/[`Write`] target as a
//! [`StorageReadProvider`]/[`StorageWriteProvider`].
//!
//! These exist to bridge the new `diskann-record` save/load APIs (which produce
//! borrowed `Writer`/`Reader` handles backed by a manifest) and the existing
//! byte-level helpers (which expect a [`StorageReadProvider`] /
//! [`StorageWriteProvider`] and a path).
//!
//! Each adapter is a "single use" thing: it lets the wrapped writer/reader be
//! handed out exactly once, for a single known item name. A second call, or a
//! call with a mismatched name, returns an [`io::Error`]. This is the right
//! contract for our use case — every leaf component's `Save` impl produces one
//! artifact through one helper invocation.
//!
//! When the existing helpers are migrated to a direct `Read`/`Write` API,
//! these adapters (and the underlying `StorageRead/WriteProvider` traits) can
//! be deprecated.

use std::{
    io::{self, Read, Seek, SeekFrom, Write},
    sync::Mutex,
};

use crate::storage::{StorageReadProvider, StorageWriteProvider, WriteSeek};

/// Trait alias for types that implement both [`Read`] and [`Seek`], mirroring
/// the existing [`WriteSeek`] alias used by the write side.
pub trait ReadSeek: Read + Seek {}
impl<T> ReadSeek for T where T: Read + Seek {}

//////////////////////////
// Write side           //
//////////////////////////

/// A [`StorageWriteProvider`] backed by a single borrowed writer.
///
/// The wrapped writer is handed out exactly once via [`create_for_write`]
/// (or [`open_writer`]) for the configured `name`. Any other call — repeated,
/// or with a mismatched name — returns an [`io::Error`].
///
/// [`create_for_write`]: StorageWriteProvider::create_for_write
/// [`open_writer`]: StorageWriteProvider::open_writer
pub struct SingleUseWriteProvider<'w> {
    name: String,
    inner: Mutex<Option<&'w mut (dyn WriteSeek + Send)>>,
}

impl<'w> SingleUseWriteProvider<'w> {
    /// Wrap `writer` so that calls to `create_for_write(name)` (or
    /// `open_writer(name)`) on this provider yield it exactly once.
    pub fn new<W>(name: impl Into<String>, writer: &'w mut W) -> Self
    where
        W: WriteSeek + Send,
    {
        Self {
            name: name.into(),
            inner: Mutex::new(Some(writer)),
        }
    }

    fn take_for(&self, requested: &str) -> io::Result<&'w mut (dyn WriteSeek + Send)> {
        if requested != self.name {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "SingleUseWriteProvider only serves {:?}; got request for {:?}",
                    self.name, requested,
                ),
            ));
        }
        // Lint: PoisonError here would mean the underlying writer is in a
        // broken state from a previous panicked write; surfacing it as an
        // io::Error rather than panicking lets callers handle it.
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("SingleUseWriteProvider lock poisoned"))?;
        guard.take().ok_or_else(|| {
            io::Error::other(format!(
                "SingleUseWriteProvider for {:?} has already been used",
                self.name,
            ))
        })
    }
}

impl std::fmt::Debug for SingleUseWriteProvider<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleUseWriteProvider")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

/// Borrowed writer produced by [`SingleUseWriteProvider`].
pub struct SingleUseWriter<'w>(&'w mut (dyn WriteSeek + Send));

impl std::fmt::Debug for SingleUseWriter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SingleUseWriter(..)")
    }
}

impl Write for SingleUseWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
    fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        self.0.write_vectored(bufs)
    }
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.0.write_all(buf)
    }
    fn write_fmt(&mut self, fmt: std::fmt::Arguments<'_>) -> io::Result<()> {
        self.0.write_fmt(fmt)
    }
}

impl Seek for SingleUseWriter<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.0.seek(pos)
    }
    fn rewind(&mut self) -> io::Result<()> {
        self.0.rewind()
    }
    fn stream_position(&mut self) -> io::Result<u64> {
        self.0.stream_position()
    }
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        self.0.seek_relative(offset)
    }
}

impl<'w> StorageWriteProvider for SingleUseWriteProvider<'w> {
    type Writer = SingleUseWriter<'w>;

    fn open_writer(&self, item_identifier: &str) -> io::Result<Self::Writer> {
        self.take_for(item_identifier).map(SingleUseWriter)
    }

    fn create_for_write(&self, item_identifier: &str) -> io::Result<Self::Writer> {
        self.take_for(item_identifier).map(SingleUseWriter)
    }

    fn delete(&self, _item_identifier: &str) -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "SingleUseWriteProvider does not support delete",
        ))
    }
}

//////////////////////////
// Read side            //
//////////////////////////

/// A [`StorageReadProvider`] backed by a single borrowed reader.
///
/// Symmetric to [`SingleUseWriteProvider`]: hands out the wrapped reader
/// exactly once via [`open_reader`] for the configured `name`.
///
/// [`open_reader`]: StorageReadProvider::open_reader
pub struct SingleUseReadProvider<'r> {
    name: String,
    length: u64,
    inner: Mutex<Option<&'r mut (dyn ReadSeek + Send)>>,
}

impl<'r> SingleUseReadProvider<'r> {
    /// Wrap `reader` so that a call to `open_reader(name)` on this provider
    /// yields it exactly once.
    ///
    /// The constructor probes the reader for its byte length (by seeking to
    /// the end and restoring the original cursor) so that subsequent
    /// [`StorageReadProvider::get_length`] calls can be answered without
    /// re-touching the stream. Fails if either seek fails.
    pub fn new<R>(name: impl Into<String>, reader: &'r mut R) -> io::Result<Self>
    where
        R: ReadSeek + Send,
    {
        let start = reader.stream_position()?;
        let end = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(start))?;
        Ok(Self {
            name: name.into(),
            length: end,
            inner: Mutex::new(Some(reader)),
        })
    }

    fn take_for(&self, requested: &str) -> io::Result<&'r mut (dyn ReadSeek + Send)> {
        if requested != self.name {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "SingleUseReadProvider only serves {:?}; got request for {:?}",
                    self.name, requested,
                ),
            ));
        }
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("SingleUseReadProvider lock poisoned"))?;
        guard.take().ok_or_else(|| {
            io::Error::other(format!(
                "SingleUseReadProvider for {:?} has already been used",
                self.name,
            ))
        })
    }
}

impl std::fmt::Debug for SingleUseReadProvider<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleUseReadProvider")
            .field("name", &self.name)
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

/// Borrowed reader produced by [`SingleUseReadProvider`].
pub struct SingleUseReader<'r>(&'r mut (dyn ReadSeek + Send));

impl std::fmt::Debug for SingleUseReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SingleUseReader(..)")
    }
}

impl Read for SingleUseReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.read(buf)
    }
    fn read_vectored(&mut self, bufs: &mut [io::IoSliceMut<'_>]) -> io::Result<usize> {
        self.0.read_vectored(bufs)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.0.read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> io::Result<usize> {
        self.0.read_to_string(buf)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.0.read_exact(buf)
    }
}

impl Seek for SingleUseReader<'_> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.0.seek(pos)
    }
    fn rewind(&mut self) -> io::Result<()> {
        self.0.rewind()
    }
    fn stream_position(&mut self) -> io::Result<u64> {
        self.0.stream_position()
    }
    fn seek_relative(&mut self, offset: i64) -> io::Result<()> {
        self.0.seek_relative(offset)
    }
}

impl<'r> StorageReadProvider for SingleUseReadProvider<'r> {
    type Reader = SingleUseReader<'r>;

    fn open_reader(&self, item_identifier: &str) -> io::Result<Self::Reader> {
        self.take_for(item_identifier).map(SingleUseReader)
    }

    fn get_length(&self, item_identifier: &str) -> io::Result<u64> {
        if item_identifier != self.name {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "SingleUseReadProvider only knows {:?}; got request for {:?}",
                    self.name, item_identifier,
                ),
            ));
        }
        Ok(self.length)
    }

    fn exists(&self, item_identifier: &str) -> bool {
        item_identifier == self.name
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::io::{Read, Seek, SeekFrom, Write};

    use super::*;

    #[test]
    fn write_provider_routes_one_create_to_the_writer() {
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let provider = SingleUseWriteProvider::new("only.bin", &mut buf);
        {
            let mut w = provider
                .create_for_write("only.bin")
                .expect("first create_for_write");
            w.write_all(b"hello").unwrap();
            w.write_all(b" world").unwrap();
            w.flush().unwrap();
        }
        drop(provider);
        assert_eq!(buf.into_inner(), b"hello world");
    }

    #[test]
    fn write_provider_rejects_second_take() {
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let provider = SingleUseWriteProvider::new("only.bin", &mut buf);
        let _first = provider.create_for_write("only.bin").unwrap();
        let err = provider
            .create_for_write("only.bin")
            .expect_err("second take must fail");
        assert!(
            err.to_string().contains("already been used"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn write_provider_rejects_unknown_name() {
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let provider = SingleUseWriteProvider::new("only.bin", &mut buf);
        let err = provider
            .create_for_write("other.bin")
            .expect_err("mismatched name must fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn write_provider_does_not_support_delete() {
        let mut buf = std::io::Cursor::new(Vec::<u8>::new());
        let provider = SingleUseWriteProvider::new("only.bin", &mut buf);
        let err = provider.delete("only.bin").expect_err("delete must fail");
        assert_eq!(err.kind(), io::ErrorKind::Unsupported);
    }

    #[test]
    fn read_provider_routes_one_open_to_the_reader() {
        let mut src = std::io::Cursor::new(b"abcdefg".to_vec());
        let provider = SingleUseReadProvider::new("only.bin", &mut src).expect("new");

        assert!(provider.exists("only.bin"));
        assert!(!provider.exists("other.bin"));
        assert_eq!(provider.get_length("only.bin").unwrap(), 7);

        let mut r = provider.open_reader("only.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        assert_eq!(out, b"abcdefg");
    }

    #[test]
    fn read_provider_rejects_second_take() {
        let mut src = std::io::Cursor::new(b"abc".to_vec());
        let provider = SingleUseReadProvider::new("only.bin", &mut src).expect("new");
        let _first = provider.open_reader("only.bin").unwrap();
        let err = provider
            .open_reader("only.bin")
            .expect_err("second take must fail");
        assert!(err.to_string().contains("already been used"));
    }

    #[test]
    fn read_provider_preserves_cursor_after_construction() {
        let mut src = std::io::Cursor::new(b"123456789".to_vec());
        src.seek(SeekFrom::Start(2)).unwrap();
        let provider = SingleUseReadProvider::new("only.bin", &mut src).expect("new");
        assert_eq!(provider.get_length("only.bin").unwrap(), 9);
        let mut r = provider.open_reader("only.bin").unwrap();
        let mut out = Vec::new();
        r.read_to_end(&mut out).unwrap();
        // Cursor was restored to position 2, so we should read from there.
        assert_eq!(out, b"3456789");
    }
}
