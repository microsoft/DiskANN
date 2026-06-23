/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Save-side context and side-car writer.
//!
//! [`Context`] is the cheap handle handed to every [`super::Save::save`] impl. It owns
//! nothing visible to the user; cloning it is free, and it can be passed to children to
//! propagate the same artifact-tracking state.
//!
//! [`Writer`] is the borrowed side-car artifact handle returned by [`Context::write`].
//! It implements [`std::io::Write`] and [`std::io::Seek`]; calling
//! [`Writer::finish`] flushes the buffer and yields a [`Handle`] that can be inserted
//! into a [`super::Record`].

use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Cursor},
    sync::Mutex,
};

use crate::save::{Error, Handle, Result, Value};

/// The backing store for a save operation.
///
/// A `SaveContext` decides where side-car artifacts are written ([`SaveContext::write`])
/// and how the final manifest is committed ([`SaveContext::finish`]). The concrete
/// implementations live under [`crate::backend`]: a disk-backed context (under the `disk`
/// feature) and an in-memory context. Alternative implementations (e.g. a virtual
/// filesystem) can be supplied for testing or to avoid touching the filesystem.
///
/// The generic [`save`](super::save) entry point is parameterized over this trait so
/// that the base crate carries no hard dependency on any particular implementation.
pub trait SaveContext {
    /// The value produced once the manifest has been committed by
    /// [`SaveContext::finish`]. For the disk-backed context this is `()`.
    type Output;

    /// Allocate a new side-car artifact, optionally tagged with a human-readable `key`.
    ///
    /// The artifact's on-disk name (and the value stored in its [`Handle`]) is prefixed
    /// with the count of artifacts written so far; when `key` is `Some`, it is appended as
    /// `{count}-{key}` purely to aid debugging. Reusing the same `key` across calls is
    /// allowed — the count prefix disambiguates the artifacts.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `key` is `Some` but not a simple relative file name, or if the
    /// underlying artifact cannot be created.
    fn write(&self, key: Option<&str>) -> Result<Writer<'_>>;

    /// Commit the manifest `value`, consuming the context.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the manifest cannot be serialized or committed.
    fn finish(self, value: Value<'_>) -> Result<Self::Output>;
}

/// Object-safe view of the artifact-allocating half of a [`SaveContext`].
///
/// [`Context`] holds a `&dyn GetWrite` so that the same handle can be threaded through
/// every [`Save::save`](super::Save) impl regardless of the concrete context type.
/// [`SaveContext::finish`] (which consumes `self` and names an associated type) is not
/// object safe, so it is deliberately excluded from this trait.
pub(super) trait GetWrite {
    fn write(&self, key: Option<&str>) -> Result<Writer<'_>>;
}

impl<T> GetWrite for T
where
    T: SaveContext,
{
    fn write(&self, key: Option<&str>) -> Result<Writer<'_>> {
        <T as SaveContext>::write(self, key)
    }
}

/// A cheap, clonable handle threaded through every [`Save::save`](super::Save) impl.
///
/// `Context` exposes one operation — [`Context::write`] — for allocating a side-car
/// artifact. The same context is passed to nested [`Save`](super::Save) impls (typically
/// via the [`save_fields!`](crate::save_fields) macro), so a single save tree shares
/// artifact-name bookkeeping. It borrows the backing [`SaveContext`] as an object-safe
/// `GetWrite` so that the save tree is agnostic to the concrete context type.
#[derive(Clone)]
pub struct Context<'a> {
    inner: &'a dyn GetWrite,
}

impl<'a> Context<'a> {
    pub(super) fn new(inner: &'a dyn GetWrite) -> Self {
        Self { inner }
    }

    /// Allocate a new side-car artifact in the manifest directory, optionally tagging it
    /// with a human-readable `key`.
    ///
    /// The artifact is named with the count of artifacts written so far (with `key`, when
    /// `Some`, appended as a readability hint), so the same `key` may be passed to
    /// multiple calls. The returned [`Writer`] is positioned at offset 0 and implements
    /// [`std::io::Write`] / [`std::io::Seek`]. Call [`Writer::finish`] to obtain a
    /// [`Handle`] that may be inserted into a [`Record`](super::Record).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `key` is `Some` but not a simple relative file name, or if the
    /// underlying file cannot be created.
    pub fn write(&self, key: Option<&str>) -> Result<Writer<'_>> {
        self.inner.write(key)
    }
}

/// A borrowed side-car artifact writer produced by [`Context::write`].
///
/// Implements [`std::io::Write`] and [`std::io::Seek`]. Writes are buffered; calling
/// [`Writer::finish`] flushes the buffer (or, for an in-memory context, deposits the
/// completed buffer into the store) and returns a [`Handle`].
#[derive(Debug)]
pub struct Writer<'a> {
    inner: Backend<'a>,
    name: String,
}

/// The backing store a [`Writer`] writes into.
#[derive(Debug)]
enum Backend<'a> {
    /// A file on disk; the bytes are persisted as they are written.
    #[cfg_attr(not(feature = "disk"), allow(dead_code))]
    File(BufWriter<File>),
    /// An in-memory buffer; on [`Writer::finish`] the completed buffer is inserted into
    /// `store` under the writer's name.
    Memory {
        buffer: Cursor<Vec<u8>>,
        store: &'a Mutex<HashMap<String, Vec<u8>>>,
    },
}

impl<'a> Writer<'a> {
    /// Construct an in-memory writer that deposits its buffer into `store` on finish.
    pub(crate) fn memory(name: String, store: &'a Mutex<HashMap<String, Vec<u8>>>) -> Self {
        Self {
            inner: Backend::Memory {
                buffer: Cursor::new(Vec::new()),
                store,
            },
            name,
        }
    }

    /// Construct a file-backed writer that streams bytes straight to `file`.
    #[cfg(feature = "disk")]
    pub(crate) fn file(name: String, file: File) -> Self {
        Self {
            inner: Backend::File(BufWriter::new(file)),
            name,
        }
    }
}

impl Writer<'_> {
    /// Flush and close the writer, returning a [`Handle`] for the artifact.
    ///
    /// Insert the returned handle into a [`Record`](super::Record) (typically via
    /// [`Record::insert`](super::Record::insert)) so that load-side code can locate the
    /// artifact through the manifest.
    pub fn finish(self) -> Result<Handle> {
        match self.inner {
            // NOTE: into_inner() will flush the buffer and close the file.
            Backend::File(io) => {
                io.into_inner()
                    .map_err(|err| Error::new(err.into_error()))?;
            }
            Backend::Memory { buffer, store } => {
                store
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .insert(self.name.clone(), buffer.into_inner());
            }
        }
        Ok(Handle::new(self.name))
    }
}

impl std::io::Write for Writer<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match &mut self.inner {
            Backend::File(io) => io.write(buf),
            Backend::Memory { buffer, .. } => buffer.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match &mut self.inner {
            Backend::File(io) => io.flush(),
            Backend::Memory { buffer, .. } => buffer.flush(),
        }
    }
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        match &mut self.inner {
            Backend::File(io) => io.write_vectored(bufs),
            Backend::Memory { buffer, .. } => buffer.write_vectored(bufs),
        }
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        match &mut self.inner {
            Backend::File(io) => io.write_all(buf),
            Backend::Memory { buffer, .. } => buffer.write_all(buf),
        }
    }
    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> std::io::Result<()> {
        match &mut self.inner {
            Backend::File(io) => io.write_fmt(args),
            Backend::Memory { buffer, .. } => buffer.write_fmt(args),
        }
    }
}

impl std::io::Seek for Writer<'_> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match &mut self.inner {
            Backend::File(io) => io.seek(pos),
            Backend::Memory { buffer, .. } => buffer.seek(pos),
        }
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        match &mut self.inner {
            Backend::File(io) => io.rewind(),
            Backend::Memory { buffer, .. } => buffer.rewind(),
        }
    }
    fn stream_position(&mut self) -> std::io::Result<u64> {
        match &mut self.inner {
            Backend::File(io) => io.stream_position(),
            Backend::Memory { buffer, .. } => buffer.stream_position(),
        }
    }
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
        match &mut self.inner {
            Backend::File(io) => io.seek_relative(offset),
            Backend::Memory { buffer, .. } => buffer.seek_relative(offset),
        }
    }
}
