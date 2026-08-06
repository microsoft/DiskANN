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

use std::io::BufWriter;

use crate::save::{Error, Handle, Result, Value};

/// Generate forwarding [`std::io::Write`] and [`std::io::Seek`] impls for `$T` that
/// delegate every method to its `$field` member.
macro_rules! delegate_write_and_seek {
    ($field:ident, $T:ty) => {
        impl std::io::Write for $T {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.$field.write(buf)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.$field.flush()
            }
            fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
                self.$field.write_vectored(bufs)
            }
            fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
                self.$field.write_all(buf)
            }
            fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> std::io::Result<()> {
                self.$field.write_fmt(args)
            }
        }

        impl std::io::Seek for $T {
            fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
                self.$field.seek(pos)
            }
            fn rewind(&mut self) -> std::io::Result<()> {
                self.$field.rewind()
            }
            fn stream_position(&mut self) -> std::io::Result<u64> {
                self.$field.stream_position()
            }
            fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
                self.$field.seek_relative(offset)
            }
        }
    };
}

pub(crate) use delegate_write_and_seek;

/// The backing store for a save operation.
///
/// A `SaveContext` decides where side-car artifacts are written ([`SaveContext::write`])
/// and how the final manifest is committed ([`SaveContext::finish`]). The concrete
/// implementations live under `crate::backend`: a disk-backed context (under the `disk`
/// feature) and an in-memory context. Alternative implementations (e.g. a virtual
/// filesystem) can be supplied for testing or to avoid touching the filesystem.
///
/// The generic [`save`](super::save) entry point is parameterized over this trait so
/// that the base crate carries no hard dependency on any particular implementation.
pub(crate) trait SaveContext {
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
/// artifact-name bookkeeping. It borrows the backing `SaveContext` as an object-safe
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

/// The backend-specific half of a [`Writer`].
///
/// Each [`SaveContext`](super::SaveContext) implementation supplies its own
/// `WriterInner` (e.g. an in-memory cursor or an on-disk file). [`Writer`] wraps it in a
/// [`BufWriter`] and forwards [`std::io::Write`] / [`std::io::Seek`] to it; on
/// [`Writer::finish`] the (flushed) inner is consumed to commit the artifact and yield a
/// [`Handle`].
pub(crate) trait WriterInner: std::io::Write + std::io::Seek + std::fmt::Debug {
    /// Commit the completed artifact under `name`, returning its [`Handle`].
    fn finish(self: Box<Self>, name: String) -> Result<Handle>;
}

/// A borrowed side-car artifact writer produced by [`Context::write`].
///
/// Implements [`std::io::Write`] and [`std::io::Seek`]. Writes are buffered; calling
/// [`Writer::finish`] flushes the buffer, commits the artifact through the backing
/// writer, and returns a [`Handle`].
#[derive(Debug)]
pub struct Writer<'a> {
    inner: BufWriter<Box<dyn WriterInner + 'a>>,
    name: String,
}

impl<'a> Writer<'a> {
    /// Wrap a backend-specific [`WriterInner`] into a buffered [`Writer`] named `name`.
    pub(crate) fn new<T>(inner: T, name: String) -> Self
    where
        T: WriterInner + 'a,
    {
        Self {
            inner: BufWriter::new(Box::new(inner)),
            name,
        }
    }

    /// Flush and close the writer, returning a [`Handle`] for the artifact.
    ///
    /// Insert the returned handle into a [`Record`](super::Record) (typically via
    /// [`Record::insert`](super::Record::insert)) so that load-side code can locate the
    /// artifact through the manifest.
    pub fn finish(self) -> Result<Handle> {
        // into_inner() flushes the buffered bytes into the backend before we commit it.
        let inner = self
            .inner
            .into_inner()
            .map_err(|err| Error::new(err.into_error()))?;
        inner.finish(self.name)
    }
}

delegate_write_and_seek!(inner, Writer<'_>);
