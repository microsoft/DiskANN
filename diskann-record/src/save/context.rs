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

use std::{fs::File, io::BufWriter};

#[cfg(feature = "disk")]
use std::{collections::HashSet, path::PathBuf, sync::Mutex};

use crate::save::{Error, Handle, Result, Value};

/// The backing store for a save operation.
///
/// A `SaveContext` decides where side-car artifacts are written ([`SaveContext::write`])
/// and how the final manifest is committed ([`SaveContext::finish`]). The default,
/// disk-backed implementation (`DiskContext`) lives in this module under the `disk`
/// feature; alternative implementations (e.g. a virtual filesystem or a purely in-memory
/// store) can be supplied for testing or to avoid touching the filesystem.
///
/// The generic [`save`](super::save) entry point is parameterized over this trait so
/// that the base crate carries no hard dependency on any particular implementation.
pub trait SaveContext {
    /// The value produced once the manifest has been committed by
    /// [`SaveContext::finish`]. For the disk-backed context this is `()`.
    type Output;

    /// Allocate a new side-car artifact named `key`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `key` is not a simple relative file name, if it has already
    /// been registered, or if the underlying artifact cannot be created.
    fn write(&self, key: &str) -> Result<Writer<'_>>;

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
    fn write(&self, key: &str) -> Result<Writer<'_>>;
}

impl<T> GetWrite for T
where
    T: SaveContext,
{
    fn write(&self, key: &str) -> Result<Writer<'_>> {
        <T as SaveContext>::write(self, key)
    }
}

/// The disk-backed [`SaveContext`].
///
/// Holds the manifest directory, the manifest path, and the set of artifact file names
/// registered so far. Lookup and insertion go through a [`Mutex`] so that concurrent
/// [`Save`](super::Save) impls cannot accidentally hand out the same artifact name twice.
#[cfg(feature = "disk")]
#[derive(Debug)]
pub(super) struct DiskContext {
    dir: PathBuf,
    metadata: PathBuf,
    files: Mutex<HashSet<String>>,
}

#[cfg(feature = "disk")]
#[derive(serde::Serialize)]
struct Final<'a> {
    files: Vec<&'a str>,
    value: &'a Value<'a>,
}

#[cfg(feature = "disk")]
impl DiskContext {
    // TODO: Error if the directory looks bad?
    pub(super) fn new(dir: PathBuf, metadata: PathBuf) -> Self {
        Self {
            dir,
            metadata,
            files: Mutex::new(HashSet::new()),
        }
    }
}

#[cfg(feature = "disk")]
impl SaveContext for DiskContext {
    type Output = ();

    fn write(&self, key: &str) -> Result<Writer<'_>> {
        // Reject absolute paths, parent traversal, and multi-component paths. Handles must be
        // simple file names relative to the manifest directory.
        let mut components = std::path::Path::new(key).components();
        match components.next() {
            Some(std::path::Component::Normal(_)) if components.next().is_none() => {}
            _ => {
                return Err(Error::message(format!(
                    "artifact file name {:?} must be a relative file name with no path separators",
                    key,
                )));
            }
        }

        // TODO: Proper disambiguation - making UUIDs etc.
        let mut files = self
            .files
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !files.insert(key.into()) {
            return Err(Error::message(format!(
                "file name {:?} has already been registered with this save context",
                key,
            )));
        }
        let full = self.dir.join(key);
        if full.exists() {
            return Err(Error::message(format!(
                "file {} already exists",
                full.display()
            )));
        }
        let file = std::fs::File::create_new(&full).map_err(|err| {
            Error::new(err).context(format!("while creating new file {}", full.display()))
        })?;
        Ok(Writer {
            io: BufWriter::new(file),
            name: key.into(),
            _lifetime: std::marker::PhantomData,
        })
    }

    /// Finalize the manifest.
    ///
    /// Writes the manifest JSON atomically: serializes to a `<metadata>.temp` file first,
    /// then renames it into place. Fails if the temp file already exists (an in-flight
    /// save is in progress, or a previous run aborted between rename steps).
    fn finish(self, value: Value<'_>) -> Result<()> {
        let files = self
            .files
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner());
        let f = Final {
            files: files.iter().map(|k| &**k).collect(),
            value: &value,
        };

        // Fail if the temp file already exists
        let mut temp = self.metadata.clone().into_os_string();
        temp.push(".temp");
        let temp = PathBuf::from(temp);
        let buffer = std::fs::File::create_new(&temp).map_err(|err| {
            if err.kind() == std::io::ErrorKind::AlreadyExists {
                Error::message(format!(
                    "Temporary file {} already exists. Aborting!",
                    temp.display()
                ))
            } else {
                Error::new(err).context(format!(
                    "while creating temp manifest file {}",
                    temp.display()
                ))
            }
        })?;

        serde_json::to_writer_pretty(buffer, &f)
            .map_err(|err| Error::new(err).context("while serializing manifest to JSON"))?;
        std::fs::rename(&temp, &self.metadata).map_err(|err| {
            Error::new(err).context(format!(
                "while renaming temp manifest {} to final path {}",
                temp.display(),
                self.metadata.display()
            ))
        })?;
        Ok(())
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

    /// Allocate a new side-car artifact named `key` in the manifest directory.
    ///
    /// The returned [`Writer`] is positioned at offset 0 and implements
    /// [`std::io::Write`] / [`std::io::Seek`]. Call [`Writer::finish`] to obtain a
    /// [`Handle`] that may be inserted into a [`Record`](super::Record).
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `key` has already been registered with this context (names
    /// must be unique within a single save), or if the underlying file cannot be created
    /// (e.g. because the artifact already exists on disk).
    pub fn write(&self, key: &str) -> Result<Writer<'_>> {
        self.inner.write(key)
    }
}

/// A borrowed side-car artifact writer produced by [`Context::write`].
///
/// Implements [`std::io::Write`] and [`std::io::Seek`]. Writes are buffered; calling
/// [`Writer::finish`] flushes the buffer, closes the file, and returns a [`Handle`].
#[derive(Debug)]
pub struct Writer<'a> {
    io: BufWriter<File>,
    name: String,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl Writer<'_> {
    /// Flush and close the writer, returning a [`Handle`] for the artifact.
    ///
    /// Insert the returned handle into a [`Record`](super::Record) (typically via
    /// [`Record::insert`](super::Record::insert)) so that load-side code can locate the
    /// artifact through the manifest.
    pub fn finish(self) -> Result<Handle> {
        // NOTE: self.io.into_inner() will flush the buffer and close the file.
        self.io
            .into_inner()
            .map_err(|err| Error::new(err.into_error()))?;
        Ok(Handle::new(self.name))
    }
}

impl std::io::Write for Writer<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.io.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.io.flush()
    }
    fn write_vectored(&mut self, bufs: &[std::io::IoSlice<'_>]) -> std::io::Result<usize> {
        self.io.write_vectored(bufs)
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.io.write_all(buf)
    }
    fn write_fmt(&mut self, args: std::fmt::Arguments<'_>) -> std::io::Result<()> {
        self.io.write_fmt(args)
    }
}

impl std::io::Seek for Writer<'_> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.io.seek(pos)
    }

    fn rewind(&mut self) -> std::io::Result<()> {
        self.io.rewind()
    }
    fn stream_position(&mut self) -> std::io::Result<u64> {
        self.io.stream_position()
    }
    fn seek_relative(&mut self, offset: i64) -> std::io::Result<()> {
        self.io.seek_relative(offset)
    }
}
