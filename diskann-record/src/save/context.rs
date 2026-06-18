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

use std::{collections::HashSet, fs::File, io::BufWriter, path::PathBuf, sync::Mutex};

use crate::save::{Error, Handle, Result, Value};

/// The owned context behind a [`Context`].
///
/// Holds the manifest directory, the manifest path, and the set of artifact file names
/// registered so far. Lookup and insertion go through a [`Mutex`] so that concurrent
/// [`Save`](super::Save) impls cannot accidentally hand out the same artifact name twice.
#[derive(Debug)]
pub(super) struct ContextInner {
    dir: PathBuf,
    metadata: PathBuf,
    files: Mutex<HashSet<String>>,
}

#[derive(serde::Serialize)]
struct Final<'a> {
    files: Vec<&'a str>,
    value: &'a Value<'a>,
}

impl ContextInner {
    // TODO: Error if the directory looks bad?
    pub(super) fn new(dir: PathBuf, metadata: PathBuf) -> Self {
        Self {
            dir,
            metadata,
            files: Mutex::new(HashSet::new()),
        }
    }

    pub(super) fn context(&self) -> Context<'_> {
        Context { inner: self }
    }

    pub(super) fn write(&self, key: &str) -> Result<Writer<'_>> {
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
    pub fn finish(self, value: Value<'_>) -> Result<()> {
        let mut temp = self.metadata.clone().into_os_string();
        temp.push(".temp");
        let temp = PathBuf::from(temp);
        if temp.exists() {
            return Err(Error::message(format!(
                "Temporary file {} already exists. Aborting!",
                temp.display()
            )));
        }
        let files = self
            .files
            .into_inner()
            .unwrap_or_else(|poison| poison.into_inner());
        let f = Final {
            files: files.iter().map(|k| &**k).collect(),
            value: &value,
        };

        let buffer = std::fs::File::create(&temp).map_err(|err| {
            Error::new(err).context(format!(
                "while creating temp manifest file {}",
                temp.display()
            ))
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
/// artifact-name bookkeeping.
#[derive(Debug, Clone)]
pub struct Context<'a> {
    inner: &'a ContextInner,
}

impl<'a> Context<'a> {
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
