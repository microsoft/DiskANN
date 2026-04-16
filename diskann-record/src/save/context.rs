/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, fs::File, io::BufWriter, path::PathBuf, sync::Mutex};

use crate::save::{Error, Handle, Value};

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

    pub(super) fn write(&self, key: &str) -> Writer<'_> {
        // TODO: Proper disambiguation - making UUIDs etc.
        let mut files = self.files.lock().unwrap();
        if files.insert(key.into()) {
            let full = self.dir.join(key);
            let file = std::fs::File::create_new(full).unwrap();
            Writer {
                io: BufWriter::new(file),
                name: key.into(),
                _lifetime: std::marker::PhantomData,
            }
        } else {
            panic!("you done goofed!");
        }
    }

    pub fn finish(self, value: Value<'_>) -> Result<(), Error> {
        let temp = format!("{}.temp", self.metadata.display());
        if std::path::Path::new(&temp).exists() {
            return Err(Error::message(format!(
                "Temporary file {} already exists. Aborting!",
                temp
            )));
        }

        let files = self.files.into_inner().unwrap();
        let f = Final {
            files: files.iter().map(|k| &**k).collect(),
            value: &value,
        };

        let buffer = std::fs::File::create(&temp).unwrap();
        serde_json::to_writer_pretty(buffer, &f).unwrap();
        std::fs::rename(&temp, &self.metadata).unwrap();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Context<'a> {
    inner: &'a ContextInner,
}

impl<'a> Context<'a> {
    pub fn write(&self, key: &str) -> Writer<'_> {
        self.inner.write(key)
    }
}

#[derive(Debug)]
pub struct Writer<'a> {
    io: BufWriter<File>,
    name: String,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl Writer<'_> {
    pub fn finish(self) -> Handle {
        Handle::new(self.name)
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
