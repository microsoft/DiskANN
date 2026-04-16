/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::HashSet,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use crate::{
    Number, Version,
    load::{Error, Loadable, Result, error},
    save,
};

#[derive(Debug, serde::Deserialize)]
pub(super) struct ContextInner {
    dir: PathBuf,
    files: HashSet<PathBuf>,
    value: save::Value<'static>,
}

#[derive(Debug, serde::Deserialize)]
struct FileRepr {
    files: HashSet<PathBuf>,
    value: save::Value<'static>,
}

impl ContextInner {
    pub(super) fn new(metadata: &Path, dir: &Path) -> Result<Self> {
        let file = std::fs::File::open(metadata).map_err(|e| {
            Error::new(e).context(format!("while trying to open {}", metadata.display()))
        })?;

        let reader = std::io::BufReader::new(file);
        let repr: FileRepr = serde_json::from_reader(reader)
            .map_err(|e| Error::new(e).context("could not deserialize manifest"))?;

        let this = Self {
            dir: dir.into(),
            files: repr.files,
            value: repr.value,
        };
        Ok(this)
    }

    pub(super) fn read(&self, key: &str) -> Result<Reader<'_>> {
        let key_as_path: &Path = key.as_ref();
        if !self.files.contains(key_as_path) {
            panic!("this should return an error instead");
        }

        let full = self.dir.join(key);
        let file = std::fs::File::open(full).unwrap();
        let reader = Reader {
            io: BufReader::new(file),
            _lifetime: std::marker::PhantomData,
        };

        Ok(reader)
    }

    pub(super) fn context(&self) -> Context<'_> {
        Context::new(self, &self.value)
    }
}

pub struct Reader<'a> {
    io: BufReader<File>,
    _lifetime: std::marker::PhantomData<&'a ()>,
}

impl std::io::Read for Reader<'_> {
    // Required method
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.io.read(buf)
    }

    // Provided methods
    fn read_vectored(&mut self, bufs: &mut [std::io::IoSliceMut<'_>]) -> std::io::Result<usize> {
        self.io.read_vectored(bufs)
    }
    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> std::io::Result<usize> {
        self.io.read_to_end(buf)
    }
    fn read_to_string(&mut self, buf: &mut String) -> std::io::Result<usize> {
        self.io.read_to_string(buf)
    }
    fn read_exact(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        self.io.read_exact(buf)
    }
}

impl std::io::Seek for Reader<'_> {
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

///////////////////////
// User facing types //
///////////////////////

#[derive(Debug, Clone)]
pub struct Context<'a> {
    inner: &'a ContextInner,
    value: &'a save::Value<'a>,
}

impl<'a> Context<'a> {
    fn new(inner: &'a ContextInner, value: &'a save::Value<'a>) -> Self {
        Self { inner, value }
    }

    fn context(&self) -> &'a ContextInner {
        self.inner
    }

    pub fn load<T>(&self) -> Result<T>
    where
        T: Loadable<'a>,
    {
        T::load(self.clone())
    }

    pub fn as_object(&self) -> Option<Object<'a>> {
        match self.value {
            save::Value::Object(versioned) => {
                let object = Object {
                    inner: self.inner,
                    record: versioned.record(),
                    version: versioned.version(),
                };
                Some(object)
            }
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match self.value {
            save::Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<Array<'a>> {
        match self.value {
            save::Value::Array(array) => Some(Array::new(self.context(), array)),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<Number> {
        match self.value {
            save::Value::Number(number) => Some(*number),
            _ => None,
        }
    }

    pub(crate) fn as_handle(&self) -> Option<&save::Handle> {
        match self.value {
            save::Value::Handle(handle) => Some(handle),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Object<'a> {
    inner: &'a ContextInner,
    record: &'a save::Record<'a>,
    version: Version,
}

impl<'a> Object<'a> {
    pub fn version(&self) -> Version {
        self.version
    }

    pub fn field<T>(&self, key: &str) -> Result<T>
    where
        T: Loadable<'a>,
    {
        match self.record.get(key) {
            Some(value) => T::load(Context::new(self.context(), value)),
            None => Err((error::Kind::MissingField).into()),
        }
    }

    pub fn read(&self, handle: &save::Handle) -> Result<Reader<'_>> {
        self.inner.read(handle.as_str())
    }

    fn context(&self) -> &'a ContextInner {
        self.inner
    }
}

#[derive(Debug)]
pub struct Array<'a> {
    inner: &'a ContextInner,
    array: &'a [save::Value<'a>],
}

impl<'a> Array<'a> {
    fn new(inner: &'a ContextInner, array: &'a [save::Value<'a>]) -> Self {
        Self { inner, array }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> Iter<'a> {
        Iter::new(self.context(), self.array.iter())
    }

    fn context(&self) -> &'a ContextInner {
        self.inner
    }
}

pub struct Iter<'a> {
    inner: &'a ContextInner,
    iter: std::slice::Iter<'a, save::Value<'a>>,
}

impl<'a> Iter<'a> {
    fn new(inner: &'a ContextInner, iter: std::slice::Iter<'a, save::Value<'a>>) -> Self {
        Self { inner, iter }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Context<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .map(|value| Context::new(self.inner, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl ExactSizeIterator for Iter<'_> {}
