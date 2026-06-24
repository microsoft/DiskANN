/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Load-side context, object, array, and side-car reader.
//!
//! [`Context`] is the cheap, clonable handle threaded through every
//! [`super::Load::load`] / [`super::Loadable::load`] impl. From a [`Context`], loaders
//! ask:
//!
//! * [`Context::as_object`] / [`Object::field`] for nested records.
//! * [`Context::as_array`] / [`Array::iter`] for sequences.
//! * [`Context::as_str`] / [`Context::as_number`] / [`Context::as_bool`] / [`Context::is_null`] for scalars.
//! * [`Object::read`] for side-car artifacts referenced by a
//!   [`save::Handle`](super::save::Handle).
//!
//! [`Reader`] implements [`std::io::Read`] and [`std::io::Seek`] over a side-car
//! artifact, regardless of the provider's backing store.

use std::io::BufReader;

use crate::{
    Number, Version,
    load::{Loadable, Result, error},
    save,
};

/// The backing store for a load operation.
///
/// A `LoadContext` supplies the root manifest [`save::Value`] ([`LoadContext::value`])
/// and resolves side-car artifacts referenced by handles ([`LoadContext::read`]). The
/// concrete implementations live under `crate::backend`: a disk-backed context (under
/// the `disk` feature) and an in-memory context. Alternative implementations (e.g. a
/// virtual filesystem) can be supplied for testing.
///
/// The generic [`load`](super::load) entry point is parameterized over this trait, and
/// [`Context`] / [`Object`] / `Array` borrow it as an object-safe `&dyn LoadContext`
/// so the load tree is agnostic to the concrete context type.
pub trait LoadContext {
    /// The root value of the manifest.
    ///
    /// # Errors
    ///
    /// Returns [`Error`](crate::load::Error) if the root value cannot be produced.
    fn value(&self) -> Result<&save::Value<'_>>;

    /// Open the side-car artifact identified by `key` for reading.
    ///
    /// # Errors
    ///
    /// Returns [`error::Kind::MissingFile`] if the file is not registered with this
    /// context or if `key` escapes the manifest directory.
    fn read(&self, key: &str) -> Result<Reader<'_>>;
}

/// The backend-specific half of a [`Reader`].
///
/// Each [`LoadContext`] implementation supplies its own `ReaderInner` (e.g. an in-memory
/// cursor or an on-disk file). The blanket impl covers any type that is both
/// [`std::io::Read`] and [`std::io::Seek`].
pub(crate) trait ReaderInner: std::io::Read + std::io::Seek {}

impl<T> ReaderInner for T where T: std::io::Read + std::io::Seek {}

/// A borrowed reader over a side-car artifact.
///
/// Produced by [`Object::read`]. Implements [`std::io::Read`] and [`std::io::Seek`] over
/// whatever backing store the [`LoadContext`] provides, so non-file-backed providers
/// (like an in-memory byte buffer) can supply an arbitrary seekable reader.
pub struct Reader<'a> {
    io: BufReader<Box<dyn ReaderInner + 'a>>,
}

impl<'a> Reader<'a> {
    /// Build a reader over an arbitrary borrowed [`ReaderInner`] source.
    ///
    /// Used by [`LoadContext`] implementations to expose a side-car artifact backed by a
    /// file or an in-memory [`std::io::Cursor`].
    pub(crate) fn new<T>(io: T) -> Self
    where
        T: ReaderInner + 'a,
    {
        Self {
            io: BufReader::new(Box::new(io)),
        }
    }
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

/// A cheap, clonable handle threaded through every load impl.
///
/// Loaders use the `as_*` accessors to peek at the underlying [`save::Value`] kind
/// (e.g. [`Context::as_object`], [`Context::as_array`], [`Context::as_str`]) and
/// [`Context::load`] to recursively deserialize a nested value into a concrete type.
#[derive(Clone)]
pub struct Context<'a> {
    inner: &'a dyn LoadContext,
    value: &'a save::Value<'a>,
}

impl<'a> Context<'a> {
    pub(super) fn new(inner: &'a dyn LoadContext, value: &'a save::Value<'a>) -> Self {
        Self { inner, value }
    }

    fn context(&self) -> &'a dyn LoadContext {
        self.inner
    }

    /// Recursively deserialize the underlying value into a `T`.
    ///
    /// Equivalent to calling `T::load(self.clone())`. Use this from inner loaders that
    /// want to delegate to another [`Loadable`].
    pub fn load<T>(&self) -> Result<T>
    where
        T: Loadable<'a>,
    {
        T::load(self.clone())
    }

    /// Returns `Some(Object)` if the value is a versioned object, else `None`.
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

    /// Returns `Some(s)` if the value is a string, else `None`.
    pub fn as_str(&self) -> Option<&'a str> {
        match self.value {
            save::Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns `Some(Array)` if the value is an array, else `None`.
    pub fn as_array(&self) -> Option<Array<'a>> {
        match self.value {
            save::Value::Array(array) => Some(Array::new(self.context(), array)),
            _ => None,
        }
    }

    /// Returns `Some(Number)` if the value is numeric, else `None`.
    ///
    /// Use the conversion methods on [`Number`] (e.g. `as_u32`, `as_i64`) to narrow to
    /// the target Rust type; out-of-range conversions return `None` and should be
    /// surfaced as [`error::Kind::NumberOutOfRange`].
    pub fn as_number(&self) -> Option<Number> {
        match self.value {
            save::Value::Number(number) => Some(*number),
            _ => None,
        }
    }

    /// Returns `Some(b)` if the value is a boolean, else `None`.
    pub fn as_bool(&self) -> Option<bool> {
        match self.value {
            save::Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns `true` if the value is null.
    ///
    /// Used by [`Loadable`] impls for [`Option<T>`] to detect the absent variant.
    pub fn is_null(&self) -> bool {
        matches!(self.value, save::Value::Null)
    }

    pub(crate) fn as_handle(&self) -> Option<&save::Handle> {
        match self.value {
            save::Value::Handle(handle) => Some(handle),
            _ => None,
        }
    }
}

/// A versioned record reached through [`Context::as_object`].
///
/// `Object` is the entry point for record-based deserialization: it exposes the schema
/// version via [`Object::version`], the user keys via [`Object::keys`], typed field
/// extraction via [`Object::field`] (and the [`load_fields!`](crate::load_fields)
/// macro), and side-car artifact access via [`Object::read`].
pub struct Object<'a> {
    inner: &'a dyn LoadContext,
    record: &'a save::Record<'a>,
    version: Version,
}

impl<'a> Object<'a> {
    /// The schema [`Version`] recorded in the manifest for this object.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Iterate over the user keys of this record. Reserved keys (`$version`,
    /// `$handle`) are tracked separately and never appear here.
    pub fn keys(&self) -> save::Keys<'_, 'a> {
        self.record.keys()
    }

    /// Number of user keys in this record.
    pub fn len(&self) -> usize {
        self.record.len()
    }

    /// Whether this record has no user keys.
    pub fn is_empty(&self) -> bool {
        self.record.is_empty()
    }

    /// Return the sole user key of this record, used by enum loaders to dispatch
    /// to a variant arm. Errors with a recoverable [`error::Kind::TypeMismatch`]
    /// if the record has zero or multiple user keys (i.e. the wire shape does
    /// not look like an enum).
    pub fn single_key(&self) -> Result<&str> {
        let mut keys = self.record.keys();
        let Some(first) = keys.next() else {
            return Err(error::Kind::TypeMismatch.into());
        };
        if keys.next().is_some() {
            return Err(error::Kind::TypeMismatch.into());
        }
        Ok(first)
    }

    /// Descend into the raw [`Context`] for `key`, without imposing a type.
    /// Useful for enum variants whose payload is itself an [`Object`], an array,
    /// or any other [`save::Value`]. Returns [`error::Kind::MissingField`] when
    /// the key is absent.
    pub fn child(&self, key: &str) -> Result<Context<'a>> {
        match self.record.get(key) {
            Some(value) => Ok(Context::new(self.context(), value)),
            None => Err(error::Kind::MissingField.into()),
        }
    }

    /// Extract the value under `key` and deserialize it into a `T`.
    ///
    /// This is the typed counterpart to [`Object::child`] and the primitive used by the
    /// [`load_fields!`](crate::load_fields) macro.
    ///
    /// # Errors
    ///
    /// Returns [`error::Kind::MissingField`] if the key is absent. Errors raised by
    /// `T::load` (e.g. [`error::Kind::TypeMismatch`]) are propagated unchanged.
    pub fn field<T>(&self, key: &str) -> Result<T>
    where
        T: Loadable<'a>,
    {
        match self.record.get(key) {
            Some(value) => T::load(Context::new(self.context(), value)),
            None => Err((error::Kind::MissingField).into()),
        }
    }

    /// Open the side-car artifact identified by `handle` for reading.
    ///
    /// The handle must have been previously written through the matching
    /// [`save::Context::write`](super::save::Context::write) call and embedded in this
    /// record. Returns [`error::Kind::MissingFile`] if the file is not registered in the
    /// manifest or if the handle attempts to escape the manifest directory.
    pub fn read(&self, handle: &save::Handle) -> Result<Reader<'_>> {
        self.inner.read(handle.as_str())
    }

    fn context(&self) -> &'a dyn LoadContext {
        self.inner
    }
}

/// A homogeneous sequence of values reached through [`Context::as_array`].
///
/// Backed by a borrowed `&[Value]`. Use [`Array::iter`] to walk the elements; each
/// item is yielded as a [`Context`] that can be further deserialized via
/// [`Context::load`].
pub struct Array<'a> {
    inner: &'a dyn LoadContext,
    array: &'a [save::Value<'a>],
}

impl<'a> Array<'a> {
    fn new(inner: &'a dyn LoadContext, array: &'a [save::Value<'a>]) -> Self {
        Self { inner, array }
    }

    /// Number of elements in the array.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over the elements, each as a [`Context`] ready for further deserialization.
    pub fn iter(&self) -> Iter<'a> {
        Iter::new(self.context(), self.array.iter())
    }

    fn context(&self) -> &'a dyn LoadContext {
        self.inner
    }
}

/// Iterator returned by [`Array::iter`].
pub struct Iter<'a> {
    inner: &'a dyn LoadContext,
    iter: std::slice::Iter<'a, save::Value<'a>>,
}

impl<'a> Iter<'a> {
    fn new(inner: &'a dyn LoadContext, iter: std::slice::Iter<'a, save::Value<'a>>) -> Self {
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
