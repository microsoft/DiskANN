/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Loading Records from Disk
//!
//! This module mirrors the [`super::save`] side. User types implement [`Load`] (or, for
//! primitive-like leaves, [`Loadable`]) and obtain an [`Object`] / [`Context`] from which
//! they extract individual fields and side-car artifacts.
//!
//! Implement [`Loader`] to supply runtime settings or resources without changing the
//! saved format. 
//! NOTE:: Do not also implement [`Load`], which already provides a blanket 
//! [`Loadable`] implementation.
//!
//! The generic entry point is `load`; `load_from_disk` (available under the `disk`
//! feature) is the disk-backed convenience wrapper that reads a manifest and dispatches
//! into the user type's [`Load`] impl.
//!
//! # Reading Records
//!
//! The [`load_fields!`](crate::load_fields) macro is the idiomatic way to extract a fixed
//! set of named fields from an [`Object`] into local bindings. It mirrors the structure
//! of [`save_fields!`](crate::save_fields).
//!
//! # Version Dispatch
//!
//! Each [`Load`] impl declares a [`VERSION`](Load::VERSION). If the version stored in the
//! manifest matches, [`Load::load`] is called. Otherwise [`Load::load_legacy`] is invoked
//! so the impl can perform a custom upgrade; returning an
//! [`error::Kind::UnknownVersion`] from `load_legacy` indicates the loader has no upgrade
//! path for that schema.
//!
//! # Recoverable vs. Critical Errors
//!
//! Load errors are tagged as recoverable or critical. Probing call sites that try
//! multiple loaders should only retry when [`Error::is_recoverable`] returns `true`. See
//! [`error::Kind::is_recoverable`] for the classification.

pub mod error;
pub use error::{Error, Result};

mod context;
pub(crate) use context::LoadContext;
pub use context::{Context, Object, Reader};

use std::marker::PhantomData;

#[cfg(feature = "disk")]
use std::path::Path;

use crate::{Version, save};

/// Deserialize a `T` from `context`.
///
/// This is the generic entry point: it asks `context` for its root value and threads a
/// [`Context`] borrowing `context` through `T`'s [`Loadable::load`] impl.
/// `load_from_disk` is the disk-backed convenience wrapper available under the `disk`
/// feature.
///
/// # Errors
///
/// Returns [`Error`] if the context cannot produce a root value or if `T`'s loader fails.
pub(crate) fn load<'a, T>(context: &'a dyn LoadContext) -> Result<T>
where
    T: Loadable<'a>,
{
    load_with(context, PhantomData::<T>)
}

/// Load the root value using caller-supplied settings or resources.
pub(crate) fn load_with<'a, L>(context: &'a dyn LoadContext, loader: L) -> Result<L::Value>
where
    L: Loader<'a>,
{
    let value = context.value()?;
    loader.load(Context::new(context, value))
}

/// Reload a value previously written by [`save::save_to_disk`].
///
/// `metadata` is the manifest JSON path produced by the saver, and `dir` is the
/// directory holding any side-car artifacts.
///
/// # Errors
///
/// Returns [`Error`] if the manifest is missing or malformed, if a referenced artifact is
/// missing, or if a user [`Load`] impl fails (e.g. due to a version mismatch with no
/// upgrade path).
#[cfg(feature = "disk")]
pub fn load_from_disk<T>(metadata: &Path, dir: &Path) -> Result<T>
where
    T: for<'a> Loadable<'a>,
{
    let context = crate::backend::disk::DiskLoadContext::new(metadata, dir)?;
    load(&context)
}

/// Reload a saved value using caller-supplied settings or resources.
///
/// The loader must work for any metadata lifetime, so the result cannot borrow from
/// the disk context, which is dropped before returning. It may borrow resources
/// supplied by the caller; those resources need not be `'static`.
///
/// A reference into the saved metadata cannot escape:
///
/// ```compile_fail
/// use std::{marker::PhantomData, path::Path};
/// use diskann_record::load::load_from_disk_with;
///
/// let name: &str = load_from_disk_with(
///     Path::new("metadata.json"),
///     Path::new("artifacts"),
///     PhantomData::<&str>,
/// ).unwrap();
/// ```
///
/// # Errors
///
/// Returns the same manifest and artifact errors as [`load_from_disk`], or any error
/// returned by the loader.
#[cfg(feature = "disk")]
pub fn load_from_disk_with<L, T>(metadata: &Path, dir: &Path, loader: L) -> Result<T>
where
    L: for<'a> Loader<'a, Value = T>,
{
    let context = crate::backend::disk::DiskLoadContext::new(metadata, dir)?;
    load_with(&context, loader)
}

/// Reconstruct a value using caller-supplied settings or resources.
pub trait Loader<'a>: Sized {
    /// The value produced by this loader.
    type Value;

    /// Load from the supplied context, consuming this loader.
    fn load(self, context: Context<'a>) -> Result<Self::Value>;
}

/// Trick to load a [`Loadable`] type through [`Loader`].
impl<'a, T> Loader<'a> for PhantomData<T>
where
    T: Loadable<'a>,
{
    type Value = T;

    fn load(self, context: Context<'a>) -> Result<Self::Value> {
        T::load(context)
    }
}

/// Implemented by user types that can be reloaded from a versioned [`Object`].
///
/// This is the symmetric counterpart to [`super::save::Save`]. Implementations describe
/// how to reconstruct `Self` from the manifest representation, and how to upgrade
/// records written by older schemas via [`Self::load_legacy`].
///
/// # Enums
///
/// Enum types dispatch on the single non-reserved key of the object (see
/// [`Object::single_key`]) and recurse via [`Object::child`] into the payload.
pub trait Load<'a>: Sized {
    /// The schema version this impl was written against.
    ///
    /// Compared with the manifest's version to choose between [`Self::load`] and
    /// [`Self::load_legacy`].
    const VERSION: Version;

    /// Reconstruct `Self` from an object whose `$version` matches [`Self::VERSION`].
    fn load(object: Object<'a>) -> Result<Self>;

    /// Reconstruct `Self` from an object whose `$version` does *not* match
    /// [`Self::VERSION`].
    ///
    /// Implementations may either upgrade the older record or refuse with
    /// [`error::Kind::UnknownVersion`] when no upgrade is possible.
    fn load_legacy(object: Object<'a>) -> Result<Self>;
}

/// Implemented by any value that can be deserialized from a [`Context`].
///
/// This is the bottom of the trait hierarchy and is implemented for the same set of
/// primitive-like types as [`super::save::Saveable`]. Most user types should implement
/// [`Load`] (which gets a [`Loadable`] impl for free via the blanket below) rather than
/// [`Loadable`] directly. A type whose shared load path lives in a [`Loader`] can
/// instead implement this trait directly by delegating to that loader with defaults.
pub trait Loadable<'a>: Sized {
    /// Deserialize `Self` from a [`Context`].
    fn load(context: Context<'a>) -> Result<Self>;
}

impl<'a, T> Loadable<'a> for T
where
    T: Load<'a>,
{
    fn load(context: Context<'a>) -> Result<Self> {
        let object = context.as_object().ok_or(error::Kind::TypeMismatch)?;
        let version = object.version();
        if version == T::VERSION {
            T::load(object)
        } else {
            T::load_legacy(object)
        }
    }
}

////////////
// Macros //
////////////

/// Extract a fixed set of named fields from an [`Object`] into local bindings.
///
/// Each name in the list becomes a `let` binding of the same name. An optional `: T`
/// suffix selects the [`Loadable`] target type; without it, type inference picks the
/// type from the surrounding context. Errors from individual fields are propagated with
/// `?`.
///
/// ```ignore
/// load_fields!(object, [
///     dim: usize,
///     label,                       // type inferred
///     vectors: save::Handle,
/// ]);
/// ```
#[macro_export]
macro_rules! load_fields {
    (@field $object:ident, $field:ident: $T:ty) => {
        let $field: $T = $object.field(stringify!($field))?;
    };
    (@field $object:ident, $field:ident) => {
        let $field = $object.field(stringify!($field))?;
    };
    ($object:ident, [$($field:ident $(: $ty:ty)?),+ $(,)?]) => {
        $(
            $crate::load_fields!(@field $object, $field $(: $ty)?);
        )+
    };
}

///////////////
// Bootstrap //
///////////////

impl<'a> Loadable<'a> for &'a str {
    fn load(context: Context<'a>) -> Result<Self> {
        context
            .as_str()
            .ok_or_else(|| error::Kind::TypeMismatch.into())
    }
}

impl Loadable<'_> for String {
    fn load(context: Context<'_>) -> Result<Self> {
        context.load::<&str>().map(|s| s.into())
    }
}

impl Loadable<'_> for save::Handle {
    fn load(context: Context<'_>) -> Result<Self> {
        context
            .as_handle()
            .cloned()
            .ok_or_else(|| error::Kind::TypeMismatch.into())
    }
}

impl Loadable<'_> for bool {
    fn load(context: Context<'_>) -> Result<Self> {
        context
            .as_bool()
            .ok_or_else(|| error::Kind::TypeMismatch.into())
    }
}

impl<'a, T> Loadable<'a> for Option<T>
where
    T: Loadable<'a>,
{
    fn load(context: Context<'a>) -> Result<Self> {
        if context.is_null() {
            Ok(None)
        } else {
            T::load(context).map(Some)
        }
    }
}

impl<'a, T> Loadable<'a> for Vec<T>
where
    T: Loadable<'a>,
{
    fn load(context: Context<'a>) -> Result<Self> {
        match context.as_array() {
            Some(array) => array.iter().map(T::load).collect(),
            None => Err((error::Kind::TypeMismatch).into()),
        }
    }
}

macro_rules! load_number {
    ($T:ty) => {
        impl Loadable<'_> for $T {
            fn load(context: Context<'_>) -> Result<Self> {
                match context.as_number() {
                    Some(n) => n.try_into().map_err(|_| error::Kind::NumberOutOfRange.into()),
                    None => Err((error::Kind::TypeMismatch).into()),
                }
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(load_number!($Ts);)+
    }
}

load_number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);

// NonZero* primitives are loaded by deserializing the inner numeric type and then
// validating it is non-zero. A zero value produces a `NumberOutOfRange` light error.
macro_rules! load_nonzero {
    ($T:ty, $Inner:ty) => {
        impl Loadable<'_> for $T {
            fn load(context: Context<'_>) -> Result<Self> {
                let inner: $Inner = context.load()?;
                <$T>::new(inner).ok_or_else(|| error::Kind::NumberOutOfRange.into())
            }
        }
    };
}

load_nonzero!(std::num::NonZeroU32, u32);
load_nonzero!(std::num::NonZeroU64, u64);
load_nonzero!(std::num::NonZeroUsize, usize);
