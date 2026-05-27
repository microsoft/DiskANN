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
//! The top-level entry point is [`load_from_disk`], which reads a manifest and dispatches
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
pub use context::{Context, Object};

use std::path::Path;

use crate::{Version, save};

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
pub fn load_from_disk<T>(metadata: &Path, dir: &Path) -> Result<T>
where
    T: for<'a> Loadable<'a>,
{
    let inner = context::ContextInner::new(metadata, dir)?;
    inner.context().load()
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
/// [`Loadable`] directly.
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
