/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! # Saving Records to Disk
//!
//! This module provides the writer-side of the framework. User types implement [`Save`]
//! (or, for primitive-like leaves, [`Saveable`]) and obtain a [`Context`] from which they
//! request side-car artifact writers and assemble a [`Record`] of named fields.
//!
//! The top-level entry point is [`save_to_disk`], which serializes a value into a
//! caller-chosen directory plus a manifest path.
//!
//! # Building Records
//!
//! The [`save_fields!`](crate::save_fields) macro is the idiomatic way to build a record
//! from a struct or destructured enum variant. It handles per-field error context and
//! invokes [`Saveable::save`] on each value.
//!
//! # Side-Car Artifacts
//!
//! Binary blobs (e.g. vector buffers) are written to side-car files via
//! [`Context::write`], which returns a [`Writer`]. The handle returned by
//! [`Writer::finish`](context::Writer::finish) can be embedded into the record as a
//! [`Handle`]; it serializes as a `$handle` reference and is rehydrated on the load side.
pub use crate::value::{Handle, Keys, Record, Value, Versioned};

mod context;
pub use context::{Context, Writer};

mod error;
pub use error::{Error, Result};

use crate::Version;

/// Serialize `x` to disk.
///
/// The manifest (a JSON document) is written atomically to `metadata`; any side-car
/// artifacts the type's [`Save::save`] impl creates via [`Context::write`] are written
/// into `dir`.
///
/// # Errors
///
/// Returns [`Error`] if the directory cannot be written to, if the manifest cannot be
/// serialized, or if a user impl returns an error.
pub fn save_to_disk<T>(
    x: &T,
    dir: impl AsRef<std::path::Path>,
    metadata: impl AsRef<std::path::Path>,
) -> Result<()>
where
    T: Saveable,
{
    let inner = context::ContextInner::new(dir.as_ref().into(), metadata.as_ref().into());
    let value = x.save(inner.context())?;
    inner.finish(value)
}

/// Implemented by user types that map to a versioned [`Record`].
///
/// This is the primary trait for structured user types. A [`Save`] impl describes the
/// versioned schema of `Self`: its associated [`VERSION`](Self::VERSION) is attached to
/// the [`Record`] produced by [`Self::save`](Self::save).
///
/// # Enums
///
/// Enum types are encoded by returning a [`Record`] with a single user key whose name
/// is the variant tag and whose value is the variant's payload (frequently
/// [`Value::Null`] for unit variants). See the crate-level docs for a worked example.
pub trait Save {
    /// The schema version attached to records produced by this impl.
    ///
    /// Loaders compare this against the version stored in the manifest to decide
    /// between [`Load::load`](crate::load::Load::load) and
    /// [`Load::load_legacy`](crate::load::Load::load_legacy).
    const VERSION: Version;

    /// Serialize `self` into a [`Record`].
    ///
    /// Use the supplied [`Context`] to request side-car artifact writers. Use the
    /// [`save_fields!`](crate::save_fields) macro to populate the record.
    fn save(&self, context: Context<'_>) -> Result<Record<'_>>;
}

/// Implemented by any value that can be written into a [`Value`].
///
/// This is the bottom of the trait hierarchy and is implemented for:
///
/// * Primitive numeric types (signed, unsigned, floats, `NonZero*`).
/// * [`bool`], [`str`], [`String`], and [`Handle`].
/// * [`Option<T>`] (serializes `None` as [`Value::Null`]).
/// * `&[T]` and [`Vec<T>`] (serialize as [`Value::Array`]).
/// * Any `T: Save` (wraps the produced record in [`Value::Object`] with the type's
///   [`Save::VERSION`]).
///
/// Most user types should implement [`Save`] (which gets a [`Saveable`] impl for free
/// via the blanket below) rather than [`Saveable`] directly.
pub trait Saveable {
    /// Serialize `self` into a [`Value`].
    fn save(&self, context: Context<'_>) -> Result<Value<'_>>;
}

impl<T> Saveable for T
where
    T: Save,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        let record = self.save(context)?;
        Ok(record.into_value(T::VERSION))
    }
}

//////////////////
// Random Stuff //
//////////////////

impl<T> Saveable for [T]
where
    T: Saveable,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        let values: Result<Vec<_>> = self.iter().map(|t| t.save(context.clone())).collect();
        values.map(Value::Array)
    }
}

impl<T> Saveable for Vec<T>
where
    T: Saveable,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        self.as_slice().save(context)
    }
}

impl Saveable for str {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::String(self.into()))
    }
}

impl Saveable for String {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::String(self.as_str().into()))
    }
}

impl Saveable for Handle {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::Handle(self.clone()))
    }
}

impl Saveable for bool {
    fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
        Ok(Value::Bool(*self))
    }
}

impl<T> Saveable for Option<T>
where
    T: Saveable,
{
    fn save(&self, context: Context<'_>) -> Result<Value<'_>> {
        match self {
            None => Ok(Value::Null),
            Some(t) => t.save(context),
        }
    }
}

macro_rules! save_number {
    ($T:ty) => {
        impl Saveable for $T {
            fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
                Ok(Value::Number((*self).into()))
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(save_number!($Ts);)+
    }
}

save_number!(usize, u64, u32, u16, u8, i64, i32, i16, i8, f32, f64);

// NonZero* primitives serialize as their inner numeric type. Loaders reject zero.
macro_rules! save_nonzero {
    ($T:ty) => {
        impl Saveable for $T {
            fn save(&self, _: Context<'_>) -> Result<Value<'_>> {
                Ok(Value::Number(self.get().into()))
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(save_nonzero!($Ts);)+
    }
}

save_nonzero!(
    std::num::NonZeroU32,
    std::num::NonZeroU64,
    std::num::NonZeroUsize
);

#[derive(Debug, Clone, Copy)]
#[doc(hidden)]
pub struct Serializing(pub &'static str);

impl std::fmt::Display for Serializing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "while serializing field \"{}\"", self.0)
    }
}

/// Build a [`Record`] from a list of fields.
///
/// Two forms are supported:
///
/// * `save_fields!(self, context, [a, b, c])` reads each field as `self.a`,
///   `self.b`, etc. Use this from `Save::save` for plain structs.
/// * `save_fields!(context, [a, b, c])` reads each field from a local binding of
///   the same name. Use this inside enum match arms where the variant's payload
///   has already been destructured into local bindings. Those bindings are
///   assumed to be references (which is automatic when matching against `&self`);
///   for an owned local, take a reference explicitly first.
#[macro_export]
macro_rules! save_fields {
    ($me:ident, $context:ident, [$($field:ident),+ $(,)?]) => {{
        $crate::save::Record::from_iter(
            [
                $(
                    (
                        ::std::borrow::Cow::Borrowed(stringify!($field)),
                        <_ as $crate::save::Saveable>::save(
                            &$me.$field,
                            $context.clone()
                        ).map_err(|err| {
                            err.context($crate::save::Serializing(stringify!($field)))
                        })?
                    ),
                )+
            ]
        )
    }};
    ($context:ident, [$($field:ident),+ $(,)?]) => {{
        $crate::save::Record::from_iter(
            [
                $(
                    (
                        ::std::borrow::Cow::Borrowed(stringify!($field)),
                        <_ as $crate::save::Saveable>::save(
                            $field,
                            $context.clone()
                        ).map_err(|err| {
                            err.context($crate::save::Serializing(stringify!($field)))
                        })?
                    ),
                )+
            ]
        )
    }};
}
