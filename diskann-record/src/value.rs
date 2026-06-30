/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Wire-level value types used in the on-disk manifest.
//!
//! These types are the shared currency of both halves of the framework:
//! user value -> [`save`](crate::save) -> [`Value`] in the save path, and the
//! [`Value`] -> [`load`](crate::load) -> user value in the load path.
//!
//! Every field stored in a manifest is one of:
//!
//! * [`Value::Null`] / [`Value::Bool`] / [`Value::Number`] / [`Value::String`] —
//!   primitive scalars.
//! * [`Value::Array`] — a homogeneous sequence (used by `Vec<T>` and `&[T]`).
//! * [`Value::Object`] — a [`Versioned`] [`Record`] (the canonical encoding for a
//!   `T: crate::save::Save`).
//! * [`Value::Handle`] — a reference to a side-car artifact (produced by
//!   [`crate::save::Context::write`] + [`crate::save::Writer::finish`]).
//!
//! Most user code never touches these enums directly. On the save side,
//! [`crate::save::Saveable`] impls turn Rust values into [`Value`]s and the
//! [`save_fields!`](crate::save_fields) macro assembles the surrounding [`Record`]; on
//! the load side, the [`crate::load`] accessors walk the same [`Value`] tree back into
//! Rust values.

use std::{borrow::Cow, collections::HashMap};

#[cfg(feature = "serde")]
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
};

use crate::{Number, Version, save::Error};

/// The wire-level union of every saveable kind.
///
/// See the module-level docs for an overview of when each variant is produced. The
/// borrowing parameter `'a` lets [`Value::String`] and nested
/// records reuse memory owned by the caller without copying.
#[derive(Debug)]
pub enum Value<'a> {
    Null,
    Bool(bool),
    Number(Number),
    String(Cow<'a, str>),
    Array(Vec<Value<'a>>),
    Object(Versioned<'a>),
    Handle(Handle),
}

#[cfg(feature = "serde")]
impl Serialize for Value<'_> {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => ser.serialize_none(),
            Self::Bool(b) => ser.serialize_bool(*b),
            Self::Number(n) => n.serialize(ser),
            Self::String(s) => ser.serialize_str(s),
            Self::Array(a) => a.serialize(ser),
            Self::Object(v) => v.serialize(ser),
            Self::Handle(h) => h.serialize(ser),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Value<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Inner;

        impl<'de> Visitor<'de> for Inner {
            type Value = Value<'static>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a valid Value")
            }

            fn visit_unit<E: de::Error>(self) -> Result<Value<'static>, E> {
                Ok(Value::Null)
            }

            fn visit_none<E: de::Error>(self) -> Result<Value<'static>, E> {
                Ok(Value::Null)
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Value<'static>, D::Error>
            where
                D: Deserializer<'de>,
            {
                Value::deserialize(deserializer)
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value<'static>, E> {
                Ok(Value::Bool(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value<'static>, E> {
                Ok(Value::Number(Number::U64(v)))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value<'static>, E> {
                Ok(Value::Number(Number::I64(v)))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value<'static>, E> {
                Ok(Value::Number(Number::F64(v)))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Value<'static>, E> {
                if let Some(n) = Number::from_sentinel(v) {
                    return Ok(Value::Number(n));
                }
                Ok(Value::String(Cow::Owned(v.to_owned())))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Value<'static>, E> {
                if let Some(n) = Number::from_sentinel(&v) {
                    return Ok(Value::Number(n));
                }
                Ok(Value::String(Cow::Owned(v)))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Value<'static>, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(v) = seq.next_element()? {
                    values.push(v);
                }
                Ok(Value::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Value<'static>, A::Error>
            where
                A: MapAccess<'de>,
            {
                // TODO: Handle invariants that only one of our reserved words are present.
                let mut version: Option<Version> = None;
                let mut handle_name: Option<String> = None;
                let mut fields: HashMap<Cow<'static, str>, Value<'static>> = HashMap::new();

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "$version" => {
                            version = Some(map.next_value()?);
                        }
                        "$handle" => {
                            handle_name = Some(map.next_value()?);
                        }
                        _ => {
                            let value = map.next_value()?;
                            fields.insert(Cow::Owned(key), value);
                        }
                    }
                }

                if let Some(name) = handle_name {
                    if version.is_some() || !fields.is_empty() {
                        return Err(de::Error::custom(
                            "handle object must contain only a \"$handle\" field",
                        ));
                    }
                    return Ok(Value::Handle(Handle(name)));
                }

                if let Some(version) = version {
                    let record = Record { record: fields };
                    return Ok(Value::Object(Versioned { record, version }));
                }

                Err(de::Error::custom(
                    "map must contain either \"$version\" or \"$handle\"",
                ))
            }
        }

        deserializer.deserialize_any(Inner)
    }
}

impl From<Handle> for Value<'_> {
    fn from(handle: Handle) -> Self {
        Self::Handle(handle)
    }
}

impl Value<'_> {
    /// Convert this value into a fully owned [`Value<'static>`], deep-copying any borrowed
    /// string or byte data.
    ///
    /// This is the allocation-based equivalent of round-tripping through the wire format:
    /// it severs every borrow from the originating data so the result can be stored
    /// independently of its source (for example inside an in-memory
    /// [`crate::MemoryContext`]).
    pub fn into_owned(self) -> Value<'static> {
        match self {
            Self::Null => Value::Null,
            Self::Bool(b) => Value::Bool(b),
            Self::Number(n) => Value::Number(n),
            Self::String(s) => Value::String(Cow::Owned(s.into_owned())),
            Self::Array(values) => {
                Value::Array(values.into_iter().map(Value::into_owned).collect())
            }
            Self::Object(versioned) => Value::Object(versioned.into_owned()),
            Self::Handle(handle) => Value::Handle(handle),
        }
    }
}

/// A map of named [`Value`]s.
///
/// `Record` is the body of an object in the manifest. On the save side each call to
/// [`crate::save::Save::save`] returns one, and [`Record::into_value`] wraps it as a
/// [`Versioned`] [`Value::Object`] ready for insertion into another record; on the load
/// side the same record is read back through [`crate::load::Object`]. Keys beginning
/// with `$` are reserved for framework metadata (see [`crate::is_reserved`]).
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Record<'a> {
    record: HashMap<Cow<'a, str>, Value<'a>>,
}

impl<'a> Record<'a> {
    /// Construct an empty record. Useful for unit enum variants.
    pub fn empty() -> Self {
        Self {
            record: HashMap::new(),
        }
    }

    /// Returns `true` if a value is registered under `key`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.record.contains_key(key)
    }

    /// Look up the [`Value`] registered under `key`, if any.
    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.record.get(key)
    }

    /// Number of (user) keys in this record. Reserved keys (`$version`, `$handle`)
    /// are tracked elsewhere and never appear here.
    pub fn len(&self) -> usize {
        self.record.len()
    }

    /// Returns `true` if this record has no user keys.
    pub fn is_empty(&self) -> bool {
        self.record.is_empty()
    }

    /// Iterate over the user keys in this record. Order is unspecified.
    pub fn keys(&self) -> Keys<'_, 'a> {
        Keys {
            inner: self.record.keys(),
        }
    }

    /// Insert `value` under `key`.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if `key` begins with `$`, which is reserved for the
    /// save/load framework (see [`crate::is_reserved`]).
    pub fn insert<K, V>(&mut self, key: K, value: V) -> crate::save::Result<Option<Value<'a>>>
    where
        K: Into<Cow<'a, str>>,
        V: Into<Value<'a>>,
    {
        let key = key.into();
        if crate::is_reserved(&key) {
            return Err(Error::message(format!(
                "record key {:?} is reserved (keys starting with `$` are reserved for the \
                 save/load framework)",
                key,
            )));
        }

        Ok(self.record.insert(key, value.into()))
    }

    /// Wrap this record as a versioned [`Value`] ready for insertion into another
    /// record. Use this from enum [`Save`](crate::save::Save) impls to attach the
    /// outer type's version to an inline variant payload.
    pub fn into_value(self, version: Version) -> Value<'a> {
        Value::Object(Versioned::new(self, version))
    }

    /// Convert this record into a fully owned [`Record<'static>`], deep-copying borrowed
    /// keys and values. See [`Value::into_owned`].
    pub fn into_owned(self) -> Record<'static> {
        Record {
            record: self
                .record
                .into_iter()
                .map(|(key, value)| (Cow::Owned(key.into_owned()), value.into_owned()))
                .collect(),
        }
    }
}

/// Iterator over the keys of a [`Record`].
pub struct Keys<'r, 'a> {
    inner: std::collections::hash_map::Keys<'r, Cow<'a, str>, Value<'a>>,
}

impl<'r, 'a> Iterator for Keys<'r, 'a> {
    type Item = &'r str;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|k| k.as_ref())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for Keys<'_, '_> {}

impl<'a> FromIterator<(Cow<'a, str>, Value<'a>)> for Record<'a> {
    fn from_iter<I: IntoIterator<Item = (Cow<'a, str>, Value<'a>)>>(itr: I) -> Self {
        Self {
            record: itr.into_iter().collect(),
        }
    }
}

/// A [`Record`] paired with the schema [`Version`] used to produce it.
///
/// Serialized as a normal object plus a `$version` field on the wire. Constructed by
/// [`Record::into_value`].
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Versioned<'a> {
    #[cfg_attr(feature = "serde", serde(flatten))]
    record: Record<'a>,
    #[cfg_attr(feature = "serde", serde(rename = "$version"))]
    version: Version,
}

impl<'a> Versioned<'a> {
    pub(crate) fn new(record: Record<'a>, version: Version) -> Self {
        Self { record, version }
    }

    pub(crate) fn version(&self) -> Version {
        self.version
    }

    pub(crate) fn record(&self) -> &Record<'a> {
        &self.record
    }

    pub(crate) fn into_owned(self) -> Versioned<'static> {
        Versioned {
            record: self.record.into_owned(),
            version: self.version,
        }
    }
}

/// A reference to a side-car artifact in the manifest directory.
///
/// Produced by [`Writer::finish`](crate::save::Writer::finish) after a side-car write completes and
/// inserted into a [`Record`] like any other value. Serializes as `{"$handle": "<name>"}`
/// on the wire; the load side rehydrates it through
/// [`crate::load::Object::read`].
#[derive(Debug, Clone)]
pub struct Handle(String);

impl Handle {
    pub(crate) fn new(string: String) -> Self {
        Self(string)
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(feature = "serde")]
impl Serialize for Handle {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut handle = ser.serialize_struct("Handle", 1)?;
        handle.serialize_field("$handle", &self.0)?;
        handle.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Handle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            #[serde(rename = "$handle")]
            handle: String,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(Handle(helper.handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_rejects_reserved_key() {
        let mut record = Record::empty();
        record
            .insert("$version", Value::Null)
            .expect_err("reserved key must be rejected");
        record
            .insert("ok", Value::Bool(true))
            .expect("normal key must be accepted");
        assert!(record.contains_key("ok"));
        assert_eq!(record.len(), 1);
    }

    #[cfg(feature = "disk")]
    #[test]
    fn deserialize_rejects_handle_with_extra_fields() {
        let json = r#"{ "$handle": "a.bin", "$version": "0.0" }"#;
        serde_json::from_str::<Value<'static>>(json)
            .expect_err("handle object with extra fields must be rejected");
    }

    #[cfg(feature = "disk")]
    #[test]
    fn deserialize_rejects_object_without_version_or_handle() {
        let json = r#"{ "field": 1 }"#;
        serde_json::from_str::<Value<'static>>(json)
            .expect_err("object without $version or $handle must be rejected");
    }
}
