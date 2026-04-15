/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::Cow, collections::HashMap};

use serde::{Serialize, Serializer, ser::SerializeStruct};

use crate::{Number, Version};

#[derive(Debug)]
pub enum Value<'a> {
    Bool(bool),
    Number(Number),
    String(Cow<'a, str>),
    Bytes(Cow<'a, [u8]>),
    Array(Vec<Value<'a>>),
    Object(Versioned<'a>),
    Handle(Handle),
}

impl Serialize for Value<'_> {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bool(b) => ser.serialize_bool(*b),
            Self::Number(n) => n.serialize(ser),
            Self::String(s) => ser.serialize_str(s),
            Self::Bytes(b) => ser.serialize_bytes(b),
            Self::Array(a) => a.serialize(ser),
            Self::Object(v) => v.serialize(ser),
            Self::Handle(h) => h.serialize(ser),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Record<'a> {
    record: HashMap<Cow<'a, str>, Value<'a>>,
}

impl<'a> Record<'a> {
    pub fn from_iter<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = (Cow<'a, str>, Value<'a>)>,
    {
        Self {
            record: itr.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.record.contains_key(key)
    }

    pub fn get(&self, key: &str) -> Option<&Value<'a>> {
        self.record.get(key)
    }

    pub fn insert<K, V>(&mut self, key: K, value: V) -> Option<Value<'a>>
    where
        K: Into<Cow<'a, str>>,
        V: Into<Value<'a>>,
    {
        let key = key.into();
        if crate::is_reserved(&key) {
            panic!("key is reserved - need better error handling");
        }

        self.record.insert(key, value.into())
    }
}

#[derive(Debug, Serialize)]
pub struct Versioned<'a> {
    #[serde(flatten)]
    record: Record<'a>,
    #[serde(rename = "$version")]
    version: Version,
}

impl<'a> Versioned<'a> {
    pub(crate) fn new(record: Record<'a>, version: Version) -> Self {
        Self { record, version }
    }

    pub(crate) fn components(self) -> (Record<'a>, Version) {
        (self.record, self.version)
    }
}

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

impl Serialize for Handle {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut handle = ser.serialize_struct("Handle", 1)?;
        handle.serialize_field("$handle", &self.0)?;
        handle.end()
    }
}
