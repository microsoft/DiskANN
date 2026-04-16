/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{borrow::Cow, collections::HashMap};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, SeqAccess, Visitor},
    ser::SerializeStruct,
};

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

impl<'de, 'a> Deserialize<'de> for Value<'a> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct Inner<'a>(std::marker::PhantomData<&'a ()>);

        impl<'de, 'a> Visitor<'de> for Inner<'a> {
            type Value = Value<'a>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a valid Value")
            }

            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Value<'a>, E> {
                Ok(Value::Bool(v))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Value<'a>, E> {
                Ok(Value::Number(Number::U64(v)))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Value<'a>, E> {
                Ok(Value::Number(Number::I64(v)))
            }

            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value<'a>, E> {
                Ok(Value::Number(Number::F64(v)))
            }

            fn visit_borrowed_str<E: de::Error>(self, v: &'de str) -> Result<Value<'a>, E> {
                Ok(Value::String(Cow::Owned(v.to_owned())))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Value<'a>, E> {
                Ok(Value::String(Cow::Owned(v.to_owned())))
            }

            fn visit_string<E: de::Error>(self, v: String) -> Result<Value<'a>, E> {
                Ok(Value::String(Cow::Owned(v)))
            }

            fn visit_borrowed_bytes<E: de::Error>(self, v: &'de [u8]) -> Result<Value<'a>, E> {
                Ok(Value::Bytes(Cow::Owned(v.to_owned())))
            }

            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<Value<'a>, E> {
                Ok(Value::Bytes(Cow::Owned(v.to_owned())))
            }

            fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<Value<'a>, E> {
                Ok(Value::Bytes(Cow::Owned(v)))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Value<'a>, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(v) = seq.next_element()? {
                    values.push(v);
                }
                Ok(Value::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Value<'a>, A::Error>
            where
                A: MapAccess<'de>,
            {
                // TODO: Handle invaiants that only one of our reserved words are present.
                let mut version: Option<Version> = None;
                let mut handle_name: Option<String> = None;
                let mut fields: HashMap<Cow<'a, str>, Value<'a>> = HashMap::new();

                while let Some(key) = map.next_key::<Cow<'a, str>>()? {
                    match key.as_ref() {
                        "$version" => {
                            version = Some(map.next_value()?);
                        }
                        "$handle" => {
                            handle_name = Some(map.next_value()?);
                        }
                        _ => {
                            let value = map.next_value()?;
                            fields.insert(key, value);
                        }
                    }
                }

                if let Some(name) = handle_name {
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

        deserializer.deserialize_any(Inner(std::marker::PhantomData))
    }
}

impl From<Handle> for Value<'_> {
    fn from(handle: Handle) -> Self {
        Self::Handle(handle)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
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

#[derive(Debug, Serialize, Deserialize)]
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

    pub(crate) fn version(&self) -> Version {
        self.version
    }

    pub(crate) fn record(&self) -> &Record<'a> {
        &self.record
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
