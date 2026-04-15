/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{Version, save};

#[derive(Debug)]
struct ContextInner;

#[derive(Debug)]
pub struct Object<'a> {
    inner: &'a ContextInner,
    record: save::Record<'a>,
    version: Version,
}

impl<'a> Object<'a> {
    pub fn version(&self) -> Version {
        self.version
    }
}

#[derive(Debug)]
pub struct Context<'a> {
    inner: &'a ContextInner,
    value: save::Value<'a>,
}

impl<'a> Context<'a> {
    fn as_object(self) -> Result<Object<'a>, Self> {
        match self.value {
            save::Value::Object(versioned) => {
                let (record, version) = versioned.components();
                let object = Object {
                    inner: self.inner,
                    record,
                    version,
                };
                Ok(object)
            }
            _ => Err(self),
        }
    }
}

pub trait Load: Sized {
    const VERSION: Version;
    fn load(object: Object<'_>) -> Self;
    fn load_legacy(object: Object<'_>) -> Self;
}

pub trait Loadable: Sized {
    fn load(context: Context<'_>) -> Self;
}

impl<T> Loadable for T
where
    T: Load,
{
    fn load(context: Context<'_>) -> Self {
        let object = context.as_object().unwrap();
        let version = object.version();
        if version == T::VERSION {
            T::load(object)
        } else {
            T::load_legacy(object)
        }
    }
}

impl<T> Loadable for Vec<T>
where
    T: Loadable,
{
    fn load(context: Context<'_>) -> Self {
        if let save::Value::Array(array) = context.value {
            array
                .into_iter()
                .map(|value| {
                    let context = Context {
                        inner: context.inner,
                        value,
                    };
                    T::load(context)
                })
                .collect()
        } else {
            panic!("nope!");
        }
    }
}
