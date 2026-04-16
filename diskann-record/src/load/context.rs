/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{
    Number, Version,
    load::{Loadable, Result, error},
    save,
};

#[derive(Debug)]
struct ContextInner;

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
