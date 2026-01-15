/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::Attribute;
use diskann_utils::reborrow::Reborrow;

///Simple container class that clients can use to
/// supply diskann with a vector and its attributes
pub struct Document<'a, V> {
    vector: &'a V,
    attributes: Vec<Attribute>,
}

impl<'a, V> Document<'a, V> {
    pub fn new(vector: &'a V, attributes: Vec<Attribute>) -> Self {
        Self { vector, attributes }
    }

    pub(crate) fn vector(&self) -> &'a V {
        self.vector
    }

    pub(crate) fn attributes(&self) -> &Vec<Attribute> {
        &self.attributes
    }
}

/// Container class that represents a (vector, attributes) pair
/// where the attributes are represented as u64s.
#[derive(Clone)]
pub struct EncodedDocument<V, ST> {
    vector: V,
    attributes: ST,
}

impl<V, ST> EncodedDocument<V, ST> {
    pub fn new(vector: V, attributes: ST) -> Self {
        Self { vector, attributes }
    }

    pub fn destructure(self) -> (V, ST) {
        (self.vector, self.attributes)
    }

    pub(crate) fn attributes(&self) -> &ST {
        &self.attributes
    }
}

/// See [`Accessor::Element`]
impl<'this, V, ST> Reborrow<'this> for EncodedDocument<V, ST>
where
    V: Reborrow<'this>,
{
    type Target = EncodedDocument<V::Target, &'this ST>;

    fn reborrow(&'this self) -> Self::Target {
        EncodedDocument::new(self.vector.reborrow(), &self.attributes)
    }
}
