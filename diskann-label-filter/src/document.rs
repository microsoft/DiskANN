/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::Attribute;

/// Simple container class that clients can use to
/// supply diskann with a vector and its attributes
pub struct Document<'a, V: ?Sized> {
    vector: &'a V,
    attributes: &'a [Attribute],
}

impl<'a, V: ?Sized> Document<'a, V> {
    pub fn new(vector: &'a V, attributes: &'a [Attribute]) -> Self {
        Self { vector, attributes }
    }

    pub(crate) fn vector(&self) -> &'a V {
        self.vector
    }

    pub(crate) fn attributes(&self) -> &'a [Attribute] {
        self.attributes
    }
}
