/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::attribute::Attribute;

/// Simple container class that clients can use to
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
