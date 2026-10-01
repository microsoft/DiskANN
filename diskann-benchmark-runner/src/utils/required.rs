/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

use crate::{reflect::tree, Reflect};

/// Like `Option<T>`, but requires the containing field to be present in the input JSON.
///
/// To represent [`None`], the field must be explicitly set to `null`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequiredOption<T>(Option<T>);

impl<T> RequiredOption<T> {
    pub fn new(opt: Option<T>) -> Self {
        Self(opt)
    }

    pub fn some(value: T) -> Self {
        Self::new(Some(value))
    }

    pub fn none() -> Self {
        Self::new(None)
    }

    pub fn unwrap_or(self, default: T) -> T {
        self.0.unwrap_or(default)
    }

    pub fn get_or_insert(&mut self, v: T) -> &mut T {
        self.0.get_or_insert(v)
    }

    pub fn into_inner(self) -> Option<T> {
        self.0
    }

    pub fn as_ref(&self) -> Option<&T> {
        self.0.as_ref()
    }

    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.0.as_mut()
    }

    pub fn as_deref(&self) -> Option<&T::Target>
    where
        T: std::ops::Deref,
    {
        self.0.as_deref()
    }

    pub fn as_deref_mut(&mut self) -> Option<&mut T::Target>
    where
        T: std::ops::DerefMut,
    {
        self.0.as_deref_mut()
    }
}

impl<T> Reflect for RequiredOption<T>
where
    T: Reflect,
{
    fn ty() -> tree::Type {
        let doc = "A required optional type.\n\n\
                   Unlike `Option`, where an omitted field implies `None`, the field must be \
                   present and explicitly set to `null` to represent `None`.";

        // We're special - we get to instantiate the `Optional` type.
        tree::Type::optional::<T>(Some(doc.into()))
    }

    fn format_type_name(f: &mut dyn std::fmt::Write) -> std::fmt::Result {
        f.write_str("RequiredOption<")?;
        T::format_type_name(f)?;
        f.write_str(">")
    }
}
