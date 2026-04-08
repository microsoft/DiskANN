/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::{Any, Checker};

pub trait Input {
    /// Return the discriminant associated with this type.
    ///
    /// This is used to map inputs types to their respective parsers.
    ///
    /// Well formed implementations should always return the same result.
    fn tag() -> &'static str;

    /// Attempt to deserialize an opaque object from the raw `serialized` representation.
    ///
    /// Deserialized values can be constructed and returned via [`Checker::any`],
    /// [`Any::new`] or [`Any::raw`].
    ///
    /// If using the [`Any`] constructors directly, implementations should associate
    /// [`Self::tag`] with the returned `Any`. If [`Checker::any`] is used - this will
    /// happen automatically.
    ///
    /// Implementations are **strongly** encouraged to implement
    /// [`CheckDeserialization`](crate::CheckDeserialization) and use this API to ensure
    /// shared resources (like input files or output files) are correctly resolved and
    /// properly shared among all jobs in a benchmark run.
    fn try_deserialize(
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any>;

    /// Print an example JSON representation of objects this input is expected to parse.
    ///
    /// Well-formed implementations should ensure that passing the returned
    /// [`serde_json::Value`] back to [`Self::try_deserialize`] correctly deserializes,
    /// though it need not necessarily pass
    /// [`CheckDeserialization`](crate::CheckDeserialization).
    fn example() -> anyhow::Result<serde_json::Value>;
}

/// A registered input. See [`crate::registry::Inputs::get`].
#[derive(Clone, Copy)]
pub struct Registered<'a>(pub(crate) &'a dyn DynInput);

impl Registered<'_> {
    /// Return the input tag of the registered input.
    ///
    /// See: [`Input::tag`].
    pub fn tag(&self) -> &'static str {
        self.0.tag()
    }

    /// Try to deserialize raw JSON into the dynamic type of the input.
    ///
    /// See: [`Input::try_deserialize`].
    pub fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        self.0.try_deserialize(serialized, checker)
    }

    /// Return an example JSON for the dynamic type of the input.
    ///
    /// See: [`Input::example`].
    pub fn example(&self) -> anyhow::Result<serde_json::Value> {
        self.0.example()
    }
}

impl std::fmt::Debug for Registered<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("input::Registered")
            .field("tag", &self.tag())
            .finish()
    }
}

//////////////
// Internal //
//////////////

#[derive(Debug)]
pub(crate) struct Wrapper<T>(std::marker::PhantomData<T>);

impl<T> Wrapper<T> {
    pub(crate) const INSTANCE: Self = Self::new();

    pub(crate) const fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> Clone for Wrapper<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Wrapper<T> {}

pub(crate) trait DynInput {
    fn tag(&self) -> &'static str;
    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any>;
    fn example(&self) -> anyhow::Result<serde_json::Value>;
}

impl<T> DynInput for Wrapper<T>
where
    T: Input,
{
    fn tag(&self) -> &'static str {
        T::tag()
    }
    fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<Any> {
        T::try_deserialize(serialized, checker)
    }
    fn example(&self) -> anyhow::Result<serde_json::Value> {
        T::example()
    }
}
