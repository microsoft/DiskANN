/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::Checker;

/// Inputs to [`Benchmarks`](crate::Benchmark).
///
/// These begin as [`raw`](Self::Raw) data transfer objects before final construction via
/// [`from_raw`](Self::from_raw).
pub trait Input: Sized + std::fmt::Debug + 'static {
    /// The raw form of this input that is deserialized from input files and serialized as
    /// [`examples`](Self::example). The raw nature of this type reflects that no input
    /// validation has been performed beyond the checks performed by its
    /// [`Deserialize`](serde::Deserialize) implementation.
    ///
    /// Final object validation is performed via [`from_raw`](Self::from_raw).
    type Raw: serde::de::DeserializeOwned + serde::Serialize;

    /// Return the discriminant associated with this type.
    ///
    /// This is used to map inputs types to their respective parsers.
    ///
    /// Well formed implementations should always return the same result.
    fn tag() -> &'static str;

    /// Construct `Self` from the raw deserialized representation, performing any necessary
    /// validation checks (e.g., resolving file paths via the [`Checker`]).
    fn from_raw(raw: Self::Raw, checker: &mut Checker) -> anyhow::Result<Self>;

    /// Serialize `self` to a [`serde_json::Value`].
    fn serialize(&self) -> anyhow::Result<serde_json::Value>;

    /// Return an example of a raw input for this [`Input`].
    ///
    /// This is used to supply sample JSON layouts in the benchmark CLI.
    fn example() -> Self::Raw;
}

/// A registered input. See [`crate::Registry::input`].
#[derive(Clone, Copy)]
pub struct Registered<'a>(pub(crate) &'a dyn internal::DynInput);

impl Registered<'_> {
    /// Return the input tag of the registered input.
    ///
    /// See: [`Input::tag`].
    pub fn tag(&self) -> &'static str {
        self.0.tag()
    }

    /// Try to deserialize raw JSON into the dynamic type of the input.
    ///
    /// See: [`Input::from_raw`].
    pub(crate) fn try_deserialize(
        &self,
        serialized: &serde_json::Value,
        checker: &mut Checker,
    ) -> anyhow::Result<internal::Any> {
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

pub(crate) mod internal {
    use super::*;

    /// Runtime representation of a deserialized [`Input`].
    #[derive(Debug)]
    pub(crate) struct Any {
        any: Box<dyn RuntimeAny>,
    }

    impl Any {
        pub(crate) fn new<T>(input: T) -> Self
        where
            T: Input,
        {
            Self {
                any: Box::new(input),
            }
        }

        #[must_use = "this function has no side effects"]
        pub(crate) fn tag(&self) -> &'static str {
            self.any.tag()
        }

        #[must_use = "this function has no side effects"]
        pub(crate) fn downcast_ref<T>(&self) -> Option<&T>
        where
            T: std::any::Any,
        {
            self.any.as_any().downcast_ref::<T>()
        }

        #[must_use = "this function has no side effects"]
        pub(crate) fn serialize(&self) -> anyhow::Result<serde_json::Value> {
            self.any.serialize()
        }
    }

    trait RuntimeAny: std::fmt::Debug {
        fn tag(&self) -> &'static str;
        fn as_any(&self) -> &dyn std::any::Any;
        fn serialize(&self) -> anyhow::Result<serde_json::Value>;
    }

    impl<T> RuntimeAny for T
    where
        T: Input,
    {
        fn tag(&self) -> &'static str {
            <Self as Input>::tag()
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn serialize(&self) -> anyhow::Result<serde_json::Value> {
            <Self as Input>::serialize(self)
        }
    }

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

        // reflection
        fn as_any(&self) -> &dyn std::any::Any;
        fn type_name(&self) -> &'static str;
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
            let raw = <T::Raw as serde::Deserialize<'_>>::deserialize(serialized)?;
            Ok(Any::new(T::from_raw(raw, checker)?))
        }
        fn example(&self) -> anyhow::Result<serde_json::Value> {
            Ok(serde_json::to_value(T::example())?)
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn type_name(&self) -> &'static str {
            std::any::type_name::<T>()
        }
    }
}
