/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use crate::dispatcher::{DispatchRule, FailureScore, MatchScore};

/// An refinement of [`std::any::Any`] with an associated name (tag) and serialization.
///
/// This type represents deserialized inputs returned from [`crate::Input::try_deserialize`]
/// and is passed to beckend benchmarks for matching and execution.
#[derive(Debug)]
pub struct Any {
    any: Box<dyn SerializableAny>,
    tag: &'static str,
}

/// The score given unsuccessful downcasts in [`Any::try_match`].
pub const MATCH_FAIL: FailureScore = FailureScore(10_000);

impl Any {
    /// Construct a new [`Any`] around `any` and associate it with the name `tag`.
    ///
    /// The tag is included as merely a debugging and readability aid and usually should
    /// belong to a [`crate::Input::tag`] that generated `any`.
    pub fn new<T>(any: T, tag: &'static str) -> Self
    where
        T: serde::Serialize + std::fmt::Debug + 'static,
    {
        Self {
            any: Box::new(any),
            tag,
        }
    }

    /// A lower level API for constructing an [`Any`] that decouples the serialized
    /// representation from the inmemory representation.
    ///
    /// When serialized, the **exact** representation of `repr` will be used.
    ///
    /// This is useful in some contexts where as part of input resolution, a fully resolved
    /// input struct contains elements that are not serializable.
    ///
    /// Like [`Any::new`], the tag is included for debugging and readability.
    pub fn raw<T>(any: T, repr: serde_json::Value, tag: &'static str) -> Self
    where
        T: std::fmt::Debug + 'static,
    {
        Self {
            any: Box::new(Raw::new(any, repr)),
            tag,
        }
    }

    /// Return the benchmark tag associated with this benchmarks.
    pub fn tag(&self) -> &'static str {
        self.tag
    }

    /// Return the Rust [`std::any::TypeId`] for the contained object.
    pub fn type_id(&self) -> std::any::TypeId {
        self.any.as_any().type_id()
    }

    /// Return `true` if the runtime value is `T`. Otherwise, return false.
    ///
    /// ```rust
    /// use diskann_benchmark_runner::any::Any;
    ///
    /// let value = Any::new(42usize, "usize");
    /// assert!(value.is::<usize>());
    /// assert!(!value.is::<u32>());
    /// ```
    #[must_use = "this function has no side effects"]
    pub fn is<T>(&self) -> bool
    where
        T: std::any::Any,
    {
        self.any.as_any().is::<T>()
    }

    /// Return a reference to the contained object if it's runtime type is `T`.
    ///
    /// Otherwise return `None`.
    ///
    /// ```rust
    /// use diskann_benchmark_runner::any::Any;
    ///
    /// let value = Any::new(42usize, "usize");
    /// assert_eq!(*value.downcast_ref::<usize>().unwrap(), 42);
    /// assert!(value.downcast_ref::<u32>().is_none());
    /// ```
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: std::any::Any,
    {
        self.any.as_any().downcast_ref::<T>()
    }

    /// Attempt to downcast self to `T` and if succssful, try matching `&T` with `U` using
    /// [`crate::dispatcher::DispatchRule`].
    ///
    /// Otherwise, return `Err(diskann_benchmark_runner::any::MATCH_FAIL)`.
    ///
    /// ```rust
    /// use diskann_benchmark_runner::{
    ///     any::Any,
    ///     dispatcher::{self, MatchScore, FailureScore},
    ///     utils::datatype::{self, DataType, Type},
    /// };
    ///
    /// let value = Any::new(DataType::Float32, "datatype");
    ///
    /// // A successful down cast and successful match.
    /// assert_eq!(
    ///     value.try_match::<DataType, Type<f32>>().unwrap(),
    ///     MatchScore(0),
    /// );
    ///
    /// // A successful down cast but unsuccessful match.
    /// assert_eq!(
    ///     value.try_match::<DataType, Type<f64>>().unwrap_err(),
    ///     datatype::MATCH_FAIL,
    /// );
    ///
    /// // An unsuccessful down cast.
    /// let value = Any::new(0usize, "usize");
    /// assert_eq!(
    ///     value.try_match::<DataType, Type<f32>>().unwrap_err(),
    ///     diskann_benchmark_runner::any::MATCH_FAIL,
    /// );
    /// ```
    pub fn try_match<'a, T, U>(&'a self) -> Result<MatchScore, FailureScore>
    where
        U: DispatchRule<&'a T>,
        T: 'static,
    {
        if let Some(cast) = self.downcast_ref::<T>() {
            U::try_match(&cast)
        } else {
            Err(MATCH_FAIL)
        }
    }

    /// Attempt to downcast self to `T` and if succssful, try converting `&T` with `U` using
    /// [`crate::dispatcher::DispatchRule`].
    ///
    /// If unsuccessful, returns an error.
    ///
    /// ```rust
    /// use diskann_benchmark_runner::{
    ///     any::Any,
    ///     dispatcher::{self, MatchScore, FailureScore},
    ///     utils::datatype::{self, DataType, Type},
    /// };
    ///
    /// let value = Any::new(DataType::Float32, "datatype");
    ///
    /// // A successful down cast and successful conversion.
    /// let _: Type<f32> = value.convert::<DataType, _>().unwrap();
    /// ```
    pub fn convert<'a, T, U>(&'a self) -> anyhow::Result<U>
    where
        U: DispatchRule<&'a T>,
        anyhow::Error: From<U::Error>,
        T: 'static,
    {
        if let Some(cast) = self.downcast_ref::<T>() {
            Ok(U::convert(cast)?)
        } else {
            Err(anyhow::Error::msg("invalid dispatch"))
        }
    }

    /// A wrapper for [`DispatchRule::description`].
    ///
    /// If `from` is `None` - document the expected tag for the input and return
    /// `<U as DispatchRule<&T>>::description(f, None)`.
    ///
    /// If `from` is `Some` - attempt to downcast to `T`. If successful, return the dispatch
    /// rule description for `U` on the doncast reference. Otherwise, return the expected tag.
    ///
    /// ```rust
    /// use diskann_benchmark_runner::{
    ///     any::Any,
    ///     utils::datatype::{self, DataType, Type},
    /// };
    ///
    /// use std::io::Write;
    ///
    /// struct Display(Option<Any>);
    ///
    /// impl std::fmt::Display for Display {
    ///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    ///         match &self.0 {
    ///             Some(v) => Any::description::<DataType, Type<f32>>(f, Some(&&v), "my-tag"),
    ///             None => Any::description::<DataType, Type<f32>>(f, None, "my-tag"),
    ///         }
    ///     }
    /// }
    ///
    /// // No contained value - document the expected conversion.
    /// assert_eq!(
    ///     Display(None).to_string(),
    ///     "tag \"my-tag\"\nfloat32",
    /// );
    ///
    /// // Matching contained value.
    /// assert_eq!(
    ///     Display(Some(Any::new(DataType::Float32, "datatype"))).to_string(),
    ///     "successful match",
    /// );
    ///
    /// // Successful down cast - unsuccessful match.
    /// assert_eq!(
    ///     Display(Some(Any::new(DataType::UInt64, "datatype"))).to_string(),
    ///     "expected \"float32\" but found \"uint64\"",
    /// );
    ///
    /// // Unsuccessful down cast.
    /// assert_eq!(
    ///     Display(Some(Any::new(0usize, "another-tag"))).to_string(),
    ///     "expected tag \"my-tag\" - instead got \"another-tag\"",
    /// );
    /// ```
    pub fn description<'a, T, U>(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a Self>,
        tag: impl std::fmt::Display,
    ) -> std::fmt::Result
    where
        U: DispatchRule<&'a T>,
        T: 'static,
    {
        match from {
            Some(this) => match this.downcast_ref::<T>() {
                Some(a) => U::description(f, Some(&a)),
                None => write!(
                    f,
                    "expected tag \"{}\" - instead got \"{}\"",
                    tag,
                    this.tag(),
                ),
            },
            None => {
                writeln!(f, "tag \"{}\"", tag)?;
                U::description(f, None::<&&T>)
            }
        }
    }

    /// Serialize the contained object to a [`serde_json::Value`].
    pub fn serialize(&self) -> Result<serde_json::Value, serde_json::Error> {
        self.any.dump()
    }
}

/// Used in `DispatchRule::description(f, _)` to ensure that additional description
/// lines are properly aligned.
#[macro_export]
macro_rules! describeln {
    ($writer:ident, $fmt:literal) => {
        writeln!($writer, concat!("        ", $fmt))
    };
    ($writer:ident, $fmt:literal, $($args:expr),* $(,)?) => {
        writeln!($writer, concat!("        ", $fmt), $($args,)*)
    };
}

trait SerializableAny: std::fmt::Debug {
    fn as_any(&self) -> &dyn std::any::Any;
    fn dump(&self) -> Result<serde_json::Value, serde_json::Error>;
}

impl<T> SerializableAny for T
where
    T: std::any::Any + serde::Serialize + std::fmt::Debug,
{
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dump(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::to_value(self)
    }
}

// A backend type that allows users to decouple the serialized representation from the
// actual type.
#[derive(Debug)]
struct Raw<T> {
    value: T,
    repr: serde_json::Value,
}

impl<T> Raw<T> {
    fn new(value: T, repr: serde_json::Value) -> Self {
        Self { value, repr }
    }
}

impl<T> SerializableAny for Raw<T>
where
    T: std::any::Any + std::fmt::Debug,
{
    fn as_any(&self) -> &dyn std::any::Any {
        &self.value
    }

    fn dump(&self) -> Result<serde_json::Value, serde_json::Error> {
        Ok(self.repr.clone())
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::utils::datatype::{self, DataType, Type};

    #[test]
    fn test_new() {
        let x = Any::new(42usize, "my-tag");
        assert_eq!(x.tag(), "my-tag");
        assert_eq!(x.type_id(), std::any::TypeId::of::<usize>());
        assert!(x.is::<usize>());
        assert!(!x.is::<u32>());
        assert_eq!(*x.downcast_ref::<usize>().unwrap(), 42);
        assert!(x.downcast_ref::<u32>().is_none());

        assert!(!x.is::<Raw<usize>>());
        assert!(!x.is::<Raw<u32>>());
        assert!(x.downcast_ref::<Raw<usize>>().is_none());
        assert!(x.downcast_ref::<Raw<u32>>().is_none());

        assert_eq!(
            x.serialize().unwrap(),
            serde_json::Value::Number(serde_json::value::Number::from(42usize))
        );
    }

    #[test]
    fn test_raw() {
        let repr = serde_json::json!(1.5);
        let x = Any::raw(42usize, repr, "my-tag");
        assert_eq!(x.tag(), "my-tag");
        assert_eq!(x.type_id(), std::any::TypeId::of::<usize>());
        assert!(x.is::<usize>());
        assert!(!x.is::<u32>());
        assert_eq!(*x.downcast_ref::<usize>().unwrap(), 42);
        assert!(x.downcast_ref::<u32>().is_none());

        assert!(!x.is::<Raw<usize>>());
        assert!(!x.is::<Raw<u32>>());
        assert!(x.downcast_ref::<Raw<usize>>().is_none());
        assert!(x.downcast_ref::<Raw<u32>>().is_none());

        assert_eq!(
            x.serialize().unwrap(),
            serde_json::Value::Number(serde_json::value::Number::from_f64(1.5).unwrap())
        );
    }

    #[test]
    fn test_try_match() {
        let value = Any::new(DataType::Float32, "random-tag");

        // A successful down cast and successful match.
        assert_eq!(
            value.try_match::<DataType, Type<f32>>().unwrap(),
            MatchScore(0),
        );

        // A successful down cast but unsuccessful match.
        assert_eq!(
            value.try_match::<DataType, Type<f64>>().unwrap_err(),
            datatype::MATCH_FAIL,
        );

        // An unsuccessful down cast.
        let value = Any::new(0usize, "");
        assert_eq!(
            value.try_match::<DataType, Type<f32>>().unwrap_err(),
            MATCH_FAIL,
        );
    }

    #[test]
    fn test_convert() {
        let value = Any::new(DataType::Float32, "random-tag");

        // A successful down cast and successful conversion.
        let _: Type<f32> = value.convert::<DataType, _>().unwrap();

        // An invalid match should return an error.
        let value = Any::new(0usize, "random-rag");
        let err = value.convert::<DataType, Type<f32>>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid dispatch"), "{}", msg);
    }

    #[test]
    #[should_panic(expected = "invalid dispatch")]
    fn test_convert_inner_error() {
        let value = Any::new(DataType::Float32, "random-tag");
        let _ = value.convert::<DataType, Type<u64>>();
    }

    #[test]
    fn test_description() {
        struct Display(Option<Any>);

        impl std::fmt::Display for Display {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match &self.0 {
                    Some(v) => Any::description::<DataType, Type<f32>>(f, Some(&v), "my-tag"),
                    None => Any::description::<DataType, Type<f32>>(f, None, "my-tag"),
                }
            }
        }

        // No contained value - document the expected conversion.
        assert_eq!(Display(None).to_string(), "tag \"my-tag\"\nfloat32",);

        // Matching contained value.
        assert_eq!(
            Display(Some(Any::new(DataType::Float32, ""))).to_string(),
            "successful match",
        );

        // Successful down cast - unsuccessful match.
        assert_eq!(
            Display(Some(Any::new(DataType::UInt64, ""))).to_string(),
            "expected \"float32\" but found \"uint64\"",
        );

        // Unsuccessful down cast.
        assert_eq!(
            Display(Some(Any::new(0usize, ""))).to_string(),
            "expected tag \"my-tag\" - instead got \"\"",
        );
    }
}
