/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// An refinement of [`std::any::Any`] with an associated name (tag) and serialization.
///
/// This type represents deserialized inputs returned from [`crate::Input::try_deserialize`]
/// and is passed to beckend benchmarks for matching and execution.
#[derive(Debug)]
pub struct Any {
    any: Box<dyn SerializableAny>,
    tag: &'static str,
}

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

    /// Serialize the contained object to a [`serde_json::Value`].
    pub fn serialize(&self) -> Result<serde_json::Value, serde_json::Error> {
        self.any.dump()
    }
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
}
