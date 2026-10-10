/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use serde::{Deserialize, Serialize};

use crate::{Reflect, reflect::tree};

/// Like `Option<T>`, but requires the containing field to be present in the input JSON.
///
/// To represent [`None`], the field must be explicitly set to `null`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RequiredOption<T>(Option<T>);

impl<T> RequiredOption<T> {
    /// Wrap an optional value.
    pub fn new(opt: Option<T>) -> Self {
        Self(opt)
    }

    /// Wrap a present value.
    pub fn some(value: T) -> Self {
        Self::new(Some(value))
    }

    /// Construct an absent value.
    pub fn none() -> Self {
        Self::new(None)
    }

    /// See: [`Option::unwrap_or`].
    pub fn unwrap_or(self, default: T) -> T {
        self.0.unwrap_or(default)
    }

    /// See: [`Option::get_or_insert`].
    pub fn get_or_insert(&mut self, v: T) -> &mut T {
        self.0.get_or_insert(v)
    }

    /// Return the wrapped optional value.
    pub fn into_inner(self) -> Option<T> {
        self.0
    }

    /// See: [`Option::as_ref`].
    pub fn as_ref(&self) -> Option<&T> {
        self.0.as_ref()
    }

    /// See: [`Option::as_mut`].
    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.0.as_mut()
    }

    /// See: [`Option::as_deref`].
    pub fn as_deref(&self) -> Option<&T::Target>
    where
        T: std::ops::Deref,
    {
        self.0.as_deref()
    }

    /// See: [`Option::as_deref_mut`].
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Input {
        value: RequiredOption<u32>,
    }

    #[test]
    fn rejects_missing_field() {
        let error = serde_json::from_value::<Input>(json!({})).unwrap_err();

        assert!(error.to_string().contains("missing field `value`"));
    }

    #[test]
    fn accepts_null_field() {
        let input: Input = serde_json::from_value(json!({ "value": null })).unwrap();

        assert_eq!(input.value, RequiredOption::none());
    }

    #[test]
    fn accepts_present_value() {
        let input: Input = serde_json::from_value(json!({ "value": 42 })).unwrap();

        assert_eq!(input.value, RequiredOption::some(42));
    }

    #[test]
    fn serializes_transparently() {
        let none = Input {
            value: RequiredOption::none(),
        };
        let some = Input {
            value: RequiredOption::some(42),
        };

        assert_eq!(
            serde_json::to_value(none).unwrap(),
            json!({ "value": null })
        );
        assert_eq!(serde_json::to_value(some).unwrap(), json!({ "value": 42 }));
    }

    #[test]
    fn methods_smoke_test() {
        let value = RequiredOption::new(Some(1));
        assert_eq!(value.into_inner(), Some(1));
        assert_eq!(RequiredOption::some(2).unwrap_or(3), 2);

        let mut value = RequiredOption::none();
        assert_eq!(value.as_ref(), None);
        assert_eq!(value.get_or_insert(String::from("hello")), "hello");
        assert_eq!(value.as_deref(), Some("hello"));
        value.as_mut().unwrap().push('!');
        value.as_deref_mut().unwrap().make_ascii_uppercase();
        assert_eq!(value.into_inner(), Some(String::from("HELLO!")));
    }
}
