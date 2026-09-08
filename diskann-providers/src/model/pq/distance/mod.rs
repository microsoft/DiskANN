/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod common;
pub mod cosine;
pub mod dynamic;
pub mod innerproduct;
pub mod l2;

pub mod multi;

/// A light-weight [`std::borrow::Cow`] whose owned state is a [`std::sync::Arc`].
#[derive(Debug)]
pub enum Shared<'a, T> {
    Arc(std::sync::Arc<T>),
    Ref(&'a T),
}

impl<T> Clone for Shared<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Arc(arc) => Shared::Arc(arc.clone()),
            Self::Ref(r) => Shared::Ref(r),
        }
    }
}

impl<'a, T> From<&'a T> for Shared<'a, T> {
    fn from(r: &'a T) -> Self {
        Self::Ref(r)
    }
}

impl<'a, T> std::ops::Deref for Shared<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Arc(arc) => arc,
            Self::Ref(r) => r,
        }
    }
}

// Exports
pub use dynamic::{DistanceComputer, QueryComputer};

#[cfg(test)]
pub(crate) mod test_utils;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::{assert_matches, sync::Arc};

    fn assert_is_async_friendly<T>(_: &T)
    where
        T: Send + Sync + 'static,
    {
    }

    #[test]
    fn test_shared() {
        let s = Arc::new("a string");

        let mut shared = Shared::<'static, &'static str>::Arc(s.clone());
        assert_is_async_friendly(&shared);

        assert_eq!(shared.as_ptr(), s.as_ptr());
        assert_eq!(*s, "a string");

        let static_str = &"a static str";
        shared = static_str.into();

        assert_eq!(*shared, "a static str");
        assert_matches!(shared, Shared::Ref(_));
    }
}
