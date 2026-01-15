/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{default::Default, marker::PhantomData};

/// A utility `'static` compatible trait adding a lifetime to a type.
pub trait WithLifetime: Send + Sync + 'static {
    /// The associated type with a lifetime.
    type Of<'a>: Send + Sync;
}

/// A [`WithLifetime`] annotator for `'static` types that discards the generic lifetime.
///
/// ```
/// use diskann_utils::{WithLifetime, lifetime::Static};
///
/// fn foo<T: WithLifetime>(_: T::Of<'_>) {}
///
/// let x = f32::default();
/// foo::<Static<f32>>(x);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Static<T> {
    _type: PhantomData<T>,
}

const _: () = assert!(std::mem::size_of::<Static<f32>>() == 0);

impl<T> Static<T> {
    /// Construct a new `Static` zero sized type.
    pub const fn new() -> Self {
        Self { _type: PhantomData }
    }
}

impl<T> Default for Static<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WithLifetime for Static<T>
where
    T: Send + Sync + 'static,
{
    type Of<'a> = T;
}

/// A [`WithLifetime`] annotator for slices.
///
/// ```
/// use diskann_utils::{WithLifetime, lifetime::Slice};
///
/// fn foo<T: WithLifetime>(x: T::Of<'_>) {}
///
/// let v = vec![1usize, 2, 3];
/// foo::<Slice<usize>>(&v);
/// assert_eq!(v.len(), 3);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Slice<T> {
    _type: PhantomData<T>,
}

impl<T> Slice<T> {
    pub const fn new() -> Self {
        Self { _type: PhantomData }
    }
}

impl<T> Default for Slice<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WithLifetime for Slice<T>
where
    T: Send + Sync + 'static,
{
    type Of<'a> = &'a [T];
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static() {
        fn foo<T>(x: T::Of<'_>) -> T::Of<'_>
        where
            T: WithLifetime,
            for<'a> T::Of<'a>: 'static,
        {
            x
        }

        let x = f32::default();
        assert_eq!(foo::<Static<f32>>(x), 0.0);

        const _: Static<f32> = Static::new();

        let _ = Static::<f32>::default();
    }

    #[test]
    fn test_slice() {
        fn foo<T>(x: T::Of<'_>) -> f32
        where
            T: for<'a> WithLifetime<Of<'a> = &'a [f32]>,
        {
            x.iter().sum()
        }

        let x = vec![1.0, 2.0, 3.0];
        assert_eq!(foo::<Slice<f32>>(&x), 6.0);

        const _: Slice<f32> = Slice::new();
        let _ = Slice::<f32>::default();
    }
}
