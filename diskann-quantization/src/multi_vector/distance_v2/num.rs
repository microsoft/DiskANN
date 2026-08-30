/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, num::NonZeroUsize};

/// A offset for pointers `*const T` in terms of numbers of `T`.
#[derive(Debug)]
pub(super) struct Elements<T> {
    elements: usize,
    _type: PhantomData<T>,
}

impl<T> Clone for Elements<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Elements<T> {}

impl<T> Elements<T> {
    pub(super) const fn new(elements: usize) -> Self {
        Self {
            elements,
            _type: PhantomData,
        }
    }

    pub(super) const fn value(self) -> usize {
        self.elements
    }
}

impl<T, U> PartialEq<Elements<U>> for Elements<T> {
    fn eq(&self, other: &Elements<U>) -> bool {
        self.value() == other.value()
    }
}

impl<T> std::ops::Mul<usize> for Elements<T> {
    type Output = Self;

    fn mul(self, by: usize) -> Self {
        Self::new(self.value() * by)
    }
}

impl<T> std::ops::Add for Elements<T> {
    type Output = Self;

    fn add(self, by: Elements<T>) -> Self {
        Self::new(self.value() + by.value())
    }
}

//--------------//
// Full Columns //
//--------------//

macro_rules! newtype {
    ($(#[$doc:meta])* $vis:vis $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(transparent)]
        $vis struct $name(NonZeroUsize);

        impl $name {
            pub(super) const fn new(value: NonZeroUsize) -> Self {
                Self(value)
            }

            pub(super) const fn value(self) -> NonZeroUsize {
                self.0
            }
        }
    }
}

newtype! {
    /// A strongly typed dimension for the `K` value when multiplying `MxK` by `KxN` matrics.
    pub(super) DimK
}

