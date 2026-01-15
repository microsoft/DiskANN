/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub use sealed::Length;

/// A type for dynamically sized slices.
#[derive(Debug, Clone, Copy)]
pub struct Dynamic(pub usize);

/// Allow integers to be converted into `Dynamic`.
impl From<usize> for Dynamic {
    fn from(value: usize) -> Dynamic {
        Dynamic(value)
    }
}

/// A type for statically sized slices (slices with a length known at compile-time).
#[derive(Debug, Clone, Copy)]
pub struct Static<const N: usize>;

// SAFETY: `value` returns the same value for all copies of `self`.
unsafe impl Length for Dynamic {
    #[inline(always)]
    fn value(self) -> usize {
        self.0
    }
}

// SAFETY: `value` always returns the same value.
unsafe impl<const N: usize> Length for Static<N> {
    #[inline(always)]
    fn value(self) -> usize {
        N
    }
}

mod sealed {
    /// A trait allowing a type to yield a value.
    ///
    /// This allows the lengths of `BitSlice`s to be either static or dynamic.
    ///
    /// # Safety
    ///
    /// Implementations must ensure that `self.value()` always returns the same value for a
    /// given instance and that this value is preserved across all copies obtained from `self`.
    pub unsafe trait Length: Copy {
        fn value(self) -> usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length() {
        for i in 0..100 {
            assert_eq!(Dynamic(i).value(), i);
        }

        assert_eq!(Static::<0>.value(), 0);
        assert_eq!(Static::<10>.value(), 10);
        assert_eq!(Static::<20>.value(), 20);
        assert_eq!(Static::<37>.value(), 37);
    }
}
