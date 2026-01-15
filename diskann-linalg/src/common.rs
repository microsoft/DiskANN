/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Indicate whether a matrix should be implicitly transposed for an operation.
#[derive(Debug, Clone, Copy)]
pub enum Transpose {
    /// Use a provided matrix directly.
    None,
    /// Use the transpose of a matrix.
    Ordinary,
}

impl Transpose {
    /// Return whether or not the enum is `Transpose::Ordinary`.
    pub fn is_transpose(&self) -> bool {
        match self {
            Self::None => false,
            Self::Ordinary => true,
        }
    }

    /// Forward one of the arguments, depending on the value of `self`.
    pub fn forward<T>(&self, if_none: T, if_transpose: T) -> T {
        match self {
            Self::None => if_none,
            Self::Ordinary => if_transpose,
        }
    }

    /// Call exactly one of the arguments depending on the value of `self` and return the
    /// result.
    pub fn call<F, G, T>(&self, if_none: F, if_transpose: G) -> T
    where
        F: Fn() -> T,
        G: Fn() -> T,
    {
        match self {
            Self::None => if_none(),
            Self::Ordinary => if_transpose(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn test_is_transpose() {
        assert!(!(Transpose::None).is_transpose());
        assert!((Transpose::Ordinary).is_transpose());
    }

    #[test]
    fn test_forward() {
        assert_eq!((Transpose::None).forward(1, 2), 1);
        assert_eq!((Transpose::Ordinary).forward(1, 2), 2);
    }

    #[test]
    fn test_call() {
        // None
        let a_called = AtomicBool::new(false);
        let b_called = AtomicBool::new(false);

        let a = || {
            a_called.store(true, Ordering::Relaxed);
            1
        };

        let b = || {
            b_called.store(true, Ordering::Relaxed);
            2
        };

        assert_eq!((Transpose::None).call(a, b), 1);

        // Make sure *only* `a` was called
        assert!(a_called.load(Ordering::Relaxed));
        assert!(!b_called.load(Ordering::Relaxed));

        // Ordinary
        let a_called = AtomicBool::new(false);
        let b_called = AtomicBool::new(false);

        let a = || {
            a_called.store(true, Ordering::Relaxed);
            1
        };

        let b = || {
            b_called.store(true, Ordering::Relaxed);
            2
        };

        assert_eq!((Transpose::Ordinary).call(a, b), 2);

        // Make sure *only* `a` was called
        assert!(!a_called.load(Ordering::Relaxed));
        assert!(b_called.load(Ordering::Relaxed));
    }
}
