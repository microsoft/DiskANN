/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Dense matrix types and the `DenseData` container abstraction.
//!
//! The [`Matrix`], [`MatrixView`], and [`MatrixViewMut`] types are re-exported from
//! [`crate::matrix`]. [`DenseData`]/[`MutDenseData`] abstract over owned and borrowed
//! contiguous storage and are used by [`crate::strided`].

pub use crate::matrix::{Generator, Init, Matrix, MatrixView, MatrixViewMut, TryFromError};

/// Abstraction over a type that can yield a dense slice of its contents.
///
/// # Safety
///
/// `as_slice` must be idempotent: it must **always** return the same slice with the same
/// length (unsafe code relies on this).
pub unsafe trait DenseData {
    type Elem;

    /// Return the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Elem];
}

/// A mutable companion to [`DenseData`].
///
/// # Safety
///
/// `as_mut_slice` must be idempotent and must span the exact same memory as
/// [`DenseData::as_slice`].
pub unsafe trait MutDenseData: DenseData {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for &[T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for &mut [T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement and spans the same memory as `as_slice`.
unsafe impl<T> MutDenseData for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement.
unsafe impl<T> DenseData for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: fulfills the idempotency requirement and spans the same memory as `as_slice`.
unsafe impl<T> MutDenseData for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}
