/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod rowmajor;

/// Various view types (types such as [`MatrixView`] that add semantic meaning to blobs
/// of data) need both immutable and mutable variants.
///
/// This trait can be implemented by wrappers for immutable and mutable slice references,
/// allowing for a common code path for immutable and mutable view types.
///
/// The main goal is to provide a way of retrieving an underlying dense slice, which can
/// then be used as the building block for higher level abstractions.
///
/// # Safety
///
/// This trait is unsafe because it requires `as_slice` to be idempotent (and unsafe code
/// relies on this).
///
/// In other words: `as_slice` must **always** return the same slice with the same length.
pub unsafe trait DenseData {
    type Elem;

    /// Return the underlying data as a slice.
    fn as_slice(&self) -> &[Self::Elem];
}

/// A mutable companion to `DenseData`.
///
/// This trait allows mutable methods on view types to be selectively enabled when data
/// underlying the type is mutable.
///
/// # Safety
///
/// This trait is unsafe because it requires `as_slice` to be idempotent (and unsafe code
/// relies on this).
///
/// In other words: `as_slice` must **always** return the same slice with the same length.
///
/// Additionally, the returned slice must span the exact same memory as `as_slice`.
pub unsafe trait MutDenseData: DenseData {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for &[T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for &mut [T] {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement and returns a slice spanning the same
// range as `as_slice`.
unsafe impl<T> MutDenseData for &mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement.
unsafe impl<T> DenseData for Box<[T]> {
    type Elem = T;
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

// SAFETY: This fulfills the idempotency requirement and returns a slice spanning the same
// memory as `as_slice`.
unsafe impl<T> MutDenseData for Box<[T]> {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}
