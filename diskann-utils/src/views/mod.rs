/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod rowmajor;

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

/// A mutable companion to [`DenseData`].
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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::lazy_format;

    /// Test the that provided representation yields a slice with the expected base pointer
    /// and length.
    fn test_dense_data_repr<T, Repr>(
        ptr: *const T,
        len: usize,
        repr: Repr,
        context: &dyn std::fmt::Display,
    ) where
        T: Copy,
        Repr: DenseData<Elem = T>,
    {
        let retrieved = repr.as_slice();
        assert_eq!(retrieved.len(), len, "{}", context);
        assert_eq!(retrieved.as_ptr(), ptr, "{}", context);
    }

    /// Set the underlying data for the provided representation to the following:
    ///
    /// [base, base + increment, base + increment + increment, ...]
    fn set_mut_dense_data_repr<T, Repr>(repr: &mut Repr, base: T, increment: T)
    where
        T: Copy + std::ops::Add<Output = T>,
        Repr: DenseData<Elem = T> + MutDenseData,
    {
        let slice = repr.as_mut_slice();
        for i in 0..slice.len() {
            if i == 0 {
                slice[i] = base;
            } else {
                slice[i] = slice[i - 1] + increment;
            }
        }
    }

    #[test]
    fn slice_implements_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let data: Vec<f32> = vec![0.0; len];
            let slice = data.as_slice();
            test_dense_data_repr(slice.as_ptr(), slice.len(), slice, &context);
        }
    }

    #[test]
    fn mut_slice_implements_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let mut data: Vec<f32> = vec![0.0; len];
            let slice = data.as_mut_slice();

            let ptr = slice.as_ptr();
            let len = slice.len();
            test_dense_data_repr(ptr, len, slice, &context);
        }
    }

    #[test]
    fn mut_slice_implements_mut_dense_data_repr() {
        for len in 0..10 {
            let context = lazy_format!("len = {}", len);
            let mut data: Vec<f32> = vec![0.0; len];
            let mut slice = data.as_mut_slice();

            let base = 2.0;
            let increment = 1.0;
            set_mut_dense_data_repr(&mut slice, base, increment);

            for (i, &v) in slice.iter().enumerate() {
                let context = lazy_format!("entry {}, {}", i, context);
                assert_eq!(v, base + increment * (i as f32), "{}", context);
            }
        }
    }

    #[test]
    fn test_box_slice_dense_data_impls() {
        // Test Box<[T]> implementations
        let data: Box<[f32]> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].into();
        let ptr = data.as_ptr();
        let len = data.len();

        // Test DenseData impl for Box<[T]>
        test_dense_data_repr(ptr, len, data, &lazy_format!("Box<[T]> DenseData"));

        // Test MutDenseData impl for Box<[T]>
        let mut data: Box<[f32]> = vec![0.0; 6].into();
        set_mut_dense_data_repr(&mut data, 1.0, 2.0);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(
                v,
                1.0 + 2.0 * (i as f32),
                "Box<[T]> MutDenseData at index {}",
                i
            );
        }
    }
}
