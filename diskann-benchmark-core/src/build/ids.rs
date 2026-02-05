/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Tools for creating external IDs for index construction.

use std::marker::PhantomData;

use diskann::{ANNError, ANNErrorKind, ANNResult};
use diskann_utils::future::AsyncFriendly;

/// Convert an implicit data index to an external ID.
///
/// The [`crate::build::Build`] trait uses indices in a range `0..N` to refer to items
/// for insertion. This trait provides a bridge between these indices and the actual external
/// ID type used by a [`diskann::provider::DataProvider`].
pub trait ToId<T>: std::fmt::Debug + AsyncFriendly {
    /// Convert the given implicit index `i` to an external ID of type `T`.
    fn to_id(&self, i: usize) -> ANNResult<T>;
}

/// An extension of [`ToId`] that tracks the number of IDs.
pub trait ToIdSized<T>: ToId<T> {
    /// Returns the number of IDs that can be produced.
    ///
    /// Implementations must ensure that for all `i` in `0..self.len()`, `self.to_id(i)` returns
    /// a valid ID. Indexing outside of this range should return an error.
    fn len(&self) -> usize;

    /// Returns `true` only if `self.len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

//----------//
// Identity //
//----------//

/// A [`ToId`] implementation that treats the implicit index as the external ID.
///
/// This requires that the external ID type `T` can be constructed from a `usize`.
pub struct Identity<T>(PhantomData<T>);

impl<T> Identity<T> {
    /// Construct a new [`Identity`].
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<T> Clone for Identity<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Identity<T> {}

impl<T> Default for Identity<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for Identity<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("Identity<{}>", std::any::type_name::<T>()))
            .field(&"_")
            .finish()
    }
}

impl<T> ToId<T> for Identity<T>
where
    T: TryFrom<usize, Error: std::error::Error + AsyncFriendly> + AsyncFriendly,
{
    fn to_id(&self, i: usize) -> ANNResult<T> {
        T::try_from(i).map_err(ANNError::opaque)
    }
}

//-------//
// Slice //
//-------//

/// An implementation of [`ToId`] and [`ToIdSized`] that encodes external IDs as a slice.
///
/// ```rust
/// use diskann_benchmark_core::build::ids::{Slice, ToId, ToIdSized};
///
/// let slice = Slice::new([10u32, 20, 30].into());
/// assert_eq!(slice.len(), 3);
/// assert_eq!(slice.to_id(0).unwrap(), 10);
/// assert_eq!(slice.to_id(1).unwrap(), 20);
/// assert_eq!(slice.to_id(2).unwrap(), 30);
/// assert!(slice.to_id(3).is_err());
/// ```
#[derive(Debug)]
pub struct Slice<T>(Box<[T]>);

impl<T> Slice<T> {
    /// Construct a new [`Slice`] from the given boxed slice.
    ///
    /// The contents of the [`Slice`] will be identical to the provided box.
    pub fn new(slice: Box<[T]>) -> Self {
        Self(slice)
    }
}

impl<T> ToId<T> for Slice<T>
where
    T: Clone + std::fmt::Debug + AsyncFriendly,
{
    fn to_id(&self, i: usize) -> ANNResult<T> {
        self.0.get(i).cloned().ok_or_else(|| {
            ANNError::message(
                ANNErrorKind::Opaque,
                format!(
                    "tried to index a slice of length {} at index {}",
                    self.0.len(),
                    i
                ),
            )
        })
    }
}

impl<T> ToIdSized<T> for Slice<T>
where
    T: Clone + std::fmt::Debug + AsyncFriendly,
{
    fn len(&self) -> usize {
        self.0.len()
    }
}

//-------//
// Range //
//-------//

/// An implementation of [`ToId`] and [`ToIdSized`] that encodes external IDs as a range.
///
/// ```rust
/// use diskann_benchmark_core::build::ids::{Range, ToId, ToIdSized};
///
/// let range = Range::new(100u32..105);
/// assert_eq!(range.len(), 5);
/// assert_eq!(range.to_id(0).unwrap(), 100);
/// assert_eq!(range.to_id(1).unwrap(), 101);
/// assert_eq!(range.to_id(4).unwrap(), 104);
/// assert!(range.to_id(5).is_err());
/// ```
#[derive(Debug)]
pub struct Range<T>(std::ops::Range<T>);

impl<T> Range<T> {
    /// Construct a new [`Range`] from the given standard library range.
    pub fn new(range: std::ops::Range<T>) -> Self {
        Self(range)
    }
}

macro_rules! impl_range {
    ($T:ty) => {
        impl ToId<$T> for Range<$T> {
            fn to_id(&self, i: usize) -> ANNResult<$T> {
                self.0.clone().nth(i).ok_or_else(|| {
                    ANNError::message(
                        ANNErrorKind::Opaque,
                        format!(
                            "tried to index a range of length {} at index {}",
                            self.0.len(),
                            i
                        ),
                    )
                })
            }
        }

        impl ToIdSized<$T> for Range<$T> {
            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
}

impl_range!(u32);

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_is_copy<T: Copy>(_x: T) {}

    #[test]
    fn identity_to_id_u32() {
        let identity = Identity::<u32>::default();
        assert_eq!(identity.to_id(0).unwrap(), 0u32);
        assert_eq!(identity.to_id(42).unwrap(), 42u32);
        assert_eq!(identity.to_id(u32::MAX as usize).unwrap(), u32::MAX);

        assert_is_copy(identity);
    }

    #[test]
    fn identity_to_id_u64() {
        let identity = Identity::<u64>::new();
        assert_eq!(identity.to_id(0).unwrap(), 0u64);
        assert_eq!(identity.to_id(usize::MAX).unwrap(), usize::MAX as u64);
    }

    #[test]
    fn identity_to_id_overflow() {
        let identity = Identity::<u8>::new();
        assert!(identity.to_id(256).is_err());
    }

    #[test]
    fn identity_debug() {
        let identity = Identity::<u32>::default();
        let debug_str = format!("{:?}", identity);
        assert!(debug_str.contains("Identity"));
        assert!(debug_str.contains("u32"));
    }

    //-------------//
    // Slice Tests //
    //-------------//

    #[test]
    fn test_slice() {
        let slice = Slice::new(vec![10u32, 20, 30, 40, 50].into_boxed_slice());

        assert_eq!(slice.len(), 5);
        assert!(!slice.is_empty());

        assert_eq!(slice.to_id(0).unwrap(), 10);
        assert_eq!(slice.to_id(1).unwrap(), 20);
        assert_eq!(slice.to_id(2).unwrap(), 30);
        assert_eq!(slice.to_id(3).unwrap(), 40);
        assert_eq!(slice.to_id(4).unwrap(), 50);
        assert!(slice.to_id(5).is_err());

        // Empty Slice
        let slice = Slice::<u32>::new(vec![].into_boxed_slice());
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);
    }

    //-------------//
    // Range Tests //
    //-------------//

    #[test]
    fn test_range() {
        let range = Range::new(100u32..105);

        assert_eq!(range.len(), 5);
        assert!(!range.is_empty());

        assert_eq!(range.to_id(0).unwrap(), 100);
        assert_eq!(range.to_id(1).unwrap(), 101);
        assert_eq!(range.to_id(4).unwrap(), 104);

        assert!(range.to_id(5).is_err());

        // Empty Range
        let range = Range::new(5u32..5);
        assert_eq!(range.len(), 0);
        assert!(range.is_empty());
    }
}
