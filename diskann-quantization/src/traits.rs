/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt::Debug;

/// A trait implemented by a quantizer, indicating that it is capable of quantizing the
/// contents oF `From` into `To`.
pub trait CompressInto<From, To> {
    /// Errors that may occur during compression.
    type Error: std::error::Error + 'static + Send + Sync;

    /// An output type resulting from compression.
    type Output: Debug + Send + Sync;

    /// Compress the data in `From` into `To`.
    ///
    /// If an error is encountered, `To` must be left in a valid but undefined state.
    fn compress_into(&self, from: From, to: To) -> Result<Self::Output, Self::Error>;
}

/// A trait implemented by a quantizer, indicating that it is capable of quantizing the
/// contents oF `From` into `To` with additional help from an argument of type `A`.
///
/// One example use case of the additional argument would be a
/// [`crate::alloc::ScopedAllocator`] through which scratch allocations can be made.
pub trait CompressIntoWith<From, To, A> {
    /// Errors that may occur during compression.
    type Error: std::error::Error + 'static + Send + Sync;

    /// Compress the data in `From` into `To`.
    ///
    /// If an error is encountered, `To` must be left in a valid but undefined state.
    fn compress_into_with(&self, from: From, to: To, with: A) -> Result<(), Self::Error>;
}

/// Create a distance function (i.e. [`diskann_vector::DistanceFunction`]) from a quantizer.
pub trait AsFunctor<T> {
    fn as_functor(&self) -> T;
}
