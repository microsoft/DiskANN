/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

////////////////
// SendFuture //
////////////////

/// A simplified type alias for `Future<Output = T> + Send`.
pub trait SendFuture<T>: core::future::Future<Output = T> + Send {}
impl<T, U> SendFuture<T> for U where U: core::future::Future<Output = T> + Send {}

///////////////////
// AsyncFriendly //
///////////////////

/// The bounds `Send + Sync + 'static` are often required by type parameters to make Futures
/// `Send + Sync + 'static`.
///
/// This trait bundles these into a single super-trait for convenience.
pub trait AsyncFriendly: Send + Sync + 'static {}
impl<T> AsyncFriendly for T where T: Send + Sync + 'static {}

////////////////
// AssertSend //
////////////////

/// If your future is not `Send` enough, try this.
///
/// Async functions (i.e., those with the keyword async) rely on type inference to
/// automatically derive `Send` and `Sync` bounds for the returned `Future`s. Unfortunately,
/// `rustc` seems to throw away quite a bit of information when building large futures like
/// we have in `diskann_async`, which can result in large futures failing to be `Send`
/// due to the bound on an inner `Future` being forgotten.
///
/// The [`AssertSend`] is a hack that helps `rustc` realize that interior `Future`s are
/// indeed `Send` and helps when proving the auto trait for larger `Future`s.
///
/// This is mainly helpful when async functions take closures.
pub trait AssertSend: core::future::Future {
    fn send(self) -> impl core::future::Future<Output = Self::Output> + Send
    where
        Self: Sized + Send,
    {
        self
    }
}

impl<T: core::future::Future> AssertSend for T {}

///////////
// boxit //
///////////

/// Type erase the provided future by boxing it.
///
/// THis can potentially help with compilation time and future size bloat by factoring out
/// pieces of larger futures into opaque chunks.
pub fn boxit<'a, T>(
    fut: impl core::future::Future<Output = T> + Send + 'a,
) -> core::pin::Pin<Box<dyn core::future::Future<Output = T> + Send + 'a>> {
    Box::pin(fut)
}
