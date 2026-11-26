/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// A hook for spawning futures.
///
/// Implementations should schedule the supplied future and return its corresponding
/// [`tokio::task::JoinHandle`]. The future may be wrapped for instrumentation, but the
/// returned handle must represent its completion and support cancellation.
pub trait Spawn {
    /// Spawn `fut` as a task.
    fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static;
}

impl Spawn for tokio::runtime::Handle {
    fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        tokio::runtime::Handle::spawn(self, future)
    }
}

impl Spawn for tokio::runtime::Runtime {
    fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        tokio::runtime::Runtime::spawn(self, future)
    }
}
