/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A minimal runtime-agnostic facade for task spawning.
//!
//! DiskANN only requires three capabilities from its async backend:
//!
//! * spawning a task,
//! * awaiting the task's output (with a join error if it panicked or was cancelled),
//! * joining a dynamic set of such tasks.
//!
//! Under the default `tokio` feature, this module delegates directly to
//! `tokio::spawn` and `tokio::task::JoinSet`, preserving upstream behavior.
//! Under the `compio` feature, the same semantics are provided on top of
//! `compio::runtime`'s thread-per-core runtime (`Runtime::spawn` +
//! `block_on`; compio keeps no detached task registry, so `JoinSet` joins its
//! tasks in FIFO order instead of completion order).
//!
//! When both features are enabled, `tokio` wins so that existing users observe
//! no change. Compiling without either backend fails with a `compile_error!`
//! in the crate root.

#[cfg(all(feature = "compio", not(feature = "tokio")))]
use std::collections::VecDeque;
use std::{
    fmt,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

//////////////////
// JoinError //
//////////////////

/// An error indicating that a spawned task failed to run to completion
/// (it either panicked or was cancelled).
///
/// The compio backend reports its error as a formatted message: its native
/// error carries a panic payload (`Box<dyn Any + Send>`) that is neither
/// `Sync` nor ever resumed by this crate, while tokio's error type is kept
/// intact.
#[derive(Debug)]
pub(crate) struct JoinError(Inner);

#[derive(Debug)]
enum Inner {
    #[cfg(feature = "tokio")]
    Tokio(tokio::task::JoinError),
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio(String),
}

impl fmt::Display for JoinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            #[cfg(feature = "tokio")]
            Inner::Tokio(err) => fmt::Display::fmt(err, f),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            Inner::Compio(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for JoinError {}

////////////////
// JoinHandle //
////////////////

/// A handle for awaiting the output of a spawned task.
///
/// Dropping all handles to a task may cancel it, mirroring the semantics of
/// both backends.
#[derive(Debug)]
pub(crate) struct JoinHandle<T>(InnerJoin<T>);

#[derive(Debug)]
enum InnerJoin<T> {
    #[cfg(feature = "tokio")]
    Tokio(tokio::task::JoinHandle<T>),
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio(compio::runtime::JoinHandle<T>),
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `self` is never moved out; only the inner enum is polled in place.
        let inner = unsafe { self.get_unchecked_mut() };
        match &mut inner.0 {
            #[cfg(feature = "tokio")]
            InnerJoin::Tokio(handle) => Pin::new(handle)
                .poll(cx)
                .map(|out| out.map_err(|err| JoinError(Inner::Tokio(err)))),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerJoin::Compio(handle) => Pin::new(handle)
                .poll(cx)
                .map(|out| out.map_err(|err| JoinError(Inner::Compio(err.to_string())))),
        }
    }
}

/// Spawn a task on the active runtime backend.
///
/// `Send` bounds match the stricter (tokio) backend; the compio backend would
/// not require them, but keeping them preserves a backend-independent API.
pub(crate) fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    #[cfg(feature = "tokio")]
    {
        JoinHandle(InnerJoin::Tokio(tokio::spawn(future)))
    }
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    {
        JoinHandle(InnerJoin::Compio(compio::runtime::spawn(future)))
    }
}

//////////
// JoinSet
//////////

/// A collection of tasks spawned on the runtime backend.
///
/// `join_next` awaits a single task. Under `tokio`, tasks are joined in
/// completion order; under `compio` (which has no task registry), they are
/// joined in spawn order. Call sites only require that all tasks eventually
/// yield their result, which both orderings satisfy.
#[derive(Debug)]
pub(crate) struct JoinSet<T>(InnerSet<T>);

#[derive(Debug)]
enum InnerSet<T> {
    #[cfg(feature = "tokio")]
    Tokio(tokio::task::JoinSet<T>),
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio(VecDeque<compio::runtime::JoinHandle<T>>),
}

impl<T> JoinSet<T>
where
    T: Send + 'static,
{
    /// Create an empty set.
    pub(crate) fn new() -> Self {
        JoinSet(InnerSet::new())
    }

    /// Spawn a task and add its handle to the set.
    pub(crate) fn spawn<F>(&mut self, task: F)
    where
        F: Future<Output = T> + Send + 'static,
    {
        InnerSet::spawn(&mut self.0, task)
    }

    /// Await the completion of one task in the set.
    ///
    /// Returns `None` once the set is empty.
    pub(crate) async fn join_next(&mut self) -> Option<Result<T, JoinError>> {
        match &mut self.0 {
            #[cfg(feature = "tokio")]
            InnerSet::Tokio(set) => set
                .join_next()
                .await
                .map(|res| res.map_err(|err| JoinError(Inner::Tokio(err)))),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerSet::Compio(queue) => {
                // Take the handle out: a finished compio handle cannot be
                // polled a second time, and joining consumes it anyway.
                let handle = queue.pop_front()?;
                Some(
                    handle
                        .await
                        .map_err(|err| JoinError(Inner::Compio(err.to_string()))),
                )
            }
        }
    }
}

impl<T> InnerSet<T>
where
    T: Send + 'static,
{
    fn new() -> Self {
        #[cfg(feature = "tokio")]
        {
            InnerSet::Tokio(tokio::task::JoinSet::new())
        }
        #[cfg(all(feature = "compio", not(feature = "tokio")))]
        {
            InnerSet::Compio(VecDeque::new())
        }
    }

    fn spawn<F>(&mut self, task: F)
    where
        F: Future<Output = T> + Send + 'static,
    {
        match self {
            #[cfg(feature = "tokio")]
            InnerSet::Tokio(set) => {
                set.spawn(task);
            }
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerSet::Compio(queue) => {
                queue.push_back(compio::runtime::spawn(task));
            }
        }
    }
}

#[cfg(all(test, feature = "compio", not(feature = "tokio")))]
mod tests {
    //! End-to-end proof that the `compio` backend can spawn, join and surface
    //! task results through the runtime facade without tokio enabled.

    use super::{JoinSet, spawn};

    #[test]
    fn spawn_returns_task_output_under_compio() {
        let rt = compio::runtime::Runtime::new().unwrap();
        let out = rt.block_on(async { spawn(async { 42u32 }).await });
        assert!(matches!(out, Ok(42)));
    }

    #[test]
    fn join_set_joins_all_tasks_under_compio() {
        let rt = compio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut set = JoinSet::new();
            for i in 0..8u32 {
                set.spawn(async move { i * 2 });
            }

            let mut seen = Vec::with_capacity(8);
            while let Some(joined) = set.join_next().await {
                seen.push(joined.unwrap());
            }

            seen.sort_unstable();
            let expected: Vec<u32> = (0..8u32).map(|i| i * 2).collect();
            assert_eq!(seen, expected, "every spawned task must be joined");
        });
    }

    #[test]
    fn join_error_is_a_std_error_under_compio() {
        let rt = compio::runtime::Runtime::new().unwrap();
        // A panicking task surfaces as `Err(JoinError)` on join for both backends.
        let joined = rt.block_on(async { spawn(async { panic!("boom") }).await });
        let err = joined.unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("panic"),
            "unexpected join error message: {message}"
        );
        let _dynamic: &dyn std::error::Error = &err;
    }
}
