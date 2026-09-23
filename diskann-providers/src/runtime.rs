/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Backend facade for the synchronous index wrappers.
//!
//! [`crate::index::wrapped_async`] exposes synchronous APIs that block on an
//! owned or borrowed async runtime. Under the default `tokio` feature this
//! module wraps a `tokio::runtime::Runtime` / `Handle` pair, preserving
//! upstream behavior (including multi-threaded and current-thread flavors and
//! external handles).
//!
//! Under the `compio` feature the facade is *stateless*: compio is
//! thread-per-core, and its `Runtime` (an `Rc`-shared executor/driver pair) is
//! neither `Send` nor `Sync`, so the wrapper must not hold one — otherwise the
//! wrapped index loses `Send + Sync`. Instead, compio runtimes are owned by
//! each worker thread's entry loop, and the facade markers resolve the
//! *current thread's* runtime through `Runtime::try_current()` on every
//! blocking call. Calling outside of a compio runtime context panics. The
//! "external handle" constructors (`new_with_handle`, `load_with_handle`)
//! remain tokio-only, since a compio runtime can never be entered from a
//! foreign thread.
//!
//! When both features are enabled, `tokio` wins so that existing users observe
//! no change. Compiling without either backend fails via the `compile_error!`
//! in the `diskann` crate root (this module is only compiled when one is set).

use std::future::Future;

/// An owned async runtime kept alive by the wrapper (tokio backend only).
#[derive(Debug)]
pub(crate) struct Runtime(InnerRuntime);

/// A cloneable handle used to block on futures.
#[derive(Clone, Debug)]
pub(crate) struct Handle(InnerHandle);

#[derive(Debug)]
enum InnerRuntime {
    #[cfg(feature = "tokio")]
    Tokio(tokio::runtime::Runtime),
    /// Marker only: under compio the worker thread's entry loop owns the
    /// runtime; blocking calls resolve it via `Runtime::try_current()`.
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio,
}

#[derive(Clone, Debug)]
enum InnerHandle {
    #[cfg(feature = "tokio")]
    Tokio(tokio::runtime::Handle),
    /// Marker only, see [`InnerRuntime::Compio`].
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio,
}

impl Runtime {
    /// Create a runtime with multiple worker threads.
    ///
    /// Under `compio` (thread-per-core) there is no multi-threaded flavor and
    /// the wrapper does not own any runtime; only a marker is stored.
    pub(crate) fn multi_thread() -> Self {
        #[cfg(feature = "tokio")]
        {
            #[expect(clippy::expect_used)]
            let rt = tokio::runtime::Builder::new_multi_thread()
                .build()
                .expect("failed to create tokio runtime");
            Self(InnerRuntime::Tokio(rt))
        }
        #[cfg(all(feature = "compio", not(feature = "tokio")))]
        {
            Self(InnerRuntime::Compio)
        }
    }

    /// Create a single-threaded (current-thread) runtime.
    ///
    /// Under `compio` the wrapper does not own any runtime; only a marker is
    /// stored.
    pub(crate) fn current_thread() -> Self {
        #[cfg(feature = "tokio")]
        {
            #[expect(clippy::expect_used)]
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("failed to create tokio runtime");
            Self(InnerRuntime::Tokio(rt))
        }
        #[cfg(all(feature = "compio", not(feature = "tokio")))]
        {
            Self(InnerRuntime::Compio)
        }
    }

    /// A handle that can drive futures independently of `self`.
    pub(crate) fn handle(&self) -> Handle {
        match &self.0 {
            #[cfg(feature = "tokio")]
            InnerRuntime::Tokio(rt) => Handle(InnerHandle::Tokio(rt.handle().clone())),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerRuntime::Compio => Handle(InnerHandle::Compio),
        }
    }
}

impl Handle {
    /// Wrap an externally owned tokio runtime handle.
    #[cfg(feature = "tokio")]
    pub(crate) fn from_tokio(handle: tokio::runtime::Handle) -> Self {
        Self(InnerHandle::Tokio(handle))
    }

    /// Run `future` to completion, blocking the current thread.
    pub(crate) fn block_on<F: Future>(&self, future: F) -> F::Output {
        match &self.0 {
            #[cfg(feature = "tokio")]
            InnerHandle::Tokio(handle) => handle.block_on(future),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerHandle::Compio => current_compio().block_on(future),
        }
    }
}

/// Resolve the compio runtime of the current thread.
///
/// Worker threads run one compio runtime per core/thread; every blocking call
/// of the wrappers must therefore happen inside that runtime's context (from
/// within `block_on`/`enter` of the thread's own runtime).
#[cfg(all(feature = "compio", not(feature = "tokio")))]
fn current_compio() -> compio::runtime::Runtime {
    compio::runtime::Runtime::try_current().unwrap_or_else(|| {
        panic!(
            "no compio runtime on the current thread: the compio backend is \
             thread-per-core and resolves the thread's own runtime on every \
             blocking call; construct and use the wrappers from within the \
             worker thread's runtime context (block_on/enter)"
        )
    })
}
