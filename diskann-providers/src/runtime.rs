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
//! external handles). Under the `compio` feature the same blocking semantics
//! are provided by a `compio::runtime::Runtime`, which is single-flavored and
//! bound to the thread that created it; consequently the "external handle"
//! constructors are tokio-only and the wrapper types are `!Send` there.
//!
//! When both features are enabled, `tokio` wins so that existing users observe
//! no change. Compiling without either backend fails via the `compile_error!`
//! in the `diskann` crate root (this module is only compiled when one is set).

use std::future::Future;

/// An owned async runtime kept alive by the wrapper.
#[derive(Debug)]
pub(crate) struct Runtime(InnerRuntime);

/// A cloneable handle used to block on futures.
#[derive(Clone, Debug)]
pub(crate) struct Handle(InnerHandle);

#[derive(Debug)]
enum InnerRuntime {
    #[cfg(feature = "tokio")]
    Tokio(tokio::runtime::Runtime),
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio(compio::runtime::Runtime),
}

#[derive(Clone, Debug)]
enum InnerHandle {
    #[cfg(feature = "tokio")]
    Tokio(tokio::runtime::Handle),
    #[cfg(all(feature = "compio", not(feature = "tokio")))]
    Compio(compio::runtime::Runtime),
}

impl Runtime {
    /// Create a runtime with multiple worker threads.
    ///
    /// Under `compio` (thread-per-core) there is no multi-threaded flavor; the
    /// single-threaded runtime is used instead.
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
            Self(InnerRuntime::Compio(current_thread_compio()))
        }
    }

    /// Create a single-threaded (current-thread) runtime.
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
            Self(InnerRuntime::Compio(current_thread_compio()))
        }
    }

    /// A handle that can drive futures independently of `self`.
    pub(crate) fn handle(&self) -> Handle {
        match &self.0 {
            #[cfg(feature = "tokio")]
            InnerRuntime::Tokio(rt) => Handle(InnerHandle::Tokio(rt.handle().clone())),
            #[cfg(all(feature = "compio", not(feature = "tokio")))]
            InnerRuntime::Compio(rt) => Handle(InnerHandle::Compio(rt.clone())),
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
            InnerHandle::Compio(rt) => rt.block_on(future),
        }
    }
}

#[cfg(all(feature = "compio", not(feature = "tokio")))]
fn current_thread_compio() -> compio::runtime::Runtime {
    #[expect(clippy::expect_used)]
    compio::runtime::Runtime::new().expect("failed to create compio runtime")
}
