/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Return a [current-thread runtime](https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.new_current_thread).
///
/// This just a thin convenience wrapper around the builder interface in the linked
/// documentation.
pub(crate) fn current_thread_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current thread runtime initialization should succeed for tests")
}
