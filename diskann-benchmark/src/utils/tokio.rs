/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Create a current-thread runtime and block on the given future.
/// Only for functions that don't need multi-threading
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current thread runtime initialization failed")
        .block_on(future)
}
