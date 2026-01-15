/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Create a multi-threaded runtime with `num_threads`.
pub(crate) fn runtime(num_threads: usize) -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()?)
}

/// Create a current-thread runtime and block on the given future.
/// Only for functions that don't need multi-threading
pub(crate) fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current thread runtime initialization failed")
        .block_on(future)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtimes() {
        for num_threads in [1, 2, 4, 8] {
            let rt = runtime(num_threads).unwrap();
            let metrics = rt.metrics();
            assert_eq!(metrics.num_workers(), num_threads);
        }
    }
}
