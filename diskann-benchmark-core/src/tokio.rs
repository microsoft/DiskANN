/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Create a generic multi-threaded runtime with `num_threads`.
///
/// No guarantees are made about the returned [`tokio::runtime::Runtime`] except that it
/// will have `num_threads` workers.
pub fn runtime(num_threads: usize) -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_threads)
        .build()?)
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
