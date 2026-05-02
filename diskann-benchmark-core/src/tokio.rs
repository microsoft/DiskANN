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

/// Create a generic multi-threaded runtime with `num_threads`.
///
/// After initial setup, the [`tokio::runtime::Builder`] will be passed to the closure `f`
/// for customization. Note that the builder provided to the callback will already be
/// initialized to contain `num_threads` threads.
pub fn runtime_with<F>(num_threads: usize, f: F) -> anyhow::Result<tokio::runtime::Runtime>
where
    F: FnOnce(&mut tokio::runtime::Builder),
{
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(num_threads);
    f(&mut builder);
    Ok(builder.build()?)
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

    #[test]
    fn test_runtime_with_threads() {
        for num_threads in [1, 2, 4, 8] {
            let rt = runtime_with(num_threads, |_| {}).unwrap();
            let metrics = rt.metrics();
            assert_eq!(metrics.num_workers(), num_threads);
        }
    }

    #[test]
    fn test_runtime_with_customizes_builder() {
        let rt = runtime_with(2, |builder| {
            builder.thread_name("custom-worker");
        })
        .unwrap();

        // Verify the runtime was created with the correct number of threads.
        assert_eq!(rt.metrics().num_workers(), 2);

        // Verify the thread name was applied by spawning work on the runtime
        // and checking the thread name from within a worker.
        let name = rt.block_on(async {
            tokio::task::spawn(async { std::thread::current().name().unwrap_or("").to_string() })
                .await
                .unwrap()
        });
        assert!(
            name.starts_with("custom-worker"),
            "expected thread name starting with 'custom-worker', got '{name}'",
        );
    }
}
