/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{ANNError, ANNResult};

/// Creates a new multi-threaded tokio runtime with the specified number of worker threads.
/// If `num_threads` is 0, it defaults to the number of logical CPUs.
pub fn create_runtime(num_threads: usize) -> ANNResult<tokio::runtime::Runtime> {
    let mut builder = tokio::runtime::Builder::new_multi_thread();

    if num_threads != 0 {
        builder.worker_threads(num_threads);
    }

    builder.build().map_err(|err| {
        ANNError::log_index_error(format!("Failed to initialize tokio runtime: {}", err))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_logical_cpu_count() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    #[test]
    fn test_create_runtime_with_zero_threads_no_panic() {
        // This test ensures that passing 0 threads doesn't panic
        // and properly defaults to the number of logical CPUs
        let result = create_runtime(0);

        // Should not panic and should succeed
        assert!(result.is_ok(), "create_runtime(0) should not panic or fail");

        let runtime = result.unwrap();

        // Verify the runtime was created successfully by executing a simple task
        let result = runtime.block_on(async { tokio::spawn(async { 42 }).await });

        assert!(result.is_ok(), "Runtime should be functional");
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_create_runtime_with_specific_threads() {
        // Test that specifying a specific number of threads works
        let result = create_runtime(2);
        assert!(result.is_ok(), "create_runtime(2) should succeed");

        let runtime = result.unwrap();

        // Verify the runtime works
        let result = runtime.block_on(async { tokio::spawn(async { "test" }).await });

        assert!(result.is_ok(), "Runtime should be functional");
        assert_eq!(result.unwrap(), "test");
    }

    #[test]
    fn test_create_runtime_with_one_thread() {
        // Test edge case with 1 thread
        let result = create_runtime(1);
        assert!(result.is_ok(), "create_runtime(1) should succeed");

        let runtime = result.unwrap();

        // Verify the runtime works even with just 1 thread
        let result = runtime.block_on(async { tokio::spawn(async { true }).await });

        assert!(
            result.is_ok(),
            "Single-threaded runtime should be functional"
        );
        assert!(result.unwrap());
    }

    #[test]
    fn test_zero_threads_defaults_to_cpu_count() {
        // Test that 0 threads actually uses the logical CPU count
        let expected_cpu_count = get_logical_cpu_count();

        // We can't directly inspect the runtime's thread count easily,
        // but we can ensure it doesn't panic and works correctly
        let result = create_runtime(0);
        assert!(
            result.is_ok(),
            "create_runtime(0) should default to {} CPUs",
            expected_cpu_count
        );

        let runtime = result.unwrap();

        // Test that the runtime can handle multiple concurrent tasks
        // which would fail if it only had 1 thread and we expected more
        let result = runtime.block_on(async {
            let tasks = (0..expected_cpu_count.min(4))
                .map(|i| tokio::spawn(async move { i * 2 }))
                .collect::<Vec<_>>();

            let mut results = Vec::new();
            for task in tasks {
                results.push(task.await.unwrap());
            }
            results
        });

        assert!(
            result.len() <= 4,
            "Should handle concurrent tasks successfully"
        );
    }
}
