/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
*/

use std::future::Future;
use std::sync::Arc;

use tokio::{sync::Mutex, task::JoinSet};

/// Runs `num_tasks` concurrent tasks that pull items from `iterator` and process
/// them with `task_fn`.
///
/// This helper spawns `num_tasks` tasks, each of which pulls items from a shared
/// iterator and processes them with the provided async function. This is useful
/// for parallelizing work across an iterator without creating one task per item.
///
/// Each task will:
/// 1. Lock the iterator and pull the next item
/// 2. If an item exists, call `task_fn` on it
/// 3. Repeat until the iterator is exhausted
///
/// All tasks run concurrently and share the same iterator through an Arc<Mutex<_>>.
///
/// # Arguments
/// * `iterator` - An iterator of items to process
/// * `num_tasks` - Number of concurrent tasks to spawn
/// * `task_fn` - Async function to call on each item. Must be Clone to share across tasks.
///
/// # Panics
/// * Panics if a spawned task panics
pub async fn run<I, T, F, Fut>(iterator: I, num_tasks: usize, task_fn: F)
where
    I: Iterator<Item = T> + Send + 'static,
    T: Send + 'static,
    F: Fn(T) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future + Send + 'static,
{
    let mut tasks = JoinSet::new();
    let iterator = Arc::new(Mutex::new(iterator));

    for _ in 0..num_tasks {
        let iterator_clone = iterator.clone();
        let task_fn = task_fn.clone();

        tasks.spawn(async move {
            loop {
                let item = {
                    let mut guard = iterator_clone.lock().await;
                    guard.next()
                };

                match item {
                    Some(item) => {
                        let _ = task_fn(item).await;
                    }
                    None => break,
                }
            }
        });
    }

    // Wait for all tasks to complete
    while let Some(res) = tasks.join_next().await {
        res.expect("A spawned task panicked");
    }
}
