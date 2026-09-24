/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Task spawning for parallel graph operations.

use std::{fmt::Debug, future::Future, pin::Pin, sync::Arc};

use futures_channel::oneshot;
use futures_util::{
    future::{AbortHandle, Abortable},
    stream::{FuturesUnordered, StreamExt},
};
use thiserror::Error;

use crate::{ANNResult, provider::ExecutionContext};

/// A task that can be scheduled on an async executor.
pub type Task = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Schedules index work on an executor supplied by the caller.
///
/// Implementations must schedule tasks independently of the calling future, on the
/// executor used by the data provider. They must not block to run the task. A task
/// must be dropped if it is cancelled or its executor shuts down, so joiners can
/// observe that it did not return a result. A rejected task must not be scheduled.
pub trait TaskSpawner: Debug + Send + Sync + 'static {
    /// Schedule `task`, returning an error if it cannot be scheduled.
    fn spawn(&self, task: Task) -> ANNResult<()>;
}

/// Spawns work on the current Tokio runtime.
#[cfg(any(feature = "tokio-runtime", test))]
#[derive(Debug, Clone, Copy, Default)]
pub struct TokioSpawner;

#[cfg(any(feature = "tokio-runtime", test))]
impl TaskSpawner for TokioSpawner {
    fn spawn(&self, task: Task) -> ANNResult<()> {
        tokio::spawn(task);
        Ok(())
    }
}

/// A spawned task was cancelled or panicked before returning a result.
#[derive(Debug, Error)]
#[error("spawned task was cancelled or panicked before returning a result")]
pub struct TaskJoinError;

pub(crate) struct Spawned<T> {
    receiver: oneshot::Receiver<T>,
    abort: AbortHandle,
}

impl<T> Spawned<T> {
    pub(crate) async fn join(self) -> Result<T, TaskJoinError> {
        self.receiver.await.map_err(|_| TaskJoinError)
    }

    fn abort(&self) {
        self.abort.abort();
    }
}

pub(crate) fn spawn<C, F, T>(
    spawner: &Arc<dyn TaskSpawner>,
    context: &C,
    future: F,
) -> ANNResult<Spawned<T>>
where
    C: ExecutionContext,
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let (sender, receiver) = oneshot::channel();
    let (abort, registration) = AbortHandle::new_pair();
    let future = context.wrap_spawn(future);
    spawner.spawn(Box::pin(async move {
        let _ = Abortable::new(
            async move {
                let _ = sender.send(future.await);
            },
            registration,
        )
        .await;
    }))?;
    Ok(Spawned { receiver, abort })
}

/// Schedule a batch, cancelling previously scheduled tasks if a spawn is rejected.
pub(crate) fn spawn_all<C, I, F, T>(
    spawner: &Arc<dyn TaskSpawner>,
    context: &C,
    futures: I,
) -> ANNResult<Vec<Spawned<T>>>
where
    C: ExecutionContext,
    I: IntoIterator<Item = F>,
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    let mut handles = Vec::new();
    for future in futures {
        match spawn(spawner, context, future) {
            Ok(handle) => handles.push(handle),
            Err(err) => {
                for handle in &handles {
                    handle.abort();
                }
                return Err(err);
            }
        }
    }
    Ok(handles)
}

/// Tasks whose remaining work is cancelled when the group is dropped.
pub(crate) struct TaskGroup<T> {
    tasks: FuturesUnordered<oneshot::Receiver<T>>,
    aborts: Vec<AbortHandle>,
}

impl<T> TaskGroup<T> {
    pub(crate) fn new() -> Self {
        Self {
            tasks: FuturesUnordered::new(),
            aborts: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, spawned: Spawned<T>) {
        self.tasks.push(spawned.receiver);
        self.aborts.push(spawned.abort);
    }

    pub(crate) async fn join_next(&mut self) -> Option<Result<T, TaskJoinError>> {
        self.tasks
            .next()
            .await
            .map(|result| result.map_err(|_| TaskJoinError))
    }
}

impl<T> Drop for TaskGroup<T> {
    fn drop(&mut self) {
        for abort in &self.aborts {
            abort.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use super::*;
    use crate::provider::DefaultContext;

    #[derive(Debug)]
    struct RejectSpawner;

    impl TaskSpawner for RejectSpawner {
        fn spawn(&self, _task: Task) -> ANNResult<()> {
            Err(crate::ANNError::message("task rejected"))
        }
    }

    #[derive(Debug, Default)]
    struct RejectAfterFirst(AtomicUsize);

    impl TaskSpawner for RejectAfterFirst {
        fn spawn(&self, task: Task) -> ANNResult<()> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                TokioSpawner.spawn(task)
            } else {
                Err(crate::ANNError::message("task rejected"))
            }
        }
    }

    struct MarkDrop(Arc<AtomicBool>);
    impl Drop for MarkDrop {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[tokio::test]
    async fn reports_spawn_rejection() {
        let spawner: Arc<dyn TaskSpawner> = Arc::new(RejectSpawner);
        let result = spawn(&spawner, &DefaultContext, async { 42 })
            .err()
            .unwrap();
        assert!(result.to_string().contains("task rejected"));
    }

    #[tokio::test]
    async fn reports_task_panics_to_joiner() {
        let spawner: Arc<dyn TaskSpawner> = Arc::new(TokioSpawner);
        let result = spawn(&spawner, &DefaultContext, async {
            panic!("task failure");
        })
        .unwrap()
        .join()
        .await;
        assert!(matches!(result, Err(TaskJoinError)));
    }

    #[tokio::test]
    async fn rejected_batch_aborts_scheduled_tasks() {
        let spawner: Arc<dyn TaskSpawner> = Arc::new(RejectAfterFirst::default());
        let first_dropped = Arc::new(AtomicBool::new(false));
        let futures = (0..2).map(|i| {
            let marker = MarkDrop(if i == 0 {
                first_dropped.clone()
            } else {
                Arc::new(AtomicBool::new(false))
            });
            async move {
                let _marker = marker;
                pending::<()>().await;
            }
        });
        let error = spawn_all(&spawner, &DefaultContext, futures).err().unwrap();
        assert!(error.to_string().contains("task rejected"));

        for _ in 0..100 {
            if first_dropped.load(Ordering::Acquire) {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert!(first_dropped.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn dropping_group_aborts_pending_tasks() {
        let spawner: Arc<dyn TaskSpawner> = Arc::new(TokioSpawner);
        let dropped = Arc::new(AtomicBool::new(false));
        let (started, ready) = oneshot::channel();
        let mut group = TaskGroup::new();
        let dropped_clone = dropped.clone();
        group.push(
            spawn(&spawner, &DefaultContext, async move {
                let _mark_drop = MarkDrop(dropped_clone);
                started.send(()).unwrap();
                pending::<()>().await;
            })
            .unwrap(),
        );

        ready.await.unwrap();
        drop(group);
        for _ in 0..100 {
            if dropped.load(Ordering::Acquire) {
                return;
            }
            tokio::task::yield_now().await;
        }
        assert!(dropped.load(Ordering::Acquire));
    }
}
