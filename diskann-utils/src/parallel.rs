/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(not(feature = "rayon"))]
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A struct to run independent tasks, either on the calling thread or on a Rayon thread
/// pool.
///
/// Results do not depend on this choice: outputs are in task order, and a failed operation
/// returns the error of its first failed task.
///
/// ```
/// use diskann_utils::Parallelism;
///
/// fn norms_sq(rows: &[Vec<f32>], parallelism: Parallelism<'_>) -> Vec<f32> {
///     parallelism.map(rows.len(), |i| rows[i].iter().map(|x| x * x).sum::<f32>())
/// }
///
/// let a = vec![vec![3.0, 4.0], vec![5.0, 12.0]];
///
/// assert_eq!(norms_sq(&a, Parallelism::sequential()), [25.0, 169.0]);
///
/// # #[cfg(feature = "rayon")]
/// # {
/// let pool = rayon::ThreadPoolBuilder::new().build().unwrap();
/// assert_eq!(norms_sq(&a, Parallelism::rayon(&pool)), [25.0, 169.0]);
/// # }
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Parallelism<'a> {
    #[cfg(feature = "rayon")]
    pool: Option<&'a rayon::ThreadPool>,
    #[cfg(not(feature = "rayon"))]
    lifetime: PhantomData<&'a ()>,
}

impl<'a> Parallelism<'a> {
    /// Construct a new `Parallelism` that runs every task on the calling thread. This is
    /// the default.
    pub const fn sequential() -> Self {
        Self {
            #[cfg(feature = "rayon")]
            pool: None,
            #[cfg(not(feature = "rayon"))]
            lifetime: PhantomData,
        }
    }

    /// Construct a new `Parallelism` that runs tasks on `pool`. Each operation blocks the
    /// calling thread until its tasks finish.
    ///
    /// # Note
    ///
    /// Do not start an operation from a worker of `pool` while holding a lock. While the
    /// worker waits, it may run other tasks from `pool`, and one that needs the same lock
    /// deadlocks.
    #[cfg(feature = "rayon")]
    pub const fn rayon(pool: &'a rayon::ThreadPool) -> Self {
        Self { pool: Some(pool) }
    }

    /// Return `f(0)`, ..., `f(tasks - 1)` in task order.
    pub fn map<T, F>(self, tasks: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send,
    {
        #[cfg(feature = "rayon")]
        if let Some(pool) = self.pool {
            return pool.install(|| (0..tasks).into_par_iter().map(f).collect());
        }

        (0..tasks).map(f).collect()
    }

    /// Return the outputs of `f(0)`, ..., `f(tasks - 1)` in task order if every task
    /// succeeds.
    ///
    /// # Errors
    ///
    /// Returns the error of the first task, in task order, that fails. Later tasks may
    /// still run.
    pub fn try_map<T, E, F>(self, tasks: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync + Send,
    {
        // On a pool, every task runs before the errors are checked, so the error returned
        // does not depend on scheduling.
        #[cfg(feature = "rayon")]
        if self.pool.is_some() {
            return self.map(tasks, f).into_iter().collect();
        }

        (0..tasks).map(f).collect()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert that `parallelism` returns outputs, and the first error, in task order.
    fn assert_task_order(parallelism: Parallelism<'_>) {
        let squares: Vec<usize> = (0..100).map(|i| i * i).collect();
        assert_eq!(parallelism.map(100, |i| i * i), squares);
        assert_eq!(
            parallelism.try_map(100, |i| Ok::<_, usize>(i * i)),
            Ok(squares)
        );

        // With no tasks, `f` is never called.
        assert!(parallelism.map(0, |i| i).is_empty());
        assert_eq!(parallelism.try_map(0, Err::<usize, usize>), Ok(Vec::new()));

        // Tasks 37 and 74 fail; the earlier one is reported.
        let failed = parallelism.try_map(100, |i| if i == 37 || i == 74 { Err(i) } else { Ok(i) });
        assert_eq!(failed, Err(37));
    }

    /// Assert that both operations run every task on a thread where `expected` returns
    /// `true`.
    fn assert_tasks_run_where<P>(parallelism: Parallelism<'_>, expected: P)
    where
        P: Fn() -> bool + Sync + Send,
    {
        assert_eq!(parallelism.map(8, |_| expected()), [true; 8]);
        assert_eq!(
            parallelism.try_map(8, |_| Ok::<_, ()>(expected())),
            Ok(vec![true; 8])
        );
    }

    #[test]
    fn test_sequential() {
        let parallelism = Parallelism::sequential();
        let caller = std::thread::current().id();

        assert_task_order(parallelism);
        assert_tasks_run_where(parallelism, || std::thread::current().id() == caller);
    }

    #[test]
    #[cfg(all(not(miri), feature = "rayon"))]
    fn test_rayon() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallelism = Parallelism::rayon(&pool);

        assert_task_order(parallelism);
        assert_tasks_run_where(parallelism, || pool.current_thread_index().is_some());
    }
}
