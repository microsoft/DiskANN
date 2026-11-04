/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(not(feature = "rayon"))]
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// Where to run independent tasks: on the calling thread, or on a Rayon thread pool.
///
/// Results do not depend on this choice: outputs are in task order, and a failed operation
/// returns the error of its first failed task. Tasks must be thread-safe even when they run
/// on the calling thread, so code that works with one choice works with all of them. The
/// default runs every task on the calling thread.
///
/// # Example
///
/// ```
/// use diskann_utils::Parallelism;
///
/// fn norms(rows: &[Vec<f32>], parallelism: Parallelism<'_>) -> Vec<f32> {
///     parallelism.map(rows.len(), |i| rows[i].iter().map(|x| x * x).sum::<f32>().sqrt())
/// }
///
/// assert_eq!(norms(&[vec![3.0, 4.0]], Parallelism::sequential()), [5.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Parallelism<'a> {
    #[cfg(feature = "rayon")]
    pool: Option<&'a rayon::ThreadPool>,
    #[cfg(not(feature = "rayon"))]
    lifetime: PhantomData<&'a ()>,
}

impl<'a> Parallelism<'a> {
    /// Run every task on the calling thread.
    pub const fn sequential() -> Self {
        Self {
            #[cfg(feature = "rayon")]
            pool: None,
            #[cfg(not(feature = "rayon"))]
            lifetime: PhantomData,
        }
    }

    /// Run tasks on `pool`. The calling thread blocks until they finish.
    ///
    /// Do not start an operation from a worker of `pool` while holding a lock: while it
    /// waits, the worker may run other tasks from `pool`, and one that needs the same lock
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

    /// Return `f(0)`, ..., `f(tasks - 1)` in task order.
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

    /// Check that `parallelism` returns results, and the first error, in task order.
    fn check_task_order(parallelism: Parallelism<'_>) {
        let squares: Vec<usize> = (0..100).map(|i| i * i).collect();
        assert_eq!(parallelism.map(100, |i| i * i), squares);
        assert_eq!(
            parallelism.try_map(100, |i| Ok::<_, usize>(i * i)),
            Ok(squares)
        );

        assert!(parallelism.map(0, |i| i).is_empty());
        assert_eq!(parallelism.try_map(0, Err::<usize, usize>), Ok(Vec::new()));

        // Tasks 37 and 74 fail; the earlier one is reported.
        let failed =
            parallelism.try_map(100, |i| if i % 37 == 0 && i > 0 { Err(i) } else { Ok(i) });
        assert_eq!(failed, Err(37));
    }

    /// Check that both operations of `parallelism` run every task where `placed` returns
    /// `true`.
    fn check_placement<P>(parallelism: Parallelism<'_>, placed: P)
    where
        P: Fn() -> bool + Sync + Send,
    {
        assert!(parallelism.map(8, |_| placed()).into_iter().all(|p| p));

        let placed = parallelism.try_map(8, |_| if placed() { Ok(()) } else { Err(()) });
        assert!(placed.is_ok());
    }

    #[test]
    fn sequential_runs_in_order_on_the_calling_thread() {
        let caller = std::thread::current().id();
        for parallelism in [Parallelism::sequential(), Parallelism::default()] {
            check_task_order(parallelism);
            check_placement(parallelism, || std::thread::current().id() == caller);
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_runs_in_order_on_the_pool() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let parallelism = Parallelism::rayon(&pool);

        check_task_order(parallelism);
        check_placement(parallelism, || pool.current_thread_index().is_some());
    }
}
