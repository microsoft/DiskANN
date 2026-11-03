/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! How the index runs work that is independent across lists.

#[cfg(not(feature = "rayon"))]
use std::marker::PhantomData;

#[cfg(feature = "rayon")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// How the index runs work that is independent across lists, such as fitting the children
/// of each split list.
///
/// The changes an operation makes do not depend on this choice.
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
    /// Do not call into the index from a worker of `pool` while holding a lock that other
    /// tasks on `pool` may need: waiting for the tasks lets the worker run them, which can
    /// deadlock.
    #[cfg(feature = "rayon")]
    pub const fn rayon(pool: &'a rayon::ThreadPool) -> Self {
        Self { pool: Some(pool) }
    }

    /// Return `f(0)`, ..., `f(tasks - 1)` in task order, or the first error in task order.
    pub(crate) fn try_map<T, E, F>(self, tasks: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync + Send,
    {
        #[cfg(feature = "rayon")]
        if let Some(pool) = self.pool {
            // Every task runs before the errors are checked, so the error returned does not
            // depend on scheduling.
            let results: Vec<Result<T, E>> =
                pool.install(|| (0..tasks).into_par_iter().map(f).collect());
            return results.into_iter().collect();
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

    /// Check that `parallelism` returns results and errors in task order.
    fn check_task_order(parallelism: Parallelism<'_>) {
        let squares = parallelism.try_map(100, |i| Ok::<_, usize>(i * i));
        assert_eq!(squares, Ok((0..100).map(|i| i * i).collect()));

        let none = parallelism.try_map(0, Err::<usize, usize>);
        assert_eq!(none, Ok(Vec::new()));

        // Tasks 37 and 74 fail; the earlier one is reported.
        let failed =
            parallelism.try_map(100, |i| if i % 37 == 0 && i > 0 { Err(i) } else { Ok(i) });
        assert_eq!(failed, Err(37));
    }

    #[test]
    fn sequential_keeps_task_order() {
        check_task_order(Parallelism::sequential());
        check_task_order(Parallelism::default());
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn rayon_keeps_task_order() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        check_task_order(Parallelism::rayon(&pool));
    }
}
