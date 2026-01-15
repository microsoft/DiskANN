/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::ops::Range;

use diskann::{ANNError, ANNResult};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

/// based on thread_num, execute the task in parallel using Rayon or serial
#[inline]
pub fn execute_with_rayon<F>(range: Range<usize>, num_threads: usize, f: F) -> ANNResult<()>
where
    F: Fn(usize) -> ANNResult<()> + Sync + Send + Copy,
{
    if num_threads == 1 {
        for i in range {
            f(i)?;
        }
        Ok(())
    } else {
        let pool = create_thread_pool(num_threads)?;
        range.into_par_iter().try_for_each_in_pool(&pool, f)
    }
}

/// Creates a new thread pool with the specified number of threads.
/// If `num_threads` is 0, it defaults to the number of logical CPUs.
pub fn create_thread_pool(num_threads: usize) -> ANNResult<RayonThreadPool> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|err| ANNError::log_thread_pool_error(err.to_string()))?;
    Ok(RayonThreadPool(pool))
}

/// Creates a thread pool with a configurable number of threads for testing purposes.
/// The number of threads can be set using the environment variable `DISKANN_TEST_POOL_THREADS`.
/// If the environment variable is not set or cannot be parsed, it defaults to 3 threads.
#[allow(clippy::unwrap_used)]
pub fn create_thread_pool_for_test() -> RayonThreadPool {
    use std::env;

    let num_threads = env::var("DISKANN_TEST_POOL_THREADS")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(3);

    create_thread_pool(num_threads).unwrap()
}
/// Creates a thread pool for benchmarking purposes without specifying the number of threads.
/// The Rayon runtime will automatically determine the optimal number of threads to use.
/// It uses the `RAYON_NUM_THREADS` environment variable if set,
/// or defaults to the number of logical CPUs otherwise
#[allow(clippy::unwrap_used)]
pub fn create_thread_pool_for_bench() -> RayonThreadPool {
    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .map_err(|err| ANNError::log_thread_pool_error(err.to_string()))
        .unwrap();
    RayonThreadPool(pool)
}

pub struct RayonThreadPool(rayon::ThreadPool);

impl RayonThreadPool {
    pub fn install<OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce() -> R + Send,
        R: Send,
    {
        self.0.install(op)
    }
}

mod sealed {
    pub trait Sealed {}
}

/// This allows either an integer to be provided or an explicit `&RayonThreadPool`.
/// If an integer is provided, we create a new thread-pool with the requested number of
/// threads.
///
/// This trait should be "sealed" to avoid external users being able to implement it.
/// See [as_threadpool_tests] for examples of how to use this trait.
pub trait AsThreadPool: sealed::Sealed + Send + Sync {
    type Returns: std::ops::Deref<Target = RayonThreadPool>;
    fn as_threadpool(&self) -> ANNResult<Self::Returns>;
}

impl sealed::Sealed for usize {}
impl sealed::Sealed for &RayonThreadPool {}

impl AsThreadPool for usize {
    type Returns = diskann_utils::reborrow::Place<RayonThreadPool>;
    fn as_threadpool(&self) -> ANNResult<Self::Returns> {
        create_thread_pool(*self).map(diskann_utils::reborrow::Place)
    }
}

impl<'a> AsThreadPool for &'a RayonThreadPool {
    type Returns = &'a RayonThreadPool;
    fn as_threadpool(&self) -> ANNResult<Self::Returns> {
        Ok(self)
    }
}

/// The `forward_threadpool` macro simplifies obtaining a thread pool from an input
/// that implements the `AsThreadPool` trait.
#[macro_export]
macro_rules! forward_threadpool {
    ($out:ident = $in:ident) => {
        $crate::forward_threadpool!($out = $in: _);
    };
    ($out:ident = $in:ident: $type:ty) => {
        let $out = &*<$type as $crate::utils::AsThreadPool>::as_threadpool(&$in)?;
    };
}

// Allow use of disallowed methods within this trait to provide custom
// implementations of common parallel operations that enforce execution
// within a specified thread pool.
#[allow(clippy::disallowed_methods)]
pub trait ParallelIteratorInPool: ParallelIterator + Sized {
    fn for_each_in_pool<OP>(self, pool: &RayonThreadPool, op: OP)
    where
        OP: Fn(Self::Item) + Sync + Send,
    {
        pool.install(|| self.for_each(op));
    }

    fn for_each_with_in_pool<OP, T>(self, pool: &RayonThreadPool, init: T, op: OP)
    where
        OP: Fn(&mut T, Self::Item) + Sync + Send,
        T: Send + Clone,
    {
        pool.install(|| self.for_each_with(init, op))
    }

    fn for_each_init_in_pool<OP, INIT, T>(self, pool: &RayonThreadPool, init: INIT, op: OP)
    where
        OP: Fn(&mut T, Self::Item) + Sync + Send,
        INIT: Fn() -> T + Sync + Send,
    {
        pool.install(|| self.for_each_init(init, op))
    }

    fn try_for_each_in_pool<OP, E>(self, pool: &RayonThreadPool, op: OP) -> Result<(), E>
    where
        OP: Fn(Self::Item) -> Result<(), E> + Sync + Send,
        E: Send,
    {
        pool.install(|| self.try_for_each(op))
    }

    fn try_for_each_with_in_pool<OP, T, E>(
        self,
        pool: &RayonThreadPool,
        init: T,
        op: OP,
    ) -> Result<(), E>
    where
        OP: Fn(&mut T, Self::Item) -> Result<(), E> + Sync + Send,
        E: Send,
        T: Send + Clone,
    {
        pool.install(|| self.try_for_each_with(init, op))
    }

    fn try_for_each_init_in_pool<OP, INIT, T, E>(
        self,
        pool: &RayonThreadPool,
        init: INIT,
        op: OP,
    ) -> Result<(), E>
    where
        OP: Fn(&mut T, Self::Item) -> Result<(), E> + Sync + Send,
        INIT: Fn() -> T + Sync + Send,
        E: Send,
    {
        pool.install(|| self.try_for_each_init(init, op))
    }

    fn count_in_pool(self, pool: &RayonThreadPool) -> usize {
        pool.install(|| self.count())
    }

    fn collect_in_pool<C>(self, pool: &RayonThreadPool) -> C
    where
        C: rayon::iter::FromParallelIterator<Self::Item> + Send,
    {
        pool.install(|| self.collect())
    }

    fn sum_in_pool<S>(self, pool: &RayonThreadPool) -> S
    where
        S: Send + std::iter::Sum<Self::Item> + std::iter::Sum<S>,
    {
        pool.install(|| self.sum())
    }
}

// Implement the `ParallelIteratorInPool` trait for any type that implements `ParallelIterator`.
impl<T> ParallelIteratorInPool for T where T: ParallelIterator {}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, mpsc::channel};

    use super::*;

    fn get_num_cpus() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap()
    }

    #[test]
    fn test_create_thread_pool_for_test_default() {
        // Ensure the environment variable is not set
        //
        // SAFETY: These environment variables are only set and removed using `std::env`
        // functions (probably).
        unsafe { std::env::remove_var("DISKANN_TEST_POOL_THREADS") };
        let pool = create_thread_pool_for_test();
        // Assuming RayonThreadPool has a method to get the number of threads
        assert_eq!(pool.0.current_num_threads(), 3);
    }

    #[test]
    fn test_create_thread_pool_for_test_from_env() {
        // Set the environment variable to a specific value
        //
        // SAFETY: These environment variables are only set and removed using `std::env`
        // functions (probably).
        unsafe { std::env::set_var("DISKANN_TEST_POOL_THREADS", "5") };
        let pool = create_thread_pool_for_test();
        // Assuming RayonThreadPool has a method to get the number of threads
        assert_eq!(pool.0.current_num_threads(), 5);

        // Clean up the environment variable
        //
        // SAFETY: These environment variables are only set and removed using `std::env`
        // functions (probably).
        unsafe { std::env::remove_var("DISKANN_TEST_POOL_THREADS") };
    }

    #[test]
    fn test_create_thread_pool_for_test_invalid_env() {
        // Set the environment variable to an invalid value
        //
        // SAFETY: These environment variables are only set and removed using `std::env`
        // functions (probably).
        unsafe { std::env::set_var("DISKANN_TEST_POOL_THREADS", "invalid") };
        let pool = create_thread_pool_for_test();
        // Assuming RayonThreadPool has a method to get the number of threads
        assert_eq!(pool.0.current_num_threads(), 3);

        // Clean up the environment variable
        //
        // SAFETY: These environment variables are only set and removed using `std::env`
        // functions (probably).
        unsafe { std::env::remove_var("DISKANN_TEST_POOL_THREADS") };
    }

    #[test]
    fn test_create_thread_pool_for_bench() {
        let pool = create_thread_pool_for_bench();
        assert_eq!(pool.0.current_num_threads(), get_num_cpus());
    }

    fn assert_run_in_rayon_thread() {
        println!(
            "Thread name: {:?}, Thread id: {:?}, Rayon thread index: {:?}, Rayon num_threads: {:?}",
            std::thread::current().name(),
            std::thread::current().id(),
            rayon::current_thread_index(),
            rayon::current_num_threads()
        );
        assert!(rayon::current_thread_index().is_some());
    }

    #[test]
    fn test_for_each_in_pool() {
        let pool = create_thread_pool(4).unwrap();

        let res = Mutex::new(Vec::new());
        (0..5).into_par_iter().for_each_in_pool(&pool, |x| {
            let mut res = res.lock().unwrap();
            res.push(x);
            assert_run_in_rayon_thread();
        });

        let mut res = res.lock().unwrap();
        res.sort();

        assert_eq!(&res[..], &[0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_for_each_with_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let (sender, receiver) = channel();

        (0..5)
            .into_par_iter()
            .for_each_with_in_pool(&pool, sender, |s, x| s.send(x).unwrap());

        let mut res: Vec<_> = receiver.iter().collect();

        res.sort();

        assert_eq!(&res[..], &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_for_each_init_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        iter.for_each_init_in_pool(
            &pool,
            || 0,
            |s, i| {
                assert_run_in_rayon_thread();
                *s += i;
            },
        );
    }

    #[test]
    fn test_map_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let mapped_iter = iter.map(|i| {
            assert_run_in_rayon_thread();
            i as f32
        });
        let list = mapped_iter.collect_in_pool::<Vec<f32>>(&pool);
        assert!(list.len() == 100);
    }

    #[test]
    fn test_try_for_each_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let result = iter.try_for_each_in_pool(&pool, |i| {
            assert_run_in_rayon_thread();
            if i < 50 { Ok(()) } else { Err("Error") }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_try_for_each_init_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let result = iter.try_for_each_init_in_pool(
            &pool,
            || 0,
            |_, i| {
                assert_run_in_rayon_thread();
                if i < 50 { Ok(()) } else { Err("Error") }
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_try_for_each_with_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let result = iter.try_for_each_with_in_pool(&pool, 0, |acc, i| {
            assert_run_in_rayon_thread();
            if i < 50 {
                *acc += i;
                Ok(())
            } else {
                Err("Error")
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_count_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let count = iter.count_in_pool(&pool);
        assert_eq!(count, 100);
    }

    #[test]
    fn test_collect_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let vec = iter.collect_in_pool::<Vec<_>>(&pool);
        assert_eq!(vec.len(), 100);
    }

    #[test]
    fn test_sum_in_pool() {
        let pool = create_thread_pool(4).unwrap();
        let iter = (0..100).into_par_iter();
        let sum: i32 = iter.sum_in_pool(&pool);
        assert_eq!(sum, (0..100).sum::<i32>());
    }
}

#[cfg(test)]
mod as_threadpool_tests {
    use super::*;

    fn some_parallel_op<P: AsThreadPool>(pool: P) -> ANNResult<f32> {
        forward_threadpool!(pool = pool);

        let ret = (0..100).into_par_iter().map(|i| i as f32).sum_in_pool(pool);
        Ok(ret)
    }

    fn another_parallel_op<P: AsThreadPool>(pool: P) -> ANNResult<f32> {
        forward_threadpool!(pool = pool);
        let ret = (0..100).into_par_iter().map(|i| i as f32).sum_in_pool(pool);
        Ok(ret)
    }

    fn execute_single_parallel_op<P: AsThreadPool>(pool: P) -> ANNResult<f32> {
        // Directly pass the thread pool to the function.
        some_parallel_op(pool)
    }

    fn execute_two_parallel_ops<P: AsThreadPool>(pool: P) -> ANNResult<f32> {
        // Need a reference to the thread pool to share it with multiple functions.
        forward_threadpool!(pool = pool);

        let ret1 = some_parallel_op(pool)?;
        let ret2 = another_parallel_op(pool)?;
        Ok(ret1 + ret2)
    }

    fn execute_combined_parallel_ops<P: AsThreadPool>(pool: P) -> ANNResult<f32> {
        // Need a Threadpool reference to execute the operations.
        forward_threadpool!(pool = pool);

        let ret1: f32 = (0..100).into_par_iter().map(|i| i as f32).sum_in_pool(pool);
        let ret2 = some_parallel_op(pool)?;
        Ok(ret1 + ret2)
    }

    #[test]
    fn test_execute_single_parallel_op_with_usize() {
        let num_threads = 4;
        let result = execute_single_parallel_op(num_threads);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_execute_single_parallel_op_with_existing_pool() {
        let pool = create_thread_pool(4).unwrap();
        let result = execute_single_parallel_op(&pool);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_execute_two_parallel_ops_with_usize() {
        let num_threads = 4;
        let result = execute_two_parallel_ops(num_threads);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_execute_two_parallel_ops_with_existing_pool() {
        let pool = create_thread_pool(4).unwrap();
        let result = execute_two_parallel_ops(&pool);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_execute_combined_parallel_ops_with_usize() {
        let num_threads = 4;
        let result = execute_combined_parallel_ops(num_threads);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }

    #[test]
    fn test_execute_combined_parallel_ops_with_existing_pool() {
        let pool = create_thread_pool(4).unwrap();
        let result = execute_combined_parallel_ops(&pool);
        assert!(result.is_ok());
        assert!(result.unwrap() > 0.0);
    }
}
