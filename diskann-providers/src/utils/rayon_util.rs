/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::{ANNError, ANNResult};
use rayon::prelude::ParallelIterator;

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

    use rayon::prelude::IntoParallelIterator;

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
