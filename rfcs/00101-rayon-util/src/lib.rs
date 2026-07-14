/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(test)]
mod tests {
    use rayon::prelude::*;
    use rayon::ThreadPoolBuilder;
    use std::sync::Condvar;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    #[ignore = "This test will cause deadlock. Run test manually via cargo test tests::test_deadlock_with_mutex -- --exact  --nocapture --ignored"]
    fn test_deadlock_with_mutex() {
        let mutex = std::sync::Mutex::new(());
        (0..100).into_par_iter().for_each(|_| {
            let _lock = mutex.lock().unwrap();
            rayon::yield_now();
        });
    }
    #[test]
    #[ignore = "This test will cause deadlock. Run test manually via cargo test tests::test_deadlock_with_mutex2 -- --exact  --nocapture --ignored"]
    fn test_deadlock_with_mutex2() {
        let mutex = std::sync::Mutex::new(());
        (0..100).into_par_iter().for_each(|_| {
            let _lock = mutex.lock().unwrap();
            let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
            pool.install(|| {
                println!("Run op inside thread pool");
            });
            println!("Finish one iteration");
        });
    }

    #[test]
    #[ignore = "Run test manually via cargo test tests::test_deadlock_with_mutex1_fix -- --exact  --nocapture --ignored"]
    fn fix_deadlock_with_mutex1_by_mutex_thread() {
        let mutex = std::sync::Mutex::new(());
        (0..100).into_par_iter().for_each(|_| {
            let pool_mutex = ThreadPoolBuilder::default().num_threads(1).build().unwrap();
            pool_mutex.install(|| {
                let _lock = mutex.lock().unwrap();
                rayon::yield_now();
                println!("Finish one iteration");
            });
        });
    }

    #[test]
    #[ignore = "Run test manually via cargo test tests::test_deadlock_with_mutex2_fix -- --exact  --nocapture --ignored"]
    fn fix_deadlock_with_mutex2_by_mutex_thread() {
        let mutex = std::sync::Mutex::new(());
        (0..100).into_par_iter().for_each(|_| {
            let pool_mutex = ThreadPoolBuilder::default().num_threads(1).build().unwrap();
            pool_mutex.install(|| {
                let _lock = mutex.lock().unwrap();
                let pool = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
                pool.install(|| {
                    println!("Run op inside thread pool");
                });
                println!("Finish one iteration");
            });
        });
    }

    pub struct RayonAwareMutex<T> {
        inner: std::sync::Mutex<T>,
    }

    impl<T> RayonAwareMutex<T> {
        pub fn new(data: T) -> Self {
            RayonAwareMutex {
                inner: Mutex::new(data),
            }
        }

        pub fn lock(&self) -> std::sync::MutexGuard<T> {
            loop {
                match self.inner.try_lock() {
                    Ok(guard) => return guard,
                    Err(_) => {
                        thread::sleep(Duration::from_micros(10));
                        rayon::yield_now();
                    }
                }
            }
        }
    }
    #[test]
    #[ignore = "It doesn't fix deadlock.Run test manually via cargo test tests::fix_deadlock_with_mutex_by_rayon_aware_mutex -- --exact  --nocapture --ignored"]
    fn fix_deadlock_with_mutex1_by_rayon_aware_mutex() {
        let mutex = RayonAwareMutex::new(());
        (0..100).into_par_iter().for_each(|_| {
            let _lock = mutex.lock();
            rayon::yield_now();
            println!("Finish one iteration");
        });
    }

    #[test]
    #[ignore = "This test will cause deadlock. Run test manually via cargo test tests::test_deadlock_with_nested_pools -- --exact  --nocapture --ignored"]
    fn test_deadlock_with_nested_pools() {
        // Define the number of available resources
        let max_resources = 10;
        // Create a shared state with a counter and a condition variable
        let state = Arc::new((Mutex::new(max_resources), Condvar::new()));
        let pool_outer = ThreadPoolBuilder::new()
            .num_threads(max_resources)
            .build()
            .unwrap();
        pool_outer.install(|| {
            (0..100).into_par_iter().for_each(|_| {
                let state = Arc::clone(&state);
                let (lock, cvar) = &*state;

                // Acquire a resource
                {
                    let mut resources = lock.lock().unwrap();
                    while *resources == 0 {
                        resources = cvar.wait(resources).unwrap();
                    }
                    *resources -= 1;
                    println!("Acquire resource, remaining resources: {}", *resources);
                }

                // Run parallel operations in a nested thread pool
                let pool_inner = ThreadPoolBuilder::new().num_threads(2).build().unwrap();
                pool_inner.install(|| {
                    println!("Run parallel operations in a nested thread pool");
                    (0..10).into_par_iter().for_each(|_| {
                        thread::sleep(Duration::from_millis(100));
                    });
                });
                println!("Finish one iteration");

                // Release the resource
                {
                    let mut resources = lock.lock().unwrap();
                    *resources += 1;
                    cvar.notify_one();
                    println!("Release resource");
                }
            })
        });
    }
}
