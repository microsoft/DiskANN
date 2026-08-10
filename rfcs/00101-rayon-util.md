# Rayon Util
## Goals of the API
### Problem Statement
DiskANN currently accepts the `num_threads` parameter and implements parallel operations using the [Rayon crate](https://github.com/rayon-rs/rayon). Rayon simplifies converting sequential iterators into parallel ones by calling `par_iter()` instead of `iter()`:
```rust
use rayon::prelude::*;
fn sum_of_squares(input: &[i32]) -> i32 {
    input.par_iter() // <-- just change that!
         .map(|&i| i * i)
         .sum()
}
```

DiskANN typically leverages Rayon’s default global thread pool and sets `RAYON_NUM_THREADS` to control the default thread pool value. However, there are some drawbacks to this approach:
1. **Side Effects on the Entire Program**: Setting the `RAYON_NUM_THREADS` environment variable in DiskANN impacts the entire program. This variable is utilized elsewhere when creating a Rayon thread pool if the number of threads is not explicitly specified.
2. **Initialization Limitation**: The global thread pool is initialized only once. Consequently, once the pool is initiated, changing `RAYON_NUM_THREADS` does not take effect, rendering `num_threads` ineffective.

### Proposed Solution
To address these issues, the API should provide a mechanism for safer and more efficient interaction with Rayon thread pools, while avoiding common pitfalls such as deadlocks.

1. **Isolate Thread Pool Configuration:** Rather than relying on the global `RAYON_NUM_THREADS` variable, the API should allow the creation and management of dedicated thread pools, preventing unintended side effects on other parts of the program.
2. **Local Thread Pool Initialization:** The API should provide the ability to initialize local thread pools with a specified number of threads. This can be achieved using Rayon’s `ThreadPoolBuilder`, ensuring the thread pools are scoped specifically to DiskANN operations.
3. **Deadlock Prevention:**  The risk of deadlocks arises from the combination of `par_*` and Yields (explicitly or implicitly via `pool.install()`), Usage of local thread pools is always combined with `pool.install()`, so one should be careful when installing a local pool inside `par_*`. See the section on [Potential Rayon deadlock issues](#potential-rayon-deadlock-issues) for more details.

# API Proposal

## RayonThreadPool
The `RayonThreadPool` type is a wrapper for `rayon::ThreadPool`, designed for executing Rayon parallel iterators with control over the thread pool configuration. 

```rust
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
```

### Creating a RayonThreadPool
The following functions initialize and manage thread pools with specific configurations.

#### `create_thread_pool`
Creates a new thread pool with the specified number of threads. If `num_threads` is 0, it defaults to the number of logical CPUs.
```rust
pub fn create_thread_pool(num_threads: usize) -> ANNResult<RayonThreadPool> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|err| ANNError::log_thread_pool_error(err.to_string()))?;
    Ok(RayonThreadPool(pool))
}
```

#### `create_thread_pool_for_test`
Creates a thread pool with a configurable number of threads for testing purposes.
The number of threads can be set using the environment variable `DISKANN_TEST_POOL_THREADS`. If the environment variable is not set or cannot be parsed, it defaults to 3 threads.
```rust
pub fn create_thread_pool_for_test() -> RayonThreadPool {
    let num_threads = env::var("DISKANN_TEST_POOL_THREADS")
        .ok()
        .and_then(|val| val.parse().ok())
        .unwrap_or(3);

    create_thread_pool(num_threads).unwrap()
}
```

#### `create_thread_pool_for_bench`
Creates a thread pool for benchmarking purposes without specifying the number of threads. The Rayon runtime will automatically determine the optimal number of threads to use.
It uses the `RAYON_NUM_THREADS` environment variable if set, or defaults to the number of logical CPUs otherwise.
```rust
#[allow(clippy::unwrap_used)]
pub fn create_thread_pool_for_bench() -> RayonThreadPool {
    let pool = rayon::ThreadPoolBuilder::new()
        .build()
        .map_err(|err| ANNError::log_thread_pool_error(err.to_string()))
        .unwrap();
    RayonThreadPool(pool)
}
```
### **Guideline for Parallel Iterators**

> **All Rayon parallel iterators should execute within a specific `RayonThreadPool` instance** to ensure controlled and consistent thread usage. Avoid using the global thread pool for these operations.

```rust
fn run_parallel_task(pool: &RayonThreadPool){
	pool.install(||{
		(0..100).into_par_iter().for_each(||{})
	})
}
```

## Enforcing Rayon parallel iterators be executed within a specified thread Pool
To ensure parallel execution occurs within a specific `RayonThreadPool` and prevent unintentional use of Rayon’s global thread pool, Clippy is configured to disallow the original Rayon methods, encouraging `_in_pool` versions instead.

### Custom `_in_pool` Methods
The `_in_pool` methods wrap common Rayon parallel iterator methods (such as `for_each` and `try_for_each`), enforcing execution within a specified thread pool and avoiding the global pool. This provides explicit control over which pool executes the parallel task.

The `ParallelIteratorInPool` trait extends `ParallelIterator` with custom `_in_pool` methods, ensuring operations execute within a defined thread pool. This includes methods like `for_each_in_pool`, `collect_in_pool`, `try_for_each_in_pool`, and `sum_in_pool`.

The following is the example usage of `for_each_in_pool`:
```rust
pub trait ParallelIteratorInPool: ParallelIterator + Sized {
    fn for_each_in_pool<OP>(self, pool: &RayonThreadPool, op: OP)
    where
        OP: Fn(Self::Item) + Sync + Send,
    {
        pool.install(|| self.for_each(op));
    }
}

impl<T> ParallelIteratorInPool for T where T: ParallelIterator {}
```
Example usage:
```rust
#[test]
fn test_for_each_in_pool() {
    let pool = create_thread_pool(4).unwrap();
    let iter = (0..100).into_par_iter();
    iter.for_each_in_pool(&pool, |i| {
        println!("{}",i);
    });
}
```
### Available `_in_pool` Methods
The following `_in_pool` methods provide pool-specific alternatives to commonly used Rayon methods:

- **`for_each_in_pool`**: Replaces `for_each`.
- **`for_each_with_in_pool`** Replaces `for_each_with`.
- **`for_each_init_in_pool`** Replaces `for_each_init`.
- **`try_for_each_in_pool`**: Replaces `try_for_each`.
- **`try_for_each_with_in_pool`**: Replaces `try_for_each_with`.
- **`try_for_each_init_in_pool`**: Replaces `try_for_each_init`.
- **`count_in_pool`**: Replaces `count`.
- **`collect_in_pool`**: Replaces `collect`.
- **`sum_in_pool`**: Replaces `sum`.

### Enforcing `_in_pool` Methods with Clippy
To standardize the use of `_in_pool` methods, we configure Clippy to disallow traditional Rayon parallel iterator methods, recommending `_in_pool` variants instead.

Add the following configuration to `clippy.toml` file to enforce `_in_pool` method usage:
```toml
# Disallow methods that consume the iterator, enforcing execution in a specified pool
disallowed-methods = [
    { path = "rayon::iter::ParallelIterator::for_each", reason = "Use `for_each_in_pool` instead to enforce execution within a specified thread pool." },
    { path = "rayon::iter::ParallelIterator::try_for_each", reason = "Use `try_for_each_in_pool` instead to enforce execution within a specified thread pool." },
    { path = "rayon::iter::ParallelIterator::collect", reason = "Use `collect_in_pool` or an equivalent method to enforce execution within a specified thread pool." },
    #...
]
```

## Helper function

### `AsThreadPool` trait
This allows either an integer to be provided or an explicit `&RayonThreadPool`. If an integer is provided, we create a new thread-pool with the requested number of threads.

```rust
pub trait AsThreadPool: Send + Sync {
    type Returns: DerefsTo<RayonThreadPool>;

    fn as_threadpool(&self) -> ANNResult<Self::Returns>;
}


impl DerefsTo<RayonThreadPool> for &RayonThreadPool {
    fn deref_to(&self) -> &RayonThreadPool {
        self
    }
}

impl AsThreadPool for usize {
    type Returns = RayonThreadPool;
    fn as_threadpool(&self) -> ANNResult<Self::Returns> {
        create_thread_pool(*self)
    }
}

impl<'a> AsThreadPool for &'a RayonThreadPool {
    type Returns = &'a RayonThreadPool;
    fn as_threadpool(&self) -> ANNResult<Self::Returns> {
        Ok(self)
    }
}
```

# Potential Rayon deadlock issues 
You can find related test cases in `00101-rayon-util/src/lib.rs`.

Rayon can encounter deadlock issues in the following scenarios:
## 1. Using Rayon while holding a Mutex
Invoking `yield_now()` inside Rayon parallel iterators while holding a Mutex can lead to deadlock.
Here's a minimal example:
```rust
fn test_deadlock_with_mutex() {
	let mutex = std::sync::Mutex::new(());
	(0..100).into_par_iter().for_each(|_| {
		let _lock = mutex.lock().unwrap();
		rayon::yield_now();
	});
}
```
Explanation:
1. Thread 1 starts processing calls inside `(0..100).into_par_iter().for_each`
	- Grabs the lock
	- Invokes `yield_now()` to steal other work from the thread pool to execute, based on the work-stealing mechanism in Rayon.
2. Then Thread 1 finds another call inside `(0..100).into_par_iter().for_each`
    - Tries to grab the lock again
    - Since Rust mutex isn't a recursive lock, recursing the lock causes deadlock.
	
Furthermore, `Rayon::ThreadPool.install()` has similar behavior to `yield_now()`, as it can schedule other tasks to run on the current thread. This also leads to deadlock in the following case:

```rust
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
```

Rayon tracks this issue in
[Using rayon under a Mutex can lead to deadlocks · Issue #592 · rayon-rs/rayon (github.com)](https://github.com/rayon-rs/rayon/issues/592)

### Issues in DiskANN and Prevention
While DiskANN hasn't encountered this issue so far, it is **very likely** to occur if Rayon is invoked while holding a Mutex.

To understand the risk, here are the prerequisites for a potential deadlock:
- The code is running inside a parallel iterator.
- A `Mutex` is being acquired.
- A `yield` is invoked explicitly or implicitly (e.g., by calling pool.install()) before releasing the lock. Rayon’s work-stealing mechanism then attempts to recurse and acquire the lock again, causing a deadlock.
To mitigate this risk, we propose the following solutions:

#### 1. Isolate Locks in a Separate Thread Pool
One solution is to isolate the lock within a separate thread pool, which can have as few as one thread since Mutex operations are sequential anyway. This prevents recursive locks from occurring:
```rust
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
```
#### 2. Disable Rayon’s Work-Stealing Strategy
Rayon is working on an option `full_blocking` to disable work-stealing in nested thread pools. You can follow the progress in [Add full_blocking feature to thread pool](https://github.com/rayon-rs/rayon/pull/1175).

When a job on a parent thread pool creates a job in a child thread pool and `full_blocking` is true, the parent job will block until the child job is completed. This is different from the default behavior where the parent thread is allowed to work on other jobs in the parent thread pool while the child job completes.

This behavior is useful for avoiding deadlocks caused by work-stealing and is helpful for instrumentation based profiling in multi-threaded settings.
#### 3. Rayon-Aware Mutex (Did Not Work)
We also experimented with a `RayonAwareMutex` that uses `try_lock()` combined with `yield_now()` to prevent deadlocks. However, this approach did not resolve the issue, possibly due to lack of prioritization or control over the lock acquisition process:

```rust
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

fn fix_deadlock_with_mutex1_by_rayon_aware_mutex() {
    let mutex = RayonAwareMutex::new(());
    (0..100).into_par_iter().for_each(|_| {
        let _lock = mutex.lock();
        rayon::yield_now();
        println!("Finish one iteration");
    });
}
```

## 2. Nested Rayon Thread Pools with Limited Resources
Here is an example:
1. Inside `(0..100).into_par_iter().for_each`,
2. Thread 1 acquires a resource, then calls `pool.install()` to execute parallel operations.
3. `pool_inner.install()` implicitly calls yield, causing the outer thread pool to cooperatively schedule another iteration of the outer loop on the same thread.
4. Thread 1 acquires a resource again.
5. Repeat steps 2-3 until all resources are exhausted.
6. All 10 threads in `pool_outer` are stuck waiting for resources, and no one can pick up the code after `pool_inner.install()` and release the resources.
7. Hence, deadlock.

```rust
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
```

### Issues in DiskANN and prevention
Although DiskANN hasn't faced this issue yet, the`DiskIndexSearcher` has a limited resource `ssd_query_scratch_queue` to provide `SSDQueryScratch` during search operations.
To prevent a similar situation that could lead to a deadlock, we should avoid using parallel operations in search.