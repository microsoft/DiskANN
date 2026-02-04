/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    any::Any,
    future::Future,
    num::NonZeroUsize,
    ops::Range,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use diskann::{ANNError, ANNResult};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_utils::future::{AsyncFriendly, boxit};

/// The core build API.
///
/// This uses a model where the data over which the index build is stored internally and
/// identified by its index. Data is numbered from `0` to `N - 1` where `N = Build::num_data()`
/// is the total number of data points.
///
/// The trait is used in conjunction with [`build`] and [`build_tracked`]. See the documentation
/// of those methods for more details.
pub trait Build: AsyncFriendly {
    /// Custom output parameters. This augments the standard metrics collected by [`build`] and
    /// allows implementation-specific data to be returned.
    type Output: AsyncFriendly;

    /// Return the number of data points to build the index over. The machinery in [`build`] and
    /// [`build_tracked`] will partition the range `0..num_data()` into disjoint ranges and call
    /// [`Build::build`] on each range in an unspecified order.
    fn num_data(&self) -> usize;

    /// Insert the data points specified by the range. Implementations may assume that the range is
    /// non-empty, within `0..num_data()`, and disjoint from other ranges passed to concurrent calls
    /// while in [`build`] or [`build_tracked`].
    ///
    /// Multiple calls may be made in parallel.
    fn build(&self, range: Range<usize>) -> impl Future<Output = ANNResult<Self::Output>> + Send;
}

/// The results of processing a single batch during build.
///
/// This struct is marked as `#[non_exhaustive]` to allow for future extension.
///
/// See: [`BuildResults`], [`build`] and [`build_tracked`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BatchResult<T> {
    /// The index of the task that executed this batch. This will be in the range `0..ntasks` where
    /// `ntasks` is the number of tasks specified to [`build`].
    pub taskid: usize,

    /// The range of data points processed by this batch.
    pub batch: Range<usize>,

    /// The wall clock time taken to process this batch.
    pub latency: MicroSeconds,

    /// The customized [`Build::Output`] for this batch.
    pub output: T,
}

impl<T> BatchResult<T> {
    /// Return the number of points in the batch associated with this result.
    pub fn batchsize(&self) -> usize {
        self.batch.len()
    }
}

/// Aggregated results for a build operation.
///
/// See: [`build`] and [`build_tracked`].
#[derive(Debug)]
pub struct BuildResults<T> {
    output: Vec<BatchResult<T>>,
    end_to_end_latency: MicroSeconds,
}

impl<T> BuildResults<T> {
    /// Return the total wall-clock time for the entire build operation.
    pub fn end_to_end_latency(&self) -> MicroSeconds {
        self.end_to_end_latency
    }

    /// Return the per-batch results by reference.
    pub fn output(&self) -> &[BatchResult<T>] {
        &self.output
    }

    /// Consume `self` and return the per-batch results by value.
    pub fn take_output(self) -> Vec<BatchResult<T>> {
        self.output
    }
}

impl<T> BuildResults<T>
where
    T: Any,
{
    /// This is a private inner constructor that converts the type-erased `BuildResultsInner` into
    /// a fully typed container.
    ///
    /// This requires that the dynamic type of the boxed [`Any`] outputs in `inner` is `T`.
    fn new(inner: BuildResultsInner) -> Self {
        let BuildResultsInner {
            end_to_end_latency,
            task_results,
        } = inner;
        let mut output = Vec::with_capacity(task_results.iter().map(|t| t.len()).sum());

        task_results
            .into_iter()
            .enumerate()
            .for_each(|(taskid, results)| {
                results.into_iter().for_each(|r| {
                    output.push(BatchResult {
                        taskid,
                        batch: r.batch,
                        latency: r.latency,
                        output: *r
                            .output
                            .downcast::<T>()
                            .expect("incorrect downcast applied"),
                    })
                })
            });

        Self {
            output,
            end_to_end_latency,
        }
    }
}

/// Control the parallel partitioning strategy for [`build`] and [`build_tracked`].
///
/// Many aspects of this enum are `#[non_exhaustive]` to allow for future extension.
/// Users should use the associated constructors instead to create instances.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Parallelism {
    /// Use dynamic load balancing to partition the work into batches of at most `batchsize`.
    /// When the batchsize is 1, the implementation guarantees sequential execution.
    ///
    /// The batches assigned to each task can be assumed to be monotonically increasing.
    ///
    /// See: [`Parallelism::dynamic`].
    #[non_exhaustive]
    Dynamic {
        batchsize: NonZeroUsize,
        ntasks: NonZeroUsize,
    },

    /// Run the build with just a single task. Input data is still batched.
    ///
    /// See: [`Parallelism::sequential`].
    #[non_exhaustive]
    Sequential { batchsize: NonZeroUsize },

    /// Create a fixed parallelism strategy with `ntasks` executors. This strategy
    /// partitions the problem space into roughly `ntasks` balanced contiguous chunks.
    ///
    /// If `batchsize` is `Some`, than each chunk will be further subdivided into at most
    /// `batchsize` sized subchunks which are then provided to [`Build::build`].
    ///
    /// If `batchsize` is `None`, then the entire task partition is supplied in a single call
    /// to [`Build::build`].
    ///
    /// See: [`Parallelism::fixed`].
    #[non_exhaustive]
    Fixed {
        batchsize: Option<NonZeroUsize>,
        ntasks: NonZeroUsize,
    },
}

impl Parallelism {
    /// Create a dynamic parallelism strategy with the specified `batchsize` and `ntasks`.
    ///
    /// Returns [`Self::Dynamic`].
    pub fn dynamic(batchsize: NonZeroUsize, ntasks: NonZeroUsize) -> Self {
        Self::Dynamic { batchsize, ntasks }
    }

    /// Create a fixed parallelism strategy with `ntasks` executors and possible
    /// sub-partitioning into the specified `batchsize`.
    ///
    /// Returns [`Self::Fixed`].
    pub fn fixed(batchsize: Option<NonZeroUsize>, ntasks: NonZeroUsize) -> Self {
        Self::Fixed { batchsize, ntasks }
    }

    /// Create a sequential parallelism strategy with the specified `batchsize`.
    ///
    /// Returns [`Self::Sequential`].
    pub fn sequential(batchsize: NonZeroUsize) -> Self {
        Self::Sequential { batchsize }
    }
}

/// Enable lazy creation of a progress reporter for the long running build operation.
///
/// See: [`Progress`].
pub trait AsProgress {
    /// Construct a progress reporter for an operation consisting of `max` points.
    fn as_progress(&self, max: usize) -> Arc<dyn Progress>;
}

/// A simple progress reporter for long running operations.
pub trait Progress: AsyncFriendly {
    /// Indicate that `handled` points have been processed.
    fn progress(&self, handled: usize);

    /// Indicate that the operation has finished.
    fn finish(&self);
}

/// Perform a build operation and return the results.
///
/// See [`build_tracked`] for more details.
pub fn build<B>(
    builder: Arc<B>,
    parallelism: Parallelism,
    runtime: &tokio::runtime::Runtime,
) -> anyhow::Result<BuildResults<B::Output>>
where
    B: Build,
{
    build_tracked(builder, parallelism, runtime, None)
}

/// Perform a build operation.
///
/// Work will be performed by spawning `ntasks` concurrent tasks in the provided `runtime`.
/// These tasks will partition the problem space `0..builder.num_data()` into batches according
/// to the policy in `parallelism`.
///
/// If `as_progress` is provided, it will be used to create a progress reporter.
pub fn build_tracked<B>(
    builder: Arc<B>,
    parallelism: Parallelism,
    runtime: &tokio::runtime::Runtime,
    as_progress: Option<&dyn AsProgress>,
) -> anyhow::Result<BuildResults<B::Output>>
where
    B: Build,
{
    let max = builder.num_data();
    let results = runtime.block_on(build_inner(
        builder,
        parallelism,
        as_progress.map(|p| p.as_progress(max)),
    ))?;
    Ok(BuildResults::new(results))
}

///////////
// Inner //
///////////

/// An inner build method with no generic parameters to reduce code-generation.
fn build_inner(
    build: Arc<dyn BuildInner>,
    parallelism: Parallelism,
    progress: Option<Arc<dyn Progress>>,
) -> impl Future<Output = anyhow::Result<BuildResultsInner>> + Send {
    match parallelism {
        Parallelism::Dynamic { batchsize, ntasks } => {
            boxit(build_inner_dynamic(build, batchsize, ntasks, progress))
        }
        Parallelism::Sequential { batchsize } => {
            // Sequential is just dynamic with one task. The dynamic load balancer will ensure that batches
            // are processed in order.
            boxit(build_inner_dynamic(
                build,
                batchsize,
                diskann::utils::ONE,
                progress,
            ))
        }
        Parallelism::Fixed { batchsize, ntasks } => {
            boxit(build_inner_fixed(build, batchsize, ntasks, progress))
        }
    }
}

type Pinned<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// A dyn-compatible version of [`Build`] to reduce monomorphization bloat.
trait BuildInner: AsyncFriendly {
    fn num_data(&self) -> usize;

    fn build(&self, range: Range<usize>) -> Pinned<'_, ANNResult<Box<dyn Any + Send>>>;
}

impl<T> BuildInner for T
where
    T: Build,
{
    fn num_data(&self) -> usize {
        <T as Build>::num_data(self)
    }

    fn build(&self, range: Range<usize>) -> Pinned<'_, ANNResult<Box<dyn Any + Send>>> {
        use futures_util::TryFutureExt;

        boxit(<T as Build>::build(self, range).map_ok(|r| -> Box<dyn Any + Send> { Box::new(r) }))
    }
}

/// Type erased inner build results.
#[derive(Debug)]
struct BuildResultsInner {
    end_to_end_latency: MicroSeconds,

    /// This field has an implicit correspondence with the task-id.
    ///
    /// Index `0` corresponds to task `0`, index `1` to task `1` and so on.
    task_results: Vec<Vec<BatchResultsInner>>,
}

#[derive(Debug)]
struct BatchResultsInner {
    batch: Range<usize>,
    latency: MicroSeconds,
    /// Note that this has dynamic type `Build::Output`.
    output: Box<dyn Any + Send>,
}

//---------//
// Dynamic //
//---------//

/// The inner implementation for [`Parallelism::Dynamic`].
async fn build_inner_dynamic(
    build: Arc<dyn BuildInner>,
    batchsize: NonZeroUsize,
    ntasks: NonZeroUsize,
    progress: Option<Arc<dyn Progress>>,
) -> anyhow::Result<BuildResultsInner> {
    let start = std::time::Instant::now();
    let control = ControlBlock::new(build.num_data(), batchsize);
    let handles: Vec<_> = (0..ntasks.get())
        .map(|_| {
            let build_clone = build.clone();
            let control_clone = control.clone();
            let progress_clone = progress.clone();
            tokio::spawn(async move {
                let mut results = Vec::new();
                while let Some(batch) = control_clone.next() {
                    let start = std::time::Instant::now();
                    let output = build_clone.build(batch.clone()).await?;
                    let latency: MicroSeconds = start.elapsed().into();

                    if let Some(p) = progress_clone.as_deref() {
                        p.progress(batch.len());
                    }

                    results.push(BatchResultsInner {
                        batch,
                        latency,
                        output,
                    });
                }
                Ok::<_, ANNError>(results)
            })
        })
        .collect();

    let mut task_results = Vec::with_capacity(ntasks.into());
    for h in handles {
        task_results.push(h.await??);
    }

    let end_to_end_latency: MicroSeconds = start.elapsed().into();
    if let Some(p) = progress.as_deref() {
        p.finish();
    }

    Ok(BuildResultsInner {
        end_to_end_latency,
        task_results,
    })
}

#[derive(Debug, Clone)]
struct ControlBlock(Arc<ControlBlockInner>);

impl ControlBlock {
    fn new(max: usize, batchsize: NonZeroUsize) -> Self {
        Self(Arc::new(ControlBlockInner::new(max, batchsize)))
    }

    fn next(&self) -> Option<Range<usize>> {
        // We need to be careful about overflowing and the potential conflict with multiple
        // threads working with changes.
        //
        // The solution, unfortunately, is to use a compare-exchange loop.
        let mut start = self.0.head.load(Ordering::Relaxed);

        loop {
            let next = start.saturating_add(self.0.batchsize.get()).min(self.0.max);
            if next == start {
                return None;
            }

            match self
                .0
                .head
                .compare_exchange(start, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return Some(start..next),
                Err(current) => {
                    start = current;
                }
            }
        }
    }
}

#[derive(Debug)]
struct ControlBlockInner {
    head: AtomicUsize,
    max: usize,
    batchsize: NonZeroUsize,
}

impl ControlBlockInner {
    fn new(max: usize, batchsize: NonZeroUsize) -> Self {
        Self {
            head: AtomicUsize::new(0),
            max,
            batchsize,
        }
    }
}

//-------//
// Fixed //
//-------//

async fn build_inner_fixed(
    build: Arc<dyn BuildInner>,
    batchsize: Option<NonZeroUsize>,
    ntasks: NonZeroUsize,
    progress: Option<Arc<dyn Progress>>,
) -> anyhow::Result<BuildResultsInner> {
    use diskann::utils::async_tools::PartitionIter;

    let start = std::time::Instant::now();
    let handles: Vec<_> = PartitionIter::new(build.num_data(), ntasks)
        .map(|range| {
            let build_clone = build.clone();
            let progress_clone = progress.clone();
            tokio::spawn(async move {
                let mut results = Vec::new();
                match batchsize {
                    Some(batchsize) => {
                        for batch in Chunks::new(range, batchsize) {
                            let start = std::time::Instant::now();
                            let output = build_clone.build(batch.clone()).await?;
                            let latency: MicroSeconds = start.elapsed().into();

                            if let Some(p) = progress_clone.as_deref() {
                                p.progress(batch.len());
                            }

                            results.push(BatchResultsInner {
                                batch,
                                latency,
                                output,
                            });
                        }
                    }
                    None => {
                        let start = std::time::Instant::now();
                        let output = build_clone.build(range.clone()).await?;
                        let latency: MicroSeconds = start.elapsed().into();

                        if let Some(p) = progress_clone.as_deref() {
                            p.progress(range.len());
                        }

                        results.push(BatchResultsInner {
                            batch: range,
                            latency,
                            output,
                        });
                    }
                }
                Ok::<_, ANNError>(results)
            })
        })
        .collect();

    let mut task_results = Vec::with_capacity(ntasks.into());
    for h in handles {
        task_results.push(h.await??);
    }

    let end_to_end_latency: MicroSeconds = start.elapsed().into();
    if let Some(p) = progress.as_deref() {
        p.finish();
    }

    Ok(BuildResultsInner {
        end_to_end_latency,
        task_results,
    })
}

/// An iterator that partitions a [`Range<usize>`] into equal-sized sub-ranges.
#[derive(Debug, Clone)]
struct Chunks {
    /// The current position in the range.
    current: usize,
    /// The end of the range.
    end: usize,
    /// The size of each chunk (except possibly the last).
    chunk_size: NonZeroUsize,
}

impl Chunks {
    fn new(range: Range<usize>, chunk_size: NonZeroUsize) -> Self {
        Self {
            current: range.start,
            end: range.end,
            chunk_size,
        }
    }
}

impl Iterator for Chunks {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.end {
            return None;
        }

        let start = self.current;
        let end = (start + self.chunk_size.get()).min(self.end);
        self.current = end;

        Some(start..end)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.current >= self.end {
            return (0, Some(0));
        }

        let remaining = self.end - self.current;
        let count = remaining.div_ceil(self.chunk_size.get());
        (count, Some(count))
    }
}

impl ExactSizeIterator for Chunks {}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::AtomicBool;

    /////////////////////////////////
    // BatchResult / BuildResults //
    /////////////////////////////////

    #[test]
    fn test_batch_result_batchsize() {
        let result = BatchResult {
            taskid: 0,
            batch: 10..25,
            latency: MicroSeconds::new(1000),
            output: "test",
        };
        assert_eq!(result.batchsize(), 15);

        let empty_result = BatchResult {
            taskid: 1,
            batch: 5..5,
            latency: MicroSeconds::new(0),
            output: 42,
        };
        assert_eq!(empty_result.batchsize(), 0);
    }

    #[test]
    fn test_build_results_accessors() {
        let batch1 = BatchResult {
            taskid: 0,
            batch: 0..10,
            latency: MicroSeconds::new(100),
            output: "first",
        };
        let batch2 = BatchResult {
            taskid: 1,
            batch: 10..20,
            latency: MicroSeconds::new(200),
            output: "second",
        };

        let results = BuildResults {
            output: vec![batch1, batch2],
            end_to_end_latency: MicroSeconds::new(500),
        };

        assert_eq!(results.end_to_end_latency(), MicroSeconds::new(500));
        assert_eq!(results.output().len(), 2);
        assert_eq!(results.output()[0].output, "first");
        assert_eq!(results.output()[1].output, "second");

        let output = results.take_output();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].output, "first");
        assert_eq!(output[1].output, "second");
    }

    ///////////////////
    // Control Block //
    ///////////////////

    fn sort_ranges(x: &Range<usize>, y: &Range<usize>) -> std::cmp::Ordering {
        x.start.cmp(&y.start)
    }

    fn check_ranges(x: &mut [Range<usize>], total: usize) {
        x.sort_by(sort_ranges);
        let mut expected_start = 0;
        for r in x {
            assert_eq!(r.start, expected_start);
            expected_start = r.end;
        }
        assert_eq!(expected_start, total);
    }

    /// Helper to collect all ranges from a ControlBlock.
    fn collect_all_ranges(control: &ControlBlock) -> Vec<Range<usize>> {
        let mut ranges = Vec::new();
        while let Some(range) = control.next() {
            ranges.push(range);
        }
        ranges
    }

    #[test]
    fn test_control_block() {
        // (max, batchsize, description)
        let test_cases: &[(usize, usize, &str)] = &[
            (10, 3, "not evenly divisible"),
            (9, 3, "exact multiple of batchsize"),
            (0, 5, "empty range"),
            (1, 1, "single element"),
            (3, 10, "batchsize larger than max"),
            (5, 5, "batchsize equals max"),
            (5, 1, "batchsize one (sequential)"),
            (10000, 128, "larger range"),
            (usize::MAX, usize::MAX / 2 - 1, "very large numbers"),
        ];

        for &(max, batchsize, desc) in test_cases {
            let control = ControlBlock::new(max, NonZeroUsize::new(batchsize).unwrap());
            let mut ranges = collect_all_ranges(&control);
            let expected_num_ranges = max.div_ceil(batchsize);

            assert_eq!(
                ranges.len(),
                expected_num_ranges,
                "{desc}: max={max}, batchsize={batchsize}: expected {expected_num_ranges} ranges, got {}",
                ranges.len()
            );
            check_ranges(&mut ranges, max);
            for _ in 1..3 {
                assert!(control.next().is_none(), "{desc}: expected no more ranges");
            }
        }
    }

    #[test]
    fn concurrent_access_yields_disjoint_complete_ranges() {
        let max = 10000;
        let control = ControlBlock::new(max, NonZeroUsize::new(7).unwrap());
        let num_threads = 4;

        let barrier = std::sync::Barrier::new(num_threads);
        let mut all_ranges = std::thread::scope(|s| {
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    s.spawn(|| {
                        barrier.wait();
                        collect_all_ranges(&control.clone())
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });

        check_ranges(&mut all_ranges, max);
    }

    ////////////
    // Chunks //
    ////////////

    #[test]
    fn test_chunks_basic() {
        // Basic cases: (range, chunk_size, expected_chunks)
        #[expect(
            clippy::single_range_in_vec_init,
            reason = "these are test cases - sometimes we do need an array of a simgle element range"
        )]
        let test_cases: &[(_, _, &[_])] = &[
            // Evenly divisible
            (0..9, 3, &[0..3, 3..6, 6..9]),
            // Not evenly divisible - last chunk is smaller
            (0..10, 3, &[0..3, 3..6, 6..9, 9..10]),
            // Chunk size equals range length
            (0..5, 5, &[0..5]),
            // Chunk size larger than range length
            (0..3, 10, &[0..3]),
            // Single element
            (0..1, 1, &[0..1]),
            // Single element with larger chunk size
            (0..1, 5, &[0..1]),
            // Empty range
            (0..0, 3, &[]),
            // Non-zero start
            (5..15, 3, &[5..8, 8..11, 11..14, 14..15]),
            // Non-zero start, evenly divisible
            (10..16, 2, &[10..12, 12..14, 14..16]),
        ];

        for (range, chunk_size, expected) in test_cases {
            let chunks: Vec<_> = Chunks::new(range.clone(), nz(*chunk_size)).collect();
            assert_eq!(
                &chunks, expected,
                "Chunks::new({:?}, {}) produced {:?}, expected {:?}",
                range, chunk_size, chunks, expected
            );
        }
    }

    #[test]
    fn test_chunks_size_hint() {
        // Test that size_hint is accurate
        let mut chunks = Chunks::new(0..10, nz(3));

        assert_eq!(chunks.size_hint(), (4, Some(4)));
        assert_eq!(chunks.len(), 4);

        chunks.next(); // consume 0..3
        assert_eq!(chunks.size_hint(), (3, Some(3)));
        assert_eq!(chunks.len(), 3);

        chunks.next(); // consume 3..6
        assert_eq!(chunks.size_hint(), (2, Some(2)));

        chunks.next(); // consume 6..9
        assert_eq!(chunks.size_hint(), (1, Some(1)));

        chunks.next(); // consume 9..10
        assert_eq!(chunks.size_hint(), (0, Some(0)));
        assert_eq!(chunks.len(), 0);

        // After exhaustion
        assert!(chunks.next().is_none());
        assert_eq!(chunks.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_chunks_empty_range() {
        let chunks: Vec<_> = Chunks::new(0..0, nz(5)).collect();
        assert!(chunks.is_empty());

        let chunks: Vec<_> = Chunks::new(10..10, nz(3)).collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunks_covers_entire_range() {
        // Verify that chunks cover the entire range without gaps or overlaps
        let test_cases: &[(Range<usize>, usize)] = &[
            (0..100, 7),
            (0..1000, 13),
            (50..150, 11),
            (0..1, 1),
            (0..17, 17),
            (0..17, 18),
        ];

        for (range, chunk_size) in test_cases {
            let chunks: Vec<_> = Chunks::new(range.clone(), nz(*chunk_size)).collect();

            // Verify no gaps and no overlaps
            let mut expected_start = range.start;
            for chunk in &chunks {
                assert_eq!(
                    chunk.start, expected_start,
                    "Gap detected at {} (expected {})",
                    chunk.start, expected_start
                );
                assert!(chunk.end > chunk.start, "Empty chunk detected: {:?}", chunk);
                expected_start = chunk.end;
            }
            assert_eq!(expected_start, range.end, "Chunks don't cover entire range");

            // Verify chunk sizes
            for (i, chunk) in chunks.iter().enumerate() {
                if i < chunks.len() - 1 {
                    assert_eq!(chunk.len(), *chunk_size, "Non-final chunk has wrong size");
                } else {
                    assert!(
                        chunk.len() <= *chunk_size,
                        "Final chunk is larger than chunk_size"
                    );
                }
            }
        }
    }

    #[test]
    fn test_chunks_large_range() {
        // Test with a large range to ensure no overflow issues
        let range = 0..1_000_000;
        let chunk_size = 1000;
        let chunks: Vec<_> = Chunks::new(range.clone(), nz(chunk_size)).collect();

        assert_eq!(chunks.len(), 1000);
        assert_eq!(chunks.first(), Some(&(0..1000)));
        assert_eq!(chunks.last(), Some(&(999_000..1_000_000)));
    }

    ///////////////////////////
    // Build / Build Tracked //
    ///////////////////////////

    /// Helper to construct a `NonZeroUsize` from a `usize` in tests.
    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    /// A mock implementation of [`Build`] that returns the range it was called with.
    struct MockBuild {
        num_data: usize,
    }

    impl MockBuild {
        fn new(num_data: usize) -> Self {
            Self { num_data }
        }
    }

    impl Build for MockBuild {
        type Output = Range<usize>;

        fn num_data(&self) -> usize {
            self.num_data
        }

        async fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
            Ok(range)
        }
    }

    /// A mock implementation of [`Progress`] that tracks calls.
    struct MockProgress {
        total_handled: AtomicUsize,
        finish_called: AtomicBool,
    }

    impl MockProgress {
        fn new() -> Self {
            Self {
                total_handled: AtomicUsize::new(0),
                finish_called: AtomicBool::new(false),
            }
        }

        fn total_handled(&self) -> usize {
            self.total_handled.load(Ordering::Relaxed)
        }

        fn was_finished(&self) -> bool {
            self.finish_called.load(Ordering::Relaxed)
        }
    }

    impl Progress for MockProgress {
        fn progress(&self, handled: usize) {
            self.total_handled.fetch_add(handled, Ordering::Relaxed);
        }

        fn finish(&self) {
            self.finish_called.store(true, Ordering::Relaxed);
        }
    }

    /// A mock implementation of [`AsProgress`] that creates a [`MockProgress`].
    struct MockAsProgress {
        progress: Arc<MockProgress>,
        expected_max: AtomicUsize,
    }

    impl MockAsProgress {
        fn new() -> Self {
            Self {
                progress: Arc::new(MockProgress::new()),
                expected_max: AtomicUsize::new(0),
            }
        }

        fn progress(&self) -> &Arc<MockProgress> {
            &self.progress
        }

        fn received_max(&self) -> usize {
            self.expected_max.load(Ordering::Relaxed)
        }
    }

    impl AsProgress for MockAsProgress {
        fn as_progress(&self, max: usize) -> Arc<dyn Progress> {
            self.expected_max.store(max, Ordering::Relaxed);
            self.progress.clone()
        }
    }

    #[test]
    fn test_build() {
        // (num_threads, num_data, parallelism, description)
        let test_cases: &[(usize, usize, Parallelism, &str)] = &[
            (
                4,
                100,
                Parallelism::dynamic(nz(10), nz(4)),
                "basic multi-task",
            ),
            (1, 50, Parallelism::dynamic(nz(10), nz(1)), "single task"),
            (4, 0, Parallelism::dynamic(nz(10), nz(4)), "empty data"),
            (
                4,
                5,
                Parallelism::dynamic(nz(100), nz(4)),
                "batchsize larger than data",
            ),
            (2, 20, Parallelism::dynamic(nz(5), nz(2)), "small dataset"),
            (
                8,
                1000,
                Parallelism::dynamic(nz(7), nz(8)),
                "larger dataset with odd batchsize",
            ),
            (
                4,
                100,
                Parallelism::dynamic(nz(10), nz(1)),
                "multiple threads but single task",
            ),
            (
                2,
                50,
                Parallelism::sequential(nz(10)),
                "sequential execution",
            ),
            // Fixed parallelism test cases
            (
                4,
                100,
                Parallelism::fixed(Some(nz(10)), nz(4)),
                "fixed with batchsize",
            ),
            (
                4,
                100,
                Parallelism::fixed(None, nz(4)),
                "fixed without batchsize (whole partition per task)",
            ),
            (
                2,
                50,
                Parallelism::fixed(Some(nz(5)), nz(2)),
                "fixed with small batchsize",
            ),
            (
                8,
                1000,
                Parallelism::fixed(Some(nz(100)), nz(8)),
                "fixed larger dataset",
            ),
            (
                4,
                0,
                Parallelism::fixed(Some(nz(10)), nz(4)),
                "fixed empty data",
            ),
            (
                4,
                5,
                Parallelism::fixed(Some(nz(100)), nz(4)),
                "fixed batchsize larger than partition",
            ),
            (
                1,
                50,
                Parallelism::fixed(Some(nz(10)), nz(1)),
                "fixed single task with batchsize",
            ),
            (
                1,
                50,
                Parallelism::fixed(None, nz(1)),
                "fixed single task without batchsize",
            ),
            (
                4,
                7,
                Parallelism::fixed(Some(nz(2)), nz(4)),
                "fixed uneven partition with batchsize",
            ),
        ];

        for (num_threads, num_data, parallelism, desc) in test_cases {
            let num_data = *num_data;
            let runtime = crate::tokio::runtime(*num_threads).unwrap();

            let (ntasks, expected_batches) = match parallelism {
                Parallelism::Dynamic { batchsize, ntasks } => {
                    let expected = num_data.div_ceil(batchsize.get());
                    (*ntasks, expected)
                }
                Parallelism::Sequential { batchsize } => {
                    let expected = num_data.div_ceil(batchsize.get());
                    (nz(1), expected)
                }
                Parallelism::Fixed { batchsize, ntasks } => {
                    // For Fixed, data is first partitioned among tasks, then each partition is batched.
                    // We need to calculate how many batches each task produces.
                    use diskann::utils::async_tools::PartitionIter;
                    let expected: usize = PartitionIter::new(num_data, *ntasks)
                        .map(|partition| match batchsize {
                            Some(bs) => partition.len().div_ceil(bs.get()),
                            None => {
                                if partition.is_empty() {
                                    0
                                } else {
                                    1
                                }
                            }
                        })
                        .sum();
                    (*ntasks, expected)
                }
            };

            let builder = Arc::new(MockBuild::new(num_data));
            let mock_as_progress = MockAsProgress::new();

            let check_results = |results: BuildResults<Range<usize>>| {
                if num_data == 0 {
                    assert!(
                        results.output().is_empty(),
                        "{desc}: no batches for empty data"
                    );
                    return;
                }

                // Verify that each BatchResult's output matches its batch range.
                for batch_result in results.output() {
                    assert_eq!(
                        batch_result.output, batch_result.batch,
                        "{desc}: output range should match batch range"
                    );
                    assert!(
                        batch_result.taskid < ntasks.get(),
                        "{desc}: taskid {} should be less than ntasks {}",
                        batch_result.taskid,
                        ntasks.get()
                    );
                }

                assert_eq!(
                    results.output().len(),
                    expected_batches,
                    "{desc}: expected {expected_batches} batches, got {}",
                    results.output().len()
                );

                // Verify all data points are covered exactly once.
                let mut ranges: Vec<_> = results.output().iter().map(|r| r.batch.clone()).collect();
                check_ranges(&mut ranges, num_data);
            };

            // Tracked build
            let results = build_tracked(
                builder.clone(),
                *parallelism,
                &runtime,
                Some(&mock_as_progress),
            )
            .unwrap_or_else(|_| panic!("{desc}: build_tracked should succeed"));

            // Verify progress tracking.
            assert_eq!(
                mock_as_progress.received_max(),
                num_data,
                "{desc}: as_progress should receive num_data as max"
            );
            assert_eq!(
                mock_as_progress.progress().total_handled(),
                num_data,
                "{desc}: total progress should equal num_data"
            );
            assert!(
                mock_as_progress.progress().was_finished(),
                "{desc}: finish should be called"
            );

            check_results(results);

            // Untracked Build
            let results = build(builder, *parallelism, &runtime)
                .unwrap_or_else(|_| panic!("{desc}: build should succeed"));
            check_results(results);
        }
    }
}
