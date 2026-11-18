/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{any::Any, num::NonZeroUsize, ops::Range};

use diskann::{ANNResult, utils::async_tools};
use diskann_benchmark_runner::utils::MicroSeconds;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::build::{
    AsProgress, BuildResults, Parallelism, Progress,
    api::{BatchResultsInner, BuildResultsInner, Chunks, ControlBlock},
};

/// The core build API.
///
/// This uses a model where the data over which the index build is stored internally and
/// identified by its index. Data is numbered from `0` to `N - 1` where `N = Build::num_data()`
/// is the total number of data points.
///
/// The trait is used in conjunction with [`build`] and [`build_tracked`]. See the documentation
/// of those methods for more details.
pub trait Build: Sync {
    /// Custom output parameters. This augments the standard metrics collected by [`build`] and
    /// allows implementation-specific data to be returned.
    type Output: Send + 'static;

    /// Return the number of data points to build the index over. The machinery in [`build`] and
    /// [`build_tracked`] will partition the range `0..num_data()` into disjoint ranges and call
    /// [`Build::build`] on each range in an unspecified order.
    fn num_data(&self) -> usize;

    /// Insert the data points specified by the range. Implementations may assume that the range is
    /// non-empty, within `0..num_data()`, and disjoint from other ranges passed to concurrent calls
    /// while in [`build`] or [`build_tracked`].
    ///
    /// Multiple calls may be made in parallel.
    fn build(&self, range: Range<usize>) -> ANNResult<Self::Output>;
}

/// Perform a build operation and return the results.
///
/// See [`build_tracked`] for more details.
pub fn build<B>(builder: &B, parallelism: Parallelism) -> anyhow::Result<BuildResults<B::Output>>
where
    B: Build,
{
    build_tracked(builder, parallelism, None)
}

/// Perform a build operation.
///
/// Work will be performed by creating `ntasks` concurrent tasks in the dynamically scoped
/// [`rayon::ThreadPool`]. These tasks will partition the problem space `0..builder.num_data()`
/// into batches according to the policy in `parallelism`.
///
/// If `as_progress` is provided, it will be used to create a progress reporter.
pub fn build_tracked<B>(
    builder: &B,
    parallelism: Parallelism,
    as_progress: Option<&dyn AsProgress>,
) -> anyhow::Result<BuildResults<B::Output>>
where
    B: Build,
{
    let max = builder.num_data();
    let progress = as_progress.map(|p| p.as_progress(max));
    let results = build_inner(builder, parallelism, progress.as_deref())?;
    Ok(BuildResults::new(results))
}

///////////
// Inner //
///////////

/// An inner build method with no generic parameters to reduce code-generation.
fn build_inner(
    build: &dyn BuildInner,
    parallelism: Parallelism,
    progress: Option<&dyn Progress>,
) -> anyhow::Result<BuildResultsInner> {
    match parallelism {
        Parallelism::Dynamic { batchsize, ntasks } => {
            build_inner_dynamic(build, batchsize, ntasks, progress)
        }
        Parallelism::Sequential { batchsize } => {
            // Sequential is just dynamic with one task. The dynamic load balancer will ensure that batches
            // are processed in order.
            build_inner_dynamic(build, batchsize, diskann::utils::ONE, progress)
        }
        Parallelism::Fixed { batchsize, ntasks } => {
            build_inner_fixed(build, batchsize, ntasks, progress)
        }
    }
}

/// A dyn-compatible version of [`Build`] to reduce monomorphization bloat.
trait BuildInner: Sync {
    fn num_data(&self) -> usize;
    fn build(&self, range: Range<usize>) -> ANNResult<Box<dyn Any + Send>>;
}

impl<T> BuildInner for T
where
    T: Build,
{
    fn num_data(&self) -> usize {
        <T as Build>::num_data(self)
    }

    fn build(&self, range: Range<usize>) -> ANNResult<Box<dyn Any + Send>> {
        <T as Build>::build(self, range).map(|r| -> Box<dyn Any + Send> { Box::new(r) })
    }
}

//---------//
// Dynamic //
//---------//

/// The inner implementation for [`Parallelism::Dynamic`].
#[expect(
    clippy::disallowed_methods,
    reason = "this is scoped behind a threadpool"
)]
fn build_inner_dynamic(
    build: &dyn BuildInner,
    batchsize: NonZeroUsize,
    ntasks: NonZeroUsize,
    progress: Option<&dyn Progress>,
) -> anyhow::Result<BuildResultsInner> {
    let start = std::time::Instant::now();
    let control = ControlBlock::new(build.num_data(), batchsize);
    let task_results = (0..ntasks.get())
        .into_par_iter()
        .map(|_| {
            let mut results = Vec::new();
            while let Some(batch) = control.next() {
                let start = std::time::Instant::now();
                let output = build.build(batch.clone())?;
                let latency: MicroSeconds = start.elapsed().into();

                if let Some(p) = progress {
                    p.progress(batch.len());
                }

                results.push(BatchResultsInner::new(batch, latency, output));
            }
            Ok::<_, anyhow::Error>(results)
        })
        .collect::<anyhow::Result<_>>()?;

    let end_to_end_latency: MicroSeconds = start.elapsed().into();
    if let Some(p) = progress {
        p.finish();
    }

    Ok(BuildResultsInner::new(end_to_end_latency, task_results))
}

//-------//
// Fixed //
//-------//

#[expect(
    clippy::disallowed_methods,
    reason = "this is scoped behind a threadpool"
)]
fn build_inner_fixed(
    build: &dyn BuildInner,
    batchsize: Option<NonZeroUsize>,
    ntasks: NonZeroUsize,
    progress: Option<&dyn Progress>,
) -> anyhow::Result<BuildResultsInner> {
    let start = std::time::Instant::now();
    let task_results = (0..ntasks.get())
        .into_par_iter()
        .map(|t| {
            let range = async_tools::partition(build.num_data(), ntasks, t)?;
            let mut results = Vec::new();
            match batchsize {
                Some(batchsize) => {
                    for batch in Chunks::new(range, batchsize) {
                        let start = std::time::Instant::now();
                        let output = build.build(batch.clone())?;
                        let latency: MicroSeconds = start.elapsed().into();

                        if let Some(p) = progress {
                            p.progress(batch.len());
                        }

                        results.push(BatchResultsInner::new(batch, latency, output));
                    }
                }
                None => {
                    if !range.is_empty() {
                        let start = std::time::Instant::now();
                        let output = build.build(range.clone())?;
                        let latency: MicroSeconds = start.elapsed().into();

                        if let Some(p) = progress {
                            p.progress(range.len());
                        }

                        results.push(BatchResultsInner::new(range, latency, output));
                    }
                }
            }
            Ok::<_, anyhow::Error>(results)
        })
        .collect::<anyhow::Result<_>>()?;

    let end_to_end_latency: MicroSeconds = start.elapsed().into();
    if let Some(p) = progress {
        p.finish();
    }

    Ok(BuildResultsInner::new(end_to_end_latency, task_results))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use crate::build::api::tests::{MockAsProgress, MockBuild, check_results, test_cases};

    impl Build for MockBuild {
        type Output = Range<usize>;

        fn num_data(&self) -> usize {
            MockBuild::num_data(self)
        }

        fn build(&self, range: Range<usize>) -> ANNResult<Self::Output> {
            Ok(range)
        }
    }

    #[test]
    fn test_build() {
        for (num_threads, num_data, parallelism, desc) in test_cases() {
            let pool = crate::rayon::runtime(NonZeroUsize::new(num_threads).unwrap()).unwrap();
            let (ntasks, expected_batches) = parallelism.ntasks_and_nbatches(num_data);

            let builder = MockBuild::new(num_data);
            let mock_as_progress = MockAsProgress::new();

            // Tracked build
            let results = pool
                .install(|| build_tracked(&builder, parallelism, Some(&mock_as_progress)))
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

            check_results(results, num_data, ntasks, expected_batches, desc);

            // Untracked Build
            let results = pool
                .install(|| build(&builder, parallelism))
                .unwrap_or_else(|_| panic!("{desc}: build should succeed"));
            check_results(results, num_data, ntasks, expected_batches, desc);
        }
    }
}
