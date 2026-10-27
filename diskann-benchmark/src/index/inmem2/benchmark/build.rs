/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Use [`Config`] to create a valid [`diskann::graph::Config`], then [`Config::engine`] to
//! create a [`diskann::graph::DiskANNIndex`] and [`tokio::runtime::Runtime`].
//!
//! Finally, [`Engine::build`] or [`Engine::build_for`] are used to run the actual index
//! construction.
//!
//! Results are collected as [`Stats`].

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core::{self as dbc};
use diskann_benchmark_runner::{
    self as dbr,
    report::{
        tolerance::{FmtSpec, MaxRelativeIncrease},
        Report,
    },
    utils::{percentiles, MicroSeconds},
};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{self as dv};
use serde::{Deserialize, Serialize};

use crate::{
    index::inmem2::{Index, Provider},
    utils::{nz, progress::Progress},
};

pub(in crate::index::inmem2) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct Config {
        pub(super) pruned_degree: usize,
        pub(super) max_degree: usize,
        pub(super) l_build: usize,
        pub(super) alpha: f32,
        pub(super) num_threads: NonZeroUsize,
    }

    impl Config {
        pub(in crate::index::inmem2) fn example() -> Self {
            nz!(NUM_THREADS = 8);
            Self {
                pruned_degree: 28,
                max_degree: 32,
                l_build: 100,
                alpha: 1.2,
                num_threads: NUM_THREADS,
            }
        }
    }

    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct Tolerance {
        /// This is the maximum amount of slow-down accepted.
        pub(super) build_time: MaxRelativeIncrease,
    }

    impl Tolerance {
        pub(in crate::index::inmem2) fn example() -> Self {
            const BUILD_TIME: MaxRelativeIncrease = MaxRelativeIncrease::new(0.05).unwrap();

            Self {
                build_time: BUILD_TIME,
            }
        }
    }
}

/// A parsed [`diskann::graph::Config`] and more.
#[derive(Debug, Clone)]
pub(in crate::index::inmem2) struct Config {
    config: diskann::graph::Config,
    num_threads: NonZeroUsize,
}

impl Config {
    /// Validate that a [`diskann::graph::Config`] can be built from the provided parameters.
    ///
    /// Argument `metric` is used to select the pruning rule.
    pub(in crate::index::inmem2) fn from_raw(
        raw: &dto::Config,
        metric: dv::distance::Metric,
    ) -> anyhow::Result<Self> {
        let dto::Config {
            pruned_degree,
            max_degree,
            l_build,
            alpha,
            num_threads,
        } = raw;

        let config = diskann::graph::config::Builder::new_with(
            *pruned_degree,
            diskann::graph::config::MaxDegree::new(*max_degree),
            *l_build,
            metric.into(),
            |b| {
                b.alpha(*alpha);
            },
        )
        .build()?;

        Ok(Self {
            config,
            num_threads: *num_threads,
        })
    }

    /// Return the number of threads used for build operations.
    pub(in crate::index::inmem2) fn num_threads(&self) -> NonZeroUsize {
        self.num_threads
    }

    /// Return the maximum degree in the built index.
    pub(in crate::index::inmem2) fn max_degree(&self) -> usize {
        self.config.max_degree().get()
    }

    /// Return an [`Engine`] to be used for index build operations around `provider`.
    ///
    /// The resulting [`diskann::graph::DiskANNIndex`] will use the parsed
    /// [`diskann::graph::Config`] contained within `self`.
    pub(in crate::index::inmem2) fn engine<R>(
        &self,
        provider: Provider<R>,
    ) -> anyhow::Result<Engine<R>>
    where
        R: di::repr::Representation,
    {
        let Self {
            config,
            num_threads,
        } = self;

        Engine::new(
            Arc::new(diskann::graph::DiskANNIndex::new(
                config.clone(),
                provider,
                None,
            )),
            *num_threads,
        )
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            config,
            num_threads,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();

        let pruned_degree = config.pruned_degree();
        let max_degree = config.max_degree();
        let alpha = config.alpha();
        let l_build = config.l_build();

        kv.push("pruned_degree", &pruned_degree);
        kv.push("max_degree", &max_degree);
        kv.push("alpha", &alpha);
        kv.push("l_build", &l_build);
        kv.push("num_threads", &num_threads);
        write!(f, "{}", kv)
    }
}

/// An execution engine for running index builds on the [`diskann::graph::DiskANNIndex`]
/// constructed from the [`di::repr::Representation`] `R`.
#[derive(Debug)]
pub(in crate::index::inmem2) struct Engine<R>
where
    R: Send + Sync + 'static,
{
    index: Arc<Index<R>>,
    runtime: tokio::runtime::Runtime,
    num_threads: NonZeroUsize,
}

impl<R> Engine<R>
where
    R: di::repr::Representation,
{
    fn new(index: Arc<Index<R>>, num_threads: NonZeroUsize) -> anyhow::Result<Self> {
        let runtime = dbc::tokio::runtime(num_threads.get())?;
        Ok(Self {
            index,
            runtime,
            num_threads,
        })
    }

    /// Return the underlying [`Index`].
    pub(in crate::index::inmem2) fn index(&self) -> &Arc<Index<R>> {
        &self.index
    }

    /// Return the underlying [`tokio::runtime::Runtime`].
    pub(in crate::index::inmem2) fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Run single-insert builds for all rows in `data` into the contained index. Use `ids`
    /// as the external id for their corresponding rows in `data`.
    ///
    /// Progress bars and final prints will be directed to `output` if provided.
    pub(in crate::index::inmem2) fn build_for<T>(
        &self,
        data: Arc<rowmajor::Owned<T>>,
        ids: std::ops::Range<u32>,
        mut output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Stats>
    where
        R: for<'a> di::repr::Insert<Query<'a> = &'a [T]>,
        T: Clone + Send + Sync + 'static,
    {
        if ids.is_empty() {
            anyhow::bail!("empty builds are not supported");
        }

        if ids.len() != data.nrows() {
            anyhow::bail!(
                "ID/Row mismatch - id-len = {}, nrows = {}",
                ids.len(),
                data.nrows()
            );
        }

        let runner = dbc::build::graph::SingleInsert::new(
            self.index.clone(),
            data,
            di::Strategy,
            dbc::build::ids::Range::<u32>::new(ids),
        );

        let progress = output.as_mut().map(|o| Progress::new(*o));
        let results = dbc::build::build_tracked(
            runner,
            dbc::build::Parallelism::dynamic(diskann::utils::ONE, self.num_threads),
            self.runtime(),
            progress
                .as_ref()
                .map(|v| -> &dyn dbc::build::AsProgress { v }),
        )?;

        let stats = Stats::new(results)?;
        if let Some(mut output) = output {
            writeln!(output, "\n\n{}\n", stats)?;
        }
        Ok(stats)
    }

    /// Run single-insert builds for all rows in `data` using the rows index as its ID.
    ///
    /// Progress bars and final prints will be directed to `output` if provided.
    pub(in crate::index::inmem2) fn build<T>(
        &self,
        data: Arc<rowmajor::Owned<T>>,
        output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Stats>
    where
        R: for<'a> di::repr::Insert<Query<'a> = &'a [T]>,
        T: Clone + Send + Sync + 'static,
    {
        let nrows = u32::try_from(data.nrows()).map_err(|_| {
            anyhow::anyhow!(
                "dataset contains more than u32::MAX rows ({})",
                data.nrows()
            )
        })?;

        self.build_for(data, 0..nrows, output)
    }
}

/// Stats for [`Engine::build`] and [`Engine::build_for`].
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index::inmem2) struct Stats {
    total_time: MicroSeconds,
    inserted: usize,
    latencies: percentiles::Percentiles<MicroSeconds>,
}

impl Stats {
    fn new(results: dbc::build::BuildResults<()>) -> anyhow::Result<Self> {
        let total_time = results.end_to_end_latency();

        let mut latencies = Vec::new();
        let mut inserted = 0;
        results.take_output().into_iter().for_each(|r| {
            inserted += r.batchsize();
            latencies.push(r.latency);
        });

        Ok(Self {
            total_time,
            inserted,
            latencies: dbr::utils::percentiles::compute_percentiles(&mut latencies)?,
        })
    }

    /// Return the latency percentiles for insert operations.
    pub(in crate::index::inmem2) fn latencies(&self) -> &percentiles::Percentiles<MicroSeconds> {
        &self.latencies
    }

    /// Return the time spent executing a build operation. This does not count setup time
    /// or time for result aggregation.
    pub(in crate::index::inmem2) fn total_time(&self) -> MicroSeconds {
        self.total_time
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = dbr::utils::fmt::KeyValue::new();

        kv.push_eager(
            "Build time",
            format_args!("{}s", self.total_time.as_seconds()),
        );
        kv.push("Vectors Inserted", &self.inserted);

        let mut latencies = dbr::utils::fmt::KeyValue::new();
        latencies.push_eager("average", format_args!("{}us", self.latencies.mean));
        latencies.push("p90", &self.latencies.p90);
        latencies.push("p99", &self.latencies.p99);

        kv.push("Insert Latencies", &latencies);

        write!(f, "{}", kv)
    }
}

//-------------//
// Regresssion //
//-------------//

#[derive(Debug)]
pub(in crate::index::inmem2) struct Tolerance {
    build_time: MaxRelativeIncrease,
}

impl Tolerance {
    pub(in crate::index::inmem2) fn from_raw(raw: &dto::Tolerance) -> Self {
        let dto::Tolerance { build_time } = *raw;
        Self { build_time }
    }

    pub(in crate::index::inmem2) fn check(&self, old: &Stats, new: &Stats) -> Report {
        let Self { build_time } = self;
        let mut builder = Report::builder();
        builder.push("vectors", Report::equal(old.inserted, new.inserted));
        builder.push(
            "build_time (s)",
            Report::lower_is_better(
                build_time,
                old.total_time.as_seconds(),
                new.total_time.as_seconds(),
                FmtSpec::new().with_precision(3),
            ),
        );
        builder.finish()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_example() {
        use dv::distance::Metric;

        let example = dto::Config::example();
        let metrics = [
            Metric::L2,
            Metric::InnerProduct,
            Metric::Cosine,
            Metric::CosineNormalized,
        ];

        for metric in metrics {
            assert!(Config::from_raw(&example, metric).is_ok());
        }
    }
}
