/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core as dbc;
use diskann_benchmark_runner as dbr;
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector as dv;
use serde::{Deserialize, Serialize};

use crate::{
    index::inmem2::{nz, Index, Provider},
    utils::progress::Progress,
};

pub(in crate::index::inmem2) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
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
}

#[derive(Debug, Clone)]
pub(in crate::index::inmem2) struct Config {
    config: diskann::graph::Config,
    num_threads: NonZeroUsize,
}

impl Config {
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

    pub(in crate::index::inmem2) fn num_threads(&self) -> NonZeroUsize {
        self.num_threads
    }

    pub(in crate::index::inmem2) fn max_degree(&self) -> usize {
        self.config.max_degree().get()
    }

    pub(in crate::index::inmem2) fn engine<R>(&self, provider: Provider<R>) -> Engine<R>
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
        let mut kv = dbr::utils::fmt::KeyValue::new();

        let pruned_degree = self.config.pruned_degree();
        let max_degree = self.config.max_degree();
        let alpha = self.config.alpha();
        let l_build = self.config.l_build();

        kv.push("pruned_degree", &pruned_degree);
        kv.push("max_degree", &max_degree);
        kv.push("alpha", &alpha);
        kv.push("l_build", &l_build);
        kv.push("num_threads", &self.num_threads);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Engine<R>
where
    R: Send + Sync + 'static,
{
    index: Arc<Index<R>>,
    num_threads: NonZeroUsize,
}

impl<R> Engine<R>
where
    R: di::repr::Representation,
{
    fn new(index: Arc<Index<R>>, num_threads: NonZeroUsize) -> Self {
        Self { index, num_threads }
    }

    pub(in crate::index::inmem2) fn index(&self) -> &Arc<Index<R>> {
        &self.index
    }

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
        let rt = dbc::tokio::runtime(self.num_threads.get())?;
        let runner = dbc::build::graph::SingleInsert::new(
            self.index.clone(),
            data,
            di::Strategy,
            dbc::build::ids::Range::<u32>::new(ids),
        );

        let progress = match output.as_mut() {
            Some(o) => Some(Progress::new(*o)),
            None => None,
        };

        let results = dbc::build::build_tracked(
            runner,
            dbc::build::Parallelism::dynamic(diskann::utils::ONE, self.num_threads),
            &rt,
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

    pub(in crate::index::inmem2) fn build<T>(
        &self,
        data: Arc<rowmajor::Owned<T>>,
        mut output: Option<&mut dyn dbr::Output>,
    ) -> anyhow::Result<Stats>
    where
        R: for<'a> di::repr::Insert<Query<'a> = &'a [T]>,
        T: Clone + Send + Sync + 'static,
    {
        let nrows = data.nrows() as u32;
        self.build_for(data, 0..nrows, output)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(in crate::index::inmem2) struct Stats {
    total_time: dbr::utils::MicroSeconds,
    inserted: usize,
    latencies: dbr::utils::percentiles::Percentiles<dbr::utils::MicroSeconds>,
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
