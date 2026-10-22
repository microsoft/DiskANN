/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann_benchmark_core as dbc;
use diskann_benchmark_runner as dbr;
use diskann_utils::views::rowmajor;
use diskann_vector as dv;
use serde::Serialize;

use crate::utils::progress::Progress;

pub(in crate::index::inmem2) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Config {
        pub(super) pruned_degree: usize,
        pub(super) max_degree: usize,
        pub(super) l_build: usize,
        pub(super) alpha: f32,
        pub(super) num_threads: NonZeroUsize,
    }
}

#[derive(Debug, Clone)]
pub(in crate::index::inmem2) struct Config {
    pub(in crate::index::inmem2) config: diskann::graph::Config,
    pub(in crate::index::inmem2) num_threads: NonZeroUsize,
}

impl Config {
    fn from_raw(raw: &dto::Config, metric: dv::distance::Metric) -> anyhow::Result<Self> {
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

    fn engine<DP, S>(&self, provider: DP, strategy: S) -> Engine<DP, S>
    where
        DP: diskann::provider::DataProvider,
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
            strategy,
            *num_threads,
            self.clone(),
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

struct Engine<DP, S>
where
    DP: diskann::provider::DataProvider,
{
    index: Arc<diskann::graph::DiskANNIndex<DP>>,
    strategy: S,
    num_threads: NonZeroUsize,
    config: Config,
}

impl<DP, S> Engine<DP, S>
where
    DP: diskann::provider::DataProvider,
{
    fn new(
        index: Arc<diskann::graph::DiskANNIndex<DP>>,
        strategy: S,
        num_threads: NonZeroUsize,
        config: Config,
    ) -> Self {
        Self {
            index,
            strategy,
            num_threads,
            config,
        }
    }

    fn index(&self) -> &Arc<diskann::graph::DiskANNIndex<DP>> {
        &self.index
    }

    fn build<T>(
        &self,
        data: Arc<rowmajor::Owned<T>>,
        output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Stats>
    where
        dbc::build::graph::SingleInsert<DP, T, S>: dbc::build::Build<Output = ()>,
        DP: diskann::provider::DataProvider<InternalId = u32, ExternalId = u32>,
        S: Clone,
    {
        let rt = dbc::tokio::runtime(self.num_threads.get())?;
        let runner = dbc::build::graph::SingleInsert::new(
            self.index.clone(),
            data,
            self.strategy.clone(),
            dbc::build::ids::Identity::<u32>::new(),
        );

        let results = dbc::build::build_tracked(
            runner,
            dbc::build::Parallelism::dynamic(diskann::utils::ONE, self.num_threads),
            &rt,
            Some(&Progress::new(output)),
        )?;

        Stats::new(results)
    }
}

#[derive(Debug, Serialize)]
struct Stats {
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
