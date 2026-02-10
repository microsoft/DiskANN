/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
    error::DiskANNError::StartPointComputeError,
    graph::{DiskANNIndex, StartPointStrategy},
    provider::{self, DataProvider, DefaultContext},
    ANNError, ANNResult,
};
use diskann_benchmark_core::build as build_core;
use diskann_benchmark_runner::{
    output::Output,
    utils::{percentiles, MicroSeconds},
};
use diskann_inmem::SetStartPoints;
use diskann_providers::{
    self,
    model::configuration::IndexConfiguration,
    storage::{AsyncIndexMetadata, LoadWith, SaveWith},
};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::inputs::async_::IndexBuild;

///////////////////////////////
// Start Point Configuration //
///////////////////////////////

pub(super) fn set_start_points<DP, T>(
    provider: &DP,
    data: MatrixView<'_, T>,
    start_strategy: StartPointStrategy,
) -> ANNResult<()>
where
    DP: SetStartPoints<[T]>,
    T: diskann::graph::SampleableForStart
        + diskann_utils::sampling::WithApproximateNorm
        + AsyncFriendly,
{
    let start_points = start_strategy
        .compute(data)
        .map_err(|e| ANNError::new(diskann::ANNErrorKind::DiskANN(StartPointComputeError), e))?;
    provider.set_start_points(start_points.row_iter())
}

///////////
// Build //
///////////

pub(super) fn single_or_multi_insert<DP, T, S>(
    index: Arc<DiskANNIndex<DP>>,
    strategy: S,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    output: &mut dyn Output,
) -> anyhow::Result<BuildStats>
where
    DP: DataProvider,
    build_core::graph::SingleInsert<DP, T, S>: build_core::Build<Output = ()>,
    build_core::graph::MultiInsert<DP, T, S>: build_core::Build<Output = ()>,
    build_core::ids::Identity<DP::ExternalId>: build_core::ids::ToId<DP::ExternalId>,
{
    let rt = diskann_benchmark_core::tokio::runtime(input.num_threads)?;
    match &input.multi_insert {
        None => {
            let runner = build_core::graph::SingleInsert::new(
                index,
                data,
                strategy,
                build_core::ids::Identity::<DP::ExternalId>::new(),
            );

            let results = build_core::build_tracked(
                runner,
                build_core::Parallelism::dynamic(
                    diskann::utils::ONE,
                    NonZeroUsize::new(input.num_threads).unwrap(),
                ),
                &rt,
                Some(&ProgressMeter::new(output)),
            )?;

            Ok(BuildStats::new(BuildKind::SingleInsert, results)?)
        }
        Some(multi_insert) => {
            let runner = build_core::graph::MultiInsert::new(
                index,
                data,
                strategy,
                build_core::ids::Identity::<DP::ExternalId>::new(),
            );

            let results = build_core::build_tracked(
                runner,
                build_core::Parallelism::sequential(multi_insert.batch_size),
                &rt,
                Some(&ProgressMeter::new(output)),
            )?;

            Ok(BuildStats::new(BuildKind::MultiInsert, results)?)
        }
    }
}

#[cfg(any(feature = "scalar-quantization", feature = "spherical-quantization"))]
pub(super) fn only_single_insert<DP, T, S>(
    index: Arc<DiskANNIndex<DP>>,
    strategy: S,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    output: &mut dyn Output,
) -> anyhow::Result<BuildStats>
where
    DP: DataProvider,
    build_core::graph::SingleInsert<DP, T, S>: build_core::Build<Output = ()>,
    build_core::ids::Identity<DP::ExternalId>: build_core::ids::ToId<DP::ExternalId>,
{
    let rt = diskann_benchmark_core::tokio::runtime(input.num_threads)?;
    match &input.multi_insert {
        None => {
            let runner = build_core::graph::SingleInsert::new(
                index,
                data,
                strategy,
                build_core::ids::Identity::<DP::ExternalId>::new(),
            );

            let results = build_core::build_tracked(
                runner,
                build_core::Parallelism::dynamic(
                    diskann::utils::ONE,
                    NonZeroUsize::new(input.num_threads).unwrap(),
                ),
                &rt,
                Some(&ProgressMeter::new(output)),
            )?;

            Ok(BuildStats::new(BuildKind::SingleInsert, results)?)
        }
        Some(_) => Err(anyhow::anyhow!(
            "please file a bug report, this quantization does not \
             support multi-insert and this should have been rejected \
             by the benchmark front-end"
        )),
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename = "kebab-case")]
pub(super) enum BuildKind {
    SingleInsert,
    MultiInsert,
}

impl std::fmt::Display for BuildKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleInsert => write!(f, "single insert"),
            Self::MultiInsert => write!(f, "multi insert"),
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct BuildStats {
    pub(super) kind: BuildKind,
    pub(super) total_time: MicroSeconds,
    pub(super) vectors_inserted: usize,
    pub(super) insert_latencies: percentiles::Percentiles<MicroSeconds>,
}

impl BuildStats {
    pub(crate) fn new(
        kind: BuildKind,
        results: build_core::BuildResults<()>,
    ) -> anyhow::Result<Self> {
        let total_time = results.end_to_end_latency();

        let mut latencies = Vec::new();
        let mut vectors_inserted = 0;
        results.take_output().into_iter().for_each(|r| {
            vectors_inserted += r.batchsize();
            latencies.push(r.latency);
        });

        Ok(Self {
            kind,
            total_time,
            vectors_inserted,
            insert_latencies: percentiles::compute_percentiles(&mut latencies)?,
        })
    }
}

impl std::fmt::Display for BuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index Build Time: {}s", self.total_time.as_seconds())?;
        writeln!(f, "Vectors Inserted: {}", self.vectors_inserted)?;
        writeln!(f, "Kind: {}", self.kind)?;
        write!(
            f,
            "Insert Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
            self.insert_latencies.mean, self.insert_latencies.p90, self.insert_latencies.p99,
        )
    }
}

pub struct ProgressMeter<'a> {
    output: &'a mut dyn Output,
}

impl<'a> ProgressMeter<'a> {
    pub fn new(output: &'a mut dyn Output) -> Self {
        Self { output }
    }
}

impl build_core::AsProgress for ProgressMeter<'_> {
    fn as_progress(&self, max: usize) -> Arc<dyn build_core::Progress> {
        let target = self.output.draw_target();
        let meter = ProgressBar::with_draw_target(Some(max as u64), target);
        meter.set_style(
            ProgressStyle::with_template("Building [{elapsed_precise}] {wide_bar} {percent}")
                .expect("This format should be valid"),
        );
        Arc::new(Meter { meter })
    }
}

#[derive(Debug)]
struct Meter {
    meter: ProgressBar,
}

impl build_core::Progress for Meter {
    fn progress(&self, handled: usize) {
        self.meter.inc(handled as u64)
    }
    fn finish(&self) {
        self.meter.finish()
    }
}

////////////////////////
// Save and Load API ///
////////////////////////

pub(super) async fn save_index<DP, T>(
    index: Arc<DiskANNIndex<DP>>,
    save_path: &str,
) -> anyhow::Result<()>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32> + provider::SetElement<[T]>,
    DiskANNIndex<DP>: SaveWith<AsyncIndexMetadata, Error = ANNError>,
{
    index
        .save_with(
            &diskann_providers::storage::FileStorageProvider,
            &AsyncIndexMetadata::new(save_path),
        )
        .await?;

    Ok(())
}

// for now, this only works with full-precision indices
pub(super) async fn load_index<'a, DP>(
    load_path: &'a str,
    index_config: &IndexConfiguration,
) -> anyhow::Result<DiskANNIndex<DP>>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32>,
    DiskANNIndex<DP>:
        diskann_providers::storage::LoadWith<(&'a str, IndexConfiguration), Error = ANNError>,
{
    let index = DiskANNIndex::<DP>::load_with(
        &diskann_providers::storage::FileStorageProvider,
        &(load_path, index_config.clone()),
    )
    .await?;

    Ok(index)
}
