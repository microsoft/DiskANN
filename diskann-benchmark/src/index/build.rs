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
use diskann_providers::{
    self,
    model::{configuration::IndexConfiguration, graph::provider::async_::inmem::SetStartPoints},
    storage::{AsyncIndexMetadata, LoadWith, SaveWith},
};
#[cfg(feature = "pipnn")]
use diskann_providers::{
    index::diskann_async, model::graph::provider::async_::common::SetElementHelper,
};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::inputs::graph_index::IndexBuild;

///////////////////////////////
// Start Point Configuration //
///////////////////////////////

pub(crate) fn set_start_points<DP, T>(
    provider: &DP,
    data: MatrixView<'_, T>,
    start_strategy: StartPointStrategy,
) -> ANNResult<()>
where
    DP: SetStartPoints<[T]>,
    T: diskann::graph::SampleableForStart + AsyncFriendly,
{
    let start_points = start_strategy
        .compute(data)
        .map_err(|e| ANNError::new(diskann::ANNErrorKind::DiskANN(StartPointComputeError), e))?;
    provider.set_start_points(start_points.row_iter())
}

///////////
// Build //
///////////

pub(crate) fn single_or_multi_insert<DP, T, S>(
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
    let rt = diskann_benchmark_core::tokio::runtime(input.num_threads())?;
    match input.multi_insert() {
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
                    NonZeroUsize::new(input.num_threads()).unwrap(),
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

#[cfg(feature = "pipnn")]
pub(crate) fn pipnn_build<T>(
    index: diskann_async::MemoryIndex<T>,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    parameters: &diskann_disk::PiPNNParameters,
) -> anyhow::Result<BuildStats>
where
    T: diskann::graph::SampleableForStart + diskann::utils::VectorRepr,
{
    use anyhow::Context;

    let npoints = data.nrows();
    let dimensions = data.ncols();
    let row_bytes = dimensions
        .checked_mul(std::mem::size_of::<T>())
        .context("dataset row size overflow")?;
    let graph = input.try_as_config()?.build()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(input.num_threads())
        .build()
        .context("failed to create PiPNN build thread pool")?;
    let mut context = diskann_pipnn::PiPNNBuildContext::new(
        parameters.into(),
        &graph,
        input.distance().into(),
        &pool,
    )?;
    if let Some(hash_prune) = &parameters.hash_prune {
        context = context.with_hash_prune(hash_prune.into())?;
    }

    let started = std::time::Instant::now();
    for (id, vector) in data.row_iter().enumerate() {
        let id = u32::try_from(id).context("PiPNN point ID exceeds u32::MAX")?;
        index.data_provider.base_vectors.set_element(&id, vector)?;
    }

    let mut source = Some(data);
    let (adjacency, start_points, real_start_id) = {
        let build_data = if row_bytes.is_multiple_of(64) {
            drop(source.take());
            // SAFETY: every real row is initialized above, dense rows have no
            // inter-row padding, and no writes race graph construction.
            let values = unsafe { index.data_provider.base_vectors.flat_prefix(npoints) };
            MatrixView::try_from(values, npoints, dimensions).map_err(|error| error.as_static())?
        } else {
            source
                .as_ref()
                .context("PiPNN source data was released before graph construction")?
                .as_view()
        };
        let adjacency = diskann_pipnn::build_graph(build_data, &context)?;
        let start_points = StartPointStrategy::Medoid
            .compute(build_data)
            .map_err(|error| {
                ANNError::new(
                    diskann::ANNErrorKind::DiskANN(StartPointComputeError),
                    error,
                )
            })?;
        let start_vector = start_points.row(0);
        let start_bytes: &[u8] = bytemuck::cast_slice(start_vector);
        let real_start_id = build_data
            .row_iter()
            .position(|row| bytemuck::cast_slice::<T, u8>(row) == start_bytes)
            .context("PiPNN medoid is not a dataset vector")?;
        (adjacency, start_points, real_start_id)
    };
    drop(source);
    for (id, neighbors) in adjacency.iter().enumerate() {
        index
            .provider()
            .neighbors()
            .set_neighbors_sync(id, neighbors)?;
    }
    index.provider().set_start_points(start_points.row_iter())?;
    let start_id = index
        .provider()
        .starting_points()?
        .into_iter()
        .next()
        .context("PiPNN provider has no search start slot")?;
    index
        .provider()
        .neighbors()
        .set_neighbors_sync(start_id as usize, &adjacency[real_start_id])?;

    let total_time = MicroSeconds::from(started.elapsed());
    let per_vector = MicroSeconds::new(
        total_time
            .as_micros()
            .checked_div(u64::try_from(npoints)?)
            .context("PiPNN build received an empty dataset")?,
    );
    Ok(BuildStats {
        kind: BuildKind::PiPNN,
        total_time,
        vectors_inserted: npoints,
        insert_latencies: percentiles::compute_percentiles(&mut [per_vector])?,
    })
}

#[cfg(any(feature = "scalar-quantization", feature = "spherical-quantization"))]
pub(crate) fn only_single_insert<DP, T, S>(
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
    let rt = diskann_benchmark_core::tokio::runtime(input.num_threads())?;
    match input.multi_insert() {
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
                    NonZeroUsize::new(input.num_threads()).unwrap(),
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
pub(crate) enum BuildKind {
    SingleInsert,
    MultiInsert,
    #[cfg(feature = "pipnn")]
    PiPNN,
}

impl std::fmt::Display for BuildKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleInsert => write!(f, "single insert"),
            Self::MultiInsert => write!(f, "multi insert"),
            #[cfg(feature = "pipnn")]
            Self::PiPNN => write!(f, "PiPNN"),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct BuildStats {
    pub(crate) kind: BuildKind,
    pub(crate) total_time: MicroSeconds,
    pub(crate) vectors_inserted: usize,
    pub(crate) insert_latencies: percentiles::Percentiles<MicroSeconds>,
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

pub(crate) async fn save_index<DP, T>(
    index: Arc<DiskANNIndex<DP>>,
    save_path: &str,
) -> anyhow::Result<()>
where
    DP: DataProvider<Context = DefaultContext, ExternalId = u32>
        + for<'a> provider::SetElement<&'a [T]>,
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
pub(crate) async fn load_index<'a, DP>(
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
