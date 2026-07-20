/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

#[cfg(feature = "pipnn")]
use std::{
    fs::File,
    io::{BufReader, Read},
    time::Instant,
};

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
#[cfg(feature = "pipnn")]
use diskann_utils::io::Metadata;
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
pub(crate) fn run_pipnn_build<T>(
    input: &IndexBuild,
    config: &diskann_pipnn::PiPNNConfig,
    mut output: &mut dyn Output,
) -> anyhow::Result<(diskann_async::MemoryIndex<T>, BuildStats)>
where
    T: diskann::graph::SampleableForStart + diskann::utils::VectorRepr + AsyncFriendly,
{
    use anyhow::Context;
    use std::io::Write;

    let max_degree = NonZeroUsize::new(input.max_degree() as usize)
        .context("max_degree must be non-zero for PiPNN build")?;
    let context = diskann_pipnn::PiPNNBuildContext::new(
        config.clone(),
        max_degree,
        input.alpha(),
        input.distance().into(),
        input.num_threads(),
    )?;

    let file = File::open(&**input.data())
        .with_context(|| format!("failed to open dataset {}", input.data().display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let (npoints, ndims) = Metadata::read(&mut reader)?.into_dims();
    let row_bytes = ndims
        .checked_mul(std::mem::size_of::<T>())
        .context("dataset row size overflow")?;
    let expected = npoints
        .checked_mul(row_bytes)
        .context("dataset size overflow")? as u64;
    anyhow::ensure!(
        file_len.saturating_sub(8) >= expected,
        "dataset {} is shorter than its {npoints} x {ndims} header",
        input.data().display(),
    );
    anyhow::ensure!(
        row_bytes.is_multiple_of(64),
        "PiPNN in-memory build requires densely packed provider rows; got {row_bytes} bytes",
    );

    let index = diskann_async::new_index::<T, _>(
        input.try_as_config()?.build()?,
        input.inmem_parameters(npoints, ndims),
        diskann_providers::model::graph::provider::async_::common::NoDeletes,
    )?;
    let t_load = Instant::now();
    let mut row = vec![T::default(); ndims];
    for id in 0..npoints {
        reader.read_exact(bytemuck::must_cast_slice_mut(&mut row))?;
        index
            .data_provider
            .base_vectors
            .set_element(&(id as u32), &row)?;
    }
    let load_secs = t_load.elapsed().as_secs_f64();

    let t_start = Instant::now();
    let data = unsafe { index.data_provider.base_vectors.flat_prefix(npoints) };
    let data = MatrixView::try_from(data, npoints, ndims).map_err(|error| error.as_static())?;
    set_start_points(index.provider(), data, *input.start_point_strategy())?;
    let start_secs = t_start.elapsed().as_secs_f64();

    let total_points = npoints + input.start_point_strategy().count();
    let data = unsafe { index.data_provider.base_vectors.flat_prefix(total_points) };
    let graph = diskann_pipnn::builder::build_typed(data, total_points, ndims, &context)?;
    let diskann_pipnn::builder::PiPNNBuildOutput {
        adjacency,
        build_stats,
    } = graph;
    let build_secs = build_stats.total_secs;

    let t_transfer = Instant::now();
    for (id, neighbors) in adjacency.into_iter().enumerate() {
        index
            .provider()
            .neighbors()
            .set_neighbors_sync(id, &neighbors)?;
    }
    let transfer_secs = t_transfer.elapsed().as_secs_f64();
    let total_secs = load_secs + start_secs + build_secs + transfer_secs;

    writeln!(output, "{}", build_stats)?;
    writeln!(
        output,
        "PiPNN load={load_secs:.3}s start={start_secs:.3}s transfer={transfer_secs:.3}s total={total_secs:.3}s"
    )?;

    let total_time = MicroSeconds::from(std::time::Duration::from_secs_f64(total_secs));
    let per_vector = MicroSeconds::from(std::time::Duration::from_secs_f64(
        total_secs / npoints as f64,
    ));
    let stats = BuildStats {
        kind: BuildKind::PiPNN,
        total_time,
        vectors_inserted: npoints,
        insert_latencies: percentiles::compute_percentiles(&mut [per_vector])?,
    };
    Ok((index, stats))
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
