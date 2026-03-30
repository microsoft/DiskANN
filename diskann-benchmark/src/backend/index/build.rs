/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc, time::Instant};

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

/// Build using PiPNN instead of Vamana.
///
/// Has the same `BF` signature as `single_or_multi_insert` for drop-in use in `run_build`.
pub(super) fn pipnn_insert<U, V, D, T, S>(
    index: Arc<DiskANNIndex<diskann_providers::model::graph::provider::async_::inmem::DefaultProvider<U, V, D>>>,
    _strategy: S,
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    mut output: &mut dyn Output,
) -> anyhow::Result<BuildStats>
where
    T: diskann::utils::VectorRepr + Send + Sync + bytemuck::Pod,
    U: AsyncFriendly + diskann_providers::model::graph::provider::async_::common::SetElementHelper<T>,
    V: AsyncFriendly + diskann_providers::model::graph::provider::async_::common::SetElementHelper<T>,
    D: AsyncFriendly,
{
    use std::io::Write;

    let pipnn_cfg = input
        .pipnn
        .as_ref()
        .expect("pipnn_insert called without PiPNN config");
    let config = pipnn_cfg.to_pipnn_config(input);

    let npoints = data.nrows();
    let ndims = data.ncols();

    writeln!(output, "PiPNN build: {} points x {} dims", npoints, ndims)?;

    // Step 1: Store all data vectors into the provider so search can read them.
    let t_store = Instant::now();
    {
        use diskann::provider::SetElement;
        let provider = &index.data_provider;
        let ctx = &DefaultContext;
        let rt = diskann_benchmark_core::tokio::runtime(input.num_threads)?;
        rt.block_on(async {
            for i in 0..npoints {
                let id = i as u32;
                let row: &[T] = data.row(i);
                SetElement::<[T]>::set_element(provider, ctx, &id, row)
                    .await
                    .map_err(|e| anyhow::anyhow!("set_element failed for id {}: {}", i, e))?;
            }
            Ok::<(), anyhow::Error>(())
        })?;
    }
    let store_secs = t_store.elapsed().as_secs_f64();
    writeln!(output, "  Vector store:  {:.3}s", store_secs)?;

    // Step 2: Run PiPNN build on the raw flat data.
    let flat_data = data.as_slice();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(input.num_threads)
        .build()?;
    let graph = pool.install(|| {
        diskann_pipnn::builder::build_typed::<T>(flat_data, npoints, ndims, &config)
    })?;

    writeln!(output, "{}", graph.build_stats)?;
    writeln!(
        output,
        "  Avg degree: {:.1}, Max degree: {}, Isolated: {}",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated(),
    )?;

    // Step 3: Transfer adjacency lists into the index.
    let t_transfer = Instant::now();
    {
        let provider = index.provider();
        let neighbors = provider.neighbors();
        for (i, adj) in graph.adjacency.iter().enumerate() {
            neighbors.set_neighbors_sync(i, adj)?;
        }
        // The frozen start point lives at index `npoints` (after data points).
        // Search begins there, so it needs edges. Copy the PiPNN medoid's edges.
        let medoid_adj = graph.adjacency[graph.medoid].clone();
        neighbors.set_neighbors_sync(npoints, &medoid_adj)?;
    }
    let transfer_secs = t_transfer.elapsed().as_secs_f64();
    writeln!(output, "  Edge transfer: {:.3}s (medoid={})", transfer_secs, graph.medoid)?;

    let total_secs = graph.build_stats.total_secs + store_secs + transfer_secs;
    writeln!(output, "  Total (incl. store + transfer): {:.3}s\n", total_secs)?;

    let total_us = MicroSeconds::from(std::time::Duration::from_secs_f64(total_secs));
    let per_vec = MicroSeconds::from(std::time::Duration::from_secs_f64(total_secs / npoints as f64));
    Ok(BuildStats {
        kind: BuildKind::PiPNN,
        total_time: total_us,
        vectors_inserted: npoints,
        insert_latencies: percentiles::compute_percentiles(&mut vec![per_vec; 1])?,
    })
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
    PiPNN,
}

impl std::fmt::Display for BuildKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleInsert => write!(f, "single insert"),
            Self::MultiInsert => write!(f, "multi insert"),
            Self::PiPNN => write!(f, "PiPNN"),
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
