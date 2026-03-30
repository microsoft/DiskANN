/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::File,
    io::{BufReader, Read},
    num::NonZeroUsize,
    sync::Arc,
    time::Instant,
};

use anyhow::Context;
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
    index::diskann_async,
    model::{
        configuration::IndexConfiguration,
        graph::provider::async_::{common::SetElementHelper, inmem::SetStartPoints},
    },
    storage::{AsyncIndexMetadata, LoadWith, SaveWith},
};
use diskann_utils::{
    future::AsyncFriendly,
    io::Metadata,
    sampling::WithApproximateNorm,
    views::{Matrix, MatrixView},
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::inputs::async_::IndexBuild;

/// Process peak resident-set size in GiB (high-water mark since start).
/// Linux: `/proc/self/status` VmHWM. Windows: `PeakWorkingSetSize`. Returns
/// `None` on platforms/paths where it can't be read.
pub(super) fn peak_rss_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb / 1024.0 / 1024.0);
            }
        }
        None
    }
    #[cfg(windows)]
    {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(
                process: isize,
                counters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        // SAFETY: passing a zeroed, correctly-sized PROCESS_MEMORY_COUNTERS to
        // the psapi call; we read peak_working_set_size only on success.
        unsafe {
            let mut c: ProcessMemoryCounters = std::mem::zeroed();
            c.cb = std::mem::size_of::<ProcessMemoryCounters>() as u32;
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) != 0 {
                Some(c.peak_working_set_size as f64 / 1024.0 / 1024.0 / 1024.0)
            } else {
                None
            }
        }
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

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

/// Load a dataset directly into the search provider, then build it with PiPNN.
pub(super) fn run_pipnn_build<T>(
    input: &IndexBuild,
    mut output: &mut dyn Output,
) -> anyhow::Result<(diskann_async::MemoryIndex<T>, BuildStats)>
where
    T: diskann::graph::SampleableForStart
        + diskann::utils::VectorRepr
        + WithApproximateNorm
        + AsyncFriendly,
{
    use std::io::Write;

    let pipnn_cfg = input
        .pipnn
        .as_ref()
        .expect("run_pipnn_build called without PiPNN config");
    let diskann_disk::build::configuration::BuildAlgorithm::PiPNN(algo_config) = pipnn_cfg else {
        anyhow::bail!("run_pipnn_build called but build algorithm is not PiPNN");
    };
    let max_degree = std::num::NonZeroUsize::new(input.max_degree)
        .context("max_degree must be non-zero for PiPNN build")?;
    let ctx = diskann_pipnn::PiPNNBuildContext::new(
        algo_config.clone(),
        max_degree,
        input.distance.into(),
        input.num_threads,
    )
    .map_err(|e| anyhow::anyhow!("invalid PiPNN config: {e}"))?;

    let file = File::open(&*input.data)
        .with_context(|| format!("failed to open dataset {}", input.data.display()))?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let (npoints, ndims) = Metadata::read(&mut reader)
        .with_context(|| format!("failed to read dataset header {}", input.data.display()))?
        .into_dims();
    let row_bytes = ndims
        .checked_mul(std::mem::size_of::<T>())
        .context("dataset row size overflow")?;
    let expected = npoints
        .checked_mul(row_bytes)
        .context("dataset size overflow")? as u64;
    let available = file_len.saturating_sub(8);
    anyhow::ensure!(
        available >= expected,
        "dataset {} declares {npoints} x {ndims} values ({} bytes), but only {available} payload bytes are available",
        input.data.display(),
        std::mem::size_of::<T>(),
    );
    anyhow::ensure!(
        row_bytes.is_multiple_of(64),
        "PiPNN direct provider build requires rows whose byte size is a multiple of 64; got {ndims} dimensions x {} bytes = {row_bytes} bytes",
        std::mem::size_of::<T>(),
    );

    let index = diskann_async::new_index::<T, _>(
        input.try_as_config()?.build()?,
        input.inmem_parameters(npoints, ndims),
        diskann_providers::model::graph::provider::async_::common::NoDeletes,
    )?;
    writeln!(output, "PiPNN build: {} points x {} dims", npoints, ndims)?;

    let t_store = Instant::now();
    let mut row = vec![T::default(); ndims];
    for id in 0..npoints {
        reader.read_exact(bytemuck::must_cast_slice_mut(&mut row))?;
        index
            .data_provider
            .base_vectors
            .set_element(&(id as u32), &row)?;
    }
    let store_secs = t_store.elapsed().as_secs_f64();
    writeln!(output, "  Vector load:   {:.3}s", store_secs)?;

    // SAFETY: loading is complete, no concurrent writes occur, and the first
    // `npoints` rows are initialized. `flat_prefix` verifies dense packing.
    let flat = unsafe { index.data_provider.base_vectors.flat_prefix(npoints) };
    let data = MatrixView::try_from(flat, npoints, ndims).map_err(|e| e.as_static())?;
    set_start_points(index.provider(), data, input.start_point_strategy)?;

    // SAFETY: start-point writes target the frozen rows after `npoints`; the
    // loaded prefix is initialized, densely packed, and no longer mutated.
    let flat: &[T] = unsafe { index.data_provider.base_vectors.flat_prefix(npoints) };
    let graph = diskann_pipnn::builder::build_typed::<T>(flat, npoints, ndims, &ctx)?;

    writeln!(output, "{}", graph.build_stats)?;
    writeln!(
        output,
        "  Avg degree: {:.1}, Max degree: {}, Isolated: {}",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated(),
    )?;

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
    writeln!(
        output,
        "  Edge transfer: {:.3}s (medoid={})",
        transfer_secs, graph.medoid
    )?;

    let total_secs = graph.build_stats.total_secs + store_secs + transfer_secs;
    writeln!(
        output,
        "  Total (incl. load + transfer): {:.3}s\n",
        total_secs
    )?;

    let total_us = MicroSeconds::from(std::time::Duration::from_secs_f64(total_secs));
    let per_vec = MicroSeconds::from(std::time::Duration::from_secs_f64(
        total_secs / npoints as f64,
    ));
    let stats = BuildStats {
        kind: BuildKind::PiPNN,
        total_time: total_us,
        vectors_inserted: npoints,
        insert_latencies: percentiles::compute_percentiles(&mut [per_vec; 1])?,
    };

    if let Some(gb) = peak_rss_gb() {
        writeln!(output, "Peak RSS (build stage): {:.2} GiB", gb)?;
    }

    Ok((index, stats))
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

#[cfg(test)]
mod tests {
    use std::fs::File;

    use diskann::graph::StartPointStrategy;
    use diskann_benchmark_runner::{files::InputFile, output::Memory, utils::datatype::DataType};
    use diskann_disk::build::configuration::BuildAlgorithm;
    use diskann_utils::{io::write_bin, views::MatrixView};

    use super::run_pipnn_build;
    use crate::{inputs::async_::IndexBuild, utils::SimilarityMeasure};

    #[test]
    fn pipnn_build_streams_bin_directly_into_provider() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.fbin");
        const NDIMS: usize = 16;
        let data: Vec<f32> = (0..64 * NDIMS).map(|x| x as f32).collect();
        let view = MatrixView::try_from(data.as_slice(), 64, NDIMS).unwrap();
        write_bin(view, &mut File::create(&path).unwrap()).unwrap();

        let mut input = IndexBuild {
            data_type: DataType::Float32,
            data: InputFile::new(path),
            distance: SimilarityMeasure::SquaredL2,
            max_degree: 8,
            l_build: 8,
            insert_retry: None,
            start_point_strategy: StartPointStrategy::FirstVector,
            alpha: 1.2,
            backedge_ratio: 1.0,
            num_threads: 1,
            multi_insert: None,
            save_path: None,
            pipnn: Some(BuildAlgorithm::PiPNN(diskann_pipnn::PiPNNConfig {
                num_hash_planes: 4,
                c_max: 64,
                c_min: 2,
                p_samp: 0.5,
                fanout: vec![2],
                k: 2,
                replicas: 1,
                l_max: 8,
                final_prune: true,
                alpha: 1.2,
            })),
        };
        let mut output = Memory::new();
        let (index, stats) = run_pipnn_build::<f32>(&input, &mut output).unwrap();

        let loaded = unsafe { index.data_provider.base_vectors.flat_prefix(64) };
        assert_eq!(loaded, data);
        assert_eq!(stats.vectors_inserted, 64);

        let unaligned_path = dir.path().join("unaligned.fbin");
        let unaligned = vec![0.0f32; 64 * 17];
        let view = MatrixView::try_from(unaligned.as_slice(), 64, 17).unwrap();
        write_bin(view, &mut File::create(&unaligned_path).unwrap()).unwrap();
        input.data = InputFile::new(unaligned_path);

        let error = run_pipnn_build::<f32>(&input, &mut output).err().unwrap();
        assert!(error.to_string().contains(
            "PiPNN direct provider build requires rows whose byte size is a multiple of 64"
        ));
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
