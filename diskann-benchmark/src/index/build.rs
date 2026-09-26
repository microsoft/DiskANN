/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::{
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
    index::diskann_async,
    model::graph::provider::async_::common::{self, SetElementHelper},
};
use diskann_utils::{
    future::AsyncFriendly,
    views::{Matrix, MatrixView},
};
#[cfg(feature = "pipnn")]
use diskann_vector::DistanceFunction;
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
    let start_points = start_strategy.compute(data).map_err(ANNError::new)?;
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
/// Build a PiPNN graph and install it in a searchable provider.
///
/// PiPNN completes its graph before provider creation. The function also computes
/// start vectors and their source IDs. It then installs vectors, edges, and start
/// slots in a new `MemoryIndex`.
///
/// `BuildStats` includes graph construction and provider installation. It does
/// not include provider allocation.
pub(crate) fn pipnn_build<T>(
    data: Arc<Matrix<T>>,
    input: &IndexBuild,
    parameters: &diskann_disk::PiPNNParameters,
) -> anyhow::Result<(diskann_async::MemoryIndex<T>, BuildStats)>
where
    T: diskann::graph::SampleableForStart + diskann::utils::VectorRepr,
{
    use anyhow::Context;
    use rayon::prelude::*;

    let npoints = data.nrows();
    let dimensions = data.ncols();
    let metric = input.distance().into();
    let graph = input.try_as_config()?.build()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(input.num_threads())
        .build()
        .context("failed to create PiPNN build thread pool")?;

    let started = std::time::Instant::now();
    let adjacency = {
        let context = diskann::graph::pipnn::PiPNNBuildContext::new(
            parameters.into(),
            &graph,
            metric,
            &pool,
        )?;
        diskann::graph::pipnn::build_graph(data.as_view(), &context)?
    };
    let start_points = input
        .start_point_strategy()
        .compute(data.as_view())
        .map_err(ANNError::new)?;
    let distance = T::distance(metric, Some(dimensions));
    // Start vectors use frozen IDs outside the real point-ID range. Connect each
    // frozen ID to its source row. For a synthetic start vector, connect it to
    // the nearest real row. `total_cmp` gives a deterministic total order.
    let start_sources = start_points
        .row_iter()
        .map(|start| {
            let bytes: &[u8] = bytemuck::cast_slice(start);
            data.row_iter()
                .position(|row| bytemuck::cast_slice::<T, u8>(row) == bytes)
                .or_else(|| {
                    data.row_iter()
                        .enumerate()
                        .map(|(index, row)| (index, distance.evaluate_similarity(start, row)))
                        .min_by(|(_, left), (_, right)| left.total_cmp(right))
                        .map(|(index, _)| index)
                })
                .context("PiPNN cannot connect a start point to an empty dataset")
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // A frozen slot stores the selected source vector. Its adjacency must equal
    // that source row. Adding the source ID removes one graph edge because the
    // row has a fixed degree.
    let start_neighbors: Vec<_> = start_sources
        .into_iter()
        .map(|source| adjacency[source].clone())
        .collect();
    let batch_elapsed = started.elapsed();

    let index = diskann_async::new_index::<T, _>(
        graph,
        input.inmem_parameters(npoints, dimensions),
        common::NoDeletes,
    )?;
    // `BuildStats` excludes provider allocation and includes provider installation.
    let install_started = std::time::Instant::now();
    // Install vectors before edges. A returned index must contain a vector for
    // every graph ID.
    #[expect(
        clippy::disallowed_methods,
        reason = "the supplied pool owns both terminal operations"
    )]
    pool.install(|| -> anyhow::Result<()> {
        (0..npoints)
            .into_par_iter()
            .try_for_each(|id| -> anyhow::Result<()> {
                let point_id = u32::try_from(id).context("PiPNN point ID exceeds u32::MAX")?;
                index
                    .data_provider
                    .base_vectors
                    .set_element(&point_id, data.row(id))?;
                Ok(())
            })?;
        adjacency
            .into_par_iter()
            .enumerate()
            .try_for_each(|(id, neighbors)| {
                index
                    .provider()
                    .neighbors()
                    .set_neighbors_sync(id, &neighbors)
            })?;
        Ok(())
    })?;
    index.provider().set_start_points(start_points.row_iter())?;
    let start_ids = index.provider().starting_points()?;
    anyhow::ensure!(
        start_ids.len() == start_neighbors.len(),
        "PiPNN provider created {} start slots for {} start vectors",
        start_ids.len(),
        start_neighbors.len()
    );
    for (start_id, neighbors) in start_ids.into_iter().zip(&start_neighbors) {
        index
            .provider()
            .neighbors()
            .set_neighbors_sync(start_id as usize, neighbors)?;
    }

    let total_time = MicroSeconds::from(batch_elapsed + install_started.elapsed());
    let stats = BuildStats {
        kind: BuildKind::PiPNN,
        total_time,
        vectors_inserted: npoints,
        insert_latencies: None,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) insert_latencies: Option<percentiles::Percentiles<MicroSeconds>>,
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
            insert_latencies: Some(percentiles::compute_percentiles(&mut latencies)?),
        })
    }
}

impl std::fmt::Display for BuildStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index Build Time: {}s", self.total_time.as_seconds())?;
        writeln!(f, "Vectors Inserted: {}", self.vectors_inserted)?;
        writeln!(f, "Kind: {}", self.kind)?;
        if let Some(latencies) = &self.insert_latencies {
            write!(
                f,
                "Insert Latencies:\n  average: {}us\n      p90: {}\n      p99: {}\n\n",
                latencies.mean, latencies.p90, latencies.p99,
            )
        } else {
            writeln!(f, "Insert Latencies: not measured for batch construction\n")
        }
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

#[cfg(all(test, feature = "pipnn"))]
mod pipnn_tests {
    use super::*;
    use diskann::graph::AdjacencyList;

    fn assert_vectors_edges_and_start_points_are_installed(threads: usize) {
        // Given: coordinates i² have increasing gaps, so k=1 selects the previous
        // point for every i>0; point 0 selects 1. Reverse edges form a chain.
        // Seventeen rows cross the provider's sixteen-ID write-lock group.
        let coordinates: [f32; 17] = std::array::from_fn(|id| (id * id) as f32);
        let expected_neighbors: [Vec<u32>; 17] = std::array::from_fn(|id| {
            (0..coordinates.len())
                .filter(|&other| id.abs_diff(other) == 1)
                .map(|other| other as u32)
                .collect()
        });
        let input: IndexBuild = serde_json::from_value(serde_json::json!({
            "data_type": "float32",
            "data": "unused.fbin",
            "distance": "squared_l2",
            "max_degree": 2,
            "l_build": 8,
            "start_point_strategy": "first_vector",
            "alpha": 1.2,
            "backedge_ratio": 1.0,
            "num_threads": threads,
            "multi_insert": null,
            "save_path": null,
            "build_algorithm": {
                "algorithm": "PiPNN",
                "c_max": 17,
                "c_min": 1,
                "p_samp": 0.5,
                "fanout": [2],
                "k": 1,
                "replicas": 1
            }
        }))
        .unwrap();
        let diskann_disk::BuildAlgorithm::PiPNN(parameters) = input.build_algorithm() else {
            panic!("expected PiPNN parameters");
        };
        let mut data = Matrix::new(0.0_f32, coordinates.len(), 1);
        data.as_mut_slice().copy_from_slice(&coordinates);

        // When
        let (index, stats) = pipnn_build(Arc::new(data), &input, parameters).unwrap();

        // Then: every real row contains its own vector and adjacency.
        assert_eq!(stats.vectors_inserted, coordinates.len());
        assert!(stats.insert_latencies.is_none());
        for (id, (&coordinate, expected)) in coordinates.iter().zip(&expected_neighbors).enumerate()
        {
            // SAFETY: id is a real dataset row; installation finished before return.
            let vector = unsafe { index.data_provider.base_vectors.get_vector_sync(id) };
            assert_eq!(vector, &[coordinate], "vector id={id}, threads={threads}");
            let mut neighbors = AdjacencyList::new();
            index
                .provider()
                .neighbors()
                .get_neighbors_sync(id, &mut neighbors)
                .unwrap();
            assert_eq!(
                &*neighbors, expected,
                "adjacency id={id}, threads={threads}"
            );
        }
        let starts = index.provider().starting_points().unwrap();
        assert_eq!(starts.len(), 1);
        // SAFETY: starting_points returns installed frozen IDs and no writer remains.
        let start = unsafe {
            index
                .data_provider
                .base_vectors
                .get_vector_sync(starts[0] as usize)
        };
        assert_eq!(start, &[coordinates[0]]);
        let mut start_neighbors = AdjacencyList::new();
        index
            .provider()
            .neighbors()
            .get_neighbors_sync(starts[0] as usize, &mut start_neighbors)
            .unwrap();
        assert_eq!(&*start_neighbors, &expected_neighbors[0]);
    }

    #[test]
    fn one_thread_installs_vectors_edges_and_requested_start_point() {
        assert_vectors_edges_and_start_points_are_installed(1);
    }

    #[test]
    fn four_threads_install_vectors_edges_and_requested_start_point() {
        assert_vectors_edges_and_start_points_are_installed(4);
    }
}
