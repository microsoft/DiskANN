/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph build of a disk index.
//!
//! PiPNN replaces the Vamana graph build and writes the same in-memory index
//! file, so the PQ and disk layout steps of the disk build do not change.

use diskann::{
    ANNResult,
    graph::{AdjacencyList, pipnn},
    utils::VectorRepr,
};
use diskann_providers::{
    model::IndexConfiguration,
    storage::{StorageReadProvider, StorageWriteProvider, save_adjacency_graph},
    utils::RayonThreadPoolRef,
};
use diskann_utils::{io::read_bin, views::rowmajor::Matrix};

use super::start_point_from_medoid;
use crate::{
    PiPNNParameters, QuantizationType,
    error::{ErrorKind, diskann_error},
};

/// Make the PiPNN configuration of a disk build. The index configuration
/// supplies the prune policy and the seed.
///
/// # Errors
///
/// Returns an error if `parameters` are invalid, or if `quantization` is not
/// [`QuantizationType::FP`]: PiPNN reads the full-precision rows and does not
/// use a build quantizer.
pub(super) fn config(
    parameters: &PiPNNParameters,
    index_configuration: &IndexConfiguration,
    quantization: &QuantizationType,
) -> ANNResult<pipnn::Config> {
    if !matches!(quantization, QuantizationType::FP) {
        return Err(diskann_error!(
            ErrorKind::IndexConfigError("build_quantization"),
            "PiPNN reads full-precision rows and needs FP build quantization, not {quantization}"
        ));
    }
    let mut builder = parameters.builder(&index_configuration.config);
    if let Some(seed) = index_configuration.random_seed {
        builder = builder.seed(seed);
    }
    Ok(builder.build()?)
}

/// Build the PiPNN graph of the dataset at `data_path` and write it to
/// `save_path` as an in-memory index file.
///
/// The start point is the medoid of a storage sample, as in the Vamana build.
/// The file records the pruned degree as the maximum degree and no frozen
/// points.
pub(super) fn build_inmem_index<T, StorageProvider>(
    index_configuration: &IndexConfiguration,
    config: &pipnn::Config,
    data_path: &str,
    save_path: &str,
    storage_provider: &StorageProvider,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<()>
where
    T: VectorRepr + pipnn::Element,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let start_point = start_point_from_medoid::<T, _>(
        data_path,
        index_configuration.random_seed,
        storage_provider,
    )?
    .id();
    let graph = build_graph::<T, _>(
        index_configuration,
        config,
        data_path,
        storage_provider,
        pool,
    )?;
    save_adjacency_graph(
        &graph,
        index_configuration.config.pruned_degree_u32().get(),
        storage_provider,
        start_point,
        save_path,
    )?;
    Ok(())
}

/// Read the dataset at `data_path` and build its PiPNN graph on `pool`.
///
/// The dataset is dropped on return, so only the graph stays in memory while
/// it is written.
fn build_graph<T, StorageProvider>(
    index_configuration: &IndexConfiguration,
    config: &pipnn::Config,
    data_path: &str,
    storage_provider: &StorageProvider,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<Vec<AdjacencyList<u32>>>
where
    T: VectorRepr + pipnn::Element,
    StorageProvider: StorageReadProvider,
{
    let data = read_bin::<T>(&mut storage_provider.open_reader(data_path)?)?;
    // PiPNN builds one graph row for each dataset row. The other build steps
    // size their work by the index configuration, so the shapes must agree.
    let (points, dimensions) = (data.nrows(), data.ncols());
    if (points, dimensions) != (index_configuration.max_points, index_configuration.dim) {
        return Err(diskann_error!(
            ErrorKind::IndexError,
            "the dataset has {points} points of {dimensions} dimensions, but the index \
             configuration has {} points of {} dimensions",
            index_configuration.max_points,
            index_configuration.dim
        ));
    }
    let rows = pipnn::Rows::new(data.as_view(), index_configuration.dist_metric)?;
    pool.install(|| pipnn::build_graph(&rows, config))
}

#[cfg(test)]
mod tests {
    use diskann::{graph::config, utils::ONE};
    use diskann_providers::{storage::VirtualStorageProvider, utils::create_thread_pool};
    use diskann_utils::{io::write_bin, views::rowmajor};
    use diskann_vector::distance::Metric;
    use vfs::MemoryFS;

    use super::*;

    const DATA: &str = "/data.fbin";
    const GRAPH: &str = "/graph";
    const DEGREE: usize = 32;
    const SEED: u64 = 42;

    fn index_configuration(points: usize, dimensions: usize) -> IndexConfiguration {
        let graph = config::Builder::new(
            DEGREE,
            config::MaxDegree::default_slack(),
            50,
            Metric::L2.into(),
        )
        .build()
        .unwrap();
        IndexConfiguration::new(Metric::L2, dimensions, points, ONE, 1, graph)
            .with_pseudo_rng_from_seed(SEED)
    }

    fn storage_with_data(points: usize, dimensions: usize) -> VirtualStorageProvider<MemoryFS> {
        let storage = VirtualStorageProvider::new_memory();
        let data: Vec<f32> = (0..points * dimensions)
            .map(|index| ((index * 17) % 251) as f32)
            .collect();
        write_bin(
            rowmajor::Ref::try_from_data(data.as_slice(), points, dimensions).unwrap(),
            &mut storage.create_for_write(DATA).unwrap(),
        )
        .unwrap();
        storage
    }

    fn build(
        storage: &VirtualStorageProvider<MemoryFS>,
        index_configuration: &IndexConfiguration,
    ) -> ANNResult<()> {
        let pipnn_config = config(
            &PiPNNParameters::default(),
            index_configuration,
            &QuantizationType::FP,
        )?;
        let pool = create_thread_pool(1)?;
        build_inmem_index::<f32, _>(
            index_configuration,
            &pipnn_config,
            DATA,
            GRAPH,
            storage,
            pool.as_ref(),
        )
    }

    #[test]
    fn config_takes_the_index_seed_and_rejects_build_quantization() {
        let index_configuration = index_configuration(1, 1);
        let parameters = PiPNNParameters::default();

        let pipnn_config = config(&parameters, &index_configuration, &QuantizationType::FP);
        assert_eq!(pipnn_config.unwrap().seed(), SEED);

        let quantized = QuantizationType::PQ { num_chunks: 1 };
        let error = config(&parameters, &index_configuration, &quantized).unwrap_err();
        assert!(error.to_string().contains("needs FP"), "{error}");
    }

    #[test]
    fn build_rejects_a_dataset_that_does_not_match_the_configuration() {
        let (points, dimensions) = (2, 8);
        for (configured_points, configured_dimensions) in [(3, dimensions), (points, 4)] {
            let storage = storage_with_data(points, dimensions);
            let index_configuration = index_configuration(configured_points, configured_dimensions);

            let error = build(&storage, &index_configuration).unwrap_err();

            assert!(error.to_string().contains("has 2 points of 8"), "{error}");
            assert!(!storage.exists(GRAPH));
        }
    }

    #[test]
    fn build_writes_the_pruned_degree_and_the_medoid() {
        let (points, dimensions) = (256, 8);
        let storage = storage_with_data(points, dimensions);

        build(&storage, &index_configuration(points, dimensions)).unwrap();

        // Header: file size (u64), maximum degree (u32), start point (u32),
        // frozen points (u64).
        let mut header = [0_u8; 24];
        std::io::Read::read_exact(&mut storage.open_reader(GRAPH).unwrap(), &mut header).unwrap();
        let medoid = start_point_from_medoid::<f32, _>(DATA, Some(SEED), &storage)
            .unwrap()
            .id();
        assert_eq!(header[8..12], (DEGREE as u32).to_le_bytes());
        assert_eq!(header[12..16], medoid.to_le_bytes());
        assert_eq!(header[16..24], 0_u64.to_le_bytes());
    }
}
