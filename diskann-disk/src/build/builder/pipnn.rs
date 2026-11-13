/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN graph step of the disk index build.
//!
//! PiPNN replaces the Vamana graph step and writes the same in-memory index
//! file, so the PQ and disk layout steps of the disk build do not change.

use std::io::{BufWriter, Write};

use diskann::{
    ANNResult,
    graph::{AdjacencyList, pipnn},
    utils::VectorRepr,
};
use diskann_providers::{
    storage::{StorageReadProvider, StorageWriteProvider},
    utils::RayonThreadPoolRef,
};
use diskann_utils::{io::read_bin, views::rowmajor::Matrix};

use super::build::start_point_from_medoid;
use crate::{
    build::configuration::DiskIndexBuildConfig,
    error::{ErrorKind, diskann_error},
};

/// Build the PiPNN graph of the dataset at `data_path` and write it to
/// `save_path` as an in-memory index file.
///
/// The start point is the medoid of a storage sample, as in the Vamana build.
pub(super) fn build_inmem_index<T, StorageProvider>(
    index_configuration: &DiskIndexBuildConfig,
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
    let start_point =
        start_point_from_medoid::<T, _>(data_path, index_configuration.seed, storage_provider)?
            .id();
    let graph = build_graph::<T, _>(
        index_configuration,
        config,
        data_path,
        storage_provider,
        pool,
    )?;
    save_graph(&graph, config, start_point, save_path, storage_provider)
}

/// Read the dataset at `data_path` and build its PiPNN graph on `pool`.
///
/// The dataset is dropped on return, so only the graph stays in memory while
/// it is written.
fn build_graph<T, StorageProvider>(
    index_configuration: &DiskIndexBuildConfig,
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
    if (points, dimensions) != (index_configuration.num_points, index_configuration.dim) {
        return Err(diskann_error!(
            ErrorKind::IndexError,
            "the dataset has {points} points of {dimensions} dimensions, but the index \
             configuration has {} points of {} dimensions",
            index_configuration.num_points,
            index_configuration.dim
        ));
    }
    let rows = pipnn::Rows::new(data.as_view(), index_configuration.metric)?;
    pool.install(|| pipnn::build_graph(&rows, config))
}

/// Write `graph` in the in-memory index format, as the merged Vamana build
/// does.
///
/// The header holds the file size (`u64`), the maximum degree (`u32`), the
/// start point (`u32`) and the number of frozen points (`u64`). Each row
/// follows as its length (`u32`) and its neighbor IDs (`u32`). The disk layout
/// sizes each node by the maximum degree, so the header records the PiPNN
/// degree, which bounds every row. PiPNN adds no frozen points.
fn save_graph<StorageProvider>(
    graph: &[AdjacencyList<u32>],
    config: &pipnn::Config,
    start_point: u32,
    save_path: &str,
    storage_provider: &StorageProvider,
) -> ANNResult<()>
where
    StorageProvider: StorageWriteProvider,
{
    let max_degree = u32::try_from(config.degree().get()).map_err(|_| {
        diskann_error!(
            ErrorKind::IndexError,
            "graph degree {} exceeds u32::MAX",
            config.degree()
        )
    })?;
    const HEADER_BYTES: usize = 24;
    let row_bytes: usize = graph
        .iter()
        .map(|row| size_of::<u32>() * (1 + row.len()))
        .sum();
    let file_bytes = (HEADER_BYTES + row_bytes) as u64;

    let mut writer = BufWriter::new(storage_provider.create_for_write(save_path)?);
    writer.write_all(&file_bytes.to_le_bytes())?;
    writer.write_all(&max_degree.to_le_bytes())?;
    writer.write_all(&start_point.to_le_bytes())?;
    writer.write_all(&0_u64.to_le_bytes())?;
    for row in graph {
        writer.write_all(&(row.len() as u32).to_le_bytes())?;
        for neighbor in row.iter() {
            writer.write_all(&neighbor.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use diskann_providers::{storage::VirtualStorageProvider, utils::create_thread_pool};
    use diskann_utils::{io::write_bin, views::rowmajor};
    use diskann_vector::distance::Metric;
    use vfs::MemoryFS;

    use super::*;
    use crate::build::configuration::{GraphBuildConfig, NumPQChunks};

    const DATA: &str = "/data.fbin";
    const GRAPH: &str = "/graph";
    const DEGREE: usize = 32;
    const SEED: u64 = 42;

    fn common_configuration(points: usize, dimensions: usize) -> DiskIndexBuildConfig {
        DiskIndexBuildConfig::new(
            Metric::L2,
            dimensions,
            points,
            1,
            NumPQChunks::new_with(1, dimensions).unwrap(),
            GraphBuildConfig::Pipnn(pipnn_config()),
        )
        .with_seed(SEED)
    }

    fn pipnn_config() -> pipnn::Config {
        pipnn::Config::builder(std::num::NonZeroUsize::new(DEGREE).unwrap())
            .seed(SEED)
            .build()
            .unwrap()
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
        index_configuration: &DiskIndexBuildConfig,
    ) -> ANNResult<()> {
        let pool = create_thread_pool(1)?;
        build_inmem_index::<f32, _>(
            index_configuration,
            &pipnn_config(),
            DATA,
            GRAPH,
            storage,
            pool.as_ref(),
        )
    }

    #[test]
    fn build_rejects_a_dataset_that_does_not_match_the_configuration() {
        let (points, dimensions) = (2, 8);
        for (configured_points, configured_dimensions) in [(3, dimensions), (points, 4)] {
            let storage = storage_with_data(points, dimensions);
            let config = common_configuration(configured_points, configured_dimensions);
            let error = build(&storage, &config).unwrap_err();
            assert!(error.to_string().contains("has 2 points of 8"), "{error}");
            assert!(!storage.exists(GRAPH));
        }
    }

    #[test]
    fn build_writes_the_graph_in_the_inmem_index_format() {
        let (points, dimensions) = (256, 8);
        let storage = storage_with_data(points, dimensions);

        build(&storage, &common_configuration(points, dimensions)).unwrap();

        let mut file = Vec::new();
        std::io::Read::read_to_end(&mut storage.open_reader(GRAPH).unwrap(), &mut file).unwrap();
        let word = |index: usize| u32::from_le_bytes(file[4 * index..][..4].try_into().unwrap());
        let medoid = start_point_from_medoid::<f32, _>(DATA, Some(SEED), &storage)
            .unwrap()
            .id();

        assert_eq!(
            u64::from_le_bytes(file[..8].try_into().unwrap()),
            file.len() as u64
        );
        assert_eq!(word(2), DEGREE as u32);
        assert_eq!(word(3), medoid);
        assert_eq!(u64::from_le_bytes(file[16..24].try_into().unwrap()), 0);
        // Each row is its length and then its neighbor IDs.
        let mut index = 6;
        for _ in 0..points {
            let len = word(index) as usize;
            assert!(len <= DEGREE);
            assert!(
                (index + 1..index + 1 + len).all(|neighbor| (word(neighbor) as usize) < points)
            );
            index += 1 + len;
        }
        assert_eq!(4 * index, file.len());
    }
}
