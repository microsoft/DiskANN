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
use diskann_vector::distance::Metric;

use crate::error::{ErrorKind, diskann_error};

/// Build the PiPNN graph of the dataset at `data_path` and write it to
/// `save_path` as an in-memory index file.
///
/// The caller's `start_id` is recorded in the graph header.
pub(super) fn build_inmem_index<T, StorageProvider>(
    metric: Metric,
    start_id: u32,
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
    let graph = build_graph::<T, _>(metric, config, data_path, storage_provider, pool)?;
    save_graph(&graph, config, start_id, save_path, storage_provider)
}

/// Read the dataset at `data_path` and build its PiPNN graph on `pool`.
///
/// The dataset is dropped on return, so only the graph stays in memory while
/// it is written.
fn build_graph<T, StorageProvider>(
    metric: Metric,
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
    let rows = pipnn::Rows::new(data.as_view(), metric)?;
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

    const DATA: &str = "/data.fbin";
    const GRAPH: &str = "/graph";
    const DEGREE: usize = 32;
    const SEED: u64 = 42;

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

    fn build(storage: &VirtualStorageProvider<MemoryFS>, start_id: u32) -> ANNResult<()> {
        let pool = create_thread_pool(1)?;
        build_inmem_index::<f32, _>(
            Metric::L2,
            start_id,
            &pipnn_config(),
            DATA,
            GRAPH,
            storage,
            pool.as_ref(),
        )
    }

    #[test]
    fn build_writes_the_graph_in_the_inmem_index_format() {
        let (points, dimensions) = (256, 8);
        let storage = storage_with_data(points, dimensions);

        let start_id = 17;
        build(&storage, start_id).unwrap();

        let mut file = Vec::new();
        std::io::Read::read_to_end(&mut storage.open_reader(GRAPH).unwrap(), &mut file).unwrap();
        let word = |index: usize| u32::from_le_bytes(file[4 * index..][..4].try_into().unwrap());

        assert_eq!(
            u64::from_le_bytes(file[..8].try_into().unwrap()),
            file.len() as u64
        );
        assert_eq!(word(2), DEGREE as u32);
        assert_eq!(word(3), start_id);
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
