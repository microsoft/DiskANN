/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
*/

use diskann::{
    graph::config,
    utils::{IntoUsize, ONE},
    ANNError, ANNResult,
};
use diskann_disk::{
    build::{
        builder::build::DiskIndexBuilder,
        disk_index_build_parameter::{DiskIndexBuildParameters, MemoryBudget, NumPQChunks},
    },
    storage::DiskIndexWriter,
    QuantizationType,
};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    model::{
        graph::{provider::async_::PrefetchCacheLineLevel, traits::GraphDataType},
        IndexConfiguration,
    },
    storage::{get_pq_pivot_file, FileStorageProvider},
    utils::Timer,
};
use diskann_utils::io::Metadata;
use diskann_vector::distance::Metric;
use pyo3::prelude::*;

use crate::{
    utils::{
        ann_result_py::*, DataType, GraphDataF32Vector, GraphDataInt8Vector, GraphDataU8Vector,
    },
    MetricPy,
};

// Disk sector length in bytes. This is used as the offset alignment and smallest block size when reading/writing index data from/to disk.
pub const DISK_SECTOR_LEN: usize = 4096;

/// The main function to build a disk index
#[allow(clippy::too_many_arguments)]
fn build_index<Data>(
    metric: Metric,
    data_path: &str,
    r: u32,
    l: u32,
    index_path_prefix: &str,
    num_threads: u32,
    index_build_ram_limit_gb: f64,
    build_pq_bytes: usize,
    alpha: f32,
    num_of_pq_chunks: usize,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32> + 'static,
{
    let storage_provider = &FileStorageProvider;

    let build_quantization_type = if build_pq_bytes > 0 {
        QuantizationType::PQ {
            num_chunks: build_pq_bytes,
        }
    } else {
        QuantizationType::FP
    };

    if num_of_pq_chunks == 0 {
        return Err(ANNError::log_index_config_error(
            "num_of_pq_chunks".to_string(),
            "Number of PQ chunks must be > 0".to_string(),
        ));
    }

    let config = config::Builder::new_with(
        r.into_usize(),
        config::MaxDegree::default_slack(),
        l.into_usize(),
        metric.into(),
        |b| {
            b.alpha(alpha).saturate_after_prune(true);
        },
    )
    .build()?;

    let metadata = Metadata::read(&mut storage_provider.open_reader(data_path)?)?;

    let memory_budget = MemoryBudget::try_from_gb(index_build_ram_limit_gb)?;
    let num_pq_chunks = NumPQChunks::new_with(num_of_pq_chunks, metadata.ndims())?;

    let disk_index_build_parameters =
        DiskIndexBuildParameters::new(memory_budget, build_quantization_type, num_pq_chunks);

    let config = IndexConfiguration::new(
        metric,
        metadata.ndims(),
        metadata.npoints(),
        ONE,
        num_threads.into_usize(),
        config,
    )
    .with_prefetch_cache_line_level(Some(PrefetchCacheLineLevel::All));

    let disk_index_writer = DiskIndexWriter::new(
        data_path.to_string(),
        index_path_prefix.to_string(),
        Option::None,
        DISK_SECTOR_LEN,
    )?;

    // Delete the PQ pivot file if it exists to avoid using stale data
    let pq_pivot_file = get_pq_pivot_file(index_path_prefix);
    if storage_provider.exists(&pq_pivot_file) {
        storage_provider.delete(&pq_pivot_file).map_err(|e| {
            ANNError::log_index_error(format_args!(
                "Failed to delete PQ pivot file: {}, err: {:?}",
                pq_pivot_file, e
            ))
        })?;
    }

    let mut disk_index = DiskIndexBuilder::<Data, FileStorageProvider>::new(
        storage_provider,
        disk_index_build_parameters,
        config,
        disk_index_writer,
    )?;

    let timer = Timer::new();
    disk_index.build()?;
    let diff = timer.elapsed();
    println!("Indexing time: {} seconds", diff.as_secs_f64());

    Ok(())
}

/// The main function to build a disk index
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn build_disk_index(
    data_type: DataType,
    distance_metric: MetricPy,
    data_path: String,
    index_path_prefix: String,
    graph_degree: u32,
    complexity: u32,
    build_dram_budget: f64,
    num_threads: u32,
    build_pq_bytes: usize,
    alpha: f32,
    num_of_pq_chunks: usize,
) -> ANNResultPy<()> {
    println!(
         "Starting index build with R: {}  Lbuild: {}  alpha: {}  #threads: {} num_of_pq_chunks: {} build_DRAM_budget: {}",
         graph_degree, complexity, alpha, num_threads, num_of_pq_chunks, build_dram_budget
     );

    let err = match data_type {
        DataType::Int8 => build_index::<GraphDataInt8Vector>(
            distance_metric.into(),
            &data_path,
            graph_degree,
            complexity,
            &index_path_prefix,
            num_threads,
            build_dram_budget,
            build_pq_bytes,
            alpha,
            num_of_pq_chunks,
        ),
        DataType::Uint8 => build_index::<GraphDataU8Vector>(
            distance_metric.into(),
            &data_path,
            graph_degree,
            complexity,
            &index_path_prefix,
            num_threads,
            build_dram_budget,
            build_pq_bytes,
            alpha,
            num_of_pq_chunks,
        ),
        DataType::Float => build_index::<GraphDataF32Vector>(
            distance_metric.into(),
            &data_path,
            graph_degree,
            complexity,
            &index_path_prefix,
            num_threads,
            build_dram_budget,
            build_pq_bytes,
            alpha,
            num_of_pq_chunks,
        ),
    };

    match err {
        Ok(_) => {
            println!("Index build completed successfully");
            Ok(())
        }
        Err(err) => {
            eprintln!("Error: {:?}", err);
            Err(ANNErrorPy::new(err))
        }
    }
}

#[cfg(test)]
mod build_disk_index_test {
    use super::build_disk_index;

    const TEST_DATA_PATH: &str = "../diskann/tests/data/siftsmall_learn.bin";

    // Building disk index with PQ takes too long (>5 minutes), so we're ignoring this test
    #[ignore]
    #[test]
    fn test_build_disk_index_with_pq() {
        const TEST_DIRECTORY: &str = "tests/data/ann";
        build_disk_index(
            crate::utils::DataType::Float,
            super::MetricPy::L2,
            String::from(TEST_DATA_PATH),
            String::from(TEST_DIRECTORY),
            10,
            10,
            0.1,
            1,
            0,
            1.2f32,
            1,
        )
        .unwrap();
    }
}
