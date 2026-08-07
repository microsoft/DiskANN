/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Write a PiPNN graph in the DiskANN disk-index format.
//!
//! This adapter checks dataset metadata and loads one contiguous matrix. It runs
//! PiPNN in the supplied Rayon pool. It computes the start point with the sampled
//! medoid policy. It then writes the common graph header and adjacency layout.
//!
//! PiPNN and Vamana use the same disk graph format.

use diskann::graph::pipnn::{PiPNNBuildContext, PiPNNConfig};
use diskann::{utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    storage::{save_adjacency_graph, StorageReadProvider, StorageWriteProvider},
    utils::{find_medoid_with_sampling, RayonThreadPoolRef, MAX_MEDOID_SAMPLE_SIZE},
};
use diskann_utils::io::{read_bin, Metadata};

use super::{u32_try_from, DiskIndexBuilder};
use crate::data_model::GraphDataType;

/// Build PiPNN adjacency and persist it through the canonical disk graph writer.
pub(super) fn build_graph<Data, StorageProvider>(
    builder: &DiskIndexBuilder<'_, Data, StorageProvider>,
    pool: RayonThreadPoolRef<'_>,
    config: PiPNNConfig,
) -> ANNResult<()>
where
    Data: GraphDataType<VectorIdType = u32>,
    Data::VectorDataType: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let data_path = builder.index_writer.get_dataset_file();
    // Check metadata before the code loads the full matrix. Report a configuration
    // error instead of a matrix-shape error.
    let (points, dimensions) =
        Metadata::read(&mut builder.storage_provider.open_reader(&data_path)?)?.into_dims();
    if dimensions != builder.index_configuration.dim {
        return Err(ANNError::message(format!(
            "configured dimension {} does not match dataset dimension {dimensions}",
            builder.index_configuration.dim
        )));
    }
    if points != builder.index_configuration.max_points {
        return Err(ANNError::message(format!(
            "configured point count {} does not match dataset point count {points}",
            builder.index_configuration.max_points
        )));
    }

    // PiPNN requires one contiguous matrix. Partition and leaf work use the
    // supplied Rayon pool.
    let data =
        read_bin::<Data::VectorDataType>(&mut builder.storage_provider.open_reader(&data_path)?)?;
    let context = PiPNNBuildContext::new(
        config,
        &builder.index_configuration.config,
        builder.index_configuration.dist_metric,
        pool.as_rayon(),
    )?;
    let adjacency = diskann::graph::pipnn::build_graph(data.as_view(), &context)?;

    // The disk header requires a start point. Use the same sampled medoid policy
    // as the Vamana disk builder.
    let mut rng = diskann_providers::utils::create_rnd_from_optional_seed(
        builder.index_configuration.random_seed,
    );
    let (_, start_id) = find_medoid_with_sampling::<Data::VectorDataType, _>(
        &data_path,
        builder.storage_provider,
        MAX_MEDOID_SAMPLE_SIZE,
        &mut rng,
    )?;
    save_adjacency_graph(
        &adjacency,
        u32_try_from(builder.index_configuration.config.pruned_degree().get())?,
        builder.storage_provider,
        u32_try_from(start_id)?,
        &builder.index_writer.get_mem_index_file(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use diskann::{graph::config, utils::ONE};
    use diskann_providers::utils::create_thread_pool;
    use diskann_providers::{
        model::IndexConfiguration,
        storage::{
            get_disk_index_file, StorageReadProvider, StorageWriteProvider, VirtualStorageProvider,
        },
    };
    use diskann_utils::{io::write_bin, views::MatrixView};
    use diskann_vector::distance::Metric;
    use vfs::MemoryFS;

    use crate::{
        build::{
            builder::build::DiskIndexBuilder,
            configuration::{MemoryBudget, NumPQChunks, PiPNNParameters},
        },
        data_model::AdHoc,
        storage::DiskIndexWriter,
        DiskIndexBuildParameters,
    };

    fn pipnn() -> PiPNNParameters {
        PiPNNParameters {
            c_max: 512,
            c_min: 64,
            p_samp: 0.01,
            fanout: vec![10, 3],
            k: 2,
            replicas: 1,
        }
    }

    fn write_data(storage: &VirtualStorageProvider<MemoryFS>, points: usize, dimensions: usize) {
        let data: Vec<f32> = (0..points * dimensions)
            .map(|index| ((index * 17) % 251) as f32)
            .collect();
        write_bin(
            MatrixView::try_from(data.as_slice(), points, dimensions).unwrap(),
            &mut storage.create_for_write("/data.fbin").unwrap(),
        )
        .unwrap();
    }

    fn graph_config(degree: usize, alpha: f32) -> diskann::graph::Config {
        config::Builder::new_with(
            degree,
            config::MaxDegree::default_slack(),
            50,
            Metric::L2.into(),
            |builder| {
                builder.alpha(alpha);
            },
        )
        .build()
        .unwrap()
    }

    fn builder<'a>(
        storage: &'a VirtualStorageProvider<MemoryFS>,
        points: usize,
        dimensions: usize,
        budget_gib: f64,
        alpha: f32,
        parameters: PiPNNParameters,
    ) -> DiskIndexBuilder<'a, AdHoc<f32>, VirtualStorageProvider<MemoryFS>> {
        let params = DiskIndexBuildParameters::new_pipnn(
            MemoryBudget::try_from_gb(budget_gib).unwrap(),
            NumPQChunks::new_with(dimensions, dimensions).unwrap(),
            parameters,
        );
        let config = IndexConfiguration::new(
            Metric::L2,
            dimensions,
            points,
            ONE,
            1,
            graph_config(32, alpha),
        )
        .with_pseudo_rng_from_seed(42);
        let writer =
            DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
        DiskIndexBuilder::new(storage, params, config, writer).unwrap()
    }

    #[test]
    fn disk_build_rejects_dataset_shape_mismatch() {
        let storage = VirtualStorageProvider::new_memory();
        write_data(&storage, 2, 8);
        let params = DiskIndexBuildParameters::new_pipnn(
            MemoryBudget::try_from_gb(10_000.0).unwrap(),
            NumPQChunks::new_with(4, 4).unwrap(),
            PiPNNParameters::default(),
        );
        let config = IndexConfiguration::new(Metric::L2, 4, 3, ONE, 1, graph_config(4, 1.2));
        let writer =
            DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();
        let mut builder =
            DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, config, writer).unwrap();

        let error = builder.build().unwrap_err();
        assert!(format!("{error:?}").contains("configured dimension 4"));
        assert!(storage.exists("/index_pq_compressed.bin"));
    }

    #[test]
    fn graph_adapter_rejects_point_count_mismatch() {
        let storage = VirtualStorageProvider::new_memory();
        write_data(&storage, 2, 8);
        let parameters = pipnn();
        let builder = builder(&storage, 3, 8, 1.0, 1.2, parameters.clone());
        let pool = create_thread_pool(1).unwrap();

        let error = super::build_graph(&builder, pool.as_ref(), (&parameters).into()).unwrap_err();
        assert!(format!("{error:?}").contains("configured point count 3"));
        assert!(!storage.exists(&builder.index_writer.get_mem_index_file()));
    }

    #[test]
    fn graph_adapter_writes_degree_medoid_and_frozen_count() {
        let storage = VirtualStorageProvider::new_memory();
        let (points, dimensions) = (256, 8);
        write_data(&storage, points, dimensions);
        let parameters = pipnn();
        let builder = builder(&storage, points, dimensions, 1.0, 1.2, parameters.clone());
        let pool = create_thread_pool(1).unwrap();

        super::build_graph(&builder, pool.as_ref(), (&parameters).into()).unwrap();

        let mut header = [0_u8; 24];
        std::io::Read::read_exact(
            &mut storage
                .open_reader(&builder.index_writer.get_mem_index_file())
                .unwrap(),
            &mut header,
        )
        .unwrap();
        assert_eq!(u32::from_le_bytes(header[8..12].try_into().unwrap()), 32);
        assert!(u32::from_le_bytes(header[12..16].try_into().unwrap()) < points as u32);
        assert_eq!(u64::from_le_bytes(header[16..24].try_into().unwrap()), 0);
    }

    #[test]
    fn explicit_selection_ignores_the_vamana_memory_strategy() {
        let storage = VirtualStorageProvider::new_memory();
        let (points, dimensions) = (256, 8);
        write_data(&storage, points, dimensions);
        let mut builder = builder(&storage, points, dimensions, 0.000001, 1.3, pipnn());

        assert!(matches!(
            builder.disk_build_param.build_algorithm(),
            crate::BuildAlgorithm::PiPNN(_)
        ));
        assert_eq!(
            builder.disk_build_param.build_quantization(),
            &crate::QuantizationType::FP
        );
        assert_eq!(builder.index_configuration.config.pruned_degree().get(), 32);
        assert_eq!(builder.index_configuration.config.l_build().get(), 50);
        assert_eq!(builder.index_configuration.config.alpha(), 1.3);
        builder.build().unwrap();
        assert!(storage.exists(&get_disk_index_file("/index")));
        assert!(storage.exists("/index_pq_compressed.bin"));
    }

    #[test]
    fn builder_rejects_invalid_pipnn_config() {
        let storage = VirtualStorageProvider::new_memory();
        let invalid = PiPNNParameters {
            c_max: 0,
            ..PiPNNParameters::default()
        };
        let params = DiskIndexBuildParameters::new_pipnn(
            MemoryBudget::try_from_gb(0.0001).unwrap(),
            NumPQChunks::new_with(1, 1).unwrap(),
            invalid,
        );
        let config = IndexConfiguration::new(Metric::L2, 1, 1, ONE, 1, graph_config(4, 1.2));
        let writer =
            DiskIndexWriter::new("/data.fbin".into(), "/index".into(), None, 4096).unwrap();

        let error = match DiskIndexBuilder::<AdHoc<f32>, _>::new(&storage, params, config, writer) {
            Ok(_) => panic!("invalid PiPNN config must be rejected"),
            Err(error) => error,
        };

        assert!(format!("{error:?}").contains("c_max must be greater than zero"));
    }
}
