/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builder tests.
#[cfg(test)]
mod disk_index_build_tests {
    use std::io::{Read, Write};

    use crate::test_utils::{GraphDataF32VectorUnitData, GraphDataMinMaxVectorUnitData};
    use diskann_providers::storage::{
        get_compressed_pq_file, get_pq_pivot_file, StorageReadProvider, StorageWriteProvider,
    };
    use diskann_utils::io::Metadata;
    use diskann_vector::distance::Metric;
    use rstest::rstest;

    use crate::{
        build::builder::core::disk_index_builder_tests::{
            new_vfs, verify_search_result_with_ground_truth, IndexBuildFixture, TestParams,
        },
        QuantizationType, SphericalBits,
    };

    #[derive(PartialEq)]
    enum BuildType {
        AsyncFP,
        AsyncSQ1Bit,
        AsyncSpherical1Bit,
        AsyncPQ,
    }

    #[rstest]
    pub fn test_disk_index_builder(
        #[values(false, true)] use_sharded_build: bool,
        #[values(
            BuildType::AsyncFP,
            BuildType::AsyncSQ1Bit,
            BuildType::AsyncSpherical1Bit,
            BuildType::AsyncPQ
        )]
        build_type: BuildType,
    ) {
        let index_path_prefix = "/disk_index_build/test_disk_index_build".to_string();

        run_disk_index_builder_test(index_path_prefix, use_sharded_build, build_type);
    }

    #[rstest]
    #[case::empty_dataset(true, 10, "Cannot generate compressed data for an empty dataset")]
    #[case::zero_batch(
        false,
        0,
        "Data compression chunk vector count must be greater than zero"
    )]
    fn invalid_compression_input_preserves_search_pq_outputs(
        #[case] empty_dataset: bool,
        #[case] batch_size: usize,
        #[case] expected_error: &str,
    ) {
        let storage = new_vfs();
        let mut params = TestParams {
            index_path_prefix: "/invalid_compression_input".into(),
            data_compression_chunk_vector_count: Some(batch_size),
            ..TestParams::default()
        };
        if empty_dataset {
            params.data_path = "/empty_compression_input.bin".into();
            Metadata::new(0, params.dim)
                .unwrap()
                .write(&mut storage.create_for_write(&params.data_path).unwrap())
                .unwrap();
        }

        let outputs = [
            (
                get_pq_pivot_file(&params.index_path_prefix),
                b"existing search PQ pivots".as_slice(),
            ),
            (
                get_compressed_pq_file(&params.index_path_prefix),
                b"existing compressed vectors".as_slice(),
            ),
        ];
        for (path, bytes) in &outputs {
            storage
                .create_for_write(path)
                .unwrap()
                .write_all(bytes)
                .unwrap();
        }
        // FP isolates disk-search PQ validation from graph-quantizer training.
        let fixture = IndexBuildFixture::new(storage, params).unwrap();
        let error = fixture.build::<GraphDataF32VectorUnitData>().unwrap_err();
        assert!(error.to_string().contains(expected_error), "{error}");

        for (path, expected_bytes) in outputs {
            let mut actual_bytes = Vec::new();
            fixture
                .storage_provider
                .open_reader(&path)
                .unwrap()
                .read_to_end(&mut actual_bytes)
                .unwrap();
            assert_eq!(actual_bytes, expected_bytes);
        }
    }

    #[rstest]
    fn test_spherical_disk_index_builder_with_metric(
        #[values(Metric::InnerProduct, Metric::Cosine)] metric: Metric,
    ) {
        let index_path_prefix = format!(
            "/disk_index_build/test_spherical_disk_index_build_{:?}_async_SPHERICAL_1",
            metric
        );
        let params = TestParams {
            l_build: 64,
            max_degree: 16,
            index_path_prefix,
            data_compression_chunk_vector_count: Some(10),
            build_quantization_type: QuantizationType::Spherical(SphericalBits::One),
            metric,
            ..TestParams::default()
        };
        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        fixture.build::<GraphDataF32VectorUnitData>().unwrap();

        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            8,
            130,
            &fixture.storage_provider,
        )
        .unwrap();
    }

    // Helper function to run the tests with consistent behavior
    fn run_disk_index_builder_test(
        index_path_prefix: String,
        use_sharded_build: bool,
        build_type: BuildType,
    ) {
        match build_type {
            BuildType::AsyncFP => {
                run_test(
                    index_path_prefix,
                    QuantizationType::FP,
                    use_sharded_build,
                    10, // top_k
                    32, // search_l
                );
            }
            BuildType::AsyncSQ1Bit => {
                run_test(
                    index_path_prefix,
                    QuantizationType::SQ {
                        nbits: 1,
                        standard_deviation: None,
                    },
                    use_sharded_build,
                    8,   // top_k
                    130, // search_l
                );
            }
            BuildType::AsyncSpherical1Bit => {
                run_test(
                    index_path_prefix,
                    QuantizationType::Spherical(SphericalBits::One),
                    use_sharded_build,
                    8,   // top_k
                    130, // search_l
                );
            }
            BuildType::AsyncPQ => {
                run_test(
                    index_path_prefix,
                    QuantizationType::PQ { num_chunks: 32 },
                    use_sharded_build,
                    10,  // top_k
                    100, // search_l
                );
            }
        }
    }

    fn run_test(
        index_path_prefix: String,
        build_quantization_type: QuantizationType,
        use_sharded_build: bool,
        top_k: usize,
        search_l: u32,
    ) {
        // Use the same parameters from [test_sift_build_and_search] in diskann_index
        let l_build = 64;
        let max_degree = 16;

        let index_path_prefix = format!(
            "{}_sharded{}_async_{}",
            index_path_prefix, use_sharded_build, build_quantization_type
        );
        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix: index_path_prefix.clone(),
            index_build_ram_gb: get_index_build_ram_gb(use_sharded_build),
            data_compression_chunk_vector_count: Some(10),
            build_quantization_type,
            ..TestParams::default()
        };

        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        fixture.build::<GraphDataF32VectorUnitData>().unwrap();

        fixture.compare_pq_compressed_files();
        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            top_k,
            search_l,
            &fixture.storage_provider,
        )
        .unwrap();
    }

    ///////////////////////////////
    // MinMax Integration Tests //
    //////////////////////////////

    #[rstest]
    pub fn test_disk_minmax_index_builder(
        #[values(false, true)] use_sharded_build: bool,
    ) -> anyhow::Result<()> {
        const DATA_FILE: &str = "/sift/siftsmall_learn_256pts_minmax.fbin";
        let index_path_prefix = "/disk_index_build/test_minmax_disk_index_build".to_string();
        let params = TestParams {
            dim: 148,
            full_dim: 128,
            data_path: DATA_FILE.to_string(),
            ..TestParams::default()
        };
        run_minmax_test(index_path_prefix, params, use_sharded_build, 10, 32);
        Ok(())
    }

    fn run_minmax_test(
        index_path_prefix: String,
        params: TestParams,
        use_sharded_build: bool,
        top_k: usize,
        search_l: u32,
    ) {
        // Use the same parameters from [test_sift_build_and_search] in diskann_index
        let l_build = 64;
        let max_degree = 16;

        let index_path_prefix =
            format!("{}_minmax_sharded={}", index_path_prefix, use_sharded_build,);
        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix: index_path_prefix.clone(),
            index_build_ram_gb: get_index_build_ram_gb(use_sharded_build),
            build_quantization_type: QuantizationType::FP,
            ..params
        };

        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        fixture.build::<GraphDataMinMaxVectorUnitData>().unwrap();

        verify_search_result_with_ground_truth::<GraphDataMinMaxVectorUnitData>(
            &fixture.params,
            top_k,
            search_l,
            &fixture.storage_provider,
        )
        .unwrap();
    }

    fn get_index_build_ram_gb(use_sharded_build: bool) -> f64 {
        if use_sharded_build {
            0.0001 // small enough to trigger sharded build.
        } else {
            1.0
        }
    }
}
