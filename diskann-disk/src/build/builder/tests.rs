/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builder tests.
#[cfg(test)]
mod disk_index_build_tests {
    use crate::test_utils::{GraphDataF32VectorUnitData, GraphDataMinMaxVectorUnitData};
    use rstest::rstest;

    use crate::{
        build::builder::core::disk_index_builder_tests::{
            new_vfs, verify_search_result_with_ground_truth, IndexBuildFixture, TestParams,
        },
        QuantizationType,
    };

    #[derive(PartialEq)]
    enum BuildType {
        AsyncFP,
        AsyncSQ1Bit,
        AsyncPQ,
    }

    #[rstest]
    pub fn test_disk_index_builder(
        #[values(false, true)] use_sharded_build: bool,
        #[values(BuildType::AsyncFP, BuildType::AsyncSQ1Bit, BuildType::AsyncPQ)]
        build_type: BuildType,
    ) {
        let index_path_prefix = "/disk_index_build/test_disk_index_build".to_string();

        run_disk_index_builder_test(index_path_prefix, use_sharded_build, build_type);
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
