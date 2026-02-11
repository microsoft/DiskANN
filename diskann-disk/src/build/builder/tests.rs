/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index builder tests.
#[cfg(test)]
mod chunkable_disk_index_build_tests {
    use std::{
        fs,
        sync::{Arc, RwLock},
        time::Duration,
    };

    use diskann_providers::test_utils::{
        graph_data_type_utils::GraphDataF32VectorUnitData, GraphDataMinMaxVectorUnitData,
    };
    use diskann_utils::test_data_root;
    use rstest::rstest;

    use crate::{
        build::{
            builder::core::disk_index_builder_tests::{
                new_vfs, verify_search_result_with_ground_truth, CheckpointParams,
                IndexBuildFixture, TestParams,
            },
            chunking::{
                checkpoint::{CheckpointManagerClone, CheckpointRecordManagerWithFileStorage},
                continuation::{
                    ChunkingConfig, ContinuationGrant, ContinuationTrackerTrait,
                    NaiveContinuationTracker,
                },
            },
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
    pub fn test_disk_index_builder_with_chunking_no_break(
        #[values(false, true)] use_sharded_build: bool,
        #[values(BuildType::AsyncFP, BuildType::AsyncSQ1Bit, BuildType::AsyncPQ)]
        build_type: BuildType,
    ) {
        let chunking_config = ChunkingConfig {
            continuation_checker: Box::<NaiveContinuationTracker>::default(),
            data_compression_chunk_vector_count: 10,
            inmemory_build_chunk_vector_count: 10,
        };

        let index_path_prefix = "/disk_index_build/test_disk_index_build_with_chunking".to_string();
        run_disk_index_builder_test(
            index_path_prefix,
            chunking_config,
            use_sharded_build,
            build_type,
        );
    }

    #[rstest]
    pub fn test_disk_index_builder_without_chunking_no_break(
        #[values(false, true)] use_sharded_build: bool,
        #[values(BuildType::AsyncFP, BuildType::AsyncSQ1Bit, BuildType::AsyncPQ)]
        build_type: BuildType,
    ) {
        let chunking_config = ChunkingConfig::default();
        let index_path_prefix =
            "/disk_index_build/test_disk_index_build_without_chunking".to_string();

        run_disk_index_builder_test(
            index_path_prefix,
            chunking_config,
            use_sharded_build,
            build_type,
        );
    }

    #[rstest]
    pub fn test_disk_index_builder_with_chunking_and_yield_in_between(
        #[values(false, true)] use_sharded_build: bool,
        #[values(BuildType::AsyncFP, BuildType::AsyncSQ1Bit, BuildType::AsyncPQ)]
        build_type: BuildType,
    ) {
        let chunking_config = ChunkingConfig {
            continuation_checker: Box::new(MockCanYeildContinuationChecker {
                count: Arc::new(RwLock::new(0)),
            }),
            data_compression_chunk_vector_count: 50,
            inmemory_build_chunk_vector_count: 50,
        };

        let index_path_prefix =
            "/disk_index_build/test_disk_index_build_without_chunking_and_yeild".to_string();

        run_disk_index_builder_test(
            index_path_prefix,
            chunking_config,
            use_sharded_build,
            build_type,
        );
    }

    // Helper function to run the tests with consistent behavior
    fn run_disk_index_builder_test(
        index_path_prefix: String,
        chunking_config: ChunkingConfig,
        use_sharded_build: bool,
        build_type: BuildType,
    ) {
        match build_type {
            BuildType::AsyncFP => {
                run_test(
                    index_path_prefix,
                    QuantizationType::FP,
                    chunking_config,
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
                    chunking_config,
                    use_sharded_build,
                    8,   // top_k
                    130, // search_l
                );
            }
            BuildType::AsyncPQ => {
                run_test(
                    index_path_prefix,
                    QuantizationType::PQ { num_chunks: 32 },
                    chunking_config,
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
        chunking_config: ChunkingConfig,
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
        let checkpoint_params = new_checkpoint_params(&index_path_prefix, chunking_config);

        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix: index_path_prefix.clone(),
            index_build_ram_gb: get_index_build_ram_gb(use_sharded_build),
            checkpoint_params,
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

        remove_checkpoint_record_file(&index_path_prefix);
    }

    #[rstest]
    #[case(false, 1, QuantizationType::FP)]
    #[case(false, 1, QuantizationType::PQ {num_chunks : 32})]
    #[case(true, 1, QuantizationType::PQ {num_chunks : 32})]
    #[case(false, 2, QuantizationType::FP)]
    #[case(true, 1, QuantizationType::FP)]
    // Disallow flaky tests with multiple threads
    // #[case(true, false, 2)]
    // #[case(true, true, 2)]
    pub fn test_disk_index_builder_with_stop_in_between(
        #[case] use_sharded_build: bool,
        #[case] num_threads: usize,
        #[case] build_quantization_type: QuantizationType,
    ) {
        // Use the same parameters from [test_sift_build_and_search] in diskann_index
        let l_build = 64;
        let max_degree = 16;
        let top_k = 10;
        let search_l = 32;

        let chunking_config = ChunkingConfig {
            continuation_checker: Box::new(MockCanStopContinuationChecker {
                count: Arc::new(RwLock::new(0)),
            }),
            data_compression_chunk_vector_count: 50,
            inmemory_build_chunk_vector_count: 50,
        };

        let index_path_prefix = format!(
            "/disk_index_build/test_disk_index_build_with_chunking_and_stop_sharded{}_T{}_Q({})",
            use_sharded_build, num_threads, build_quantization_type
        );

        // Convert VFS path to real filesystem path for checkpoint manager
        let vfs_path = index_path_prefix.trim_start_matches('/');
        let real_path = test_data_root().join(vfs_path);
        let checkpoint_manager = Box::new(CheckpointRecordManagerWithFileStorage::new(
            real_path.to_str().unwrap(),
            0,
        ));
        let checkpoint_params = Some(CheckpointParams {
            chunking_config,
            checkpoint_record_manager: checkpoint_manager.clone_box(),
        });

        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix: index_path_prefix.clone(),
            index_build_ram_gb: get_index_build_ram_gb(use_sharded_build),
            checkpoint_params,
            num_threads,
            build_quantization_type,
            ..TestParams::default()
        };

        let fixture = IndexBuildFixture::new(new_vfs(), params).unwrap();

        loop {
            if checkpoint_manager.has_completed().unwrap() {
                break;
            }

            fixture.build::<GraphDataF32VectorUnitData>().unwrap();
        }

        fixture.compare_pq_compressed_files();

        // Cannot compare the index with truth data because stop/resume operations break the
        // deterministic random seed sequence, resulting in a different index structure.
        verify_search_result_with_ground_truth::<GraphDataF32VectorUnitData>(
            &fixture.params,
            top_k,
            search_l,
            &fixture.storage_provider,
        )
        .unwrap();

        remove_checkpoint_record_file(&index_path_prefix);
    }

    ///////////////////////////////
    // MinMax Integration Tests //
    //////////////////////////////

    #[rstest]
    pub fn test_disk_minmax_index_builder(
        #[values(false, true)] use_sharded_build: bool,
    ) -> anyhow::Result<()> {
        const DATA_FILE: &str = "/sift/siftsmall_learn_256pts_minmax.fbin";
        let chunking_config = ChunkingConfig::default();
        let index_path_prefix =
            "/disk_index_build/test_minmax_disk_index_build_without_chunking".to_string();
        let params = TestParams {
            dim: 148,
            full_dim: 128,
            data_path: DATA_FILE.to_string(),
            ..TestParams::default()
        };
        run_minmax_test(
            index_path_prefix,
            chunking_config,
            params,
            use_sharded_build,
            10,
            32,
        );
        Ok(())
    }

    fn run_minmax_test(
        index_path_prefix: String,
        chunking_config: ChunkingConfig,
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
        let checkpoint_params = new_checkpoint_params(&index_path_prefix, chunking_config);

        let params = TestParams {
            l_build,
            max_degree,
            index_path_prefix: index_path_prefix.clone(),
            index_build_ram_gb: get_index_build_ram_gb(use_sharded_build),
            checkpoint_params,
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

        remove_checkpoint_record_file(&index_path_prefix);
    }

    fn remove_checkpoint_record_file(index_path_prefix: &str) {
        // Convert VFS path (e.g., "/disk_index_build/...") to real filesystem path
        let vfs_path = index_path_prefix.trim_start_matches('/');
        let real_path = test_data_root().join(vfs_path);
        fs::remove_file(format!("{}_0.checkpoint", real_path.display()))
            .expect("Failed to delete file");
    }

    fn get_index_build_ram_gb(use_sharded_build: bool) -> f64 {
        if use_sharded_build {
            0.0001 // small enough to trigger sharded build.
        } else {
            1.0
        }
    }

    fn new_checkpoint_params(
        index_path_prefix: &str,
        chunking_config: ChunkingConfig,
    ) -> Option<CheckpointParams> {
        // Convert VFS path (e.g., "/disk_index_build/...") to real filesystem path
        let vfs_path = index_path_prefix.trim_start_matches('/');
        let real_path = test_data_root().join(vfs_path);
        Some(CheckpointParams {
            chunking_config,
            checkpoint_record_manager: Box::new(CheckpointRecordManagerWithFileStorage::new(
                real_path.to_str().unwrap(),
                0,
            )),
        })
    }

    struct MockCanYeildContinuationChecker {
        count: Arc<RwLock<usize>>,
    }

    impl Clone for MockCanYeildContinuationChecker {
        fn clone(&self) -> Self {
            MockCanYeildContinuationChecker {
                count: self.count.clone(),
            }
        }
    }

    impl ContinuationTrackerTrait for MockCanYeildContinuationChecker {
        fn get_continuation_grant(&self) -> ContinuationGrant {
            let mut count = self.count.write().unwrap();
            *count += 1;
            if (*count).is_multiple_of(2) {
                ContinuationGrant::Continue
            } else {
                ContinuationGrant::Yield(Duration::from_millis(100))
            }
        }
    }

    struct MockCanStopContinuationChecker {
        count: Arc<RwLock<usize>>,
    }

    impl Clone for MockCanStopContinuationChecker {
        fn clone(&self) -> Self {
            MockCanStopContinuationChecker {
                count: self.count.clone(),
            }
        }
    }

    impl ContinuationTrackerTrait for MockCanStopContinuationChecker {
        fn get_continuation_grant(&self) -> ContinuationGrant {
            let mut count = self.count.write().unwrap();
            *count += 1;
            if !(*count).is_multiple_of(3) {
                ContinuationGrant::Continue
            } else {
                ContinuationGrant::Stop
            }
        }
    }
}
