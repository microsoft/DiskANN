/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod tests {
    use std::{ffi::c_void, mem, ptr};

    use diskann_vector::distance::Metric;

    use crate::{
        VectorQuantType, VectorValueType, card, check_external_id_valid, create_index, drop_index,
        garnet::Context, insert, remove, search_vector, set_attribute, test_utils::Store,
    };

    /// Creates an index with default test values and returns (index_ptr, Context).
    /// The caller is responsible for calling drop_index when done.
    fn create_test_index(store: &Store) -> (*const c_void, Context) {
        let (index_ptr, ctx) = create_test_index_with_metric(store, Metric::L2 as i32);
        assert!(
            !index_ptr.is_null(),
            "create_test_index failed to create index"
        );
        (index_ptr, ctx)
    }

    /// Creates an index with specified metric type and returns (index_ptr, Context).
    /// The caller is responsible for calling drop_index when done.
    fn create_test_index_with_metric(store: &Store, metric_type: i32) -> (*const c_void, Context) {
        store.clear();

        let callbacks = store.callbacks();
        let ctx = Context(0);

        let dim: u32 = 2;
        let reduce_dim = 0;
        let quant_type = VectorQuantType::NoQuant;
        let l_build = 10;
        let max_degree = 20;

        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dim,
                reduce_dim,
                quant_type,
                metric_type,
                l_build,
                max_degree,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };

        (index_ptr, ctx)
    }

    #[test]
    fn basic_create_index() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);
        assert!(!index_ptr.is_null());

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn create_index_with_invalid_metric_returns_null() {
        let store = Store;

        // Test with invalid metric type values — passed as raw i32
        let invalid_metrics = [-1, -2, 99, i32::MAX, i32::MIN];

        for invalid_metric in invalid_metrics {
            let (index_ptr, _ctx) = create_test_index_with_metric(&store, invalid_metric);
            assert!(
                index_ptr.is_null(),
                "Expected null for invalid metric_type={}",
                invalid_metric
            );
        }
    }

    #[test]
    fn create_index_with_valid_metrics() {
        let store = Store;

        // Test all valid metric types
        let valid_metrics = [
            Metric::Cosine as i32,
            Metric::L2 as i32,
            Metric::InnerProduct as i32,
            Metric::CosineNormalized as i32,
        ];

        for valid_metric in valid_metrics {
            let (index_ptr, ctx) = create_test_index_with_metric(&store, valid_metric);
            assert!(
                !index_ptr.is_null(),
                "Expected non-null for valid metric_type_raw={}",
                valid_metric
            );
            unsafe {
                drop_index(ctx.0, index_ptr);
            }
        }
    }

    #[test]
    fn add_check_and_remove_vector() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        // Vector id with 4 bytes size
        let garnet_vector_id = 42u32;
        let id_bytes = bytemuck::bytes_of(&garnet_vector_id);

        let vector: [f32; 2] = [1.0, 2.0];
        let vector_bytes = bytemuck::cast_slice(&vector);
        let vector_len = 2;

        let attributes_bytes = b"wololo";
        let attributes_len = attributes_bytes.len();

        let result: bool = unsafe {
            insert(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                VectorValueType::FP32,
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_len,
            )
        };

        assert!(result);

        // Confirm vector exists using FFI function
        let exists =
            unsafe { check_external_id_valid(ctx.0, index_ptr, id_bytes.as_ptr(), id_bytes.len()) };
        assert!(exists);

        let mut cardinality = unsafe { card(ctx.0, index_ptr) };
        assert_eq!(cardinality, 1);

        let removed = unsafe { remove(ctx.0, index_ptr, id_bytes.as_ptr(), id_bytes.len()) };
        assert!(removed);

        // Currently we're not tracking deletions for cardinality, so it should still be 1
        cardinality = unsafe { card(ctx.0, index_ptr) };
        assert_eq!(cardinality, 1);

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn update_vector_attributes() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        // Vector id with 4 bytes size
        let garnet_vector_id = 42u32;
        let id_bytes = bytemuck::bytes_of(&garnet_vector_id);

        // Try to update attributes from non-existing vector
        let attributes1 = b"wololo";
        let set_attribute_result1 = unsafe {
            set_attribute(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                attributes1.as_ptr(),
                attributes1.len(),
            )
        };
        assert!(!set_attribute_result1);

        let vector: [f32; 2] = [1.0, 2.0];
        let vector_bytes = bytemuck::cast_slice(&vector);

        let result: bool = unsafe {
            insert(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                VectorValueType::FP32,
                vector_bytes.as_ptr(),
                vector_bytes.len() / 4,
                attributes1.as_ptr(),
                attributes1.len(),
            )
        };
        assert!(result);

        // Set attributes after insertion
        let attributes2 = b"new_attributes";
        let set_attribute_result2 = unsafe {
            set_attribute(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                attributes2.as_ptr(),
                attributes2.len(),
            )
        };
        assert!(set_attribute_result2);

        // Set attributes to empty using null ptr
        let set_attribute_result3 = unsafe {
            set_attribute(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                ptr::null(),
                0,
            )
        };
        assert!(set_attribute_result3);

        let empty_attribute = b"";
        let set_attribute_result4 = unsafe {
            set_attribute(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                empty_attribute.as_ptr(),
                0,
            )
        };
        assert!(set_attribute_result4);

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn external_id_exists_lifecycle() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        // EID 1
        let eid1 = 1u32;
        let eid1_bytes = bytemuck::bytes_of(&eid1);

        // EID 2
        let eid2 = 2u32;
        let eid2_bytes = bytemuck::bytes_of(&eid2);

        let vector: [f32; 2] = [1.0, 2.0];
        let vector_bytes = bytemuck::cast_slice(&vector);
        let vector_len = 2;

        let attributes_bytes = b"test_attr";

        // Check external_id exists with EID 1 (should not exist initially)
        let exists1 = unsafe {
            check_external_id_valid(ctx.0, index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists1, "EID 1 should not exist initially");

        // Add vector with EID 1
        let insert_result1 = unsafe {
            insert(
                ctx.0,
                index_ptr,
                eid1_bytes.as_ptr(),
                eid1_bytes.len(),
                VectorValueType::FP32,
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_bytes.len(),
            )
        };
        assert!(insert_result1, "Insert with EID 1 should succeed");

        // Check external_id exists with EID 1 (should exist after insert)
        let exists2 = unsafe {
            check_external_id_valid(ctx.0, index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(exists2, "EID 1 should exist after insert");

        // Remove vector with EID 1
        let removed = unsafe { remove(ctx.0, index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len()) };
        assert!(removed, "Remove with EID 1 should succeed");

        // Check external_id exists with EID 1 (should not exist after removal)
        let exists3 = unsafe {
            check_external_id_valid(ctx.0, index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists3, "EID 1 should not exist after removal");

        // Add vector with EID 2
        let insert_result2 = unsafe {
            insert(
                ctx.0,
                index_ptr,
                eid2_bytes.as_ptr(),
                eid2_bytes.len(),
                VectorValueType::FP32,
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_bytes.len(),
            )
        };
        assert!(insert_result2, "Insert with EID 2 should succeed");

        // Check external_id exists with EID 2 (should exist after insert)
        let exists4 = unsafe {
            check_external_id_valid(ctx.0, index_ptr, eid2_bytes.as_ptr(), eid2_bytes.len())
        };
        assert!(exists4, "EID 2 should exist after insert");

        // Check external_id exists with EID 1 (should still not exist)
        let exists5 = unsafe {
            check_external_id_valid(ctx.0, index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists5, "EID 1 should still not exist");

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }

    /// Using u64 external IDs, insert some vectors and ensure search results are same.
    #[test]
    fn search_with_large_external_ids() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        let id1 = 1234u64;
        let v1 = &[1u8, 0u8];
        let id2 = 5678u64;
        let v2 = &[0u8, 1u8];

        let id1_bytes = bytemuck::bytes_of(&id1);
        let id2_bytes = bytemuck::bytes_of(&id2);

        assert!(unsafe {
            insert(
                ctx.0,
                index_ptr,
                id1_bytes.as_ptr(),
                id1_bytes.len(),
                VectorValueType::XB8,
                v1.as_ptr(),
                v1.len(),
                b"".as_ptr(),
                0,
            )
        });

        assert!(unsafe {
            insert(
                ctx.0,
                index_ptr,
                id2_bytes.as_ptr(),
                id2_bytes.len(),
                VectorValueType::XB8,
                v2.as_ptr(),
                v2.len(),
                b"".as_ptr(),
                0,
            )
        });

        let qv = &[0u8, 0u8];
        let mut output_id_buffer = vec![0u8; 2 * (mem::size_of::<u64>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; 2];

        let count = unsafe {
            search_vector(
                ctx.0,
                index_ptr,
                VectorValueType::XB8,
                qv.as_ptr(),
                qv.len(),
                2.0,
                10,
                ptr::null(),
                0,
                0,
                output_id_buffer.as_mut_ptr(),
                output_id_buffer.len(),
                output_dists.as_mut_ptr(),
                output_dists.len(),
                ptr::null_mut(),
            )
        };

        assert_eq!(count, 2);

        let mut output_ids = vec![];
        let mut offset = 0;
        for _ in 0..(count as usize) {
            let mut id_len = 0u32;
            bytemuck::bytes_of_mut(&mut id_len)
                .copy_from_slice(&output_id_buffer[offset..offset + mem::size_of::<u32>()]);
            offset += mem::size_of::<u32>();

            assert_eq!(id_len, mem::size_of::<u64>() as u32);

            let mut id = 0u64;
            bytemuck::bytes_of_mut(&mut id)
                .copy_from_slice(&output_id_buffer[offset..offset + mem::size_of::<u64>()]);
            offset += mem::size_of::<u64>();

            output_ids.push(id);
        }

        for &d in &output_dists[2..] {
            assert_eq!(d, 0.0);
        }
        match (output_ids[0], output_ids[1]) {
            (1234u64, 5678u64) => {
                assert_eq!(output_dists[0], 1.0);
                assert_eq!(output_dists[1], 1.0);
            }
            (5678u64, 1234u64) => {
                assert_eq!(output_dists[0], 1.0);
                assert_eq!(output_dists[1], 1.0);
            }
            _ => {
                panic!("got unexpected ids {} and {}", output_ids[0], output_ids[1]);
            }
        }

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }

    /// Helper to insert a vector with u32 external ID and FP32 data.
    unsafe fn insert_f32_vector(
        ctx: &Context,
        index_ptr: *const c_void,
        eid: u32,
        vector: &[f32],
    ) -> bool {
        let id_bytes = bytemuck::bytes_of(&eid);
        let vector_bytes = bytemuck::cast_slice(vector);
        unsafe {
            insert(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                VectorValueType::FP32,
                vector_bytes.as_ptr(),
                vector.len(),
                b"".as_ptr(),
                0,
            )
        }
    }

    /// Helper to run search_vector and parse the output IDs (u32) and distances.
    unsafe fn do_search(
        ctx: &Context,
        index_ptr: *const c_void,
        query: &[f32],
        k: usize,
        bitmap: Option<&[u8]>,
    ) -> (Vec<u32>, Vec<f32>) {
        let query_bytes = bytemuck::cast_slice(query);
        let mut output_id_buffer = vec![0u8; k * (mem::size_of::<u32>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; k];

        let (bitmap_ptr, bitmap_len) = match bitmap {
            Some(b) => (b.as_ptr(), b.len()),
            None => (ptr::null(), 0),
        };

        let count = unsafe {
            search_vector(
                ctx.0,
                index_ptr,
                VectorValueType::FP32,
                query_bytes.as_ptr(),
                query.len(),
                0.0,
                (k * 2) as u32, // search exploration factor
                bitmap_ptr,
                bitmap_len,
                0,
                output_id_buffer.as_mut_ptr(),
                output_id_buffer.len(),
                output_dists.as_mut_ptr(),
                output_dists.len(),
                ptr::null_mut(),
            )
        };

        assert!(count >= 0, "search failed with {count}");
        let count = count as usize;

        let mut ids = vec![];
        let mut offset = 0;
        for _ in 0..count {
            let mut id_len = 0u32;
            bytemuck::bytes_of_mut(&mut id_len)
                .copy_from_slice(&output_id_buffer[offset..offset + mem::size_of::<u32>()]);
            offset += mem::size_of::<u32>();
            let mut id = 0u32;
            bytemuck::bytes_of_mut(&mut id)
                .copy_from_slice(&output_id_buffer[offset..offset + id_len as usize]);
            offset += id_len as usize;
            ids.push(id);
        }

        output_dists.truncate(count);
        (ids, output_dists)
    }

    #[test]
    fn search_without_filter() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        unsafe {
            assert!(insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]));
            assert!(insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]));
            assert!(insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]));

            let (ids, _dists) = do_search(&ctx, index_ptr, &[1.0, 0.0], 3, None);
            assert!(ids.len() >= 2, "should return at least 2 vectors");
            // Closest to [1,0] should be id=10 (exact match)
            assert_eq!(ids[0], 10);

            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn search_with_bitmap_all_match() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        unsafe {
            assert!(insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]));
            assert!(insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]));
            assert!(insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]));

            // Bitmap with bits 1,2,3 set (internal IDs for the 3 inserted vectors;
            // internal ID 0 is the start point)
            let bitmap: [u8; 8] = [0b00001110, 0, 0, 0, 0, 0, 0, 0];
            let (ids, _dists) = do_search(&ctx, index_ptr, &[1.0, 0.0], 3, Some(&bitmap));
            // Start point (internal ID 0) is filtered out from results,
            // so we may get fewer than k results.
            assert!(ids.len() >= 2, "should return at least 2 matching vectors");
            assert_eq!(ids[0], 10, "closest should still be id=10");

            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn search_with_bitmap_partial_match() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        unsafe {
            // Internal ID 0 -> EID 10, vector [1,0]
            assert!(insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]));
            // Internal ID 1 -> EID 20, vector [0,1]
            assert!(insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]));
            // Internal ID 2 -> EID 30, vector [1,1]
            assert!(insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]));

            // Bitmap with only bit 2 set (internal ID 2 = EID 20, second inserted vector)
            let bitmap: [u8; 8] = [0b00000100, 0, 0, 0, 0, 0, 0, 0];
            // Query close to EID 20's vector [0,1] to ensure it appears in results
            let (ids, _dists) = do_search(&ctx, index_ptr, &[0.0, 1.0], 3, Some(&bitmap));
            // BetaFilter biases toward matching vectors by scaling their distances.
            assert!(!ids.is_empty(), "should return at least one result");
            // EID 20 should appear since it's the closest to query AND matches the filter
            assert!(
                ids.contains(&20),
                "filtered vector EID 20 should be in results"
            );

            drop_index(ctx.0, index_ptr);
        }
    }

    #[test]
    fn search_with_null_bitmap_same_as_unfiltered() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store);

        unsafe {
            assert!(insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]));
            assert!(insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]));

            // Null bitmap should behave like no filter
            let (ids_null, dists_null) = do_search(&ctx, index_ptr, &[1.0, 0.0], 2, None);
            let (ids_empty, dists_empty) = do_search(&ctx, index_ptr, &[1.0, 0.0], 2, Some(&[]));

            // Both should return same results (empty slice triggers has_filter=false
            // because bitmap_len=0)
            assert_eq!(ids_null, ids_empty);
            assert_eq!(dists_null, dists_empty);

            drop_index(ctx.0, index_ptr);
        }
    }

    /// Insert 27 vectors in a 3x3x3 grid with string external IDs like "v_0_0_0",
    /// then run multiple VSIM queries each requesting the 3 nearest neighbors.
    #[test]
    fn search_grid_with_string_ids() {
        let store = Store;
        store.clear();

        let callbacks = store.callbacks();
        let ctx = Context(0);

        let dim: u32 = 3;
        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dim,
                0,
                VectorQuantType::NoQuant,
                Metric::L2 as i32,
                10,
                20,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };
        assert!(!index_ptr.is_null());

        // Insert 27 vectors in a 3x3x3 grid with string IDs "v_x_y_z"
        let mut ids: Vec<String> = Vec::new();
        let mut vectors: Vec<[f32; 3]> = Vec::new();
        for x in 0..3u32 {
            for y in 0..3u32 {
                for z in 0..3u32 {
                    let id = format!("v_{x}_{y}_{z}");
                    let vec = [x as f32, y as f32, z as f32];

                    let id_bytes = id.as_bytes();
                    let vec_bytes: &[u8] = bytemuck::cast_slice(&vec);
                    let inserted = unsafe {
                        insert(
                            ctx.0,
                            index_ptr,
                            id_bytes.as_ptr(),
                            id_bytes.len(),
                            VectorValueType::FP32,
                            vec_bytes.as_ptr(),
                            vec.len(),
                            b"".as_ptr(),
                            0,
                        )
                    };
                    assert!(inserted, "failed to insert {id}");

                    ids.push(id);
                    vectors.push(vec);
                }
            }
        }

        assert_eq!(unsafe { card(ctx.0, index_ptr) }, 27);

        // Helper to run a search and parse variable-length string IDs from the output buffer.
        let do_grid_search = |query: &[f32; 3], k: usize| -> (Vec<String>, Vec<f32>) {
            let query_bytes: &[u8] = bytemuck::cast_slice(query);
            // Each result: 4 bytes id_len + up to 7 bytes for "v_X_Y_Z"
            let max_id_size = mem::size_of::<u32>() + 7;
            let mut output_id_buffer = vec![0u8; k * max_id_size];
            let mut output_dists = vec![0f32; k];

            let delta = 2.0f32;
            let search_exploration_factor = 100u32;
            let bitmap_data = ptr::null();
            let bitmap_len = 0;
            let max_filtering_effort = 30;

            let count = unsafe {
                search_vector(
                    ctx.0,
                    index_ptr,
                    VectorValueType::FP32,
                    query_bytes.as_ptr(),
                    query.len(),
                    delta,
                    search_exploration_factor,
                    bitmap_data,
                    bitmap_len,
                    max_filtering_effort,
                    output_id_buffer.as_mut_ptr(),
                    output_id_buffer.len(),
                    output_dists.as_mut_ptr(),
                    output_dists.len(),
                    ptr::null_mut(),
                )
            };
            assert!(count >= 0, "search failed with {count}");
            let count = count as usize;

            let mut result_ids = Vec::new();
            let mut offset = 0;
            for _ in 0..count {
                let mut id_len = 0u32;
                bytemuck::bytes_of_mut(&mut id_len)
                    .copy_from_slice(&output_id_buffer[offset..offset + mem::size_of::<u32>()]);
                offset += mem::size_of::<u32>();

                let id_str =
                    std::str::from_utf8(&output_id_buffer[offset..offset + id_len as usize])
                        .expect("id should be valid utf8");
                result_ids.push(id_str.to_string());
                offset += id_len as usize;
            }

            output_dists.truncate(count);
            (result_ids, output_dists)
        };

        // Query 1: exact match at origin — nearest should be v_0_0_0
        let (ids1, dists1) = do_grid_search(&[0.0, 0.0, 0.0], 3);
        assert_eq!(ids1.len(), 3, "expected 3 results");
        assert_eq!(ids1[0], "v_0_0_0", "closest to origin should be v_0_0_0");
        assert_eq!(dists1[0], 0.0, "exact match should have distance 0");

        // Query 2: exact match at corner — nearest should be v_2_2_2
        let (ids2, dists2) = do_grid_search(&[2.0, 2.0, 2.0], 3);
        assert_eq!(ids2.len(), 3, "expected 3 results");
        assert_eq!(ids2[0], "v_2_2_2", "closest to (2,2,2) should be v_2_2_2");
        assert_eq!(dists2[0], 0.0, "exact match should have distance 0");

        // Query 3: center of grid — nearest should be v_1_1_1
        let (ids3, dists3) = do_grid_search(&[1.0, 1.0, 1.0], 3);
        assert_eq!(ids3.len(), 3, "expected 3 results");
        assert_eq!(ids3[0], "v_1_1_1", "closest to center should be v_1_1_1");
        assert_eq!(dists3[0], 0.0, "exact match should have distance 0");
        // The next 2 results should all be distance 1.0 (one axis off by 1)
        for d in &dists3[1..] {
            assert_eq!(
                *d, 1.0,
                "neighbors of center should be at squared L2 distance 1.0"
            );
        }

        // Query 4: off-grid point — nearest should be v_0_0_0 at squared distance 0.75
        let (ids4, dists4) = do_grid_search(&[0.5, 0.5, 0.5], 3);
        assert_eq!(ids4.len(), 3, "expected 3 results");
        // All 8 corners {0,1}^3 are equidistant at squared L2 = 3*0.25 = 0.75,
        // but v_0_0_0 and v_1_1_1 etc. are among them — just check the distance.
        assert_eq!(
            dists4[0], 0.75,
            "nearest to (0.5,0.5,0.5) should be at squared distance 0.75"
        );

        unsafe {
            drop_index(ctx.0, index_ptr);
        }
    }
}
