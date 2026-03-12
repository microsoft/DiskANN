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
    fn insert_f32_vector(
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

    /// Helper to insert a vector with a string external ID and FP32 data.
    fn insert_f32_vector_str(
        ctx: &Context,
        index_ptr: *const c_void,
        eid: &str,
        vector: &[f32],
    ) -> bool {
        let id_bytes = eid.as_bytes();
        let vector_bytes: &[u8] = bytemuck::cast_slice(vector);
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
    fn do_search(
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

    // ── Grid sanity-check tests (L2 distance) ──────────────────────────

    /// Generates `grid_size ^ dimensions` vectors on an integer grid.
    /// Returns `(ids, vectors)` where each id is a descriptive string
    /// like `"grid_vector_00000001_dim3"`.
    fn generate_grid_vectors(dimensions: usize, grid_size: usize) -> (Vec<String>, Vec<Vec<f32>>) {
        let total = grid_size.pow(dimensions as u32);
        let mut ids = Vec::with_capacity(total);
        let mut vectors = Vec::with_capacity(total);

        for i in 0..total {
            let mut vec = vec![0.0f32; dimensions];
            let mut pos = i;
            for d in (0..dimensions).rev() {
                vec[d] = (pos % grid_size) as f32;
                pos /= grid_size;
            }
            ids.push(format!("grid_vector_{:08}_dim{}", i + 1, dimensions));
            vectors.push(vec);
        }

        (ids, vectors)
    }

    /// Brute-force k nearest neighbors using squared L2 distance.
    /// Returns the set of IDs of the k closest vectors.
    fn brute_force_l2_knn(
        ids: &[String],
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
    ) -> Vec<String> {
        let mut scored: Vec<(&String, f32)> = ids
            .iter()
            .zip(vectors.iter())
            .map(|(id, vec)| {
                let dist: f32 = vec
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (id, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id.clone()).collect()
    }

    fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// Count intersection by grouping results by distance (handles ties).
    /// When multiple vectors are equidistant, the exact IDs may differ
    /// between brute-force and ANN, but the distance counts must match.
    fn distance_based_intersection(
        vectors: &[Vec<f32>],
        ids: &[String],
        query: &[f32],
        expected: &[String],
        actual: &[String],
    ) -> usize {
        use std::collections::HashMap;

        let id_to_idx: HashMap<&str, usize> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let count_per_dist = |id_set: &[String]| -> HashMap<u64, usize> {
            let mut counts = HashMap::new();
            for id in id_set {
                let idx = id_to_idx[id.as_str()];
                let dist = squared_l2(&vectors[idx], query);
                let key = dist.to_bits() as u64;
                *counts.entry(key).or_insert(0) += 1;
            }
            counts
        };

        let expected_counts = count_per_dist(expected);
        let actual_counts = count_per_dist(actual);

        let mut intersection = 0;
        for (&dist, &exp_count) in &expected_counts {
            if let Some(&act_count) = actual_counts.get(&dist) {
                intersection += exp_count.min(act_count);
            }
        }
        intersection
    }

    /// Parse variable-length string IDs from a search output buffer.
    /// Each entry is a `u32` length prefix followed by that many UTF-8 bytes.
    fn parse_string_ids(output_id_buffer: &[u8], count: usize) -> Vec<String> {
        let mut result_ids = Vec::with_capacity(count);
        let mut offset = 0;
        for _ in 0..count {
            let mut id_len = 0u32;
            bytemuck::bytes_of_mut(&mut id_len)
                .copy_from_slice(&output_id_buffer[offset..offset + mem::size_of::<u32>()]);
            offset += mem::size_of::<u32>();
            let id_str = std::str::from_utf8(&output_id_buffer[offset..offset + id_len as usize])
                .expect("id should be valid utf8");
            result_ids.push(id_str.to_string());
            offset += id_len as usize;
        }
        result_ids
    }

    /// Helper: create an L2 index, insert grid vectors, query each, return recall.
    unsafe fn run_grid_recall(store: &Store, dimensions: u32, grid_size: usize, k: usize) -> f64 {
        store.clear();
        let callbacks = store.callbacks();
        let ctx = Context(0);

        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dimensions,
                0,
                VectorQuantType::NoQuant,
                Metric::L2 as i32,
                100,
                32,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };
        assert!(!index_ptr.is_null());

        let (ids, vectors) = generate_grid_vectors(dimensions as usize, grid_size);
        let max_id_len = ids.iter().map(|id| id.len()).max().unwrap_or(0);

        for (eid, vec) in ids.iter().zip(vectors.iter()) {
            assert!(
                insert_f32_vector_str(&ctx, index_ptr, eid, vec),
                "insert failed for eid={eid}"
            );
        }

        let mut total_matches = 0usize;
        let mut total_expected = 0usize;

        for vec in &vectors {
            let query_bytes: &[u8] = bytemuck::cast_slice(vec);
            let max_id_size = mem::size_of::<u32>() + max_id_len;
            let mut output_id_buffer = vec![0u8; k * max_id_size];
            let mut output_dists = vec![0f32; k];

            let count = unsafe {
                search_vector(
                    ctx.0,
                    index_ptr,
                    VectorValueType::FP32,
                    query_bytes.as_ptr(),
                    vec.len(),
                    2.0,
                    200,
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
            assert!(count >= 0, "search failed");

            let result_ids = parse_string_ids(&output_id_buffer, count as usize);
            let expected_ids = brute_force_l2_knn(&ids, &vectors, vec, k);
            let matches =
                distance_based_intersection(&vectors, &ids, vec, &expected_ids, &result_ids);
            total_matches += matches;
            total_expected += expected_ids.len();
        }

        unsafe { drop_index(ctx.0, index_ptr) };

        total_matches as f64 / total_expected as f64
    }

    #[test]
    fn grid_l2_recall_1d_100() {
        let store = Store;
        let recall = unsafe { run_grid_recall(&store, 1, 100, 3) };
        assert!(recall >= 0.99, "1D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_2d_10() {
        let store = Store;
        let recall = unsafe { run_grid_recall(&store, 2, 10, 3) };
        assert!(recall >= 0.99, "2D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_3d_7() {
        let store = Store;
        let recall = unsafe { run_grid_recall(&store, 3, 7, 3) };
        assert!(recall >= 0.99, "3D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_4d_5() {
        let store = Store;
        let recall = unsafe { run_grid_recall(&store, 4, 5, 3) };
        assert!(recall >= 0.99, "4D grid recall too low: {recall:.4}");
    }

    // ── Circle sanity-check tests (Cosine distance) ────────────────────

    /// Generates `point_count` 2D vectors evenly spaced on a circle of the given radius.
    fn generate_circle_vectors(point_count: usize, radius: f32) -> (Vec<String>, Vec<Vec<f32>>) {
        let mut ids = Vec::with_capacity(point_count);
        let mut vectors = Vec::with_capacity(point_count);

        for i in 0..point_count {
            let theta = 2.0 * std::f32::consts::PI * (i as f32) / (point_count as f32);
            ids.push(format!(
                "circle_point_{:08}_r{:.2}_theta{:.6}",
                i + 1,
                radius,
                theta
            ));
            vectors.push(vec![theta.cos() * radius, theta.sin() * radius]);
        }

        (ids, vectors)
    }

    /// Brute-force k nearest neighbors using cosine distance = 1 - cos_sim.
    fn brute_force_cosine_knn(
        ids: &[String],
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
    ) -> Vec<String> {
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(&String, f32)> = ids
            .iter()
            .zip(vectors.iter())
            .map(|(id, vec)| {
                let dot: f32 = vec.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let v_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos_sim = if q_norm > 0.0 && v_norm > 0.0 {
                    dot / (q_norm * v_norm)
                } else {
                    0.0
                };
                (id, 1.0 - cos_sim)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id.clone()).collect()
    }

    /// Helper: create a cosine index, insert circle vectors, query each, return recall.
    unsafe fn run_circle_recall(store: &Store, point_count: usize, radius: f32, k: usize) -> f64 {
        store.clear();
        let callbacks = store.callbacks();
        let ctx = Context(0);

        let dim: u32 = 2;
        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dim,
                0,
                VectorQuantType::NoQuant,
                Metric::Cosine as i32,
                100,
                32,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };
        assert!(!index_ptr.is_null());

        let (ids, vectors) = generate_circle_vectors(point_count, radius);
        let max_id_len = ids.iter().map(|id| id.len()).max().unwrap_or(0);

        for (eid, vec) in ids.iter().zip(vectors.iter()) {
            assert!(
                insert_f32_vector_str(&ctx, index_ptr, eid, vec),
                "insert failed for eid={eid}"
            );
        }

        let mut total_matches = 0usize;
        let mut total_expected = 0usize;

        for vec in &vectors {
            let query_bytes: &[u8] = bytemuck::cast_slice(vec);
            let max_id_size = mem::size_of::<u32>() + max_id_len;
            let mut output_id_buffer = vec![0u8; k * max_id_size];
            let mut output_dists = vec![0f32; k];

            let count = unsafe {
                search_vector(
                    ctx.0,
                    index_ptr,
                    VectorValueType::FP32,
                    query_bytes.as_ptr(),
                    vec.len(),
                    2.0,
                    200,
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
            assert!(count >= 0, "search failed");

            let result_ids = parse_string_ids(&output_id_buffer, count as usize);
            let expected_ids = brute_force_cosine_knn(&ids, &vectors, vec, k);
            let matches = result_ids
                .iter()
                .filter(|id| expected_ids.contains(id))
                .count();
            total_matches += matches;
            total_expected += expected_ids.len();
        }

        unsafe { drop_index(ctx.0, index_ptr) };

        total_matches as f64 / total_expected as f64
    }

    /// Circle with 100 points, radius=1.0, cosine distance, k=5
    #[test]
    fn circle_cosine_recall_r1_100pt() {
        let store = Store;
        let recall = unsafe { run_circle_recall(&store, 100, 1.0, 5) };
        assert!(recall >= 0.99, "circle r=1 recall too low: {recall:.4}");
    }

    /// Circle with 93 points, radius=534.0, cosine distance, k=5
    #[test]
    fn circle_cosine_recall_r534_93pt() {
        let store = Store;
        let recall = unsafe { run_circle_recall(&store, 93, 534.0, 5) };
        assert!(recall >= 0.99, "circle r=534 recall too low: {recall:.4}");
    }

    /// Circle with 50 points, radius=10.0, cosine distance, k=3
    #[test]
    fn circle_cosine_recall_r10_50pt() {
        let store = Store;
        let recall = unsafe { run_circle_recall(&store, 50, 10.0, 3) };
        assert!(recall >= 0.99, "circle r=10 recall too low: {recall:.4}");
    }
}
