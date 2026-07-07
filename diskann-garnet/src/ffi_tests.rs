/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod tests {
    use std::{ffi::c_void, mem, ptr};

    use diskann_vector::distance::Metric;
    use rand::{Rng, seq::SliceRandom};

    use crate::{
        Index, InsertResult, VectorQuantType, backfill_quant_vectors, build_quant_table, card,
        check_external_id_valid, check_internal_id_valid, create_index, drop_index,
        garnet::{Context, Term},
        insert,
        quantization::{GarnetQuantizer, Spherical1Bit},
        remove, search_vector, set_attribute,
        test_utils::Store,
    };

    /// Creates an index with default test values and returns (index_ptr, Context).
    /// The caller is responsible for calling drop_index when done.
    fn create_test_index(store: &Store, quant_type: VectorQuantType) -> (*const c_void, Context) {
        let (index_ptr, ctx) = create_test_index_with_metric(store, quant_type, Metric::L2 as i32);
        assert!(
            !index_ptr.is_null(),
            "create_test_index failed to create index"
        );
        (index_ptr, ctx)
    }

    /// Creates an index with specified metric type and returns (index_ptr, Context).
    /// The caller is responsible for calling drop_index when done.
    fn create_test_index_with_metric(
        store: &Store,
        quant_type: VectorQuantType,
        metric_type: i32,
    ) -> (*const c_void, Context) {
        let callbacks = store.callbacks();
        let ctx = Context::new(0);

        let dim: u32 = 2;
        let reduce_dim = 0;
        let l_build = 10;
        let max_degree = 20;

        let index_ptr = unsafe {
            create_index(
                ctx.get(),
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
                callbacks.filter_callback(),
            )
        };

        (index_ptr, ctx)
    }

    #[test]
    fn basic_create_index() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        assert!(!index_ptr.is_null());

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn create_index_with_invalid_metric_returns_null() {
        let store = Store;

        // Test with invalid metric type values — passed as raw i32
        let invalid_metrics = [-1, -2, 99, i32::MAX, i32::MIN];

        for invalid_metric in invalid_metrics {
            let (index_ptr, _ctx) =
                create_test_index_with_metric(&store, VectorQuantType::NoQuant, invalid_metric);
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
            let (index_ptr, ctx) =
                create_test_index_with_metric(&store, VectorQuantType::NoQuant, valid_metric);
            assert!(
                !index_ptr.is_null(),
                "Expected non-null for valid metric_type_raw={}",
                valid_metric
            );
            unsafe {
                drop_index(ctx.get(), index_ptr);
            }
        }
    }

    #[test]
    fn add_check_and_remove_vector() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        // Vector id with 4 bytes size
        let garnet_vector_id = 42u32;
        let id_bytes = bytemuck::bytes_of(&garnet_vector_id);

        let vector: [f32; 2] = [1.0, 2.0];
        let vector_bytes = bytemuck::cast_slice(&vector);
        let vector_len = 2;

        let attributes_bytes = b"wololo";
        let attributes_len = attributes_bytes.len();

        let result = unsafe {
            insert(
                ctx.get(),
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_len,
            )
        };

        assert!(result > 0);

        // Confirm vector exists using FFI function
        let exists = unsafe {
            check_external_id_valid(ctx.get(), index_ptr, id_bytes.as_ptr(), id_bytes.len())
        };
        assert!(exists);

        let mut cardinality = unsafe { card(ctx.get(), index_ptr) };
        assert_eq!(cardinality, 1);

        let removed = unsafe { remove(ctx.get(), index_ptr, id_bytes.as_ptr(), id_bytes.len()) };
        assert!(removed);

        // Currently we're not tracking deletions for cardinality, so it should still be 1
        cardinality = unsafe { card(ctx.get(), index_ptr) };
        assert_eq!(cardinality, 1);

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn update_vector_attributes() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        // Vector id with 4 bytes size
        let garnet_vector_id = 42u32;
        let id_bytes = bytemuck::bytes_of(&garnet_vector_id);

        // Try to update attributes from non-existing vector
        let attributes1 = b"wololo";
        let set_attribute_result1 = unsafe {
            set_attribute(
                ctx.get(),
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

        let result = unsafe {
            insert(
                ctx.get(),
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                vector_bytes.as_ptr(),
                vector_bytes.len() / 4,
                attributes1.as_ptr(),
                attributes1.len(),
            )
        };
        assert!(result > 0);

        // Set attributes after insertion
        let attributes2 = b"new_attributes";
        let set_attribute_result2 = unsafe {
            set_attribute(
                ctx.get(),
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
                ctx.get(),
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
                ctx.get(),
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                empty_attribute.as_ptr(),
                0,
            )
        };
        assert!(set_attribute_result4);

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn external_id_exists_lifecycle() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

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
            check_external_id_valid(ctx.get(), index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists1, "EID 1 should not exist initially");

        // Add vector with EID 1
        let insert_result1 = unsafe {
            insert(
                ctx.get(),
                index_ptr,
                eid1_bytes.as_ptr(),
                eid1_bytes.len(),
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_bytes.len(),
            )
        };
        assert!(insert_result1 > 0, "Insert with EID 1 should succeed");

        // Check external_id exists with EID 1 (should exist after insert)
        let exists2 = unsafe {
            check_external_id_valid(ctx.get(), index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(exists2, "EID 1 should exist after insert");

        // Remove vector with EID 1
        let removed =
            unsafe { remove(ctx.get(), index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len()) };
        assert!(removed, "Remove with EID 1 should succeed");

        // Check external_id exists with EID 1 (should not exist after removal)
        let exists3 = unsafe {
            check_external_id_valid(ctx.get(), index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists3, "EID 1 should not exist after removal");

        // Add vector with EID 2
        let insert_result2 = unsafe {
            insert(
                ctx.get(),
                index_ptr,
                eid2_bytes.as_ptr(),
                eid2_bytes.len(),
                vector_bytes.as_ptr(),
                vector_len,
                attributes_bytes.as_ptr(),
                attributes_bytes.len(),
            )
        };
        assert!(insert_result2 > 0, "Insert with EID 2 should succeed");

        // Check external_id exists with EID 2 (should exist after insert)
        let exists4 = unsafe {
            check_external_id_valid(ctx.get(), index_ptr, eid2_bytes.as_ptr(), eid2_bytes.len())
        };
        assert!(exists4, "EID 2 should exist after insert");

        // Check external_id exists with EID 1 (should still not exist)
        let exists5 = unsafe {
            check_external_id_valid(ctx.get(), index_ptr, eid1_bytes.as_ptr(), eid1_bytes.len())
        };
        assert!(!exists5, "EID 1 should still not exist");

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn internal_id_exists_lifecycle() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        let bad_iid_bytes = [0u8; 5];
        let exists0 = unsafe {
            check_internal_id_valid(
                ctx.get(),
                index_ptr,
                bad_iid_bytes.as_ptr(),
                bad_iid_bytes.len(),
            )
        };
        assert!(!exists0, "Bad ID should not exist");

        let eid = 1u32;
        let eid_bytes = bytemuck::bytes_of(&eid);

        let vector: [f32; 2] = [1.0, 2.0];
        let vector_bytes = bytemuck::cast_slice(&vector);
        let vector_len = 2;

        // First insert will get ID=1
        let iid = 1u32;
        let iid_bytes = bytemuck::bytes_of(&iid);

        // Check internal ID does not exist
        let exists1 = unsafe {
            check_internal_id_valid(ctx.get(), index_ptr, iid_bytes.as_ptr(), iid_bytes.len())
        };
        assert!(!exists1, "ID should not exist initially");

        // Insert vector
        let insert_result = unsafe {
            insert(
                ctx.get(),
                index_ptr,
                eid_bytes.as_ptr(),
                eid_bytes.len(),
                vector_bytes.as_ptr(),
                vector_len,
                ptr::null(),
                0,
            )
        };
        assert!(insert_result > 0, "Insert should succeed");

        // Check internal id exists
        let exists2 = unsafe {
            check_internal_id_valid(ctx.get(), index_ptr, iid_bytes.as_ptr(), iid_bytes.len())
        };
        assert!(exists2, "ID should exist after insert");

        // Remove vector
        let removed = unsafe { remove(ctx.get(), index_ptr, eid_bytes.as_ptr(), eid_bytes.len()) };
        assert!(removed, "Remove with EID 1 should succeed");

        // Check internal ID does not exist now
        let exists3 = unsafe {
            check_internal_id_valid(ctx.get(), index_ptr, iid_bytes.as_ptr(), iid_bytes.len())
        };
        assert!(!exists3, "ID should not exist after removal");

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    /// Using u64 external IDs, insert some vectors and ensure search results are same.
    #[test]
    fn search_with_large_external_ids() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        let id1 = 1234u64;
        let v1 = &[1.0f32, 0.0];
        let id2 = 5678u64;
        let v2 = &[0.0f32, 1.0];

        let id1_bytes = bytemuck::bytes_of(&id1);
        let id2_bytes = bytemuck::bytes_of(&id2);

        assert!(
            unsafe {
                insert(
                    ctx.get(),
                    index_ptr,
                    id1_bytes.as_ptr(),
                    id1_bytes.len(),
                    bytemuck::cast_slice::<f32, u8>(v1).as_ptr(),
                    v1.len(),
                    b"".as_ptr(),
                    0,
                )
            } > 0
        );

        assert!(
            unsafe {
                insert(
                    ctx.get(),
                    index_ptr,
                    id2_bytes.as_ptr(),
                    id2_bytes.len(),
                    bytemuck::cast_slice::<f32, u8>(v2).as_ptr(),
                    v2.len(),
                    b"".as_ptr(),
                    0,
                )
            } > 0
        );

        let qv = &[0.0f32, 0.0];
        let mut output_id_buffer = vec![0u8; 2 * (mem::size_of::<u64>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; 2];

        let count = unsafe {
            search_vector(
                ctx.get(),
                index_ptr,
                bytemuck::cast_slice::<f32, u8>(qv).as_ptr(),
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
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn search_element() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        let id1 = 1234u64;
        let v1 = &[1.0f32, 0.0];
        let id2 = 5678u64;
        let v2 = &[0.0f32, 1.0];

        let id1_bytes = bytemuck::bytes_of(&id1);
        let id2_bytes = bytemuck::bytes_of(&id2);

        assert!(
            unsafe {
                insert(
                    ctx.get(),
                    index_ptr,
                    id1_bytes.as_ptr(),
                    id1_bytes.len(),
                    bytemuck::cast_slice::<f32, u8>(v1).as_ptr(),
                    v1.len(),
                    b"".as_ptr(),
                    0,
                )
            } > 0
        );

        assert!(
            unsafe {
                insert(
                    ctx.get(),
                    index_ptr,
                    id2_bytes.as_ptr(),
                    id2_bytes.len(),
                    bytemuck::cast_slice::<f32, u8>(v2).as_ptr(),
                    v2.len(),
                    b"".as_ptr(),
                    0,
                )
            } > 0
        );

        let mut output_id_buffer = vec![0u8; 2 * (mem::size_of::<u64>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; 2];

        let count = unsafe {
            crate::search_element(
                ctx.get(),
                index_ptr,
                id1_bytes.as_ptr(),
                id1_bytes.len(),
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

        match (output_ids[0], output_ids[1]) {
            (1234u64, 5678u64) => {
                assert_eq!(output_dists[0], 0.0);
                assert_eq!(output_dists[1], 2.0);
            }
            (5678u64, 1234u64) => {
                assert_eq!(output_dists[0], 2.0);
                assert_eq!(output_dists[1], 0.0);
            }
            _ => {
                panic!("got unexpected ids {} and {}", output_ids[0], output_ids[1]);
            }
        }

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn continue_search() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        let mut output_id_buffer = vec![0u8; 2 * (mem::size_of::<u64>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; 2];
        let res = unsafe {
            crate::continue_search(
                ctx.get(),
                index_ptr,
                ptr::null_mut(),
                output_id_buffer.as_mut_ptr(),
                output_id_buffer.len(),
                output_dists.as_mut_ptr(),
                output_dists.len(),
                ptr::null_mut(),
            )
        };
        assert_eq!(res, -1);
    }

    /// Helper to insert a vector with u32 external ID and FP32 data.
    fn insert_f32_vector(
        ctx: &Context,
        index_ptr: *const c_void,
        eid: u32,
        vector: &[f32],
    ) -> InsertResult {
        let id_bytes = bytemuck::bytes_of(&eid);
        let vector_bytes = bytemuck::cast_slice(vector);
        unsafe {
            insert(
                ctx.get(),
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                vector_bytes.as_ptr(),
                vector.len(),
                b"".as_ptr(),
                0,
            )
            .into()
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
                ctx.get(),
                index_ptr,
                query_bytes.as_ptr(),
                query.len(),
                0.0,
                (k * 2) as u32,
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
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        unsafe {
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]),
                InsertResult::Success
            );
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]),
                InsertResult::Success
            );
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]),
                InsertResult::Success
            );

            let (ids, _dists) = do_search(&ctx, index_ptr, &[1.0, 0.0], 3, None);
            assert!(ids.len() >= 2, "should return at least 2 vectors");
            // Closest to [1,0] should be id=10 (exact match)
            assert_eq!(ids[0], 10);

            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn search_with_null_bitmap_same_as_unfiltered() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);

        unsafe {
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]),
                InsertResult::Success
            );
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]),
                InsertResult::Success
            );

            // Null bitmap should behave like no filter
            let (ids_null, dists_null) = do_search(&ctx, index_ptr, &[1.0, 0.0], 2, None);
            let (ids_empty, dists_empty) = do_search(&ctx, index_ptr, &[1.0, 0.0], 2, Some(&[]));

            // Both should return same results (empty slice triggers has_filter=false
            // because bitmap_len=0)
            assert_eq!(ids_null, ids_empty);
            assert_eq!(dists_null, dists_empty);

            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn basic_quant_bootstrap_lifecycle_bin() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::Bin);
        let index = unsafe { &*index_ptr.cast::<Index>() };

        let quantizer = Spherical1Bit::new(2);
        let required_vectors = quantizer.required_vectors();

        let mut rng = rand::rng();

        // pre-quantization phase

        assert_eq!(index.inner.approximate_count(), 0);
        for id in 0..required_vectors - 1 {
            let v = [rng.random(), rng.random()];
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, id as u32, &v),
                InsertResult::Success
            );
        }
        assert_eq!(
            index.inner.approximate_count() as usize,
            required_vectors - 1
        );

        // transition phase

        let v = [rng.random(), rng.random()];
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, required_vectors as u32 - 1, &v),
            InsertResult::SuccessStartTraining
        );

        // signal to train the quantizer

        let res = unsafe { build_quant_table(ctx.get(), index_ptr) };
        assert!(res, "quantizer training failed");

        // new inserts will be quantized

        let v = [rng.random(), rng.random()];
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, required_vectors as u32, &v),
            InsertResult::Success
        );

        // previous insert is unquantized; inserted before training
        let iid = required_vectors as u32;
        assert!(
            store
                .get(ctx.term(Term::Quantized).get(), bytemuck::bytes_of(&iid))
                .is_none()
        );

        // latest insert is quantized
        let iid = required_vectors as u32 + 1; // +1 to account for the start vector
        let qv = store
            .get(ctx.term(Term::Quantized).get(), bytemuck::bytes_of(&iid))
            .expect("missing quant vector");
        assert_eq!(qv.len(), quantizer.bytes());

        // backfill quant vectors

        unsafe { backfill_quant_vectors(ctx.get(), index_ptr, 0, 1) };

        // all previous inserts are now quantized
        for iid in 1..=required_vectors as u32 {
            let qv = store
                .get(ctx.term(Term::Quantized).get(), bytemuck::bytes_of(&iid))
                .expect("missing quant vector");
            assert_eq!(qv.len(), quantizer.bytes());
        }

        // do a search
        let qv = [0.5f32, 0.5];
        let (ids, _dists) = do_search(&ctx, index_ptr, &qv, 10, None);
        assert!(!ids.is_empty(), "no results found");

        // delete some vectors
        let mut to_delete = (0..required_vectors as u32).collect::<Vec<_>>();
        to_delete.shuffle(&mut rng);
        for id in to_delete.into_iter().take(100) {
            assert!(unsafe {
                remove(
                    ctx.get(),
                    index_ptr,
                    bytemuck::bytes_of(&id).as_ptr(),
                    mem::size_of::<u32>(),
                )
            });
        }

        // do another search
        let qv = [0.5f32, 0.5];
        let (ids, _dists) = do_search(&ctx, index_ptr, &qv, 10, None);
        assert!(!ids.is_empty(), "no results found");

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn recreate_basic() {
        let store = Store;
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::Q8);

        // add vectors
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]),
            InsertResult::Success
        );
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]),
            InsertResult::Success
        );
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]),
            InsertResult::Success
        );

        // do a search; save results
        let qv = [0.0f32, 0.0];
        let (orig_ids, orig_dists) = do_search(&ctx, index_ptr, &qv, 10, None);
        assert!(!orig_ids.is_empty());
        assert!(!orig_dists.is_empty());

        let orig_num_vectors = unsafe { card(ctx.get(), index_ptr) } as usize;

        // drop index
        unsafe {
            drop_index(ctx.get(), index_ptr);
        }

        // create_index with the same store
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::Q8);

        // check num vectors is the same
        let num_vectors = unsafe { card(ctx.get(), index_ptr) } as usize;
        assert_eq!(num_vectors, orig_num_vectors);

        // do a search; check results match above
        let (ids, dists) = do_search(&ctx, index_ptr, &qv, 10, None);
        assert_eq!(ids, orig_ids);
        assert_eq!(dists, orig_dists);

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }
}
