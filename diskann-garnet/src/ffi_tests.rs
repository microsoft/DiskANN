/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod tests {
    use std::{
        ffi::c_void,
        mem, ptr,
        sync::{
            Barrier,
            atomic::{AtomicBool, Ordering},
        },
        thread,
    };

    use diskann_utils::views::Matrix;
    use diskann_vector::distance::Metric;
    use rand::{Rng, seq::SliceRandom};

    use crate::{
        ImportResult, Index, IndexState, InsertResult, Overflow, VectorQuantType,
        backfill_quant_vectors, build_quant_table, can_import, card, check_external_id_valid,
        check_internal_id_valid, create_index, drop_index, finish_import,
        garnet::{Context, Term, WriteCallback},
        import_term, insert,
        quantization::{GarnetQuantizer, MinMax8Bit, Spherical1Bit},
        remove, search_neighbors, search_vector, set_attribute, set_quant_state,
        test_utils::{STORE, Store},
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
        create_test_index_with_write_callback(
            store,
            quant_type,
            metric_type,
            store.callbacks().write_callback(),
        )
    }

    fn create_test_index_with_write_callback(
        store: &Store,
        quant_type: VectorQuantType,
        metric_type: i32,
        write_callback: WriteCallback,
    ) -> (*const c_void, Context) {
        let callbacks = store.callbacks();
        let ctx = Context::new(0);
        let mut quant_needed = false;

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
                write_callback,
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
                callbacks.filter_callback(),
                callbacks.log_callback(),
                &mut quant_needed,
            )
        };

        (index_ptr, ctx)
    }

    #[test]
    fn basic_create_index() {
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        assert!(!index_ptr.is_null());

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn create_index_with_invalid_metric_returns_null() {
        let store = Store::new();

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
        let store = Store::new();

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
        let store = Store::new();
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

        cardinality = unsafe { card(ctx.get(), index_ptr) };
        assert_eq!(cardinality, 0);

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn external_id_ffi_calls_reject_null_or_empty_ids() {
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        let valid_id = b"vector";
        let internal_id = 1u32.to_ne_bytes();
        let vector = [1.0f32, 2.0];
        let vector_bytes: &[u8] = bytemuck::cast_slice(&vector);
        let attributes = b"attributes";
        let mut output_ids = [0u8; 128];
        let mut output_distances = [0f32; 20];
        let mut overflow = ptr::null_mut();

        for (case, id_data, id_len) in [
            ("null and zero length", ptr::null(), 0),
            ("non-null and zero length", valid_id.as_ptr(), 0),
            ("null and nonzero length", ptr::null(), valid_id.len()),
        ] {
            unsafe {
                assert_eq!(
                    insert(
                        ctx.get(),
                        index_ptr,
                        id_data,
                        id_len,
                        vector_bytes.as_ptr(),
                        vector.len(),
                        attributes.as_ptr(),
                        attributes.len(),
                    ),
                    u8::from(InsertResult::Fail),
                    "insert: {case}"
                );
                assert!(
                    !set_attribute(
                        ctx.get(),
                        index_ptr,
                        id_data,
                        id_len,
                        attributes.as_ptr(),
                        attributes.len(),
                    ),
                    "set_attribute: {case}"
                );
                assert_eq!(
                    crate::search_element(
                        ctx.get(),
                        index_ptr,
                        id_data,
                        id_len,
                        0.0,
                        10,
                        ptr::null(),
                        0,
                        0,
                        output_ids.as_mut_ptr(),
                        output_ids.len(),
                        output_distances.as_mut_ptr(),
                        output_distances.len(),
                        4,
                        &mut overflow,
                    ),
                    -1,
                    "search_element: {case}"
                );
                assert!(
                    !remove(ctx.get(), index_ptr, id_data, id_len),
                    "remove: {case}"
                );
                assert!(
                    !check_external_id_valid(ctx.get(), index_ptr, id_data, id_len),
                    "check_external_id_valid: {case}"
                );
                assert_eq!(
                    search_neighbors(
                        ctx.get(),
                        index_ptr,
                        id_data,
                        id_len,
                        output_ids.as_mut_ptr(),
                        output_ids.len(),
                        output_distances.as_mut_ptr(),
                        output_distances.len(),
                        &mut overflow,
                    ),
                    -1,
                    "search_neighbors: {case}"
                );
                assert!(
                    !import_term(
                        ctx.get(),
                        index_ptr,
                        Term::IntMap as u32,
                        id_data,
                        id_len,
                        internal_id.as_ptr(),
                        internal_id.len(),
                    ),
                    "import_term INTMAP: {case}"
                );
                assert!(
                    !import_term(
                        ctx.get(),
                        index_ptr,
                        Term::ExtMap as u32,
                        internal_id.as_ptr(),
                        internal_id.len(),
                        id_data,
                        id_len,
                    ),
                    "import_term EXTMAP: {case}"
                );
                assert_eq!(card(ctx.get(), index_ptr), 0, "cardinality: {case}");
            }
        }

        unsafe {
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn can_import_tracks_eligibility() {
        for quant_type in [VectorQuantType::NoQuant, VectorQuantType::Q8] {
            let store = Store::new();
            let (index_ptr, ctx) = create_test_index(&store, quant_type);
            assert_eq!(
                unsafe { can_import(ctx.get(), index_ptr) },
                quant_type == VectorQuantType::NoQuant
            );
            if quant_type == VectorQuantType::Q8 {
                let quantizer = MinMax8Bit::new(2, Metric::L2).unwrap();
                let state = quantizer.serialize().unwrap();
                assert!(unsafe {
                    set_quant_state(ctx.get(), index_ptr, state.as_ptr(), state.len())
                });
            }

            assert!(unsafe { can_import(ctx.get(), index_ptr) });
            // A duplicate call to make sure calling can_import doesn't
            // itself diable imports.
            assert!(unsafe { can_import(ctx.get(), index_ptr) });

            let id = 1u32.to_ne_bytes();
            let attributes = b"attributes";
            assert!(unsafe {
                import_term(
                    ctx.get(),
                    index_ptr,
                    Term::Attributes as u32,
                    id.as_ptr(),
                    id.len(),
                    attributes.as_ptr(),
                    attributes.len(),
                )
            });
            assert!(unsafe { can_import(ctx.get(), index_ptr) });
            assert_eq!(
                unsafe { finish_import(ctx.get(), index_ptr, 0, 1) },
                u8::from(ImportResult::TaskFailed)
            );
            assert!(!unsafe { can_import(ctx.get(), index_ptr) });
            unsafe { drop_index(ctx.get(), index_ptr) };

            let (index_ptr, ctx) = create_test_index(&store, quant_type);
            assert!(!unsafe { can_import(ctx.get(), index_ptr) });
            unsafe { drop_index(ctx.get(), index_ptr) };
        }

        for insert_vector in [false, true] {
            let store = Store::new();
            let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
            assert!(unsafe { can_import(ctx.get(), index_ptr) });
            if insert_vector {
                assert_eq!(
                    insert_f32_vector(&ctx, index_ptr, 1, &[1.0, 2.0]),
                    InsertResult::Success
                );
            } else {
                assert_eq!(unsafe { card(ctx.get(), index_ptr) }, 0);
            }
            assert!(!unsafe { can_import(ctx.get(), index_ptr) });
            assert!(!unsafe { can_import(ctx.get(), index_ptr) });
            unsafe { drop_index(ctx.get(), index_ptr) };
        }
    }

    #[test]
    fn import_terms_and_finish_q8() {
        let quantizer = MinMax8Bit::new(2, Metric::L2).unwrap();
        check_import_terms_and_finish(VectorQuantType::Q8, Some(&quantizer));
    }

    #[test]
    fn import_terms_and_finish_noquant() {
        check_import_terms_and_finish(VectorQuantType::NoQuant, None);
    }

    #[test]
    fn import_terms_and_finish_bin() {
        let quantizer = Spherical1Bit::new(Metric::L2, 2);
        let mut training_data = Matrix::new(0.0f32, quantizer.required_vectors(), 2);
        for row in 0..quantizer.required_vectors() {
            training_data.row_mut(row).fill((row % 100 + 1) as f32);
        }
        quantizer
            .train(Metric::L2, training_data.as_view())
            .unwrap();
        check_import_terms_and_finish(VectorQuantType::Bin, Some(&quantizer));
    }

    #[test]
    fn import_terms_and_finish_multi_worker() {
        check_import_terms_and_finish_with(
            VectorQuantType::NoQuant,
            None,
            None,
            |ctx, index_ptr| {
                const TASK_COUNT: usize = 104;
                let index = unsafe { &*index_ptr.cast::<Index>() };
                assert_eq!(
                    unsafe { finish_import(ctx.get(), index_ptr, 0, TASK_COUNT) },
                    0
                );
                let snapshot = STORE.with(Clone::clone);
                let barrier = Barrier::new(4);
                thread::scope(|scope| {
                    for worker in 0..4 {
                        let snapshot = snapshot.clone();
                        let barrier = &barrier;
                        scope.spawn(move || {
                            STORE.with(|store| {
                                for (key, value) in snapshot {
                                    store.insert(key, value);
                                }
                            });
                            barrier.wait();
                            for task_idx in (worker + 1..TASK_COUNT - 1).step_by(4) {
                                assert_eq!(
                                    unsafe {
                                        finish_import(
                                            ctx.get(),
                                            ptr::from_ref(index).cast(),
                                            task_idx,
                                            TASK_COUNT,
                                        )
                                    },
                                    0
                                );
                            }
                        });
                    }
                });
                assert_eq!(
                    index.state.load(Ordering::Acquire),
                    IndexState::NoStartPoints as usize
                );
                assert_eq!(
                    unsafe { finish_import(ctx.get(), index_ptr, TASK_COUNT - 1, TASK_COUNT) },
                    0
                );
                assert_eq!(
                    index.state.load(Ordering::Acquire),
                    IndexState::Ready as usize
                );
            },
        );
    }

    #[test]
    fn import_terms_and_finish_persistence_failure() {
        static FAIL_WRITE: AtomicBool = AtomicBool::new(false);
        unsafe extern "C" fn write(
            context: u64,
            key: *const u8,
            key_len: usize,
            value: *const u8,
            value_len: usize,
        ) -> bool {
            !FAIL_WRITE.swap(false, Ordering::AcqRel)
                && unsafe {
                    (Store::attach().callbacks().write_callback())(
                        context, key, key_len, value, value_len,
                    )
                }
        }

        check_import_terms_and_finish_with(
            VectorQuantType::NoQuant,
            None,
            Some(write),
            |ctx, index_ptr| {
                FAIL_WRITE.store(true, Ordering::Release);
                assert_eq!(unsafe { card(ctx.get(), index_ptr) }, u64::MAX);
                FAIL_WRITE.store(true, Ordering::Release);
                assert_eq!(
                    unsafe { finish_import(ctx.get(), index_ptr, 0, 1) },
                    u8::from(ImportResult::TaskFailed)
                );
                let index = unsafe { &*index_ptr.cast::<Index>() };
                assert_eq!(
                    index.state.load(Ordering::Acquire),
                    IndexState::NoStartPoints as usize
                );
                assert_eq!(
                    unsafe { finish_import(ctx.get(), index_ptr, 0, 1) },
                    u8::from(ImportResult::Success)
                );
                assert_eq!(
                    Store::attach().get(
                        ctx.term(Term::Metadata).get(),
                        &u32::from_be_bytes(*b"_imp").to_ne_bytes()
                    ),
                    Some(vec![0u8])
                );
            },
        );
    }

    /// Exercises both import orderings, finalization, and subsequent insertion and search.
    fn check_import_terms_and_finish(
        quant_type: VectorQuantType,
        quantizer: Option<&dyn GarnetQuantizer>,
    ) {
        check_import_terms_and_finish_with(quant_type, quantizer, None, |ctx, index_ptr| {
            assert_eq!(
                unsafe { finish_import(ctx.get(), index_ptr, 0, 1) },
                u8::from(ImportResult::Success)
            );
        });
    }

    fn check_import_terms_and_finish_with(
        quant_type: VectorQuantType,
        quantizer: Option<&dyn GarnetQuantizer>,
        write_callback: Option<WriteCallback>,
        finalize: impl Fn(&Context, *const c_void),
    ) {
        const VECTOR_COUNT: usize = 100;
        const MAX_DEGREE: usize = 20;

        let quant_state = quantizer.map(|quantizer| quantizer.serialize().unwrap());
        let terms: Vec<_> = (1..=VECTOR_COUNT as u32)
            .map(|id| {
                let vector = [id as f32; 2];
                let mut vector_terms =
                    vec![(Term::Vector as u32, bytemuck::cast_slice(&vector).to_vec())];
                if let Some(quantizer) = quantizer {
                    let mut quantized = vec![0u8; quantizer.bytes()];
                    quantizer.compress(&vector, &mut quantized).unwrap();
                    vector_terms.push((Term::Quantized as u32, quantized));
                }
                let mut neighbors = [0u32; MAX_DEGREE + 1];
                for (offset, neighbor) in neighbors[..5].iter_mut().enumerate() {
                    *neighbor =
                        (id + VECTOR_COUNT as u32 - 2 - offset as u32) % VECTOR_COUNT as u32 + 1;
                }
                neighbors[MAX_DEGREE] = 5;
                let id_bytes = id.to_ne_bytes();

                vector_terms.extend([
                    (
                        Term::Neighbors as u32,
                        bytemuck::cast_slice(&neighbors).to_vec(),
                    ),
                    (Term::Attributes as u32, id_bytes.to_vec()),
                    (Term::IntMap as u32, id_bytes.to_vec()),
                    (Term::ExtMap as u32, id_bytes.to_vec()),
                ]);
                vector_terms
            })
            .collect();
        let term_count = terms[0].len();

        for by_term in [true, false] {
            let store = Store::new();
            let (index_ptr, ctx) = create_test_index_with_write_callback(
                &store,
                quant_type,
                Metric::L2 as i32,
                write_callback.unwrap_or_else(|| store.callbacks().write_callback()),
            );
            assert!(!index_ptr.is_null());
            let index = unsafe { &*index_ptr.cast::<Index>() };
            assert_eq!(index.inner.max_degree(), MAX_DEGREE);
            if let Some(quant_state) = &quant_state {
                let id = 1u32.to_ne_bytes();
                for (term, value) in &terms[0] {
                    assert!(
                        !unsafe {
                            import_term(
                                ctx.get(),
                                index_ptr,
                                *term,
                                id.as_ptr(),
                                id.len(),
                                value.as_ptr(),
                                value.len(),
                            )
                        },
                        "import accepted without preset quant state: quant_type={quant_type:?}, by_term={by_term}, term={term}"
                    );
                    assert!(store.get(ctx.get() | u64::from(*term), &id).is_none());
                }
                assert_eq!(unsafe { card(ctx.get(), index_ptr) }, 0);
                assert!(!unsafe {
                    check_internal_id_valid(ctx.get(), index_ptr, id.as_ptr(), id.len())
                });
                assert!(!unsafe {
                    check_external_id_valid(ctx.get(), index_ptr, id.as_ptr(), id.len())
                });
                assert!(unsafe {
                    set_quant_state(
                        ctx.get(),
                        index_ptr,
                        quant_state.as_ptr(),
                        quant_state.len(),
                    )
                });
            }

            for position in 0..VECTOR_COUNT * term_count {
                let (vector_index, term_index) = if by_term {
                    (position % VECTOR_COUNT, position / VECTOR_COUNT)
                } else {
                    (position / term_count, position % term_count)
                };
                let id = (vector_index as u32 + 1).to_ne_bytes();
                let (term, value) = &terms[vector_index][term_index];
                assert!(
                    unsafe {
                        import_term(
                            ctx.get(),
                            index_ptr,
                            *term,
                            id.as_ptr(),
                            id.len(),
                            value.as_ptr(),
                            value.len(),
                        )
                    },
                    "import failed: by_term={by_term}, id={}, term={term}",
                    vector_index + 1
                );
            }

            assert!(unsafe { can_import(ctx.get(), index_ptr) });
            finalize(&ctx, index_ptr);
            assert!(!unsafe { can_import(ctx.get(), index_ptr) });
            for rejected_id in [1u32, 101] {
                let id = rejected_id.to_ne_bytes();
                for (term, value) in &terms[1] {
                    let term_context = ctx.get() | u64::from(*term);
                    let before = store.get(term_context, &id);
                    assert!(
                        !unsafe {
                            import_term(
                                ctx.get(),
                                index_ptr,
                                *term,
                                id.as_ptr(),
                                id.len(),
                                value.as_ptr(),
                                value.len(),
                            )
                        },
                        "import accepted after finish_import: by_term={by_term}, id={rejected_id}, term={term}"
                    );
                    assert_eq!(
                        store.get(term_context, &id),
                        before,
                        "rejected import changed storage: by_term={by_term}, id={rejected_id}, term={term}"
                    );
                }
            }
            let unused_id = 101u32.to_ne_bytes();
            assert!(!unsafe {
                check_internal_id_valid(ctx.get(), index_ptr, unused_id.as_ptr(), unused_id.len())
            });
            assert!(!unsafe {
                check_external_id_valid(ctx.get(), index_ptr, unused_id.as_ptr(), unused_id.len())
            });
            assert_eq!(unsafe { card(ctx.get(), index_ptr) }, VECTOR_COUNT as u64);

            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 101, &[101.0, 101.0]),
                InsertResult::Success,
                "insert failed: by_term={by_term}"
            );
            assert_eq!(
                unsafe { card(ctx.get(), index_ptr) },
                VECTOR_COUNT as u64 + 1
            );
            let added_id = 101u32.to_ne_bytes();
            assert!(unsafe {
                check_external_id_valid(ctx.get(), index_ptr, added_id.as_ptr(), added_id.len())
            });

            let (ids, distances) = do_search(&ctx, index_ptr, &[50.0, 50.0], 5, None);
            assert_eq!(ids.first(), Some(&50), "nearest vector: by_term={by_term}");
            let mut sorted_ids = ids;
            sorted_ids.sort_unstable();
            assert_eq!(sorted_ids, [48, 49, 50, 51, 52]);
            assert_eq!(distances.len(), 5);
            assert!(distances.iter().all(|distance| distance.is_finite()));

            unsafe {
                drop_index(ctx.get(), index_ptr);
            }
        }
    }

    #[test]
    fn update_vector_attributes() {
        let store = Store::new();
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
        let store = Store::new();
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
        let store = Store::new();
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
        let store = Store::new();
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
        let mut overflow = ptr::null_mut();

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
                1,
                &mut overflow,
            )
        };

        assert_eq!(count, 2);
        assert!(overflow.is_null());

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
        let store = Store::new();
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
        let mut overflow = ptr::null_mut();

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
                1,
                &mut overflow,
            )
        };

        assert_eq!(count, 2);
        assert!(overflow.is_null());

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
    fn search_element_writes_overflow_output() {
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        let id = 1u32;
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, id, &[0.0, 1.0]),
            InsertResult::Success
        );

        let mut output_ids: [u8; 0] = [];
        let mut output_distances: [f32; 1] = [0f32];
        let mut overflow = ptr::null_mut();
        let count = unsafe {
            crate::search_element(
                ctx.get(),
                index_ptr,
                bytemuck::bytes_of(&id).as_ptr(),
                mem::size_of::<u32>(),
                1.0,
                10,
                ptr::null(),
                0,
                0,
                output_ids.as_mut_ptr(),
                output_ids.len(),
                output_distances.as_mut_ptr(),
                output_distances.len(),
                1,
                &mut overflow,
            )
        };

        assert_eq!(count, 0);
        assert!(!overflow.is_null());

        unsafe {
            drop(Overflow::from_ptr(overflow));
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn overflow_results() {
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        let mut output_id_buffer = vec![0u8; 2 * (mem::size_of::<u64>() + mem::size_of::<u32>())];
        let mut output_dists = vec![0f32; 2];
        let res = unsafe {
            crate::overflow_results(
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
        let mut overflow = ptr::null_mut();

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
                1,
                &mut overflow,
            )
        };

        if !overflow.is_null() {
            unsafe { drop(Overflow::from_ptr(overflow)) };
        }

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
    fn search_vector_writes_overflow_output() {
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::NoQuant);
        let vector = [0.0f32, 1.0];
        assert_eq!(
            insert_f32_vector(&ctx, index_ptr, 1, &vector),
            InsertResult::Success
        );

        let query_bytes = bytemuck::cast_slice(&vector);
        let mut output_ids: [u8; 0] = [];
        let mut output_distances: [f32; 1] = [0f32];
        let mut overflow = ptr::null_mut();
        let count = unsafe {
            search_vector(
                ctx.get(),
                index_ptr,
                query_bytes.as_ptr(),
                vector.len(),
                1.0,
                10,
                ptr::null(),
                0,
                0,
                output_ids.as_mut_ptr(),
                output_ids.len(),
                output_distances.as_mut_ptr(),
                output_distances.len(),
                1,
                &mut overflow,
            )
        };

        assert_eq!(count, 0);
        assert!(!overflow.is_null());

        unsafe {
            drop(Overflow::from_ptr(overflow));
            drop_index(ctx.get(), index_ptr);
        }
    }

    #[test]
    fn search_without_filter() {
        let store = Store::new();
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
        let store = Store::new();
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
        let store = Store::new();
        let (index_ptr, ctx) = create_test_index(&store, VectorQuantType::Bin);
        let index = unsafe { &*index_ptr.cast::<Index>() };

        let quantizer = Spherical1Bit::new(Metric::L2, 2);
        let required_vectors = quantizer.required_vectors();

        let mut rng = rand::rng();

        // pre-quantization phase

        assert_eq!(index.inner.approximate_count(&ctx).unwrap(), 0);
        for id in 0..required_vectors - 1 {
            let v = [rng.random(), rng.random()];
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, id as u32, &v),
                InsertResult::Success
            );
        }
        assert_eq!(
            index.inner.approximate_count(&ctx).unwrap() as usize,
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
    fn recreate_basic_with_float_vectors() {
        for quant_type in [
            VectorQuantType::NoQuant,
            VectorQuantType::Bin,
            VectorQuantType::Q8,
        ] {
            let store = Store::new();
            let (index_ptr, ctx) = create_test_index(&store, quant_type);

            // add vectors
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 10, &[1.0, 0.0]),
                InsertResult::Success,
                "failed to insert id=10 with quant_type={quant_type:?}",
            );
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 20, &[0.0, 1.0]),
                InsertResult::Success,
                "failed to insert id=20 with quant_type={quant_type:?}",
            );
            assert_eq!(
                insert_f32_vector(&ctx, index_ptr, 30, &[1.0, 1.0]),
                InsertResult::Success,
                "failed to insert id=30 with quant_type={quant_type:?}",
            );

            // do a search; save results
            let qv = [0.0f32, 0.0];
            let (orig_ids, orig_dists) = do_search(&ctx, index_ptr, &qv, 10, None);
            assert!(
                !orig_ids.is_empty(),
                "got empty id list for quant_type={quant_type:?}"
            );
            assert!(
                !orig_dists.is_empty(),
                "got empty dist list for quant_type={quant_type:?}"
            );

            let orig_num_vectors = unsafe { card(ctx.get(), index_ptr) } as usize;

            // drop index
            unsafe {
                drop_index(ctx.get(), index_ptr);
            }

            // create_index with the same store
            let (index_ptr, ctx) = create_test_index(&store, quant_type);

            // check num vectors is the same
            let num_vectors = unsafe { card(ctx.get(), index_ptr) } as usize;
            assert_eq!(
                num_vectors, orig_num_vectors,
                "term count mismatch for quant_type={quant_type:?}"
            );

            // do a search; check results match above
            let (ids, dists) = do_search(&ctx, index_ptr, &qv, 10, None);
            assert_eq!(
                ids, orig_ids,
                "recreated search didn't match ids for quant_type={quant_type:?}"
            );
            assert_eq!(
                dists, orig_dists,
                "recreated search didn't match dists for quant_type={quant_type:?}"
            );

            unsafe {
                drop_index(ctx.get(), index_ptr);
            }
        }
    }
}
