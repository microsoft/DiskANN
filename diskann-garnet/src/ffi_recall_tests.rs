/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod tests {
    use std::{collections::HashMap, ffi::c_void, mem, ptr};

    use diskann_vector::PureDistanceFunction;
    use diskann_vector::distance::{Cosine, Metric, SquaredL2};

    use crate::{
        VectorQuantType, VectorValueType, create_index, drop_index, garnet::Context, insert,
        search_vector, test_utils::Store,
    };

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

    /// Wraps a `PureDistanceFunction` into a plain `fn` pointer so it can be
    /// passed to helpers that accept `fn(&[f32], &[f32]) -> f32`.
    fn distance_fn<T>(a: &[f32], b: &[f32]) -> f32
    where
        T: for<'a, 'b> PureDistanceFunction<&'a [f32], &'b [f32], f32>,
    {
        T::evaluate(a, b)
    }

    /// Brute-force k nearest neighbors. Returns the IDs of the k closest
    /// vectors according to `distance_fn`.
    fn brute_force_knn(
        ids: &[String],
        vectors: &[Vec<f32>],
        query: &[f32],
        k: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> Vec<String> {
        let mut scored: Vec<(&String, f32)> = Vec::with_capacity(ids.len());
        for i in 0..ids.len() {
            scored.push((&ids[i], distance_fn(&vectors[i], query)));
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        let mut result = Vec::with_capacity(scored.len());
        for (id, _) in scored {
            result.push(id.clone());
        }
        result
    }

    /// Bucket a set of result IDs by distance, quantizing to integer keys.
    /// Distances are multiplied by `BUCKET_SCALE` (1000) and rounded, so
    /// results within 0.001 of each other land in the same bucket.
    fn bucket_counts(
        result_ids: &[String],
        id_to_vector: &HashMap<&str, &[f32]>,
        query: &[f32],
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> HashMap<i64, usize> {
        const BUCKET_SCALE: f32 = 1000.0;

        let mut counts: HashMap<i64, usize> = HashMap::new();
        for id in result_ids {
            let dist = distance_fn(id_to_vector[id.as_str()], query);
            let key = (dist * BUCKET_SCALE).round() as i64;
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    /// Count intersection by grouping results by distance (handles ties).
    /// When multiple vectors are equidistant, the exact IDs may differ
    /// between brute-force and ANN, but the distance counts must match.
    ///
    /// Distances are bucketed with an epsilon tolerance so that small
    /// floating-point differences do not cause spurious mismatches.
    fn distance_based_intersection(
        vectors: &[Vec<f32>],
        ids: &[String],
        query: &[f32],
        expected: &[String],
        actual: &[String],
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> usize {
        let mut id_to_vector: HashMap<&str, &[f32]> = HashMap::with_capacity(ids.len());
        for i in 0..ids.len() {
            id_to_vector.insert(ids[i].as_str(), &vectors[i]);
        }

        let expected_counts = bucket_counts(expected, &id_to_vector, query, distance_fn);
        let actual_counts = bucket_counts(actual, &id_to_vector, query, distance_fn);

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
        assert_eq!(
            offset,
            output_id_buffer.len(),
            "buffer not fully consumed: parsed {offset} bytes but buffer is {} bytes",
            output_id_buffer.len()
        );
        result_ids
    }

    /// Common helper: create an index, insert vectors, query each, return recall.
    fn run_recall(
        store: &Store,
        dimensions: u32,
        metric: Metric,
        ids: &[String],
        vectors: &[Vec<f32>],
        k: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> f64 {
        store.clear();
        let callbacks = store.callbacks();
        let ctx = Context(0);

        let reduce_dimensions = 0;
        let l_build = 100;
        let max_degree = 32;
        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dimensions,
                reduce_dimensions,
                VectorQuantType::NoQuant,
                metric as i32,
                l_build,
                max_degree,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };
        assert!(!index_ptr.is_null());

        let max_id_len = ids.iter().map(|id| id.len()).max().unwrap_or(0);

        for i in 0..ids.len() {
            assert!(
                insert_f32_vector_str(&ctx, index_ptr, &ids[i], &vectors[i]),
                "insert failed for eid={}",
                ids[i]
            );
        }

        let mut total_matches = 0usize;
        let mut total_expected = 0usize;

        let delta = 2.0_f32;
        let search_exploration_factor = 200_u32;
        let max_filtering_effort = 0_usize;
        let continuation = ptr::null_mut();

        for vec in vectors {
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
                    delta,
                    search_exploration_factor,
                    ptr::null(),
                    0,
                    max_filtering_effort,
                    output_id_buffer.as_mut_ptr(),
                    output_id_buffer.len(),
                    output_dists.as_mut_ptr(),
                    output_dists.len(),
                    continuation,
                )
            };
            assert!(count >= 0, "search failed");

            let result_ids = parse_string_ids(&output_id_buffer, count as usize);
            let expected_ids = brute_force_knn(ids, vectors, vec, k, distance_fn);
            let matches = distance_based_intersection(
                vectors,
                ids,
                vec,
                &expected_ids,
                &result_ids,
                distance_fn,
            );
            total_matches += matches;
            total_expected += expected_ids.len();
        }

        unsafe { drop_index(ctx.0, index_ptr) };

        total_matches as f64 / total_expected as f64
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

    /// Helper: create an L2 index, insert grid vectors, query each, return recall.
    fn run_grid_recall(store: &Store, dimensions: u32, grid_size: usize, k: usize) -> f64 {
        let (ids, vectors) = generate_grid_vectors(dimensions as usize, grid_size);
        run_recall(
            store,
            dimensions,
            Metric::L2,
            &ids,
            &vectors,
            k,
            distance_fn::<SquaredL2>,
        )
    }

    #[test]
    fn grid_l2_recall_1d_100() {
        let store = Store;
        let recall = run_grid_recall(&store, 1, 100, 3);
        assert!(recall >= 0.99, "1D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_2d_10() {
        let store = Store;
        let recall = run_grid_recall(&store, 2, 10, 3);
        assert!(recall >= 0.99, "2D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_3d_7() {
        let store = Store;
        let recall = run_grid_recall(&store, 3, 7, 3);
        assert!(recall >= 0.99, "3D grid recall too low: {recall:.4}");
    }

    #[test]
    fn grid_l2_recall_4d_5() {
        let store = Store;
        let recall = run_grid_recall(&store, 4, 5, 3);
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

    /// Helper: create a cosine index, insert circle vectors, query each, return recall.
    fn run_circle_recall(store: &Store, point_count: usize, radius: f32, k: usize) -> f64 {
        let (ids, vectors) = generate_circle_vectors(point_count, radius);
        run_recall(
            store,
            2,
            Metric::Cosine,
            &ids,
            &vectors,
            k,
            distance_fn::<Cosine>,
        )
    }

    /// Circle with 100 points, radius=1.0, cosine distance, k=5
    #[test]
    fn circle_cosine_recall_r1_100pt() {
        let store = Store;
        let recall = run_circle_recall(&store, 100, 1.0, 5);
        assert!(recall >= 0.99, "circle r=1 recall too low: {recall:.4}");
    }

    /// Circle with 93 points, radius=534.0, cosine distance, k=5
    #[test]
    fn circle_cosine_recall_r534_93pt() {
        let store = Store;
        let recall = run_circle_recall(&store, 93, 534.0, 5);
        assert!(recall >= 0.99, "circle r=534 recall too low: {recall:.4}");
    }

    /// Circle with 50 points, radius=10.0, cosine distance, k=3
    #[test]
    fn circle_cosine_recall_r10_50pt() {
        let store = Store;
        let recall = run_circle_recall(&store, 50, 10.0, 3);
        assert!(recall >= 0.99, "circle r=10 recall too low: {recall:.4}");
    }

    // ── SB8 (signed int8) recall tests ─────────────────────────────────

    /// Helper to insert a vector with a string external ID and SB8 data.
    fn insert_sb8_vector_str(
        ctx: &Context,
        index_ptr: *const c_void,
        eid: &str,
        vector: &[i8],
    ) -> bool {
        let id_bytes = eid.as_bytes();
        let vector_bytes: &[u8] = bytemuck::cast_slice(vector);
        unsafe {
            insert(
                ctx.0,
                index_ptr,
                id_bytes.as_ptr(),
                id_bytes.len(),
                VectorValueType::SB8,
                vector_bytes.as_ptr(),
                vector.len(),
                b"".as_ptr(),
                0,
            )
        }
    }

    /// Common helper: create an index, insert SB8 vectors, query each with SB8, return recall.
    /// Distances are computed in f32 space (SB8 values are converted to f32 internally).
    fn run_sb8_recall(
        store: &Store,
        dimensions: u32,
        metric: Metric,
        ids: &[String],
        vectors_i8: &[Vec<i8>],
        k: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> f64 {
        run_sb8_recall_with_quant(
            store,
            dimensions,
            metric,
            ids,
            vectors_i8,
            k,
            distance_fn,
            VectorQuantType::NoQuant,
        )
    }

    /// Common helper: create an index with specified quant type, insert SB8 vectors, query each
    /// with SB8, return recall. Brute-force comparison uses f32 space; index distance computation
    /// depends on the quant type (f32 for NoQuant, i8 for Q8).
    #[allow(clippy::too_many_arguments)]
    fn run_sb8_recall_with_quant(
        store: &Store,
        dimensions: u32,
        metric: Metric,
        ids: &[String],
        vectors_i8: &[Vec<i8>],
        k: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
        quant_type: VectorQuantType,
    ) -> f64 {
        store.clear();
        let callbacks = store.callbacks();
        let ctx = Context(0);

        let reduce_dimensions = 0;
        let l_build = 100;
        let max_degree = 32;
        let index_ptr = unsafe {
            create_index(
                ctx.0,
                dimensions,
                reduce_dimensions,
                quant_type,
                metric as i32,
                l_build,
                max_degree,
                callbacks.read_callback(),
                callbacks.write_callback(),
                callbacks.delete_callback(),
                callbacks.rmw_callback(),
            )
        };
        assert!(!index_ptr.is_null());

        let max_id_len = ids.iter().map(|id| id.len()).max().unwrap_or(0);

        // Convert i8 vectors to f32 for brute-force comparison
        let vectors_f32: Vec<Vec<f32>> = vectors_i8
            .iter()
            .map(|v| v.iter().map(|&x| x as f32).collect())
            .collect();

        for i in 0..ids.len() {
            assert!(
                insert_sb8_vector_str(&ctx, index_ptr, &ids[i], &vectors_i8[i]),
                "insert failed for eid={}",
                ids[i]
            );
        }

        let mut total_matches = 0usize;
        let mut total_expected = 0usize;

        let delta = 2.0_f32;
        let search_exploration_factor = 200_u32;
        let max_filtering_effort = 0_usize;
        let continuation = ptr::null_mut();

        for (idx, vec_i8) in vectors_i8.iter().enumerate() {
            let query_bytes: &[u8] = bytemuck::cast_slice(vec_i8);
            let max_id_size = mem::size_of::<u32>() + max_id_len;
            let mut output_id_buffer = vec![0u8; k * max_id_size];
            let mut output_dists = vec![0f32; k];

            let count = unsafe {
                search_vector(
                    ctx.0,
                    index_ptr,
                    VectorValueType::SB8,
                    query_bytes.as_ptr(),
                    vec_i8.len(),
                    delta,
                    search_exploration_factor,
                    ptr::null(),
                    0,
                    max_filtering_effort,
                    output_id_buffer.as_mut_ptr(),
                    output_id_buffer.len(),
                    output_dists.as_mut_ptr(),
                    output_dists.len(),
                    continuation,
                )
            };
            assert!(count >= 0, "search failed");

            let result_ids = parse_string_ids(&output_id_buffer, count as usize);
            let expected_ids =
                brute_force_knn(ids, &vectors_f32, &vectors_f32[idx], k, distance_fn);
            let matches = distance_based_intersection(
                &vectors_f32,
                ids,
                &vectors_f32[idx],
                &expected_ids,
                &result_ids,
                distance_fn,
            );
            total_matches += matches;
            total_expected += expected_ids.len();
        }

        unsafe { drop_index(ctx.0, index_ptr) };

        total_matches as f64 / total_expected as f64
    }

    /// Generates `grid_size ^ dimensions` SB8 vectors on a signed integer grid
    /// centered around 0 (values from -(grid_size/2) to +(grid_size/2)-1).
    fn generate_sb8_grid_vectors(
        dimensions: usize,
        grid_size: usize,
    ) -> (Vec<String>, Vec<Vec<i8>>) {
        let total = grid_size.pow(dimensions as u32);
        let offset = (grid_size / 2) as i8;
        let mut ids = Vec::with_capacity(total);
        let mut vectors = Vec::with_capacity(total);

        for i in 0..total {
            let mut vec = vec![0i8; dimensions];
            let mut pos = i;
            for d in (0..dimensions).rev() {
                vec[d] = (pos % grid_size) as i8 - offset;
                pos /= grid_size;
            }
            ids.push(format!("sb8_grid_{:08}_dim{}", i + 1, dimensions));
            vectors.push(vec);
        }

        (ids, vectors)
    }

    /// Helper: create an L2 index, insert SB8 grid vectors, query each, return recall.
    fn run_sb8_grid_recall(store: &Store, dimensions: u32, grid_size: usize, k: usize) -> f64 {
        let (ids, vectors) = generate_sb8_grid_vectors(dimensions as usize, grid_size);
        run_sb8_recall(
            store,
            dimensions,
            Metric::L2,
            &ids,
            &vectors,
            k,
            distance_fn::<SquaredL2>,
        )
    }

    #[test]
    fn sb8_grid_l2_recall_1d_100() {
        let store = Store;
        let recall = run_sb8_grid_recall(&store, 1, 100, 3);
        assert!(recall >= 0.99, "SB8 1D grid recall too low: {recall:.4}");
    }

    #[test]
    fn sb8_grid_l2_recall_2d_10() {
        let store = Store;
        let recall = run_sb8_grid_recall(&store, 2, 10, 3);
        assert!(recall >= 0.99, "SB8 2D grid recall too low: {recall:.4}");
    }

    #[test]
    fn sb8_grid_l2_recall_3d_7() {
        let store = Store;
        let recall = run_sb8_grid_recall(&store, 3, 7, 3);
        assert!(recall >= 0.99, "SB8 3D grid recall too low: {recall:.4}");
    }

    #[test]
    fn sb8_grid_l2_recall_4d_5() {
        let store = Store;
        let recall = run_sb8_grid_recall(&store, 4, 5, 3);
        assert!(recall >= 0.99, "SB8 4D grid recall too low: {recall:.4}");
    }

    // ── Q8 (native int8 index) recall tests ─────────────────────────────

    /// Helper: create a Q8 L2 index, insert SB8 grid vectors, query each, return recall.
    fn run_q8_grid_recall(store: &Store, dimensions: u32, grid_size: usize, k: usize) -> f64 {
        let (ids, vectors) = generate_sb8_grid_vectors(dimensions as usize, grid_size);
        run_sb8_recall_with_quant(
            store,
            dimensions,
            Metric::L2,
            &ids,
            &vectors,
            k,
            distance_fn::<SquaredL2>,
            VectorQuantType::Q8,
        )
    }

    #[test]
    fn q8_grid_l2_recall_1d_100() {
        let store = Store;
        let recall = run_q8_grid_recall(&store, 1, 100, 3);
        assert!(recall >= 0.99, "Q8 1D grid recall too low: {recall:.4}");
    }

    #[test]
    fn q8_grid_l2_recall_2d_10() {
        let store = Store;
        let recall = run_q8_grid_recall(&store, 2, 10, 3);
        assert!(recall >= 0.99, "Q8 2D grid recall too low: {recall:.4}");
    }

    #[test]
    fn q8_grid_l2_recall_3d_7() {
        let store = Store;
        let recall = run_q8_grid_recall(&store, 3, 7, 3);
        assert!(recall >= 0.99, "Q8 3D grid recall too low: {recall:.4}");
    }

    #[test]
    fn q8_grid_l2_recall_4d_5() {
        let store = Store;
        let recall = run_q8_grid_recall(&store, 4, 5, 3);
        assert!(recall >= 0.99, "Q8 4D grid recall too low: {recall:.4}");
    }
}
