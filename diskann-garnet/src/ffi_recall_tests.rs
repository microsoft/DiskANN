/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod tests {
    use std::{collections::HashMap, ffi::c_void, mem, ptr};

    use diskann_vector::distance::Metric;

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

    fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut a_norm_sq = 0.0f32;
        let mut b_norm_sq = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            a_norm_sq += a[i] * a[i];
            b_norm_sq += b[i] * b[i];
        }
        let a_norm = a_norm_sq.sqrt();
        let b_norm = b_norm_sq.sqrt();
        if a_norm > 0.0 && b_norm > 0.0 {
            1.0 - dot / (a_norm * b_norm)
        } else {
            1.0
        }
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
        run_recall(store, dimensions, Metric::L2, &ids, &vectors, k, squared_l2)
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
        run_recall(store, 2, Metric::Cosine, &ids, &vectors, k, cosine_distance)
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
}
