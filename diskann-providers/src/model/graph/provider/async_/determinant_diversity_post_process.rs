/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::{MathematicalValue, PureDistanceFunction, distance::InnerProduct};

pub fn determinant_diversity_post_process<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    query: &[f32],
    k: usize,
    determinant_diversity_eta: f32,
    determinant_diversity_power: f32,
) -> Vec<(Id, f32)> {
    if candidates.is_empty() || query.is_empty() {
        return Vec::new();
    }

    let candidates: Vec<_> = candidates
        .into_iter()
        .filter(|(_, _, vector)| vector.len() == query.len())
        .collect();

    if candidates.is_empty() {
        return Vec::new();
    }

    let k = k.min(candidates.len());
    if k == 0 {
        return Vec::new();
    }

    if candidates[0].2.is_empty() {
        return Vec::new();
    }

    let distance_range = {
        let mut min_distance = f32::INFINITY;
        let mut max_distance = f32::NEG_INFINITY;

        for (_, distance, _) in &candidates {
            min_distance = min_distance.min(*distance);
            max_distance = max_distance.max(*distance);
        }

        (min_distance, max_distance)
    };

    if determinant_diversity_eta > 0.0 {
        post_process_with_eta_f32(
            candidates,
            k,
            determinant_diversity_eta,
            determinant_diversity_power,
            distance_range,
        )
    } else {
        post_process_greedy_orthogonalization_f32(
            candidates,
            k,
            determinant_diversity_power,
            distance_range,
        )
    }
}

fn post_process_with_eta_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    k: usize,
    eta: f32,
    power: f32,
    distance_range: (f32, f32),
) -> Vec<(Id, f32)> {
    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let inv_sqrt_eta = 1.0 / eta.sqrt();
    let mut residuals = Vec::with_capacity(n);
    let mut norms_sq = Vec::with_capacity(n);

    for (_, distance_to_query, v) in &candidates {
        let scale =
            distance_to_similarity(*distance_to_query, distance_range).powf(power) * inv_sqrt_eta;
        let residual: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let norm_sq = dot_product(&residual, &residual);
        residuals.push(residual);
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);
    let mut projections = vec![0.0f32; n];

    for _ in 0..k {
        let best_idx = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let Some(selected_index) = best_idx else {
            break;
        };

        selected.push(selected_index);
        available[selected_index] = false;

        if selected.len() == k {
            break;
        }

        let best_norm_sq = norms_sq[selected_index];
        if best_norm_sq <= 0.0 {
            continue;
        }

        let inv_norm_sq = 1.0 / best_norm_sq;
        let r_star_copy = residuals[selected_index].clone();

        for i in 0..n {
            if !available[i] {
                projections[i] = 0.0;
            } else {
                projections[i] = dot_product(&residuals[i], &r_star_copy) * inv_norm_sq;
            }
        }

        for i in 0..n {
            if !available[i] {
                continue;
            }

            let projection = projections[i];
            for (residual, &star) in residuals[i].iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            norms_sq[i] = (norms_sq[i] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = &candidates[idx];
            (*id, *dist)
        })
        .collect()
}

fn post_process_greedy_orthogonalization_f32<Id: Copy>(
    candidates: Vec<(Id, f32, Vec<f32>)>,
    k: usize,
    power: f32,
    distance_range: (f32, f32),
) -> Vec<(Id, f32)> {
    let n = candidates.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let mut residuals = Vec::with_capacity(n);
    let mut norms_sq = Vec::with_capacity(n);

    for (_, distance_to_query, v) in &candidates {
        let scale = distance_to_similarity(*distance_to_query, distance_range).powf(power);
        let residual: Vec<f32> = v.iter().map(|&x| x * scale).collect();
        let norm_sq = dot_product(&residual, &residual);
        residuals.push(residual);
        norms_sq.push(norm_sq);
    }

    let mut available = vec![true; n];
    let mut selected = Vec::with_capacity(k);
    let mut projections = vec![0.0f32; n];

    for _ in 0..k {
        let best = available
            .iter()
            .enumerate()
            .filter(|&(_, &avail)| avail)
            .max_by(|(i, _), (j, _)| {
                norms_sq[*i]
                    .partial_cmp(&norms_sq[*j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let Some((best_index, _)) = best else {
            break;
        };

        let best_norm_sq = norms_sq[best_index];
        selected.push(best_index);
        available[best_index] = false;

        if selected.len() == k {
            break;
        }

        if best_norm_sq <= 0.0 {
            continue;
        }

        let inv_norm_sq_star = 1.0 / best_norm_sq;
        let r_star_copy = residuals[best_index].clone();

        for j in 0..n {
            if !available[j] {
                projections[j] = 0.0;
            } else {
                projections[j] = dot_product(&residuals[j], &r_star_copy) * inv_norm_sq_star;
            }
        }

        for j in 0..n {
            if !available[j] {
                continue;
            }

            let projection = projections[j];
            for (residual, &star) in residuals[j].iter_mut().zip(r_star_copy.iter()) {
                *residual -= projection * star;
            }

            norms_sq[j] = (norms_sq[j] - projection * projection * best_norm_sq).max(0.0);
        }
    }

    selected
        .iter()
        .map(|&idx| {
            let (id, dist, _) = &candidates[idx];
            (*id, *dist)
        })
        .collect()
}

fn distance_to_similarity(distance: f32, distance_range: (f32, f32)) -> f32 {
    let (min_distance, max_distance) = distance_range;
    let span = (max_distance - min_distance).max(f32::EPSILON);

    // Distances are lower-is-better in DiskANN distance semantics.
    ((max_distance - distance) / span).max(0.0) + f32::EPSILON
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    <InnerProduct as PureDistanceFunction<&[f32], &[f32], MathematicalValue<f32>>>::evaluate(a, b)
        .into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_candidates() {
        let result =
            determinant_diversity_post_process::<u32>(Vec::new(), &[1.0, 2.0], 5, 0.5, 1.0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_empty_query() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 2.0])];
        let result = determinant_diversity_post_process(candidates, &[], 5, 0.5, 1.0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_mismatched_dimensions() {
        let candidates = vec![
            (0u32, 0.5, vec![1.0, 2.0]),
            (1u32, 0.3, vec![1.0]), // Wrong dimension
        ];
        let query = &[1.0, 2.0, 3.0];
        let result = determinant_diversity_post_process(candidates, query, 5, 0.5, 1.0);
        assert_eq!(result.len(), 0); // All candidates filtered due to dimension mismatch
    }

    #[test]
    fn test_single_candidate() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 2.0])];
        let query = &[1.0, 2.0];
        let result = determinant_diversity_post_process(candidates, query, 5, 0.5, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
    }

    #[test]
    fn test_k_larger_than_candidates() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 10, 0.5, 1.0);
        assert_eq!(result.len(), 2); // Should return min(k, candidates.len())
    }

    #[test]
    fn test_with_eta_diversity() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.2, vec![0.9, 0.1]),
            (2u32, 0.3, vec![0.8, 0.2]),
        ];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 1.0, 1.0);

        assert_eq!(result.len(), 2);
        // Should select based on diversity metric with eta > 0
        assert!(result.iter().all(|(id, _)| *id < 3));
    }

    #[test]
    fn test_without_eta_greedy() {
        let candidates = vec![
            (0u32, 0.1, vec![1.0, 0.0]),
            (1u32, 0.2, vec![0.9, 0.1]),
            (2u32, 0.3, vec![0.8, 0.2]),
        ];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        assert_eq!(result.len(), 2);
        // Should select based on greedy orthogonalization (eta == 0)
        assert!(result.iter().all(|(id, _)| *id < 3));
    }

    #[test]
    fn test_power_parameter() {
        let candidates = vec![(0u32, 0.1, vec![1.0, 0.0]), (1u32, 0.2, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];

        // Test with different power values - should still work without panicking
        let result1 = determinant_diversity_post_process(candidates.clone(), query, 2, 0.0, 1.0);
        let result2 = determinant_diversity_post_process(candidates, query, 2, 0.0, 2.0);

        assert_eq!(result1.len(), 2);
        assert_eq!(result2.len(), 2);
    }

    #[test]
    fn test_distances_preserved() {
        let candidates = vec![(0u32, 0.5, vec![1.0, 0.0]), (1u32, 0.3, vec![0.0, 1.0])];
        let query = &[1.0, 1.0];
        let result = determinant_diversity_post_process(candidates, query, 2, 0.0, 1.0);

        // Verify that distances are preserved from input
        assert!(result.iter().all(|(_, dist)| *dist == 0.5 || *dist == 0.3));
    }
}
