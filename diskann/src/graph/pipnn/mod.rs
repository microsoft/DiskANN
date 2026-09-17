/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Numerical kernels for PiPNN graph construction.
//!
//! Metric modules fill portable distance buffers. Kernel modules use architecture
//! `A` to traverse those buffers. Callers select metric `M` once per graph build.

#[allow(dead_code)]
mod leaf_kernel;
#[allow(dead_code)]
mod leaf_metric;
#[allow(dead_code)]
mod partition_kernel;
#[allow(dead_code)]
mod partition_metric;
#[allow(dead_code)]
mod simd;
#[allow(dead_code)]
mod topk;

pub(super) struct L2;
pub(super) struct Cosine;
pub(super) struct CosineNormalized;
pub(super) struct InnerProduct;

/// Convert one dot product and two norms to cosine distance.
///
/// Treat a zero or subnormal norm as zero similarity. This rule takes precedence
/// over the dot value. Clamp finite similarity to the cosine range. Otherwise,
/// a NaN input produces a NaN distance.
#[inline(always)]
fn cosine_distance(dot: f32, source_norm: f32, target_norm: f32) -> f32 {
    if source_norm < f32::MIN_POSITIVE.sqrt() || target_norm < f32::MIN_POSITIVE.sqrt() {
        1.0
    } else {
        1.0 - (dot / (source_norm * target_norm)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
#[cfg(not(miri))]
mod scalar_ranking {
    use diskann_vector::distance::Metric;

    // Sample an irregular arc with increasing radius. Repeat its x/y coordinates
    // through every dimension, so an odd final coordinate changes the geometry.
    // Unequal angle steps avoid systematic nearest-neighbor ties.
    pub(super) fn arc_vectors(
        parameters: impl IntoIterator<Item = f64>,
        dimensions: usize,
        normalize: bool,
    ) -> Vec<f32> {
        parameters
            .into_iter()
            .flat_map(|t| {
                let angle = 0.03 * t + 0.001 * t * t;
                let radius = 1.0 + t / 64.0;
                let coordinates = [radius * angle.cos(), radius * angle.sin()];
                let mut vector: Vec<_> = (0..dimensions)
                    .map(|dimension| coordinates[dimension % 2] as f32)
                    .collect();
                if normalize {
                    let norm = vector
                        .iter()
                        .map(|&x| f64::from(x).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    for x in &mut vector {
                        *x = (f64::from(*x) / norm) as f32;
                    }
                }
                vector
            })
            .collect()
    }

    // Use f64 scalar arithmetic on the actual f32 inputs, not GEMM or a kernel
    // metric. These fixtures have finite coordinates and nonzero normal norms.
    // L2 includes the point norm; partition ranking can omit this row constant.
    pub(super) fn distance(metric: Metric, point: &[f32], target: &[f32]) -> f64 {
        let dot: f64 = point
            .iter()
            .zip(target)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum();
        match metric {
            Metric::L2 => point
                .iter()
                .zip(target)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum(),
            Metric::Cosine => {
                let point_norm = point
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let target_norm = target
                    .iter()
                    .map(|&x| f64::from(x).powi(2))
                    .sum::<f64>()
                    .sqrt();
                1.0 - (dot / (point_norm * target_norm)).clamp(-1.0, 1.0)
            }
            Metric::CosineNormalized | Metric::InnerProduct => -dot,
        }
    }

    // Keep rank comparisons tight enough to distinguish nearby candidates.
    // L2 and inner-product scores grow with dimension. Cosine scores stay unit-scale.
    pub(super) fn ranking_tolerance(metric: Metric, dimensions: usize) -> f64 {
        let scale = match metric {
            Metric::L2 | Metric::InnerProduct => dimensions as f64,
            Metric::Cosine | Metric::CosineNormalized => 1.0,
        };
        32.0 * f64::from(f32::EPSILON) * scale
    }

    // Bound distance error from the accumulated terms, not the final distance.
    // L2 can subtract large dot products from large squared norms.
    // This bound applies to the finite, nonzero-norm geometry above.
    pub(super) fn distance_tolerance(metric: Metric, point: &[f32], target: &[f32]) -> f64 {
        // gamma = n*u/(1-n*u) bounds rounding across n operations.
        // Allow norm reduction, Gram accumulation, and the final arithmetic.
        let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
        let steps = 2.0 * point.len() as f64 + 3.0;
        let accumulated_roundoff = steps * unit_roundoff;
        let gamma = accumulated_roundoff / (1.0 - accumulated_roundoff);
        match metric {
            Metric::L2 => {
                let terms: f64 = point
                    .iter()
                    .zip(target)
                    .map(|(&x, &y)| (f64::from(x).abs() + f64::from(y).abs()).powi(2))
                    .sum();
                gamma * terms
            }
            Metric::InnerProduct | Metric::CosineNormalized => {
                let terms: f64 = point
                    .iter()
                    .zip(target)
                    .map(|(&x, &y)| (f64::from(x) * f64::from(y)).abs())
                    .sum();
                gamma * terms
            }
            Metric::Cosine => {
                // Both the dot product and the norm product can carry error.
                // Division adds rounding. The final subtraction adds at most 2*u.
                (2.0 * gamma + unit_roundoff * (1.0 + gamma)) / (1.0 - gamma) + 2.0 * unit_roundoff
            }
        }
    }

    // Compare in output order. An isolated score admits only its expected ID.
    // Scores within the rounding bound can exchange IDs, but cannot duplicate one.
    #[track_caller]
    pub(super) fn assert_ranked_ids(actual: &[u32], sorted: &[(u32, f64)], tolerance: f64) {
        let mut previous = f64::NEG_INFINITY;
        for (rank, &id) in actual.iter().enumerate() {
            assert!(
                !actual[..rank].contains(&id),
                "duplicate ID {id} at rank {rank}"
            );
            let score = sorted
                .iter()
                .find(|&&(candidate, _)| candidate == id)
                .unwrap_or_else(|| panic!("unexpected ID {id} at rank {rank}"))
                .1;
            let (expected_id, expected_score) = sorted[rank];
            assert!(
                (score - expected_score).abs() <= tolerance,
                "rank {rank}: ID {id} score {score}, expected ID {expected_id} score {expected_score}, tolerance {tolerance}"
            );
            assert!(
                previous <= score + tolerance,
                "rank {rank}: score {score} follows farther score {previous}"
            );
            previous = score;
        }
    }
}

#[cfg(test)]
mod cosine_distance_tests {
    use super::cosine_distance;
    use rstest::rstest;

    #[rstest]
    #[case::zero_source(0.0, 0.0, 1.0)]
    #[case::zero_target(0.0, 1.0, 0.0)]
    #[case::subnormal_source(f32::MIN_POSITIVE.sqrt() / 2.0, f32::MIN_POSITIVE.sqrt() / 2.0, 1.0)]
    #[case::subnormal_target(f32::MIN_POSITIVE.sqrt() / 2.0, 1.0, f32::MIN_POSITIVE.sqrt() / 2.0)]
    #[case::zero_norm_before_nan_dot(f32::NAN, 0.0, 1.0)]
    fn small_norm_produces_unit_distance(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert_eq!(cosine_distance(dot, source_norm, target_norm), 1.0);
    }

    #[test]
    fn norm_at_cutoff_preserves_positive_similarity() {
        // At the cutoff, similarity remains 0.5 instead of zero.
        let norm = f32::MIN_POSITIVE.sqrt();
        assert_eq!(cosine_distance(0.5 * norm, norm, 1.0), 0.5);
    }

    #[rstest]
    #[case::above_one(1.0 + f32::EPSILON, 0.0)]
    #[case::below_negative_one(-1.0 - f32::EPSILON, 2.0)]
    fn cosine_clamps_finite_similarity(#[case] similarity: f32, #[case] expected: f32) {
        assert_eq!(cosine_distance(similarity, 1.0, 1.0), expected);
    }

    #[rstest]
    #[case::nan_dot(f32::NAN, 1.0, 1.0)]
    #[case::nan_source_norm(0.0, f32::NAN, 1.0)]
    #[case::nan_target_norm(0.0, 1.0, f32::NAN)]
    fn nan_propagates_without_small_norms(
        #[case] dot: f32,
        #[case] source_norm: f32,
        #[case] target_norm: f32,
    ) {
        assert!(cosine_distance(dot, source_norm, target_norm).is_nan());
    }
}
