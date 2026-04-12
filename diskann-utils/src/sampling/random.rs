/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{rngs::StdRng, Rng};
use rand_distr::StandardNormal;

pub trait RoundFromf32 {
    fn round_from_f32(x: f32) -> Self;
}

impl RoundFromf32 for f32 {
    fn round_from_f32(x: f32) -> Self {
        x
    }
}
impl RoundFromf32 for i8 {
    fn round_from_f32(x: f32) -> Self {
        x.round() as i8
    }
}
impl RoundFromf32 for u8 {
    fn round_from_f32(x: f32) -> Self {
        x.round() as u8
    }
}
impl RoundFromf32 for half::f16 {
    fn round_from_f32(x: f32) -> Self {
        half::f16::from_f32(x)
    }
}

pub trait WithApproximateNorm: Sized {
    fn with_approximate_norm(dim: usize, norm: f32, rng: &mut StdRng) -> Vec<Self>;
}

impl WithApproximateNorm for f32 {
    fn with_approximate_norm(dim: usize, norm: f32, rng: &mut StdRng) -> Vec<Self> {
        generate_random_vector_with_norm_signed(dim, norm, true, rng, |x: f32| x)
    }
}

impl WithApproximateNorm for half::f16 {
    fn with_approximate_norm(dim: usize, norm: f32, rng: &mut StdRng) -> Vec<Self> {
        // Small QOL improvement, `diskann_wide::cast_f32_to_f16` works under `Miri` while `half::f16::from_f32`
        // does not.
        generate_random_vector_with_norm_signed(dim, norm, true, rng, diskann_wide::cast_f32_to_f16)
    }
}

impl WithApproximateNorm for u8 {
    fn with_approximate_norm(dim: usize, norm: f32, rng: &mut StdRng) -> Vec<Self> {
        generate_random_vector_with_norm_signed(dim, norm, false, rng, |x| x as u8)
    }
}

impl WithApproximateNorm for i8 {
    fn with_approximate_norm(dim: usize, norm: f32, rng: &mut StdRng) -> Vec<Self> {
        generate_random_vector_with_norm_signed(dim, norm, true, rng, |x| x as i8)
    }
}

// This function uses StandardNormal distribution. StandardNormal creates more uniformly distributed points on sphere surface (mathematically proven property), making the graph easier to navigate. Uniform distribution creates clustering artifacts that make navigation harder, requiring larger search budgets.
fn generate_random_vector_with_norm_signed<T, F>(
    dim: usize,
    norm: f32,
    signed: bool,
    rng: &mut StdRng,
    f: F,
) -> Vec<T>
where
    F: Fn(f32) -> T,
{
    let mut vec: Vec<f32> = (0..dim).map(|_| rng.sample(StandardNormal)).collect();
    let current_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = norm / current_norm;
    if signed {
        vec.iter_mut().for_each(|x| *x *= scale);
    } else {
        vec.iter_mut().for_each(|x| *x = (*x * scale).abs());
    };
    vec.into_iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rstest::rstest;

    #[rstest]
    #[case(1, 0.01)]
    #[case(100, 0.01)]
    #[case(171, 5.0)]
    #[case(1024, 100.7)]
    fn test_generate_random_vector_with_norm_f32(#[case] dim: usize, #[case] norm: f32) {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let vec: Vec<f32> = WithApproximateNorm::with_approximate_norm(dim, norm, &mut rng);
        let computed_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let tolerance = 1e-5;
        assert!((computed_norm - norm).abs() / norm < tolerance);
    }

    #[rstest]
    #[case(1, 0.01)]
    #[case(100, 0.01)]
    #[case(171, 5.0)]
    #[case(1024, 100.7)]
    fn test_generate_random_vector_with_norm_half_f16(#[case] dim: usize, #[case] norm: f32) {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let vec: Vec<half::f16> = WithApproximateNorm::with_approximate_norm(dim, norm, &mut rng);
        let computed_norm: f32 = vec
            .iter()
            .map(|x| {
                let val: f32 = x.to_f32();
                val * val
            })
            .sum::<f32>()
            .sqrt();
        let tolerance = 1e-2; // half precision
        assert!((computed_norm - norm).abs() / norm < tolerance);
    }

    #[rstest]
    #[case(17, 50.0)]
    #[case(1024, 1007.0)]
    fn test_generate_random_vector_with_norm_u8(#[case] dim: usize, #[case] norm: f32) {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let vec: Vec<u8> = WithApproximateNorm::with_approximate_norm(dim, norm, &mut rng);
        let computed_norm: f32 = vec
            .iter()
            .map(|&x| {
                let val: f32 = x as f32;
                val * val
            })
            .sum::<f32>()
            .sqrt();
        let tolerance = 1e-1; // due to quantization
        assert!((computed_norm - norm).abs() / norm < tolerance);
    }

    #[rstest]
    #[case(17, 50.0)]
    #[case(1024, 1007.0)]
    fn test_generate_random_vector_with_norm_i8(#[case] dim: usize, #[case] norm: f32) {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let vec: Vec<i8> = WithApproximateNorm::with_approximate_norm(dim, norm, &mut rng);
        let computed_norm: f32 = vec
            .iter()
            .map(|&x| {
                let val: f32 = x as f32;
                val * val
            })
            .sum::<f32>()
            .sqrt();
        let tolerance = 1e-1; // due to quantization
        assert!((computed_norm - norm).abs() / norm < tolerance);
    }

    #[rstest]
    #[case(3.6f32, 4i8)]
    #[case(2.3f32, 2i8)]
    #[case(-1.5f32, -2i8)]
    fn test_round_f32_to_i8(#[case] input: f32, #[case] expected: i8) {
        let result: i8 = RoundFromf32::round_from_f32(input);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(3.6f32, 4u8)]
    #[case(2.3f32, 2u8)]
    #[case(-1.5f32, 0u8)]
    fn test_round_f32_to_u8(#[case] input: f32, #[case] expected: u8) {
        let result: u8 = RoundFromf32::round_from_f32(input);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(3.6f32, half::f16::from_f32(3.6f32))]
    #[case(2.3f32, half::f16::from_f32(2.3f32))]
    #[case(-1.5f32, half::f16::from_f32(-1.5f32))]
    fn test_round_f32_to_f16(#[case] input: f32, #[case] expected: half::f16) {
        let result: half::f16 = RoundFromf32::round_from_f32(input);
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case(3.6f32, 3.6f32)]
    #[case(2.3f32, 2.3f32)]
    #[case(-1.5f32, -1.5f32)]
    fn test_round_f32_to_f32(#[case] input: f32, #[case] expected: f32) {
        let result: f32 = RoundFromf32::round_from_f32(input);
        assert_eq!(result, expected);
    }

    /// Test that generated points are evenly distributed on a circle.
    ///
    /// Tolerance levels:
    ///   - tolerance_sigmas = 1.0 → Very strict, only allows ±1σ deviation (about 68% of buckets would naturally fall within this)
    ///   - tolerance_sigmas = 3.0 → Moderate, allows ±3σ deviation (99.7% would naturally fall within this)
    ///   - tolerance_sigmas = 6.0 → Very lenient, allows ±6σ deviation (99.9997% would naturally fall within this)
    #[rstest]
    #[case(true, 500, 3.0, 42)]
    #[case(true, 500, 3.0, 43)]
    #[case(true, 500, 3.0, 44)]
    #[case(false, 500, 3.0, 42)]
    #[case(false, 500, 3.0, 43)]
    #[case(false, 500, 3.0, 44)]
    fn test_generate_random_vector_with_norm_signed_produces_uniform_distribution_on_circle(
        #[case] signed: bool,
        #[case] expected_per_bucket: usize,
        #[case] tolerance_sigmas: f32,
        #[case] seed: u64,
    ) {
        let dim = 2;
        let norm = 1.0;
        let mut rng = StdRng::seed_from_u64(seed);

        // Step 1: Pick number of buckets and calculate samples
        let num_buckets = if signed { 36 } else { 9 };
        let num_samples = num_buckets * expected_per_bucket;

        // Generate samples
        let samples: Vec<Vec<f32>> = (0..num_samples)
            .map(|_| generate_random_vector_with_norm_signed(dim, norm, signed, &mut rng, |x| x))
            .collect();

        // Step 2: Count hits per bucket
        let mut counts = vec![0usize; num_buckets];

        for sample in &samples {
            let theta = sample[1].atan2(sample[0]); // atan2(y, x) returns [-π, π]

            // Map to bucket: floor(θ / 2π × buckets)
            let bucket = if signed {
                // Full circle [0, 2π) → [0, 36)
                let normalized_theta = if theta < 0.0 {
                    theta + 2.0 * std::f32::consts::PI
                } else {
                    theta
                };
                ((normalized_theta / (2.0 * std::f32::consts::PI)) * num_buckets as f32).floor()
                    as usize
                    % num_buckets
            } else {
                // First quadrant [0, π/2) → [0, 9)
                ((theta / (std::f32::consts::PI / 2.0)) * num_buckets as f32).floor() as usize
            };

            counts[bucket] += 1;
        }

        // Step 3: Check each bucket is within tolerance_sigmas × σ
        // Noise per bucket: σ ≈ sqrt(expected)
        // Threshold: |observed - expected| / expected > tolerance_sigmas / sqrt(expected)
        let sigma = (expected_per_bucket as f32).sqrt();
        let threshold = tolerance_sigmas / sigma;

        let failed_count = counts
            .iter()
            .filter(|&&observed| {
                let deviation = (observed as f32 - expected_per_bucket as f32).abs()
                    / expected_per_bucket as f32;
                deviation > threshold
            })
            .count();

        assert_eq!(
            failed_count,
            0,
            "Distribution not uniform: {} out of {} bucket(s) had point counts that deviated more than {}σ from expected. \
             This indicates the generator is producing clustered points instead of evenly distributed points on the circle surface.",
            failed_count,
            num_buckets,
            tolerance_sigmas
        );
    }
}
