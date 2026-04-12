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

    #[test]
    fn test_even_distribution_on_circle_signed() {
        // Test that signed distribution produces uniform points across full circle (360 degrees)
        let dim = 2;
        let norm = 1.0;
        let num_samples = 10000; // 200000 samples per bucket on average
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate samples
        let samples: Vec<Vec<f32>> = (0..num_samples)
            .map(|_| generate_random_vector_with_norm_signed(dim, norm, true, &mut rng, |x| x))
            .collect();

        // Count samples in each 10-degree bucket (36 buckets total)
        const NUM_BUCKETS: usize = 36;
        let mut buckets = [0usize; NUM_BUCKETS];

        for sample in samples {
            let angle = sample[1].atan2(sample[0]); // atan2(y, x) gives angle in radians
            let degrees = angle.to_degrees();
            // Normalize to 0-360 range
            let normalized_degrees = if degrees < 0.0 {
                degrees + 360.0
            } else {
                degrees
            };
            let bucket = (normalized_degrees / 10.0).floor() as usize % NUM_BUCKETS;
            buckets[bucket] += 1;
        }

        // Find min and max counts
        let min_count = *buckets.iter().min().unwrap();
        let max_count = *buckets.iter().max().unwrap();
        let avg_count = num_samples / NUM_BUCKETS;
        let variation_pct = ((max_count - min_count) as f32 / avg_count as f32) * 100.0;

        // Check if variation is less than 1% (with more samples, variation decreases)
        if variation_pct >= 5.0 {
            eprintln!("Test failed! Distribution is not uniform.");
            eprintln!("Bucket counts:");
            for (i, count) in buckets.iter().enumerate() {
                eprintln!(
                    "  Bucket {} ({:3}-{:3} degrees): {} samples",
                    i,
                    i * 10,
                    (i + 1) * 10,
                    count
                );
            }
            eprintln!("Min count: {}", min_count);
            eprintln!("Max count: {}", max_count);
            eprintln!("Average count: {}", avg_count);
            eprintln!("Observed variation: {:.2}%", variation_pct);
            panic!("Distribution test failed: {:.2}% > 1.0%", variation_pct);
        }
    }

    #[test]
    fn test_even_distribution_on_circle_unsigned() {
        // Test that unsigned distribution produces uniform points across first quadrant (0-90 degrees)
        let dim = 2;
        let norm = 1.0;
        let num_samples = 3600000; // 400000 samples per bucket on average
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate samples
        let samples: Vec<Vec<f32>> = (0..num_samples)
            .map(|_| generate_random_vector_with_norm_signed(dim, norm, false, &mut rng, |x| x))
            .collect();

        // For unsigned, all points should be in first quadrant (0-90 degrees)
        // Divide into 9 buckets of 10 degrees each
        const NUM_BUCKETS: usize = 9;
        let mut buckets = [0usize; NUM_BUCKETS];

        for sample in samples {
            // Verify both coordinates are non-negative
            assert!(
                sample[0] >= 0.0 && sample[1] >= 0.0,
                "Unsigned should produce only non-negative coordinates"
            );

            let angle = sample[1].atan2(sample[0]); // atan2(y, x)
            let degrees = angle.to_degrees();

            // All angles should be in 0-90 degree range
            assert!(
                degrees >= 0.0 && degrees <= 90.0,
                "Unsigned should only produce 0-90 degree arc, got {} degrees",
                degrees
            );

            let bucket = (degrees / 10.0).floor() as usize;
            if bucket < NUM_BUCKETS {
                buckets[bucket] += 1;
            }
        }

        // Find min and max counts
        let min_count = *buckets.iter().min().unwrap();
        let max_count = *buckets.iter().max().unwrap();
        let avg_count = num_samples / NUM_BUCKETS;
        let difference_pct = ((max_count - min_count) as f32 / avg_count as f32) * 100.0;

        // Check if difference is less than 1%
        if difference_pct >= 1.0 {
            eprintln!("Test failed! Distribution is not uniform.");
            eprintln!("Bucket counts:");
            for (i, count) in buckets.iter().enumerate() {
                eprintln!(
                    "  Bucket {} ({:2}-{:2} degrees): {} samples",
                    i,
                    i * 10,
                    (i + 1) * 10,
                    count
                );
            }
            eprintln!("Min count: {}", min_count);
            eprintln!("Max count: {}", max_count);
            eprintln!("Average count: {}", avg_count);
            eprintln!("Observed difference: {:.2}%", difference_pct);
            panic!("Distribution test failed: {:.2}% > 1.0%", difference_pct);
        }
    }
}
