/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{rngs::StdRng, Rng};

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

// Note: private function
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
    let mut vec: Vec<f32> = if signed {
        (0..dim)
            .map(|_| rng.random_range(-1.0f32..1.0f32))
            .collect()
    } else {
        (0..dim).map(|_| rng.random_range(0.0f32..1.0f32)).collect()
    };
    let current_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = norm / current_norm;
    vec.iter_mut().for_each(|x| *x *= scale);
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
        assert!((computed_norm - norm).abs() < tolerance);
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
}
