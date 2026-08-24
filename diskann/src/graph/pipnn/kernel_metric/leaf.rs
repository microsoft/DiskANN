/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::super::simd::{PiPNNSIMDSchema, PiPNNSIMDVector};
use super::{
    Cosine, CosineNormalized, InnerProduct, L2, cosine_distance_simd, cosine_distance_single,
};

/// Compute leaf distances for one concrete metric.
pub(in super::super) trait LeafMetric: Send + Sync + 'static {
    /// Prepare one contiguous metric-specific norm for each leaf-local point.
    fn prepare_leaf_norms(_dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one source norm for reuse across SIMD target groups.
    #[inline(always)]
    fn source_simd<A>(arch: A, _norms: &[f32], _source: usize) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        A::Vector::default(arch)
    }

    /// Prepare one source norm for reuse across single target values.
    #[inline(always)]
    fn source_single(_norms: &[f32], _source: usize) -> f32 {
        0.0
    }

    /// Compute distances for one complete SIMD group.
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: A::Vector,
        dot_products: A::Vector,
        first_target: usize,
    ) -> A::Vector
    where
        A: PiPNNSIMDSchema;

    /// Compute one distance outside the complete SIMD prefix.
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32;
}

/// Load one complete SIMD group of prepared norms.
#[inline(always)]
fn load_norms_simd<F>(arch: F::Arch, norms: &[f32], first_norm: usize) -> F
where
    F: PiPNNSIMDVector,
{
    let last_norm = first_norm + F::LANES;
    let norm_group = &norms[first_norm..last_norm];

    // SAFETY: `norm_group` contains one complete SIMD group.
    unsafe { F::load_simd(arch, norm_group.as_ptr()) }
}

impl LeafMetric for L2 {
    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)];
        }
    }

    #[inline(always)]
    fn source_simd<A>(arch: A, norms: &[f32], source: usize) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        A::Vector::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: A::Vector,
        dot_products: A::Vector,
        first_target: usize,
    ) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        let target_norms = load_norms_simd::<A::Vector>(arch, norms, first_target);
        (A::Vector::splat(arch, -2.0).mul_add_simd(dot_products, source_norms) + target_norms)
            .max_simd(A::Vector::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        ((-2.0_f32).mul_add(dot_product, source_norm) + norms[target]).max(0.0)
    }
}

impl LeafMetric for Cosine {
    fn prepare_leaf_norms(dots: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(dots.nrows(), 0.0);
        for (point, norm) in norms.iter_mut().enumerate() {
            *norm = dots[(point, point)].sqrt();
        }
    }

    #[inline(always)]
    fn source_simd<A>(arch: A, norms: &[f32], source: usize) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        A::Vector::splat(arch, norms[source])
    }

    #[inline(always)]
    fn source_single(norms: &[f32], source: usize) -> f32 {
        norms[source]
    }

    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        norms: &[f32],
        source_norms: A::Vector,
        dot_products: A::Vector,
        first_target: usize,
    ) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        let target_norms = load_norms_simd::<A::Vector>(arch, norms, first_target);
        cosine_distance_simd(arch, dot_products, source_norms, target_norms)
            .max_simd(A::Vector::default(arch))
    }

    #[inline(always)]
    fn distance_single(norms: &[f32], source_norm: f32, dot_product: f32, target: usize) -> f32 {
        cosine_distance_single(dot_product, source_norm, norms[target]).max(0.0)
    }
}

impl LeafMetric for CosineNormalized {
    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        _norms: &[f32],
        _source_norms: A::Vector,
        dot_products: A::Vector,
        _first_target: usize,
    ) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        A::Vector::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        1.0 - dot_product
    }
}

impl LeafMetric for InnerProduct {
    #[inline(always)]
    fn distances_simd<A>(
        arch: A,
        _norms: &[f32],
        _source_norms: A::Vector,
        dot_products: A::Vector,
        _first_target: usize,
    ) -> A::Vector
    where
        A: PiPNNSIMDSchema,
    {
        A::Vector::default(arch) - dot_products
    }

    #[inline(always)]
    fn distance_single(_norms: &[f32], _source_norm: f32, dot_product: f32, _target: usize) -> f32 {
        -dot_product
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    reason = "deterministic test matrices must abort on invalid setup"
)]
mod tests {
    use super::*;
    use diskann_wide::{ARCH, SIMDVector, arch::Current};

    fn simd_distances<M: LeafMetric>(
        norms: &[f32],
        source: usize,
        dot_products: [f32; 16],
        first_target: usize,
    ) -> [f32; 16] {
        type TestVector = <Current as PiPNNSIMDSchema>::Vector;
        assert_eq!(16 % TestVector::LANES, 0);
        let source_norms = M::source_simd(ARCH, norms, source);
        let mut output = [0.0; 16];
        for first in (0..16).step_by(TestVector::LANES) {
            // SAFETY: each offset starts one complete SIMD group in both arrays.
            unsafe {
                let dots = TestVector::load_simd(ARCH, dot_products.as_ptr().add(first));
                M::distances_simd(ARCH, norms, source_norms, dots, first_target + first)
                    .store_simd(output.as_mut_ptr().add(first));
            }
        }
        output
    }

    mod prepare_leaf_norms_tests {
        use super::*;

        #[test]
        fn returns_the_squared_euclidean_norm_of_each_point_for_l2() {
            // Given
            let first_point = [2.0_f32, 1.0];
            let second_point = [1.0_f32, 3.0];
            let first_self_dot = first_point[0] * first_point[0] + first_point[1] * first_point[1];
            let cross_dot = first_point[0] * second_point[0] + first_point[1] * second_point[1];
            let second_self_dot =
                second_point[0] * second_point[0] + second_point[1] * second_point[1];
            let gram_values = [first_self_dot, cross_dot, cross_dot, second_self_dot];
            let gram = MatrixView::try_from(&gram_values[..], 2, 2).unwrap();
            let expected_point_self_dots = [first_self_dot, second_self_dot];
            let mut actual_norms = Vec::new();

            // When
            L2::prepare_leaf_norms(gram, &mut actual_norms);

            // Then
            assert_eq!(actual_norms, expected_point_self_dots);
        }

        #[test]
        fn returns_the_euclidean_norm_of_each_point_for_cosine() {
            // Given
            let first_point = [2.0_f32, 1.0];
            let second_point = [1.0_f32, 3.0];
            let first_self_dot = first_point[0] * first_point[0] + first_point[1] * first_point[1];
            let cross_dot = first_point[0] * second_point[0] + first_point[1] * second_point[1];
            let second_self_dot =
                second_point[0] * second_point[0] + second_point[1] * second_point[1];
            let gram_values = [first_self_dot, cross_dot, cross_dot, second_self_dot];
            let gram = MatrixView::try_from(&gram_values[..], 2, 2).unwrap();
            let expected_point_l2_norms = [first_self_dot.sqrt(), second_self_dot.sqrt()];
            let mut actual_norms = Vec::new();

            // When
            Cosine::prepare_leaf_norms(gram, &mut actual_norms);

            // Then
            assert_eq!(actual_norms, expected_point_l2_norms);
        }
    }

    mod distance_single_tests {
        use super::*;

        #[test]
        fn distance_equals_squared_norm_sum_minus_twice_the_dot_product_with_l2() {
            // Given
            let source_squared_norm = 4.0;
            let target_squared_norm = 9.0;
            let dot_product = 6.0;
            let squared_norms = [source_squared_norm, target_squared_norm];
            let expected_squared_l2_distance =
                source_squared_norm + target_squared_norm - 2.0 * dot_product;

            // When
            let actual_distance =
                L2::distance_single(&squared_norms, source_squared_norm, dot_product, 1);

            // Then
            assert_eq!(actual_distance, expected_squared_l2_distance);
        }

        #[test]
        fn negative_roundoff_is_clamped_to_zero_with_l2() {
            // Given
            let source_squared_norm = 1.0;
            let target_squared_norm = 1.0;
            let dot_product_above_exact_norm = 1.000_001;
            let squared_norms = [source_squared_norm, target_squared_norm];
            let expected_non_negative_distance = 0.0;

            // When
            let actual_distance = L2::distance_single(
                &squared_norms,
                source_squared_norm,
                dot_product_above_exact_norm,
                1,
            );

            // Then
            assert_eq!(actual_distance, expected_non_negative_distance);
        }

        #[test]
        fn distance_equals_one_minus_dot_over_norm_product_with_cosine() {
            // Given
            let source_norm = 2.0;
            let target_norm = 4.0;
            let dot_product = 4.0;
            let norms = [source_norm, target_norm];
            let expected_one_minus_normalized_dot = 1.0 - dot_product / (source_norm * target_norm);

            // When
            let actual_distance = Cosine::distance_single(&norms, source_norm, dot_product, 1);

            // Then
            assert_eq!(actual_distance, expected_one_minus_normalized_dot);
        }

        #[test]
        fn distance_equals_one_minus_the_dot_product_with_normalized_cosine() {
            // Given
            let dot_product = 0.25;
            let expected_one_minus_dot = 1.0 - dot_product;

            // When
            let actual_distance = CosineNormalized::distance_single(&[], 0.0, dot_product, 0);

            // Then
            assert_eq!(actual_distance, expected_one_minus_dot);
        }

        #[test]
        fn distance_equals_the_negative_dot_product_with_inner_product() {
            // Given
            let dot_product = 3.0;
            let expected_negative_dot = -dot_product;

            // When
            let actual_distance = InnerProduct::distance_single(&[], 0.0, dot_product, 0);

            // Then
            assert_eq!(actual_distance, expected_negative_dot);
        }
    }

    mod distances_simd_tests {
        use super::*;

        #[test]
        fn every_lane_uses_squared_norm_sum_minus_twice_the_dot_product_with_l2() {
            let source_squared_norm = 4.0;
            let target_squared_norm = 9.0;
            let dot_product = 6.0;
            let expected_distance = source_squared_norm + target_squared_norm - 2.0 * dot_product;
            let mut squared_norms = [target_squared_norm; 17];
            squared_norms[0] = source_squared_norm;

            let actual_distances = simd_distances::<L2>(&squared_norms, 0, [dot_product; 16], 1);

            assert_eq!(actual_distances, [expected_distance; 16]);
        }

        #[test]
        fn nan_dot_products_clamp_to_zero_in_every_lane_with_l2() {
            let squared_norms = [1.0; 17];
            let expected_non_negative_distance = 0.0;

            let actual_distances = simd_distances::<L2>(&squared_norms, 0, [f32::NAN; 16], 1);

            assert_eq!(actual_distances, [expected_non_negative_distance; 16]);
        }

        #[test]
        fn every_lane_uses_one_minus_normalized_dot_with_cosine() {
            let source_norm = 2.0;
            let target_norm = 4.0;
            let dot_product = 4.0;
            let expected_distance = 1.0 - dot_product / (source_norm * target_norm);
            let mut norms = [target_norm; 17];
            norms[0] = source_norm;

            let actual_distances = simd_distances::<Cosine>(&norms, 0, [dot_product; 16], 1);

            assert_eq!(actual_distances, [expected_distance; 16]);
        }

        #[test]
        fn every_lane_uses_one_minus_dot_product_with_normalized_cosine() {
            let dot_product = 0.25;
            let expected_distance = 1.0 - dot_product;

            let actual_distances = simd_distances::<CosineNormalized>(&[], 0, [dot_product; 16], 0);

            assert_eq!(actual_distances, [expected_distance; 16]);
        }

        #[test]
        fn every_lane_uses_negative_dot_product_with_inner_product() {
            let dot_product = 3.0;
            let expected_distance = -dot_product;

            let actual_distances = simd_distances::<InnerProduct>(&[], 0, [dot_product; 16], 0);

            assert_eq!(actual_distances, [expected_distance; 16]);
        }
    }
}
