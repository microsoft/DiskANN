/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::MatrixView;
use diskann_vector::{Norm, norm::FastL2NormSquared};
use diskann_wide::{SIMDMulAdd, SIMDVector};

use super::super::simd::{PiPNNSIMDSchema, PiPNNSIMDVector};
use super::{
    Cosine, CosineNormalized, InnerProduct, L2, PartitionNorms, cosine_distance_simd,
    cosine_distance_single,
};

/// Compute partition rankings for one concrete metric.
pub(in super::super) trait PartitionMetric: Send + Sync + 'static {
    /// SIMD representation for partition ranking scores.
    type Simd<A>: PiPNNSIMDVector<Arch = A>
    where
        A: PiPNNSIMDSchema;

    /// Prepare one norm value for each point in the active stripe.
    fn prepare_point_norms(_points: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one norm value for each sampled leader.
    fn prepare_leader_norms(_leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.clear();
    }

    /// Prepare one point norm for reuse across SIMD leader groups.
    #[inline(always)]
    fn point_simd<A>(arch: A, _norms: PartitionNorms<'_>, _point: usize) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::default(arch)
    }

    /// Prepare one point norm for reuse across single leader values.
    #[inline(always)]
    fn point_single(_norms: PartitionNorms<'_>, _point: usize) -> f32 {
        0.0
    }

    /// Compute rankings for one complete SIMD group.
    fn rankings_simd<A>(
        arch: A,
        norms: PartitionNorms<'_>,
        point_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_leader: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema;

    /// Compute one ranking outside the complete SIMD prefix.
    fn ranking_single(
        norms: PartitionNorms<'_>,
        point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32;
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

impl PartitionMetric for L2 {
    type Simd<A>
        = A::PartitionScore
    where
        A: PiPNNSIMDSchema;

    fn prepare_leader_norms(leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(leaders.nrows(), 0.0);
        for (norm, leader) in norms.iter_mut().zip(leaders.row_iter()) {
            *norm = leader.iter().map(|value| value * value).sum();
        }
    }

    #[inline(always)]
    fn rankings_simd<A>(
        arch: A,
        norms: PartitionNorms<'_>,
        _point_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_leader: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        let leader_norms = load_norms_simd::<Self::Simd<A>>(arch, norms.leader_norms, first_leader);
        Self::Simd::<A>::splat(arch, -2.0).mul_add_simd(dot_products, leader_norms)
    }

    #[inline(always)]
    fn ranking_single(
        norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32 {
        (-2.0_f32).mul_add(dot_product, norms.leader_norms[leader])
    }
}

impl PartitionMetric for Cosine {
    type Simd<A>
        = A::PartitionScore
    where
        A: PiPNNSIMDSchema;

    fn prepare_point_norms(points: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(points.nrows(), 0.0);
        for (norm, point) in norms.iter_mut().zip(points.row_iter()) {
            *norm = FastL2NormSquared.evaluate(point).sqrt();
        }
    }

    fn prepare_leader_norms(leaders: MatrixView<'_, f32>, norms: &mut Vec<f32>) {
        norms.resize(leaders.nrows(), 0.0);
        for (norm, leader) in norms.iter_mut().zip(leaders.row_iter()) {
            *norm = leader.iter().map(|value| value * value).sum::<f32>().sqrt();
        }
    }

    #[inline(always)]
    fn point_simd<A>(arch: A, norms: PartitionNorms<'_>, point: usize) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::splat(arch, norms.point_norms[point])
    }

    #[inline(always)]
    fn point_single(norms: PartitionNorms<'_>, point: usize) -> f32 {
        norms.point_norms[point]
    }

    #[inline(always)]
    fn rankings_simd<A>(
        arch: A,
        norms: PartitionNorms<'_>,
        point_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        first_leader: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        let leader_norms = load_norms_simd::<Self::Simd<A>>(arch, norms.leader_norms, first_leader);
        cosine_distance_simd(arch, dot_products, point_norms, leader_norms)
    }

    #[inline(always)]
    fn ranking_single(
        norms: PartitionNorms<'_>,
        point_norm: f32,
        dot_product: f32,
        leader: usize,
    ) -> f32 {
        cosine_distance_single(dot_product, point_norm, norms.leader_norms[leader])
    }
}

impl PartitionMetric for CosineNormalized {
    type Simd<A>
        = A::PartitionScore
    where
        A: PiPNNSIMDSchema;

    #[inline(always)]
    fn rankings_simd<A>(
        arch: A,
        _norms: PartitionNorms<'_>,
        _point_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        _first_leader: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::splat(arch, 1.0) - dot_products
    }

    #[inline(always)]
    fn ranking_single(
        _norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        _leader: usize,
    ) -> f32 {
        1.0 - dot_product
    }
}

impl PartitionMetric for InnerProduct {
    type Simd<A>
        = A::PartitionScore
    where
        A: PiPNNSIMDSchema;

    #[inline(always)]
    fn rankings_simd<A>(
        arch: A,
        _norms: PartitionNorms<'_>,
        _point_norms: Self::Simd<A>,
        dot_products: Self::Simd<A>,
        _first_leader: usize,
    ) -> Self::Simd<A>
    where
        A: PiPNNSIMDSchema,
    {
        Self::Simd::<A>::default(arch) - dot_products
    }

    #[inline(always)]
    fn ranking_single(
        _norms: PartitionNorms<'_>,
        _point_norm: f32,
        dot_product: f32,
        _leader: usize,
    ) -> f32 {
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

    fn rank_single_leader<M: PartitionMetric>(
        dot_product: f32,
        point_norm: f32,
        leader_norm: f32,
    ) -> f32 {
        let point_norms = [point_norm];
        let leader_norms = [leader_norm];
        let norms = PartitionNorms {
            point_norms: &point_norms,
            leader_norms: &leader_norms,
        };
        M::ranking_single(norms, M::point_single(norms, 0), dot_product, 0)
    }

    mod prepare_leader_norms_tests {
        use super::*;

        #[test]
        fn returns_the_squared_euclidean_norm_of_each_leader_for_l2() {
            // Given
            let first_leader = [1.0_f32, 2.0];
            let second_leader = [3.0_f32, 4.0];
            let leader_values = [
                first_leader[0],
                first_leader[1],
                second_leader[0],
                second_leader[1],
            ];
            let leaders = MatrixView::try_from(&leader_values[..], 2, 2).unwrap();
            let expected_row_squared_norms = [
                first_leader[0].powi(2) + first_leader[1].powi(2),
                second_leader[0].powi(2) + second_leader[1].powi(2),
            ];
            let mut actual_norms = Vec::new();

            // When
            L2::prepare_leader_norms(leaders, &mut actual_norms);

            // Then
            assert_eq!(actual_norms, expected_row_squared_norms);
        }

        #[test]
        fn returns_the_euclidean_norm_of_each_leader_for_cosine() {
            // Given
            let first_leader = [1.0_f32, 2.0];
            let second_leader = [3.0_f32, 4.0];
            let leader_values = [
                first_leader[0],
                first_leader[1],
                second_leader[0],
                second_leader[1],
            ];
            let leaders = MatrixView::try_from(&leader_values[..], 2, 2).unwrap();
            let expected_row_norms = [
                (first_leader[0].powi(2) + first_leader[1].powi(2)).sqrt(),
                (second_leader[0].powi(2) + second_leader[1].powi(2)).sqrt(),
            ];
            let mut actual_norms = Vec::new();

            // When
            Cosine::prepare_leader_norms(leaders, &mut actual_norms);

            // Then
            assert_eq!(actual_norms, expected_row_norms);
        }
    }

    #[test]
    fn returns_the_euclidean_norm_of_each_point_for_cosine() {
        // Given
        let first_point = [1.0_f32, 2.0];
        let second_point = [3.0_f32, 4.0];
        let point_values = [
            first_point[0],
            first_point[1],
            second_point[0],
            second_point[1],
        ];
        let points = MatrixView::try_from(&point_values[..], 2, 2).unwrap();
        let expected_row_norms = [
            (first_point[0].powi(2) + first_point[1].powi(2)).sqrt(),
            (second_point[0].powi(2) + second_point[1].powi(2)).sqrt(),
        ];
        let mut actual_norms = Vec::new();

        // When
        Cosine::prepare_point_norms(points, &mut actual_norms);

        // Then
        assert_eq!(actual_norms, expected_row_norms);
    }

    mod ranking_single_tests {
        use super::*;

        #[test]
        fn ranking_equals_leader_squared_norm_minus_twice_the_dot_product_with_l2() {
            // Given
            let dot_product = 2.0;
            let leader_squared_norm = 9.0;
            let expected_leader_norm_minus_twice_dot = leader_squared_norm - 2.0 * dot_product;

            // When
            let actual_ranking = rank_single_leader::<L2>(dot_product, 0.0, leader_squared_norm);

            // Then
            assert_eq!(actual_ranking, expected_leader_norm_minus_twice_dot);
        }

        #[test]
        fn ranking_equals_one_minus_dot_over_norm_product_with_cosine() {
            // Given
            let dot_product = 4.0;
            let point_norm = 2.0;
            let leader_norm = 4.0;
            let expected_one_minus_normalized_dot = 1.0 - dot_product / (point_norm * leader_norm);

            // When
            let actual_ranking = rank_single_leader::<Cosine>(dot_product, point_norm, leader_norm);

            // Then
            assert_eq!(actual_ranking, expected_one_minus_normalized_dot);
        }

        #[test]
        fn ranking_equals_one_minus_the_dot_product_with_normalized_cosine() {
            // Given
            let dot_product = 0.25;
            let expected_one_minus_dot = 1.0 - dot_product;

            // When
            let actual_ranking = rank_single_leader::<CosineNormalized>(dot_product, 0.0, 0.0);

            // Then
            assert_eq!(actual_ranking, expected_one_minus_dot);
        }

        #[test]
        fn ranking_equals_the_negative_dot_product_with_inner_product() {
            // Given
            let dot_product = 3.0;
            let expected_negative_dot = -dot_product;

            // When
            let actual_ranking = rank_single_leader::<InnerProduct>(dot_product, 0.0, 0.0);

            // Then
            assert_eq!(actual_ranking, expected_negative_dot);
        }
    }
}
