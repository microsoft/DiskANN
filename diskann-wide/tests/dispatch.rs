/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(target_arch = "x86_64")]
use diskann_wide::{SIMDMulAdd, SIMDSumTree, SIMDVector};

#[cfg(target_arch = "x86_64")]
type V3 = diskann_wide::arch::x86_64::V3;

#[cfg(target_arch = "x86_64")]
type V4 = diskann_wide::arch::x86_64::V4;

struct InnerProduct;

impl diskann_wide::arch::Target2<diskann_wide::arch::Scalar, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar, x: &[f32], y: &[f32]) -> f32 {
        std::iter::zip(x.iter(), y.iter()).map(|(a, b)| a * b).sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V3, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, arch: V3, a: &[f32], b: &[f32]) -> f32 {
        diskann_wide::alias!(f32s = <V3>::f32x8);

        // The number of lanes to use in a SIMD register.
        const LANES: usize = f32s::LANES;

        let len = a.len();
        let a = a.as_ptr();
        let b = b.as_ptr();

        let mut sum = f32s::default(arch);
        let trips = len / LANES;
        let remainder = len % LANES;
        for i in 0..trips {
            // SAFETY: By loop construction, `a.add(LANES * (i + 1) - 1)` is always in-bounds.
            let wa = unsafe { f32s::load_simd(arch, a.add(LANES * i)) };
            // SAFETY: By loop construction, `b.add(LANES * (i + 1) - 1)` is always in-bounds.
            let wb = unsafe { f32s::load_simd(arch, b.add(LANES * i)) };
            sum = wa.mul_add_simd(wb, sum);
        }

        // Handle and remaining using predicated loads.
        if remainder != 0 {
            // SAFETY: By loop construction, `a.add(LANES * trips)` is always in-bounds.
            let wa = unsafe { f32s::load_simd_first(arch, a.add(trips * LANES), remainder) };
            // SAFETY: By loop construction, `b.add(LANES * trips)` is always in-bounds.
            let wb = unsafe { f32s::load_simd_first(arch, b.add(trips * LANES), remainder) };
            sum = wa.mul_add_simd(wb, sum);
        }

        sum.sum_tree()
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V4, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, arch: V4, a: &[f32], b: &[f32]) -> f32 {
        diskann_wide::alias!(f32s = <V4>::f32x16);

        // The number of lanes to use in a SIMD register.
        const LANES: usize = f32s::LANES;

        let len = a.len();
        let a = a.as_ptr();
        let b = b.as_ptr();

        let mut sum = f32s::default(arch);
        let trips = len / LANES;
        let remainder = len % LANES;
        for i in 0..trips {
            // SAFETY: By loop construction, `a.add(LANES * (i + 1) - 1)` is always in-bounds.
            let wa = unsafe { f32s::load_simd(arch, a.add(LANES * i)) };
            // SAFETY: By loop construction, `b.add(LANES * (i + 1) - 1)` is always in-bounds.
            let wb = unsafe { f32s::load_simd(arch, b.add(LANES * i)) };
            sum = wa.mul_add_simd(wb, sum);
        }

        // Handle and remaining using predicated loads.
        if remainder != 0 {
            // SAFETY: By loop construction, `a.add(LANES * trips)` is always in-bounds.
            let wa = unsafe { f32s::load_simd_first(arch, a.add(trips * LANES), remainder) };
            // SAFETY: By loop construction, `b.add(LANES * trips)` is always in-bounds.
            let wb = unsafe { f32s::load_simd_first(arch, b.add(trips * LANES), remainder) };
            sum = wa.mul_add_simd(wb, sum);
        }

        sum.sum_tree()
    }
}

// This example
struct SquaredL2;

impl diskann_wide::arch::Target2<diskann_wide::arch::Scalar, f32, &[f32], &[f32]> for SquaredL2 {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar, x: &[f32], y: &[f32]) -> f32 {
        std::iter::zip(x.iter(), y.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V3, f32, &[f32], &[f32]> for SquaredL2 {
    #[inline(always)]
    fn run(self, arch: V3, x: &[f32], y: &[f32]) -> f32 {
        let arch: diskann_wide::arch::Scalar = arch.into();
        self.run(arch, x, y)
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V4, f32, &[f32], &[f32]> for SquaredL2 {
    #[inline(always)]
    fn run(self, arch: V4, x: &[f32], y: &[f32]) -> f32 {
        let arch: diskann_wide::arch::Scalar = arch.into();
        self.run(arch, x, y)
    }
}

#[inline(never)]
pub fn test_inner_product(a: &[f32], b: &[f32]) -> f32 {
    diskann_wide::arch::dispatch2(InnerProduct, a, b)
}

#[inline(never)]
pub fn test_euclidean(a: &[f32], b: &[f32]) -> f32 {
    diskann_wide::arch::dispatch2(SquaredL2, a, b)
}

#[test]
fn test() {
    assert_eq!(test_inner_product(&[1.0], &[2.0]), 2.0);
    assert_eq!(test_inner_product(&[2.0], &[4.0]), 8.0);

    assert_eq!(test_euclidean(&[1.0], &[2.0]), 1.0);
    assert_eq!(test_euclidean(&[2.0], &[4.0]), 4.0);
}
