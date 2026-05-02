/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{SIMDFloat, SIMDSumTree, SIMDVector};

#[cfg(target_arch = "x86_64")]
type V3 = diskann_wide::arch::x86_64::V3;

#[cfg(target_arch = "x86_64")]
type V4 = diskann_wide::arch::x86_64::V4;

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;

struct InnerProduct;

impl diskann_wide::arch::Target2<diskann_wide::arch::Scalar, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, _: diskann_wide::arch::Scalar, x: &[f32], y: &[f32]) -> f32 {
        std::iter::zip(x.iter(), y.iter()).map(|(a, b)| a * b).sum()
    }
}

#[inline(always)]
fn inner_product<F>(arch: F::Arch, a: &[f32], b: &[f32]) -> f32
where
    F: SIMDVector<Scalar = f32> + SIMDFloat + SIMDSumTree,
{
    assert_eq!(a.len(), b.len());

    let lanes: usize = F::LANES;

    let len = a.len();
    let a = a.as_ptr();
    let b = b.as_ptr();

    let mut sum = F::default(arch);
    let trips = len / lanes;
    let remainder = len % lanes;
    for i in 0..trips {
        // SAFETY: By loop construction, `a.add(lanes * (i + 1) - 1)` is always in-bounds.
        let wa = unsafe { F::load_simd(arch, a.add(lanes * i)) };
        // SAFETY: By loop construction, `b.add(lanes * (i + 1) - 1)` is always in-bounds.
        let wb = unsafe { F::load_simd(arch, b.add(lanes * i)) };
        sum = wa.mul_add_simd(wb, sum);
    }

    // Handle and remaining using predicated loads.
    if remainder != 0 {
        // SAFETY: By loop construction, `a.add(lanes * trips)` is always in-bounds.
        let wa = unsafe { F::load_simd_first(arch, a.add(trips * lanes), remainder) };
        // SAFETY: By loop construction, `b.add(lanes * trips)` is always in-bounds.
        let wb = unsafe { F::load_simd_first(arch, b.add(trips * lanes), remainder) };
        sum = wa.mul_add_simd(wb, sum);
    }

    sum.sum_tree()
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V3, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, arch: V3, a: &[f32], b: &[f32]) -> f32 {
        diskann_wide::alias!(f32s = <V3>::f32x8);

        inner_product::<f32s>(arch, a, b)
    }
}

#[cfg(target_arch = "x86_64")]
impl diskann_wide::arch::Target2<V4, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, arch: V4, a: &[f32], b: &[f32]) -> f32 {
        diskann_wide::alias!(f32s = <V4>::f32x16);

        inner_product::<f32s>(arch, a, b)
    }
}

#[cfg(target_arch = "aarch64")]
impl diskann_wide::arch::Target2<Neon, f32, &[f32], &[f32]> for InnerProduct {
    #[inline(always)]
    fn run(self, arch: Neon, a: &[f32], b: &[f32]) -> f32 {
        diskann_wide::alias!(f32s = <Neon>::f32x4);

        inner_product::<f32s>(arch, a, b)
    }
}

// This example shows how `Architectures` can be used in the context of auto-vectorization.
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

#[cfg(target_arch = "aarch64")]
impl diskann_wide::arch::Target2<Neon, f32, &[f32], &[f32]> for SquaredL2 {
    #[inline(always)]
    fn run(self, arch: Neon, x: &[f32], y: &[f32]) -> f32 {
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
