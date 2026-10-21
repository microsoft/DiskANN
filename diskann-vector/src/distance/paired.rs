/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{arch::Target3, Architecture, SIMDVector, ARCH};

use super::{
    simd::{self, SIMDSchema, L2},
    SquaredL2,
};

impl SquaredL2 {
    /// Compute squared L2 distances from `point` to two centers, sharing point loads.
    ///
    /// Uses the same architecture, accumulation order, and tail handling as two
    /// independent FP32 evaluations. Finite inputs produce bitwise-identical results,
    /// including distances that overflow to infinity. No temporary vectors are allocated.
    ///
    /// ```rust
    /// use diskann_vector::distance::SquaredL2;
    /// assert_eq!(SquaredL2::evaluate_pair(&[1.0, 2.0], &[0.0, 0.0], &[3.0, 4.0]), [5.0, 8.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics unless all three slices have equal lengths.
    #[inline(always)]
    pub fn evaluate_pair(point: &[f32], first: &[f32], second: &[f32]) -> [f32; 2] {
        pair(ARCH, point, first, second)
    }
}

impl<A: Architecture> Target3<A, [f32; 2], &[f32], &[f32], &[f32]> for SquaredL2
where
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    #[inline(always)]
    fn run(self, arch: A, point: &[f32], first: &[f32], second: &[f32]) -> [f32; 2] {
        pair(arch, point, first, second)
    }
}

type Acc<A> = <L2 as SIMDSchema<f32, f32, A>>::Accumulator;
type Left<A> = <L2 as SIMDSchema<f32, f32, A>>::Left;
type Right<A> = <L2 as SIMDSchema<f32, f32, A>>::Right;

/// All three pointers must be readable for one full L2 SIMD vector.
#[inline(always)]
unsafe fn accumulate<A: Architecture>(
    arch: A,
    point: *const f32,
    first: *const f32,
    second: *const f32,
    sums: [Acc<A>; 2],
) -> [Acc<A>; 2]
where
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    // SAFETY: The caller provides three full vectors. The architecture witness
    // establishes that these SIMD loads and arithmetic are supported.
    unsafe {
        let x = Left::<A>::load_simd(arch, point);
        [
            L2.accumulate(x, Right::<A>::load_simd(arch, first), sums[0]),
            L2.accumulate(x, Right::<A>::load_simd(arch, second), sums[1]),
        ]
    }
}

#[inline(always)]
fn pair<A: Architecture>(arch: A, point: &[f32], first: &[f32], second: &[f32]) -> [f32; 2]
where
    L2: SIMDSchema<f32, f32, A, Return = f32>,
{
    let len = point.len();
    if first.len() != len {
        simd::emit_length_error(len, first.len());
    }
    if second.len() != len {
        simd::emit_length_error(len, second.len());
    }

    // Scalar L2 uses Strategy2x1, whose final full vector is accumulated after
    // combining its two sums. Preserve that implementation rather than impose
    // the four-accumulator SIMD reduction tree on it. This branch is constant.
    if <L2 as SIMDSchema<f32, f32, A>>::get_main_bocksize() != 4 {
        return [
            simd::simd_op(&L2, arch, point, first),
            simd::simd_op(&L2, arch, point, second),
        ];
    }

    let width = <L2 as SIMDSchema<f32, f32, A>>::get_simd_width();
    let block = 4 * width;
    let full = len / block * block;
    let vectors = len / width * width;
    let x = point.as_ptr();
    let c0 = first.as_ptr();
    let c1 = second.as_ptr();
    let mut s0 = [L2.init(arch); 2];
    let mut s1 = s0;
    let mut s2 = s0;
    let mut s3 = s0;

    // SAFETY: Lengths were checked above. Each loop iteration reads one complete
    // block, and each peeled vector ends at or before `vectors <= len`.
    unsafe {
        for i in (0..full).step_by(block) {
            s0 = accumulate(arch, x.add(i), c0.add(i), c1.add(i), s0);
            s1 = accumulate(
                arch,
                x.add(i + width),
                c0.add(i + width),
                c1.add(i + width),
                s1,
            );
            s2 = accumulate(
                arch,
                x.add(i + 2 * width),
                c0.add(i + 2 * width),
                c1.add(i + 2 * width),
                s2,
            );
            s3 = accumulate(
                arch,
                x.add(i + 3 * width),
                c0.add(i + 3 * width),
                c1.add(i + 3 * width),
                s3,
            );
        }
        if vectors - full >= width {
            s0 = accumulate(arch, x.add(full), c0.add(full), c1.add(full), s0);
        }
        if vectors - full >= 2 * width {
            let i = full + width;
            s1 = accumulate(arch, x.add(i), c0.add(i), c1.add(i), s1);
        }
        if vectors - full >= 3 * width {
            let i = full + 2 * width;
            s2 = accumulate(arch, x.add(i), c0.add(i), c1.add(i), s2);
        }
    }
    let mut a = L2.combine(L2.combine(s0[0], s1[0]), L2.combine(s2[0], s3[0]));
    let mut b = L2.combine(L2.combine(s0[1], s1[1]), L2.combine(s2[1], s3[1]));
    let remaining = len - vectors;
    if remaining != 0 {
        // SAFETY: `vectors < len`, and all three slices have `remaining` elements
        // left. Reuse each architecture's original partial-vector arithmetic.
        unsafe {
            a = L2.epilogue(arch, x.add(vectors), c0.add(vectors), remaining, a);
            b = L2.epilogue(arch, x.add(vectors), c1.add(vectors), remaining, b);
        }
    }
    [L2.reduce(a), L2.reduce(b)]
}

#[cfg(test)]
mod tests;
