/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two independent eight-lane reductions share one sixteen-lane register.
//! No reduction crosses the point boundary.

#[cfg(target_arch = "x86_64")]
use diskann_wide::{arch::Target3, Architecture, SIMDVector};

#[cfg(target_arch = "x86_64")]
use super::simd::{self, SIMDSchema, L2};
use super::SquaredL2;

type Evaluate = fn([&[f32]; 2], &[f32], &[f32]) -> [[f32; 2]; 2];

/// A two-point, two-center L2 evaluator with the compiled baseline's arithmetic.
///
/// On an x86 V3/V4 build and V4-capable CPU, each half of a wide register holds
/// the baseline's eight coordinate lanes for one point. The four accumulation
/// chains, FMA, final reduction and partial-vector tail are unchanged. Other
/// configurations use two baseline paired evaluations. ISA selection happens
/// at construction; rows are borrowed without packing or allocation.
#[derive(Clone, Copy)]
pub struct TwoPointSquaredL2 {
    evaluate: Evaluate,
}

impl Default for TwoPointSquaredL2 {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoPointSquaredL2 {
    pub fn new() -> Self {
        // Scalar uses a different reduction schedule. A V4-capable host must
        // never silently promote a scalar-compiled baseline to that schedule.
        #[cfg(target_arch = "x86_64")]
        if diskann_wide::arch::Current::level() >= diskann_wide::arch::x86_64::V3::level()
            && diskann_wide::arch::x86_64::V4::new_checked_miri().is_some()
        {
            return Self {
                // The witness is checked here once per prepared parent.
                // The wrapper below applies target features to the whole tile.
                evaluate: v4_evaluate,
            };
        }
        Self { evaluate: fallback }
    }

    /// Return `[point][center]` distances, bitwise equal to baseline paired L2
    /// for finite coordinates (including overflow to infinity).
    ///
    /// # Panics
    /// Panics if any row or center has a different dimension.
    #[inline]
    pub fn evaluate(&self, points: [&[f32]; 2], first: &[f32], second: &[f32]) -> [[f32; 2]; 2] {
        (self.evaluate)(points, first, second)
    }
}

fn fallback(points: [&[f32]; 2], first: &[f32], second: &[f32]) -> [[f32; 2]; 2] {
    points.map(|point| SquaredL2::evaluate_pair(point, first, second))
}

#[cfg(target_arch = "x86_64")]
fn v4_evaluate(points: [&[f32]; 2], first: &[f32], second: &[f32]) -> [[f32; 2]; 2] {
    // SAFETY: This private function is selected only after V4::new_checked_miri
    // succeeds. As with other architecture witnesses, CPU features cannot change
    // during the evaluator's lifetime.
    let arch = unsafe { diskann_wide::arch::x86_64::V4::new() };
    arch.run3(Tile, points, first, second)
}

#[cfg(target_arch = "x86_64")]
struct Tile;

#[cfg(target_arch = "x86_64")]
impl<A: Architecture> Target3<A, [[f32; 2]; 2], [&[f32]; 2], &[f32], &[f32]> for Tile
where
    L2: SIMDSchema<
        f32,
        f32,
        A,
        Return = f32,
        Accumulator = A::f32x8,
        Left = A::f32x8,
        Right = A::f32x8,
    >,
{
    #[inline(always)]
    fn run(self, arch: A, points: [&[f32]; 2], first: &[f32], second: &[f32]) -> [[f32; 2]; 2] {
        use diskann_wide::SplitJoin;

        let len = first.len();
        for actual in [second.len(), points[0].len(), points[1].len()] {
            if actual != len {
                simd::emit_length_error(len, actual);
            }
        }
        let full = len / 32 * 32;
        let vectors = len / 8 * 8;
        let mut sums = [[A::f32x16::default(arch); 2]; 4];

        for offset in (0..full).step_by(32) {
            accumulate(arch, points, [first, second], offset, &mut sums[0]);
            accumulate(arch, points, [first, second], offset + 8, &mut sums[1]);
            accumulate(arch, points, [first, second], offset + 16, &mut sums[2]);
            accumulate(arch, points, [first, second], offset + 24, &mut sums[3]);
        }
        for (group, offset) in (full..vectors).step_by(8).enumerate() {
            accumulate(arch, points, [first, second], offset, &mut sums[group]);
        }

        let mut result = [[0.0; 2]; 2];
        for (center_id, center) in [first, second].into_iter().enumerate() {
            let reduced = ((sums[0][center_id] + sums[1][center_id])
                + (sums[2][center_id] + sums[3][center_id]))
                .split();
            for (point_id, mut sum) in [reduced.lo, reduced.hi].into_iter().enumerate() {
                if vectors != len {
                    // SAFETY: The checked rows contain `len - vectors < 8`
                    // coordinates. Preserve the original masked FMA epilogue.
                    sum = unsafe {
                        L2.epilogue(
                            arch,
                            points[point_id].as_ptr().add(vectors),
                            center.as_ptr().add(vectors),
                            len - vectors,
                            sum,
                        )
                    };
                }
                result[point_id][center_id] = L2.reduce(sum);
            }
        }
        result
    }
}

// Keep arithmetic in the target-feature-enabled caller. An uninlined closure
// here compiles for the base ISA and spills/calls out for each vector operation.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn accumulate<A: Architecture>(
    arch: A,
    points: [&[f32]; 2],
    centers: [&[f32]; 2],
    offset: usize,
    sums: &mut [A::f32x16; 2],
) {
    use diskann_wide::{LoHi, SIMDMulAdd, SplitJoin};
    // Each register is [point 0: d..d+8 | point 1: d..d+8]. No cross-point sum.
    // SAFETY: The private caller checks equal lengths and calls only for complete
    // eight-coordinate vectors. The architecture witness covers SIMD execution.
    unsafe {
        let x = A::f32x16::join(LoHi::new(
            A::f32x8::load_simd(arch, points[0].as_ptr().add(offset)),
            A::f32x8::load_simd(arch, points[1].as_ptr().add(offset)),
        ));
        for (sum, center) in sums.iter_mut().zip(centers) {
            let c = A::f32x8::load_simd(arch, center.as_ptr().add(offset));
            let delta = x - A::f32x16::join(LoHi::new(c, c));
            *sum = delta.mul_add_simd(delta, *sum);
        }
    }
}

#[cfg(test)]
mod tests;
