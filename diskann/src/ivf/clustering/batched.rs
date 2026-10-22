/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Two-center SIMD assignment with center loads shared by four or eight rows.

use diskann_wide::{Architecture, SIMDMulAdd, SIMDSumTree, SIMDVector, arch::Target3};

use super::assignment::{choose, paired};
use crate::ivf::workingset::PointRef;

#[derive(Debug, Clone, Copy)]
pub(super) struct Batched<const ROWS: usize> {
    pub(super) dim: usize,
}

impl<A: Architecture, I, const ROWS: usize> Target3<A, (), &[PointRef<'_, I>], &[f32], &mut [u8]>
    for Batched<ROWS>
{
    #[inline]
    fn run(self, arch: A, points: &[PointRef<'_, I>], centers: &[f32], assignments: &mut [u8]) {
        diskann_wide::alias!(V<A> = f32x8);
        // Shapes are checked before entering the fitter or by the provider working set.
        assert_eq!(centers.len(), 2 * self.dim);
        assert_eq!(assignments.len(), points.len());
        assert!(points.iter().all(|point| point.vector.len() == self.dim));
        let mut groups = points.chunks_exact(ROWS);
        let mut outputs = assignments.chunks_exact_mut(ROWS);
        for (group, output) in groups.by_ref().zip(outputs.by_ref()) {
            let mut first = [V::<A>::default(arch); ROWS];
            let mut second = first;
            let full = self.dim / V::<A>::LANES * V::<A>::LANES;
            for d in (0..full).step_by(V::<A>::LANES) {
                // SAFETY: Shapes were checked. Each complete SIMD load ends within dim.
                unsafe {
                    let a = V::<A>::load_simd(arch, centers.as_ptr().add(d));
                    let b = V::<A>::load_simd(arch, centers.as_ptr().add(self.dim + d));
                    for row in 0..ROWS {
                        let x = V::<A>::load_simd(arch, group[row].vector.as_ptr().add(d));
                        let da = x - a;
                        let db = x - b;
                        first[row] = da.mul_add_simd(da, first[row]);
                        second[row] = db.mul_add_simd(db, second[row]);
                    }
                }
            }
            let tail = self.dim - full;
            if tail != 0 {
                // SAFETY: The masked loads access exactly the tail coordinates.
                unsafe {
                    let a = V::<A>::load_simd_first(arch, centers.as_ptr().add(full), tail);
                    let b =
                        V::<A>::load_simd_first(arch, centers.as_ptr().add(self.dim + full), tail);
                    for row in 0..ROWS {
                        let x = V::<A>::load_simd_first(
                            arch,
                            group[row].vector.as_ptr().add(full),
                            tail,
                        );
                        let da = x - a;
                        let db = x - b;
                        first[row] = da.mul_add_simd(da, first[row]);
                        second[row] = db.mul_add_simd(db, second[row]);
                    }
                }
            }
            for row in 0..ROWS {
                output[row] = choose(first[row].sum_tree(), second[row].sum_tree());
            }
        }
        // A partial final group uses the reference kernel, including its architecture tail.
        for (point, output) in groups.remainder().iter().zip(outputs.into_remainder()) {
            *output = paired(point.vector, centers, self.dim);
        }
    }
}

#[cfg(test)]
mod tests;
