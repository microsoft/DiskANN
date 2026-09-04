// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MinMax4 document adapter for the packed `u8` by unpacked `u8` MaxSim kernel.

use std::num::NonZeroUsize;

use diskann_wide::Architecture;

use crate::matrix_kernels::{
    blocks::{packed, unpacked},
    driver,
    maxsim::packed_u8_x_unpacked_u8,
    num::DimK,
    ptr::Slice,
};
use crate::minmax::{MinMaxCompensation, MinMaxMeta};
use crate::multi_vector::MatRef;

pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a_values: &'a [u8],
    a_meta: &'a [MinMaxCompensation],
    a_rows: usize,
    b: MatRef<'a, MinMaxMeta<4>>,
    c: &'a mut [f32],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    pub(crate) fn new(
        arch: A,
        a_values: &'a [u8],
        a_meta: &'a [MinMaxCompensation],
        a_rows: usize,
        b: MatRef<'a, MinMaxMeta<4>>,
        c: &'a mut [f32],
        k: DimK,
    ) -> Self {
        debug_assert_eq!(a_values.len(), a_rows.div_ceil(MR) * k.value().get() * MR);
        debug_assert_eq!(a_meta.len(), a_rows);
        debug_assert_eq!(c.len(), a_rows);
        Self {
            arch,
            a_values,
            a_meta,
            a_rows,
            b,
            c,
            k,
        }
    }
}

impl<A, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR>
where
    A: Architecture,
    for<'a> packed_u8_x_unpacked_u8::Driver<'a, A, MR, NR>: driver::Drive,
{
    fn drive(&mut self) {
        if self.a_rows == 0 {
            self.c.fill(f32::MAX);
            return;
        }

        let mut b_values = Vec::with_capacity(self.b.num_vectors() * self.k.value().get());
        let mut b_meta = Vec::with_capacity(self.b.num_vectors());
        for row in self.b.rows() {
            b_meta.push(row.meta());
            let vector = row.vector();
            for i in 0..self.k.value().get() {
                // SAFETY: `i` is bounded by the common intrinsic dimension.
                b_values.push(unsafe { vector.get_unchecked(i) } as u8);
            }
        }

        if b_values.is_empty() {
            self.c.fill(f32::MAX);
            return;
        }

        let Some(a_blocks) = NonZeroUsize::new(self.a_rows.div_ceil(MR)) else {
            return;
        };
        let Some(b_extent) = NonZeroUsize::new(self.b.num_vectors()) else {
            return;
        };
        // SAFETY: Query packing allocates exactly `a_blocks * MR * k` values.
        let a = unsafe { packed::View::new(Slice::new(self.a_values), a_blocks, self.k) };
        // SAFETY: Every document contributes exactly `k` unpacked values.
        let b = unsafe { unpacked::View::new(Slice::new(&b_values), b_extent, self.k) };
        // SAFETY: The views and compensation slices describe the same matrices.
        let mut driver = unsafe {
            packed_u8_x_unpacked_u8::Driver::new(
                self.arch,
                a,
                b,
                packed_u8_x_unpacked_u8::Compensations {
                    a: self.a_meta,
                    b: &b_meta,
                },
                self.c,
                self.k,
            )
        };
        driver::Drive::drive(&mut driver);
    }
}
