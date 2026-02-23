/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{SIMDMulAdd, SIMDMinMax, SIMDVector, arch::x86_64::V3};
use diskann_utils::views::MatrixView;

use crate::algorithms::kmeans::BlockTranspose;

diskann_wide::alias!(f32x8 = <V3>::f32x8);

// pub unsafe fn test_function(
//     arch: V3,
//     a_packed: *const f32,
//     b: *const f32,
//     k: usize,
//     r0: &mut f32x8,
//     r1: &mut f32x8,
// ) {
//     let op = |x: f32x8, y: f32x8| x.max_simd(y);
//     unsafe { microkernel(arch, a_packed, b, k, r0, r1, op) }
// }

#[inline(never)]
#[cold]
fn test_function_panic(
) {
    // assert_eq!(scratch.len(), a.nrows())
    //     || !a.nrows().is_multiple_of(32)
    //     || !b.nrows().is_multiple_of(4)
    //     || a.ncols() != b.ncols() {
    //     test_function_panic();
}

pub fn test_function(
    arch: V3,
    a: &BlockTranspose<16>,
    b: MatrixView<'_, f32>,
    scratch: &mut [f32],
) {
    // Let's get this out of the way.
    if scratch.len() != a.nrows()
        || !a.nrows().is_multiple_of(32)
        || !b.nrows().is_multiple_of(4)
        || a.ncols() != b.ncols() {
        test_function_panic();
    }

    let pa_packed = a.as_ptr();
    let pb = b.as_ptr();
    let pr = scratch.as_mut_ptr();

    let k = a.ncols();

    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

    const A_TILE: usize = 128;
    const B_TILE: usize = 128;

    const { assert!(A_TILE.is_multiple_of(A_PANEL)) };
    const { assert!(B_TILE.is_multiple_of(B_PANEL)) };

    let op = |x: f32x8, y: f32x8| x.max_simd(y);

    // Precompute strides (in elements, not bytes).
    let a_panel_stride = A_PANEL * k;
    let a_tile_stride = A_TILE * k;
    let b_panel_stride = B_PANEL * k;
    let b_tile_stride = B_TILE * k;

    let pa_end = pa_packed.wrapping_add(a.nrows() * k);
    let pb_end = pb.wrapping_add(b.nrows() * k);

    // Party time!
    //
    // TODO: Maybe peel off the last iteration to keep the loop bounds a bit tighter.
    unsafe {
        let mut pa_tile = pa_packed;
        let mut pr_tile = pr;
        while pa_tile < pa_end {
            let pa_tile_end = pa_tile.add(a_tile_stride.min(pa_end.offset_from_unsigned(pa_tile)));

            let mut pb_tile = pb;
            while pb_tile < pb_end {
                let pb_tile_end =
                    pb_tile.add(b_tile_stride.min(pb_end.offset_from_unsigned(pb_tile)));

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;
                    while pb_panel < pb_tile_end {
                        microkernel(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }
                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
                pb_tile = pb_tile.add(b_tile_stride);
            }
            pa_tile = pa_tile.add(a_tile_stride);
            pr_tile = pr_tile.add(A_TILE);
        }
    }
}

// TODO: Unroll loops.
#[inline(always)]
pub unsafe fn microkernel<Op>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
    reduce: Op,
) where
    Op: Fn(f32x8, f32x8) -> f32x8,
{
    let mut p00 = f32x8::default(arch);
    let mut p10 = f32x8::default(arch);

    let mut p01 = f32x8::default(arch);
    let mut p11 = f32x8::default(arch);

    let mut p02 = f32x8::default(arch);
    let mut p12 = f32x8::default(arch);

    let mut p03 = f32x8::default(arch);
    let mut p13 = f32x8::default(arch);

    let o0 = 0;
    let o1 = k;
    let o2 = 2 * k;
    let o3 = 3 * k;

    let a_stride = 2 * f32x8::LANES;
    let a_stride_half = f32x8::LANES;

    for i in 0..k {
        unsafe {
            let a0 = f32x8::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32x8::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            let b0 = f32x8::splat(arch, b.add(i + o0).read_unaligned());
            p00 = a0.mul_add_simd(b0, p00);
            p10 = a1.mul_add_simd(b0, p10);

            let b1 = f32x8::splat(arch, b.add(i + o1).read_unaligned());
            p01 = a0.mul_add_simd(b1, p01);
            p11 = a1.mul_add_simd(b1, p11);

            let b2 = f32x8::splat(arch, b.add(i + o2).read_unaligned());
            p02 = a0.mul_add_simd(b2, p02);
            p12 = a1.mul_add_simd(b2, p12);

            let b3 = f32x8::splat(arch, b.add(i + o3).read_unaligned());
            p03 = a0.mul_add_simd(b3, p03);
            p13 = a1.mul_add_simd(b3, p13);
        }
    }

    let mut r0 = unsafe { f32x8::load_simd(arch, r) };
    let mut r1 = unsafe { f32x8::load_simd(arch, r.add(f32x8::LANES)) };

    r0 = reduce(r0, reduce(reduce(p00, p01), reduce(p02, p03)));
    r1 = reduce(r1, reduce(reduce(p10, p11), reduce(p12, p13)));

    unsafe { r0.store_simd(r) };
    unsafe { r1.store_simd(r.add(f32x8::LANES)) };
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_utils::views::Matrix;

    #[test]
    fn test() {
        let a = Matrix::new(1.0f32, 32, 128);
        let a_packed = BlockTranspose::<16>::from_matrix_view(a.as_view());
        let b = Matrix::new(1.0f32, 128, 128);
        let mut c = vec![f32::NEG_INFINITY; a.nrows()];

        test_function(
            diskann_wide::ARCH,
            &a_packed,
            b.as_view(),
            &mut c
        );

        let f = c.iter().map(|i| *i).sum::<f32>();
        println!("f = {:?}", f);
    }
}

