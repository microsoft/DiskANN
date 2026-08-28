/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_wide::{Architecture, SIMDMinMax, SIMDMulAdd, SIMDVector, arch::Scalar};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::multi_vector::distance_v2::{
    Check, blocks::fixed, num::Elements, ptr::Slice, util::Fold,
    kernel::micro::{Kernel, maxsim::MaxSim},
};

diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

macro_rules! stamp {
    ($arch:ty, $na:literal, $nb:literal) => {
        impl Kernel<
            fixed::FullBlockTranspose<'_, f32, $na, 1>,
            fixed::FullRowMajor<'_, f32, $nb>,
            &mut [f32; $na],
        > for MaxSim<$arch> {
            #[inline(always)]
            fn kernel(
                &self,
                a: fixed::FullBlockTranspose<'_, f32, $na, 1>,
                b: fixed::FullRowMajor<'_, f32, $nb>,
                cols: NonZeroUsize,
                c: &mut [f32; $na],
            ) {
                unsafe { block_transposed_x_row_major(*self, a, b, cols, c) }
            }
        }
    };
    ($arch:ty, $na:literal, { $($nb:literal),+ $(,)? }) => {
        $(stamp!($arch, $na, $nb);)+
    }
}

stamp!(V3, 16, { 4, 3, 2, 1 });
stamp!(V4, 16, { 4, 3, 2, 1 });

trait ExtraWide<const ELEMENTS: usize>: Copy {
    type Wide: Copy;
    type Splat: Copy;

    fn default(self) -> Self::Wide;
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide;
    fn splat(self, value: f32) -> Self::Splat;
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide;
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide;
    fn max_into(max: Self::Wide, into: &mut [f32; ELEMENTS]);
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<16> for MaxSim<V3> {
    type Wide = [f32x8<V3>; 2];
    type Splat = f32x8<V3>;

    fn default(self) -> Self::Wide {
        [SIMDVector::default(self.0), SIMDVector::default(self.0)]
    }

    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        slice.length().check(Check::eq(), 16);
        unsafe {
            [
                SIMDVector::load_simd(self.0, slice.as_ptr()),
                SIMDVector::load_simd(self.0, slice.add(Elements::new(8)).as_ptr()),
            ]
        }
    }

    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self.0, value)
    }

    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| a[i].mul_add_simd(b, acc[i]))
    }

    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| lhs[i].max_simd(rhs[i]))
    }

    fn max_into(lhs: Self::Wide, into: &mut [f32; 16]) {
        unsafe {
            lhs[0].store_simd(into.as_mut_ptr());
            lhs[1].store_simd(into.as_mut_ptr().add(8));
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<16> for MaxSim<V4> {
    type Wide = f32x16<V4>;
    type Splat = f32x16<V4>;

    fn default(self) -> Self::Wide {
        SIMDVector::default(self.0)
    }

    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        slice.length().check(Check::eq(), 16);
        unsafe { SIMDVector::load_simd(self.0, slice.as_ptr()) }
    }

    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self.0, value)
    }

    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        a.mul_add_simd(b, acc)
    }

    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        lhs.max_simd(rhs)
    }

    fn max_into(lhs: Self::Wide, into: &mut [f32; 16]) {
        unsafe {
            lhs.store_simd(into.as_mut_ptr());
        }
    }
}

#[inline(always)]
unsafe fn block_transposed_x_row_major<W, const NA: usize, const NB: usize>(
    wide: W,
    a: fixed::FullBlockTranspose<'_, f32, NA, 1>,
    b: fixed::FullRowMajor<'_, f32, NB>,
    k: NonZeroUsize,
    c: &mut [f32; NA],
) where
    W: ExtraWide<NA>,
    [W::Wide; NB]: Fold<W::Wide>,
{
    let k = k.get();

    // Check that everyone agrees.
    a.ncols().check(Check::eq(), k);
    b.ncols().check(Check::eq(), k);

    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut acc = [wide.default(); NB];

    let astride = a.stride(k);
    let bstride = b.stride(k);

    for i in 0..k {
        let ai = unsafe { wide.load(ap.add(astride * i).truncate(Elements::new(NA))) };

        for j in 0..NB {
            let bj = wide.splat(unsafe { bp.add(bstride * j + Elements::new(i)).read() });
            acc[j] = W::mul_add_splat(ai, bj, acc[j]);
        }
    }

    W::max_into(Fold::fold(acc, W::max), c);
}
