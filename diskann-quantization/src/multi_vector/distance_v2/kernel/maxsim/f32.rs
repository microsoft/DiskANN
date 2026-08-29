/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector, arch::Scalar};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::multi_vector::distance_v2::{
    blocks, bounds,
    kernel::{MicroKernel, PanelKernel, maxsim::MaxSim},
    num::{AllColumns, Elements},
    ptr::Slice,
    util::{Fold, Folder},
};

#[derive(Debug)]
pub(crate) struct BlockWithRowMajor<'a, A, const NA: usize, const NB: usize> {
    kernel: MaxSim<A>,
    a: blocks::fixed::FullBlockTranspose<'a, f32, NA>,
    b: blocks::dynamic::RowMajor<'a, f32>,
    c: &'a mut [f32; NA],
    cols: AllColumns,
}

impl<'a, A, const NA: usize, const NB: usize> BlockWithRowMajor<'a, A, NA, NB> {
    /// Construct a new kernel.
    ///
    /// # Safety
    ///
    /// Callers asserts that the number of columns in `a` and `b` are equal to `cols`.
    pub(crate) unsafe fn new(
        kernel: MaxSim<A>,
        a: blocks::fixed::FullBlockTranspose<'a, f32, NA>,
        b: blocks::dynamic::RowMajor<'a, f32>,
        c: &'a mut [f32; NA],
        cols: AllColumns,
    ) -> Self {
        bounds::check_eq!(a.ncols(), cols);
        bounds::check_eq!(b.ncols(), cols);

        Self {
            kernel,
            a,
            b,
            c,
            cols,
        }
    }
}

trait TailDispatch {
    fn tail_dispatch(&mut self);
}

macro_rules! tail_dispatch {
    ($arch:ty, $na:literal, $nb: literal, [ $($ns:literal),+ $(,)? ]) => {
        impl PanelKernel for BlockWithRowMajor<'_, $arch, $na, $nb> {
            #[inline]
            fn panel_kernel(&mut self) {
                let b_tail = unsafe { self.b.visit_all_rows_fixed::<$nb>(
                    self.cols,
                    |b| {
                        MicroKernel::kernel(
                            &self.kernel,
                            self.a,
                            b,
                            self.cols.value(),
                            self.c
                        );
                    }
                ) };

                if let Some(b_tail) = b_tail {
                    let nrows = b_tail.nrows();

                    // Repeitition Pattern.
                    $(
                        const { assert!($ns < $nb) };
                        if nrows == $ns {
                            MicroKernel::kernel(
                                &self.kernel,
                                self.a,
                                unsafe { b_tail.materialize::<$ns>() },
                                self.cols.value(),
                                self.c
                            );
                        }
                    )+
                }
            }
        }
    }
}

tail_dispatch!(Scalar, 8, 2, [1]);
tail_dispatch!(V3, 16, 4, [1, 2, 3]);
tail_dispatch!(V3, 16, 6, [1, 2, 3, 4, 5]);

tail_dispatch!(V4, 16, 4, [1, 2, 3]);

//--------------//
// Micro Kernel //
//--------------//

#[inline(always)]
unsafe fn micro_kernel<W, const NA: usize, const NB: usize>(
    wide: W,
    a: blocks::fixed::FullBlockTranspose<'_, f32, NA, 1>,
    b: blocks::fixed::FullRowMajor<'_, f32, NB>,
    k: NonZeroUsize,
    c: &mut [f32; NA],
) where
    W: ExtraWide<NA>,
    Folder: Fold<NB>,
{
    let k = k.get();

    // Check that everyone agrees.
    bounds::check_eq!(a.ncols(), k);
    bounds::check_eq!(b.ncols(), k);

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

    wide.max_into(Folder::fold(acc, W::max), c);
}

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

macro_rules! stamp {
    ($arch:ty, $na:literal, $nb:literal) => {
        impl MicroKernel<
            blocks::fixed::FullBlockTranspose<'_, f32, $na, 1>,
            blocks::fixed::FullRowMajor<'_, f32, $nb>,
            &mut [f32; $na],
        > for MaxSim<$arch> {
            #[inline(always)]
            fn kernel(
                &self,
                a: blocks::fixed::FullBlockTranspose<'_, f32, $na, 1>,
                b: blocks::fixed::FullRowMajor<'_, f32, $nb>,
                cols: NonZeroUsize,
                c: &mut [f32; $na],
            ) {
                unsafe { micro_kernel(*self, a, b, cols, c) }
            }
        }
    };
    ($arch:ty, $na:literal, { $($nb:literal),+ $(,)? }) => {
        $(stamp!($arch, $na, $nb);)+
    }
}

stamp!(Scalar, 8, { 2, 1 });
stamp!(V3, 16, { 6, 5, 4, 3, 2, 1 });
stamp!(V4, 16, { 4, 3, 2, 1 });

trait ExtraWide<const ELEMENTS: usize>: Copy {
    type Wide: Copy;
    type Splat: Copy;

    fn default(self) -> Self::Wide;
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide;
    fn splat(self, value: f32) -> Self::Splat;
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide;
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide;
    fn max_into(self, max: Self::Wide, into: &mut [f32; ELEMENTS]);
}

impl ExtraWide<8> for MaxSim<Scalar> {
    type Wide = [f32x4<Scalar>; 2];
    type Splat = f32x4<Scalar>;

    fn default(self) -> Self::Wide {
        [SIMDVector::default(self.0), SIMDVector::default(self.0)]
    }

    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 8);

        unsafe {
            [
                SIMDVector::load_simd(self.0, slice.as_ptr()),
                SIMDVector::load_simd(self.0, slice.add(Elements::new(4)).as_ptr()),
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

    fn max_into(self, lhs: Self::Wide, into: &mut [f32; 8]) {
        let previous = unsafe { self.load(Slice::new(into)) };
        let max = Self::max(lhs, previous);

        unsafe {
            max[0].store_simd(into.as_mut_ptr());
            max[1].store_simd(into.as_mut_ptr().add(4));
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<16> for MaxSim<V3> {
    type Wide = [f32x8<V3>; 2];
    type Splat = f32x8<V3>;

    fn default(self) -> Self::Wide {
        [SIMDVector::default(self.0), SIMDVector::default(self.0)]
    }

    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);
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

    fn max_into(self, lhs: Self::Wide, into: &mut [f32; 16]) {
        let previous = unsafe { self.load(Slice::new(into)) };
        let max = Self::max(lhs, previous);

        unsafe {
            max[0].store_simd(into.as_mut_ptr());
            max[1].store_simd(into.as_mut_ptr().add(8));
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
        bounds::check_eq!(slice.len(), 16);
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

    fn max_into(self, lhs: Self::Wide, into: &mut [f32; 16]) {
        let previous = unsafe { self.load(Slice::new(into)) };
        let max = Self::max(lhs, previous);

        unsafe {
            max.store_simd(into.as_mut_ptr());
        }
    }
}
