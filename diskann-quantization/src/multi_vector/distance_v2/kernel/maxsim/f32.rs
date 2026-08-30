/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector, arch::Scalar};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::multi_vector::distance_v2::{
    Cache,
    blocks::{packed, unpacked},
    bounds,
    kernel::{Drive, MicroKernel, PanelKernel, maxsim::MaxSim},
    num::{DimK, Elements, value_or_one},
    ptr::{MutSlice, Slice},
    util::{Fold, Folder},
};

//--------//
// Driver //
//--------//

#[derive(Debug, Clone, Copy)]
pub(super) struct Params {
    /// The (approximate) number of blocks of `A` that fit in the L2 cache.
    pub(super) a_panels_in_l2: NonZeroUsize,
    /// The (approximate) number of columns of `B` that fis in the L1 cache.
    pub(super) b_cols_in_l1: NonZeroUsize,
}

pub(crate) struct PackedXUnpacked<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: MutSlice<'a, f32>,
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> PackedXUnpacked<'a, A, MR, NR> {
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: MutSlice<'a, f32>,
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            c.len(),
            a.extent(),
            "output slice must have one entry for every row in `a`",
        );

        // Pick the number of A-panels to process at a time so the working set is within
        // the L2 cache.
        let a_panel_bytes = a.block_stride(k).bytes();
        let a_panels_in_l2 = value_or_one(cache.l2().get() / a_panel_bytes);

        // Pick the number of B-panels to process to the `B` working set plus a single
        // panel of `A` fits in the L1 cache.
        let b_budget = cache.l1().get().saturating_sub(a_panel_bytes);
        let b_cols_in_l1 = value_or_one(NR * (b_budget / (NR * b.stride(k).bytes())));

        let params = Params {
            a_panels_in_l2,
            b_cols_in_l1,
        };

        Self {
            kernel: MaxSim::new(arch),
            a,
            b,
            c,
            k,
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize> Drive for PackedXUnpacked<'_, A, MR, NR>
where
    A: Copy,
    for<'a> BlockWithRowMajor<'a, A, MR, NR>: PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        let on_a_panels = |a_panels: packed::View<'_, f32, MR>, a_block_base| {
            let on_b_panels = |b_panels: unpacked::View<'_, f32>| {
                let panel_kernel = |a_panel: packed::Panel<'_, f32, MR>, a_block_offset| {
                    let mut c = unsafe {
                        self.c
                            .subslice(MR * (a_block_base + a_block_offset), bounds::Bound::new(MR))
                    };

                    let mut kernel = unsafe {
                        BlockWithRowMajor::new(
                            self.kernel,
                            a_panel,
                            b_panels,
                            unsafe { c.materialize::<MR>() },
                            self.k,
                        )
                    };

                    kernel.panel_kernel();
                };

                unsafe {
                    a_panels.visit_panels(self.k, panel_kernel);
                }
            };

            unsafe {
                self.b
                    .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
            }
        };

        unsafe {
            self.a
                .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels)
        };
    }
}

//-------------//
// PanelKernel //
//-------------//

#[derive(Debug)]
pub(crate) struct BlockWithRowMajor<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::Panel<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> BlockWithRowMajor<'a, A, MR, NR> {
    /// Construct a new kernel.
    pub(crate) unsafe fn new(
        kernel: MaxSim<A>,
        a: packed::Panel<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: &'a mut [f32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { kernel, a, b, c, k }
    }
}

macro_rules! stamp {
    ($arch:ty, $na:literal, $nb: literal, [ $($ns:literal),+ $(,)? ]) => {
        impl PanelKernel for BlockWithRowMajor<'_, $arch, $na, $nb> {
            #[inline]
            fn panel_kernel(&mut self) {
                let b_tail = unsafe { self.b.visit_panels::<$nb>(
                    self.k,
                    |b| {
                        MicroKernel::kernel(
                            &self.kernel,
                            self.a,
                            b,
                            self.k,
                            self.c
                        );
                    }
                ) };

                if let Some(b_tail) = b_tail {
                    let ncols = b_tail.extent().get();

                    // Repeitition Pattern.
                    $(
                        const { assert!($ns < $nb) };
                        if ncols == $ns {
                            MicroKernel::kernel(
                                &self.kernel,
                                self.a,
                                unsafe { b_tail.materialize::<$ns>() },
                                self.k,
                                self.c
                            );
                        }
                    )+
                }
            }
        }
    }
}

stamp!(Scalar, 8, 2, [1]);
stamp!(V3, 16, 4, [1, 2, 3]);
stamp!(V3, 16, 6, [1, 2, 3, 4, 5]);

stamp!(V4, 16, 4, [1, 2, 3]);

//--------------//
// Micro Kernel //
//--------------//

#[inline(always)]
unsafe fn micro_kernel<W, const MR: usize, const NR: usize>(
    wide: W,
    a: packed::Panel<'_, f32, MR>,
    b: unpacked::Panel<'_, f32, NR>,
    k: DimK,
    c: &mut [f32; MR],
) where
    W: ExtraWide<MR>,
    Folder: Fold<NR>,
{
    // Check that everyone agrees.
    bounds::check_eq!(a.k(), k);
    bounds::check_eq!(b.k(), k);

    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut acc = [wide.default(); NR];

    let astride = a.stride(k);
    let bstride = b.stride(k);

    for i in 0..k.value().get() {
        let ai = unsafe { wide.load(ap.add(astride * i).truncate(Elements::new(MR))) };

        for j in 0..NR {
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
            packed::Panel<'_, f32, $na>,
            unpacked::Panel<'_, f32, $nb>,
            &mut [f32; $na],
        > for MaxSim<$arch> {
            #[inline(always)]
            fn kernel(
                &self,
                a: packed::Panel<'_, f32, $na>,
                b: unpacked::Panel<'_, f32, $nb>,
                k: DimK,
                c: &mut [f32; $na],
            ) {
                unsafe { micro_kernel(*self, a, b, k, c) }
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
