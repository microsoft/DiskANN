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
    kernel::{self, maxsim::MaxSim},
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

pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: MutSlice<'a, f32>,
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
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

impl<A, const MR: usize, const NR: usize> kernel::Drive for Driver<'_, A, MR, NR>
where
    A: Copy,
    for<'a> PanelKernel<'a, A, MR, NR>: kernel::PanelKernel,
{
    #[inline(never)]
    fn drive(&mut self) {
        let on_a_panels = |a_panels: packed::View<'_, f32, MR>, a_block_base| {
            let on_b_panels = |b_panels: unpacked::View<'_, f32>, _| {
                let panel_kernel = |a_panel: packed::Panel<'_, f32, MR>, a_block_offset| {
                    let mut c = unsafe {
                        self.c
                            .subslice(MR * (a_block_base + a_block_offset), bounds::Bound::new(MR))
                    };

                    let c = unsafe { c.as_array::<MR>() };

                    let mut kernel =
                        unsafe { PanelKernel::new(self.kernel, a_panel, b_panels, c, self.k) };

                    kernel::PanelKernel::panel_kernel(&mut kernel);
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
pub(crate) struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::Panel<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
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

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr: literal, [ $($ns:literal),+ $(,)? ]) => {
        impl kernel::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline]
            fn panel_kernel(&mut self) {
                let on_b_panels = |b: unpacked::Panel<'_, f32, $nr>, _| {
                    let mut micro = unsafe {
                        MicroKernel::new(
                            self.kernel,
                            self.a,
                            b,
                            self.c,
                            self.k,
                        )
                    };

                    kernel::MicroKernel::micro_kernel(&mut micro);
                };

                let b_tail = unsafe { self.b.visit_panels::<$nr>(self.k, on_b_panels) };

                if let Some(b_tail) = b_tail {
                    // Repeitition Pattern.
                    $(
                        const { assert!($ns < $nr) };
                        if let Some(b_panel) = b_tail.try_as_panel::<$ns>() {
                            let mut micro = unsafe {
                                MicroKernel::new(
                                    self.kernel,
                                    self.a,
                                    b_panel,
                                    self.c,
                                    self.k,
                                )
                            };

                            kernel::MicroKernel::micro_kernel(&mut micro);
                        }
                    )+
                }
            }
        }
    }
}

panel_kernel!(Scalar, 8, 2, [1]);

panel_kernel!(V3, 16, 4, [1, 2, 3]);
panel_kernel!(V3, 16, 6, [1, 2, 3, 4, 5]);

panel_kernel!(V4, 16, 4, [1, 2, 3]);

//--------------//
// Micro Kernel //
//--------------//

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    kernel: MaxSim<A>,
    a: packed::Panel<'a, f32, MR>,
    b: unpacked::Panel<'a, f32, NR>,
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> MicroKernel<'a, A, MR, NR> {
    unsafe fn new(
        kernel: MaxSim<A>,
        a: packed::Panel<'a, f32, MR>,
        b: unpacked::Panel<'a, f32, NR>,
        c: &'a mut [f32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { kernel, a, b, c, k }
    }
}

#[inline(always)]
unsafe fn micro_kernel<W, const MR: usize, const NR: usize>(
    wide: W,
    a: packed::Panel<'_, f32, MR>,
    b: unpacked::Panel<'_, f32, NR>,
    c: &mut [f32; MR],
    k: DimK,
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
            let bj =
                wide.splat(*unsafe { bp.add(bstride * j + Elements::new(i)).as_unit().as_ref() });

            acc[j] = W::mul_add_splat(ai, bj, acc[j]);
        }
    }

    wide.max_into(Folder::fold(acc, W::max), c);
}

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl kernel::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                unsafe { micro_kernel(self.kernel, self.a, self.b, self.c, self.k) }
            }
        }
    };
    ($arch:ty, $mr:literal, { $($nr:literal),+ $(,)? }) => {
        $(micro_kernel!($arch, $mr, $nr);)+
    }
}

micro_kernel!(Scalar, 8, { 2, 1 });
micro_kernel!(V3, 16, { 6, 5, 4, 3, 2, 1 });
micro_kernel!(V4, 16, { 4, 3, 2, 1 });

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

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::views::Matrix;
    use rand::{SeedableRng, rngs::StdRng};

    use crate::multi_vector::distance_v2::test_util::TestDistr;

    fn generate_test_problem(
        mr: usize,
        k: usize,
        n: usize,
        rng: &mut impl rand::Rng,
    ) -> (Matrix<f32>, Matrix<f32>, Vec<f32>) {
        // What's happening here with the terminology is very confusing:
        //
        // * We go across logical rows of the left-hand side. Due to the block-transposed
        //   layout a logical row is actually a physical **column** of the `ref_a` we are
        //   using as a representation.
        //
        // * We go across logical columns of the right-hand side. Due to the **row-major**
        //   nature of `Matrix`, this means that logical columns are physical **rowsd**
        //   of `ref_b`.
        let ref_a = TestDistr::matrix::<f32>(k, mr, rng);
        let ref_b = TestDistr::matrix::<f32>(n, k, rng);

        let ref_c: Vec<f32> = (0..mr)
            .map(|lhs_logical_row| {
                let mut max_ip = f32::NEG_INFINITY;
                ref_b.row_iter().for_each(|rhs_physical_col| {
                    let mut this_ip = 0.0;
                    for (k, b) in rhs_physical_col.iter().enumerate() {
                        this_ip = b.mul_add(ref_a[(k, lhs_logical_row)], this_ip);
                    }

                    max_ip = max_ip.max(this_ip);
                });

                max_ip
            })
            .collect();

        (ref_a, ref_b, ref_c)
    }

    /////////////////
    // MicroKernel //
    /////////////////

    fn test_micro_kernel<A, const MR: usize, const NR: usize>(
        arch: A,
        k: DimK,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) where
        for<'a> MicroKernel<'a, A, MR, NR>: kernel::MicroKernel,
    {
        let (ref_a, ref_b, ref_c) = generate_test_problem(MR, k.value().get(), NR, rng);

        let mut c = [f32::NEG_INFINITY; MR];

        // Run the test kernel.
        let mut kernel = unsafe {
            MicroKernel::new(
                MaxSim::new(arch),
                packed::Panel::new(Slice::new(ref_a.as_slice()), k),
                unpacked::Panel::new(Slice::new(ref_b.as_slice()), k),
                &mut c,
                k,
            )
        };

        kernel::MicroKernel::micro_kernel(&mut kernel);
        assert_eq!(&*ref_c, kernel.c, "{ctx}");

        // Try again - but this time use a value that is much bigger than the what should
        // be generated by the test problem.
        //
        // This checks that we don't just overwite existing contents.
        let new_c = kernel.c.map(|i| i + 1.0);
        *kernel.c = new_c;

        kernel::MicroKernel::micro_kernel(&mut kernel);
        assert_eq!(new_c, *kernel.c, "{ctx}");
    }

    macro_rules! test_micro_kernel {
        (
            $fn:ident,
            $arch:expr,
            $seed:literal,
            $(
                $MR:literal => { $($NR:literal),+ $(,)? }
            ),+ $(,)?
        ) => {
            #[test]
            fn $fn() {
                if let Some(arch) = $arch {
                    let mut rng = StdRng::seed_from_u64($seed);

                    for k in [1, 2, 5, 8] {
                        let k = DimK::new(NonZeroUsize::new(k).unwrap());

                        $(
                            $(
                                test_micro_kernel::<_, $MR, $NR>(
                                    arch,
                                    k,
                                    &mut rng,
                                    format_args!("k = {:?}", k),
                                );
                            )+
                        )+
                    }
                }
            }
        }
    }

    test_micro_kernel!(
        test_micro_kernel_scalar,
        Some(Scalar::new()),
        0x759722e1a83e4566,
        8 => { 2, 1 },
    );

    test_micro_kernel!(
        test_micro_kernel_v3,
        V3::new_checked(),
        0x157f59d99f648437,
        16 => { 6, 5, 4, 3, 2, 1},
    );

    test_micro_kernel!(
        test_micro_kernel_v4,
        V4::new_checked_miri(),
        0xca13f736977f96fe,
        16 => { 4, 3, 2, 1},
    );

    /////////////////
    // PanelKernel //
    /////////////////

    // The panel kernel operates on a single A-panel with multiple B-panels.
    //
    // This test sweeps over a number of rows for the B-panels to exercise all possible
    // corner cases.
    //
    // We can reuse `generate_test_problem` as the expected answer remains the same as
    // for the micro-kernel.
    fn test_panel_kernel<A, const MR: usize, const NR: usize>(
        arch: A,
        k: DimK,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) where
        A: Copy,
        for<'a> PanelKernel<'a, A, MR, NR>: kernel::PanelKernel,
    {
        for blocks in 0..4 {
            for remainder in 0..NR {
                let cols = NR * blocks + remainder;
                if cols == 0 {
                    continue;
                }

                let (ref_a, ref_b, ref_c) = generate_test_problem(MR, k.value().get(), cols, rng);

                let extent = NonZeroUsize::new(cols).unwrap();

                let mut c = [f32::NEG_INFINITY; MR];

                let mut kernel = unsafe {
                    PanelKernel::new(
                        MaxSim::new(arch),
                        packed::Panel::new(Slice::new(ref_a.as_slice()), k),
                        unpacked::View::new(Slice::new(ref_b.as_slice()), extent, k),
                        &mut c,
                        k,
                    )
                };

                kernel::PanelKernel::panel_kernel(&mut kernel);
                assert_eq!(&*ref_c, kernel.c, "{ctx}");

                // Try again - but this time use a value that is much bigger than the what
                // should be generated by the test problem.
                //
                // This checks that we don't just overwite existing contents.
                let new_c = kernel.c.map(|i| i + 1.0);
                *kernel.c = new_c;

                kernel::PanelKernel::panel_kernel(&mut kernel);
                assert_eq!(new_c, *kernel.c, "{ctx}");
            }
        }
    }

    macro_rules! test_panel_kernel {
        (
            $fn:ident,
            $arch:expr,
            $seed:literal,
            $(
                (
                    $MR:literal, $NR:literal
                )
            ),+ $(,)?
        ) => {
            #[test]
            fn $fn() {
                if let Some(arch) = $arch {
                    let mut rng = StdRng::seed_from_u64($seed);

                    for k in [1, 2, 5, 8] {
                        let k = DimK::new(NonZeroUsize::new(k).unwrap());

                        $(
                            test_panel_kernel::<_, $MR, $NR>(
                                arch,
                                k,
                                &mut rng,
                                format_args!("k = {:?}", k),
                            );
                        )+
                    }
                }
            }
        }
    }

    test_panel_kernel!(
        test_panel_kernel_scalar_8x2,
        Some(Scalar::new()),
        0x2c03eb9ee51d30c3,
        (8, 2),
    );

    test_panel_kernel!(
        test_panel_kernel_v3_16x4,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 4),
    );

    test_panel_kernel!(
        test_panel_kernel_v3_16x6,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 6),
    );

    test_panel_kernel!(
        test_panel_kernel_v4_16x4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 4),
    );
}
