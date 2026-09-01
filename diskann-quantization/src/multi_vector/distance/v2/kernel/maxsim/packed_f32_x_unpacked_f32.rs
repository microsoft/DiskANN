/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector, arch::Scalar};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::multi_vector::distance::v2::{
    Cache,
    blocks::{packed, unpacked},
    bounds,
    kernel::{self, maxsim::MaxSim},
    num::{Bytes, DimK, Elements, value_or_one},
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

impl Params {
    pub(super) fn new(cache: Cache, a_panel: Bytes, b_col: Bytes, nr: usize) -> Self {
        // Pick the number of A-panels to process at a time so the working set is within
        // the L2 cache.
        let a_panels_in_l2 = value_or_one(cache.l2().get() / a_panel.value());

        // Pick the number of B-panels to process to the `B` working set plus a single
        // panel of `A` fits in the L1 cache.
        let b_budget = cache.l1().get().saturating_sub(a_panel.value());
        let b_cols_in_l1 = value_or_one(nr * (b_budget.div_ceil(nr * b_col.value())));

        Self {
            a_panels_in_l2,
            b_cols_in_l1,
        }
    }
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
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must be equal to `k`. Bound `c.len()` must be equal
    /// to `a.extent()`.
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

        unsafe {
            Self::new_inner(
                arch,
                a,
                b,
                c,
                k,
                Params::new(cache, a.block_stride(k).bytes(), b.stride(k).bytes(), NR),
            )
        }
    }

    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: MutSlice<'a, f32>,
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            c.len(),
            a.extent(),
            "output slice must have one entry for every row in `a`",
        );

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
    fn drive(&mut self) {
        // SAFETY: Class invariant - the length of `self.c` must be equal to `self.a.extent()`
        unsafe { self.c.as_std_mut_slice(self.a.extent().get()) }.fill(f32::NEG_INFINITY);

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
    ///
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must both be equal to `k`.
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
panel_kernel!(Scalar, 8, 4, [1, 2, 3]);
panel_kernel!(Scalar, 8, 6, [1, 2, 3, 4, 5]);

panel_kernel!(V3, 16, 4, [1, 2, 3]);
panel_kernel!(V3, 16, 6, [1, 2, 3, 4, 5]);

panel_kernel!(V4, 16, 4, [1, 2, 3]);
panel_kernel!(V4, 16, 6, [1, 2, 3, 4, 5]);

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
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must be equal to `k`.
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

/// # Safety
///
/// Bounds `a.k()` and `b.k()` must be equal to `k`.
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

micro_kernel!(Scalar, 8, { 6, 5, 4, 3, 2, 1 });
micro_kernel!(V3, 16, { 6, 5, 4, 3, 2, 1 });
micro_kernel!(V4, 16, { 6, 5, 4, 3, 2, 1 });

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

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [SIMDVector::default(self.0), SIMDVector::default(self.0)]
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 8);

        unsafe {
            [
                SIMDVector::load_simd(self.0, slice.as_ptr()),
                SIMDVector::load_simd(self.0, slice.add(Elements::new(4)).as_ptr()),
            ]
        }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self.0, value)
    }

    #[inline(always)]
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| a[i].mul_add_simd(b, acc[i]))
    }

    #[inline(always)]
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| lhs[i].max_simd(rhs[i]))
    }

    #[inline(always)]
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

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [SIMDVector::default(self.0), SIMDVector::default(self.0)]
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);
        unsafe {
            [
                SIMDVector::load_simd(self.0, slice.as_ptr()),
                SIMDVector::load_simd(self.0, slice.add(Elements::new(8)).as_ptr()),
            ]
        }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self.0, value)
    }

    #[inline(always)]
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| a[i].mul_add_simd(b, acc[i]))
    }

    #[inline(always)]
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| lhs[i].max_simd(rhs[i]))
    }

    #[inline(always)]
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

    #[inline(always)]
    fn default(self) -> Self::Wide {
        SIMDVector::default(self.0)
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);

        unsafe { SIMDVector::load_simd(self.0, slice.as_ptr()) }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self.0, value)
    }

    #[inline(always)]
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        a.mul_add_simd(b, acc)
    }

    #[inline(always)]
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        lhs.max_simd(rhs)
    }

    #[inline(always)]
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

    use rand::{SeedableRng, rngs::StdRng};

    use crate::multi_vector::{BlockTransposed, distance::v2::kernel::maxsim};

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
        let (ref_a, ref_b, ref_c) = maxsim::test::generate(MR, k.value().get(), NR, rng);

        // From the reference problem, we need to transpose both `ref_a` and `ref_b` to get
        // them into the desired format.
        let ref_a = ref_a.transpose();
        let ref_b = ref_b.transpose();

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

                let (ref_a, ref_b, ref_c) = maxsim::test::generate(MR, k.value().get(), cols, rng);

                // From the reference problem, we need to transpose both `ref_a` and `ref_b` to get
                // them into the desired format.
                let ref_a = ref_a.transpose();
                let ref_b = ref_b.transpose();

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
        test_panel_kernel_scalar,
        Some(Scalar::new()),
        0x2c03eb9ee51d30c3,
        (8, 2),
    );

    test_panel_kernel!(
        test_panel_kernel_v3,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 4),
        (16, 6),
    );

    test_panel_kernel!(
        test_panel_kernel_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 4),
    );

    ////////////
    // Driver //
    ////////////

    fn test_driver<A, const MR: usize, const NR: usize>(arch: A, rng: &mut impl rand::Rng)
    where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR>: kernel::Drive,
    {
        // (a-panels-per-tile, a-rows, b-cols-per-tile, b-cols, k)
        let cases = [
            (1, MR * 1, 1, 1, 1),           // Smallest valid setup
            (1, MR * 3, 1, 3, 1),           // Unit advancement, no reuse.
            (2, MR * 2, 2 * NR, 2 * NR, 3), // Values a direct multiple of the blocking.
            (2, MR * 3, 2 * NR, NR, 3),     //
            (2, MR * 1, 2 * NR, 2 * NR + 1, 3),
            (2, MR * 3, 2 * NR, 2 * NR + 1, 5),
            (2, MR * 5, 2 * NR, 4 * NR + 1, 1),
        ];

        for case in cases {
            let (a_panels_per_tile, a_rows, b_cols_per_tile, b_cols, k) = case;

            let k = DimK::new(NonZeroUsize::new(k).unwrap());

            let (ref_a, ref_b, ref_c) =
                maxsim::test::generate(a_rows, k.value().get(), b_cols, rng);

            // Massage the input data in the form needed by the kernel.
            let a_bt = BlockTransposed::<f32, MR>::from_matrix_view(ref_a.as_view());
            let b = ref_b.transpose();

            let mut c = vec![f32::NAN; a_bt.padded_nrows()];

            let mut driver = unsafe {
                Driver::new_inner(
                    arch,
                    packed::View::from_block_transposed(a_bt.as_view()),
                    unpacked::View::from_matrix_view(b.as_view()),
                    MutSlice::new(&mut c),
                    k,
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(a_panels_per_tile).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(b_cols_per_tile).unwrap(),
                    },
                )
            };

            kernel::Drive::drive(&mut driver);

            assert_eq!(
                ref_c, c,
                "a_panels_per_tile: {}, a_rows: {}, b_cols_per_tile: {}, b_cols: {}, k: {:?}",
                a_panels_per_tile, a_rows, b_cols_per_tile, b_cols, k,
            );
        }
    }

    macro_rules! test_driver {
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

                    $(test_driver::<_, $MR, $NR>(arch, &mut rng);)+
                }
            }
        }
    }

    test_driver!(
        test_driver_scalar,
        Some(Scalar::new()),
        0x2c03eb9ee51d30c3,
        (8, 2),
    );

    test_driver!(
        test_driver_v3,
        V3::new_checked(),
        0x2c03eb9ee51d30c3,
        (16, 4),
        (16, 6),
    );

    test_driver!(
        test_driver_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 4),
    );
}
