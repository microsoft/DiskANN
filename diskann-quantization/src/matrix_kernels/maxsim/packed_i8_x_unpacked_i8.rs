/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The integer counterpart of [`super::packed_f32_x_unpacked_f32`], accumulating in `i32`.
//!
//! The blocking strategy is identical. The difference is that `a` interleaves `PACK`
//! consecutive contraction indices within each packed row so that a single lane of the
//! widening dot-product instructions consumes `PACK` products at a time. `PACK` is chosen
//! per architecture to match the instruction used.
//!
//! Each product is bounded by `128 * 128`, so the `i32` accumulator is exact and cannot
//! overflow for contraction dimensions up to `131_071`.

use diskann_wide::arch::{Architecture, Scalar};
use diskann_wide::{SIMDDotProduct, SIMDMinMax, SIMDReinterpret, SIMDVector};

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{DimK, Elements},
    ptr::{MutSlice, Slice},
    util::{self, Fold, Folder},
};

use super::packed_f32_x_unpacked_f32::Params;

diskann_wide::alias!(i8x16<A> = i8x16);
diskann_wide::alias!(i16x16<A> = i16x16);
diskann_wide::alias!(i32x8<A> = i32x8);
diskann_wide::alias!(u32x8<A> = u32x8);

/// Widen a `PACK = 2` group into the little-endian `i16` lane pair consumed by the 16-bit
/// dot products.
#[inline(always)]
fn i16_pair([lo, hi]: [i8; 2]) -> u32 {
    u32::from(i16::from(lo) as u16) | (u32::from(i16::from(hi) as u16) << 16)
}

//--------//
// Driver //
//--------//

/// A driver for prepacked by unpacked integer "maxsim" computations.
///
/// See [`super::packed_f32_x_unpacked_f32::Driver`] for the blocking strategy and for the
/// rationale behind invariant (2).
///
/// # Class Invariants
///
/// 1. `a.k()` and `b.k()` must be equal to `k`.
/// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: packed::View<'a, i8, MR, PACK>,
    b: unpacked::View<'a, i8>,
    c: &'a mut [i32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize, const PACK: usize> Driver<'a, A, MR, NR, PACK> {
    /// Prepare for a maxsim on `a` and `b` with the results stored directly into `c`.
    ///
    /// `c` does not require any specific initial value.
    ///
    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, i8, MR, PACK>,
        b: unpacked::View<'a, i8>,
        c: &'a mut [i32],
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks",
        );

        // SAFETY: Inherited from caller.
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

    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, i8, MR, PACK>,
        b: unpacked::View<'a, i8>,
        c: &'a mut [i32],
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(b.k(), k, "contraction dimensions do not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupy exactly the packed A blocks",
        );

        Self {
            arch,
            a,
            b,
            c,
            k,
            params,
        }
    }
}

impl<A, const MR: usize, const NR: usize, const PACK: usize> driver::Drive
    for Driver<'_, A, MR, NR, PACK>
where
    A: util::LoadStore<i32, MR> + Architecture,
    for<'a> PanelKernel<'a, A, MR, NR, PACK>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                // Pre-fill `c`.
                self.c.fill(i32::MIN);

                // We allow `c` to be slightly under-filled.
                //
                // These variables track if under-fill is happening.
                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;

                let mut c = MutSlice::new(self.c);

                let on_a_panels = |a_panels: packed::View<'_, i8, MR, PACK>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, i8>, _| {
                        let panel_kernel =
                            |a_panel: packed::Panel<'_, i8, MR, PACK>, a_block_offset| {
                                // If we are in the very last block and we need to sub-fill, do
                                // that. Otherwise, reference the output in place.
                                let a_block = a_block_base + a_block_offset;
                                let handling_tail = a_block == last_a_block && remainder != 0;

                                let bound = bounds::Bound::from_fn(|| {
                                    if handling_tail { remainder } else { MR }
                                });

                                // SAFETY: By class invariant,
                                //
                                // `MR * (self.a.blocks() - 1) < c.len() <= MR * self.a.blocks()`.
                                //
                                // From the visitor, `a_block <= self.a.blocks()`.
                                let mut region = unsafe { c.subslice(MR * a_block, bound) };
                                let c = if handling_tail {
                                    util::LoadStore::<i32, MR>::load(
                                        self.arch,
                                        // SAFETY: `region` as length exactly `remainder`.
                                        unsafe { region.as_std_slice(remainder) },
                                    )
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    unsafe { *region.as_array::<MR>() }
                                };

                                // run the kernel
                                //
                                // SAFETY: By class invariant, `a_panel.k()` and `b_panels.k()`
                                // are both equal to `self.k`.
                                let mut kernel = unsafe {
                                    PanelKernel::new(self.arch, a_panel, b_panels, c, self.k)
                                };

                                driver::PanelKernel::panel_kernel(&mut kernel);

                                let c_final = kernel.take();

                                // Put back `C`.
                                if handling_tail {
                                    util::LoadStore::<i32, MR>::store(
                                        self.arch,
                                        c_final,
                                        // SAFETY: `region` has length exactly `remainder`.
                                        unsafe { region.as_std_mut_slice(remainder) },
                                    );
                                } else {
                                    // SAFETY: `region` has length exactly `MR`.
                                    unsafe { *region.as_array::<MR>() = c_final };
                                }
                            };

                        // SAFETY: By class invariant, `a_panels.k() == self.k`.
                        unsafe {
                            a_panels.visit_panels(self.k, panel_kernel);
                        }
                    };

                    // SAFETY: By class invariant, `self.b.k() == self.k`.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_panels);
                    }
                };

                // SAFETY: By class invariant, `self.a.k() == self.k`.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_panels)
                };
            },
        );
    }
}

//-------------//
// PanelKernel //
//-------------//

#[derive(Debug)]
pub(super) struct PanelKernel<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: packed::Panel<'a, i8, MR, PACK>,
    b: unpacked::View<'a, i8>,
    c: [i32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize, const PACK: usize> PanelKernel<'a, A, MR, NR, PACK> {
    /// Construct a new kernel.
    ///
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must both be equal to `k`.
    pub(super) unsafe fn new(
        arch: A,
        a: packed::Panel<'a, i8, MR, PACK>,
        b: unpacked::View<'a, i8>,
        c: [i32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { arch, a, b, c, k }
    }

    pub(super) fn take(self) -> [i32; MR] {
        self.c
    }
}

/// A custom visitor for the [`MicroKernel`].
///
/// This is needed to ensure the visitor body is inlined to inherit target features.
#[derive(Debug)]
struct Visitor<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: packed::Panel<'a, i8, MR, PACK>,
    c: &'a mut [i32; MR],
    k: DimK,
}

impl<A, const MR: usize, const NR: usize, const PACK: usize> unpacked::PanelVisitor<i8, NR>
    for Visitor<'_, A, MR, NR, PACK>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR, PACK>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, i8, NR>, _: usize) {
        // SAFETY: This is only used on contexts where `self.a.k()`, `b.k()`, and `self.k`
        // are all equal.
        let mut micro = unsafe { MicroKernel::new(self.arch, self.a, b, self.c, self.k) };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, $pack:literal, [ $($ns:literal),+ $(,)? ]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr, $pack> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                // NOTE: A `Visitor` is used here instead of a closure because a `Visitor`
                // is more reliably inlined, which means that target-features are inherited
                // more reliably.
                let on_b_panels = Visitor {
                    arch: self.arch,
                    a: self.a,
                    c: &mut self.c,
                    k: self.k,
                };

                // SAFETY: By class invariant, `self.k` is equal to `self.b.k()`.
                let b_tail = unsafe { self.b.visit_panels::<$nr>(self.k, on_b_panels) };

                if let Some(b_tail) = b_tail {
                    // Repetition Pattern.
                    $(
                        const { assert!($ns < $nr) };
                        if let Some(b_panel) = b_tail.try_as_panel::<$ns>() {
                            // SAFETY: By class invariant, `self.a.k()` and `self.b.k()`
                            // are equal to `self.k`.
                            let mut micro = unsafe {
                                MicroKernel::new(
                                    self.arch,
                                    self.a,
                                    b_panel,
                                    &mut self.c,
                                    self.k,
                                )
                            };

                            driver::MicroKernel::micro_kernel(&mut micro);
                        }
                    )+
                }
            }
        }
    }
}

panel_kernel!(Scalar, 8, 2, 2, [1]);

//--------------//
// Micro Kernel //
//--------------//

/// # Class Invariants
///
/// `a.k()` and `b.k()` are equal to `k`.
struct MicroKernel<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: packed::Panel<'a, i8, MR, PACK>,
    b: unpacked::Panel<'a, i8, NR>,
    c: &'a mut [i32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize, const PACK: usize> MicroKernel<'a, A, MR, NR, PACK> {
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must be equal to `k`.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, i8, MR, PACK>,
        b: unpacked::Panel<'a, i8, NR>,
        c: &'a mut [i32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { arch, a, b, c, k }
    }
}

/// Gather the `PACK` contraction indices starting at `ptr` for a single column of `b`,
/// zero filling the last group when `k` is not a multiple of `PACK`.
///
/// # Safety
///
/// `valid` must not exceed `PACK` and the first `valid` elements of `ptr` must be readable.
#[inline(always)]
unsafe fn group<const PACK: usize>(ptr: Slice<'_, i8>, valid: usize) -> [i8; PACK] {
    core::array::from_fn(|p| {
        if p < valid {
            // SAFETY: Since `p < valid`, the pointer offset is valid and readable.
            unsafe { *ptr.add(Elements::new(p)).as_unit().as_ref() }
        } else {
            0
        }
    })
}

/// # Safety
///
/// Bounds `a.k()` and `b.k()` must be equal to `k`.
#[inline(always)]
unsafe fn micro_kernel<W, const MR: usize, const NR: usize, const PACK: usize>(
    wide: W,
    a: packed::Panel<'_, i8, MR, PACK>,
    b: unpacked::Panel<'_, i8, NR>,
    c: &mut [i32; MR],
    k: DimK,
) where
    W: ExtraWide<MR, PACK>,
    Folder: Fold<NR>,
{
    // Check that everyone agrees.
    bounds::check_eq!(a.k(), k);
    bounds::check_eq!(b.k(), k);

    let ap = a.as_ptr();
    let bp = b.as_ptr();

    let mut acc = [wide.default(); NR];

    let astride = a.row_stride();
    let bstride = b.stride(k);

    let rows = a.rows(k);
    let k = k.value().get();

    for row in 0..rows {
        // SAFETY: By preconditions, `ap.len() == astride * rows`. Since `row < rows`:
        //
        // * The pointer offset is valid.
        // * The subsequent truncation is valid.
        // * The slice passed to `wide.load` has a length equal to `astride`.
        let ai = unsafe { wide.load(ap.add(astride * row).truncate(astride)) };

        // The trailing row of `a` is zero padded, so zero filling `b` past `k` keeps every
        // padded product at zero.
        let i = row * PACK;
        let valid = PACK.min(k - i);

        for (j, acc) in acc.iter_mut().enumerate() {
            // SAFETY: By preconditions, `bp.len() == bstride * NR`. Since `i < k`, `j < NR`
            // and `i + valid <= k`:
            //
            // * The pointer offset is valid and its first `valid` elements are readable.
            let bj =
                wide.splat(unsafe { group::<PACK>(bp.add(bstride * j + Elements::new(i)), valid) });

            *acc = W::dot(ai, bj, *acc);
        }
    }

    wide.max_into(Folder::fold(acc, W::max), c);
}

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal, $pack:literal) => {
        impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr, $pack> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                // SAFETY: By class invariant, `self.a.k()` and `self.b.k()` equal `self.k`.
                unsafe { micro_kernel(self.arch, self.a, self.b, self.c, self.k) }
            }
        }
    };
    ($arch:ty, $mr:literal, $pack:literal, { $($nr:literal),+ $(,)? }) => {
        $(micro_kernel!($arch, $mr, $nr, $pack);)+
    }
}

micro_kernel!(Scalar, 8, 2, { 2, 1 });

trait ExtraWide<const ELEMENTS: usize, const PACK: usize>: Copy {
    type Wide: Copy;
    type Splat: Copy;
    type Acc: Copy;

    /// # Safety
    ///
    /// `slice.len()` must be exactly `ELEMENTS * PACK`.
    unsafe fn load(self, slice: Slice<'_, i8>) -> Self::Wide;

    fn default(self) -> Self::Acc;
    fn splat(self, group: [i8; PACK]) -> Self::Splat;
    fn dot(a: Self::Wide, b: Self::Splat, acc: Self::Acc) -> Self::Acc;
    fn max(lhs: Self::Acc, rhs: Self::Acc) -> Self::Acc;
    fn max_into(self, max: Self::Acc, into: &mut [i32; ELEMENTS]);
}

impl ExtraWide<8, 2> for Scalar {
    type Wide = i16x16<Scalar>;
    type Splat = i16x16<Scalar>;
    type Acc = i32x8<Scalar>;

    #[inline(always)]
    fn default(self) -> Self::Acc {
        SIMDVector::default(self)
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, i8>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);

        // SAFETY: Since `slice.len()` must be 16, the 16-wide SIMD load is valid.
        let bytes: i8x16<Scalar> = unsafe { SIMDVector::load_simd(self, slice.as_ptr()) };

        Self::Wide::from(bytes)
    }

    #[inline(always)]
    fn splat(self, group: [i8; 2]) -> Self::Splat {
        u32x8::<Scalar>::splat(self, i16_pair(group)).reinterpret_simd()
    }

    #[inline(always)]
    fn dot(a: Self::Wide, b: Self::Splat, acc: Self::Acc) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    #[inline(always)]
    fn max(lhs: Self::Acc, rhs: Self::Acc) -> Self::Acc {
        lhs.max_simd(rhs)
    }

    #[inline(always)]
    fn max_into(self, lhs: Self::Acc, into: &mut [i32; 8]) {
        // SAFETY: Since `into.len()` is 8, the 8-wide SIMD load is valid.
        let previous: Self::Acc = unsafe { SIMDVector::load_simd(self, into.as_ptr()) };

        // SAFETY: Since `into.len()` is 8, the 8-wide SIMD store is valid.
        unsafe { Self::max(lhs, previous).store_simd(into.as_mut_ptr()) };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::arch::x86_64::V3;

    panel_kernel!(V3, 16, 6, 2, [1, 2, 3, 4, 5]);

    micro_kernel!(V3, 16, 2, { 6, 5, 4, 3, 2, 1 });

    //-----------//
    // ExtraWide //
    //-----------//

    impl ExtraWide<16, 2> for V3 {
        type Wide = [i16x16<V3>; 2];
        type Splat = i16x16<V3>;
        type Acc = [i32x8<V3>; 2];

        #[inline(always)]
        fn default(self) -> Self::Acc {
            [SIMDVector::default(self), SIMDVector::default(self)]
        }

        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, i8>) -> Self::Wide {
            bounds::check_eq!(slice.len(), 32);

            // SAFETY: Since `slice.len()` must be 32, the pointer offset and 16-wide SIMD loads
            // are valid.
            let bytes: [i8x16<V3>; 2] = unsafe {
                [
                    SIMDVector::load_simd(self, slice.as_ptr()),
                    SIMDVector::load_simd(self, slice.add(Elements::new(16)).as_ptr()),
                ]
            };

            bytes.map(Self::Splat::from)
        }

        #[inline(always)]
        fn splat(self, group: [i8; 2]) -> Self::Splat {
            u32x8::<V3>::splat(self, i16_pair(group)).reinterpret_simd()
        }

        #[inline(always)]
        fn dot(a: Self::Wide, b: Self::Splat, acc: Self::Acc) -> Self::Acc {
            core::array::from_fn(|i| acc[i].dot_simd(a[i], b))
        }

        #[inline(always)]
        fn max(lhs: Self::Acc, rhs: Self::Acc) -> Self::Acc {
            core::array::from_fn(|i| lhs[i].max_simd(rhs[i]))
        }

        #[inline(always)]
        fn max_into(self, lhs: Self::Acc, into: &mut [i32; 16]) {
            // SAFETY: Since `into.len()` is 16, the pointer offset and 8-wide SIMD loads are
            // valid.
            let previous: Self::Acc = unsafe {
                [
                    SIMDVector::load_simd(self, into.as_ptr()),
                    SIMDVector::load_simd(self, into.as_ptr().add(8)),
                ]
            };

            let max = Self::max(lhs, previous);

            // SAFETY: Since `into.len()` is 16, the pointer offset and 8-wide SIMD stores are
            // valid.
            unsafe {
                max[0].store_simd(into.as_mut_ptr());
                max[1].store_simd(into.as_mut_ptr().add(8));
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::num::NonZeroUsize;

    use rand::{SeedableRng, rngs::StdRng};

    #[cfg(target_arch = "x86_64")]
    use diskann_wide::arch::x86_64::V3;

    use crate::{matrix_kernels::maxsim, multi_vector::BlockTransposed};

    /////////////////
    // MicroKernel //
    /////////////////

    fn test_micro_kernel<A, const MR: usize, const NR: usize, const PACK: usize>(
        arch: A,
        k: DimK,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) where
        for<'a> MicroKernel<'a, A, MR, NR, PACK>: driver::MicroKernel,
    {
        let (ref_a, ref_b, ref_c) = maxsim::test::generate_i8(MR, k.value().get(), NR, rng);

        // From the reference problem, `ref_a` needs to be packed and `ref_b` transposed to
        // get them into the desired format.
        let a_bt = BlockTransposed::<i8, MR, PACK>::from_matrix_view(ref_a.as_view());
        let ref_b = ref_b.transpose();

        let mut c = [i32::MIN; MR];

        // Run the test kernel.
        //
        // SAFETY: Test builds will verify the bounds we passed.
        let mut kernel = unsafe {
            MicroKernel::new(
                arch,
                packed::Panel::new(Slice::new(a_bt.as_slice()), k),
                unpacked::Panel::new(Slice::new(ref_b.as_slice()), k),
                &mut c,
                k,
            )
        };

        driver::MicroKernel::micro_kernel(&mut kernel);
        assert_eq!(&*ref_c, kernel.c, "{ctx}");

        // Try again - but this time use a value that is much bigger than the what should
        // be generated by the test problem.
        //
        // This checks that we don't just overwrite existing contents.
        let new_c = kernel.c.map(|i| i + 1);
        *kernel.c = new_c;

        driver::MicroKernel::micro_kernel(&mut kernel);
        assert_eq!(new_c, *kernel.c, "{ctx}");
    }

    macro_rules! test_micro_kernel {
        (
            $fn:ident,
            $arch:expr,
            $seed:literal,
            $PACK:literal,
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
                                test_micro_kernel::<_, $MR, $NR, $PACK>(
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
        0x4b1d09c2a77e5310,
        2,
        8 => { 2, 1 },
    );

    #[cfg(target_arch = "x86_64")]
    test_micro_kernel!(
        test_micro_kernel_v3,
        V3::new_checked(),
        0xe0a5c31f8b62d94a,
        2,
        16 => { 6, 5, 4, 3, 2, 1 },
    );

    /////////////////
    // PanelKernel //
    /////////////////

    // The panel kernel operates on a single A-panel with multiple B-panels.
    //
    // This test sweeps over a number of rows for the B-panels to exercise all possible
    // corner cases.
    fn test_panel_kernel<A, const MR: usize, const NR: usize, const PACK: usize>(
        arch: A,
        k: DimK,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) where
        A: Copy,
        for<'a> PanelKernel<'a, A, MR, NR, PACK>: driver::PanelKernel,
    {
        for blocks in 0..4 {
            for remainder in 0..NR {
                let cols = NR * blocks + remainder;
                if cols == 0 {
                    continue;
                }

                let (ref_a, ref_b, ref_c) =
                    maxsim::test::generate_i8(MR, k.value().get(), cols, rng);

                let a_bt = BlockTransposed::<i8, MR, PACK>::from_matrix_view(ref_a.as_view());
                let ref_b = ref_b.transpose();

                let extent = NonZeroUsize::new(cols).unwrap();

                let c = [i32::MIN; MR];

                // SAFETY: Test builds will verify the bounds we passed.
                let mut kernel = unsafe {
                    PanelKernel::new(
                        arch,
                        packed::Panel::new(Slice::new(a_bt.as_slice()), k),
                        unpacked::View::new(Slice::new(ref_b.as_slice()), extent, k),
                        c,
                        k,
                    )
                };

                driver::PanelKernel::panel_kernel(&mut kernel);
                assert_eq!(&*ref_c, kernel.c, "{ctx}");

                // Try again - but this time use a value that is much bigger than the what
                // should be generated by the test problem.
                //
                // This checks that we don't just overwrite existing contents.
                let new_c = kernel.c.map(|i| i + 1);
                kernel.c = new_c;

                driver::PanelKernel::panel_kernel(&mut kernel);
                assert_eq!(new_c, kernel.c, "{ctx}");
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
                    $MR:literal, $NR:literal, $PACK:literal
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
                            test_panel_kernel::<_, $MR, $NR, $PACK>(
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
        0x9f3e7ab4c05d1268,
        (8, 2, 2),
    );

    #[cfg(target_arch = "x86_64")]
    test_panel_kernel!(
        test_panel_kernel_v3,
        V3::new_checked(),
        0x9f3e7ab4c05d1268,
        (16, 6, 2),
    );

    ////////////
    // Driver //
    ////////////

    fn test_driver<A, const MR: usize, const NR: usize, const PACK: usize>(
        arch: A,
        rng: &mut impl rand::Rng,
    ) where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR, PACK>: driver::Drive,
    {
        let cases = maxsim::test::packed_x_unpacked_test_dims(MR, NR);
        for case in cases {
            let maxsim::test::TestDims {
                a_panels_per_tile,
                total_a_rows,
                b_cols_per_tile,
                total_b_cols,
                k,
            } = case.clone();

            let k = DimK::new(NonZeroUsize::new(k).unwrap());

            let (ref_a, ref_b, ref_c) =
                maxsim::test::generate_i8(total_a_rows, k.value().get(), total_b_cols, rng);

            // Massage the input data in the form needed by the kernel.
            let a_bt = BlockTransposed::<i8, MR, PACK>::from_matrix_view(ref_a.as_view());
            let b = ref_b.transpose();

            let mut c = vec![i32::MAX; a_bt.nrows()];

            // SAFETY: Test builds will verify the bounds we passed.
            let mut driver = unsafe {
                Driver::new_inner(
                    arch,
                    packed::View::from_block_transposed(a_bt.as_view()).unwrap(),
                    unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                    &mut c,
                    k,
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(a_panels_per_tile).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(b_cols_per_tile).unwrap(),
                    },
                )
            };

            driver::Drive::drive(&mut driver);

            assert_eq!(ref_c, c, "setup: {:?}", case)
        }
    }

    macro_rules! test_driver {
        (
            $fn:ident,
            $arch:expr,
            $seed:literal,
            $(
                (
                    $MR:literal, $NR:literal, $PACK:literal
                )
            ),+ $(,)?
        ) => {
            #[test]
            fn $fn() {
                if let Some(arch) = $arch {
                    let mut rng = StdRng::seed_from_u64($seed);

                    $(test_driver::<_, $MR, $NR, $PACK>(arch, &mut rng);)+
                }
            }
        }
    }

    test_driver!(
        test_driver_scalar,
        Some(Scalar::new()),
        0x63c8ed19f4720ab5,
        (8, 2, 2),
    );

    #[cfg(target_arch = "x86_64")]
    test_driver!(
        test_driver_v3,
        V3::new_checked(),
        0x63c8ed19f4720ab5,
        (16, 6, 2),
    );
}
