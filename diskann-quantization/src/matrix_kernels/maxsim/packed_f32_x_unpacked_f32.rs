/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The lowering of operations mimics a GEMM style operation with inplace application of the
//! max-sim reduction operation. Currently, blocking across the contraction dimension "k"
//! is not implemented. As such, expect a performance penalty for large-dimensional vectors.
//!
//! The kernel is implemented as follows:
//!
//! * Partition `a` into sub-views `suba` that roughly occupy the L2 cache.
//! * Partition `b` into sub-views `subb` that roughly occupy a portion of the L1 cache.
//! * Partition `suba` into panels `pa`. We want `pa + subb` to fit in L1.
//! * Perform micro-kernel operations on `pa + subb`. This computes the max-sim in-place.
//!
//! There is plenty of room for improvement. This is just a starting point.

use std::num::NonZeroUsize;

use diskann_wide::{
    SIMDMinMax, SIMDMulAdd, SIMDVector,
    arch::{Architecture, Scalar},
};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    num::{Bytes, DimK, Elements, value_or_one},
    ptr::{MutSlice, Slice},
    util::{self, Fold, Folder},
};

/// Blocking parameters for the `packed x unpacked` kernel.
#[derive(Debug, Clone, Copy)]
pub(super) struct Params {
    /// The (approximate) number of blocks of `A` that fit in the L2 cache.
    pub(super) a_panels_in_l2: NonZeroUsize,
    /// The (approximate) number of columns of `B` that fis in the L1 cache.
    pub(super) b_cols_in_l1: NonZeroUsize,
}

impl Params {
    /// Select hyper-parameters for the `packed x unpacked` kernel based on cache size.
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

//--------//
// Driver //
//--------//

/// A driver for prepacked by unpacked "maxsim" computations.
///
/// Results are returned directly in `c`.
///
/// Note that class invariant (2) allows the physical length of the output to be less than
/// the packed extent of `a` as long as it resides within the last physical block of `a`.
///
/// This allows the kernel to write directly into an output buffer when `a` is not logically
/// filled. `a` being *physically* filled is still a requirement.
///
/// # Class Invariants
///
/// 1. `a.k()` and `b.k()` must be equal to `k`.
/// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: &'a mut [f32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// Prepare for a maxsim on `a` and `b` with the results stored directly into `c`.
    ///
    /// `c` does not any specific initial value.
    ///
    /// # Safety
    ///
    /// 1. `a.k()` and `b.k()` must be equal to `k`.
    /// 2. `c.len().div_ceil(MR)` must be equal to `a.blocks()`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: &'a mut [f32],
        k: DimK,
        cache: Cache,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupiy exactly the packed A blocks",
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
        a: packed::View<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: &'a mut [f32],
        k: DimK,
        params: Params,
    ) -> Self {
        bounds::check_eq!(a.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(b.k(), k, "constraction dimensions to not agree");
        bounds::check_eq!(
            bounds::Bound::new(a.blocks().get()),
            c.len().div_ceil(MR),
            "output length must occupiy exactly the packed A blocks",
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

impl<A, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR>
where
    A: util::LoadStore<f32, MR> + Architecture,
    for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                // Pre-fill `c`.
                self.c.fill(f32::NEG_INFINITY);

                // We allow `c` to be slightly under-filled.
                //
                // These variables track if under-fill is happening.
                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;

                let mut c = MutSlice::new(self.c);

                let on_a_panels = |a_panels: packed::View<'_, f32, MR>, a_block_base| {
                    let on_b_panels = |b_panels: unpacked::View<'_, f32>, _| {
                        let panel_kernel = |a_panel: packed::Panel<'_, f32, MR>, a_block_offset| {
                            // If we are in the very last block and we need to sub-fill, do that.
                            // Otherwise, reference the output in place.
                            let a_block = a_block_base + a_block_offset;
                            let handling_tail = a_block == last_a_block && remainder != 0;

                            let bound =
                                bounds::Bound::from_fn(
                                    || if handling_tail { remainder } else { MR },
                                );

                            // SAFETY: By class invariant,
                            //
                            // `MR * (self.a.blocks() - 1) < c.len() <= MR * self.a.blocks()`.
                            //
                            // From the visitor, `a_block <= self.a.blocks()`.
                            let mut region = unsafe { c.subslice(MR * a_block, bound) };
                            let c = if handling_tail {
                                util::LoadStore::<f32, MR>::load(
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
                                util::LoadStore::<f32, MR>::store(
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
pub(super) struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, f32, MR>,
    b: unpacked::View<'a, f32>,
    c: [f32; MR],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// Construct a new kernel.
    ///
    /// # Safety
    ///
    /// Bounds `a.k()` and `b.k()` must both be equal to `k`.
    pub(super) unsafe fn new(
        arch: A,
        a: packed::Panel<'a, f32, MR>,
        b: unpacked::View<'a, f32>,
        c: [f32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { arch, a, b, c, k }
    }

    pub(super) fn take(self) -> [f32; MR] {
        self.c
    }
}

/// A custom visitor for the [`MicroKernel`].
///
/// This is needed to ensure the visitor body is inlined to inherit target features.
#[derive(Debug)]
struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, f32, MR>,
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<f32, NR> for Visitor<'_, A, MR, NR>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, f32, NR>, _: usize) {
        // SAFETY: This is only used on contexts where `self.a.k()`, `b.k()`, and `self.k`
        // are all equal.
        let mut micro = unsafe { MicroKernel::new(self.arch, self.a, b, self.c, self.k) };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr: literal, [ $($ns:literal),+ $(,)? ]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
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
                    // Repeitition Pattern.
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

panel_kernel!(Scalar, 8, 2, [1]);

panel_kernel!(V3, 16, 6, [1, 2, 3, 4, 5]);

panel_kernel!(V4, 16, 6, [1, 2, 3, 4, 5]);
panel_kernel!(V4, 32, 6, [1, 2, 3, 4, 5]);

//--------------//
// Micro Kernel //
//--------------//

/// # Class Invariants
///
/// `a.k()` and `b.k()` are equal to `k`.
struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
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
        arch: A,
        a: packed::Panel<'a, f32, MR>,
        b: unpacked::Panel<'a, f32, NR>,
        c: &'a mut [f32; MR],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);

        Self { arch, a, b, c, k }
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

    let astride = Elements::<f32>::new(MR);
    let bstride = b.stride(k);

    for i in 0..k.value().get() {
        // SAFETY: By preconditions, `ap.len() == astride * k`. Since `i < k` and `astride == MR`:
        //
        // * The pointer offset is valid.
        // * The subsequent truncation is valid.
        // * The slice passed to `wide.load` has a length equal to `astride` (and hence `MR`).
        let ai = unsafe { wide.load(ap.add(astride * i).truncate(astride)) };

        for (j, acc) in acc.iter_mut().enumerate() {
            // SAFETY: By precionditions, `bp.len() == bstride * NR`. Since `i < k` and `j < NR`:
            //
            // * The pointer offset is valid and readable.
            let bj =
                wide.splat(*unsafe { bp.add(bstride * j + Elements::new(i)).as_unit().as_ref() });

            *acc = W::mul_add_splat(ai, bj, *acc);
        }
    }

    wide.max_into(Folder::fold(acc, W::max), c);
}

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(f32x16<A> = f32x16);

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                // SAFETY: By class invariant, `self.a.k()` and `self.b.k()` equal `self.k`.
                unsafe { micro_kernel(self.arch, self.a, self.b, self.c, self.k) }
            }
        }
    };
    ($arch:ty, $mr:literal, { $($nr:literal),+ $(,)? }) => {
        $(micro_kernel!($arch, $mr, $nr);)+
    }
}

micro_kernel!(Scalar, 8, { 2, 1 });
micro_kernel!(V3, 16, { 6, 5, 4, 3, 2, 1 });
micro_kernel!(V4, 16, { 6, 5, 4, 3, 2, 1 });
micro_kernel!(V4, 32, { 6, 5, 4, 3, 2, 1 });

trait ExtraWide<const ELEMENTS: usize>: Copy {
    type Wide: Copy;
    type Splat: Copy;

    /// # Safety
    ///
    /// `slice.len()` must be exactly `ELEMENTS`.
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide;

    fn default(self) -> Self::Wide;
    fn splat(self, value: f32) -> Self::Splat;
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide;
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide;
    fn max_into(self, max: Self::Wide, into: &mut [f32; ELEMENTS]);
}

impl ExtraWide<8> for Scalar {
    type Wide = [f32x4<Scalar>; 2];
    type Splat = f32x4<Scalar>;

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [SIMDVector::default(self), SIMDVector::default(self)]
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 8);

        // SAFETY: Since `slice.len()` must be 8, the pointer offset and 4-wide SIMD loads
        // are valid.
        unsafe {
            [
                SIMDVector::load_simd(self, slice.as_ptr()),
                SIMDVector::load_simd(self, slice.add(Elements::new(4)).as_ptr()),
            ]
        }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self, value)
    }

    #[inline(always)]
    fn mul_add_splat(a: Self::Wide, b: Self::Splat, acc: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| (a[i] * b) + acc[i])
    }

    #[inline(always)]
    fn max(lhs: Self::Wide, rhs: Self::Wide) -> Self::Wide {
        core::array::from_fn(|i| lhs[i].max_simd(rhs[i]))
    }

    #[inline(always)]
    fn max_into(self, lhs: Self::Wide, into: &mut [f32; 8]) {
        // SAFETY: `into` has a length of exactly 8.
        let previous = unsafe { self.load(Slice::new(into)) };
        let max = Self::max(lhs, previous);

        // SAFETY: Since `into.len()` is 8, the pointer offset and 4-wide SIMD stores are valid.
        unsafe {
            max[0].store_simd(into.as_mut_ptr());
            max[1].store_simd(into.as_mut_ptr().add(4));
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<16> for V3 {
    type Wide = [f32x8<V3>; 2];
    type Splat = f32x8<V3>;

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [SIMDVector::default(self), SIMDVector::default(self)]
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);

        // SAFETY: Since `slice.len()` must be 16, the pointer offset and 8-wide SIMD loads
        // are valid.
        unsafe {
            [
                SIMDVector::load_simd(self, slice.as_ptr()),
                SIMDVector::load_simd(self, slice.add(Elements::new(8)).as_ptr()),
            ]
        }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self, value)
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
        // SAFETY: `into` has a length of exactly 16.
        let previous = unsafe { self.load(Slice::new(into)) };
        let max = Self::max(lhs, previous);

        // SAFETY: Since `into.len()` is 16, the pointer offset and 8-wide SIMD stores are valid.
        unsafe {
            max[0].store_simd(into.as_mut_ptr());
            max[1].store_simd(into.as_mut_ptr().add(8));
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<16> for V4 {
    type Wide = f32x16<V4>;
    type Splat = f32x16<V4>;

    #[inline(always)]
    fn default(self) -> Self::Wide {
        SIMDVector::default(self)
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 16);

        // SAFETY: Since `slice.len()` must be 16, the 16-wide SIMD load is safe.
        unsafe { SIMDVector::load_simd(self, slice.as_ptr()) }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self, value)
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
        // SAFETY: `into` has a length of exactly 16.
        let previous = unsafe { ExtraWide::<16>::load(self, Slice::new(into)) };
        let max = <Self as ExtraWide<16>>::max(lhs, previous);

        // SAFETY: Since `into.len()` is 16, the store is valid.
        unsafe {
            max.store_simd(into.as_mut_ptr());
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl ExtraWide<32> for V4 {
    type Wide = [f32x16<V4>; 2];
    type Splat = f32x16<V4>;

    #[inline(always)]
    fn default(self) -> Self::Wide {
        [SIMDVector::default(self), SIMDVector::default(self)]
    }

    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, f32>) -> Self::Wide {
        bounds::check_eq!(slice.len(), 32);

        // SAFETY: Since `slice.len()` must be 32, the pointer offset and 16-wide SIMD loads
        // are valid.
        unsafe {
            [
                SIMDVector::load_simd(self, slice.as_ptr()),
                SIMDVector::load_simd(self, slice.add(Elements::new(16)).as_ptr()),
            ]
        }
    }

    #[inline(always)]
    fn splat(self, value: f32) -> Self::Splat {
        SIMDVector::splat(self, value)
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
    fn max_into(self, lhs: Self::Wide, into: &mut [f32; 32]) {
        // SAFETY: `into` has a length of exactly 32.
        let previous = unsafe { ExtraWide::<32>::load(self, Slice::new(into)) };
        let max = <Self as ExtraWide<32>>::max(lhs, previous);

        // SAFETY: Since `into.len()` is 32, the pointer offset and 16-wide SIMD stores are
        // valid.
        unsafe {
            max[0].store_simd(into.as_mut_ptr());
            max[1].store_simd(into.as_mut_ptr().add(16));
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

    use crate::{matrix_kernels::maxsim, multi_vector::BlockTransposed};

    /////////////////
    // MicroKernel //
    /////////////////

    fn test_micro_kernel<A, const MR: usize, const NR: usize>(
        arch: A,
        k: DimK,
        rng: &mut impl rand::Rng,
        ctx: std::fmt::Arguments<'_>,
    ) where
        for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
    {
        let (ref_a, ref_b, ref_c) = maxsim::test::generate(MR, k.value().get(), NR, rng);

        // From the reference problem, we need to transpose both `ref_a` and `ref_b` to get
        // them into the desired format.
        let ref_a = ref_a.transpose();
        let ref_b = ref_b.transpose();

        let mut c = [f32::NEG_INFINITY; MR];

        // Run the test kernel.
        //
        // SAFETY: Test builds will verify the bounds we passed.
        let mut kernel = unsafe {
            MicroKernel::new(
                arch,
                packed::Panel::new(Slice::new(ref_a.as_slice()), k),
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
        // This checks that we don't just overwite existing contents.
        let new_c = kernel.c.map(|i| i + 1.0);
        *kernel.c = new_c;

        driver::MicroKernel::micro_kernel(&mut kernel);
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
        16 => { 6, 5, 4, 3, 2, 1},
        32 => { 6, 5, 4, 3, 2, 1},
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
        for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
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

                let c = [f32::NEG_INFINITY; MR];

                // SAFETY: Test builds will verify the bounds we passed.
                let mut kernel = unsafe {
                    PanelKernel::new(
                        arch,
                        packed::Panel::new(Slice::new(ref_a.as_slice()), k),
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
                // This checks that we don't just overwite existing contents.
                let new_c = kernel.c.map(|i| i + 1.0);
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
        (16, 6),
    );

    test_panel_kernel!(
        test_panel_kernel_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 6),
        (32, 6),
    );

    ////////////
    // Driver //
    ////////////

    fn test_driver<A, const MR: usize, const NR: usize>(arch: A, rng: &mut impl rand::Rng)
    where
        A: Copy,
        for<'a> Driver<'a, A, MR, NR>: driver::Drive,
    {
        // (a-panels-per-tile, a-rows, b-cols-per-tile, b-cols, k)
        let cases = [
            (1, 1, 1, 1, 1),                        // Smallest logical output
            (1, MR / 2, 1, 1, 1),                   // Partial first panel
            (1, MR - 1, 2 * NR, NR, 3),             // Nearly full first panel
            (2, MR + 1, 2 * NR, NR, 3),             // Partial second panel
            (2, 2 * MR - 1, 2 * NR, 2 * NR + 1, 5), // Partial panel and split B
            (2, 2 * MR + 1, 2 * NR, 2 * NR + 1, 5), // Split A and B with a partial panel
            (1, MR * 3, 1, 3, 1),                   // Unit advancement, no reuse.
            (2, MR * 2, 2 * NR, 2 * NR, 3),         // Values a direct multiple of the blocking.
            (2, MR * 3, 2 * NR, NR, 3),
            (2, MR, 2 * NR, 2 * NR + 1, 3),
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

            let mut c = vec![f32::NAN; a_bt.nrows()];

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
        (16, 6),
    );

    test_driver!(
        test_driver_v4,
        V4::new_checked_miri(),
        0x2c03eb9ee51d30c3,
        (16, 6),
        (32, 6),
    );
}
