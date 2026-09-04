// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Matrix-packed `u8` by matrix-unpacked `u8` MaxSim kernel.

use diskann_wide::Architecture;
use diskann_wide::arch::Scalar;

use crate::matrix_kernels::{
    Cache,
    blocks::{packed, unpacked},
    bounds, driver,
    maxsim::packed_f32_x_unpacked_f32::Params,
    num::DimK,
    ptr::MutSlice,
    util,
};
use crate::minmax::MinMaxCompensation;

pub(super) struct Compensations<'a> {
    pub(super) a: &'a [MinMaxCompensation],
    pub(super) b: &'a [MinMaxCompensation],
}

pub(super) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    a_meta: &'a [MinMaxCompensation],
    b_meta: &'a [MinMaxCompensation],
    c: &'a mut [f32],
    k: DimK,
    params: Params,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a`, `b`, and the metadata slices must describe the same logical matrices.
    pub(super) unsafe fn new(
        arch: A,
        a: packed::View<'a, u8, MR>,
        b: unpacked::View<'a, u8>,
        compensation: Compensations<'a>,
        c: &'a mut [f32],
        k: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_eq!(bounds::Bound::new(a.blocks().get()), c.len().div_ceil(MR));
        bounds::check_eq!(bounds::Bound::new(compensation.a.len()), c.len());
        bounds::check_eq!(bounds::Bound::new(compensation.b.len()), b.extent().get());

        let params = Params::new(
            Cache::detect(),
            a.block_stride(k).bytes(),
            b.stride(k).bytes(),
            NR,
        );
        Self {
            arch,
            a,
            b,
            a_meta: compensation.a,
            b_meta: compensation.b,
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
                self.c.fill(f32::MAX);
                let remainder = self.c.len() % MR;
                let last_a_block = self.a.blocks().get() - 1;
                let mut c = MutSlice::new(self.c);

                let on_a_subview = |a: packed::View<'_, u8, MR>, a_block_base| {
                    let on_b_subview = |b: unpacked::View<'_, u8>, b_base| {
                        let b_meta = &self.b_meta[b_base..][..b.extent().get()];

                        let on_a_panel = |a_panel: packed::Panel<'_, u8, MR>, a_block_offset| {
                            let a_block = a_block_base + a_block_offset;
                            let a_base = a_block * MR;
                            let handling_tail = a_block == last_a_block && remainder != 0;
                            let valid_rows = if handling_tail { remainder } else { MR };
                            let bound = bounds::Bound::new(valid_rows);
                            // SAFETY: Each A block maps to one output region; only the final
                            // region may contain fewer than `MR` values.
                            let mut region = unsafe { c.subslice(a_base, bound) };
                            let c_panel = if handling_tail {
                                // SAFETY: A tail region contains exactly `remainder` values.
                                util::LoadStore::<f32, MR>::load(self.arch, unsafe {
                                    region.as_std_slice(remainder)
                                })
                            } else {
                                // SAFETY: A complete region contains exactly `MR` values.
                                unsafe { *region.as_array::<MR>() }
                            };
                            let mut kernel = PanelKernel {
                                arch: self.arch,
                                a: a_panel,
                                b,
                                a_meta: &self.a_meta[a_base..][..valid_rows],
                                b_meta,
                                c: c_panel,
                                k: self.k,
                            };
                            driver::PanelKernel::panel_kernel(&mut kernel);
                            if handling_tail {
                                // SAFETY: A tail region contains exactly `remainder` values.
                                util::LoadStore::<f32, MR>::store(self.arch, kernel.c, unsafe {
                                    region.as_std_mut_slice(remainder)
                                });
                            } else {
                                // SAFETY: A complete region contains exactly `MR` values.
                                unsafe { *region.as_array::<MR>() = kernel.c };
                            }
                        };

                        // SAFETY: `a` retains the driver's contraction dimension.
                        unsafe { a.visit_panels(self.k, on_a_panel) };
                    };

                    // SAFETY: `self.b` retains the driver's contraction dimension.
                    unsafe {
                        self.b
                            .visit_sub_views(self.params.b_cols_in_l1, self.k, on_b_subview)
                    };
                };

                // SAFETY: `self.a` retains the driver's contraction dimension.
                unsafe {
                    self.a
                        .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a_subview)
                };
            },
        );
    }
}

pub(super) struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::View<'a, u8>,
    a_meta: &'a [MinMaxCompensation],
    b_meta: &'a [MinMaxCompensation],
    c: [f32; MR],
    k: DimK,
}

struct Visitor<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    a_meta: &'a [MinMaxCompensation],
    b_meta: &'a [MinMaxCompensation],
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<u8, NR> for Visitor<'_, A, MR, NR>
where
    A: Copy,
    for<'a> MicroKernel<'a, A, MR, NR>: driver::MicroKernel,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, u8, NR>, start: usize) {
        let mut micro = MicroKernel {
            arch: self.arch,
            a: self.a,
            b,
            a_meta: self.a_meta,
            b_meta: &self.b_meta[start..][..NR],
            c: self.c,
            k: self.k,
        };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [$($tail:literal),+ $(,)?]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let visitor = Visitor {
                    arch: self.arch,
                    a: self.a,
                    a_meta: self.a_meta,
                    b_meta: self.b_meta,
                    c: &mut self.c,
                    k: self.k,
                };
                // SAFETY: `self.k` is the contraction dimension tracked by `self.b`.
                let tail = unsafe { self.b.visit_panels::<$nr>(self.k, visitor) };

                if let Some(tail) = tail {
                    $(
                        if let Some(b) = tail.try_as_panel::<$tail>() {
                            let start = self.b.extent().get() - $tail;
                            let mut micro = MicroKernel {
                                arch: self.arch,
                                a: self.a,
                                b,
                                a_meta: self.a_meta,
                                b_meta: &self.b_meta[start..],
                                c: &mut self.c,
                                k: self.k,
                            };
                            driver::MicroKernel::micro_kernel(&mut micro);
                        }
                    )+
                }
            }
        }
    };
}

panel_kernel!(Scalar, 8, 6, [1, 2, 3, 4, 5]);

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR>,
    b: unpacked::Panel<'a, u8, NR>,
    a_meta: &'a [MinMaxCompensation],
    b_meta: &'a [MinMaxCompensation],
    c: &'a mut [f32; MR],
    k: DimK,
}

#[inline(always)]
fn micro_kernel<W, const MR: usize, const NR: usize>(
    wide: W,
    a: packed::Panel<'_, u8, MR>,
    b: unpacked::Panel<'_, u8, NR>,
    a_meta: &[MinMaxCompensation],
    b_meta: &[MinMaxCompensation],
    c: &mut [f32; MR],
    k: DimK,
) where
    W: ExtraWide<MR>,
{
    let accum = wide.multiply(a, b, k);

    for (i, query) in a_meta.iter().enumerate() {
        for (j, doc) in b_meta.iter().enumerate() {
            let similarity = compensated_product(accum[j][i], k, query, doc);
            c[i] = c[i].min(-similarity);
        }
    }
}

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn micro_kernel(&mut self) {
                micro_kernel(
                    self.arch,
                    self.a,
                    self.b,
                    self.a_meta,
                    self.b_meta,
                    self.c,
                    self.k,
                )
            }
        }
    };
    ($arch:ty, $mr:literal, { $($nr:literal),+ $(,)? }) => {
        $(micro_kernel!($arch, $mr, $nr);)+
    };
}

micro_kernel!(Scalar, 8, { 6, 5, 4, 3, 2, 1 });

trait ExtraWide<const MR: usize>: Copy {
    fn multiply<const NR: usize>(
        self,
        a: packed::Panel<'_, u8, MR>,
        b: unpacked::Panel<'_, u8, NR>,
        k: DimK,
    ) -> [[u32; MR]; NR];
}

impl<const MR: usize> ExtraWide<MR> for Scalar {
    #[inline(always)]
    fn multiply<const NR: usize>(
        self,
        a: packed::Panel<'_, u8, MR>,
        b: unpacked::Panel<'_, u8, NR>,
        k: DimK,
    ) -> [[u32; MR]; NR] {
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let bstride = b.stride(k);
        let mut accum = [[0; MR]; NR];

        for kk in 0..k.value().get() {
            for (j, row) in accum.iter_mut().enumerate() {
                for (i, value) in row.iter_mut().enumerate() {
                    // SAFETY: The packed A panel contains `k * MR` values.
                    let q = unsafe { *ap.as_ptr().add(kk * MR + i) };
                    // SAFETY: The B panel contains `NR` rows of `k` values.
                    let d = unsafe { *bp.as_ptr().add(j * bstride.value() + kk) };
                    *value += q as u32 * d as u32;
                }
            }
        }
        accum
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::arch::x86_64::{V3, V4};

    panel_kernel!(V3, 16, 6, [1, 2, 3, 4, 5]);
    panel_kernel!(V4, 16, 6, [1, 2, 3, 4, 5]);

    micro_kernel!(V3, 16, { 6, 5, 4, 3, 2, 1 });
    micro_kernel!(V4, 16, { 6, 5, 4, 3, 2, 1 });

    impl ExtraWide<16> for V3 {
        #[inline(always)]
        fn multiply<const NR: usize>(
            self,
            a: packed::Panel<'_, u8, 16>,
            b: unpacked::Panel<'_, u8, NR>,
            k: DimK,
        ) -> [[u32; 16]; NR] {
            use std::arch::x86_64::{
                __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepu8_epi32,
                _mm256_mullo_epi32, _mm256_set1_epi32, _mm256_setzero_si256, _mm256_storeu_si256,
            };

            let ap = a.as_ptr().as_ptr();
            let bp = b.as_ptr().as_ptr();
            let bstride = b.stride(k).value();
            let mut output = [[0_u32; 16]; NR];

            // SAFETY: V3 proves AVX2 availability, and panel invariants bound all accesses.
            unsafe {
                let zero: __m256i = _mm256_setzero_si256();
                let mut lo = [zero; NR];
                let mut hi = [zero; NR];
                for kk in 0..k.value().get() {
                    let q = ap.add(kk * 16);
                    let qlo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(q.cast()));
                    let qhi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(q.add(8).cast()));
                    for j in 0..NR {
                        let d = _mm256_set1_epi32(*bp.add(j * bstride + kk) as i32);
                        lo[j] = _mm256_add_epi32(lo[j], _mm256_mullo_epi32(qlo, d));
                        hi[j] = _mm256_add_epi32(hi[j], _mm256_mullo_epi32(qhi, d));
                    }
                }
                for j in 0..NR {
                    _mm256_storeu_si256(output[j].as_mut_ptr().cast(), lo[j]);
                    _mm256_storeu_si256(output[j].as_mut_ptr().add(8).cast(), hi[j]);
                }
            }
            output
        }
    }

    impl ExtraWide<16> for V4 {
        #[inline(always)]
        fn multiply<const NR: usize>(
            self,
            a: packed::Panel<'_, u8, 16>,
            b: unpacked::Panel<'_, u8, NR>,
            k: DimK,
        ) -> [[u32; 16]; NR] {
            use std::arch::x86_64::{
                __m512i, _mm_loadu_si128, _mm512_add_epi32, _mm512_cvtepu8_epi32,
                _mm512_mullo_epi32, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
            };

            let ap = a.as_ptr().as_ptr();
            let bp = b.as_ptr().as_ptr();
            let bstride = b.stride(k).value();
            let mut output = [[0_u32; 16]; NR];

            // SAFETY: V4 proves AVX-512F/BW availability, and panel invariants bound accesses.
            unsafe {
                let zero: __m512i = _mm512_setzero_si512();
                let mut accum = [zero; NR];
                for kk in 0..k.value().get() {
                    let q = _mm512_cvtepu8_epi32(_mm_loadu_si128(ap.add(kk * 16).cast()));
                    for j in 0..NR {
                        let d = _mm512_set1_epi32(*bp.add(j * bstride + kk) as i32);
                        accum[j] = _mm512_add_epi32(accum[j], _mm512_mullo_epi32(q, d));
                    }
                }
                for j in 0..NR {
                    _mm512_storeu_si512(output[j].as_mut_ptr().cast(), accum[j]);
                }
            }
            output
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    use diskann_wide::arch::aarch64::Neon;

    panel_kernel!(Neon, 8, 8, [1, 2, 3, 4, 5, 6, 7]);
    micro_kernel!(Neon, 8, { 8, 7, 6, 5, 4, 3, 2, 1 });

    impl ExtraWide<8> for Neon {
        #[inline(always)]
        fn multiply<const NR: usize>(
            self,
            a: packed::Panel<'_, u8, 8>,
            b: unpacked::Panel<'_, u8, NR>,
            k: DimK,
        ) -> [[u32; 8]; NR] {
            use std::arch::aarch64::{
                uint32x4_t, vdupq_n_u32, vget_high_u16, vget_low_u16, vld1_u8, vmlal_n_u16,
                vmovl_u8, vst1q_u32,
            };

            let ap = a.as_ptr().as_ptr();
            let bp = b.as_ptr().as_ptr();
            let bstride = b.stride(k).value();
            let mut output = [[0_u32; 8]; NR];

            // SAFETY: Neon proves instruction availability, and panel invariants bound accesses.
            unsafe {
                let zero: uint32x4_t = vdupq_n_u32(0);
                let mut lo = [zero; NR];
                let mut hi = [zero; NR];
                for kk in 0..k.value().get() {
                    let q = vmovl_u8(vld1_u8(ap.add(kk * 8)));
                    let qlo = vget_low_u16(q);
                    let qhi = vget_high_u16(q);
                    for j in 0..NR {
                        let d = *bp.add(j * bstride + kk) as u16;
                        lo[j] = vmlal_n_u16(lo[j], qlo, d);
                        hi[j] = vmlal_n_u16(hi[j], qhi, d);
                    }
                }
                for j in 0..NR {
                    vst1q_u32(output[j].as_mut_ptr(), lo[j]);
                    vst1q_u32(output[j].as_mut_ptr().add(4), hi[j]);
                }
            }
            output
        }
    }
}

#[inline(always)]
fn compensated_product(
    raw: u32,
    k: DimK,
    query: &MinMaxCompensation,
    doc: &MinMaxCompensation,
) -> f32 {
    query.a * doc.a * raw as f32
        + query.n * doc.b
        + doc.n * query.b
        + query.b * doc.b * k.value().get() as f32
}
