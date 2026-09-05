// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MinMax8 A by MinMax4 B MaxSim kernel.

use diskann_wide::arch::Scalar;
use diskann_wide::{Architecture, SIMDMinMax, SIMDVector};

use crate::matrix_kernels::{
    bounds::Bound,
    driver,
    num::{DimK, Elements},
    ptr::Slice,
    util,
};
use crate::minmax::{MinMaxCompensation, MinMaxMeta};
use crate::multi_vector::MatRef;

#[derive(Debug, Clone, Copy)]
pub(crate) enum APacking {
    RowMajor,
    Grouped4,
    #[cfg(target_arch = "x86_64")]
    Grouped8,
}

#[derive(Debug)]
pub(crate) struct PackedMinMax8<const MR: usize> {
    nrows: usize,
    dim: usize,
    block_stride: usize,
    values: Vec<u8>,
    scale: Vec<f32>,
    bias: Vec<f32>,
    scaled_sum: Vec<f32>,
}

impl<const MR: usize> PackedMinMax8<MR> {
    pub(crate) fn new(a: MatRef<'_, MinMaxMeta<8>>, packing: APacking) -> Self {
        let nrows = a.num_vectors();
        let dim = a.repr().intrinsic_dim();
        let padded_rows = nrows.div_ceil(MR) * MR;
        let block_stride = match packing {
            APacking::RowMajor => dim * MR,
            APacking::Grouped4 => dim.div_ceil(4) * 4 * MR,
            #[cfg(target_arch = "x86_64")]
            APacking::Grouped8 => dim.div_ceil(8) * 8 * MR,
        };
        let mut values = vec![0; nrows.div_ceil(MR) * block_stride];
        let mut scale = vec![0.0; padded_rows];
        let mut bias = vec![0.0; padded_rows];
        let mut scaled_sum = vec![0.0; padded_rows];

        for (row_index, row) in a.rows().enumerate() {
            let meta = row.meta();
            scale[row_index] = meta.a;
            bias[row_index] = meta.b;
            scaled_sum[row_index] = meta.n;
            let block = row_index / MR;
            let lane = row_index % MR;
            let vector = row.vector();
            for k in 0..dim {
                let index = match packing {
                    APacking::RowMajor => block * block_stride + lane * dim + k,
                    APacking::Grouped4 => {
                        let chunk = k / 4;
                        let offset = k % 4;
                        block * block_stride + chunk * MR * 4 + lane * 4 + offset
                    }
                    #[cfg(target_arch = "x86_64")]
                    APacking::Grouped8 => {
                        let chunk = k / 8;
                        let offset = k % 8;
                        block * block_stride + chunk * MR * 8 + lane * 8 + offset
                    }
                };
                // SAFETY: `k` is bounded by the common intrinsic dimension.
                values[index] = unsafe { vector.get_unchecked(k) } as u8;
            }
        }

        Self {
            nrows,
            dim,
            block_stride,
            values,
            scale,
            bias,
            scaled_sum,
        }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

//--------//
// Driver //
//--------//

pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: &'a PackedMinMax8<MR>,
    b: MatRef<'a, MinMaxMeta<4>>,
    c: &'a mut [f32],
    k: DimK,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// `a.dim()`, `b`'s intrinsic dimension, and `k` must agree. A must use the packing
    /// required by `A`, and `c` must contain exactly `a.nrows()` values.
    pub(crate) unsafe fn new(
        arch: A,
        a: &'a PackedMinMax8<MR>,
        b: MatRef<'a, MinMaxMeta<4>>,
        c: &'a mut [f32],
        k: DimK,
    ) -> Self {
        let a_blocks = a.nrows.div_ceil(MR);
        let padded_rows = a_blocks * MR;
        debug_assert_eq!(a.dim, k.value().get());
        debug_assert_eq!(b.repr().intrinsic_dim(), k.value().get());
        debug_assert_eq!(a.values.len(), a_blocks * a.block_stride);
        debug_assert!(a.block_stride >= k.value().get() * MR);
        debug_assert_eq!(a.scale.len(), padded_rows);
        debug_assert_eq!(a.bias.len(), padded_rows);
        debug_assert_eq!(a.scaled_sum.len(), padded_rows);
        debug_assert_eq!(c.len(), a.nrows);
        Self { arch, a, b, c, k }
    }
}

impl<A, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR>,
    for<'a> PanelKernel<'a, A, MR, NR>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                self.c.fill(f32::MAX);
                let all_a_values = Slice::new(&self.a.values);

                for block in 0..self.a.nrows.div_ceil(MR) {
                    let a_base = block * MR;
                    let valid_rows = (self.a.nrows - a_base).min(MR);
                    let values_start = block * self.a.block_stride;
                    // SAFETY: Each A block occupies exactly `a_block_stride` bytes.
                    let a_values = unsafe {
                        all_a_values
                            .add(Elements::new(values_start))
                            .truncate(Elements::new(self.a.block_stride))
                    };

                    // SAFETY: A metadata arrays are padded to complete `MR`-row blocks.
                    let (a_scale, a_bias, a_scaled_sum) = unsafe {
                        (
                            &*self.a.scale.as_ptr().add(a_base).cast::<[f32; MR]>(),
                            &*self.a.bias.as_ptr().add(a_base).cast::<[f32; MR]>(),
                            &*self.a.scaled_sum.as_ptr().add(a_base).cast::<[f32; MR]>(),
                        )
                    };

                    let c = util::LoadStore::<f32, MR>::load(
                        self.arch,
                        &self.c[a_base..][..valid_rows],
                    );
                    let mut panel = PanelKernel {
                        arch: self.arch,
                        a_values,
                        a_scale,
                        a_bias,
                        a_scaled_sum,
                        b: self.b,
                        c,
                        k: self.k,
                        valid_rows,
                    };
                    driver::PanelKernel::panel_kernel(&mut panel);
                    util::LoadStore::<f32, MR>::store(
                        self.arch,
                        panel.c,
                        &mut self.c[a_base..][..valid_rows],
                    );
                }
            },
        );
    }
}

//-------------//
// PanelKernel //
//-------------//

struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a_values: Slice<'a, u8>,
    a_scale: &'a [f32; MR],
    a_bias: &'a [f32; MR],
    a_scaled_sum: &'a [f32; MR],
    b: MatRef<'a, MinMaxMeta<4>>,
    c: [f32; MR],
    k: DimK,
    valid_rows: usize,
}

impl<A: Copy, const MR: usize, const NR: usize> PanelKernel<'_, A, MR, NR> {
    #[inline(always)]
    fn run_micro<const N: usize>(&mut self, b_start: usize)
    where
        for<'a> MicroKernel<'a, A, MR, N>: driver::MicroKernel,
    {
        let mut micro = MicroKernel {
            arch: self.arch,
            a_values: self.a_values,
            a_scale: self.a_scale,
            a_bias: self.a_bias,
            a_scaled_sum: self.a_scaled_sum,
            b: BPanel::new(self.b, b_start, self.k.value().get()),
            c: &mut self.c,
            k: self.k,
            valid_rows: self.valid_rows,
        };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [$($tail:literal),+ $(,)?]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let full_rows = self.b.num_vectors() - self.b.num_vectors() % $nr;
                for b_start in (0..full_rows).step_by($nr) {
                    self.run_micro::<$nr>(b_start);
                }

                let remainder = self.b.num_vectors() - full_rows;
                $(
                    if remainder == $tail {
                        self.run_micro::<$tail>(full_rows);
                        return;
                    }
                )+
                debug_assert_eq!(remainder, 0);
            }
        }
    };
}

panel_kernel!(Scalar, 8, 6, [1, 2, 3, 4, 5]);

struct BPanel<'a, const N: usize> {
    values: [Slice<'a, u8>; N],
    meta: [MinMaxCompensation; N],
}

impl<'a, const N: usize> BPanel<'a, N> {
    #[inline(always)]
    fn new(b: MatRef<'a, MinMaxMeta<4>>, start: usize, k: usize) -> Self {
        let values = core::array::from_fn(|j| {
            // SAFETY: Panel dispatch ensures `start + j < b.num_vectors()`.
            let row = unsafe { b.get_row_unchecked(start + j) };
            let vector = row.vector();
            // SAFETY: The row owns `ceil(k / 2)` densely packed MinMax4 bytes for lifetime
            // `'a`; the `MatRef` retains that allocation for the returned panel.
            unsafe {
                Slice::from_raw(
                    std::ptr::NonNull::new_unchecked(vector.as_ptr().cast_mut()),
                    Bound::new(k.div_ceil(2)),
                )
            }
        });
        let mut meta = [MinMaxCompensation::default(); N];
        for (j, value) in meta.iter_mut().enumerate() {
            // SAFETY: Panel dispatch ensures `start + j < b.num_vectors()`.
            let row = unsafe { b.get_row_unchecked(start + j) };
            *value = row.meta();
        }
        Self { values, meta }
    }
}

//-------------//
// MicroKernel //
//-------------//

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a_values: Slice<'a, u8>,
    a_scale: &'a [f32; MR],
    a_bias: &'a [f32; MR],
    a_scaled_sum: &'a [f32; MR],
    b: BPanel<'a, NR>,
    c: &'a mut [f32; MR],
    k: DimK,
    valid_rows: usize,
}

macro_rules! micro_kernel {
    ($arch:ty, $mr:literal, $function:path, {$($nr:literal),+ $(,)?}) => {
        $(
            impl driver::MicroKernel for MicroKernel<'_, $arch, $mr, $nr> {
                #[inline(always)]
                fn micro_kernel(&mut self) {
                    self.arch.run_inline(
                        #[inline]
                        || {
                            // SAFETY: `Driver` and `BPanel` retain the complete A and B
                            // spans required by the architecture-specific micro-kernel.
                            unsafe {
                                $function(
                                    self.arch,
                                    self.a_values,
                                    self.a_scale,
                                    self.a_bias,
                                    self.a_scaled_sum,
                                    &self.b,
                                    self.c,
                                    self.k.value().get(),
                                    self.valid_rows,
                                )
                            }
                        },
                    )
                }
            }
        )+
    };
}

#[inline(always)]
unsafe fn expand_full_u4(values: Slice<'_, u8>, chunk: usize) -> u32 {
    // SAFETY: A full chunk contains two packed MinMax4 bytes.
    let packed = unsafe {
        values
            .add(Elements::new(chunk * 2))
            .truncate(Elements::new(2))
    };
    // SAFETY: `packed` tracks exactly two bytes.
    let first = unsafe { *packed.as_unit().as_ref() };
    // SAFETY: The second byte is within the tracked two-byte span.
    let second = unsafe { *packed.add(Elements::new(1)).as_unit().as_ref() };
    u32::from_le_bytes([first & 0x0f, first >> 4, second & 0x0f, second >> 4])
}

#[inline(always)]
unsafe fn expand_tail_u4(values: Slice<'_, u8>, byte_offset: usize, remainder: usize) -> u32 {
    debug_assert!((1..4).contains(&remainder));
    // SAFETY: Every non-empty tail contains its first byte.
    let tail = unsafe { values.add(Elements::new(byte_offset)) };
    // SAFETY: `tail` tracks at least one byte.
    let first = unsafe { *tail.as_unit().as_ref() };
    let second = if remainder == 3 {
        // SAFETY: A three-value tail contains a second byte.
        unsafe { *tail.add(Elements::new(1)).as_unit().as_ref() }
    } else {
        0
    };
    u32::from_le_bytes([
        first & 0x0f,
        if remainder >= 2 { first >> 4 } else { 0 },
        second & 0x0f,
        0,
    ])
}

trait ExtraWide<const MR: usize>: Architecture + Copy {
    type A: SIMDVector<Arch = Self, Scalar = u8>;
    type B: Copy;
    type Accumulator: SIMDVector<Arch = Self> + Copy;
    type Float: SIMDVector<Arch = Self, Scalar = f32>
        + SIMDMinMax
        + std::ops::Add<Output = Self::Float>
        + std::ops::Sub<Output = Self::Float>
        + std::ops::Mul<Output = Self::Float>;

    const HALF_ROWS: usize;
    const DIMENSIONS: usize;

    /// # Safety
    ///
    /// The packed B group at `byte_offset` must contain `dimensions` values.
    unsafe fn unpack_b(
        self,
        values: Slice<'_, u8>,
        byte_offset: usize,
        dimensions: usize,
    ) -> Self::B;

    fn dot(self, accumulator: Self::Accumulator, a: Self::A, b: Self::B) -> Self::Accumulator;

    fn to_float(self, accumulator: Self::Accumulator) -> Self::Float;
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn micro_kernel<W, const MR: usize, const NR: usize>(
    wide: W,
    a_values: Slice<'_, u8>,
    a_scale: &[f32; MR],
    a_bias: &[f32; MR],
    a_scaled_sum: &[f32; MR],
    b: &BPanel<'_, NR>,
    scores: &mut [f32; MR],
    k: usize,
    valid_rows: usize,
) where
    W: ExtraWide<MR>,
{
    debug_assert_eq!(MR, 2 * W::HALF_ROWS);

    // SAFETY: A is padded to complete dimension groups, and every `Slice` operation
    // retains debug bounds until the final SIMD load.
    unsafe {
        let has_hi = valid_rows > W::HALF_ROWS;
        let mut lo = [W::Accumulator::default(wide); NR];
        let mut hi = [W::Accumulator::default(wide); NR];

        for chunk in 0..k / W::DIMENSIONS {
            let a_panel = a_values
                .add(Elements::new(chunk * MR * W::DIMENSIONS))
                .truncate(Elements::new(MR * W::DIMENSIONS));
            let half_bytes = W::HALF_ROWS * W::DIMENSIONS;
            let a_lo = W::A::load_simd(wide, a_panel.truncate(Elements::new(half_bytes)).as_ptr());
            let a_hi = if has_hi {
                Some(W::A::load_simd(
                    wide,
                    a_panel
                        .add(Elements::new(half_bytes))
                        .truncate(Elements::new(half_bytes))
                        .as_ptr(),
                ))
            } else {
                None
            };

            for (j, accumulator) in lo.iter_mut().enumerate() {
                let b_panel = wide.unpack_b(
                    b.values[j],
                    chunk * W::DIMENSIONS.div_ceil(2),
                    W::DIMENSIONS,
                );
                *accumulator = wide.dot(*accumulator, a_lo, b_panel);
                if let Some(a_hi) = a_hi {
                    hi[j] = wide.dot(hi[j], a_hi, b_panel);
                }
            }
        }

        let remainder = k % W::DIMENSIONS;
        if remainder != 0 {
            let chunk = k / W::DIMENSIONS;
            let a_panel = a_values
                .add(Elements::new(chunk * MR * W::DIMENSIONS))
                .truncate(Elements::new(MR * W::DIMENSIONS));
            let half_bytes = W::HALF_ROWS * W::DIMENSIONS;
            let a_lo = W::A::load_simd(wide, a_panel.truncate(Elements::new(half_bytes)).as_ptr());
            let a_hi = if has_hi {
                Some(W::A::load_simd(
                    wide,
                    a_panel
                        .add(Elements::new(half_bytes))
                        .truncate(Elements::new(half_bytes))
                        .as_ptr(),
                ))
            } else {
                None
            };
            for (j, accumulator) in lo.iter_mut().enumerate() {
                let b_panel =
                    wide.unpack_b(b.values[j], chunk * W::DIMENSIONS.div_ceil(2), remainder);
                *accumulator = wide.dot(*accumulator, a_lo, b_panel);
                if let Some(a_hi) = a_hi {
                    hi[j] = wide.dot(hi[j], a_hi, b_panel);
                }
            }
        }

        let a_scale_lo = W::Float::load_simd(wide, a_scale.as_ptr());
        let a_scale_hi = W::Float::load_simd(wide, a_scale.as_ptr().add(W::HALF_ROWS));
        let a_bias_lo = W::Float::load_simd(wide, a_bias.as_ptr());
        let a_bias_hi = W::Float::load_simd(wide, a_bias.as_ptr().add(W::HALF_ROWS));
        let a_sum_lo = W::Float::load_simd(wide, a_scaled_sum.as_ptr());
        let a_sum_hi = W::Float::load_simd(wide, a_scaled_sum.as_ptr().add(W::HALF_ROWS));
        let mut score_lo = W::Float::load_simd(wide, scores.as_ptr());
        let mut score_hi =
            has_hi.then(|| W::Float::load_simd(wide, scores.as_ptr().add(W::HALF_ROWS)));
        let zero = W::Float::default(wide);

        for (j, accumulator) in lo.iter().copied().enumerate() {
            let doc = b.meta[j];
            let raw_lo = wide.to_float(accumulator);
            let doc_scale = W::Float::splat(wide, doc.a);
            let doc_bias = W::Float::splat(wide, doc.b);
            let doc_sum = W::Float::splat(wide, doc.n);
            let dim = W::Float::splat(wide, k as f32);

            let mut similarity_lo = (a_scale_lo * doc_scale) * raw_lo;
            similarity_lo = similarity_lo + a_sum_lo * doc_bias;
            similarity_lo = similarity_lo + doc_sum * a_bias_lo;
            similarity_lo = similarity_lo + (a_bias_lo * doc_bias) * dim;
            score_lo = score_lo.min_simd_standard(zero - similarity_lo);

            if let Some(score_hi) = score_hi.as_mut() {
                let raw_hi = wide.to_float(hi[j]);
                let mut similarity_hi = (a_scale_hi * doc_scale) * raw_hi;
                similarity_hi = similarity_hi + a_sum_hi * doc_bias;
                similarity_hi = similarity_hi + doc_sum * a_bias_hi;
                similarity_hi = similarity_hi + (a_bias_hi * doc_bias) * dim;
                *score_hi = score_hi.min_simd_standard(zero - similarity_hi);
            }
        }

        score_lo.store_simd(scores.as_mut_ptr());
        if let Some(score_hi) = score_hi {
            score_hi.store_simd(scores.as_mut_ptr().add(W::HALF_ROWS));
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn scalar_micro_kernel<const NR: usize>(
    _arch: Scalar,
    a_values: Slice<'_, u8>,
    a_scale: &[f32; 8],
    a_bias: &[f32; 8],
    a_scaled_sum: &[f32; 8],
    b: &BPanel<'_, NR>,
    scores: &mut [f32; 8],
    k: usize,
    valid_rows: usize,
) {
    // SAFETY: The A panel and each B row retain enough tracked bytes for every access.
    unsafe {
        let a_values = a_values.as_std_slice(8 * k);
        let b_bytes = k.div_ceil(2);
        for i in 0..valid_rows {
            let a_row = a_values.get_unchecked(i * k..(i + 1) * k);
            for (j, meta) in b.meta.iter().enumerate() {
                let b_row = b.values[j].as_std_slice(b_bytes);
                let mut raw = 0_u32;
                for byte in 0..k / 2 {
                    let packed = *b_row.get_unchecked(byte);
                    let a0 = *a_row.get_unchecked(2 * byte);
                    let a1 = *a_row.get_unchecked(2 * byte + 1);
                    raw += u32::from(a0) * u32::from(packed & 0x0f)
                        + u32::from(a1) * u32::from(packed >> 4);
                }
                if !k.is_multiple_of(2) {
                    let packed = *b_row.get_unchecked(k / 2);
                    let a_value = *a_row.get_unchecked(k - 1);
                    raw += u32::from(a_value) * u32::from(packed & 0x0f);
                }

                let similarity = a_scale[i] * meta.a * raw as f32
                    + a_scaled_sum[i] * meta.b
                    + meta.n * a_bias[i]
                    + a_bias[i] * meta.b * k as f32;
                scores[i] = scores[i].min(-similarity);
            }
        }
    }
}

micro_kernel!(Scalar, 8, scalar_micro_kernel, {6, 5, 4, 3, 2, 1});

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    use diskann_wide::SIMDVector;
    use diskann_wide::arch::x86_64::{V3, V4};

    #[inline(always)]
    unsafe fn unpack_full_u4_avx2(
        arch: V3,
        values: Slice<'_, u8>,
        byte_offset: usize,
    ) -> <V3 as Architecture>::i8x32 {
        // wide does not expose 16-bit logical shifts or this byte interleave.
        use std::arch::x86_64::{_mm256_srli_epi16, _mm256_unpacklo_epi8};
        diskann_wide::alias!(i16s = <V3>::i16x16);
        diskann_wide::alias!(i8s = <V3>::i8x32);

        // SAFETY: A full group contains two packed MinMax4 bytes.
        let packed = unsafe {
            values
                .add(Elements::new(byte_offset))
                .truncate(Elements::new(2))
        };
        // SAFETY: `packed` tracks exactly two bytes.
        let first = unsafe { *packed.as_unit().as_ref() };
        // SAFETY: The second byte is within the tracked two-byte span.
        let second = unsafe { *packed.add(Elements::new(1)).as_unit().as_ref() };
        let packed = i16s::splat(arch, i16::from_le_bytes([first, second])).to_underlying();
        let mask = i8s::splat(arch, 0x0f);
        let low = i8s::from_underlying(arch, packed) & mask;
        // SAFETY: V3 provides AVX2.
        unsafe {
            let high = i8s::from_underlying(arch, _mm256_srli_epi16::<4>(packed)) & mask;
            i8s::from_underlying(
                arch,
                _mm256_unpacklo_epi8(low.to_underlying(), high.to_underlying()),
            )
        }
    }

    panel_kernel!(V3, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
    panel_kernel!(V4, 16, 8, [1, 2, 3, 4, 5, 6, 7]);

    impl ExtraWide<16> for V3 {
        type A = <V3 as Architecture>::u8x32;
        type B = <V3 as Architecture>::i8x32;
        type Accumulator = <V3 as Architecture>::i32x8;
        type Float = <V3 as Architecture>::f32x8;

        const HALF_ROWS: usize = 8;
        const DIMENSIONS: usize = 4;

        unsafe fn unpack_b(
            self,
            values: Slice<'_, u8>,
            byte_offset: usize,
            dimensions: usize,
        ) -> Self::B {
            diskann_wide::alias!(i8s = <V3>::i8x32);
            diskann_wide::alias!(u32s = <V3>::u32x8);

            if dimensions == 4 && !cfg!(miri) {
                // SAFETY: Four dimensions occupy two packed bytes, and V3 provides AVX2.
                return unsafe { unpack_full_u4_avx2(self, values, byte_offset) };
            }

            let expanded = if dimensions == 4 {
                // SAFETY: Four dimensions occupy two packed bytes.
                unsafe { expand_full_u4(values, byte_offset / 2) }
            } else {
                // SAFETY: Inherited from the trait contract.
                unsafe { expand_tail_u4(values, byte_offset, dimensions) }
            };
            if cfg!(miri) {
                let bytes = expanded.to_le_bytes();
                i8s::from_array(self, core::array::from_fn(|i| bytes[i % 4] as i8))
            } else {
                i8s::from_underlying(self, u32s::splat(self, expanded).to_underlying())
            }
        }

        fn dot(self, accumulator: Self::Accumulator, a: Self::A, b: Self::B) -> Self::Accumulator {
            use diskann_wide::SIMDDotProduct;
            // wide exposes the i16 dot product, but not the mixed-byte pair product.
            use std::arch::x86_64::_mm256_maddubs_epi16;
            diskann_wide::alias!(i16s = <V3>::i16x16);

            let products = if cfg!(miri) {
                let a = a.to_array();
                let b = b.to_array();
                i16s::from_array(
                    self,
                    core::array::from_fn(|i| {
                        let x0 = i32::from(a[2 * i]) * i32::from(b[2 * i]);
                        let x1 = i32::from(a[2 * i + 1]) * i32::from(b[2 * i + 1]);
                        (x0 + x1).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
                    }),
                )
            } else {
                i16s::from_underlying(
                    self,
                    // SAFETY: V3 provides AVX2.
                    unsafe { _mm256_maddubs_epi16(a.to_underlying(), b.to_underlying()) },
                )
            };
            accumulator.dot_simd(products, i16s::splat(self, 1))
        }

        fn to_float(self, accumulator: Self::Accumulator) -> Self::Float {
            Self::Float::from_array(self, accumulator.to_array().map(|x| (x as u32) as f32))
        }
    }

    impl ExtraWide<16> for V4 {
        type A = <V4 as Architecture>::u8x64;
        type B = <V4 as Architecture>::i8x64;
        type Accumulator = <V4 as Architecture>::i32x16;
        type Float = <V4 as Architecture>::f32x8;

        const HALF_ROWS: usize = 8;
        const DIMENSIONS: usize = 8;

        unsafe fn unpack_b(
            self,
            values: Slice<'_, u8>,
            byte_offset: usize,
            dimensions: usize,
        ) -> Self::B {
            diskann_wide::alias!(i8s = <V4>::i8x64);

            #[cfg(miri)]
            {
                let byte_count = dimensions.div_ceil(2);
                // SAFETY: The caller guarantees a valid packed B group.
                let packed = unsafe {
                    values
                        .add(Elements::new(byte_offset))
                        .truncate(Elements::new(byte_count))
                        .as_std_slice(byte_count)
                };
                i8s::from_array(
                    self,
                    core::array::from_fn(|i| {
                        let dimension = i % 8;
                        if dimension >= dimensions {
                            0
                        } else {
                            let value = packed[dimension / 2];
                            if dimension.is_multiple_of(2) {
                                (value & 0x0f) as i8
                            } else {
                                (value >> 4) as i8
                            }
                        }
                    }),
                )
            }

            #[cfg(not(miri))]
            {
                use std::arch::x86_64::_pdep_u64;
                diskann_wide::alias!(u64s = <V4>::u64x8);

                let byte_count = dimensions.div_ceil(2);
                // SAFETY: The caller guarantees a valid packed B group.
                let packed = unsafe {
                    values
                        .add(Elements::new(byte_offset))
                        .truncate(Elements::new(byte_count))
                };
                let source = if dimensions == 8 {
                    // SAFETY: A full group tracks four bytes and the load is unaligned.
                    u32::from_le(unsafe { packed.as_ptr().cast::<u32>().read_unaligned() })
                } else {
                    let mut source = 0_u32;
                    for index in 0..byte_count {
                        // SAFETY: `index` is within the tracked packed group.
                        let value = unsafe { *packed.add(Elements::new(index)).as_unit().as_ref() };
                        source |= u32::from(value) << (8 * index);
                    }
                    source
                };
                // SAFETY: V4 provides BMI2; wide has no bit-deposit operation.
                let expanded = unsafe { _pdep_u64(u64::from(source), 0x0f0f_0f0f_0f0f_0f0f) };
                i8s::from_underlying(self, u64s::splat(self, expanded).to_underlying())
            }
        }

        fn dot(self, accumulator: Self::Accumulator, a: Self::A, b: Self::B) -> Self::Accumulator {
            use diskann_wide::SIMDDotProduct;
            accumulator.dot_simd(a, b)
        }

        fn to_float(self, accumulator: Self::Accumulator) -> Self::Float {
            diskann_wide::alias!(f32s = <V4>::f32x8);

            #[cfg(miri)]
            {
                let values = accumulator.to_array();
                f32s::from_array(
                    self,
                    core::array::from_fn(|i| {
                        values[2 * i].wrapping_add(values[2 * i + 1]) as u32 as f32
                    }),
                )
            }

            #[cfg(not(miri))]
            {
                use std::arch::x86_64::_mm512_cvtepi64_epi32;
                diskann_wide::alias!(i32s8 = <V4>::i32x8);
                diskann_wide::alias!(u64s = <V4>::u64x8);

                let upper = u64s::from_underlying(self, accumulator.to_underlying()) >> 32;
                let pairs =
                    accumulator + Self::Accumulator::from_underlying(self, upper.to_underlying());
                // SAFETY: V4 provides AVX-512F; wide has no u64-to-u32 lane narrowing.
                let reduced = i32s8::from_underlying(self, unsafe {
                    _mm512_cvtepi64_epi32(pairs.to_underlying())
                });
                f32s::from_array(self, reduced.to_array().map(|x| (x as u32) as f32))
            }
        }
    }

    micro_kernel!(V3, 16, micro_kernel, {8, 7, 6, 5, 4, 3, 2, 1});
    micro_kernel!(V4, 16, micro_kernel, {8, 7, 6, 5, 4, 3, 2, 1});
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    use diskann_wide::SIMDVector;
    use diskann_wide::arch::aarch64::Neon;

    panel_kernel!(Neon, 8, 8, [1, 2, 3, 4, 5, 6, 7]);

    #[inline(always)]
    unsafe fn neon_unpack(
        arch: Neon,
        values: Slice<'_, u8>,
        byte_offset: usize,
        dimensions: usize,
    ) -> <Neon as Architecture>::u8x16 {
        diskann_wide::alias!(u32s = <Neon>::u32x4);
        diskann_wide::alias!(u8s = <Neon>::u8x16);

        let expanded = if dimensions == 4 {
            // SAFETY: Four dimensions occupy two packed bytes.
            unsafe { expand_full_u4(values, byte_offset / 2) }
        } else {
            // SAFETY: The caller guarantees a valid final partial group.
            unsafe { expand_tail_u4(values, byte_offset, dimensions) }
        };
        if cfg!(miri) {
            let bytes = expanded.to_le_bytes();
            u8s::from_array(arch, core::array::from_fn(|i| bytes[i % 4]))
        } else {
            u8s::from_underlying(
                arch,
                // SAFETY: Reinterpreting a vector does not change its bits.
                unsafe {
                    std::arch::aarch64::vreinterpretq_u8_u32(
                        u32s::splat(arch, expanded).to_underlying(),
                    )
                },
            )
        }
    }

    #[inline(always)]
    fn neon_dot(
        _arch: Neon,
        accumulator: <Neon as Architecture>::u32x4,
        a: <Neon as Architecture>::u8x16,
        b: <Neon as Architecture>::u8x16,
    ) -> <Neon as Architecture>::u32x4 {
        use diskann_wide::SIMDDotProduct;
        accumulator.dot_simd(a, b)
    }

    #[inline(always)]
    fn neon_to_float(
        arch: Neon,
        accumulator: <Neon as Architecture>::u32x4,
    ) -> <Neon as Architecture>::f32x4 {
        <Neon as Architecture>::f32x4::from_array(arch, accumulator.to_array().map(|x| x as f32))
    }

    impl ExtraWide<8> for Neon {
        type A = <Neon as Architecture>::u8x16;
        type B = <Neon as Architecture>::u8x16;
        type Accumulator = <Neon as Architecture>::u32x4;
        type Float = <Neon as Architecture>::f32x4;

        const HALF_ROWS: usize = 4;
        const DIMENSIONS: usize = 4;

        unsafe fn unpack_b(
            self,
            values: Slice<'_, u8>,
            byte_offset: usize,
            dimensions: usize,
        ) -> Self::B {
            // SAFETY: Inherited from the trait contract.
            unsafe { neon_unpack(self, values, byte_offset, dimensions) }
        }

        fn dot(self, accumulator: Self::Accumulator, a: Self::A, b: Self::B) -> Self::Accumulator {
            neon_dot(self, accumulator, a, b)
        }

        fn to_float(self, accumulator: Self::Accumulator) -> Self::Float {
            neon_to_float(self, accumulator)
        }
    }

    micro_kernel!(Neon, 8, micro_kernel, {8, 7, 6, 5, 4, 3, 2, 1});
}

#[cfg(test)]
mod tests {
    use diskann_utils::ReborrowMut;

    use super::*;
    use crate::multi_vector::{Defaulted, Mat};

    fn check_packing<const MR: usize>(packing: APacking, group: Option<usize>) {
        for nrows in [0, 1, MR - 1, MR, MR + 1] {
            for dim in [0, 1, 3, 4, 5, 7, 8, 9] {
                let mut query = Mat::new(MinMaxMeta::<8>::new(nrows, dim), Defaulted).unwrap();
                for (i, mut row) in query.reborrow_mut().rows_mut().enumerate() {
                    row.set_meta(MinMaxCompensation {
                        a: (i + 1) as f32,
                        b: -((i + 2) as f32),
                        n: (i + 3) as f32,
                        dim: dim as u32,
                        ..Default::default()
                    });
                    for k in 0..dim {
                        row.vector_mut()
                            .set(k, ((i * 17 + k) % 256) as i64)
                            .unwrap();
                    }
                }

                let packed = PackedMinMax8::<MR>::new(query.as_view(), packing);
                assert_eq!(packed.nrows(), nrows);
                assert_eq!(packed.dim(), dim);
                let group = group.unwrap_or(dim.max(1));
                let blocks = nrows.div_ceil(MR);
                assert_eq!(packed.block_stride, dim.div_ceil(group) * group * MR);

                let mut expected = Vec::new();
                for block in 0..blocks {
                    for chunk in 0..dim.div_ceil(group) {
                        for lane in 0..MR {
                            for offset in 0..group {
                                let row = block * MR + lane;
                                let k = chunk * group + offset;
                                expected.push(if row < nrows && k < dim {
                                    ((row * 17 + k) % 256) as u8
                                } else {
                                    0
                                });
                            }
                        }
                    }
                }
                assert_eq!(packed.values, expected, "{packing:?}: ({nrows}, {dim})");
                for (values, offset, sign) in [
                    (&packed.scale, 1, 1.0),
                    (&packed.bias, 2, -1.0),
                    (&packed.scaled_sum, 3, 1.0),
                ] {
                    assert_eq!(values.len(), blocks * MR);
                    for (i, &value) in values.iter().enumerate() {
                        assert_eq!(
                            value,
                            if i < nrows {
                                sign * (i + offset) as f32
                            } else {
                                0.0
                            },
                            "{packing:?}: metadata row {i} for ({nrows}, {dim})",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn row_major_packing() {
        check_packing::<8>(APacking::RowMajor, None);
    }

    #[test]
    fn grouped4_packing() {
        check_packing::<8>(APacking::Grouped4, Some(4));
        check_packing::<16>(APacking::Grouped4, Some(4));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn grouped8_packing() {
        check_packing::<16>(APacking::Grouped8, Some(8));
    }

    #[test]
    fn expands_full_u4_groups() {
        for first in 0..=u8::MAX {
            for second in [0, 0x0f, 0xf0, 0xff] {
                let packed = [0xab, 0xcd, first, second];
                // SAFETY: The second chunk contains two packed bytes.
                let expanded = unsafe { expand_full_u4(Slice::new(&packed), 1) };
                assert_eq!(
                    expanded.to_le_bytes(),
                    [first & 0x0f, first >> 4, second & 0x0f, second >> 4],
                );
            }
        }
    }

    #[test]
    fn expands_u4_tails_without_padding_nibbles() {
        let packed = [0x21, 0xf3, 0x21, 0xf3];
        for byte_offset in [0, 2] {
            for (remainder, expected) in [
                (1_usize, [1, 0, 0, 0]),
                (2, [1, 2, 0, 0]),
                (3, [1, 2, 3, 0]),
            ] {
                let values = Slice::new(&packed[..byte_offset + remainder.div_ceil(2)]);
                // SAFETY: The tracked span contains exactly the packed tail bytes.
                let expanded = unsafe { expand_tail_u4(values, byte_offset, remainder) };
                assert_eq!(expanded.to_le_bytes(), expected);
            }
        }
    }
}
