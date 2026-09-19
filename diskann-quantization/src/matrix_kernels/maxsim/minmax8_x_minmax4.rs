/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MinMax8 by MinMax4 MaxSim over the existing packed/unpacked panel views.
//!
//! A uses the shared 64-dimensional even/odd layout with explicit byte packing.
//! The driver expands each canonical MinMax4 B tile once, then reuses it across A's
//! L2 subviews and panels.
//! Integer contraction consumes padded K-dimensional panels without knowing the original
//! dimension or metadata. MinMax reduction uses the original D and opaque accumulators.
//! Scratch is owned by each call, never by the shared prepared query.

mod decode;
pub(crate) mod layout;
pub(crate) mod reader;

use diskann_wide::{Architecture, SIMDMinMax, SIMDVector, arch::Scalar};

use crate::{
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        bounds::{self, Bound},
        driver,
        num::{Bytes, DimK, Elements},
        ptr::{MutSlice, Slice},
        util,
    },
    minmax::MinMaxCompensation,
};

use super::packed_f32_x_unpacked_f32::Params as CacheParams;
use decode::Decoder;
use layout::PackedQueryView;
use reader::{BTile, MinMax4Rows};

/// Structure-of-arrays coefficients for one complete query panel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueryCompensation<const MR: usize> {
    pub(crate) scale: [f32; MR],
    pub(crate) bias: [f32; MR],
    pub(crate) scaled_sum: [f32; MR],
}

impl<const MR: usize> Default for QueryCompensation<MR> {
    fn default() -> Self {
        Self {
            scale: [0.0; MR],
            bias: [0.0; MR],
            scaled_sum: [0.0; MR],
        }
    }
}

/// B-first traversal over byte-valued panels; `k` always counts padded dimensions.
pub(crate) struct Driver<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: PackedQueryView<'a, MR, PACK>,
    a_meta: &'a [QueryCompensation<MR>],
    b: MinMax4Rows<'a>,
    b_values: Vec<u8>,
    b_metadata: Vec<MinMaxCompensation>,
    c: &'a mut [f32],
    blocking: CacheParams,
}

impl<'a, A, const PACK: usize, const MR: usize, const NR: usize> Driver<'a, A, PACK, MR, NR> {
    #[expect(
        clippy::expect_used,
        reason = "capacity overflow must fail before allocating scratch"
    )]
    pub(crate) fn new(
        arch: A,
        a: PackedQueryView<'a, MR, PACK>,
        a_meta: &'a [QueryCompensation<MR>],
        b: MinMax4Rows<'a>,
        c: &'a mut [f32],
        cache: Cache,
    ) -> Self {
        const { assert!(NR > 0) };
        let k = a.k();
        assert_eq!(b.dim(), a.layout().dim(), "document dimension mismatch");
        assert_eq!(
            a_meta.len(),
            a.nrows().div_ceil(MR),
            "query metadata length mismatch"
        );
        assert_eq!(c.len(), a.nrows(), "output length mismatch");
        let a_bytes = a
            .values()
            .block_stride(k)
            .bytes()
            .value()
            .checked_add(std::mem::size_of::<QueryCompensation<MR>>())
            .expect("query panel size overflow");
        let b_bytes = k
            .value()
            .get()
            .checked_add(std::mem::size_of::<MinMaxCompensation>())
            .expect("document row size overflow");
        NR.checked_mul(b_bytes)
            .expect("document panel size overflow");
        let blocking = CacheParams::new(cache, Bytes::new(a_bytes), Bytes::new(b_bytes), NR);
        let b_rows = b.rows().min(blocking.b_cols_in_l1).get();
        Self {
            arch,
            a,
            a_meta,
            b,
            b_values: vec![
                0;
                b_rows
                    .checked_mul(k.value().get())
                    .expect("document scratch overflow")
            ],
            b_metadata: vec![MinMaxCompensation::default(); b_rows],
            c,
            blocking,
        }
    }
}

impl<A, const PACK: usize, const MR: usize, const NR: usize> driver::Drive
    for Driver<'_, A, PACK, MR, NR>
where
    A: Decoder + util::LoadStore<f32, MR> + ExtraWide<PACK, MR>,
    for<'a> PanelKernel<'a, A, PACK, MR, NR>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                let layout = self.a.layout();
                let dim = layout.dim();
                let k = self.a.k();
                let b_rows_per_tile = self.b.rows().min(self.blocking.b_cols_in_l1);
                self.c.fill(f32::MAX);
                let output_rows = self.c.len();
                let mut c = MutSlice::new(self.c);
                let a_meta = Slice::new(self.a_meta);
                let on_b_tile = |b_tile: MinMax4Rows<'_>| {
                    let rows = b_tile.rows().get();
                    let decoded = b_tile.decode(
                        self.arch,
                        layout,
                        &mut self.b_values[..rows * k.value().get()],
                        &mut self.b_metadata[..rows],
                    );
                    let on_a_tile = |a_tile: packed::View<'_, u8, MR, PACK>,
                                     a_block_base: usize| {
                        let on_a_panel =
                            |a: packed::Panel<'_, u8, MR, PACK>, a_block_offset: usize| {
                                let block = a_block_base + a_block_offset;
                                let valid_rows = (output_rows - block * MR).min(MR);
                                // SAFETY: The global block index stays within A. Metadata
                                // and output cover every A block, including its partial tail.
                                unsafe {
                                    let mut region = c.subslice(block * MR, Bound::new(valid_rows));
                                    let output = region.as_std_mut_slice(valid_rows);
                                    let query = a_meta.add(Elements::new(block)).as_unit().as_ref();
                                    let mut panel = PanelKernel::new(
                                        self.arch, a, query, decoded, output, k, dim,
                                    );
                                    driver::PanelKernel::panel_kernel(&mut panel);
                                    util::LoadStore::<f32, MR>::store(self.arch, panel.c, output);
                                }
                            };
                        // SAFETY: The A subview inherits the validated padded dimension.
                        unsafe { a_tile.visit_panels(k, on_a_panel) };
                    };
                    // SAFETY: A was validated against k; subviews retain its bounds.
                    unsafe {
                        self.a
                            .values()
                            .visit_sub_views(self.blocking.a_panels_in_l2, k, on_a_tile)
                    };
                };
                self.b.visit_tiles(b_rows_per_tile, on_b_tile);
            },
        );
    }
}

struct PanelKernel<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR, PACK>,
    query: &'a QueryCompensation<MR>,
    b: BTile<'a>,
    c: [f32; MR],
    k: DimK,
    dim: usize,
    valid_rows: usize,
}

impl<'a, A, const PACK: usize, const MR: usize, const NR: usize> PanelKernel<'a, A, PACK, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR>,
{
    /// # Safety
    ///
    /// A and B have k padded columns. The output occupies at most MR query rows.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, u8, MR, PACK>,
        query: &'a QueryCompensation<MR>,
        b: BTile<'a>,
        c: &[f32],
        k: DimK,
        dim: usize,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.values.k(), k);
        bounds::check_le!(Bound::new(c.len()), MR);
        Self {
            arch,
            a,
            query,
            b,
            c: util::LoadStore::<f32, MR>::load(arch, c),
            k,
            dim,
            valid_rows: c.len(),
        }
    }
}

impl<A, const PACK: usize, const MR: usize, const NR: usize, const EXTENT: usize>
    unpacked::PanelVisitor<u8, EXTENT> for &mut PanelKernel<'_, A, PACK, MR, NR>
where
    A: Architecture + ExtraWide<PACK, MR>,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, u8, EXTENT>, start: usize) {
        // SAFETY: The visitor receives complete rows from the validated source.
        let b_meta = unsafe {
            self.b
                .meta
                .add(Elements::new(start))
                .truncate(Elements::new(EXTENT))
        };
        let mut micro = MicroKernel::<_, PACK, MR, EXTENT> {
            arch: self.arch,
            a: self.a,
            query: self.query,
            b,
            b_meta,
            c: &mut self.c,
            k: self.k,
            dim: self.dim,
            valid_rows: self.valid_rows,
        };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $pack:literal, $mr:literal, $nr:literal, [$($tail:literal),+]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $pack, $mr, $nr>
        {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let b = self.b.values;
                let k = self.k;
                let mut visitor = self;
                // SAFETY: The constructor checks B's row stride.
                let remainder = unsafe {
                    b.visit_panels::<$nr>(k, &mut *visitor)
                };
                if let Some(remainder) = remainder {
                    $(
                        if let Some(panel) = remainder.try_as_panel::<$tail>() {
                            unpacked::PanelVisitor::visit(
                                &mut visitor,
                                panel,
                                remainder.start(),
                            );
                        }
                    )+
                }
            }
        }
    };
}

struct MicroKernel<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, u8, MR, PACK>,
    query: &'a QueryCompensation<MR>,
    b: unpacked::Panel<'a, u8, NR>,
    b_meta: Slice<'a, MinMaxCompensation>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: usize,
    valid_rows: usize,
}

impl<A, const PACK: usize, const MR: usize, const NR: usize> driver::MicroKernel
    for MicroKernel<'_, A, PACK, MR, NR>
where
    A: Architecture + ExtraWide<PACK, MR>,
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        self.arch.run_inline(
            #[inline]
            || {
                bounds::check_eq!(self.a.k(), self.k);
                // SAFETY: The panel constructor establishes the common contraction
                // dimension and the number of valid query rows.
                let acc = unsafe { self.arch.contract(self.a, self.b, self.k, self.valid_rows) };
                // SAFETY: The metadata span contains exactly NR entries.
                unsafe {
                    self.arch
                        .reduce(acc, self.query, self.b_meta, self.dim, self.c)
                };
            },
        );
    }
}

/// Register layout and instruction selection are private to each implementation.
trait ExtraWide<const PACK: usize, const MR: usize>: Copy {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// # Safety
    ///
    /// `values` contains exactly MR * PACK bytes and `valid_rows <= MR`.
    unsafe fn load(self, values: Slice<'_, u8>, valid_rows: usize) -> Self::Query;
    fn zero(self) -> Self::Accumulator;
    fn splat(self, value: [u8; PACK]) -> Self::Splat;

    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator;

    #[cfg(test)]
    fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; MR];

    /// # Safety
    ///
    /// Both panels have contraction dimension `k`, `k % PACK == 0`, and `valid_rows <= MR`.
    /// B values are unsigned nibbles (0..=15).
    #[inline(always)]
    unsafe fn contract<const NR: usize>(
        self,
        a: packed::Panel<'_, u8, MR, PACK>,
        b: unpacked::Panel<'_, u8, NR>,
        k: DimK,
        valid_rows: usize,
    ) -> [Self::Accumulator; NR] {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_eq!(Bound::new(k.value().get() % PACK), 0);
        bounds::check_le!(Bound::new(valid_rows), MR);
        let mut acc = [self.zero(); NR];
        let bp = b.as_ptr();
        let b_stride = b.stride(k);
        for group in 0..k.value().get() / PACK {
            // SAFETY: Each group holds PACK contiguous bytes from every query row.
            let a = unsafe { self.load(a.group(group), valid_rows) };
            for (j, acc) in acc.iter_mut().enumerate() {
                // SAFETY: PACK divides K; this group lies wholly inside document row j.
                let b = unsafe {
                    bp.add(b_stride * j + Elements::new(group * PACK))
                        .truncate(Elements::new(PACK))
                        .as_ptr()
                        .cast::<[u8; PACK]>()
                        .read_unaligned()
                };
                *acc = self.dot(a, self.splat(b), *acc);
            }
        }
        acc
    }

    /// Apply MinMax compensation without exposing the accumulator's register layout.
    ///
    /// # Safety
    ///
    /// `docs` contains exactly NR metadata entries, one per accumulator.
    unsafe fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<MR>,
        docs: Slice<'_, MinMaxCompensation>,
        dim: usize,
        scores: &mut [f32; MR],
    );
}

impl ExtraWide<4, 8> for Scalar {
    type Query = [[u32; 8]; 4];
    type Splat = [u32; 4];
    type Accumulator = [u32; 8];

    #[cfg(test)]
    fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 8] {
        acc
    }

    #[inline(always)]
    unsafe fn load(self, values: Slice<'_, u8>, _: usize) -> Self::Query {
        // SAFETY: The trait contract requires exactly 8 * 4 initialized bytes.
        let values = unsafe { values.as_std_slice(32) };
        core::array::from_fn(|d| core::array::from_fn(|row| u32::from(values[row * 4 + d])))
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn splat(self, value: [u8; 4]) -> Self::Splat {
        value.map(u32::from)
    }
    #[inline(always)]
    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator {
        core::array::from_fn(|i| {
            let dot = a[0][i] * b[0] + a[1][i] * b[1] + a[2][i] * b[2] + a[3][i] * b[3];
            acc[i].wrapping_add(dot)
        })
    }

    #[inline(always)]
    unsafe fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<8>,
        docs: Slice<'_, MinMaxCompensation>,
        dim: usize,
        scores: &mut [f32; 8],
    ) {
        // SAFETY: The trait contract requires exactly NR metadata entries.
        let docs = unsafe { docs.as_std_slice(NR) };
        for (acc, doc) in acc.into_iter().zip(docs) {
            for i in 0..8 {
                let similarity = query.scale[i] * doc.a * acc[i] as f32
                    + query.scaled_sum[i] * doc.b
                    + doc.n * query.bias[i]
                    + query.bias[i] * doc.b * dim as f32;
                scores[i] = scores[i].min(-similarity);
            }
        }
    }
}

panel_kernel!(Scalar, 4, 8, 6, [1, 2, 3, 4, 5]);

// Kept separate from contraction so metadata and floating point constraints do not
// leak through ExtraWide's opaque register types.
macro_rules! compensate {
    ($arch:expr, $float:ty, $lanes:literal, $parts:literal, $acc:expr, $convert:expr, $query:expr, $docs:expr, $dim:expr, $scores:expr) => {{
        let query = $query;
        let acc = $acc;
        let docs = $docs;
        bounds::check_eq!(docs.len(), acc.len());
        let mut scores = MutSlice::new($scores);
        let scale = Slice::new(&query.scale);
        let bias = Slice::new(&query.bias);
        let sum = Slice::new(&query.scaled_sum);
        for i in 0..$parts {
            let start = i * $lanes;
            // SAFETY: The backend provides exactly MR / lanes vectors. All metadata
            // and scores have MR elements; each access is narrowed to one SIMD vector.
            unsafe {
                let scale = <$float>::load_simd(
                    $arch,
                    scale
                        .add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let bias = <$float>::load_simd(
                    $arch,
                    bias.add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let sum = <$float>::load_simd(
                    $arch,
                    sum.add(Elements::new(start))
                        .truncate(Elements::new($lanes))
                        .as_ptr(),
                );
                let mut output = scores.subslice(start, Bound::new($lanes));
                let mut best = <$float>::load_simd($arch, output.as_ptr());
                for (j, acc) in acc.iter().enumerate() {
                    let doc = docs.add(Elements::new(j)).as_unit().as_ref();
                    let raw = ($convert)(*acc, i);
                    let mut similarity = (scale * <$float>::splat($arch, doc.a)) * raw;
                    similarity = similarity + sum * <$float>::splat($arch, doc.b);
                    similarity = similarity + <$float>::splat($arch, doc.n) * bias;
                    similarity = similarity
                        + (bias * <$float>::splat($arch, doc.b))
                            * <$float>::splat($arch, $dim as f32);
                    best = best.min_simd_standard(<$float>::default($arch) - similarity);
                }
                best.store_simd(output.as_mut_ptr());
            }
        }
    }};
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::{
        SIMDDotProduct,
        arch::x86_64::{V3, V4},
    };

    impl ExtraWide<4, 16> for V3 {
        type Query = (
            <V3 as Architecture>::u8x32,
            Option<<V3 as Architecture>::u8x32>,
        );
        type Splat = <V3 as Architecture>::i8x32;
        type Accumulator = [<V3 as Architecture>::i32x8; 2];

        #[cfg(test)]
        fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 16] {
            let lanes = acc.map(|x| x.to_array());
            core::array::from_fn(|i| lanes[i / 8][i % 8] as u32)
        }

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, u8>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 64);
            // SAFETY: Each half contains eight groups, or 32 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(32)).as_ptr());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(32))
                            .truncate(Elements::new(32))
                            .as_ptr(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: [u8; 4]) -> Self::Splat {
            diskann_wide::alias!(u32s = <V3>::u32x8);
            Self::Splat::from_underlying(
                self,
                u32s::splat(self, u32::from_le_bytes(value)).to_underlying(),
            )
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            use std::arch::x86_64::_mm256_maddubs_epi16;
            diskann_wide::alias!(i16s = <V3>::i16x16);
            let dot = |a: <V3 as Architecture>::u8x32, acc: <V3 as Architecture>::i32x8| {
                // SAFETY: V3 provides AVX2. B contains unsigned nibbles, so each
                // pair sum is at most 2 * 255 * 15 and cannot saturate.
                let products = i16s::from_underlying(self, unsafe {
                    _mm256_maddubs_epi16(a.to_underlying(), b.to_underlying())
                });
                acc.dot_simd(products, i16s::splat(self, 1))
            };
            acc[0] = dot(a.0, acc[0]);
            if let Some(hi) = a.1 {
                acc[1] = dot(hi, acc[1]);
            }
            acc
        }

        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V3>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as u32 as f32))
            };
            compensate!(self, floats, 8, 2, acc, convert, query, docs, dim, scores);
        }
    }

    impl ExtraWide<8, 16> for V4 {
        type Query = (
            <V4 as Architecture>::u8x64,
            Option<<V4 as Architecture>::u8x64>,
        );
        type Splat = <V4 as Architecture>::i8x64;
        type Accumulator = [<V4 as Architecture>::i32x16; 2];

        #[cfg(test)]
        fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 16] {
            let lanes = acc.map(|x| x.to_array());
            core::array::from_fn(|i| {
                let part = &lanes[i / 8];
                part[2 * (i % 8)].wrapping_add(part[2 * (i % 8) + 1]) as u32
            })
        }

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, u8>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 128);
            // SAFETY: Each half contains eight groups, or 64 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(64)).as_ptr());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(64))
                            .truncate(Elements::new(64))
                            .as_ptr(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: [u8; 8]) -> Self::Splat {
            // Miri's V4 registers use scalar arrays, so byte reinterpretation needs an
            // explicit conversion rather than the native register's underlying type.
            #[cfg(miri)]
            {
                Self::Splat::from_array(self, core::array::from_fn(|lane| value[lane % 8] as i8))
            }

            #[cfg(not(miri))]
            {
                diskann_wide::alias!(u64s = <V4>::u64x8);
                Self::Splat::from_underlying(
                    self,
                    u64s::splat(self, u64::from_le_bytes(value)).to_underlying(),
                )
            }
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a.0, b);
            if let Some(hi) = a.1 {
                acc[1] = acc[1].dot_simd(hi, b);
            }
            acc
        }

        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V4>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                // V4's emulated arrays cannot be passed to native AVX-512 intrinsics.
                #[cfg(miri)]
                {
                    let values = acc[part].to_array();
                    floats::from_array(
                        self,
                        core::array::from_fn(|i| {
                            values[2 * i].wrapping_add(values[2 * i + 1]) as u32 as f32
                        }),
                    )
                }

                #[cfg(not(miri))]
                {
                    use std::arch::x86_64::{
                        _mm512_add_epi32, _mm512_cvtepi64_epi32, _mm512_srli_epi64,
                    };
                    diskann_wide::alias!(i32s = <V4>::i32x8);

                    let value = acc[part].to_underlying();
                    // SAFETY: V4 provides AVX-512F and AVX-512DQ.
                    let pairs = unsafe { _mm512_add_epi32(value, _mm512_srli_epi64::<32>(value)) };
                    // SAFETY: V4 provides AVX-512F and AVX-512DQ.
                    let reduced =
                        i32s::from_underlying(self, unsafe { _mm512_cvtepi64_epi32(pairs) });
                    floats::from_array(self, reduced.to_array().map(|x| x as u32 as f32))
                }
            };
            compensate!(self, floats, 8, 2, acc, convert, query, docs, dim, scores);
        }
    }

    panel_kernel!(V3, 4, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
    panel_kernel!(V4, 8, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::{SIMDDotProduct, arch::aarch64::Neon};

    impl ExtraWide<4, 8> for Neon {
        type Query = (
            <Neon as Architecture>::u8x16,
            Option<<Neon as Architecture>::u8x16>,
        );
        type Splat = <Neon as Architecture>::u8x16;
        type Accumulator = [<Neon as Architecture>::u32x4; 2];

        #[cfg(test)]
        fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 8] {
            let lanes = acc.map(|x| x.to_array());
            core::array::from_fn(|i| lanes[i / 4][i % 4])
        }

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, u8>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 32);
            // SAFETY: Each half contains four groups, or 16 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(16)).as_ptr());
                let hi = (rows > 4).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(16))
                            .truncate(Elements::new(16))
                            .as_ptr(),
                    )
                });
                (lo, hi)
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: [u8; 4]) -> Self::Splat {
            Self::Splat::from_array(self, core::array::from_fn(|i| value[i % 4]))
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a.0, b);
            if let Some(hi) = a.1 {
                acc[1] = acc[1].dot_simd(hi, b);
            }
            acc
        }

        #[inline(always)]
        unsafe fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<8>,
            docs: Slice<'_, MinMaxCompensation>,
            dim: usize,
            scores: &mut [f32; 8],
        ) {
            diskann_wide::alias!(floats = <Neon>::f32x4);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as f32))
            };
            compensate!(self, floats, 4, 2, acc, convert, query, docs, dim, scores);
        }
    }

    panel_kernel!(Neon, 4, 8, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_utils::views::Matrix;

    use super::layout::{EvenOdd64Layout, PackedQuery};
    use super::*;
    use crate::{
        matrix_kernels::{num::value_or_one, test_util::panic_message_for},
        minmax::{Data, DataMutRef, DataRef, MinMaxMeta},
        multi_vector::MatRef,
    };

    fn dimension(value: usize) -> DimK {
        DimK::new(NonZeroUsize::new(value).unwrap())
    }

    fn canonical(b: &Matrix<u8>, dim: usize) -> MinMax4Rows<'_> {
        MinMax4Rows::new(MatRef::new(MinMaxMeta::<4>::new(b.nrows(), dim), b.as_slice()).unwrap())
            .unwrap()
    }

    fn check_decoded_b<A: Decoder>(arch: A) {
        for &dim in layout::tests::DIMS.iter().filter(|&&d| d != 0) {
            for rows in [1, 7, 8, 9, 31, 32, 33] {
                if cfg!(miri) && !(matches!(dim, 1 | 8 | 9) && rows <= 8) {
                    continue;
                }
                let b = documents(rows, dim);
                let layout = EvenOdd64Layout::new(dim);
                let k = layout.padded();
                let mut values = vec![0xff; rows * k];
                let mut meta = vec![MinMaxCompensation::default(); rows];
                let decoded = canonical(&b, dim).decode(arch, layout, &mut values, &mut meta);
                // SAFETY: The decoder initialized rows * K elements.
                let values = unsafe { decoded.values.as_std_slice(dimension(k)) };
                for row in 0..rows {
                    // SAFETY: The decoder retained one metadata entry per row.
                    let meta = unsafe { decoded.meta.add(Elements::new(row)).as_unit().as_ref() };
                    assert_eq!(meta.a, (row % 4 + 1) as f32 * 0.25);
                    assert_eq!(meta.b, (row % 3) as f32 - 1.0);
                    assert_eq!(meta.n, (row % 5) as f32 * 0.5);
                    let mut expected = vec![0; k];
                    for d in 0..dim {
                        let p = d / 64 * 64 + d % 2 * 32 + d % 64 / 2;
                        expected[p] = ((row * 7 + d * 3 + 1) % 16) as u8;
                    }
                    assert_eq!(&values[row * k..(row + 1) * k], expected);
                }
            }
        }
    }

    #[test]
    fn decoded_b_groups_and_metadata() {
        check_decoded_b(Scalar::new());
    }

    fn documents(rows: usize, dim: usize) -> Matrix<u8> {
        let stride = Data::<4>::canonical_bytes(dim);
        let mut bytes = Matrix::new(0_u8, rows, stride);
        for (i, bytes) in bytes.as_mut_slice().chunks_exact_mut(stride).enumerate() {
            let mut row = DataMutRef::<4>::from_canonical_front_mut(bytes, dim).unwrap();
            row.set_meta(MinMaxCompensation {
                a: (i % 4 + 1) as f32 * 0.25,
                b: (i % 3) as f32 - 1.0,
                n: (i % 5) as f32 * 0.5,
                dim: dim as u32,
                ..Default::default()
            });
            for d in 0..dim {
                row.vector_mut()
                    .set(d, ((i * 7 + d * 3 + 1) % 16) as i64)
                    .unwrap();
            }
        }
        bytes
    }

    #[test]
    fn invalid_driver_bounds_are_detected() {
        let values = PackedQuery::<8, 4>::new(8, 9);
        let docs = documents(2, 9);
        for (dim, metadata_rows, output_rows) in [(8, 1, 8), (9, 0, 8), (9, 1, 0), (9, 1, 9)] {
            let _ = panic_message_for(|| {
                let metadata = vec![QueryCompensation::default(); metadata_rows];
                let mut scores = vec![0.0; output_rows];
                let input = if dim == 9 { &docs } else { &documents(2, dim) };
                let _ = Driver::<_, 4, 8, 6>::new(
                    Scalar::new(),
                    values.as_view().unwrap(),
                    &metadata,
                    canonical(input, dim),
                    &mut scores,
                    Cache::detect(),
                );
            });
        }
    }

    #[test]
    fn decoded_b_rejects_inexact_scratch_lengths() {
        let dim = 9;
        let layout = EvenOdd64Layout::new(dim);
        let docs = documents(2, dim);
        for (values_len, meta_len, expected) in [
            (0, 2, "decoded B value scratch"),
            (127, 2, "decoded B value scratch"),
            (129, 2, "decoded B value scratch"),
            (128, 0, "decoded B metadata scratch"),
            (128, 1, "decoded B metadata scratch"),
            (128, 3, "decoded B metadata scratch"),
        ] {
            let message = panic_message_for(|| {
                let mut values = vec![0; values_len];
                let mut metadata = vec![MinMaxCompensation::default(); meta_len];
                let _ =
                    canonical(&docs, dim).decode(Scalar::new(), layout, &mut values, &mut metadata);
            });
            assert!(message.contains(expected), "{message}");
        }
    }

    fn check_driver_case<A, const PACK: usize, const MR: usize, const NR: usize>(
        arch: A,
        rows: usize,
        cols: usize,
        dim: usize,
        a_panels_per_tile: usize,
        b_rows_per_tile: usize,
    ) where
        A: Decoder + ExtraWide<PACK, MR> + util::LoadStore<f32, MR>,
        for<'a> Driver<'a, A, PACK, MR, NR>: driver::Drive,
    {
        let k = EvenOdd64Layout::new(dim).padded();
        let mut a = PackedQuery::<MR, PACK>::new(rows, dim);
        let mut query = vec![QueryCompensation::<MR>::default(); rows.div_ceil(MR)];
        for row in 0..rows {
            let values: Vec<_> = (0..dim)
                .map(|d| ((row * 17 + d * 3 + 1) % 256) as u8)
                .collect();
            a.set_row(row, &values);
            let q = &mut query[row / MR];
            q.scale[row % MR] = (row % 3 + 1) as f32 * 0.5;
            q.bias[row % MR] = (row % 5) as f32 - 2.0;
            q.scaled_sum[row % MR] = (row % 7) as f32;
        }
        let b = documents(cols, dim);
        let expected: Vec<f32> = (0..rows)
            .map(|i| {
                let q = &query[i / MR];
                let lane = i % MR;
                let mut best = f32::MAX;
                for j in 0..cols {
                    let doc = DataRef::<4>::from_canonical_front(b.row(j), dim)
                        .unwrap()
                        .meta();
                    let raw: u32 = (0..dim)
                        .map(|d| {
                            ((i * 17 + d * 3 + 1) % 256) as u32 * ((j * 7 + d * 3 + 1) % 16) as u32
                        })
                        .sum();
                    let similarity = q.scale[lane] * doc.a * raw as f32
                        + q.scaled_sum[lane] * doc.b
                        + doc.n * q.bias[lane]
                        + q.bias[lane] * doc.b * dim as f32;
                    best = best.min(-similarity);
                }
                best
            })
            .collect();
        let mut scores = vec![12345.0; rows + 2];
        let a = a.as_view().unwrap();
        let b = canonical(&b, dim);
        let panel_bytes = a.values().block_stride(dimension(k)).bytes().value()
            + std::mem::size_of::<QueryCompensation<MR>>();
        let b_row_bytes = k + std::mem::size_of::<MinMaxCompensation>();
        let cache = Cache::new(
            value_or_one(panel_bytes + b_rows_per_tile * b_row_bytes),
            value_or_one(panel_bytes * a_panels_per_tile),
        );
        let mut driver =
            Driver::<_, PACK, MR, NR>::new(arch, a, &query, b, &mut scores[1..rows + 1], cache);
        driver::Drive::drive(&mut driver);
        driver::Drive::drive(&mut driver);
        assert_eq!(scores[0], 12345.0);
        assert_eq!(scores[rows + 1], 12345.0);
        assert_eq!(
            &scores[1..rows + 1],
            expected,
            "({rows},{cols},{dim}), A tile={a_panels_per_tile}, B tile={b_rows_per_tile}"
        );
    }

    fn check_driver<A, const PACK: usize, const MR: usize, const NR: usize>(arch: A)
    where
        A: Decoder + ExtraWide<PACK, MR> + util::LoadStore<f32, MR>,
        for<'a> Driver<'a, A, PACK, MR, NR>: driver::Drive,
    {
        let cases = crate::matrix_kernels::maxsim::test::packed_x_unpacked_test_dims(MR, NR)
            .into_iter()
            .map(|c| {
                (
                    c.total_a_rows,
                    c.total_b_cols,
                    c.k,
                    c.a_panels_per_tile,
                    c.b_cols_per_tile,
                )
            })
            .chain((1..=NR).flat_map(|n| (1..=17).map(move |dim| (MR + 1, n + NR, dim, 1, NR))))
            .chain([
                (MR + 1, 32, 1, 1, NR),
                (MR + 1, 33, 1, 1, NR),
                (MR + 1, 16, 256, 1, NR),
                (MR + 1, 17, 256, 1, NR),
                (MR + 1, 16, 257, 1, NR),
                (MR + 1, 31, 250, 1, NR),
                (MR + 1, 32, 250, 1, NR),
                (MR + 1, 33, 250, 1, NR),
                (MR + 1, 2 * NR + 1, 1024, 1, NR),
                (MR + 1, 2 * NR + 1, 1025, 1, NR),
                (MR + 1, 2 * NR + 1, 2048, 1, NR),
                (MR + 1, 2 * NR + 1, 2049, 1, NR),
            ]);
        for (rows, cols, dim, a_panels_per_tile, b_rows_per_tile) in cases {
            if cfg!(miri)
                && !(rows <= MR + 1 && cols <= NR + 1 && matches!(dim, 1 | 9)
                    || dim == 1 && matches!(cols, 32 | 33))
            {
                continue;
            }
            check_driver_case::<_, PACK, MR, NR>(
                arch,
                rows,
                cols,
                dim,
                a_panels_per_tile,
                b_rows_per_tile,
            );
        }

        for rows in [1, MR, MR + 1, 2 * MR + 1] {
            for cols in [1, NR + 1, 2 * NR + 1, 33] {
                for dim in [1, 9, 250, 256] {
                    if cfg!(miri) && !(rows == MR + 1 && cols == NR + 1 && dim == 9) {
                        continue;
                    }
                    for a_panels_per_tile in [1, 2] {
                        check_driver_case::<_, PACK, MR, NR>(
                            arch,
                            rows,
                            cols,
                            dim,
                            a_panels_per_tile,
                            NR,
                        );
                    }
                }
            }
        }
    }

    fn check_registers<A, const PACK: usize, const MR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<PACK, MR>,
    {
        arch.run_inline(|| {
            for rows in 1..=MR {
                for pattern in 0..3 {
                    let b = core::array::from_fn(|lane| match pattern {
                        0 => 0,
                        1 => 15,
                        _ => [1, 7, 3, 15, 2, 9, 4, 12][lane],
                    });
                    let values: [[u8; PACK]; MR] = core::array::from_fn(|i| {
                        core::array::from_fn(|j| {
                            if i.is_multiple_of(2) {
                                255
                            } else {
                                (i * 11 + j * 13) as u8
                            }
                        })
                    });
                    // SAFETY: The span has exactly MR groups and rows <= MR.
                    let a = unsafe { arch.load(Slice::new(values.as_flattened()), rows) };
                    let mut acc = arch.zero();
                    for _ in 0..3 {
                        acc = arch.dot(a, arch.splat(b), acc);
                    }
                    let query = QueryCompensation {
                        scale: [0.5; MR],
                        bias: [-2.0; MR],
                        scaled_sum: [7.0; MR],
                    };
                    let doc = MinMaxCompensation {
                        a: 0.25,
                        b: 3.0,
                        n: 11.0,
                        ..Default::default()
                    };
                    let mut scores = [f32::MAX; MR];
                    let group_dim = PACK;
                    let reduce = |doc: MinMaxCompensation, scores: &mut [f32; MR]| {
                        // SAFETY: One accumulator has exactly one metadata entry.
                        unsafe { arch.reduce([acc], &query, Slice::new(&[doc]), group_dim, scores) }
                    };
                    reduce(doc, &mut scores);
                    for i in 0..rows {
                        let raw: u32 = values[i]
                            .iter()
                            .zip(b.iter())
                            .map(|(&a, &b)| u32::from(a) * u32::from(b))
                            .sum::<u32>()
                            * 3;
                        let expected = -(0.5 * 0.25 * raw as f32
                            + 7.0 * 3.0
                            + 11.0 * -2.0
                            + -2.0 * 3.0 * group_dim as f32);
                        assert_eq!(scores[i], expected);
                    }
                    let previous = scores;
                    let nan = MinMaxCompensation { a: f32::NAN, ..doc };
                    reduce(nan, &mut scores);
                    assert_eq!(
                        scores, previous,
                        "NaN compensation must not discard prior scores"
                    );
                }
            }
        });
    }

    fn check_contraction<
        A: Architecture + ExtraWide<PACK, MR>,
        const PACK: usize,
        const MR: usize,
    >(
        arch: A,
    ) {
        arch.run_inline(|| {
            if !cfg!(miri) {
                let values = vec![255; MR * PACK];
                // SAFETY: The span contains exactly MR * PACK bytes.
                let a = unsafe { arch.load(Slice::new(&values), MR) };
                let b = arch.splat([15; PACK]);
                let mut acc = arch.zero();
                for iteration in 1..=300_000 {
                    acc = arch.dot(a, b, acc);
                    if matches!(iteration, 140_000 | 200_000 | 300_000) {
                        let expected = (iteration as u64 * PACK as u64 * 255 * 15) as u32;
                        assert_eq!(arch.accumulator_lanes(acc), [expected; MR]);
                        let mut scores = [f32::MAX; MR];
                        let query = QueryCompensation {
                            scale: [1.0; MR],
                            ..Default::default()
                        };
                        let docs = [MinMaxCompensation {
                            a: 1.0,
                            ..Default::default()
                        }];
                        // SAFETY: The single accumulator has one metadata entry.
                        unsafe { arch.reduce([acc], &query, Slice::new(&docs), 1, &mut scores) };
                        assert_eq!(scores, [-(expected as f32); MR]);
                    }
                }
            }
            for groups in [1, 2, 3, 8, 17, 129] {
                let k = groups * PACK;
                let a_value = |row: usize, d: usize| ((row * 17 + d * 23 + 255) % 256) as u8;
                let b_value = |row: usize, d: usize| ((row * 7 + d * 3 + 15) % 16) as u8;
                // Deliberately construct panels without query storage, the layout
                // mapping, decoder, canonical reader, quantizer, or compensation.
                let mut a = Vec::new();
                for group in 0..groups {
                    for row in 0..MR {
                        for lane in 0..PACK {
                            a.push(a_value(row, group * PACK + lane));
                        }
                    }
                }
                let b: Vec<_> = (0..3)
                    .flat_map(|row| (0..k).map(move |d| b_value(row, d)))
                    .collect();
                for rows in 1..=MR {
                    // SAFETY: The manually packed A and row-major B have K columns,
                    // PACK divides K, and all B values fit in four bits.
                    let acc = unsafe {
                        arch.contract(
                            packed::Panel::<_, MR, PACK>::new(Slice::new(&a), dimension(k)),
                            unpacked::Panel::<_, 3>::new(Slice::new(&b), dimension(k)),
                            dimension(k),
                            rows,
                        )
                    };
                    for (doc, acc) in acc.into_iter().enumerate() {
                        let lanes = arch.accumulator_lanes(acc);
                        for (row, &actual) in lanes.iter().take(rows).enumerate() {
                            let expected = (0..k)
                                .map(|d| u32::from(a_value(row, d)) * u32::from(b_value(doc, d)))
                                .sum::<u32>();
                            assert_eq!(actual, expected, "groups={groups}, row={row}, doc={doc}");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn scalar_driver_and_registers() {
        check_driver::<_, 4, 8, 6>(Scalar::new());
        check_registers::<_, 4, 8>(Scalar::new());
        check_contraction::<_, 4, 8>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check_driver::<_, 4, 16, 8>(arch);
            check_registers::<_, 4, 16>(arch);
            check_contraction::<_, 4, 16>(arch);
            check_decoded_b(arch);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            arch.run_inline(|| {
                check_decoded_b(arch);
            });
            check_driver::<_, 8, 16, 8>(arch);
            check_registers::<_, 8, 16>(arch);
            check_contraction::<_, 8, 16>(arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_driver::<_, 4, 8, 8>(arch);
            check_registers::<_, 4, 8>(arch);
            check_contraction::<_, 4, 8>(arch);
            check_decoded_b(arch);
        }
    }
}
