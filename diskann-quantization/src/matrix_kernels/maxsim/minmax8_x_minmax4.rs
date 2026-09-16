// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MinMax8 by MinMax4 MaxSim over the existing packed/unpacked panel views.
//!
//! A is block-transposed in architecture-specific contraction groups. The driver expands
//! each canonical MinMax4 B tile once, then reuses it across A's L2 subviews and panels.
//! Panel and micro-kernels only consume grouped values, never canonical packed bytes.
//! Only the original dimension participates in compensation; padded groups are zero.
//! The driver borrows all inputs and owns only temporary document scratch.
//! Scalar, V3, and Neon use four-byte even/odd groups. V4 uses contiguous eight-byte
//! groups so one broadcast feeds two adjacent VNNI lanes per query row.

use diskann_wide::{Architecture, SIMDMinMax, SIMDVector, arch::Scalar};

use crate::{
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        bounds::{self, Bound},
        driver,
        num::{Bytes, DimK, Elements, value_or_one},
        ptr::{MutSlice, Slice},
        util,
    },
    minmax::{Data, DataRef, MinMaxCompensation},
};

use super::packed_f32_x_unpacked_f32::Params as CacheParams;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Grouped<const N: usize>(pub(crate) [u8; N]);

impl<const N: usize> Default for Grouped<N> {
    fn default() -> Self {
        Self([0; N])
    }
}

impl<const N: usize> Grouped<N> {
    pub(crate) fn count(dim: usize) -> usize {
        const { assert!(N == 4 || N == 8) };
        dim.div_ceil(8) * (8 / N)
    }

    pub(crate) fn from_query(values: &[u8], group: usize) -> Self {
        assert!(group < Self::count(values.len()));
        Self(core::array::from_fn(|lane| {
            let dim = if N == 4 {
                group / 2 * 8 + 2 * lane + group % 2
            } else {
                group * 8 + lane
            };
            values.get(dim).copied().unwrap_or(0)
        }))
    }
}

impl<const N: usize> std::ops::Deref for Grouped<N> {
    type Target = [u8; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize> std::ops::DerefMut for Grouped<N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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

/// `k` counts contraction groups, not original dimensions.
pub(crate) struct Driver<'a, A, const N: usize, const MR: usize, const NR: usize> {
    arch: A,

    a: packed::View<'a, Grouped<N>, MR>,
    a_meta: &'a [QueryCompensation<MR>],

    b: unpacked::View<'a, u8>,
    b_values: Vec<Grouped<N>>,
    b_metadata: Vec<MinMaxCompensation>,

    c: &'a mut [f32],
    dim: DimK,
    blocking: CacheParams,
}

impl<'a, A, const N: usize, const MR: usize, const NR: usize> Driver<'a, A, N, MR, NR> {
    /// # Safety
    ///
    /// * A uses the `Grouped<N>` packing for `dim`, including zero padding.
    /// * `a_meta.len() == a.blocks()` and `c.len().div_ceil(MR) == a.blocks()`.
    /// * B contains canonical MinMax4 rows of dimension `dim`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, Grouped<N>, MR>,
        a_meta: &'a [QueryCompensation<MR>],
        b: unpacked::View<'a, u8>,
        c: &'a mut [f32],
        dim: DimK,
        cache: Cache,
    ) -> Self {
        const { assert!(NR > 0) };
        let k = DimK::new(value_or_one(Grouped::<N>::count(dim.value().get())));
        let b_stride = DimK::new(value_or_one(Data::<4>::canonical_bytes(dim.value().get())));
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), b_stride);
        bounds::check_eq!(Bound::new(a_meta.len()), a.blocks());
        bounds::check_eq!(Bound::new(c.len().div_ceil(MR)), a.blocks());
        let blocking = CacheParams::new(
            cache,
            Bytes::new(
                a.block_stride(k).bytes().value() + std::mem::size_of::<QueryCompensation<MR>>(),
            ),
            Bytes::new(
                Elements::<Grouped<N>>::new(k.value().get()).bytes().value()
                    + std::mem::size_of::<MinMaxCompensation>(),
            ),
            NR,
        );
        let b_rows = b.extent().min(blocking.b_cols_in_l1).get();
        Self {
            arch,
            a,
            a_meta,
            b,
            b_values: vec![Grouped::default(); b_rows * k.value().get()],
            b_metadata: vec![MinMaxCompensation::default(); b_rows],
            c,
            dim,
            blocking,
        }
    }
}

impl<A, const N: usize, const MR: usize, const NR: usize> driver::Drive for Driver<'_, A, N, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR> + ExtraWide<N, MR>,
    for<'a> PanelKernel<'a, A, N, MR, NR>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                let dim = self.dim;
                let k = DimK::new(value_or_one(Grouped::<N>::count(dim.value().get())));
                let b_stride =
                    DimK::new(value_or_one(Data::<4>::canonical_bytes(dim.value().get())));
                let b_rows_per_tile = self.b.extent().min(self.blocking.b_cols_in_l1);
                self.c.fill(f32::MAX);
                let output_rows = self.c.len();
                let mut c = MutSlice::new(self.c);
                let a_meta = Slice::new(self.a_meta);
                let on_b_tile = |b_tile: unpacked::View<'_, u8>, _| {
                    let rows = b_tile.extent().get();
                    // SAFETY: B's stride is validated; each tile fits the allocated scratch.
                    let decoded = unsafe {
                        BTile::decode(
                            b_tile,
                            b_stride,
                            dim,
                            k,
                            &mut self.b_values[..rows * k.value().get()],
                            &mut self.b_metadata[..rows],
                        )
                    };
                    let on_a_tile = |a_tile: packed::View<'_, Grouped<N>, MR>,
                                     a_block_base: usize| {
                        let on_a_panel =
                            |a: packed::Panel<'_, Grouped<N>, MR>, a_block_offset: usize| {
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
                        // SAFETY: The A subview inherits the validated grouped dimension.
                        unsafe { a_tile.visit_panels(k, on_a_panel) };
                    };
                    // SAFETY: A was validated against k; subviews retain its bounds.
                    unsafe {
                        self.a
                            .visit_sub_views(self.blocking.a_panels_in_l2, k, on_a_tile)
                    };
                };
                // SAFETY: B was validated against its canonical row stride.
                unsafe { self.b.visit_sub_views(b_rows_per_tile, b_stride, on_b_tile) };
            },
        );
    }
}

struct PanelKernel<'a, A, const N: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, Grouped<N>, MR>,
    query: &'a QueryCompensation<MR>,
    b: BTile<'a, N>,
    c: [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<'a, A, const N: usize, const MR: usize, const NR: usize> PanelKernel<'a, A, N, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR>,
{
    /// # Safety
    ///
    /// A and B have k grouped columns for dim. The output occupies at most MR query rows.
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, Grouped<N>, MR>,
        query: &'a QueryCompensation<MR>,
        b: BTile<'a, N>,
        c: &[f32],
        k: DimK,
        dim: DimK,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.values.k(), k);
        bounds::check_eq!(Bound::new(Grouped::<N>::count(dim.value().get())), k);
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

impl<A, const N: usize, const MR: usize, const NR: usize, const EXTENT: usize>
    unpacked::PanelVisitor<Grouped<N>, EXTENT> for &mut PanelKernel<'_, A, N, MR, NR>
where
    A: Architecture + ExtraWide<N, MR>,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, Grouped<N>, EXTENT>, start: usize) {
        // SAFETY: The visitor receives complete rows from the validated source.
        let b_meta = unsafe {
            self.b
                .meta
                .add(Elements::new(start))
                .truncate(Elements::new(EXTENT))
        };
        let mut micro = MicroKernel::<_, N, MR, EXTENT> {
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
    ($arch:ty, $n:literal, $mr:literal, $nr:literal, [$($tail:literal),+]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $n, $mr, $nr>
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

#[derive(Clone, Copy)]
struct BTile<'a, const N: usize> {
    values: unpacked::View<'a, Grouped<N>>,
    meta: Slice<'a, MinMaxCompensation>,
}

impl<'a, const N: usize> BTile<'a, N> {
    /// # Safety
    ///
    /// B contains canonical MinMax4 rows of dimension dim and stride b_stride.
    /// k equals `Grouped<N>::count(dim)`, and the backend supports this grouping.
    unsafe fn decode(
        b: unpacked::View<'_, u8>,
        b_stride: DimK,
        dim: DimK,
        k: DimK,
        values: &'a mut [Grouped<N>],
        meta: &'a mut [MinMaxCompensation],
    ) -> Self {
        let rows = b.extent();
        let groups = k.value().get();
        let dim = dim.value().get();
        assert_eq!(
            values.len(),
            rows.get() * groups,
            "decoded B value scratch must contain exactly rows * groups elements",
        );
        assert_eq!(
            meta.len(),
            rows.get(),
            "decoded B metadata scratch must contain exactly one entry per row",
        );
        bounds::check_eq!(Bound::new(groups), Grouped::<N>::count(dim));
        // SAFETY: The caller supplies the validated canonical stride.
        let bytes = unsafe { b.as_std_slice(b_stride) };
        for ((row, output), meta) in bytes
            .chunks_exact(b_stride.value().get())
            .zip(values.chunks_exact_mut(groups))
            .zip(meta.iter_mut())
        {
            // SAFETY: Each row has the canonical stride and dimension.
            let data = unsafe { DataRef::<4>::from_canonical_unchecked(row, dim) };
            *meta = data.meta();
            let vector = data.vector();
            // SAFETY: The vector retains exactly ceil(dim / 2) packed bytes.
            let source = unsafe {
                Slice::from_raw(
                    std::ptr::NonNull::new_unchecked(vector.as_ptr().cast_mut()),
                    Bound::new(dim.div_ceil(2)),
                )
            };
            // SAFETY: The canonical source and output cover exactly one grouped row.
            unsafe { unpack_minmax4_row(source, dim, output) };
        }
        Self {
            // SAFETY: values contains exactly rows * k initialized groups.
            values: unsafe { unpacked::View::new(Slice::new(values), rows, k) },
            meta: Slice::new(meta),
        }
    }
}

/// Unpack a MinMax4 row into the kernel's grouped byte layout.
///
/// # Safety
///
/// `values` contains `ceil(dim / 2)` bytes and `output` contains `Grouped::<N>::count(dim)`
/// groups. On x86-64, `N == 8` requires BMI2.
#[inline(always)]
unsafe fn unpack_minmax4_row<const N: usize>(
    values: Slice<'_, u8>,
    dim: usize,
    output: &mut [Grouped<N>],
) {
    bounds::check_eq!(values.len(), dim.div_ceil(2));
    bounds::check_eq!(Bound::new(output.len()), Grouped::<N>::count(dim));
    let mut tail_start = 0;
    if N == 4 {
        let blocks = dim / 8;
        // SAFETY: Each complete block contains four packed bytes.
        let packed = unsafe {
            values
                .truncate(Elements::new(blocks * 4))
                .as_std_slice(blocks * 4)
        };
        for (source, groups) in packed.chunks_exact(4).zip(output.chunks_exact_mut(2)) {
            let source = u32::from_le_bytes([source[0], source[1], source[2], source[3]]);
            let low = (source & 0x0f0f_0f0f).to_le_bytes();
            let high = ((source >> 4) & 0x0f0f_0f0f).to_le_bytes();
            groups[0] = Grouped(core::array::from_fn(|i| low[i]));
            groups[1] = Grouped(core::array::from_fn(|i| high[i]));
        }
        tail_start = blocks * 2;
    }
    for (group, output) in output.iter_mut().enumerate().skip(tail_start) {
        // SAFETY: Every remaining group lies within the complete packed row.
        *output = unsafe { unpack_minmax4_group(values, group, dim) };
    }
}

/// Unpack one group, including a final incomplete group, from a MinMax4 row.
///
/// # Safety
///
/// `values` contains exactly `dim.div_ceil(2)` bytes and
/// `group < Grouped::<N>::count(dim)`. On x86-64, `N == 8` requires BMI2.
#[inline(always)]
unsafe fn unpack_minmax4_group<const N: usize>(
    values: Slice<'_, u8>,
    group: usize,
    dim: usize,
) -> Grouped<N> {
    bounds::check_eq!(values.len(), dim.div_ceil(2));
    bounds::check_lt!(Bound::new(group), Grouped::<N>::count(dim));

    let groups_per_block = 8 / N;
    let block = group / groups_per_block;
    let within_block = group % groups_per_block;
    let byte_start = block * 4;
    let dimensions = (dim - block * 8).min(8);
    let byte_count = dimensions.div_ceil(2);
    // SAFETY: `group` is within the number of eight-dimensional blocks.
    let packed = unsafe {
        values
            .add(Elements::new(byte_start))
            .truncate(Elements::new(byte_count))
    };

    if dimensions == 8 {
        // SAFETY: A complete block tracks exactly four bytes.
        let source = u32::from_le(unsafe { packed.as_ptr().cast::<u32>().read_unaligned() });
        if N == 4 {
            let expanded = if within_block == 0 {
                (source & 0x0f0f_0f0f).to_le_bytes()
            } else {
                ((source >> 4) & 0x0f0f_0f0f).to_le_bytes()
            };
            return Grouped(core::array::from_fn(|lane| expanded[lane]));
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::_pdep_u64;

            // SAFETY: Grouped<8> is only used by V4, which provides BMI2.
            let expanded =
                unsafe { _pdep_u64(u64::from(source), 0x0f0f_0f0f_0f0f_0f0f) }.to_le_bytes();
            return Grouped(core::array::from_fn(|lane| expanded[lane]));
        }
    }

    // SAFETY: `packed` tracks exactly the bytes in the final partial block.
    let packed = unsafe { packed.as_std_slice(byte_count) };
    Grouped(core::array::from_fn(|lane| {
        let dimension = if N == 4 {
            2 * lane + within_block
        } else {
            lane
        };
        if dimension >= dimensions {
            0
        } else {
            (packed[dimension / 2] >> (4 * (dimension % 2))) & 15
        }
    }))
}

struct MicroKernel<'a, A, const N: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, Grouped<N>, MR>,
    query: &'a QueryCompensation<MR>,
    b: unpacked::Panel<'a, Grouped<N>, NR>,
    b_meta: Slice<'a, MinMaxCompensation>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<A, const N: usize, const MR: usize, const NR: usize> driver::MicroKernel
    for MicroKernel<'_, A, N, MR, NR>
where
    A: Architecture + ExtraWide<N, MR>,
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        self.arch.run_inline(
            #[inline]
            || {
                bounds::check_eq!(self.a.k(), self.k);
                // SAFETY: The panel constructor establishes the common contraction
                // dimension and the number of valid query rows.
                let acc = unsafe {
                    self.arch
                        .contract(self.a, self.b, self.k, self.dim, self.valid_rows)
                };
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
trait ExtraWide<const N: usize, const MR: usize>: Copy {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// # Safety
    ///
    /// `values` contains exactly MR groups and `valid_rows <= MR`.
    unsafe fn load(self, values: Slice<'_, Grouped<N>>, valid_rows: usize) -> Self::Query;
    fn zero(self) -> Self::Accumulator;
    fn splat(self, value: Grouped<N>) -> Self::Splat;

    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator;

    /// # Safety
    ///
    /// Both panels have contraction dimension `k` and `valid_rows <= MR`.
    #[inline(always)]
    unsafe fn contract<const NR: usize>(
        self,
        a: packed::Panel<'_, Grouped<N>, MR>,
        b: unpacked::Panel<'_, Grouped<N>, NR>,
        k: DimK,
        dim: DimK,
        valid_rows: usize,
    ) -> [Self::Accumulator; NR] {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_eq!(Bound::new(Grouped::<N>::count(dim.value().get())), k);
        bounds::check_le!(Bound::new(valid_rows), MR);
        let mut acc = [self.zero(); NR];
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let b_stride = b.stride(k);
        for i in 0..k.value().get() {
            // SAFETY: A has k complete MR-element groups.
            let a = unsafe {
                self.load(
                    ap.add(Elements::new(i * MR)).truncate(Elements::new(MR)),
                    valid_rows,
                )
            };
            for (j, acc) in acc.iter_mut().enumerate() {
                // SAFETY: The B panel contains one complete packed row per accumulator.
                let b = unsafe { *bp.add(b_stride * j + Elements::new(i)).as_unit().as_ref() };
                *acc = self.dot(a, self.splat(b), *acc);
            }
        }
        acc
    }

    /// # Safety
    ///
    /// `docs` contains exactly NR metadata entries, one per accumulator.
    unsafe fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<MR>,
        docs: Slice<'_, MinMaxCompensation>,
        dim: DimK,
        scores: &mut [f32; MR],
    );
}

impl ExtraWide<4, 8> for Scalar {
    type Query = [[u32; 8]; 4];
    type Splat = [u32; 4];
    type Accumulator = [u32; 8];

    #[inline(always)]
    unsafe fn load(self, values: Slice<'_, Grouped<4>>, _: usize) -> Self::Query {
        // SAFETY: The trait contract requires exactly eight initialized groups.
        let values = unsafe { values.as_std_slice(8) };
        core::array::from_fn(|d| core::array::from_fn(|row| u32::from(values[row][d])))
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn splat(self, value: Grouped<4>) -> Self::Splat {
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
        dim: DimK,
        scores: &mut [f32; 8],
    ) {
        // SAFETY: The trait contract requires exactly NR metadata entries.
        let docs = unsafe { docs.as_std_slice(NR) };
        for (acc, doc) in acc.into_iter().zip(docs) {
            for i in 0..8 {
                let similarity = query.scale[i] * doc.a * acc[i] as f32
                    + query.scaled_sum[i] * doc.b
                    + doc.n * query.bias[i]
                    + query.bias[i] * doc.b * dim.value().get() as f32;
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
                            * <$float>::splat($arch, $dim.value().get() as f32);
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

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<4>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 16);
            // SAFETY: Each half contains eight groups, or 32 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(8)).as_ptr().cast());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(8))
                            .truncate(Elements::new(8))
                            .as_ptr()
                            .cast(),
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
        fn splat(self, value: Grouped<4>) -> Self::Splat {
            diskann_wide::alias!(u32s = <V3>::u32x8);
            Self::Splat::from_underlying(
                self,
                u32s::splat(self, u32::from_le_bytes(*value)).to_underlying(),
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
            dim: DimK,
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

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<8>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 16);
            // SAFETY: Each half contains eight groups, or 64 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(8)).as_ptr().cast());
                let hi = (rows > 8).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(8))
                            .truncate(Elements::new(8))
                            .as_ptr()
                            .cast(),
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
        fn splat(self, value: Grouped<8>) -> Self::Splat {
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
                    u64s::splat(self, u64::from_le_bytes(*value)).to_underlying(),
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
            dim: DimK,
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

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Grouped<4>>, rows: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 8);
            // SAFETY: Each half contains four groups, or 16 bytes.
            unsafe {
                let lo =
                    SIMDVector::load_simd(self, values.truncate(Elements::new(4)).as_ptr().cast());
                let hi = (rows > 4).then(|| {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(4))
                            .truncate(Elements::new(4))
                            .as_ptr()
                            .cast(),
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
        fn splat(self, value: Grouped<4>) -> Self::Splat {
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
            dim: DimK,
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

    use super::*;
    use crate::{
        matrix_kernels::test_util::panic_message_for, minmax::DataMutRef,
        multi_vector::BlockTransposed,
    };

    #[test]
    fn even_odd_groups() {
        for dim in 1..=8 {
            let values: Vec<_> = (0..dim as u8).map(|i| i + 1).collect();
            for group in 0..Grouped::<4>::count(dim) {
                let got = Grouped::<4>::from_query(&values, group);
                let parity = group % 2;
                for i in 0..4 {
                    assert_eq!(got[i], values.get(2 * i + parity).copied().unwrap_or(0));
                }
            }
        }
    }

    fn dimension(value: usize) -> DimK {
        DimK::new(NonZeroUsize::new(value).unwrap())
    }

    fn check_packed_groups<const N: usize>() {
        let bytes = [0xff, 0x21, 0x43, 0x65, 0x87, 0xff];
        for dim in 1_usize..=8 {
            let packed = Slice::new(&bytes[1..1 + dim.div_ceil(2)]);
            let mut decoded = vec![Grouped::<N>::default(); Grouped::<N>::count(dim)];
            // SAFETY: Input and output have the exact row lengths. N=8 is called
            // in a V4 context, including BMI2 under Miri.
            unsafe { unpack_minmax4_row(packed, dim, &mut decoded) };
            for (group, decoded_group) in decoded.iter().enumerate() {
                // SAFETY: The complete row contains this group.
                let values = unsafe { unpack_minmax4_group::<N>(packed, group, dim) };
                assert_eq!(*decoded_group, values);
                for lane in 0..N {
                    let d = if N == 4 { 2 * lane + group } else { lane };
                    assert_eq!(values[lane], if d < dim { (d + 1) as u8 } else { 0 });
                }
            }
        }
    }

    #[test]
    fn packed_groups_ignore_padding_nibbles() {
        check_packed_groups::<4>();
    }

    fn check_decoded_b<const N: usize>() {
        for dim in (1..=33).chain([249, 250, 256, 257]) {
            for rows in [1, 7, 8, 9, 31, 32, 33] {
                if cfg!(miri) && !(matches!(dim, 1 | 8 | 9) && rows <= 8) {
                    continue;
                }
                let b = documents(rows, dim);
                let groups = Grouped::<N>::count(dim);
                let mut values = vec![Grouped::<N>::default(); rows * groups];
                let mut meta = vec![MinMaxCompensation::default(); rows];
                // SAFETY: The fixture contains canonical rows and exact scratch extents.
                let decoded = unsafe {
                    BTile::decode(
                        unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                        dimension(Data::<4>::canonical_bytes(dim)),
                        dimension(dim),
                        dimension(groups),
                        &mut values,
                        &mut meta,
                    )
                };
                // SAFETY: The decoder initialized rows * groups elements.
                let values = unsafe { decoded.values.as_std_slice(dimension(groups)) };
                for row in 0..rows {
                    // SAFETY: The decoder retained one metadata entry per row.
                    let meta = unsafe { decoded.meta.add(Elements::new(row)).as_unit().as_ref() };
                    assert_eq!(meta.a, (row % 4 + 1) as f32 * 0.25);
                    assert_eq!(meta.b, (row % 3) as f32 - 1.0);
                    assert_eq!(meta.n, (row % 5) as f32 * 0.5);
                    for group in 0..groups {
                        for (lane, &value) in values[row * groups + group].iter().enumerate() {
                            let d = if N == 4 {
                                group / 2 * 8 + 2 * lane + group % 2
                            } else {
                                group * 8 + lane
                            };
                            let expected = if d < dim {
                                ((row * 7 + d * 3 + 1) % 16) as u8
                            } else {
                                0
                            };
                            assert_eq!(value, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decoded_b_groups_and_metadata() {
        check_decoded_b::<4>();
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
        let values = BlockTransposed::<Grouped<4>, 8>::new(8, 4);
        let docs = documents(2, 9);
        for (dim, metadata_rows, output_rows) in [(8, 1, 8), (9, 0, 8), (9, 1, 0), (9, 1, 9)] {
            let _ = panic_message_for(|| {
                let metadata = vec![QueryCompensation::default(); metadata_rows];
                let mut scores = vec![0.0; output_rows];
                // SAFETY: In test builds the constructor checks every supplied size
                // relationship before creating or dereferencing any derived spans.
                let _ = unsafe {
                    Driver::<_, 4, 8, 6>::new(
                        Scalar::new(),
                        packed::View::from_block_transposed(values.as_view()).unwrap(),
                        &metadata,
                        unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                        &mut scores,
                        dimension(dim),
                        Cache::detect(),
                    )
                };
            });
        }
    }

    #[test]
    fn decoded_b_rejects_inexact_scratch_lengths() {
        let dim = 9;
        let groups = Grouped::<4>::count(dim);
        let docs = documents(2, dim);
        for (values_len, meta_len, expected) in [
            (0, 2, "decoded B value scratch"),
            (7, 2, "decoded B value scratch"),
            (9, 2, "decoded B value scratch"),
            (8, 0, "decoded B metadata scratch"),
            (8, 1, "decoded B metadata scratch"),
            (8, 3, "decoded B metadata scratch"),
        ] {
            let message = panic_message_for(|| {
                let mut values = vec![Grouped::<4>::default(); values_len];
                let mut metadata = vec![MinMaxCompensation::default(); meta_len];
                // SAFETY: The canonical input and group dimension agree. Scratch lengths
                // are checked unconditionally before any derived spans are accessed.
                let _ = unsafe {
                    BTile::decode(
                        unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                        dimension(Data::<4>::canonical_bytes(dim)),
                        dimension(dim),
                        dimension(groups),
                        &mut values,
                        &mut metadata,
                    )
                };
            });
            assert!(message.contains(expected), "{message}");
        }
    }

    fn check_driver_case<A, const N: usize, const MR: usize, const NR: usize>(
        arch: A,
        rows: usize,
        cols: usize,
        dim: usize,
        a_panels_per_tile: usize,
        b_rows_per_tile: usize,
    ) where
        A: Architecture + ExtraWide<N, MR> + util::LoadStore<f32, MR>,
        for<'a> Driver<'a, A, N, MR, NR>: driver::Drive,
    {
        let k = Grouped::<N>::count(dim);
        let mut grouped = Matrix::new(Grouped::<N>::default(), rows, k);
        let mut query = vec![QueryCompensation::<MR>::default(); rows.div_ceil(MR)];
        for row in 0..rows {
            let values: Vec<_> = (0..dim)
                .map(|d| ((row * 17 + d * 3 + 1) % 256) as u8)
                .collect();
            for group in 0..k {
                grouped.as_mut_slice()[row * k + group] = Grouped::<N>::from_query(&values, group);
            }
            let q = &mut query[row / MR];
            q.scale[row % MR] = (row % 3 + 1) as f32 * 0.5;
            q.bias[row % MR] = (row % 5) as f32 - 2.0;
            q.scaled_sum[row % MR] = (row % 7) as f32;
        }
        let a = BlockTransposed::<_, MR>::from_matrix_view(grouped.as_view());
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
        let a = packed::View::from_block_transposed(a.as_view()).unwrap();
        let b = unpacked::View::from_matrix_view(b.as_view()).unwrap();
        let panel_bytes = a.block_stride(dimension(k)).bytes().value()
            + std::mem::size_of::<QueryCompensation<MR>>();
        let b_row_bytes = k * N + std::mem::size_of::<MinMaxCompensation>();
        let cache = Cache::new(
            value_or_one(panel_bytes + b_rows_per_tile * b_row_bytes),
            value_or_one(panel_bytes * a_panels_per_tile),
        );
        // SAFETY: The fixture supplies matching packed groups, metadata and output size.
        let mut driver = unsafe {
            Driver::<_, N, MR, NR>::new(
                arch,
                a,
                &query,
                b,
                &mut scores[1..rows + 1],
                dimension(dim),
                cache,
            )
        };
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

    fn check_driver<A, const N: usize, const MR: usize, const NR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<N, MR> + util::LoadStore<f32, MR>,
        for<'a> Driver<'a, A, N, MR, NR>: driver::Drive,
    {
        let cases = super::super::test::packed_x_unpacked_test_dims(MR, NR)
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
            check_driver_case::<_, N, MR, NR>(
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
                        check_driver_case::<_, N, MR, NR>(
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

    fn check_registers<A, const N: usize, const MR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<N, MR>,
    {
        arch.run_inline(|| {
            for rows in 1..=MR {
                for pattern in 0..3 {
                    let b = Grouped(core::array::from_fn(|lane| match pattern {
                        0 => 0,
                        1 => 15,
                        _ => [1, 7, 3, 15, 2, 9, 4, 12][lane],
                    }));
                    let values: [Grouped<N>; MR] = core::array::from_fn(|i| {
                        Grouped(core::array::from_fn(|j| {
                            if i.is_multiple_of(2) {
                                255
                            } else {
                                (i * 11 + j * 13) as u8
                            }
                        }))
                    });
                    // SAFETY: The span has exactly MR groups and rows <= MR.
                    let a = unsafe { arch.load(Slice::new(&values), rows) };
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
                    let group_dim = N;
                    let reduce = |doc: MinMaxCompensation, scores: &mut [f32; MR]| {
                        // SAFETY: One accumulator has exactly one metadata entry.
                        unsafe {
                            arch.reduce(
                                [acc],
                                &query,
                                Slice::new(&[doc]),
                                dimension(group_dim),
                                scores,
                            )
                        }
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

    #[test]
    fn scalar_driver_and_registers() {
        check_driver::<_, 4, 8, 6>(Scalar::new());
        check_registers::<_, 4, 8>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check_driver::<_, 4, 16, 8>(arch);
            check_registers::<_, 4, 16>(arch);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            arch.run_inline(|| {
                check_packed_groups::<8>();
                check_decoded_b::<8>();
            });
            check_driver::<_, 8, 16, 8>(arch);
            check_registers::<_, 8, 16>(arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_driver::<_, 4, 8, 8>(arch);
            check_registers::<_, 4, 8>(arch);
        }
    }
}
