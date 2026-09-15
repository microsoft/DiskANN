// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! MinMax8 by MinMax4 MaxSim over the existing packed/unpacked panel views.
//!
//! A is block-transposed in four-byte, even/odd contraction groups. Each B cache
//! tile is unpacked once into the same groups, then reused across all A sub-views.
//! Only the original dimension participates in compensation; padded groups are zero.
//! The driver borrows all inputs and owns only the B conversion scratch buffers.
//! Every eight dimensions become `[0, 2, 4, 6]`, then `[1, 3, 5, 7]`, for both operands.

use diskann_wide::{Architecture, SIMDMinMax, SIMDVector, arch::Scalar};

use crate::{
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        bounds::{self, Bound},
        driver,
        num::{DimK, Elements, value_or_one},
        ptr::{MutSlice, Slice},
        util,
    },
    minmax::{Data, DataRef, MinMaxCompensation},
};

use super::packed_f32_x_unpacked_f32::Params;

pub(crate) type Group = [u8; 4];

pub(crate) fn groups(dim: usize) -> usize {
    dim.div_ceil(8) * 2
}

/// Split up to eight bytes into even and odd dimension groups.
///
/// # Panics
///
/// Panics if `values.len() > 8`.
pub(crate) fn split_u8(values: &[u8]) -> [Group; 2] {
    assert!(values.len() <= 8);
    let mut output = [[0; 4]; 2];
    for (i, &value) in values.iter().enumerate() {
        output[i % 2][i / 2] = value;
    }
    output
}

/// Unpack a densely stored unsigned four-bit row into even/odd contraction groups.
///
/// # Panics
///
/// Panics unless `from.len() == dim.div_ceil(2)` and `to.len() == groups(dim)`.
fn unpack_u4(from: &[u8], to: &mut [Group], dim: usize) {
    assert_eq!(from.len(), dim.div_ceil(2));
    assert_eq!(to.len(), groups(dim));

    let mut chunks = from.chunks_exact(4);
    for (input, output) in chunks.by_ref().zip(to.chunks_exact_mut(2)) {
        let packed = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        output[0] = (packed & 0x0f0f_0f0f).to_le_bytes();
        output[1] = ((packed >> 4) & 0x0f0f_0f0f).to_le_bytes();
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut bytes = [0; 4];
        bytes[..tail.len()].copy_from_slice(tail);
        let packed = u32::from_le_bytes(bytes);
        let last = to.len() - 2;
        to[last] = (packed & 0x0f0f_0f0f).to_le_bytes();
        to[last + 1] = ((packed >> 4) & 0x0f0f_0f0f).to_le_bytes();
    }
    if !dim.is_multiple_of(2) {
        // The high nibble of the last byte is padding, not a logical dimension.
        to[groups(dim) - 1][(dim % 8) / 2] = 0;
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

/// `k` counts four-byte contraction groups, not original dimensions.
pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::View<'a, Group, MR>,
    a_meta: &'a [QueryCompensation<MR>],
    b: unpacked::View<'a, u8>,
    c: &'a mut [f32],
    dim: DimK,
    k: DimK,
    b_stride: DimK,
    params: Params,
    b_values: Vec<Group>,
    b_meta: Vec<MinMaxCompensation>,
}

impl<'a, A, const MR: usize, const NR: usize> Driver<'a, A, MR, NR> {
    /// # Safety
    ///
    /// * A uses the even/odd packing for `dim`, including zero padding.
    /// * `a_meta.len() == a.blocks()` and `c.len().div_ceil(MR) == a.blocks()`.
    /// * B contains canonical MinMax4 rows of dimension `dim`.
    pub(crate) unsafe fn new(
        arch: A,
        a: packed::View<'a, Group, MR>,
        a_meta: &'a [QueryCompensation<MR>],
        b: unpacked::View<'a, u8>,
        c: &'a mut [f32],
        dim: DimK,
        cache: Cache,
    ) -> Self {
        let k = DimK::new(value_or_one(groups(dim.value().get())));
        let params = Params::new(
            cache,
            a.block_stride(k).bytes(),
            Elements::<Group>::new(k.value().get()).bytes(),
            NR,
        );
        // SAFETY: Inherited from caller.
        unsafe { Self::new_inner(arch, a, a_meta, b, c, dim, params) }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn new_inner(
        arch: A,
        a: packed::View<'a, Group, MR>,
        a_meta: &'a [QueryCompensation<MR>],
        b: unpacked::View<'a, u8>,
        c: &'a mut [f32],
        dim: DimK,
        params: Params,
    ) -> Self {
        let k = DimK::new(value_or_one(groups(dim.value().get())));
        let b_stride = DimK::new(value_or_one(Data::<4>::canonical_bytes(dim.value().get())));
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), b_stride);
        bounds::check_eq!(Bound::new(a_meta.len()), a.blocks());
        bounds::check_eq!(Bound::new(c.len().div_ceil(MR)), a.blocks());
        let b_rows = params.b_cols_in_l1.min(b.extent()).get();
        Self {
            arch,
            a,
            a_meta,
            b,
            c,
            dim,
            k,
            b_stride,
            params,
            b_values: vec![[0; 4]; b_rows * k.value().get()],
            b_meta: vec![MinMaxCompensation::default(); b_rows],
        }
    }
}

/// Convert a cache tile, retaining exact source and destination bounds.
///
/// # Safety
///
/// * B contains canonical MinMax4 rows of dimension `dim`.
/// * `values.len() == b.extent() * groups(dim)`.
/// * `meta.len() == b.extent()`.
#[inline]
unsafe fn unpack_b(
    b: unpacked::View<'_, u8>,
    values: &mut [Group],
    meta: &mut [MinMaxCompensation],
    dim: DimK,
) {
    let dim = dim.value().get();
    let stride = DimK::new(value_or_one(Data::<4>::canonical_bytes(dim)));
    bounds::check_eq!(b.k(), stride);
    bounds::check_eq!(Bound::new(values.len()), b.extent().get() * groups(dim));
    bounds::check_eq!(Bound::new(meta.len()), b.extent());
    let groups = groups(dim);
    let mut values = MutSlice::new(values);
    let mut meta = MutSlice::new(meta);
    let on_row = |row: unpacked::Panel<'_, u8, 1>, start: usize| {
        // SAFETY: Each visited row has exactly the canonical stride. Both destination
        // spans have one matching entry (or `groups` entries) per B row.
        unsafe {
            let row = row.as_ptr().as_std_slice(stride.value().get());
            let row = DataRef::<4>::from_canonical_unchecked(row, dim);
            *meta.subslice(start, Bound::new(1)).as_array::<1>() = [row.meta()];
            let mut output = values.subslice(start * groups, Bound::new(groups));
            unpack_u4(
                row.vector().as_slice(),
                output.as_std_mut_slice(groups),
                dim,
            );
        }
    };
    // SAFETY: B's tracked row length equals the canonical stride.
    let remainder = unsafe { b.visit_panels::<1>(stride, on_row) };
    debug_assert!(remainder.is_none());
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
                let output_rows = self.c.len();
                let mut c = MutSlice::new(self.c);
                let a_meta = Slice::new(self.a_meta);
                let on_b = |b: unpacked::View<'_, u8>, _| {
                    let b_rows = b.extent();
                    let values = &mut self.b_values[..b_rows.get() * self.k.value().get()];
                    let meta = &mut self.b_meta[..b_rows.get()];
                    // SAFETY: Sub-views preserve the canonical row format and dimension.
                    unsafe { unpack_b(b, values, meta, self.dim) };
                    // SAFETY: Conversion fills exactly `b_rows * k` groups.
                    let b = unsafe { unpacked::View::new(Slice::new(values), b_rows, self.k) };
                    let b_meta = Slice::new(meta);
                    let on_a = |a: packed::View<'_, Group, MR>, block_base: usize| {
                        let on_panel = |a: packed::Panel<'_, Group, MR>, block_offset: usize| {
                            let block = block_base + block_offset;
                            let valid_rows = (output_rows - block * MR).min(MR);
                            // SAFETY: The output occupies exactly the packed A blocks.
                            let mut region =
                                unsafe { c.subslice(block * MR, Bound::new(valid_rows)) };
                            // SAFETY: The region was truncated to `valid_rows` above.
                            let output = unsafe { region.as_std_mut_slice(valid_rows) };
                            let scores = util::LoadStore::<f32, MR>::load(self.arch, output);
                            // SAFETY: Every packed A block has exactly one metadata entry.
                            let query =
                                unsafe { a_meta.add(Elements::new(block)).as_unit().as_ref() };
                            // SAFETY: Visitors preserve `k` and metadata/output spans.
                            let mut panel = unsafe {
                                PanelKernel::new(
                                    self.arch, a, query, b, b_meta, scores, self.k, self.dim,
                                    valid_rows,
                                )
                            };
                            driver::PanelKernel::panel_kernel(&mut panel);
                            util::LoadStore::<f32, MR>::store(self.arch, panel.c, output);
                        };
                        // SAFETY: A sub-views retain the parent's contraction dimension.
                        unsafe { a.visit_panels(self.k, on_panel) };
                    };
                    // SAFETY: A was validated against `k` on construction.
                    unsafe {
                        self.a
                            .visit_sub_views(self.params.a_panels_in_l2, self.k, on_a)
                    };
                };
                // SAFETY: B was validated against the canonical row stride on construction.
                unsafe {
                    self.b
                        .visit_sub_views(self.params.b_cols_in_l1, self.b_stride, on_b)
                };
            },
        );
    }
}

struct PanelKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, Group, MR>,
    query: &'a QueryCompensation<MR>,
    b: unpacked::View<'a, Group>,
    b_meta: Slice<'a, MinMaxCompensation>,
    c: [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<'a, A, const MR: usize, const NR: usize> PanelKernel<'a, A, MR, NR> {
    /// # Safety
    ///
    /// A and B share `k` and the even/odd packing for `dim`. Metadata describes
    /// exactly their rows. Only `valid_rows` query rows are logically present.
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        arch: A,
        a: packed::Panel<'a, Group, MR>,
        query: &'a QueryCompensation<MR>,
        b: unpacked::View<'a, Group>,
        b_meta: Slice<'a, MinMaxCompensation>,
        c: [f32; MR],
        k: DimK,
        dim: DimK,
        valid_rows: usize,
    ) -> Self {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_eq!(b_meta.len(), b.extent());
        bounds::check_eq!(Bound::new(groups(dim.value().get())), k);
        bounds::check_le!(Bound::new(valid_rows), MR);
        Self {
            arch,
            a,
            query,
            b,
            b_meta,
            c,
            k,
            dim,
            valid_rows,
        }
    }
}

struct Visitor<'a, A, const MR: usize> {
    arch: A,
    a: packed::Panel<'a, Group, MR>,
    query: &'a QueryCompensation<MR>,
    b_meta: Slice<'a, MinMaxCompensation>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<A, const MR: usize, const NR: usize> unpacked::PanelVisitor<Group, NR> for Visitor<'_, A, MR>
where
    A: Architecture + ExtraWide<MR>,
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, Group, NR>, start: usize) {
        // SAFETY: The visitor receives a complete panel; metadata follows the same rows.
        let meta = unsafe {
            self.b_meta
                .add(Elements::new(start))
                .truncate(Elements::new(NR))
        };
        let mut micro = MicroKernel {
            arch: self.arch,
            a: self.a,
            query: self.query,
            b,
            b_meta: meta,
            c: self.c,
            k: self.k,
            dim: self.dim,
            valid_rows: self.valid_rows,
        };
        driver::MicroKernel::micro_kernel(&mut micro);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, [$($tail:literal),+]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let visitor = Visitor {
                    arch: self.arch, a: self.a, query: self.query, b_meta: self.b_meta,
                    c: &mut self.c, k: self.k, dim: self.dim, valid_rows: self.valid_rows,
                };
                // SAFETY: The constructor checks B's contraction dimension and metadata.
                let remainder = unsafe { self.b.visit_panels::<$nr>(self.k, visitor) };
                if let Some(remainder) = remainder {
                    $(
                        if let Some(panel) = remainder.try_as_panel::<$tail>() {
                            let mut visitor = Visitor {
                                arch: self.arch, a: self.a, query: self.query, b_meta: self.b_meta,
                                c: &mut self.c, k: self.k, dim: self.dim, valid_rows: self.valid_rows,
                            };
                            unpacked::PanelVisitor::visit(&mut visitor, panel, remainder.start());
                        }
                    )+
                }
            }
        }
    };
}

struct MicroKernel<'a, A, const MR: usize, const NR: usize> {
    arch: A,
    a: packed::Panel<'a, Group, MR>,
    query: &'a QueryCompensation<MR>,
    b: unpacked::Panel<'a, Group, NR>,
    b_meta: Slice<'a, MinMaxCompensation>,
    c: &'a mut [f32; MR],
    k: DimK,
    dim: DimK,
    valid_rows: usize,
}

impl<A, const MR: usize, const NR: usize> driver::MicroKernel for MicroKernel<'_, A, MR, NR>
where
    A: Architecture + ExtraWide<MR>,
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        self.arch.run_inline(
            #[inline]
            || {
                bounds::check_eq!(self.a.k(), self.k);
                bounds::check_eq!(self.b.k(), self.k);
                bounds::check_eq!(self.b_meta.len(), NR);
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
trait ExtraWide<const MR: usize>: Copy {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// # Safety
    ///
    /// `values` contains exactly MR groups and `valid_rows <= MR`.
    unsafe fn load(self, values: Slice<'_, Group>, valid_rows: usize) -> Self::Query;
    fn zero(self) -> Self::Accumulator;
    fn splat(self, value: Group) -> Self::Splat;
    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator;

    /// # Safety
    ///
    /// Both panels have contraction dimension `k` and `valid_rows <= MR`.
    #[inline(always)]
    unsafe fn contract<const NR: usize>(
        self,
        a: packed::Panel<'_, Group, MR>,
        b: unpacked::Panel<'_, Group, NR>,
        k: DimK,
        valid_rows: usize,
    ) -> [Self::Accumulator; NR] {
        bounds::check_eq!(a.k(), k);
        bounds::check_eq!(b.k(), k);
        bounds::check_le!(Bound::new(valid_rows), MR);
        let mut acc = [self.zero(); NR];
        let ap = a.as_ptr();
        let bp = b.as_ptr();
        let stride = b.stride(k);
        for i in 0..k.value().get() {
            // SAFETY: A has k complete MR-element groups.
            let a = unsafe {
                self.load(
                    ap.add(Elements::new(i * MR)).truncate(Elements::new(MR)),
                    valid_rows,
                )
            };
            for (j, acc) in acc.iter_mut().enumerate() {
                // SAFETY: i < k, j < NR, and B has exactly NR * k groups.
                let group = unsafe { *bp.add(stride * j + Elements::new(i)).as_unit().as_ref() };
                *acc = self.dot(a, self.splat(group), *acc);
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

impl ExtraWide<8> for Scalar {
    type Query = [[u32; 8]; 4];
    type Splat = [u32; 4];
    type Accumulator = [u32; 8];

    #[inline(always)]
    unsafe fn load(self, values: Slice<'_, Group>, _: usize) -> Self::Query {
        // SAFETY: The trait contract requires exactly eight initialized groups.
        let values = unsafe { values.as_std_slice(8) };
        core::array::from_fn(|d| core::array::from_fn(|row| u32::from(values[row][d])))
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn splat(self, value: Group) -> Self::Splat {
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

panel_kernel!(Scalar, 8, 6, [1, 2, 3, 4, 5]);

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

    impl ExtraWide<16> for V3 {
        type Query = (
            <V3 as Architecture>::u8x32,
            Option<<V3 as Architecture>::u8x32>,
        );
        type Splat = <V3 as Architecture>::i8x32;
        type Accumulator = [<V3 as Architecture>::i32x8; 2];

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Group>, rows: usize) -> Self::Query {
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
        fn splat(self, value: Group) -> Self::Splat {
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

    impl ExtraWide<16> for V4 {
        type Query = <V4 as Architecture>::u8x64;
        type Splat = <V4 as Architecture>::i8x64;
        type Accumulator = <V4 as Architecture>::i32x16;

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Group>, _: usize) -> Self::Query {
            bounds::check_eq!(values.len(), 16);
            // SAFETY: Sixteen groups contain exactly 64 bytes.
            unsafe { SIMDVector::load_simd(self, values.as_ptr().cast()) }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            SIMDVector::default(self)
        }
        #[inline(always)]
        fn splat(self, value: Group) -> Self::Splat {
            Self::Splat::from_array(self, core::array::from_fn(|i| value[i % 4] as i8))
        }
        #[inline(always)]
        fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator {
            acc.dot_simd(a, b)
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
            diskann_wide::alias!(floats = <V4>::f32x16);
            let convert = |acc: Self::Accumulator, _: usize| {
                floats::from_array(self, acc.to_array().map(|x| x as u32 as f32))
            };
            compensate!(self, floats, 16, 1, acc, convert, query, docs, dim, scores);
        }
    }

    panel_kernel!(V3, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
    panel_kernel!(V4, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::{SIMDDotProduct, arch::aarch64::Neon};

    impl ExtraWide<8> for Neon {
        type Query = (
            <Neon as Architecture>::u8x16,
            Option<<Neon as Architecture>::u8x16>,
        );
        type Splat = <Neon as Architecture>::u8x16;
        type Accumulator = [<Neon as Architecture>::u32x4; 2];

        #[inline(always)]
        unsafe fn load(self, values: Slice<'_, Group>, rows: usize) -> Self::Query {
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
        fn splat(self, value: Group) -> Self::Splat {
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

    panel_kernel!(Neon, 8, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{minmax::DataMutRef, multi_vector::BlockTransposed};
    use diskann_utils::views::Matrix;
    use std::num::NonZeroUsize;

    #[test]
    fn even_odd_groups() {
        for dim in 0..=8 {
            let values: Vec<_> = (0..dim as u8).map(|i| i + 1).collect();
            let got = split_u8(&values);
            for (parity, group) in got.iter().enumerate() {
                for (i, value) in group.iter().enumerate() {
                    assert_eq!(*value, values.get(2 * i + parity).copied().unwrap_or(0));
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_oversized_byte_group() {
        split_u8(&[0; 9]);
    }

    #[test]
    fn unpack_empty_row() {
        unpack_u4(&[], &mut [], 0);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_incorrect_source_length() {
        unpack_u4(&[0; 2], &mut [[0; 4]; 2], 1);
    }

    #[test]
    #[should_panic]
    fn unpack_rejects_incorrect_destination_length() {
        unpack_u4(&[0; 4], &mut [[0; 4]; 1], 8);
    }

    fn dimension(value: usize) -> DimK {
        DimK::new(NonZeroUsize::new(value).unwrap())
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
    fn unpack_all_nibbles_and_tails() {
        for dim in 1_usize..=41 {
            let a: Vec<u8> = (0..dim).map(|i| (i * 17 + 3) as u8).collect();
            let a_grouped: Vec<Group> = (0..groups(dim))
                .map(|group| {
                    core::array::from_fn(|lane| {
                        let d = group / 2 * 8 + 2 * lane + group % 2;
                        a.get(d).copied().unwrap_or(0)
                    })
                })
                .collect();
            for byte in 0..=u8::MAX {
                let packed: Vec<_> = (0..dim.div_ceil(2))
                    .map(|i| byte.wrapping_add((i * 17) as u8))
                    .collect();
                let stride = Data::<4>::canonical_bytes(dim);
                let mut bytes = Matrix::new(byte, 1, stride);
                let metadata = MinMaxCompensation {
                    a: 0.5,
                    b: -2.0,
                    n: 7.0,
                    dim: dim as u32,
                    ..Default::default()
                };
                {
                    let mut row =
                        DataMutRef::<4>::from_canonical_front_mut(bytes.as_mut_slice(), dim)
                            .unwrap();
                    row.set_meta(metadata);
                    for d in 0..dim {
                        row.vector_mut()
                            .set(d, i64::from((packed[d / 2] >> (4 * (d % 2))) & 15))
                            .unwrap();
                    }
                }
                let groups = groups(dim);
                let mut output = vec![[255; 4]; groups + 2];
                let mut meta = [MinMaxCompensation::default()];
                // SAFETY: One canonical row has matching group and metadata destinations.
                unsafe {
                    unpack_b(
                        unpacked::View::from_matrix_view(bytes.as_view()).unwrap(),
                        &mut output[1..groups + 1],
                        &mut meta,
                        dimension(dim),
                    )
                };
                assert_eq!(meta[0], metadata);
                assert_eq!(output[0], [255; 4]);
                assert_eq!(output[groups + 1], [255; 4]);
                let output = &output[1..groups + 1];
                for (group, values) in output.iter().enumerate() {
                    for (i, &value) in values.iter().enumerate() {
                        let d = group / 2 * 8 + 2 * i + group % 2;
                        let expected = if d < dim {
                            (packed[d / 2] >> (4 * (d % 2))) & 15
                        } else {
                            0
                        };
                        assert_eq!(value, expected, "dim={dim}, byte={byte}, d={d}");
                    }
                }
                let got: u32 = a_grouped
                    .iter()
                    .flatten()
                    .zip(output.iter().flatten())
                    .map(|(&a, &b)| u32::from(a) * u32::from(b))
                    .sum();
                let expected: u32 = a
                    .iter()
                    .enumerate()
                    .map(|(i, &a)| u32::from(a) * u32::from((packed[i / 2] >> (4 * (i % 2))) & 15))
                    .sum();
                assert_eq!(got, expected, "permuted dot: dim={dim}, byte={byte}");
            }
        }
    }

    #[test]
    fn document_conversion() {
        for rows in [1, 3, 8, 9] {
            for dim in 1_usize..=33 {
                let groups = groups(dim);
                let bytes = documents(rows, dim);
                let b = unpacked::View::from_matrix_view(bytes.as_view()).unwrap();
                let mut values = vec![[255; 4]; rows * groups];
                let mut meta = vec![MinMaxCompensation::default(); rows];
                // SAFETY: Test fixture contains canonical rows of the given dimension.
                unsafe { unpack_b(b, &mut values, &mut meta, dimension(dim)) };
                for row in 0..rows {
                    let source = DataRef::<4>::from_canonical_front(bytes.row(row), dim).unwrap();
                    assert_eq!(meta[row], source.meta());
                    for (group, values) in values[row * groups..][..groups].iter().enumerate() {
                        for (lane, &value) in values.iter().enumerate() {
                            let d = group / 2 * 8 + 2 * lane + group % 2;
                            assert_eq!(
                                value,
                                if d < dim {
                                    ((row * 7 + d * 3 + 1) % 16) as u8
                                } else {
                                    0
                                }
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn invalid_driver_bounds_are_detected() {
        use crate::matrix_kernels::test_util::panic_message_for;

        let values = BlockTransposed::<Group, 8>::new(8, 4);
        let docs = documents(2, 9);
        for (dim, metadata_rows, output_rows) in [(8, 1, 8), (9, 0, 8), (9, 1, 0), (9, 1, 9)] {
            let _ = panic_message_for(|| {
                let metadata = vec![QueryCompensation::default(); metadata_rows];
                let mut scores = vec![0.0; output_rows];
                // SAFETY: In test builds new_inner checks every supplied size relationship
                // before creating or dereferencing any derived spans.
                let _ = unsafe {
                    Driver::<_, 8, 6>::new_inner(
                        Scalar::new(),
                        packed::View::from_block_transposed(values.as_view()).unwrap(),
                        &metadata,
                        unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                        &mut scores,
                        dimension(dim),
                        Params {
                            a_panels_in_l2: NonZeroUsize::MIN,
                            b_cols_in_l1: NonZeroUsize::MIN,
                        },
                    )
                };
            });
        }
        let _ = panic_message_for(|| {
            let docs = documents(2, 3);
            // SAFETY: The canonical row-length mismatch is checked before any data access.
            unsafe {
                unpack_b(
                    unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                    &mut [[0; 4]; 8],
                    &mut [MinMaxCompensation::default(); 2],
                    dimension(9),
                )
            };
        });
    }

    #[test]
    fn invalid_conversion_output_bounds_are_detected() {
        use crate::matrix_kernels::test_util::panic_message_for;

        let docs = documents(2, 9);
        for (values, metas) in [(7, 2), (9, 2), (8, 1), (8, 3)] {
            let _ = panic_message_for(|| {
                // SAFETY: Test builds check destination sizes before any access.
                unsafe {
                    unpack_b(
                        unpacked::View::from_matrix_view(docs.as_view()).unwrap(),
                        &mut vec![[0; 4]; values],
                        &mut vec![MinMaxCompensation::default(); metas],
                        dimension(9),
                    )
                };
            });
        }
    }

    fn check_driver<A, const MR: usize, const NR: usize>(arch: A)
    where
        A: Architecture,
        for<'a> Driver<'a, A, MR, NR>: driver::Drive,
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
            .chain(
                (1..=NR).flat_map(|n| (1..=17).map(move |dim| (MR + 1, n + NR, dim, 1, NR + 1))),
            );
        for (rows, cols, dim, a_blocks, b_cols) in cases {
            if cfg!(miri) && !(rows <= MR + 1 && cols <= NR + 1 && matches!(dim, 1 | 9)) {
                continue;
            }
            let k = groups(dim);
            let mut grouped = Matrix::new([0_u8; 4], rows, k);
            let mut query = vec![QueryCompensation::<MR>::default(); rows.div_ceil(MR)];
            for row in 0..rows {
                for group in 0..k {
                    grouped.as_mut_slice()[row * k + group] = core::array::from_fn(|lane| {
                        let d = group / 2 * 8 + 2 * lane + group % 2;
                        if d < dim {
                            ((row * 17 + d * 3 + 1) % 256) as u8
                        } else {
                            0
                        }
                    });
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
                                ((i * 17 + d * 3 + 1) % 256) as u32
                                    * ((j * 7 + d * 3 + 1) % 16) as u32
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
            // SAFETY: The fixture supplies matching packed groups, metadata and output size.
            let mut driver = unsafe {
                Driver::<_, MR, NR>::new_inner(
                    arch,
                    packed::View::from_block_transposed(a.as_view()).unwrap(),
                    &query,
                    unpacked::View::from_matrix_view(b.as_view()).unwrap(),
                    &mut scores[1..rows + 1],
                    dimension(dim),
                    Params {
                        a_panels_in_l2: NonZeroUsize::new(a_blocks).unwrap(),
                        b_cols_in_l1: NonZeroUsize::new(b_cols).unwrap(),
                    },
                )
            };
            driver::Drive::drive(&mut driver);
            driver::Drive::drive(&mut driver);
            assert_eq!(scores[0], 12345.0);
            assert_eq!(scores[rows + 1], 12345.0);
            assert_eq!(
                &scores[1..rows + 1],
                expected,
                "({rows},{cols},{dim},{a_blocks},{b_cols})"
            );
        }
    }

    fn check_registers<A, const MR: usize>(arch: A)
    where
        A: Architecture + ExtraWide<MR>,
    {
        arch.run_inline(|| {
            for rows in 1..=MR {
                for b in [[0; 4], [15; 4], [1, 7, 3, 15]] {
                    let values: [Group; MR] = core::array::from_fn(|i| {
                        core::array::from_fn(|j| {
                            if i.is_multiple_of(2) {
                                255
                            } else {
                                (i * 11 + j * 13) as u8
                            }
                        })
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
                    // SAFETY: One accumulator has exactly one metadata entry.
                    unsafe {
                        arch.reduce([acc], &query, Slice::new(&[doc]), dimension(4), &mut scores)
                    };
                    for i in 0..rows {
                        let raw: u32 = values[i]
                            .iter()
                            .zip(b)
                            .map(|(&a, b)| u32::from(a) * u32::from(b))
                            .sum::<u32>()
                            * 3;
                        let expected =
                            -(0.5 * 0.25 * raw as f32 + 7.0 * 3.0 + 11.0 * -2.0 + -2.0 * 3.0 * 4.0);
                        assert_eq!(scores[i], expected);
                    }
                    let previous = scores;
                    let nan = MinMaxCompensation { a: f32::NAN, ..doc };
                    // SAFETY: One accumulator has exactly one metadata entry.
                    unsafe {
                        arch.reduce([acc], &query, Slice::new(&[nan]), dimension(4), &mut scores)
                    };
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
        check_driver::<_, 8, 6>(Scalar::new());
        check_registers::<_, 8>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check_driver::<_, 16, 8>(arch);
            check_registers::<_, 16>(arch);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
            check_driver::<_, 16, 8>(arch);
            check_registers::<_, 16>(arch);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_driver::<_, 8, 8>(arch);
            check_registers::<_, 8>(arch);
        }
    }
}
