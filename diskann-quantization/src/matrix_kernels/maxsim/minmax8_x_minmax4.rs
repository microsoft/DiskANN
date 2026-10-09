/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MinMax8 by MinMax4 MaxSim over the existing packed/unpacked panel views.
//!
//! A uses the shared column-order contract with MinMax8's 64-dimensional even/odd policy.
//! The driver expands each canonical MinMax4 B tile once, then reuses it across A's panels.
//! Integer contraction consumes padded K-dimensional panels without knowing the original
//! dimension or metadata. MinMax reduction uses the original D and opaque accumulators.
//! Row tails select a fixed register path before contraction, not inside the dot loop.
//! Scratch is owned by each call, never by the shared prepared query.

mod decode;
pub(crate) mod layout;
pub(crate) mod reader;

use diskann_wide::{Architecture, SIMDMinMax, SIMDVector, arch::Scalar};

use crate::{
    matrix_kernels::{
        Cache,
        blocks::packed,
        bounds::{self, Bound},
        driver,
        num::Elements,
        ptr::MutSlice,
        util,
    },
    minmax::MinMaxCompensation,
};

use super::packed_f32_x_unpacked_f32::b_cols_in_l1;
use decode::Decoder;
use layout::{APanel, PackedQueryView, QueryCompensation};
use reader::{BPanel, BScratch, BTile, MinMax4Rows};

/// B-first traversal over byte-valued panels; `k` always counts padded dimensions.
pub(crate) struct Driver<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: PackedQueryView<'a, MR, PACK, NR>,
    b: MinMax4Rows<'a>,
    scratch: BScratch<'a>,
    c: &'a mut [f32],
    b_rows: std::num::NonZeroUsize,
}

impl<'a, A, const PACK: usize, const MR: usize, const NR: usize> Driver<'a, A, PACK, MR, NR> {
    pub(crate) fn new(
        arch: A,
        a: PackedQueryView<'a, MR, PACK, NR>,
        b: MinMax4Rows<'a>,
        c: &'a mut [f32],
        cache: Cache,
    ) -> Self {
        const { assert!(NR > 0) };
        assert_eq!(b.dim(), a.dim(), "document dimension mismatch");
        assert_eq!(c.len(), a.nrows(), "output length mismatch");
        let layout = a.layout();
        let requested = b_cols_in_l1(cache, a.panel_bytes(), layout.decoded_row_bytes(), NR).get();
        let b_rows = crate::matrix_kernels::num::value_or_one(
            requested.min(b.rows()).min(layout.max_b_rows()),
        );
        Self {
            arch,
            a,
            b,
            scratch: BScratch::new(layout, b_rows),
            c,
            b_rows,
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
                self.c.fill(f32::MAX);
                let mut c = MutSlice::new(self.c);
                self.b.visit_tiles(self.b_rows, |b| {
                    let decoded = self.scratch.decode(self.arch, b);
                    self.a.visit_panels(|a, start| {
                        // SAFETY: Query panels partition exactly the validated output rows.
                        let mut region = unsafe { c.subslice(start, Bound::new(a.valid_rows())) };
                        // SAFETY: The region was narrowed to this panel's valid rows.
                        let output = unsafe { region.as_std_mut_slice(a.valid_rows()) };
                        let mut panel = PanelKernel::new(self.arch, a, decoded, output);
                        driver::PanelKernel::panel_kernel(&mut panel);
                        util::LoadStore::<f32, MR>::store(self.arch, panel.c, output);
                    });
                });
            },
        );
    }
}

struct PanelKernel<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: APanel<'a, MR, PACK>,
    b: BTile<'a>,
    c: [f32; MR],
}

impl<'a, A, const PACK: usize, const MR: usize, const NR: usize> PanelKernel<'a, A, PACK, MR, NR>
where
    A: Architecture + util::LoadStore<f32, MR>,
{
    fn new(arch: A, a: APanel<'a, MR, PACK>, b: BTile<'a>, c: &[f32]) -> Self {
        assert_eq!(a.dim(), b.dim(), "panel dimension mismatch");
        assert_eq!(a.k(), b.k(), "panel padded dimension mismatch");
        assert_eq!(c.len(), a.valid_rows(), "panel output length mismatch");
        bounds::check_le!(Bound::new(c.len()), MR);
        Self {
            arch,
            a,
            b,
            c: util::LoadStore::<f32, MR>::load(arch, c),
        }
    }
}

impl<A, const PACK: usize, const MR: usize, const NR: usize> PanelKernel<'_, A, PACK, MR, NR>
where
    A: Architecture + ExtraWide<PACK, MR>,
{
    #[inline(always)]
    fn visit<const EXTENT: usize>(&mut self, b: BPanel<'_, EXTENT>) {
        let mut micro = MicroKernel::<_, PACK, MR, EXTENT> {
            arch: self.arch,
            a: self.a,
            b,
            c: &mut self.c,
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
                let b = self.b;
                let remainder = b.visit_panels::<$nr>(|panel| self.visit(panel));
                if let Some(remainder) = remainder {
                    $(
                        if let Some(panel) = remainder.try_as_panel::<$tail>() {
                            self.visit(panel);
                        }
                    )+
                }
            }
        }
    };
}

struct MicroKernel<'a, A, const PACK: usize, const MR: usize, const NR: usize> {
    arch: A,
    a: APanel<'a, MR, PACK>,
    b: BPanel<'a, NR>,
    c: &'a mut [f32; MR],
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
                // SAFETY: The panels share a contraction dimension and B holds nibbles.
                let acc = unsafe {
                    if self.a.valid_rows() <= MR / 2 {
                        contract::<_, PACK, MR, NR, true>(self.arch, self.a, self.b)
                    } else {
                        contract::<_, PACK, MR, NR, false>(self.arch, self.a, self.b)
                    }
                };
                self.arch.reduce(
                    acc,
                    self.a.compensation(),
                    self.b.compensation(),
                    self.a.dim(),
                    self.c,
                );
            },
        );
    }
}

/// Contract complete, zero-padded groups. `HALF` limits the result to MR / 2 query rows.
///
/// # Safety
///
/// The query and document contraction dimensions must agree.
#[inline(always)]
unsafe fn contract<A, const PACK: usize, const MR: usize, const NR: usize, const HALF: bool>(
    arch: A,
    a: APanel<'_, MR, PACK>,
    b: BPanel<'_, NR>,
) -> [A::Accumulator; NR]
where
    A: ExtraWide<PACK, MR>,
{
    let mut acc = [arch.zero(); NR];
    for group in 0..a.k().value().get() / PACK {
        // SAFETY: The group is in bounds, and each patch includes all MR rows.
        let query = arch.load::<HALF>(unsafe { a.group(group) });
        for (j, acc) in acc.iter_mut().enumerate() {
            // SAFETY: K is a multiple of PACK and j < NR, so this whole group is in bounds.
            let doc = unsafe { b.group::<PACK>(j, group) };
            *acc = arch.dot::<HALF>(query, arch.splat(doc), *acc);
        }
    }
    acc
}

/// Opaque register operations; panel traversal belongs to the micro-kernel.
trait ExtraWide<const PACK: usize, const MR: usize>: Copy {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;

    /// Load a query group. With `HALF`, only the first `MR / 2` rows are needed.
    fn load<const HALF: bool>(self, values: packed::Patch<'_, u8, MR, PACK>) -> Self::Query;
    fn zero(self) -> Self::Accumulator;
    fn splat(self, value: [u8; PACK]) -> Self::Splat;
    /// Accumulate the rows selected by `load::<HALF>`; other rows are unspecified.
    fn dot<const HALF: bool>(
        self,
        a: Self::Query,
        b: Self::Splat,
        acc: Self::Accumulator,
    ) -> Self::Accumulator;

    #[cfg(test)]
    fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; MR];

    /// Apply MinMax compensation without exposing the accumulator's register layout.
    fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<MR>,
        docs: &[MinMaxCompensation; NR],
        dim: usize,
        scores: &mut [f32; MR],
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn dot_registers<A, B, C, const HALF: bool>(a: [A; 2], b: B, mut acc: [C; 2]) -> [C; 2]
where
    A: SIMDVector,
    B: SIMDVector,
    C: SIMDVector + diskann_wide::SIMDDotProduct<A, B>,
{
    acc[0] = acc[0].dot_simd(a[0], b);
    if !HALF {
        acc[1] = acc[1].dot_simd(a[1], b);
    }
    acc
}

impl ExtraWide<4, 8> for Scalar {
    type Query = [[u16; 8]; 4];
    type Splat = [u16; 4];
    type Accumulator = [u32; 8];

    #[cfg(test)]
    fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 8] {
        acc
    }

    #[inline(always)]
    fn load<const HALF: bool>(self, values: packed::Patch<'_, u8, 8, 4>) -> Self::Query {
        let values = values.as_array();
        core::array::from_fn(|d| core::array::from_fn(|row| u16::from(values[row][d])))
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn splat(self, value: [u8; 4]) -> Self::Splat {
        value.map(u16::from)
    }
    #[inline(always)]
    fn dot<const HALF: bool>(
        self,
        a: Self::Query,
        b: Self::Splat,
        acc: Self::Accumulator,
    ) -> Self::Accumulator {
        // B holds nibbles, so four u8 x u4 products sum to at most 15300 and fit in u16.
        core::array::from_fn(|i| {
            let dot = a[0][i] * b[0] + a[1][i] * b[1] + a[2][i] * b[2] + a[3][i] * b[3];
            acc[i].wrapping_add(u32::from(dot))
        })
    }

    #[inline(always)]
    fn reduce<const NR: usize>(
        self,
        acc: [Self::Accumulator; NR],
        query: &QueryCompensation<8>,
        docs: &[MinMaxCompensation; NR],
        dim: usize,
        scores: &mut [f32; 8],
    ) {
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

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn compensate<F, T: Copy, const MR: usize, const NR: usize>(
    arch: F::Arch,
    acc: [T; NR],
    query: &QueryCompensation<MR>,
    docs: &[MinMaxCompensation; NR],
    dim: usize,
    scores: &mut [f32; MR],
    convert: impl Fn(T, usize) -> F,
) where
    F: SIMDVector<Scalar = f32>
        + SIMDMinMax
        + std::ops::Add<Output = F>
        + std::ops::Sub<Output = F>
        + std::ops::Mul<Output = F>,
{
    const {
        assert!(MR.is_multiple_of(F::LANES));
    }
    for (part, output) in scores.chunks_exact_mut(F::LANES).enumerate() {
        let start = part * F::LANES;
        // SAFETY: MR is a multiple of LANES, and each metadata/output chunk has LANES entries.
        let (scale, bias, sum, mut best) = unsafe {
            (
                F::load_simd(arch, query.scale[start..].as_ptr()),
                F::load_simd(arch, query.bias[start..].as_ptr()),
                F::load_simd(arch, query.scaled_sum[start..].as_ptr()),
                F::load_simd(arch, output.as_ptr()),
            )
        };
        for (acc, doc) in acc.iter().zip(docs) {
            let raw = convert(*acc, part);
            let mut similarity = (scale * F::splat(arch, doc.a)) * raw;
            similarity = similarity + sum * F::splat(arch, doc.b);
            similarity = similarity + F::splat(arch, doc.n) * bias;
            similarity = similarity + (bias * F::splat(arch, doc.b)) * F::splat(arch, dim as f32);
            best = best.min_simd_standard(F::default(arch) - similarity);
        }
        // SAFETY: This output chunk contains exactly LANES writable entries.
        unsafe { best.store_simd(output.as_mut_ptr()) };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::{
        SIMDReinterpret, SplitJoin, ZipUnzip,
        arch::x86_64::{V3, V4},
    };

    impl ExtraWide<4, 16> for V3 {
        type Query = [<V3 as Architecture>::u8x32; 2];
        type Splat = <V3 as Architecture>::i8x32;
        type Accumulator = [<V3 as Architecture>::i32x8; 2];

        #[cfg(test)]
        fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 16] {
            let lanes = acc.map(|x| x.to_array());
            core::array::from_fn(|i| lanes[i / 8][i % 8] as u32)
        }

        #[inline(always)]
        fn load<const HALF: bool>(self, values: packed::Patch<'_, u8, 16, 4>) -> Self::Query {
            let values = values.as_ptr();
            // SAFETY: The patch spans 16 * 4 bytes; each half holds eight rows, or 32 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(32)).as_ptr());
                let hi = if HALF {
                    SIMDVector::default(self)
                } else {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(32))
                            .truncate(Elements::new(32))
                            .as_ptr(),
                    )
                };
                [lo, hi]
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: [u8; 4]) -> Self::Splat {
            diskann_wide::alias!(u32s = <V3>::u32x8);
            u32s::splat(self, u32::from_le_bytes(value)).reinterpret_simd()
        }
        #[inline(always)]
        fn dot<const HALF: bool>(
            self,
            a: Self::Query,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            dot_registers::<_, _, _, HALF>(a, b, acc)
        }

        #[inline(always)]
        fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: &[MinMaxCompensation; NR],
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V3>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as u32 as f32))
            };
            compensate(self, acc, query, docs, dim, scores, convert);
        }
    }

    impl ExtraWide<8, 16> for V4 {
        type Query = [<V4 as Architecture>::u8x64; 2];
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
        fn load<const HALF: bool>(self, values: packed::Patch<'_, u8, 16, 8>) -> Self::Query {
            let values = values.as_ptr();
            // SAFETY: The patch spans 16 * 8 bytes; each half holds eight rows, or 64 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(64)).as_ptr());
                let hi = if HALF {
                    SIMDVector::default(self)
                } else {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(64))
                            .truncate(Elements::new(64))
                            .as_ptr(),
                    )
                };
                [lo, hi]
            }
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn splat(self, value: [u8; 8]) -> Self::Splat {
            diskann_wide::alias!(u64s = <V4>::u64x8);
            u64s::splat(self, u64::from_le_bytes(value)).reinterpret_simd()
        }
        #[inline(always)]
        fn dot<const HALF: bool>(
            self,
            a: Self::Query,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            dot_registers::<_, _, _, HALF>(a, b, acc)
        }

        #[inline(always)]
        fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<16>,
            docs: &[MinMaxCompensation; NR],
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            diskann_wide::alias!(floats = <V4>::f32x8);
            let convert = |acc: Self::Accumulator, part: usize| {
                let reduced: <V4 as Architecture>::i32x8 = acc[part]
                    .split()
                    .map(|half| {
                        let pair = half.unzip();
                        pair.lo + pair.hi
                    })
                    .join();
                floats::from_array(self, reduced.to_array().map(|x| x as u32 as f32))
            };
            compensate(self, acc, query, docs, dim, scores, convert);
        }
    }

    panel_kernel!(V3, 4, 16, 6, [1, 2, 3, 4, 5]);
    panel_kernel!(V4, 8, 16, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::arch::aarch64::Neon;

    impl ExtraWide<4, 8> for Neon {
        type Query = [<Neon as Architecture>::u8x16; 2];
        type Splat = <Neon as Architecture>::u8x16;
        type Accumulator = [<Neon as Architecture>::u32x4; 2];

        #[cfg(test)]
        fn accumulator_lanes(self, acc: Self::Accumulator) -> [u32; 8] {
            let lanes = acc.map(|x| x.to_array());
            core::array::from_fn(|i| lanes[i / 4][i % 4])
        }

        #[inline(always)]
        fn load<const HALF: bool>(self, values: packed::Patch<'_, u8, 8, 4>) -> Self::Query {
            let values = values.as_ptr();
            // SAFETY: The patch spans 8 * 4 bytes; each half holds four rows, or 16 bytes.
            unsafe {
                let lo = SIMDVector::load_simd(self, values.truncate(Elements::new(16)).as_ptr());
                let hi = if HALF {
                    SIMDVector::default(self)
                } else {
                    SIMDVector::load_simd(
                        self,
                        values
                            .add(Elements::new(16))
                            .truncate(Elements::new(16))
                            .as_ptr(),
                    )
                };
                [lo, hi]
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
        fn dot<const HALF: bool>(
            self,
            a: Self::Query,
            b: Self::Splat,
            acc: Self::Accumulator,
        ) -> Self::Accumulator {
            dot_registers::<_, _, _, HALF>(a, b, acc)
        }

        #[inline(always)]
        fn reduce<const NR: usize>(
            self,
            acc: [Self::Accumulator; NR],
            query: &QueryCompensation<8>,
            docs: &[MinMaxCompensation; NR],
            dim: usize,
            scores: &mut [f32; 8],
        ) {
            diskann_wide::alias!(floats = <Neon>::f32x4);
            let convert = |acc: Self::Accumulator, part: usize| {
                floats::from_array(self, acc[part].to_array().map(|x| x as f32))
            };
            compensate(self, acc, query, docs, dim, scores, convert);
        }
    }

    panel_kernel!(Neon, 4, 8, 8, [1, 2, 3, 4, 5, 6, 7]);
}

#[cfg(test)]
mod tests {
    use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};

    use super::layout::{
        BLOCK, Layout,
        tests::{position, query as packed_query},
    };
    use super::*;
    use crate::{
        matrix_kernels::{num::value_or_one, test_util::panic_message_for},
        minmax::{Data, DataMutRef, DataRef, MinMaxMeta},
        multi_vector::{MatRef, block_transposed::BlockLayout},
    };

    fn canonical(b: &rowmajor::Owned<u8>, dim: usize) -> MinMax4Rows<'_> {
        MinMax4Rows::new(MatRef::new(MinMaxMeta::<4>::new(b.nrows(), dim), b.as_slice()).unwrap())
    }

    fn check_decoded_b<A: Decoder>(arch: A) {
        for &dim in layout::tests::DIMS.iter().filter(|&&d| d != 0) {
            for rows in [1, 7, 8, 9, 31, 32, 33] {
                if cfg!(miri) && !(matches!(dim, 1 | 8 | 9) && rows <= 8) {
                    continue;
                }
                let b = documents(rows, dim);
                let layout = Layout::new::<8, 4, 6>(dim).unwrap();
                let mut scratch = BScratch::new(layout.nonempty().unwrap(), value_or_one(rows));
                let decoded = scratch.decode(arch, canonical(&b, dim));
                let k = decoded.k().value().get();
                let (values, metadata) = decoded.as_slices();
                for row in 0..rows {
                    let meta = metadata[row];
                    assert_eq!(meta.a, (row % 4 + 1) as f32 * 0.25);
                    assert_eq!(meta.b, (row % 3) as f32 - 1.0);
                    assert_eq!(meta.n, (row % 5) as f32 * 0.5);
                    let mut expected = vec![0; k];
                    for d in 0..dim {
                        expected[position(d)] = ((row * 7 + d * 3 + 1) % 16) as u8;
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

    fn documents(rows: usize, dim: usize) -> rowmajor::Owned<u8> {
        let stride = Data::<4>::canonical_bytes(dim);
        let mut bytes = rowmajor::Owned::from_element(rows, stride, 0_u8);
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
        let values = packed_query::<8, 4, 6>(8, 9, |_, _| 0, |_| MinMaxCompensation::default());
        let docs = documents(2, 9);
        for (dim, output_rows) in [(8, 8), (9, 0), (9, 9)] {
            let _ = panic_message_for(|| {
                let mut scores = vec![0.0; output_rows];
                let input = if dim == 9 { &docs } else { &documents(2, dim) };
                let _ = Driver::<_, 4, 8, 6>::new(
                    Scalar::new(),
                    values.as_view().unwrap(),
                    canonical(input, dim),
                    &mut scores,
                    Cache::detect(),
                );
            });
        }
    }

    #[test]
    fn decoded_b_rejects_incompatible_tiles() {
        let dim = 9;
        let docs = documents(2, dim);
        for (rows, doc_dim, expected) in [
            (1, dim, "document tile exceeds scratch"),
            (2, dim - 1, "document dimension mismatch"),
        ] {
            let message = panic_message_for(|| {
                let layout = Layout::new::<8, 4, 6>(dim).unwrap();
                let mut scratch = BScratch::new(layout.nonempty().unwrap(), value_or_one(rows));
                let docs = if doc_dim == dim {
                    &docs
                } else {
                    &documents(2, doc_dim)
                };
                let _ = scratch.decode(Scalar::new(), canonical(docs, doc_dim));
            });
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    #[should_panic(expected = "scratch exceeds validated byte budget")]
    fn scratch_rejects_excessive_rows() {
        let layout = Layout::new::<8, 4, 6>(9).unwrap();
        let layout = layout.nonempty().unwrap();
        BScratch::new(layout, value_or_one(layout.max_b_rows() + 1));
    }

    #[test]
    #[should_panic(expected = "panel dimension mismatch")]
    fn panel_rejects_distinct_dimensions_with_same_padding() {
        let query = packed_query::<8, 4, 6>(1, 9, |_, _| 0, |_| MinMaxCompensation::default());
        let docs = documents(1, 8);
        let layout = Layout::new::<8, 4, 6>(8).unwrap();
        let mut scratch = BScratch::new(layout.nonempty().unwrap(), value_or_one(1));
        let decoded = scratch.decode(Scalar::new(), canonical(&docs, 8));
        let query = query.as_view().unwrap();
        assert_eq!(query.k(), decoded.k());
        query.visit_panels(|panel, _| {
            let _ = PanelKernel::<_, 4, 8, 6>::new(Scalar::new(), panel, decoded, &[0.0]);
        });
    }

    #[test]
    fn decoded_panels_keep_metadata_aligned_across_tiles() {
        let dim = 9;
        let docs = documents(23, dim);
        for capacity in [1, 5, 6, 7, 12, 13, 24] {
            let layout = Layout::new::<8, 4, 6>(dim).unwrap();
            let mut scratch = BScratch::new(layout.nonempty().unwrap(), value_or_one(capacity));
            let mut row = 0;
            canonical(&docs, dim).visit_tiles(value_or_one(capacity), |tile| {
                let decoded = scratch.decode(Scalar::new(), tile);
                let mut check = |meta: &[MinMaxCompensation]| {
                    for actual in meta {
                        let expected = DataRef::<4>::from_canonical_front(docs.row(row), dim)
                            .unwrap()
                            .meta();
                        assert_eq!(*actual, expected, "row={row}, capacity={capacity}");
                        row += 1;
                    }
                };
                let remainder = decoded.visit_panels::<6>(|panel| check(panel.compensation()));
                if let Some(remainder) = remainder {
                    macro_rules! tail {
                        ($($n:literal),+) => {$(
                            if let Some(panel) = remainder.try_as_panel::<$n>() {
                                check(panel.compensation());
                            }
                        )+};
                    }
                    tail!(1, 2, 3, 4, 5);
                }
            });
            assert_eq!(row, 23);
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
        let k = dim.next_multiple_of(BLOCK);
        let query_meta = |row: usize| MinMaxCompensation {
            a: (row % 3 + 1) as f32 * 0.5,
            b: (row % 5) as f32 - 2.0,
            n: (row % 7) as f32,
            ..Default::default()
        };
        let a = packed_query::<MR, PACK, NR>(
            rows,
            dim,
            |row, d| ((row * 17 + d * 3 + 1) % 256) as u8,
            query_meta,
        );
        let b = documents(cols, dim);
        let expected: Vec<f32> = (0..rows)
            .map(|i| {
                let q = query_meta(i);
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
                    let similarity = q.a * doc.a * raw as f32
                        + q.n * doc.b
                        + doc.n * q.b
                        + q.b * doc.b * dim as f32;
                    best = best.min(-similarity);
                }
                best
            })
            .collect();
        let mut scores = vec![12345.0; rows + 2];
        let a = a.as_view().unwrap();
        let b = canonical(&b, dim);
        let panel_bytes = a.panel_bytes().value();
        let b_row_bytes = k + std::mem::size_of::<MinMaxCompensation>();
        let cache = Cache::new(
            value_or_one(panel_bytes + b_rows_per_tile * b_row_bytes),
            value_or_one(panel_bytes * a_panels_per_tile),
        );
        let mut driver =
            Driver::<_, PACK, MR, NR>::new(arch, a, b, &mut scores[1..rows + 1], cache);
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
                            if i >= rows {
                                0
                            } else if i.is_multiple_of(2) {
                                255
                            } else {
                                (i * 11 + j * 13) as u8
                            }
                        })
                    });
                    let a = if rows <= MR / 2 {
                        arch.load::<true>(packed::Patch::from_array(&values))
                    } else {
                        arch.load::<false>(packed::Patch::from_array(&values))
                    };
                    let mut acc = arch.zero();
                    for _ in 0..3 {
                        acc = if rows <= MR / 2 {
                            arch.dot::<true>(a, arch.splat(b), acc)
                        } else {
                            arch.dot::<false>(a, arch.splat(b), acc)
                        };
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
                        arch.reduce([acc], &query, &[doc], group_dim, scores)
                    };
                    reduce(doc, &mut scores);
                    for i in 0..MR {
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
                let values = [[255; PACK]; MR];
                let a = arch.load::<false>(packed::Patch::from_array(&values));
                let b = arch.splat([15; PACK]);
                let mut acc = arch.zero();
                for iteration in 1..=300_000 {
                    acc = arch.dot::<false>(a, b, acc);
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
                        arch.reduce([acc], &query, &docs, 1, &mut scores);
                        assert_eq!(scores, [-(expected as f32); MR]);
                    }
                }
            }
            // All logical tails are padded before contraction.
            let groups: &[usize] = if cfg!(miri) {
                &[1, 2, 3]
            } else {
                &[1, 2, 3, 8, 17, 129]
            };
            let ks = groups
                .iter()
                .flat_map(|&groups| (0..PACK).map(move |r| groups * PACK - r));
            for dim in ks {
                let k = dim.next_multiple_of(BLOCK);
                let a_value = |row: usize, d: usize| ((row * 17 + d * 23 + 255) % 256) as u8;
                let b_value = |row: usize, d: usize| ((row * 7 + d * 3 + 15) % 16) as u8;
                // Deliberately construct panels without query storage, the layout
                // mapping, decoder, canonical reader, quantizer, or compensation. A's
                // padding is non-zero; B's dimension padding must prevent its contribution.
                let mut a = vec![0xff; BlockLayout::<MR, PACK>::block_len(k)];
                for row in 0..MR {
                    for d in 0..dim {
                        a[BlockLayout::<MR, PACK>::linear_index(row, position(d), k)] =
                            a_value(row, d);
                    }
                }
                let mut b = vec![0; 3 * k];
                for row in 0..3 {
                    for d in 0..dim {
                        b[row * k + position(d)] = b_value(row, d);
                    }
                }
                let query_meta = QueryCompensation::default();
                let doc_meta = [MinMaxCompensation::default(); 3];
                let geometry = Layout::new::<MR, PACK, 3>(dim).unwrap();
                let geometry = geometry.nonempty().unwrap();
                let ap = layout::tests::panel::<MR, PACK>(&a, &query_meta, geometry);
                let bp = BPanel::from_test_values(&b, &doc_meta, geometry);
                for half in [false, true] {
                    // SAFETY: The panels above have K columns and B holds nibbles.
                    let acc = unsafe {
                        if half {
                            contract::<_, PACK, MR, 3, true>(arch, ap, bp)
                        } else {
                            contract::<_, PACK, MR, 3, false>(arch, ap, bp)
                        }
                    };
                    let rows = if half { MR / 2 } else { MR };
                    for (doc, acc) in acc.into_iter().enumerate() {
                        let lanes = arch.accumulator_lanes(acc);
                        for (row, &actual) in lanes.iter().take(rows).enumerate() {
                            let expected = (0..dim)
                                .map(|d| u32::from(a_value(row, d)) * u32::from(b_value(doc, d)))
                                .sum::<u32>();
                            assert_eq!(
                                actual, expected,
                                "k={k}, row={row}, doc={doc}, half={half}"
                            );
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

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn dot_registers_handles_both_halves_and_wrapping() {
        use diskann_wide::Emulated;

        let arch = Scalar::new();
        let query = [[255; 32], core::array::from_fn(|i| (i * 7) as u8)];
        let doc = core::array::from_fn(|i| [0, 15, 1, 7][i % 4]);
        let initial = [[i32::MAX; 8], core::array::from_fn(|i| i as i32 * 31)];
        let b = Emulated::<i8, 32>::from_array(arch, doc);
        let acc = initial.map(|x| Emulated::<i32, 8>::from_array(arch, x));
        for hi in [[0; 32], query[1]] {
            let query = [query[0], hi];
            let a = query.map(|x| Emulated::<u8, 32>::from_array(arch, x));
            for (half, actual) in [
                (false, dot_registers::<_, _, _, false>(a, b, acc)),
                (true, dot_registers::<_, _, _, true>(a, b, acc)),
            ] {
                let expected: [[i32; 8]; 2] = core::array::from_fn(|part| {
                    core::array::from_fn(|lane| {
                        let mut sum = initial[part][lane];
                        if part == 0 || !half {
                            for d in 4 * lane..4 * (lane + 1) {
                                sum =
                                    sum.wrapping_add(i32::from(query[part][d]) * i32::from(doc[d]));
                            }
                        }
                        sum
                    })
                });
                assert_eq!(actual.map(|x| x.to_array()), expected);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_driver_and_registers() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check_driver::<_, 4, 16, 6>(arch);
            check_registers::<_, 4, 16>(arch);
            check_contraction::<_, 4, 16>(arch);
            check_decoded_b(arch);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_pairwise_reduction() {
        use diskann_wide::arch::x86_64::V4;

        let arch = V4::new_checked_miri();
        if cfg!(miri) {
            assert!(arch.is_some(), "V4 emulation requires V3 support");
        }
        let Some(arch) = arch else {
            eprintln!("V4 unavailable; use Miri to exercise the emulated reduction");
            return;
        };
        arch.run_inline(|| {
            let lanes = [
                [
                    1,
                    11,
                    23,
                    47,
                    -1,
                    2,
                    i32::MAX,
                    1,
                    i32::MIN,
                    -1,
                    i32::MAX,
                    i32::MAX,
                    i32::MIN,
                    i32::MIN,
                    -17,
                    -23,
                ],
                [
                    31,
                    5,
                    113,
                    257,
                    -7,
                    19,
                    i32::MAX,
                    2,
                    i32::MIN,
                    -2,
                    i32::MAX,
                    i32::MIN,
                    i32::MIN,
                    1,
                    0,
                    0,
                ],
            ];
            let acc = lanes.map(|x| <V4 as Architecture>::i32x16::from_array(arch, x));
            let query = QueryCompensation {
                scale: [1.0; 16],
                ..Default::default()
            };
            let docs = [MinMaxCompensation {
                a: 1.0,
                ..Default::default()
            }];
            let mut scores = [f32::MAX; 16];
            <V4 as ExtraWide<8, 16>>::reduce(arch, [acc], &query, &docs, 1, &mut scores);
            for (i, score) in scores.into_iter().enumerate() {
                let part = &lanes[i / 8];
                let pair = 2 * (i % 8);
                let expected = -(part[pair].wrapping_add(part[pair + 1]) as u32 as f32);
                assert_eq!(score, expected, "query lane {i}");
            }
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_padded_panels() {
        use diskann_wide::arch::x86_64::V4;

        let arch = V4::new_checked_miri();
        if cfg!(miri) {
            assert!(arch.is_some(), "V4 emulation requires V3 support");
        }
        let Some(arch) = arch else {
            eprintln!("V4 unavailable; use Miri to exercise the padded panels");
            return;
        };
        for rows in [1, 8, 9, 17] {
            check_driver_case::<_, 8, 16, 8>(arch, rows, 9, 9, 1, 8);
        }
        check_contraction::<_, 8, 16>(arch);
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
