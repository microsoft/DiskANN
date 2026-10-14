/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! B-first MinMax MaxSim. Preparation and synchronized panel iterators
//! establish the micro-kernel's inputs; integer contraction never reads canonical data.

pub(crate) mod a;
pub(crate) mod b;
pub(crate) mod layout;

use super::packed_f32_x_unpacked_f32::Params;
use crate::matrix_kernels::{
    Cache, bounds, driver,
    num::{Elements, value_or_one},
    ptr::Slice,
    util::LoadStore,
};
use crate::minmax::MinMaxCompensation;
use a::{APanel, AView};
use b::{BPanel, BTile, RawRows, Scratch, UnpackNibbles};
use diskann_wide::{
    Architecture,
    arch::{Scalar, Target},
};

diskann_wide::alias!(f32x4<A> = f32x4);
diskann_wide::alias!(f32x8<A> = f32x8);
diskann_wide::alias!(i8x32<A> = i8x32);
diskann_wide::alias!(i8x64<A> = i8x64);
diskann_wide::alias!(i16x16<A> = i16x16);
diskann_wide::alias!(i32x8<A> = i32x8);
diskann_wide::alias!(i32x16<A> = i32x16);
diskann_wide::alias!(u8x16<A> = u8x16);
diskann_wide::alias!(u8x32<A> = u8x32);
diskann_wide::alias!(u8x64<A> = u8x64);
diskann_wide::alias!(u32x4<A> = u32x4);
diskann_wide::alias!(u32x8<A> = u32x8);
diskann_wide::alias!(u64x8<A> = u64x8);

pub(crate) struct Driver<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: AView<'a, MR, PACK>,
    b: RawRows<'a>,
    c: &'a mut [f32],
    scratch: Scratch,
    b_rows: std::num::NonZeroUsize,
}

impl<'a, A, const MR: usize, const NR: usize, const PACK: usize> Driver<'a, A, MR, NR, PACK> {
    pub(crate) fn new(
        arch: A,
        a: AView<'a, MR, PACK>,
        b: RawRows<'a>,
        c: &'a mut [f32],
        cache: Cache,
    ) -> Self {
        const {
            assert!(NR > 0);
        }
        assert_eq!(a.rows(), c.len(), "output row mismatch");
        assert_eq!(a.layout().dim(), b.dim(), "document dimension mismatch");
        assert!(
            a.layout().dim() <= layout::MAX_DIM,
            "integer accumulator dimension limit"
        );
        let params = Params::new(cache, a.panel_bytes(), Scratch::row_bytes(a.layout()), NR);
        Self {
            arch,
            a,
            b,
            c,
            scratch: Scratch::default(),
            b_rows: value_or_one(b.rows().min(params.b_cols_in_l1.get())),
        }
    }
}

impl<A, const MR: usize, const NR: usize, const PACK: usize> driver::Drive
    for Driver<'_, A, MR, NR, PACK>
where
    A: UnpackNibbles + ExtraWide<MR, PACK> + LoadStore<f32, MR>,
    for<'b> PanelKernel<'b, A, MR, NR, PACK>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run_inline(&mut *self);
    }
}

impl<A, const MR: usize, const NR: usize, const PACK: usize> Target<A, ()>
    for &mut Driver<'_, A, MR, NR, PACK>
where
    A: UnpackNibbles + ExtraWide<MR, PACK> + LoadStore<f32, MR>,
    for<'b> PanelKernel<'b, A, MR, NR, PACK>: driver::PanelKernel,
{
    #[inline(always)]
    fn run(self, arch: A) {
        self.c.fill(f32::MAX);
        let a = self.a;
        self.b.visit_tiles(
            self.b_rows,
            #[inline(always)]
            |raw| {
                let b = self.scratch.prepare(arch, raw, a.layout());
                a.visit_panels(
                    #[inline(always)]
                    |a, start| {
                        let output = &mut self.c[start..start + a.valid_rows()];
                        let mut panel = PanelKernel::<_, MR, NR, PACK>::new(arch, a, b, output);
                        driver::PanelKernel::panel_kernel(&mut panel);
                        arch.store(panel.scores, output);
                    },
                );
            },
        );
    }
}

struct PanelKernel<'a, A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    a: APanel<'a, MR, PACK>,
    b: BTile<'a>,
    compensation: PanelCompensation<MR>,
    scores: [f32; MR],
}

impl<'a, A, const MR: usize, const NR: usize, const PACK: usize> PanelKernel<'a, A, MR, NR, PACK> {
    #[inline(always)]
    fn new(arch: A, a: APanel<'a, MR, PACK>, b: BTile<'a>, output: &[f32]) -> Self
    where
        A: Architecture + LoadStore<f32, MR>,
    {
        assert_eq!(
            a.layout(),
            b.layout(),
            "A and B must use matching MinMax layouts"
        );
        assert_eq!(output.len(), a.valid_rows());
        Self {
            arch,
            a,
            b,
            compensation: PanelCompensation::new(a.compensation(), a.valid_rows()),
            scores: arch.load(output),
        }
    }

    #[inline(always)]
    fn micro_kernel<const N: usize>(&mut self, b: BPanel<'_, N>)
    where
        A: Architecture + ExtraWide<MR, PACK>,
    {
        let mut kernel = MicroKernel {
            arch: self.arch,
            a: self.a,
            b,
            compensation: &self.compensation,
            scores: &mut self.scores,
        };
        driver::MicroKernel::micro_kernel(&mut kernel);
    }
}

macro_rules! panel_kernel {
    ($arch:ty, $mr:literal, $nr:literal, $pack:literal, [$($n:literal),+ $(,)?]) => {
        impl driver::PanelKernel for PanelKernel<'_, $arch, $mr, $nr, $pack> {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let b = self.b;
                let remainder = b.visit_panels::<$nr>(|panel| self.micro_kernel(panel));
                if let Some(remainder) = remainder {
                    $(
                        const { assert!($n > 0 && $n < $nr) };
                        if let Some(panel) = remainder.panel::<$n>() {
                            self.micro_kernel(panel);
                        }
                    )+
                }
            }
        }
    };
}

struct MicroKernel<'a, A, const MR: usize, const N: usize, const PACK: usize> {
    arch: A,
    a: APanel<'a, MR, PACK>,
    b: BPanel<'a, N>,
    compensation: &'a PanelCompensation<MR>,
    scores: &'a mut [f32; MR],
}

impl<A, const MR: usize, const N: usize, const PACK: usize> driver::MicroKernel
    for MicroKernel<'_, A, MR, N, PACK>
where
    A: Architecture + ExtraWide<MR, PACK>,
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        self.arch.run_inline(&mut *self);
    }
}

impl<A, const MR: usize, const N: usize, const PACK: usize> Target<A, ()>
    for &mut MicroKernel<'_, A, MR, N, PACK>
where
    A: Architecture + ExtraWide<MR, PACK>,
{
    #[inline(always)]
    fn run(self, arch: A) {
        let acc = contract(arch, self.a, self.b);
        arch.reduce(
            acc,
            self.compensation,
            self.b.compensation(),
            self.a.layout().dim(),
            self.scores,
        );
    }
}

/// Panel-local SIMD-friendly projection; owned query metadata remains array-of-structs.
struct PanelCompensation<const MR: usize> {
    scale: [f32; MR],
    bias: [f32; MR],
    sum: [f32; MR],
    valid_rows: usize,
}

impl<const MR: usize> PanelCompensation<MR> {
    #[inline(always)]
    fn new(query: &[MinMaxCompensation; MR], valid_rows: usize) -> Self {
        assert!(valid_rows <= MR);
        Self {
            scale: core::array::from_fn(|i| query[i].a),
            bias: core::array::from_fn(|i| query[i].b),
            sum: core::array::from_fn(|i| query[i].n),
            valid_rows,
        }
    }
}

#[inline(always)]
fn contract<A, const MR: usize, const N: usize, const PACK: usize>(
    arch: A,
    a: APanel<'_, MR, PACK>,
    b: BPanel<'_, N>,
) -> [A::Accumulator; N]
where
    A: ExtraWide<MR, PACK>,
{
    let mut acc = [arch.zero(); N];
    for group in 0..a.layout().k().value().get() / PACK {
        // SAFETY: Preparation pads K to a multiple of PACK; each A group spans MR * PACK.
        let query = unsafe { arch.load(a.group(group)) };
        for (doc, acc) in acc.iter_mut().enumerate() {
            // SAFETY: doc < N and B uses the same padded K; each B group spans PACK.
            let values = unsafe { arch.splat(b.group::<PACK>(doc, group)) };
            *acc = arch.dot(query, values, *acc);
        }
    }
    acc
}

/// Read one complete prepared B group.
///
/// # Safety
///
/// `values.len()` must be exactly `PACK`.
#[inline(always)]
unsafe fn group<const PACK: usize>(values: Slice<'_, u8>) -> [u8; PACK] {
    bounds::check_eq!(values.len(), PACK);
    core::array::from_fn(|i| {
        // SAFETY: i < PACK, so the offset selects one readable element.
        unsafe { *values.add(Elements::new(i)).as_unit().as_ref() }
    })
}

/// Register representations remain private to each architecture backend.
trait ExtraWide<const MR: usize, const PACK: usize>: Architecture {
    type Query: Copy;
    type Splat: Copy;
    type Accumulator: Copy;
    /// Load a complete prepared A patch, including zero-padded query rows.
    ///
    /// # Safety
    ///
    /// `slice.len()` must be exactly `MR * PACK`.
    unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Query;
    /// Broadcast one group of unpacked 4-bit B codes.
    ///
    /// # Safety
    ///
    /// `slice.len()` must be exactly `PACK`.
    unsafe fn splat(self, slice: Slice<'_, u8>) -> Self::Splat;
    fn zero(self) -> Self::Accumulator;
    fn dot(self, query: Self::Query, doc: Self::Splat, acc: Self::Accumulator)
    -> Self::Accumulator;
    fn reduce<const N: usize>(
        self,
        acc: [Self::Accumulator; N],
        query: &PanelCompensation<MR>,
        docs: &[MinMaxCompensation; N],
        dim: usize,
        scores: &mut [f32; MR],
    );
}

panel_kernel!(Scalar, 8, 5, 4, [1, 2, 3, 4]);

impl ExtraWide<8, 4> for Scalar {
    type Query = [[u16; 8]; 4];
    type Splat = [u16; 4];
    type Accumulator = [u32; 8];
    #[inline(always)]
    unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Query {
        // SAFETY: By precondition, the patch contains MR * PACK = 32 bytes.
        let values = unsafe { slice.as_std_slice(32) };
        core::array::from_fn(|d| core::array::from_fn(|row| u16::from(values[row * 4 + d])))
    }
    #[inline(always)]
    unsafe fn splat(self, slice: Slice<'_, u8>) -> Self::Splat {
        // SAFETY: By precondition, the span contains one complete PACK4 group.
        unsafe { group::<4>(slice) }.map(u16::from)
    }
    #[inline(always)]
    fn zero(self) -> Self::Accumulator {
        [0; 8]
    }
    #[inline(always)]
    fn dot(self, a: Self::Query, b: Self::Splat, acc: Self::Accumulator) -> Self::Accumulator {
        core::array::from_fn(|i| {
            let dot = a[0][i] * b[0] + a[1][i] * b[1] + a[2][i] * b[2] + a[3][i] * b[3];
            acc[i].wrapping_add(u32::from(dot))
        })
    }
    #[inline(always)]
    fn reduce<const N: usize>(
        self,
        acc: [Self::Accumulator; N],
        query: &PanelCompensation<8>,
        docs: &[MinMaxCompensation; N],
        dim: usize,
        scores: &mut [f32; 8],
    ) {
        for (acc, doc) in acc.iter().zip(docs) {
            for (i, (&raw, best)) in acc
                .iter()
                .zip(scores.iter_mut())
                .take(query.valid_rows)
                .enumerate()
            {
                let similarity = query.scale[i] * doc.a * raw as f32
                    + query.sum[i] * doc.b
                    + doc.n * query.bias[i]
                    + query.bias[i] * doc.b * dim as f32;
                *best = best.min(-similarity);
            }
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
fn compensate<F, T: Copy, const MR: usize, const N: usize>(
    arch: F::Arch,
    acc: [T; N],
    query: &PanelCompensation<MR>,
    docs: &[MinMaxCompensation; N],
    dim: usize,
    scores: &mut [f32; MR],
    convert: impl Fn(T, usize) -> F,
) where
    F: diskann_wide::SIMDVector<Scalar = f32>
        + diskann_wide::SIMDMinMax
        + std::ops::Add<Output = F>
        + std::ops::Sub<Output = F>
        + std::ops::Mul<Output = F>,
{
    const {
        assert!(MR.is_multiple_of(F::LANES));
    }
    for (part, output) in scores
        .chunks_exact_mut(F::LANES)
        .take(query.valid_rows.div_ceil(F::LANES))
        .enumerate()
    {
        let start = part * F::LANES;
        // SAFETY: Each metadata and output slice contains exactly LANES elements.
        let (scale, bias, sum, mut best) = unsafe {
            (
                F::load_simd(arch, query.scale[start..].as_ptr()),
                F::load_simd(arch, query.bias[start..].as_ptr()),
                F::load_simd(arch, query.sum[start..].as_ptr()),
                F::load_simd(arch, output.as_ptr()),
            )
        };
        for (&raw, doc) in acc.iter().zip(docs) {
            let mut similarity = (scale * F::splat(arch, doc.a)) * convert(raw, part);
            similarity = similarity + sum * F::splat(arch, doc.b);
            similarity = similarity + F::splat(arch, doc.n) * bias;
            similarity = similarity + (bias * F::splat(arch, doc.b)) * F::splat(arch, dim as f32);
            best = best.min_simd_standard(F::default(arch) - similarity);
        }
        // SAFETY: The output chunk contains LANES writable elements.
        unsafe { best.store_simd(output.as_mut_ptr()) };
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::{
        SIMDDotProduct, SIMDReinterpret, SIMDVector,
        arch::x86_64::{V3, V4},
    };

    panel_kernel!(V3, 16, 5, 4, [1, 2, 3, 4]);
    panel_kernel!(V4, 16, 8, 8, [1, 2, 3, 4, 5, 6, 7]);

    impl ExtraWide<16, 4> for V3 {
        type Query = [u8x32<V3>; 2];
        type Splat = i8x32<V3>;
        type Accumulator = [i32x8<V3>; 2];
        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Query {
            bounds::check_eq!(slice.len(), 64);
            // SAFETY: The 64-byte patch contains two complete 32-byte halves.
            unsafe {
                [
                    SIMDVector::load_simd(self, slice.as_ptr()),
                    SIMDVector::load_simd(self, slice.add(Elements::new(32)).as_ptr()),
                ]
            }
        }
        #[inline(always)]
        unsafe fn splat(self, slice: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: By precondition, the span contains one complete PACK4 group.
            let value = unsafe { group(slice) };
            u32x8::<V3>::splat(self, u32::from_le_bytes(value)).reinterpret_simd()
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            let dot = |a: u8x32<V3>, acc: i32x8<V3>| {
                // B is a nibble, so each pair sum is at most 7650 and cannot saturate.
                let products = self.mm256_maddubs_epi16(a, b);
                acc.dot_simd(products, i16x16::<V3>::splat(self, 1))
            };
            acc[0] = dot(a[0], acc[0]);
            acc[1] = dot(a[1], acc[1]);
            acc
        }
        #[inline(always)]
        fn reduce<const N: usize>(
            self,
            acc: [Self::Accumulator; N],
            query: &PanelCompensation<16>,
            docs: &[MinMaxCompensation; N],
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            compensate(
                self,
                acc,
                query,
                docs,
                dim,
                scores,
                |acc: Self::Accumulator, part| {
                    f32x8::<V3>::from_array(self, acc[part].to_array().map(|x| x as u32 as f32))
                },
            );
        }
    }

    impl ExtraWide<16, 8> for V4 {
        type Query = [u8x64<V4>; 2];
        type Splat = i8x64<V4>;
        type Accumulator = [i32x16<V4>; 2];
        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Query {
            bounds::check_eq!(slice.len(), 128);
            // SAFETY: The 128-byte patch contains two complete 64-byte halves.
            unsafe {
                [
                    SIMDVector::load_simd(self, slice.as_ptr()),
                    SIMDVector::load_simd(self, slice.add(Elements::new(64)).as_ptr()),
                ]
            }
        }
        #[inline(always)]
        unsafe fn splat(self, slice: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: By precondition, the span contains one complete PACK8 group.
            let value = unsafe { group(slice) };
            u64x8::<V4>::splat(self, u64::from_le_bytes(value)).reinterpret_simd()
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a[0], b);
            acc[1] = acc[1].dot_simd(a[1], b);
            acc
        }
        #[inline(always)]
        fn reduce<const N: usize>(
            self,
            acc: [Self::Accumulator; N],
            query: &PanelCompensation<16>,
            docs: &[MinMaxCompensation; N],
            dim: usize,
            scores: &mut [f32; 16],
        ) {
            let convert = |acc: Self::Accumulator, part: usize| {
                // PACK8 uses two adjacent VNNI accumulator lanes per query row.
                let lanes = acc[part].to_array();
                f32x8::<V4>::from_array(
                    self,
                    core::array::from_fn(|i| {
                        lanes[2 * i].wrapping_add(lanes[2 * i + 1]) as u32 as f32
                    }),
                )
            };
            compensate(self, acc, query, docs, dim, scores, convert);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::{SIMDDotProduct, SIMDVector, arch::aarch64::Neon};

    panel_kernel!(Neon, 8, 8, 4, [1, 2, 3, 4, 5, 6, 7]);

    impl ExtraWide<8, 4> for Neon {
        type Query = [u8x16<Neon>; 2];
        type Splat = u8x16<Neon>;
        type Accumulator = [u32x4<Neon>; 2];
        #[inline(always)]
        unsafe fn load(self, slice: Slice<'_, u8>) -> Self::Query {
            bounds::check_eq!(slice.len(), 32);
            // SAFETY: The 32-byte patch contains two complete 16-byte halves.
            unsafe {
                [
                    SIMDVector::load_simd(self, slice.as_ptr()),
                    SIMDVector::load_simd(self, slice.add(Elements::new(16)).as_ptr()),
                ]
            }
        }
        #[inline(always)]
        unsafe fn splat(self, slice: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: By precondition, the span contains one complete PACK4 group.
            let value = unsafe { group::<4>(slice) };
            Self::Splat::from_array(self, core::array::from_fn(|i| value[i % 4]))
        }
        #[inline(always)]
        fn zero(self) -> Self::Accumulator {
            [SIMDVector::default(self); 2]
        }
        #[inline(always)]
        fn dot(
            self,
            a: Self::Query,
            b: Self::Splat,
            mut acc: Self::Accumulator,
        ) -> Self::Accumulator {
            acc[0] = acc[0].dot_simd(a[0], b);
            acc[1] = acc[1].dot_simd(a[1], b);
            acc
        }
        #[inline(always)]
        fn reduce<const N: usize>(
            self,
            acc: [Self::Accumulator; N],
            query: &PanelCompensation<8>,
            docs: &[MinMaxCompensation; N],
            dim: usize,
            scores: &mut [f32; 8],
        ) {
            compensate(
                self,
                acc,
                query,
                docs,
                dim,
                scores,
                |acc: Self::Accumulator, part| {
                    f32x4::<Neon>::from_array(self, acc[part].to_array().map(|x| x as f32))
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unsigned_accumulators_are_preserved_during_conversion_and_pair_folding() {
        use diskann_wide::{
            SIMDVector,
            arch::x86_64::{V3, V4},
        };
        let query = PanelCompensation::new(
            &[MinMaxCompensation {
                a: 1.0,
                ..Default::default()
            }; 16],
            16,
        );
        let docs = [MinMaxCompensation {
            a: 1.0,
            ..Default::default()
        }; 1];
        if let Some(arch) = V3::new_checked() {
            arch.run_inline(
                #[inline]
                || {
                    let raw = 0xe000_0000u32;
                    let acc = [[i32x8::<V3>::splat(arch, raw as i32); 2]; 1];
                    let mut scores = [f32::MAX; 16];
                    arch.reduce(acc, &query, &docs, 4, &mut scores);
                    assert_eq!(scores, [-(raw as f32); 16]);
                },
            );
        }
        if let Some(arch) = V4::new_checked_miri() {
            arch.run_inline(
                #[inline]
                || {
                    let codes = [0, 1, 2, 3, 12, 13, 14, 15];
                    // SAFETY: codes contains one complete PACK8 group.
                    let splat = unsafe { arch.splat(Slice::new(&codes)) };
                    assert_eq!(
                        splat.to_array(),
                        core::array::from_fn(|i| codes[i % 8] as i8)
                    );
                    let values: [[i32; 16]; 2] = core::array::from_fn(|part| {
                        core::array::from_fn(|lane| 0x7000_0000 + (part * 16 + lane) as i32 * 1024)
                    });
                    let acc = [values.map(|part| i32x16::<V4>::from_array(arch, part))];
                    let mut scores = [f32::MAX; 16];
                    arch.reduce(acc, &query, &docs, 8, &mut scores);
                    let expected = core::array::from_fn(|row| {
                        let part = &values[row / 8];
                        let lane = 2 * (row % 8);
                        -(part[lane].wrapping_add(part[lane + 1]) as u32 as f32)
                    });
                    assert_eq!(scores, expected);
                },
            );
        }
    }

    use super::*;
    use crate::{
        bits::{Representation, Unsigned},
        minmax::{MinMaxCompensation, MinMaxMeta},
        multi_vector::{BlockTransposed, Defaulted, Mat, MaxSim, QueryMatRef},
    };
    use diskann_utils::ReborrowMut;
    use diskann_vector::DistanceFunctionMut;
    use diskann_wide::arch::Scalar;
    const NR: usize = 5;
    const PACK: usize = 4;

    fn matrix<const BITS: usize>(rows: usize, dim: usize) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut result = Mat::new(MinMaxMeta::new(rows, dim), Defaulted).unwrap();
        for (i, mut row) in result.reborrow_mut().rows_mut().enumerate() {
            let mut sum = 0u32;
            for d in 0..dim {
                let code = ((i * 17 + d * 13) % (1 << BITS)) as u32;
                row.vector_mut().set(d, i64::from(code)).unwrap();
                sum += code;
            }
            let scale = (i + 1) as f32 / 10.0;
            row.set_meta(MinMaxCompensation {
                dim: dim as u32,
                a: scale,
                b: -(i as f32) / 7.0,
                n: scale * sum as f32,
                ..Default::default()
            });
        }
        result
    }

    fn check_driver<A, const MR: usize, const NR: usize, const PACK: usize>(
        arch: A,
        dimensions: &[usize],
        panels: usize,
    ) where
        A: UnpackNibbles + ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        for<'a> PanelKernel<'a, A, MR, NR, PACK>: driver::PanelKernel,
    {
        for &dim in dimensions {
            let query = matrix::<8>(panels * MR + 1, dim);
            let docs = matrix::<4>(panels * NR + 1, dim);
            let layout = layout::Layout::new(dim).unwrap();
            let mut values =
                BlockTransposed::<u8, MR, PACK>::try_new(query.num_vectors(), layout.padded_dim())
                    .unwrap();
            let mut metadata =
                vec![[MinMaxCompensation::default(); MR]; query.num_vectors().div_ceil(MR)];
            for (i, row) in query.as_view().rows().enumerate() {
                let mut packed_row = values.get_row_mut(i).unwrap();
                for (d, &value) in row.vector().as_slice().iter().enumerate() {
                    packed_row.set(layout::Layout::position(d), value);
                }
                metadata[i / MR][i % MR] = row.meta();
            }
            let a = AView::new(values.as_view(), &metadata, &layout).unwrap();
            let mut expected = vec![0.0; query.num_vectors()];
            MaxSim::new(&mut expected).evaluate(QueryMatRef::from(query.as_view()), docs.as_view());
            let k = a.layout().k().value().get();
            let a_bytes = MR * k + std::mem::size_of::<[MinMaxCompensation; MR]>();
            let b_bytes = k + std::mem::size_of::<MinMaxCompensation>();
            assert_eq!(a.panel_bytes().value(), a_bytes);
            assert_eq!(Scratch::row_bytes(a.layout()).value(), b_bytes);
            for b_rows in [NR, 2 * NR] {
                let cache = Cache::new(value_or_one(a_bytes + b_rows * b_bytes), value_or_one(1));
                let mut scores = vec![123.0; query.num_vectors() + 2];
                let mut driver = Driver::<_, MR, NR, PACK>::new(
                    arch,
                    a,
                    RawRows::new(docs.as_view()),
                    &mut scores[1..query.num_vectors() + 1],
                    cache,
                );
                assert_eq!(driver.b_rows.get(), b_rows.min(docs.num_vectors()));
                for _ in 0..2 {
                    driver::Drive::drive(&mut driver);
                }
                assert_eq!(&scores[1..query.num_vectors() + 1], expected);
                assert_eq!((scores[0], scores[query.num_vectors() + 1]), (123.0, 123.0));
            }
        }
    }

    fn check_driver_architectures(dimensions: &[usize], panels: usize) {
        check_driver::<_, 8, 5, 4>(Scalar::new(), dimensions, panels);
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_driver::<_, 16, 5, 4>(arch, dimensions, panels);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                check_driver::<_, 16, 8, 8>(arch, dimensions, panels);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_driver::<_, 8, 8, 4>(arch, dimensions, panels);
        }
    }

    #[test]
    fn query_panels_and_document_tiles_keep_metadata_aligned() {
        check_driver_architectures(&[1, 63, 65, 129], 3);
    }

    #[test]
    fn small_odd_dimension_panels_match_pairwise() {
        check_driver_architectures(&[3], 1);
    }

    panel_kernel!(Scalar, 8, 9, 4, [1, 2, 3, 4, 5, 6, 7, 8]);

    #[test]
    fn registered_panels_are_not_limited_to_eight_columns() {
        check_driver::<_, 8, 9, 4>(Scalar::new(), &[3], 1);
    }

    fn check_inactive_half<A: ExtraWide<MR, PACK>, const MR: usize, const PACK: usize>(arch: A) {
        arch.run_inline(
            #[inline]
            || {
                let query = PanelCompensation::new(
                    &[MinMaxCompensation {
                        a: 1.0,
                        b: 1.0,
                        n: 1.0,
                        ..Default::default()
                    }; MR],
                    MR / 2,
                );
                let docs = [MinMaxCompensation {
                    a: 1.0,
                    b: 1.0,
                    n: 1.0,
                    ..Default::default()
                }];
                let mut scores = [123.0; MR];
                arch.reduce([arch.zero()], &query, &docs, 1, &mut scores);
                assert!(scores[..MR / 2].iter().all(|&score| score == -3.0));
                assert!(scores[MR / 2..].iter().all(|&score| score == 123.0));
            },
        );
    }

    #[test]
    fn compensation_skips_the_inactive_query_half() {
        check_inactive_half::<_, 8, 4>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_inactive_half::<_, 16, 4>(arch);
            }
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked_miri() {
                check_inactive_half::<_, 16, 8>(arch);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check_inactive_half::<_, 8, 4>(arch);
        }
    }

    #[test]
    fn compatible_layouts_and_document_group_bounds() {
        use crate::matrix_kernels::test_util::panic_message_for;
        use layout::Layout;

        let a_layout = Layout::new(65).unwrap();
        let b_layout = Layout::new(65).unwrap();
        assert!(!std::ptr::eq(&a_layout, &b_layout));
        let values = BlockTransposed::<u8, 8, 4>::new(1, 128);
        let metadata = [[MinMaxCompensation::default(); 8]];
        let a = AView::new(values.as_view(), &metadata, &a_layout).unwrap();
        let docs = matrix::<4>(2, 65);
        let mut scratch = Scratch::default();
        let b = scratch.prepare(Scalar::new(), RawRows::new(docs.as_view()), &b_layout);
        a.visit_panels(|a, _| {
            let mut panel = PanelKernel::<_, 8, 5, 4>::new(Scalar::new(), a, b, &[f32::MAX]);
            driver::PanelKernel::panel_kernel(&mut panel);
            assert_eq!(panel.scores[0], 0.0);
        });
        assert!(
            b.visit_panels::<2>(|panel| {
                for row in 0..2 {
                    let source = docs.get_row(row).unwrap();
                    for group in 0..32 {
                        let expected: [u8; 4] = core::array::from_fn(|i| {
                            let column = group * 4 + i;
                            let dim = column / 64 * 64
                                + 2 * (column % 32)
                                + usize::from(column % 64 >= 32);
                            if dim < 65 {
                                source.vector().get(dim).unwrap() as u8
                            } else {
                                0
                            }
                        });
                        // SAFETY: Both rows contain 32 complete PACK4 groups.
                        let values = unsafe { panel.group::<4>(row, group).as_std_slice(4) };
                        assert_eq!(values, expected.as_slice());
                    }
                }
                let message = panic_message_for(|| {
                    // SAFETY: cfg(test) checks each logical index before accessing the pointer.
                    unsafe { panel.group::<4>(0, 32) };
                });
                assert!(
                    message.contains("document group out of bounds"),
                    "{message}"
                );
                let message = panic_message_for(|| {
                    // SAFETY: cfg(test) checks the row before accessing the pointer.
                    unsafe { panel.group::<4>(2, 0) };
                });
                assert!(message.contains("document row out of bounds"), "{message}");
            })
            .is_none()
        );

        // The same padded K does not make different original dimensions compatible.
        let other_layout = Layout::new(66).unwrap();
        let other_docs = matrix::<4>(2, 66);
        let other_b = scratch.prepare(
            Scalar::new(),
            RawRows::new(other_docs.as_view()),
            &other_layout,
        );
        a.visit_panels(|a, _| {
            let message = panic_message_for(|| {
                PanelKernel::<_, 8, 5, 4>::new(Scalar::new(), a, other_b, &[f32::MAX]);
            });
            assert!(
                message.contains("A and B must use matching MinMax layouts"),
                "{message}"
            );
        });
    }

    #[test]
    fn reused_b_scratch_fully_overwrites_codes_padding_and_metadata() {
        let docs = matrix::<4>(NR + 1, 65);
        let layout = layout::Layout::new(65).unwrap();
        let mut scratch = Scratch::default();
        // First populate a full panel, then unpack a smaller tail into the same buffers.
        RawRows::new(docs.as_view()).visit_tiles(value_or_one(NR), |raw| {
            let tile = scratch.prepare(Scalar::new(), raw, &layout);
            let mut visited = 0;
            let tail = tile.visit_panels::<NR>(|b| {
                for row in 0..NR {
                    assert!(b.compensation()[row].a.is_finite());
                    visited += 1;
                }
            });
            if let Some(tail) = tail {
                let panel = tail.panel::<1>().unwrap();
                assert_eq!(panel.compensation()[0], docs.get_row(NR).unwrap().meta());
                // SAFETY: The layout contains 128 columns, so the last pack is complete.
                let values = unsafe { panel.group::<PACK>(0, 128 / PACK - 1).as_std_slice(PACK) };
                assert_eq!(values, &[0; PACK]);
                visited += 1;
            }
            assert_eq!(visited, raw.rows());
        });
    }
}
