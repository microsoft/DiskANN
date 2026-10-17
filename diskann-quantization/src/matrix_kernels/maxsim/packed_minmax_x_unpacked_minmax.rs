/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MaxSim for prepacked MinMax A and tiled MinMax B, with 1-8 bits per input code.
//!
//! Each B tile is decoded once and reused across A panels. Integer inner products
//! are compensated before reduction because B's coefficients vary by vector.
//! B rows are zero-padded to PACK; kernel `k` remains the logical dimension used
//! for compensation and accumulator limits.
//!
//! Outputs are unnegated maximum inner products. NaN candidates are ignored;
//! empty B or all-NaN comparisons leave negative infinity.
//!
//! Scalar and x86 provide a general i16 path (PACK=2) and a byte path (PACK=4).
//! The byte path supports B up to six bits on V3 and seven on Scalar/V4.
//! Neon uses unsigned byte dots (PACK=4) for every bit width.
//! A packing depends only on A's bit width and the selected numeric layout, so it
//! can be reused across B formats supported by that layout.

use std::num::NonZeroUsize;

use diskann_wide::{
    SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDReinterpret, SIMDVector,
    arch::{Architecture, Scalar, Target},
};

use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::{
        Cache,
        blocks::{packed, unpacked},
        bounds, driver,
        num::{Bytes, DimK, Elements},
        ptr::Slice,
        util::LoadStore,
    },
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{BlockTransposed, MatRef, Overflow},
};

use super::packed_f32_x_unpacked_f32::Params;

diskann_wide::alias!(f32x16<W> = f32x16);
diskann_wide::alias!(f32x8<W> = f32x8);
diskann_wide::alias!(i8x32<W> = i8x32);
diskann_wide::alias!(i8x64<W> = i8x64);
diskann_wide::alias!(i16x16<W> = i16x16);
diskann_wide::alias!(i16x32<W> = i16x32);
diskann_wide::alias!(i32x16<W> = i32x16);
diskann_wide::alias!(i32x8<W> = i32x8);
diskann_wide::alias!(u8x32<W> = u8x32);
diskann_wide::alias!(u8x64<W> = u8x64);
diskann_wide::alias!(u32x4<W> = u32x4);
diskann_wide::alias!(u32x8<W> = u32x8);
diskann_wide::alias!(u32x16<W> = u32x16);

// Signed backends determine the shared accumulation limit.
const fn max_k<const A_BITS: usize, const B_BITS: usize>() -> usize {
    assert!(A_BITS > 0 && A_BITS <= 8 && B_BITS > 0 && B_BITS <= 8);
    (i32::MAX as usize) / (((1 << A_BITS) - 1) * ((1 << B_BITS) - 1))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Coefficients<T> {
    a: T,
    b: T,
    n: T,
}

#[derive(Debug)]
pub(crate) struct PackedA<T: Copy, const A_BITS: usize, const MR: usize, const PACK: usize>
where
    Unsigned: Representation<A_BITS>,
{
    codes: BlockTransposed<T, MR, PACK>,
    meta: Coefficients<Vec<f32>>,
}

impl<T, const A_BITS: usize, const MR: usize, const PACK: usize> PackedA<T, A_BITS, MR, PACK>
where
    T: Copy + Default,
    Unsigned: Representation<A_BITS>,
{
    /// Pack A and copy its coefficients, zero-filling partial rows and K packs.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the packed data or coefficient allocation size overflows.
    #[expect(clippy::expect_used)]
    pub(crate) fn new<W: ExtraWide<MR, PACK, AElem = T>>(
        _: W,
        a: MatRef<'_, MinMaxMeta<A_BITS>>,
    ) -> Result<Self, Overflow> {
        let k = a.repr().intrinsic_dim();
        let mut codes = BlockTransposed::try_new(a.num_vectors(), k)?;
        let padded_rows = codes.padded_nrows();
        Overflow::check::<f32>(padded_rows, 1)?;
        let mut meta = Coefficients {
            a: vec![0.0; padded_rows],
            b: vec![0.0; padded_rows],
            n: vec![0.0; padded_rows],
        };
        for (i, row) in a.rows().enumerate() {
            let m = row.meta();
            meta.a[i] = m.a;
            meta.b[i] = m.b;
            meta.n[i] = m.n;
            let v = row.vector();
            let mut dst = codes
                .get_row_mut(i)
                .expect("A and its packing have equal row counts");
            for j in 0..k {
                // SAFETY: MinMaxMeta exposes exactly k codes for every row.
                dst.set(j, W::convert_a(unsafe { v.get_unchecked(j) } as u8));
            }
        }
        Ok(Self { codes, meta })
    }

    /// Cache-planning bytes for one padded A panel, including coefficients.
    fn panel_bytes(&self) -> Bytes {
        let codes = Elements::<T>::new(MR * self.codes.padded_ncols()).bytes();
        Bytes::new(codes.value() + size_of::<Coefficients<[f32; MR]>>())
    }
}

//--------//
// Driver //
//--------//

/// Each drive overwrites C; results do not accumulate across calls.
pub(crate) struct Driver<
    'a,
    W: ExtraWide<MR, PACK>,
    const B_BITS: usize,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> where
    Unsigned: Representation<B_BITS>,
{
    arch: W,
    a: packed::View<'a, W::AElem, MR, PACK>,
    a_meta: Coefficients<&'a [[f32; MR]]>,
    b: MatRef<'a, MinMaxMeta<B_BITS>>,
    c: &'a mut [f32],
    k: DimK,
    b_rows_per_tile: NonZeroUsize,
}

impl<
    'a,
    W: ExtraWide<MR, PACK>,
    const B_BITS: usize,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> Driver<'a, W, B_BITS, MR, NR, PACK>
where
    Unsigned: Representation<B_BITS>,
{
    /// # Panics
    ///
    /// Panics if K exceeds the integer accumulator limit for these bit widths.
    /// Conversion of a valid integer dot product to f32 may still round.
    ///
    /// B's bit width must fit the selected backend's [`ExtraWide::MAX_B_BITS`];
    /// incompatible combinations are rejected during monomorphization.
    ///
    /// # Safety
    ///
    /// A must be nonempty, both inputs must have logical dimension k, and C must have
    /// exactly A's logical row count.
    #[expect(clippy::expect_used)]
    pub(crate) unsafe fn new<const A_BITS: usize>(
        arch: W,
        a: &'a PackedA<W::AElem, A_BITS, MR, PACK>,
        b: MatRef<'a, MinMaxMeta<B_BITS>>,
        c: &'a mut [f32],
        k: DimK,
        cache: Cache,
    ) -> Self
    where
        Unsigned: Representation<A_BITS>,
    {
        const {
            assert!(NR > 0 && NR <= 8, "NR must be in 1..=8");
            assert!(
                B_BITS <= W::MAX_B_BITS,
                "B bit width exceeds the backend's supported range"
            )
        };
        let max_k = const { max_k::<A_BITS, B_BITS>() };
        assert!(
            k.value().get() <= max_k,
            "contraction dimension {} exceeds the MinMax{A_BITS} x MinMax{B_BITS} limit of {max_k}",
            k.value().get(),
        );
        bounds::check_eq!(bounds::Bound::new(a.codes.ncols()), k);
        bounds::check_eq!(bounds::Bound::new(b.repr().intrinsic_dim()), k);
        bounds::check_eq!(bounds::Bound::new(c.len()), a.codes.nrows());
        let a_meta = Coefficients {
            a: a.meta.a.as_chunks::<MR>().0,
            b: a.meta.b.as_chunks::<MR>().0,
            n: a.meta.n.as_chunks::<MR>().0,
        };
        let a_bytes = a.panel_bytes();
        let a = packed::View::from_block_transposed(a.codes.as_view())
            .expect("A and K must be nonempty");
        for meta in [a_meta.a, a_meta.b, a_meta.n] {
            bounds::check_eq!(bounds::Bound::new(meta.len()), a.blocks().get());
        }
        let b_rows_per_tile = Params::new(
            cache,
            a_bytes,
            BScratch::<W::BElem>::row_bytes::<PACK>(k),
            NR,
        )
        .b_cols_in_l1;
        Self {
            arch,
            a,
            a_meta,
            b,
            c,
            k,
            b_rows_per_tile,
        }
    }
}

impl<W, const B_BITS: usize, const MR: usize, const NR: usize, const PACK: usize> driver::Drive
    for Driver<'_, W, B_BITS, MR, NR, PACK>
where
    W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
    Unsigned: Representation<B_BITS>,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                self.c.fill(f32::NEG_INFINITY);
                let mut scratch = BScratch::<W::BElem>::default();
                let mut start = 0;
                while let Some(remaining) = NonZeroUsize::new(self.b.num_vectors() - start) {
                    let count = remaining.min(self.b_rows_per_tile);
                    let b =
                        scratch.prepare::<B_BITS, PACK>(self.b, start, count, self.k, W::convert_b);
                    // SAFETY: Driver's construction establishes all dimensions, metadata
                    // extents, and C's coverage of A. Visitors preserve these bounds.
                    unsafe {
                        self.a.visit_panels(self.k, |a, panel| {
                            let row = panel * MR;
                            let end = (row + MR).min(self.c.len());
                            let c = self.arch.load(&self.c[row..end]);
                            let a_meta = Coefficients {
                                a: self.arch.load_scores(&self.a_meta.a[panel]),
                                b: self.arch.load_scores(&self.a_meta.b[panel]),
                                n: self.arch.load_scores(&self.a_meta.n[panel]),
                            };
                            let kernel = PanelKernel::<W, MR, NR, PACK> {
                                arch: self.arch,
                                a,
                                a_meta,
                                b: b.codes,
                                b_meta: b.meta,
                                c,
                                k: self.k,
                            };
                            let c = self.arch.run(kernel);
                            self.arch.store(c, &mut self.c[row..end]);
                        });
                    }
                    start += count.get();
                }
            },
        );
    }
}

#[derive(Default)]
struct BScratch<T> {
    codes: Vec<T>,
    meta: Vec<MinMaxCompensation>,
}

struct BTile<'a, T> {
    codes: unpacked::View<'a, T>,
    meta: &'a [MinMaxCompensation],
}

#[inline]
#[expect(clippy::expect_used)]
fn padded_k<const PACK: usize>(k: DimK) -> DimK {
    DimK::new(
        NonZeroUsize::new(k.value().get().next_multiple_of(PACK))
            .expect("padding a positive dimension remains positive"),
    )
}

impl<T: Copy + Default> BScratch<T> {
    /// Convert logical codes once each; padding uses `T::default()` without conversion.
    #[expect(clippy::expect_used)]
    fn prepare<const B_BITS: usize, const PACK: usize>(
        &mut self,
        b: MatRef<'_, MinMaxMeta<B_BITS>>,
        start: usize,
        count: NonZeroUsize,
        k: DimK,
        mut convert: impl FnMut(u8) -> T,
    ) -> BTile<'_, T>
    where
        Unsigned: Representation<B_BITS>,
    {
        let dim = k.value().get();
        assert_eq!(b.repr().intrinsic_dim(), dim);
        assert!(start <= b.num_vectors() && count.get() <= b.num_vectors() - start);
        let physical_k = padded_k::<PACK>(k);
        let stride = physical_k.value().get();
        self.codes.resize(count.get() * stride, T::default());
        self.meta.resize(count.get(), MinMaxCompensation::default());
        for i in 0..count.get() {
            let row = b.get_row(start + i).expect("B tile range was checked");
            self.meta[i] = row.meta();
            let dst = &mut self.codes[i * stride..(i + 1) * stride];
            row.vector()
                .decode_into(&mut dst[..dim], |code| convert(code as u8))
                .expect("B output must match the logical code count");
            dst[dim..].fill(T::default());
        }
        // SAFETY: Each row has physical_k initialized elements, including zero padding.
        let codes = unsafe { unpacked::View::new(Slice::new(&self.codes), count, physical_k) };
        BTile {
            codes,
            meta: &self.meta,
        }
    }

    /// Cache-planning bytes for one decoded B row, including metadata.
    fn row_bytes<const PACK: usize>(k: DimK) -> Bytes {
        let codes = Elements::<T>::new(padded_k::<PACK>(k).value().get()).bytes();
        Bytes::new(codes.value() + size_of::<MinMaxCompensation>())
    }
}

//-------------//
// PanelKernel //
//-------------//

struct PanelKernel<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
{
    arch: W,
    a: packed::Panel<'a, W::AElem, MR, PACK>,
    a_meta: Coefficients<W::Float>,
    b: unpacked::View<'a, W::BElem>,
    b_meta: &'a [MinMaxCompensation],
    c: [f32; MR],
    k: DimK,
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    Target<W, [f32; MR]> for PanelKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn run(mut self, _: W) -> [f32; MR] {
        driver::PanelKernel::panel_kernel(&mut self);
        self.c
    }
}

struct Visitor<'a, W: ExtraWide<MR, PACK>, const MR: usize, const PACK: usize> {
    arch: W,
    a: packed::Panel<'a, W::AElem, MR, PACK>,
    a_meta: Coefficients<W::Float>,
    b_meta: &'a [MinMaxCompensation],
    c: &'a mut [f32; MR],
    k: DimK,
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    unpacked::PanelVisitor<W::BElem, NR> for Visitor<'_, W, MR, PACK>
{
    #[inline(always)]
    fn visit(&mut self, b: unpacked::Panel<'_, W::BElem, NR>, start: usize) {
        let mut kernel = MicroKernel {
            arch: self.arch,
            a: self.a,
            a_meta: self.a_meta,
            b,
            b_meta: &self.b_meta[start..start + NR],
            c: self.c,
            k: self.k,
        };
        driver::MicroKernel::micro_kernel(&mut kernel);
    }
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    driver::PanelKernel for PanelKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn panel_kernel(&mut self) {
        let mut visitor = Visitor {
            arch: self.arch,
            a: self.a,
            a_meta: self.a_meta,
            b_meta: self.b_meta,
            c: &mut self.c,
            k: self.k,
        };
        // SAFETY: B's view is constructed with PACK-padded rows and matching metadata.
        let tail = unsafe {
            self.b.visit_panels::<NR>(
                padded_k::<PACK>(self.k),
                Visitor {
                    c: &mut *visitor.c,
                    ..visitor
                },
            )
        };
        if let Some(tail) = tail {
            macro_rules! remainder {
                ($($n:literal),+) => {$(
                    if $n < NR && let Some(b) = tail.try_as_panel::<$n>() {
                        unpacked::PanelVisitor::visit(&mut visitor, b, tail.start());
                        return;
                    }
                )+};
            }
            remainder!(1, 2, 3, 4, 5, 6, 7);
            unreachable!("NR <= 8 bounds the trailing panel to at most seven B vectors");
        }
    }
}

//-------------//
// MicroKernel //
//-------------//

struct MicroKernel<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
{
    arch: W,
    a: packed::Panel<'a, W::AElem, MR, PACK>,
    a_meta: Coefficients<W::Float>,
    b: unpacked::Panel<'a, W::BElem, NR>,
    b_meta: &'a [MinMaxCompensation],
    c: &'a mut [f32; MR],
    k: DimK,
}

/// Keep the hot contraction step in the caller's ISA context.
///
/// # Safety
///
/// `a` must contain at least MR * PACK readable elements, and for every
/// `j < NR`, `b.add(bstride * j)` must contain PACK readable elements.
#[inline(always)]
unsafe fn accumulate_pack<W, const MR: usize, const NR: usize, const PACK: usize>(
    arch: W,
    a: Slice<'_, W::AElem>,
    b: Slice<'_, W::BElem>,
    bstride: Elements<W::BElem>,
    acc: &mut [W::Acc; NR],
) where
    W: ExtraWide<MR, PACK>,
{
    // SAFETY: The caller provides a full padded A pack.
    let a = unsafe { arch.load_a(a) };
    for (j, acc) in acc.iter_mut().enumerate() {
        // SAFETY: The caller guarantees a full pack in every B column.
        let b = unsafe { arch.splat(b.add(bstride * j)) };
        *acc = arch.dot(*acc, a, b);
    }
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    driver::MicroKernel for MicroKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        bounds::check_eq!(self.a.k(), self.k);
        let physical_k = padded_k::<PACK>(self.k);
        bounds::check_eq!(self.b.k(), physical_k);
        let k = self.k.value().get();
        let mut acc = [self.arch.zero_acc(); NR];
        let ap = self.a.as_ptr();
        let bp = self.b.as_ptr();
        let astride = self.a.pack_stride();
        let bstride = self.b.stride(physical_k);

        let packs = physical_k.value().get() / PACK;
        for pack in 0..packs {
            // SAFETY: A and every B row are zero-padded to a multiple of PACK.
            unsafe {
                accumulate_pack::<W, MR, NR, PACK>(
                    self.arch,
                    ap.add(astride * pack),
                    bp.add(Elements::new(pack * PACK)),
                    bstride,
                    &mut acc,
                );
            }
        }
        let mut best = self.arch.load_scores(self.c);
        for (acc, b) in acc.into_iter().zip(self.b_meta) {
            best = self.arch.compensate_max(acc, self.a_meta, b, k, best);
        }
        self.arch.store_scores(best, self.c);
    }
}
//----------//
// Backends //
//----------//

/// Backend operations for MR A rows and PACK contraction elements. Register
/// representation is private to each backend; NR only controls the number of accumulators.
///
/// All implementations accept A codes in 0..=255. B's range is limited by
/// both its element type and any intermediate arithmetic in the dot product.
pub(crate) trait ExtraWide<const MR: usize, const PACK: usize>: Architecture {
    /// Largest complete unsigned B width supported for every A code in 0..=255.
    /// This is a backend capability, not the actual A or B input width.
    const MAX_B_BITS: usize;

    type AElem: Copy + Default;
    type BElem: Copy + Default;
    type Wide: Copy;
    type Splat: Copy;
    type Acc: Copy;
    type Float: Copy;

    fn convert_a(code: u8) -> Self::AElem;
    /// `code` fits in MAX_B_BITS unsigned bits.
    fn convert_b(code: u8) -> Self::BElem;

    /// # Safety
    ///
    /// `a` must contain at least MR * PACK initialized, readable elements.
    unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide;

    fn zero_acc(self) -> Self::Acc;

    /// Accumulate MR dot products of PACK elements without mixing rows.
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc;

    fn load_scores(self, scores: &[f32; MR]) -> Self::Float;
    fn store_scores(self, scores: Self::Float, out: &mut [f32; MR]);

    /// Compensate each row using logical `k`, then update its maximum, ignoring NaNs.
    fn compensate_max(
        self,
        acc: Self::Acc,
        a: Coefficients<Self::Float>,
        b: &MinMaxCompensation,
        k: usize,
        best: Self::Float,
    ) -> Self::Float;

    /// # Safety
    ///
    /// B must contain PACK readable elements, including any zero padding.
    unsafe fn splat(self, b: Slice<'_, Self::BElem>) -> Self::Splat;
}

// Concrete array conversions check the register extents in each implementation.
macro_rules! impl_simd_ops {
    ($mr:literal, $pack:literal) => {
        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides MR * PACK initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, { $mr * $pack }>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn load_scores(self, scores: &[f32; $mr]) -> Self::Float {
            Self::Float::from_array(self, *scores)
        }

        #[inline(always)]
        fn store_scores(self, scores: Self::Float, out: &mut [f32; $mr]) {
            *out = scores.to_array();
        }

        #[inline(always)]
        fn compensate_max(
            self,
            acc: Self::Acc,
            a: Coefficients<Self::Float>,
            b: &MinMaxCompensation,
            k: usize,
            best: Self::Float,
        ) -> Self::Float {
            let dot: Self::Float = acc.simd_cast();
            let a_b = Self::Float::splat(self, b.a);
            let b_b = Self::Float::splat(self, b.b);
            let n_b = Self::Float::splat(self, b.n);
            let dim = Self::Float::splat(self, k as f32);
            let ip = ((a.a * a_b) * dot + a.n * b_b) + n_b * a.b + (a.b * b_b) * dim;
            best.max_simd_standard(ip)
        }
    };
}

/// # Safety
///
/// `b` must contain at least PACK initialized, readable elements.
#[inline(always)]
unsafe fn load_pack<T: Copy, const PACK: usize>(b: Slice<'_, T>) -> [T; PACK] {
    bounds::check_ge!(b.len(), PACK);
    // SAFETY: The caller provides a full pack; no array alignment is required.
    unsafe { b.as_ptr().cast::<[T; PACK]>().read_unaligned() }
}

/// Pack two i16 bit patterns into the low and high halves of a u32.
///
/// # Safety
///
/// B must contain two readable elements.
#[inline(always)]
unsafe fn pack_i16_pair(b: Slice<'_, i16>) -> u32 {
    // SAFETY: The caller provides two readable elements, including any zero padding.
    let [lo, hi] = unsafe { load_pack::<_, 2>(b) };
    u32::from(lo as u16) | (u32::from(hi as u16) << 16)
}

impl ExtraWide<16, 4> for Scalar {
    impl_simd_ops!(16, 4);
    const MAX_B_BITS: usize = 7;

    type AElem = u8;
    type BElem = i8;
    type Wide = u8x64<Scalar>;
    type Splat = i8x64<Scalar>;
    type Acc = i32x16<Scalar>;
    type Float = f32x16<Scalar>;

    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    fn convert_a(code: u8) -> u8 {
        code
    }

    fn convert_b(code: u8) -> i8 {
        code as i8
    }

    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
        // SAFETY: The caller provides four readable elements, including any zero padding.
        let values = unsafe { load_pack::<_, 4>(b) };
        let quad = u32::from_le_bytes(values.map(|x| x as u8));
        u32x16::<Scalar>::splat(self, quad).reinterpret_simd()
    }
}

impl ExtraWide<16, 2> for Scalar {
    impl_simd_ops!(16, 2);
    const MAX_B_BITS: usize = 8;

    type AElem = i16;
    type BElem = i16;
    type Wide = i16x32<Scalar>;
    type Splat = Self::Wide;
    type Acc = i32x16<Scalar>;
    type Float = f32x16<Scalar>;

    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    fn convert_a(code: u8) -> i16 {
        i16::from(code)
    }

    fn convert_b(code: u8) -> i16 {
        i16::from(code)
    }

    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i16>) -> Self::Splat {
        // SAFETY: Inherited from the caller.
        u32x16::<Scalar>::splat(self, unsafe { pack_i16_pair(b) }).reinterpret_simd()
    }
}

impl ExtraWide<8, 2> for Scalar {
    impl_simd_ops!(8, 2);
    const MAX_B_BITS: usize = 8;
    type AElem = i16;
    type BElem = i16;
    type Wide = i16x16<Scalar>;
    type Splat = Self::Wide;
    type Acc = i32x8<Scalar>;
    type Float = f32x8<Scalar>;
    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    fn convert_a(code: u8) -> i16 {
        i16::from(code)
    }
    fn convert_b(code: u8) -> i16 {
        i16::from(code)
    }
    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i16>) -> Self::Splat {
        // SAFETY: The caller provides two readable elements.
        u32x8::<Scalar>::splat(self, unsafe { pack_i16_pair(b) }).reinterpret_simd()
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::SplitJoin;
    use diskann_wide::arch::x86_64::{V3, V4};

    impl ExtraWide<8, 4> for V3 {
        impl_simd_ops!(8, 4);
        const MAX_B_BITS: usize = 6;
        type AElem = u8;
        type BElem = i8;
        type Wide = u8x32<V3>;
        type Splat = i8x32<V3>;
        type Acc = i32x8<V3>;
        type Float = f32x8<V3>;
        fn convert_a(code: u8) -> u8 {
            code
        }
        fn convert_b(code: u8) -> i8 {
            code as i8
        }
        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            // The largest pair is 2 * 255 * 63 = 32130, below i16 saturation.
            acc.dot_simd(self.mm256_maddubs_epi16(a, b), i16x16::<V3>::splat(self, 1))
        }
        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements.
            let values = unsafe { load_pack::<_, 4>(b) };
            let quad = u32::from_le_bytes(values.map(|v| v as u8));
            u32x8::<V3>::splat(self, quad).reinterpret_simd()
        }
    }

    impl ExtraWide<8, 2> for V3 {
        impl_simd_ops!(8, 2);
        const MAX_B_BITS: usize = 8;
        type AElem = i16;
        type BElem = i16;
        type Wide = i16x16<V3>;
        type Splat = Self::Wide;
        type Acc = i32x8<V3>;
        type Float = f32x8<V3>;
        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> i16 {
            i16::from(code)
        }
        fn convert_b(code: u8) -> i16 {
            i16::from(code)
        }
        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i16>) -> Self::Splat {
            // SAFETY: The caller provides two readable elements.
            u32x8::<V3>::splat(self, unsafe { pack_i16_pair(b) }).reinterpret_simd()
        }
    }

    impl ExtraWide<16, 4> for V3 {
        impl_simd_ops!(16, 4);
        const MAX_B_BITS: usize = 6;

        type AElem = u8;
        type BElem = i8;
        type Wide = u8x64<V3>;
        type Splat = i8x64<V3>;
        type Acc = i32x16<V3>;
        type Float = f32x16<V3>;

        fn convert_a(code: u8) -> u8 {
            code
        }

        fn convert_b(code: u8) -> i8 {
            code as i8
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            let a = a.split();
            let b = b.split();
            let acc = acc.split();
            let ones = i16x16::<V3>::splat(self, 1);
            // Each saturated i16 pair is at most 2 * 255 * 63 = 32130, so it is exact.
            i32x16::<V3>::new(
                acc.lo.dot_simd(self.mm256_maddubs_epi16(a.lo, b.lo), ones),
                acc.hi.dot_simd(self.mm256_maddubs_epi16(a.hi, b.hi), ones),
            )
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements, including any zero padding.
            let values = unsafe { load_pack::<_, 4>(b) };
            let quad = u32::from_le_bytes(values.map(|x| x as u8));
            let half: i8x32<V3> = u32x8::<V3>::splat(self, quad).reinterpret_simd();
            i8x64::<V3>::new(half, half)
        }
    }

    impl ExtraWide<16, 2> for V3 {
        impl_simd_ops!(16, 2);
        const MAX_B_BITS: usize = 8;

        type AElem = i16;
        type BElem = i16;
        type Wide = i16x32<V3>;
        type Splat = Self::Wide;
        type Acc = i32x16<V3>;
        type Float = f32x16<V3>;

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> i16 {
            i16::from(code)
        }

        fn convert_b(code: u8) -> i16 {
            i16::from(code)
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i16>) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            let pair = unsafe { pack_i16_pair(b) };
            let half: i16x16<V3> = u32x8::<V3>::splat(self, pair).reinterpret_simd();
            i16x32::<V3>::new(half, half)
        }
    }

    impl ExtraWide<16, 4> for V4 {
        impl_simd_ops!(16, 4);
        const MAX_B_BITS: usize = 7;

        type AElem = u8;
        type BElem = i8;
        type Wide = u8x64<V4>;
        type Splat = i8x64<V4>;
        type Acc = i32x16<V4>;
        type Float = f32x16<V4>;

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> u8 {
            code
        }

        fn convert_b(code: u8) -> i8 {
            code as i8
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements, including any zero padding.
            let values = unsafe { load_pack::<_, 4>(b) };
            let quad = u32::from_le_bytes(values.map(|x| x as u8));
            u32x16::<V4>::splat(self, quad).reinterpret_simd()
        }
    }

    impl ExtraWide<16, 2> for V4 {
        impl_simd_ops!(16, 2);
        const MAX_B_BITS: usize = 8;

        type AElem = i16;
        type BElem = i16;
        type Wide = i16x32<V4>;
        type Splat = Self::Wide;
        type Acc = i32x16<V4>;
        type Float = f32x16<V4>;

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> i16 {
            i16::from(code)
        }

        fn convert_b(code: u8) -> i16 {
            i16::from(code)
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i16>) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            u32x16::<V4>::splat(self, unsafe { pack_i16_pair(b) }).reinterpret_simd()
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::arch::aarch64::Neon;

    impl ExtraWide<8, 4> for Neon {
        impl_simd_ops!(8, 4);
        const MAX_B_BITS: usize = 8;
        type AElem = u8;
        type BElem = u8;
        type Wide = u8x32<Neon>;
        type Splat = Self::Wide;
        type Acc = u32x8<Neon>;
        type Float = f32x8<Neon>;
        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> u8 {
            code
        }
        fn convert_b(code: u8) -> u8 {
            code
        }
        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements.
            let values = unsafe { load_pack::<_, 4>(b) };
            let quarter = u32x4::<Neon>::splat(self, u32::from_le_bytes(values)).reinterpret_simd();
            u8x32::<Neon>::new(quarter, quarter)
        }
    }

    impl ExtraWide<16, 4> for Neon {
        impl_simd_ops!(16, 4);
        const MAX_B_BITS: usize = 8;

        type AElem = u8;
        type BElem = u8;
        type Wide = u8x64<Neon>;
        type Splat = Self::Wide;
        type Acc = u32x16<Neon>;
        type Float = f32x16<Neon>;

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_a(code: u8) -> u8 {
            code
        }

        fn convert_b(code: u8) -> u8 {
            code
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements, including any zero padding.
            let values = unsafe { load_pack::<_, 4>(b) };
            let quad = u32::from_le_bytes(values);
            let quarter = u32x4::<Neon>::splat(self, quad).reinterpret_simd();
            let half = u8x32::<Neon>::new(quarter, quarter);
            u8x64::<Neon>::new(half, half)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_vector::PureDistanceFunction;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        bits::{Representation, Unsigned},
        distances,
        matrix_kernels::maxsim::test::packed_x_unpacked_test_dims,
        minmax::{DataRef, MinMaxIP},
        multi_vector::{Defaulted, Mat},
    };

    #[test]
    fn i16_pair_preserves_lane_order_and_bits() {
        for (values, expected) in [
            ([1, 255], 0x00ff_0001),
            ([255, 1], 0x0001_00ff),
            ([-1, i16::MIN], 0x8000_ffff),
        ] {
            // SAFETY: values contains both initialized elements of the pair.
            assert_eq!(unsafe { pack_i16_pair(Slice::new(&values)) }, expected);
        }
    }

    fn random_minmax_matrix<const BITS: usize>(
        rows: usize,
        k: usize,
        rng: &mut StdRng,
    ) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut mat = Mat::new(MinMaxMeta::<BITS>::new(rows, k), Defaulted).unwrap();
        for mut row in mat.rows_mut() {
            let a = rng.random_range(0.001..2.0);
            let b = rng.random_range(-3.0..3.0);
            let mut sum = 0;
            for j in 0..k {
                let code = rng.random_range(0..(1 << BITS));
                row.vector_mut().set(j, code).unwrap();
                sum += code;
            }
            row.set_meta(MinMaxCompensation {
                dim: k as u32,
                a,
                b,
                n: a * sum as f32,
                norm_squared: 0.0,
            });
        }
        mat
    }

    fn reference_max_inner_products<const A_BITS: usize, const B_BITS: usize>(
        a: MatRef<'_, MinMaxMeta<A_BITS>>,
        b: MatRef<'_, MinMaxMeta<B_BITS>>,
    ) -> Vec<f32>
    where
        Unsigned: Representation<A_BITS> + Representation<B_BITS>,
    {
        let k = a.repr().intrinsic_dim();
        assert_eq!(k, b.repr().intrinsic_dim());
        a.rows()
            .map(|a| {
                let av = a.vector();
                let am = a.meta();
                b.rows()
                    .map(|b| {
                        let bv = b.vector();
                        let bm = b.meta();
                        let dot: u64 = (0..k)
                            .map(|j| av.get(j).unwrap() as u64 * bv.get(j).unwrap() as u64)
                            .sum();
                        ((am.a * bm.a) * dot as f32 + am.n * bm.b)
                            + bm.n * am.b
                            + (am.b * bm.b) * k as f32
                    })
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect()
    }

    #[track_caller]
    fn assert_scores_close(actual: &[f32], expected: &[f32], context: std::fmt::Arguments<'_>) {
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert!(
                got == want
                    || (got.is_finite()
                        && want.is_finite()
                        && (got - want).abs() <= 1e-5 * want.abs().max(1.0)),
                "{context}: row {i}: got {got}, expected {want}"
            );
        }
    }

    fn checked_driver<
        'a,
        W,
        const A_BITS: usize,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
        a: &'a PackedA<W::AElem, A_BITS, MR, PACK>,
        b: MatRef<'a, MinMaxMeta<B_BITS>>,
        c: &'a mut [f32],
    ) -> Driver<'a, W, B_BITS, MR, NR, PACK>
    where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<A_BITS> + Representation<B_BITS>,
    {
        assert!(a.codes.nrows() > 0);
        assert_eq!(a.codes.nrows(), c.len());
        assert_eq!(a.codes.ncols(), b.repr().intrinsic_dim());
        let k = DimK::new(NonZeroUsize::new(a.codes.ncols()).unwrap());
        // SAFETY: Nonzero A/K and matching dimensions/output were checked above;
        // PackedA's constructor establishes metadata extents and padding.
        unsafe { Driver::<W, B_BITS, MR, NR, PACK>::new(arch, a, b, c, k, Cache::detect()) }
    }

    fn assert_driver_contract<
        W,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
    ) where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<B_BITS>,
    {
        let mut rng = StdRng::seed_from_u64(0x63de91a0ba5c8742);
        for shape in packed_x_unpacked_test_dims(MR, NR) {
            for k in [shape.k, shape.k.next_multiple_of(PACK)] {
                let a = random_minmax_matrix::<8>(shape.total_a_rows, k, &mut rng);
                let b = random_minmax_matrix::<B_BITS>(shape.total_b_cols, k, &mut rng);
                let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
                let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
                let mut guarded = vec![654321.0; a.num_vectors() + 2];
                let mut driver = checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
                    arch,
                    &packed,
                    b.as_view(),
                    &mut guarded[1..=a.num_vectors()],
                );
                driver.b_rows_per_tile = NonZeroUsize::new(shape.b_cols_per_tile).unwrap();
                driver.c.fill(f32::INFINITY);
                driver::Drive::drive(&mut driver);
                assert_scores_close(
                    driver.c,
                    &expected,
                    format_args!(
                        "forced B tiling: {shape:?}, K={k}, MR={MR}, NR={NR}, PACK={PACK}"
                    ),
                );
                assert_eq!(guarded[0], 654321.0);
                assert_eq!(guarded[a.num_vectors() + 1], 654321.0);
            }
        }
        assert_compensated_reduction::<W, B_BITS, MR, NR, PACK>(arch);

        let a = random_minmax_matrix::<8>(MR + 1, 5, &mut rng);
        let b = random_minmax_matrix::<B_BITS>(0, 5, &mut rng);
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut c = vec![f32::INFINITY; a.num_vectors()];
        driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
            arch,
            &packed,
            b.as_view(),
            &mut c,
        ));
        assert_eq!(c, vec![f32::NEG_INFINITY; a.num_vectors()]);
    }

    fn constant_minmax_matrix<const BITS: usize>(
        rows: usize,
        k: usize,
        code: i64,
        a: f32,
        b: f32,
    ) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut mat = Mat::new(MinMaxMeta::<BITS>::new(rows, k), Defaulted).unwrap();
        for mut row in mat.rows_mut() {
            for j in 0..k {
                row.vector_mut().set(j, code).unwrap();
            }
            row.set_meta(MinMaxCompensation {
                dim: k as u32,
                a,
                b,
                n: a * (k as f32 * code as f32),
                norm_squared: 0.0,
            });
        }
        mat
    }

    fn assert_compensated_reduction<
        W,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
    ) where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<B_BITS>,
    {
        let a = constant_minmax_matrix::<8>(MR + 1, 3, 1, 1.0, 0.0);
        let mut b = constant_minmax_matrix::<B_BITS>(NR + 1, 3, (1 << B_BITS) - 1, 1.0, 0.0);
        let winner_scale = (1 << B_BITS) as f32;
        {
            let mut row = b.get_row_mut(0).unwrap();
            row.vector_mut().set(0, 1).unwrap();
            row.vector_mut().set(1, 1).unwrap();
            row.vector_mut().set(2, 1).unwrap();
            row.set_meta(MinMaxCompensation {
                dim: 3,
                a: winner_scale,
                b: 0.0,
                n: 3.0 * winner_scale,
                norm_squared: 0.0,
            });
        }
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut c = vec![0.0; MR + 1];
        let mut driver =
            checked_driver::<_, 8, B_BITS, MR, NR, PACK>(arch, &packed, b.as_view(), &mut c);
        // The first tile contains competing raw dots; its compensated winner must survive the tail.
        driver.b_rows_per_tile = NonZeroUsize::new(NR).unwrap();
        driver::Drive::drive(&mut driver);
        assert_eq!(c, vec![3.0 * winner_scale; MR + 1]);

        let a = constant_minmax_matrix::<8>(MR + 1, 3, 1, 1.0, -2.0);
        let b = constant_minmax_matrix::<B_BITS>((NR - 1).max(1), 3, 1, 1.0, 0.0);
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
            arch,
            &packed,
            b.as_view(),
            &mut c,
        ));
        // Invalid B lanes must not introduce a zero candidate.
        assert_eq!(c, vec![-3.0; MR + 1]);
    }

    #[cfg(not(miri))]
    #[test]
    fn accumulator_is_exact_at_dimension_limit() {
        let arch = Scalar::new();
        let k = max_k::<8, 8>();
        let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let b = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let packed = PackedA::<_, 8, 8, 2>::new(arch, a.as_view()).unwrap();
        let mut c = [0.0];
        driver::Drive::drive(&mut checked_driver::<_, 8, 8, 8, 2, 2>(
            arch,
            &packed,
            b.as_view(),
            &mut c,
        ));
        assert_eq!(c[0], (k * 255 * 255) as f32);
    }

    #[test]
    fn scalar_driver_obeys_maxsim_contract() {
        assert_driver_contract::<Scalar, 4, 16, 2, 4>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_driver_contract::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("AVX2 unavailable; backend not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_driver_contract::<_, 4, 16, 8, 4>(arch);
        } else {
            eprintln!("V4 unavailable; backend not executed");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_driver_contract::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("Neon/dotprod unavailable; backend not executed");
        }
    }

    #[test]
    fn smaller_panels_preserve_tiling_and_nan_reduction() {
        fn check<W, const B_BITS: usize, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<8, PACK> + LoadStore<f32, 8>,
            Unsigned: Representation<B_BITS>,
        {
            assert_driver_contract::<W, B_BITS, 8, NR, PACK>(arch);
            assert_microkernel_nan_reduction::<W, 8, NR, PACK>(arch);
        }
        check::<_, 8, 2, 2>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            check::<_, 4, 4, 4>(arch);
            check::<_, 8, 4, 2>(arch);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            check::<_, 8, 4, 4>(arch);
        }
    }

    fn assert_a_packing<W, const A_BITS: usize, const MR: usize, const PACK: usize>(arch: W)
    where
        W: ExtraWide<MR, PACK>,
        W::AElem: PartialEq + std::fmt::Debug,
        Unsigned: Representation<A_BITS>,
    {
        for (m, k) in [(0, 0), (0, 5), (3, 0)] {
            let a = constant_minmax_matrix::<A_BITS>(m, k, 0, 1.0, 0.0);
            let packed = PackedA::<_, A_BITS, MR, PACK>::new(arch, a.as_view()).unwrap();
            assert_eq!(packed.codes.nrows(), m);
            assert_eq!(packed.codes.ncols(), k);
            assert!(packed::View::from_block_transposed(packed.codes.as_view()).is_none());
        }
        let mut rng = StdRng::seed_from_u64(0x0ae4b97f2c186d35);
        let a = random_minmax_matrix::<A_BITS>(MR + 1, 5, &mut rng);
        let packed = PackedA::<_, A_BITS, MR, PACK>::new(arch, a.as_view()).unwrap();
        let padded_cols = 5usize.div_ceil(PACK) * PACK;
        assert_eq!(packed.codes.nrows(), MR + 1);
        assert_eq!(packed.codes.ncols(), 5);
        assert_eq!(packed.codes.padded_nrows(), 2 * MR);
        assert_eq!(packed.codes.padded_ncols(), padded_cols);
        for row in 0..2 * MR {
            for col in 0..padded_cols {
                let index = (row / MR) * MR * padded_cols
                    + (col / PACK) * MR * PACK
                    + (row % MR) * PACK
                    + col % PACK;
                let expected = if row <= MR && col < 5 {
                    W::convert_a(a.get_row(row).unwrap().vector().get(col).unwrap() as u8)
                } else {
                    W::AElem::default()
                };
                assert_eq!(packed.codes.as_slice()[index], expected);
            }
            let meta = a.get_row(row).map(|v| v.meta()).unwrap_or_default();
            assert_eq!(packed.meta.a[row], meta.a);
            assert_eq!(packed.meta.b[row], meta.b);
            assert_eq!(packed.meta.n[row], meta.n);
        }
    }

    #[test]
    fn a_packing_preserves_codes_metadata_and_padding() {
        macro_rules! check {
            ($($bits:literal),+) => {$(
                assert_a_packing::<_, $bits, 16, 4>(Scalar::new());
                assert_a_packing::<_, $bits, 16, 2>(Scalar::new());
            )+};
        }
        check!(1, 2, 3, 4, 5, 6, 7, 8);
        assert_a_packing::<_, 3, 8, 2>(Scalar::new());
    }

    #[test]
    fn cache_footprints_match_prepared_layouts() {
        for (k, a_bytes, b_bytes) in [(256, 4288, 276), (257, 4352, 280)] {
            let dim = DimK::new(NonZeroUsize::new(k).unwrap());
            assert_eq!(BScratch::<i8>::row_bytes::<4>(dim).value(), b_bytes);
            assert_eq!(BScratch::<u8>::row_bytes::<4>(dim).value(), b_bytes);
            for rows in [1, 16, 17] {
                let a = constant_minmax_matrix::<8>(rows, k, 0, 1.0, 0.0);
                let packed = PackedA::<u8, 8, 16, 4>::new(Scalar::new(), a.as_view()).unwrap();
                assert_eq!(packed.panel_bytes().value(), a_bytes);

                // Preserve the previous view-based cache estimate.
                let view = packed::View::from_block_transposed(packed.codes.as_view()).unwrap();
                assert_eq!(
                    packed.panel_bytes().value(),
                    view.block_stride(dim).bytes().value() + 3 * 16 * size_of::<f32>()
                );
            }
        }
    }

    #[test]
    fn widened_cache_footprints_match_prepared_layouts() {
        for (k, a_bytes, b_bytes) in [(256, 8384, 532), (257, 8448, 536)] {
            let dim = DimK::new(NonZeroUsize::new(k).unwrap());
            assert_eq!(BScratch::<i16>::row_bytes::<2>(dim).value(), b_bytes);
            for rows in [1, 16, 17] {
                let a = constant_minmax_matrix::<8>(rows, k, 255, 1.0, 0.0);
                let packed = PackedA::<i16, 8, 16, 2>::new(Scalar::new(), a.as_view()).unwrap();
                assert_eq!(packed.panel_bytes().value(), a_bytes);
                let view = packed::View::from_block_transposed(packed.codes.as_view()).unwrap();
                assert_eq!(
                    packed.panel_bytes().value(),
                    view.block_stride(dim).bytes().value() + 3 * 16 * size_of::<f32>(),
                );
            }
        }
    }

    #[test]
    fn b_preparation_preserves_metadata_and_zero_padding() {
        assert_b_preparation::<i8, 7, 4>(-1, |code| code as i8);
        assert_b_preparation::<i16, 8, 2>(-1, i16::from);
        assert_b_preparation::<u8, 8, 4>(u8::MAX, |code| code);
        assert_b_preparation::<u16, 3, 4>(u16::MAX, |code| u16::from(code) + 1);
    }

    fn assert_b_preparation<T, const B_BITS: usize, const PACK: usize>(
        sentinel: T,
        convert: impl Fn(u8) -> T,
    ) where
        T: Copy + Default + PartialEq + std::fmt::Debug,
        Unsigned: Representation<B_BITS>,
    {
        let mut scratch = BScratch::<T>::default();
        for dim in [PACK, PACK + 1] {
            let k = DimK::new(NonZeroUsize::new(dim).unwrap());
            let stride = crate::minmax::Data::<B_BITS>::canonical_bytes(dim);
            // Unaligned storage and dirty high bits exercise the packed input boundary.
            let mut storage = vec![0xff; 1 + 3 * stride];
            let bytes = &mut storage[1..];
            for (i, raw) in bytes.chunks_exact_mut(stride).enumerate() {
                let mut row =
                    crate::minmax::DataMutRef::<B_BITS>::from_canonical_front_mut(raw, dim)
                        .unwrap();
                let mut sum = 0;
                let max_code = (1 << B_BITS) - 1;
                for j in 0..dim {
                    let code = (max_code - ((i + j) & max_code)) as i64;
                    row.vector_mut().set(j, code).unwrap();
                    sum += code;
                }
                let scale = (i + 1) as f32 * 0.25;
                row.set_meta(MinMaxCompensation {
                    dim: dim as u32,
                    a: scale,
                    b: i as f32 - 3.0,
                    n: scale * sum as f32,
                    norm_squared: (i * i + 1) as f32,
                });
            }
            let b = MatRef::new(MinMaxMeta::<B_BITS>::new(3, dim), bytes).unwrap();
            for (start, count) in [(0, 3), (2, 1), (1, 2)] {
                scratch.codes.fill(sentinel);
                scratch.meta.fill(MinMaxCompensation::default());
                let mut conversions = 0;
                let tile = scratch.prepare::<B_BITS, PACK>(
                    b,
                    start,
                    NonZeroUsize::new(count).unwrap(),
                    k,
                    |code| {
                        conversions += 1;
                        convert(code)
                    },
                );
                assert_eq!(conversions, count * dim);
                let padded = padded_k::<PACK>(k);
                let row_len = padded.value().get();
                // SAFETY: prepare constructs the view with this padded dimension.
                let codes = unsafe { tile.codes.as_std_slice(padded) };
                assert_eq!(codes.len(), count * row_len);
                assert_eq!(tile.meta.len(), count);
                for i in 0..count {
                    let row = b.get_row(start + i).unwrap();
                    assert_eq!(tile.meta[i], row.meta());
                    for j in 0..dim {
                        assert_eq!(
                            codes[i * row_len + j],
                            convert(row.vector().get(j).unwrap() as u8),
                            "B tile: K={dim}, start={start}, count={count}, row={i}, column={j}"
                        );
                    }
                    assert!(
                        codes[i * row_len + dim..(i + 1) * row_len]
                            .iter()
                            .all(|&v| v == T::default())
                    );
                }
            }
        }
    }

    fn assert_driver_target_shapes<
        W,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
    ) where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<B_BITS>,
    {
        let mut rng = StdRng::seed_from_u64(0x8f61d9c34a257be0);
        let mut shapes: Vec<_> = (1..NR).map(|n| (MR + 1, n, PACK + 1)).collect();
        if !cfg!(miri) {
            shapes.extend([(16, 16, 256), (17, 17, 257)]);
        }
        for (m, n, k) in shapes {
            let a = random_minmax_matrix::<8>(m, k, &mut rng);
            let b = random_minmax_matrix::<B_BITS>(n, k, &mut rng);
            let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
            let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
            let mut c = vec![f32::INFINITY; m];
            driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
                arch,
                &packed,
                b.as_view(),
                &mut c,
            ));
            assert_scores_close(
                &c,
                &expected,
                format_args!("M={m}, N={n}, K={k}, MR={MR}, NR={NR}, PACK={PACK}"),
            );
        }
    }

    #[test]
    fn scalar_driver_matches_reference_at_target_shapes() {
        assert_driver_target_shapes::<_, 4, 16, 2, 4>(Scalar::new());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_matches_reference_at_target_shapes() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_driver_target_shapes::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("Neon/dotprod unavailable; target-shape regression not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_driver_matches_reference_at_target_shapes() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_driver_target_shapes::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("AVX2 unavailable; target-shape regression not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_driver_matches_reference_at_target_shapes() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_driver_target_shapes::<_, 4, 16, 8, 4>(arch);
        } else {
            eprintln!("V4 unavailable; target-shape regression not executed");
        }
    }

    fn assert_microkernel_nan_reduction<W, const MR: usize, const NR: usize, const PACK: usize>(
        arch: W,
    ) where
        W: ExtraWide<MR, PACK>,
    {
        let k = DimK::new(NonZeroUsize::new(1).unwrap());
        let a_codes = vec![W::AElem::default(); MR * PACK];
        let b_codes = vec![W::BElem::default(); NR * PACK];
        arch.run(|| {
            let mut n = [0.0; MR];
            n[MR - 1] = f32::NAN;
            let a_meta = Coefficients {
                a: arch.load_scores(&[1.0; MR]),
                b: arch.load_scores(&[1.0; MR]),
                n: arch.load_scores(&n),
            };
            for (initial, score) in [
                (f32::NEG_INFINITY, f32::NAN),
                (f32::NEG_INFINITY, 5.0),
                (f32::NEG_INFINITY, -5.0),
                (f32::NEG_INFINITY, f32::INFINITY),
                (7.0, f32::NAN),
            ] {
                for column in [0, NR - 1] {
                    let mut b_meta = [MinMaxCompensation {
                        dim: 1,
                        a: 1.0,
                        b: 0.0,
                        n: f32::NAN,
                        norm_squared: 0.0,
                    }; NR];
                    b_meta[column].n = score;
                    let mut c = [initial; MR];
                    let mut kernel = MicroKernel::<W, MR, NR, PACK> {
                        arch,
                        // SAFETY: A is one full MR panel, padded to PACK for logical K=1.
                        a: unsafe { packed::Panel::new(Slice::new(&a_codes), k) },
                        a_meta,
                        // SAFETY: B contains NR rows of PACK initialized elements.
                        b: unsafe {
                            unpacked::Panel::new(Slice::new(&b_codes), padded_k::<PACK>(k))
                        },
                        b_meta: &b_meta,
                        c: &mut c,
                        k,
                    };
                    driver::MicroKernel::micro_kernel(&mut kernel);
                    let mut expected = [initial.max(score); MR];
                    expected[MR - 1] = initial;
                    assert_eq!(
                        c, expected,
                        "NaN reduction: initial={initial}, score={score}, column={column}",
                    );
                }
            }
        });
    }

    #[test]
    fn scalar_microkernel_ignores_nan_candidates() {
        assert_microkernel_nan_reduction::<Scalar, 16, 2, 4>(Scalar::new());
        assert_microkernel_nan_reduction::<Scalar, 16, 4, 2>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_microkernel_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_microkernel_nan_reduction::<_, 16, 4, 4>(arch);
            assert_microkernel_nan_reduction::<_, 16, 4, 2>(arch);
        } else {
            eprintln!("AVX2 unavailable; NaN regression not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_microkernel_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_microkernel_nan_reduction::<_, 16, 8, 4>(arch);
            assert_microkernel_nan_reduction::<_, 16, 8, 2>(arch);
        } else {
            eprintln!("V4 unavailable; NaN regression not executed");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_microkernel_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_microkernel_nan_reduction::<_, 16, 4, 4>(arch);
        } else {
            eprintln!("Neon/dotprod unavailable; NaN regression not executed");
        }
    }

    mod all_widths {
        use super::*;

        fn assert_backend_code_range<W, const MR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<MR, PACK>,
        {
            assert!((1..=8).contains(&W::MAX_B_BITS));
            let max_b = u8::MAX >> (8 - W::MAX_B_BITS);
            arch.run(|| {
                let codes: Vec<_> = (0..MR * PACK)
                    .map(|i| {
                        W::convert_a(if (i / PACK).is_multiple_of(2) {
                            u8::MAX
                        } else {
                            1
                        })
                    })
                    .collect();
                let b_codes = [W::convert_b(max_b); PACK];
                // SAFETY: The buffers contain one complete A pack and B pack.
                let (a, b) = unsafe {
                    (
                        arch.load_a(Slice::new(&codes)),
                        arch.splat(Slice::new(&b_codes)),
                    )
                };
                let acc = arch.dot(arch.zero_acc(), a, b);
                let acc = arch.dot(acc, a, b);
                let coefficients = Coefficients {
                    a: arch.load_scores(&[1.0; MR]),
                    b: arch.load_scores(&[0.0; MR]),
                    n: arch.load_scores(&[0.0; MR]),
                };
                let score = arch.compensate_max(
                    acc,
                    coefficients,
                    &MinMaxCompensation {
                        a: 1.0,
                        ..Default::default()
                    },
                    2 * PACK,
                    arch.load_scores(&[f32::NEG_INFINITY; MR]),
                );
                let mut actual = [0.0; MR];
                arch.store_scores(score, &mut actual);
                assert_eq!(
                    actual,
                    core::array::from_fn(|i| {
                        let a = if i.is_multiple_of(2) { 255 } else { 1 };
                        (2 * PACK * a * usize::from(max_b)) as f32
                    }),
                );
            });
        }

        fn assert_b_width<
            W,
            const B_BITS: usize,
            const MR: usize,
            const NR: usize,
            const PACK: usize,
        >(
            arch: W,
            rng: &mut StdRng,
        ) where
            W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
            Unsigned: Representation<B_BITS>,
        {
            let max_a = u64::from(u8::MAX);
            let max_b = (1u64 << B_BITS) - 1;
            let limit = max_k::<8, B_BITS>() as u64;
            assert!(limit * max_a * max_b <= i32::MAX as u64);
            assert!((limit + 1) * max_a * max_b > i32::MAX as u64);

            let (m, n, k) = (MR + 1, NR + 1, PACK + 1);
            let a = random_minmax_matrix::<8>(m, k, rng);
            let b = random_minmax_matrix::<B_BITS>(n, k, rng);
            let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
            let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
            let mut guarded = vec![654321.0; m + 2];
            let mut kernel = checked_driver::<W, 8, B_BITS, MR, NR, PACK>(
                arch,
                &packed,
                b.as_view(),
                &mut guarded[1..=m],
            );
            driver::Drive::drive(&mut kernel);
            assert_scores_close(
                kernel.c,
                &expected,
                format_args!("8x{B_BITS}: M={m}, N={n}, K={k}, PACK={PACK}"),
            );
            assert_eq!(guarded[0], 654321.0);
            assert_eq!(guarded[m + 1], 654321.0);

            let a = constant_minmax_matrix::<8>(1, k, max_a as i64, 1.0, 0.0);
            let b = constant_minmax_matrix::<B_BITS>(1, k, max_b as i64, 1.0, 0.0);
            let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
            let mut c = [0.0];
            driver::Drive::drive(&mut checked_driver::<W, 8, B_BITS, MR, NR, PACK>(
                arch,
                &packed,
                b.as_view(),
                &mut c,
            ));
            assert_eq!(c, [(k as u64 * max_a * max_b) as f32]);
        }

        macro_rules! check_widths {
            ($arch:expr, $mr:literal, $nr:literal, $pack:literal; $($b:literal),+ $(,)?) => {{
                let arch = $arch;
                let mut rng = StdRng::seed_from_u64(0x948cd3e02a671fb5);
                assert_backend_code_range::<_, $mr, $pack>(arch);
                $(
                    assert_b_width::<_, $b, $mr, $nr, $pack>(arch, &mut rng);
                )+
            }};
        }

        #[test]
        fn scalar_supports_all_b_widths_and_layouts() {
            check_widths!(Scalar::new(), 16, 2, 2; 1, 2, 3, 4, 5, 6, 7, 8);
            check_widths!(Scalar::new(), 16, 2, 4; 1, 2, 3, 4, 5, 6, 7);
            check_widths!(Scalar::new(), 8, 2, 2; 1, 2, 3, 4, 5, 6, 7, 8);
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_supports_all_b_widths_and_layouts() {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_widths!(arch, 16, 4, 2; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, 16, 4, 4; 1, 2, 3, 4, 5, 6);
                check_widths!(arch, 8, 4, 2; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, 8, 4, 4; 1, 2, 3, 4, 5, 6);
            } else {
                eprintln!("AVX2 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx512_supports_all_b_widths_and_layouts() {
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                check_widths!(arch, 16, 8, 2; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, 16, 8, 4; 1, 2, 3, 4, 5, 6, 7);
            } else {
                eprintln!("V4 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn neon_supports_all_b_widths_and_layouts() {
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                check_widths!(arch, 16, 4, 4; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, 8, 4, 4; 1, 2, 3, 4, 5, 6, 7, 8);
            } else {
                eprintln!("Neon/dotprod unavailable; all-width regression not executed");
            }
        }

        fn check_reference<const A_BITS: usize, const B_BITS: usize>()
        where
            Unsigned: Representation<A_BITS> + Representation<B_BITS>,
            MinMaxIP: for<'a, 'b> PureDistanceFunction<
                    DataRef<'a, A_BITS>,
                    DataRef<'b, B_BITS>,
                    distances::MathematicalResult<f32>,
                >,
        {
            let mut rng = StdRng::seed_from_u64(0x14a7f0b2c69d538e);
            for k in [3, 65] {
                let a = random_minmax_matrix::<A_BITS>(3, k, &mut rng);
                let b = random_minmax_matrix::<B_BITS>(5, k, &mut rng);
                let expected: Vec<_> = a
                    .rows()
                    .map(|a| {
                        b.rows()
                            .map(|b| {
                                <MinMaxIP as PureDistanceFunction<
                                    DataRef<'_, A_BITS>,
                                    DataRef<'_, B_BITS>,
                                    distances::MathematicalResult<f32>,
                                >>::evaluate(a, b)
                                .unwrap()
                                .into_inner()
                            })
                            .fold(f32::NEG_INFINITY, f32::max)
                    })
                    .collect();
                assert_scores_close(
                    &reference_max_inner_products::<A_BITS, B_BITS>(a.as_view(), b.as_view()),
                    &expected,
                    format_args!("scalar reference: {A_BITS}x{B_BITS}, K={k}"),
                );
            }
        }

        #[test]
        fn reference_agrees_with_existing_pairwise_combinations() {
            check_reference::<1, 1>();
            check_reference::<2, 2>();
            check_reference::<3, 3>();
            check_reference::<4, 4>();
            check_reference::<5, 5>();
            check_reference::<6, 6>();
            check_reference::<7, 7>();
            check_reference::<8, 8>();
            check_reference::<8, 4>();
            check_reference::<8, 2>();
            check_reference::<8, 1>();
        }

        #[cfg(not(miri))]
        #[test]
        fn low_bit_inputs_can_exceed_the_eight_bit_dimension_limit() {
            fn check<const A_BITS: usize, const B_BITS: usize>()
            where
                Unsigned: Representation<A_BITS> + Representation<B_BITS>,
            {
                let k = max_k::<8, 8>() + 1;
                let max_a = (1 << A_BITS) - 1;
                let max_b = (1 << B_BITS) - 1;
                let a = constant_minmax_matrix::<A_BITS>(1, k, max_a, 1.0, 0.0);
                let b = constant_minmax_matrix::<B_BITS>(1, k, max_b, 1.0, 0.0);
                let packed = PackedA::<_, A_BITS, 16, 2>::new(Scalar::new(), a.as_view()).unwrap();
                let mut c = [0.0];
                driver::Drive::drive(&mut checked_driver::<_, A_BITS, B_BITS, 16, 2, 2>(
                    Scalar::new(),
                    &packed,
                    b.as_view(),
                    &mut c,
                ));
                assert_eq!(c[0], (k as i64 * max_a * max_b) as f32);
            }
            check::<1, 8>();
            check::<8, 1>();
        }
    }

    mod minmax8 {
        use super::*;

        fn check_backend<W, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
        {
            assert_driver_contract::<W, 8, 16, NR, PACK>(arch);
            assert_driver_target_shapes::<W, 8, 16, NR, PACK>(arch);
            assert_unsigned_byte_products::<W, NR, PACK>(arch);
        }

        fn assert_unsigned_byte_products<W, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
        {
            if cfg!(miri) {
                return;
            }
            for k in [PACK - 1, PACK, PACK + 1] {
                let boundary_codes = [0, 1, 127, 128, 255];
                let mut a = constant_minmax_matrix::<8>(17, k, 0, 1.0, 0.0);
                for (i, mut row) in a.rows_mut().enumerate() {
                    let code = boundary_codes[i % boundary_codes.len()];
                    for j in 0..k {
                        row.vector_mut().set(j, code).unwrap();
                    }
                    let mut meta = row.meta();
                    meta.n = (k as i64 * code) as f32;
                    row.set_meta(meta);
                }
                let a = PackedA::<_, 8, 16, PACK>::new(arch, a.as_view()).unwrap();
                let mut c = vec![f32::NAN; 17];
                for code in boundary_codes {
                    let b = constant_minmax_matrix::<8>(1, k, code, 1.0, 0.0);
                    driver::Drive::drive(&mut checked_driver::<W, 8, 8, 16, NR, PACK>(
                        arch,
                        &a,
                        b.as_view(),
                        &mut c,
                    ));
                    for (i, &actual) in c.iter().enumerate() {
                        assert_eq!(
                            actual,
                            (boundary_codes[i % boundary_codes.len()] as usize * code as usize * k)
                                as f32,
                            "A={}, B={code}, K={k}",
                            boundary_codes[i % boundary_codes.len()],
                        );
                    }
                }
            }
        }

        #[test]
        fn scalar_driver_obeys_maxsim_contract() {
            check_backend::<_, 2, 2>(Scalar::new());
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_backend::<_, 4, 2>(arch);
            } else {
                eprintln!("AVX2 unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx512_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                check_backend::<_, 8, 2>(arch);
            } else {
                eprintln!("V4 unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn neon_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                check_backend::<_, 4, 4>(arch);
            } else {
                eprintln!("Neon/dotprod unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(not(miri))]
        #[test]
        #[should_panic(expected = "exceeds the MinMax8 x MinMax8 limit")]
        fn driver_rejects_k_above_accumulator_limit() {
            let k = max_k::<8, 8>() + 1;
            let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let b = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let a = PackedA::<_, 8, 16, 2>::new(Scalar::new(), a.as_view()).unwrap();
            let mut c = [0.0];
            let _ = checked_driver::<_, 8, 8, 16, 2, 2>(Scalar::new(), &a, b.as_view(), &mut c);
        }
    }
}
