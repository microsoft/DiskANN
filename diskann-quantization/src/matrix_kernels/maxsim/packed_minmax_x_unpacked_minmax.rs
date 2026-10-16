/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MaxSim for prepacked MinMax A and tiled MinMax B, with 1-8 bits per input code.
//!
//! Each B tile is decoded once and reused across A panels. Integer inner products
//! are compensated before reduction because B's coefficients vary by vector.
//!
//! Outputs are unnegated maximum inner products. NaN candidates are ignored;
//! empty B or all-NaN comparisons leave negative infinity.
//!
//! Scalar and x86 provide a general i16 path (PACK=2) and a byte path (PACK=4).
//! The byte path supports B up to six bits on V3 and seven on Scalar/V4.
//! Neon uses unsigned byte dots (PACK=4) for every bit width.
//! A packing depends only on A's bit width and the selected numeric layout, so it
//! can be reused across B formats supported by that layout.

use std::{
    num::NonZeroUsize,
    ops::{Add, Mul},
};

use diskann_wide::{
    SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDReinterpret, SIMDVector,
    arch::{Architecture, Scalar, Target},
};

use crate::{
    bits::{BitSlice, Representation, Unsigned},
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
diskann_wide::alias!(i8x32<W> = i8x32);
diskann_wide::alias!(i8x64<W> = i8x64);
diskann_wide::alias!(i16x16<W> = i16x16);
diskann_wide::alias!(i16x32<W> = i16x32);
diskann_wide::alias!(i32x16<W> = i32x16);
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
struct Coefficients<T> {
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
    a_meta: Coefficients<&'a [f32]>,
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
            a: a.meta.a.as_slice(),
            b: a.meta.b.as_slice(),
            n: a.meta.n.as_slice(),
        };
        let a_bytes = a.panel_bytes();
        let a = packed::View::from_block_transposed(a.codes.as_view())
            .expect("A and K must be nonempty");
        for meta in [a_meta.a, a_meta.b, a_meta.n] {
            bounds::check_eq!(bounds::Bound::new(meta.len()), MR * a.blocks().get());
        }
        let b_rows_per_tile =
            Params::new(cache, a_bytes, BScratch::<W::BElem>::row_bytes(k), NR).b_cols_in_l1;
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
                    let b = scratch.prepare::<W, B_BITS, MR, PACK>(self.b, start, count, self.k);
                    // SAFETY: Driver's construction establishes all dimensions, metadata
                    // extents, and C's coverage of A. Visitors preserve these bounds.
                    unsafe {
                        self.a.visit_panels(self.k, |a, panel| {
                            let row = panel * MR;
                            let end = (row + MR).min(self.c.len());
                            let c = self.arch.load(&self.c[row..end]);
                            let a_meta = Coefficients {
                                a: W::Float::load_simd(self.arch, self.a_meta.a.as_ptr().add(row)),
                                b: W::Float::load_simd(self.arch, self.a_meta.b.as_ptr().add(row)),
                                n: W::Float::load_simd(self.arch, self.a_meta.n.as_ptr().add(row)),
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

impl<T: Copy + Default> BScratch<T> {
    #[expect(clippy::expect_used)]
    fn prepare<W, const B_BITS: usize, const MR: usize, const PACK: usize>(
        &mut self,
        b: MatRef<'_, MinMaxMeta<B_BITS>>,
        start: usize,
        count: NonZeroUsize,
        k: DimK,
    ) -> BTile<'_, T>
    where
        W: ExtraWide<MR, PACK, BElem = T>,
        Unsigned: Representation<B_BITS>,
    {
        let dim = k.value().get();
        assert_eq!(b.repr().intrinsic_dim(), dim);
        assert!(start <= b.num_vectors() && count.get() <= b.num_vectors() - start);
        self.codes.resize(count.get() * dim, T::default());
        self.meta.resize(count.get(), MinMaxCompensation::default());
        for i in 0..count.get() {
            let row = b.get_row(start + i).expect("B tile range was checked");
            self.meta[i] = row.meta();
            decode_b::<W, B_BITS, MR, PACK>(row.vector(), &mut self.codes[i * dim..(i + 1) * dim]);
        }
        // SAFETY: The initialized codes contain count rows of exactly dim elements.
        let codes = unsafe { unpacked::View::new(Slice::new(&self.codes), count, k) };
        BTile {
            codes,
            meta: &self.meta,
        }
    }

    /// Cache-planning bytes for one decoded B row, including metadata.
    fn row_bytes(k: DimK) -> Bytes {
        let codes = Elements::<T>::new(k.value().get()).bytes();
        Bytes::new(codes.value() + size_of::<MinMaxCompensation>())
    }
}

/// Expand a B row, ignoring unused padding bits.
///
/// # Panics
///
/// Panics if the logical input and output lengths differ.
#[inline(always)]
#[expect(clippy::expect_used)]
fn decode_b<W, const B_BITS: usize, const MR: usize, const PACK: usize>(
    codes: BitSlice<'_, B_BITS, Unsigned>,
    dst: &mut [W::BElem],
) where
    W: ExtraWide<MR, PACK>,
    Unsigned: Representation<B_BITS>,
{
    const {
        assert!(
            B_BITS <= W::MAX_B_BITS,
            "B bit width exceeds the backend's supported range"
        )
    };
    assert_eq!(codes.len(), dst.len());
    let packed = codes.as_slice();
    if B_BITS == 8 {
        for (out, &code) in dst.iter_mut().zip(packed) {
            *out = W::convert_b(code);
        }
    } else if B_BITS == 4 {
        let (pairs, tail) = dst.as_chunks_mut::<2>();
        for (pair, &byte) in pairs.iter_mut().zip(packed) {
            *pair = [W::convert_b(byte & 15), W::convert_b(byte >> 4)];
        }
        if let Some(last) = tail.first_mut() {
            *last = W::convert_b(packed[pairs.len()] & 15);
        }
    } else {
        for (i, out) in dst.iter_mut().enumerate() {
            *out = W::convert_b(codes.get(i).expect("input and output lengths match") as u8);
        }
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
        const { assert!(NR > 0 && NR <= 8) };
        let mut visitor = Visitor {
            arch: self.arch,
            a: self.a,
            a_meta: self.a_meta,
            b_meta: self.b_meta,
            c: &mut self.c,
            k: self.k,
        };
        // SAFETY: PanelKernel is constructed with equal dimensions and matching B metadata.
        let tail = unsafe {
            self.b.visit_panels::<NR>(
                self.k,
                Visitor {
                    c: &mut *visitor.c,
                    ..visitor
                },
            )
        };
        if let Some(tail) = tail {
            macro_rules! remainder {
                ($($n:literal),+) => {$(
                    if let Some(b) = tail.try_as_panel::<$n>() {
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
/// `a` must contain at least `W::Wide::LANES` readable elements. `valid <= PACK`,
/// and for every `j < NR`, `b.add(bstride * j)` must contain `valid` readable elements.
#[inline(always)]
unsafe fn accumulate_pack<W, const MR: usize, const NR: usize, const PACK: usize>(
    arch: W,
    a: Slice<'_, W::AElem>,
    b: Slice<'_, W::BElem>,
    bstride: Elements<W::BElem>,
    valid: usize,
    acc: &mut [W::Acc; NR],
) where
    W: ExtraWide<MR, PACK>,
{
    // SAFETY: The caller provides a full padded A pack.
    let a = unsafe { W::Wide::load_simd(arch, a.as_ptr()) };
    for (j, acc) in acc.iter_mut().enumerate() {
        // SAFETY: The caller guarantees valid elements in every B column.
        let b = unsafe { arch.splat(b.add(bstride * j), valid) };
        *acc = arch.dot(*acc, a, b);
    }
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    driver::MicroKernel for MicroKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn micro_kernel(&mut self) {
        const {
            assert!(W::Float::LANES == MR);
            assert!(W::Wide::LANES == MR * PACK);
        }
        bounds::check_eq!(self.a.k(), self.k);
        bounds::check_eq!(self.b.k(), self.k);
        let k = self.k.value().get();
        let mut acc = [W::Acc::default(self.arch); NR];
        let ap = self.a.as_ptr();
        let bp = self.b.as_ptr();
        let astride = self.a.pack_stride();
        let bstride = self.b.stride(self.k);
        let full = k / PACK;

        for pack in 0..full {
            // SAFETY: A contains a full pack; every B column has PACK logical elements.
            unsafe {
                accumulate_pack::<W, MR, NR, PACK>(
                    self.arch,
                    ap.add(astride * pack),
                    bp.add(Elements::new(pack * PACK)),
                    bstride,
                    PACK,
                    &mut acc,
                );
            }
        }
        if !k.is_multiple_of(PACK) {
            // SAFETY: A's last pack is padded. B has k % PACK remaining logical elements;
            // splat zero-fills the rest, without reading a padded B extent.
            unsafe {
                accumulate_pack::<W, MR, NR, PACK>(
                    self.arch,
                    ap.add(astride * full),
                    bp.add(Elements::new(full * PACK)),
                    bstride,
                    k % PACK,
                    &mut acc,
                );
            }
        }

        // SAFETY: Float has exactly MR lanes, and C is an MR-element array.
        let mut best = unsafe { W::Float::load_simd(self.arch, self.c.as_ptr()) };
        for (acc, b) in acc.into_iter().zip(self.b_meta) {
            let dot = acc.simd_cast();
            let a_b = W::Float::splat(self.arch, b.a);
            let b_b = W::Float::splat(self.arch, b.b);
            let n_b = W::Float::splat(self.arch, b.n);
            let dim = W::Float::splat(self.arch, k as f32);
            let a = self.a_meta;
            let ip = ((a.a * a_b) * dot + a.n * b_b) + n_b * a.b + (a.b * b_b) * dim;
            best = best.max_simd_standard(ip);
        }
        // SAFETY: C has exactly Float::LANES writable elements.
        unsafe { best.store_simd(self.c.as_mut_ptr()) };
    }
}

//----------//
// Backends //
//----------//

/// SIMD types for MR A rows and PACK contraction elements. NR counts accumulators,
/// not lanes, so it does not change these types.
///
/// All implementations accept A codes in 0..=255. B's range is limited by
/// both its element type and any intermediate arithmetic in the dot product.
pub(crate) trait ExtraWide<const MR: usize, const PACK: usize>: Architecture {
    /// Largest complete unsigned B width supported for every A code in 0..=255.
    /// This is a backend capability, not the actual A or B input width.
    const MAX_B_BITS: usize;

    type AElem: Copy + Default;
    type BElem: Copy + Default;
    type Wide: SIMDVector<Arch = Self, Scalar = Self::AElem>;
    type Splat: SIMDVector<Arch = Self, Scalar = Self::BElem>;
    type Acc: SIMDVector<Arch = Self>
        + SIMDDotProduct<Self::Wide, Self::Splat>
        + SIMDCast<f32, Cast = Self::Float>;
    type Float: SIMDVector<Arch = Self, Scalar = f32>
        + SIMDMinMax
        + Add<Output = Self::Float>
        + Mul<Output = Self::Float>;

    fn convert_a(code: u8) -> Self::AElem;
    /// `code` fits in MAX_B_BITS unsigned bits.
    fn convert_b(code: u8) -> Self::BElem;

    /// Accumulate codes produced by `convert_a` and `convert_b`.
    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    /// # Safety
    ///
    /// `valid <= PACK` and the first `valid` elements of B must be readable.
    unsafe fn splat(self, b: Slice<'_, Self::BElem>, valid: usize) -> Self::Splat;
}

/// # Safety
///
/// `valid <= PACK` and the first `valid` elements of B must be readable.
#[inline(always)]
unsafe fn group<T: Copy + Default, const PACK: usize>(b: Slice<'_, T>, valid: usize) -> [T; PACK] {
    core::array::from_fn(|i| {
        if i < valid {
            // SAFETY: i < valid, so the element is readable.
            unsafe { *b.add(Elements::new(i)).as_unit().as_ref() }
        } else {
            T::default()
        }
    })
}

/// # Safety
///
/// `valid <= 2` and the first `valid` elements of B must be readable.
#[inline(always)]
unsafe fn pair(b: Slice<'_, i16>, valid: usize) -> u32 {
    // SAFETY: Inherited from the caller; group zero-fills a partial pair.
    let [lo, hi] = unsafe { group::<_, 2>(b, valid) };
    u32::from(lo as u16) | (u32::from(hi as u16) << 16)
}

impl ExtraWide<16, 4> for Scalar {
    const MAX_B_BITS: usize = 7;

    type AElem = u8;
    type BElem = i8;
    type Wide = u8x64<Scalar>;
    type Splat = i8x64<Scalar>;
    type Acc = i32x16<Scalar>;
    type Float = f32x16<Scalar>;

    fn convert_a(code: u8) -> u8 {
        code
    }

    fn convert_b(code: u8) -> i8 {
        code as i8
    }

    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i8>, valid: usize) -> Self::Splat {
        // SAFETY: Inherited from the caller.
        let values = unsafe { group::<_, 4>(b, valid) };
        let quad = u32::from_le_bytes(values.map(|x| x as u8));
        u32x16::<Scalar>::splat(self, quad).reinterpret_simd()
    }
}

impl ExtraWide<16, 2> for Scalar {
    const MAX_B_BITS: usize = 8;

    type AElem = i16;
    type BElem = i16;
    type Wide = i16x32<Scalar>;
    type Splat = Self::Wide;
    type Acc = i32x16<Scalar>;
    type Float = f32x16<Scalar>;

    fn convert_a(code: u8) -> i16 {
        i16::from(code)
    }

    fn convert_b(code: u8) -> i16 {
        i16::from(code)
    }

    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i16>, valid: usize) -> Self::Splat {
        // SAFETY: Inherited from the caller.
        u32x16::<Scalar>::splat(self, unsafe { pair(b, valid) }).reinterpret_simd()
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;
    use diskann_wide::SplitJoin;
    use diskann_wide::arch::x86_64::{V3, V4};

    impl ExtraWide<16, 4> for V3 {
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
        unsafe fn splat(self, b: Slice<'_, i8>, valid: usize) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            let values = unsafe { group::<_, 4>(b, valid) };
            let quad = u32::from_le_bytes(values.map(|x| x as u8));
            let half: i8x32<V3> = u32x8::<V3>::splat(self, quad).reinterpret_simd();
            i8x64::<V3>::new(half, half)
        }
    }

    impl ExtraWide<16, 2> for V3 {
        const MAX_B_BITS: usize = 8;

        type AElem = i16;
        type BElem = i16;
        type Wide = i16x32<V3>;
        type Splat = Self::Wide;
        type Acc = i32x16<V3>;
        type Float = f32x16<V3>;

        fn convert_a(code: u8) -> i16 {
            i16::from(code)
        }

        fn convert_b(code: u8) -> i16 {
            i16::from(code)
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i16>, valid: usize) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            let pair = unsafe { pair(b, valid) };
            let half: i16x16<V3> = u32x8::<V3>::splat(self, pair).reinterpret_simd();
            i16x32::<V3>::new(half, half)
        }
    }

    impl ExtraWide<16, 4> for V4 {
        const MAX_B_BITS: usize = 7;

        type AElem = u8;
        type BElem = i8;
        type Wide = u8x64<V4>;
        type Splat = i8x64<V4>;
        type Acc = i32x16<V4>;
        type Float = f32x16<V4>;

        fn convert_a(code: u8) -> u8 {
            code
        }

        fn convert_b(code: u8) -> i8 {
            code as i8
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i8>, valid: usize) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            let values = unsafe { group::<_, 4>(b, valid) };
            let quad = u32::from_le_bytes(values.map(|x| x as u8));
            u32x16::<V4>::splat(self, quad).reinterpret_simd()
        }
    }

    impl ExtraWide<16, 2> for V4 {
        const MAX_B_BITS: usize = 8;

        type AElem = i16;
        type BElem = i16;
        type Wide = i16x32<V4>;
        type Splat = Self::Wide;
        type Acc = i32x16<V4>;
        type Float = f32x16<V4>;

        fn convert_a(code: u8) -> i16 {
            i16::from(code)
        }

        fn convert_b(code: u8) -> i16 {
            i16::from(code)
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i16>, valid: usize) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            u32x16::<V4>::splat(self, unsafe { pair(b, valid) }).reinterpret_simd()
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;
    use diskann_wide::arch::aarch64::Neon;

    impl ExtraWide<16, 4> for Neon {
        const MAX_B_BITS: usize = 8;

        type AElem = u8;
        type BElem = u8;
        type Wide = u8x64<Neon>;
        type Splat = Self::Wide;
        type Acc = u32x16<Neon>;
        type Float = f32x16<Neon>;

        fn convert_a(code: u8) -> u8 {
            code
        }

        fn convert_b(code: u8) -> u8 {
            code
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, u8>, valid: usize) -> Self::Splat {
            // SAFETY: Inherited from the caller.
            let quad = u32::from_le_bytes(unsafe { group::<_, 4>(b, valid) });
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

    const MAX_K_8X4: usize = max_k::<8, 4>();

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
            for k in [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 127, 129] {
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
                driver::Drive::drive(&mut driver);
                assert_scores_close(
                    driver.c,
                    &expected,
                    format_args!("default tiling: {shape:?}, K={k}, MR={MR}, NR={NR}, PACK={PACK}"),
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
                // Reset stale scores once per invocation, never once per B tile.
                driver.c.fill(f32::INFINITY);
                driver::Drive::drive(&mut driver);
                assert_scores_close(
                    driver.c,
                    &expected,
                    format_args!("repeated drive: {shape:?}, K={k}, MR={MR}, NR={NR}, PACK={PACK}"),
                );
                assert_eq!(guarded[0], 654321.0);
                assert_eq!(guarded[a.num_vectors() + 1], 654321.0);
            }
        }
        assert_compensated_reduction::<W, B_BITS, MR, NR, PACK>(arch);
        assert_accumulator_limit_is_exact::<W, B_BITS, MR, NR, PACK>(arch);

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
        let mut b = constant_minmax_matrix::<B_BITS>(NR + 1, 3, 15, 1.0, 0.0);
        {
            let mut row = b.get_row_mut(0).unwrap();
            row.vector_mut().set(0, 1).unwrap();
            row.vector_mut().set(1, 1).unwrap();
            row.vector_mut().set(2, 1).unwrap();
            row.set_meta(MinMaxCompensation {
                dim: 3,
                a: 100.0,
                b: 0.0,
                n: 300.0,
                norm_squared: 0.0,
            });
        }
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut c = vec![0.0; MR + 1];
        let mut driver =
            checked_driver::<_, 8, B_BITS, MR, NR, PACK>(arch, &packed, b.as_view(), &mut c);
        // The compensated winner is in the first tile, not the last.
        driver.b_rows_per_tile = NonZeroUsize::new(1).unwrap();
        driver::Drive::drive(&mut driver);
        assert_eq!(c, vec![300.0; MR + 1]);

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

    fn assert_accumulator_limit_is_exact<
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
        if cfg!(miri) {
            return;
        }
        let k = max_k::<8, B_BITS>();
        let max_b = (1 << B_BITS) - 1;
        let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let b = constant_minmax_matrix::<B_BITS>(1, k, max_b as i64, 1.0, 0.0);
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut c = [0.0];
        driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
            arch,
            &packed,
            b.as_view(),
            &mut c,
        ));
        assert_eq!(c[0], (k * 255 * max_b) as f32);
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

    fn assert_a_packing<W, const A_BITS: usize, const MR: usize, const PACK: usize>(arch: W)
    where
        W: ExtraWide<MR, PACK>,
        W::AElem: PartialEq + std::fmt::Debug,
        Unsigned: Representation<A_BITS>,
    {
        for (m, k) in [(0, 0), (0, 5), (3, 0), (0, MAX_K_8X4), (0, MAX_K_8X4 + 1)] {
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
    fn packing_preserves_codes_coefficients_and_zero_padding() {
        assert_a_packing::<_, 8, 16, 4>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_a_packing::<_, 8, 16, 4>(arch);
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_a_packing::<_, 8, 16, 4>(arch);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_a_packing::<_, 8, 16, 4>(arch);
        }
    }

    #[test]
    fn cache_footprints_match_prepared_layouts() {
        for (k, a_bytes, b_bytes) in [(256, 4288, 276), (257, 4352, 277)] {
            let dim = DimK::new(NonZeroUsize::new(k).unwrap());
            assert_eq!(BScratch::<i8>::row_bytes(dim).value(), b_bytes);
            assert_eq!(BScratch::<u8>::row_bytes(dim).value(), b_bytes);
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
        for (k, a_bytes, b_bytes) in [(256, 8384, 532), (257, 8448, 534)] {
            let dim = DimK::new(NonZeroUsize::new(k).unwrap());
            assert_eq!(BScratch::<i16>::row_bytes(dim).value(), b_bytes);
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

    #[cfg(not(miri))]
    #[test]
    #[should_panic(expected = "exceeds the MinMax8 x MinMax4 limit")]
    fn driver_rejects_k_above_accumulator_limit() {
        let k = MAX_K_8X4 + 1;
        let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let b = constant_minmax_matrix::<4>(1, k, 15, 1.0, 0.0);
        let packed = PackedA::<u8, 8, 16, 4>::new(Scalar::new(), a.as_view()).unwrap();
        let mut c = [123.0];
        let _ = checked_driver::<_, 8, 4, 16, 2, 4>(Scalar::new(), &packed, b.as_view(), &mut c);
    }

    #[test]
    fn b_preparation_masks_padding_and_reuses_storage() {
        let k = 5;
        let stride = crate::minmax::Data::<4>::canonical_bytes(k);
        let mut bytes = vec![0xff; stride * 3];
        for (row_index, raw) in bytes.chunks_exact_mut(stride).enumerate() {
            let mut row = crate::minmax::DataMutRef::<4>::from_canonical_front_mut(raw, k).unwrap();
            let a = (row_index + 1) as f32;
            row.set_meta(MinMaxCompensation {
                dim: k as u32,
                a,
                b: -(row_index as f32),
                n: a * (10 + 5 * row_index) as f32,
                ..Default::default()
            });
            for i in 0..k {
                row.vector_mut().set(i, (i + row_index) as i64).unwrap();
            }
        }
        let b = MatRef::new(MinMaxMeta::<4>::new(3, k), &bytes).unwrap();
        let mut scratch = BScratch::<i8>::default();
        let dim = DimK::new(NonZeroUsize::new(k).unwrap());
        let tile = scratch.prepare::<Scalar, 4, 16, 4>(b, 0, NonZeroUsize::new(3).unwrap(), dim);
        assert_eq!(
            // SAFETY: dim is the dimension passed to prepare.
            unsafe { tile.codes.as_std_slice(dim) },
            &[0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6]
        );
        let ptr = scratch.codes.as_ptr();
        let tile = scratch.prepare::<Scalar, 4, 16, 4>(b, 2, NonZeroUsize::new(1).unwrap(), dim);
        assert_eq!(tile.meta, &[b.get_row(2).unwrap().meta()]);
        assert_eq!(
            // SAFETY: dim is the dimension passed to prepare.
            unsafe { tile.codes.as_std_slice(dim) },
            &[2, 3, 4, 5, 6]
        );
        assert_eq!(scratch.codes.as_ptr(), ptr);
    }

    fn assert_b_decoding_boundaries<
        W,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
        sentinel: W::BElem,
    ) where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        W::BElem: PartialEq + std::fmt::Debug,
        Unsigned: Representation<B_BITS>,
    {
        decode_b::<W, B_BITS, MR, PACK>(BitSlice::new(&[] as &[u8], 0).unwrap(), &mut []);
        let mut rng = StdRng::seed_from_u64(0x8af450de13c769b2);
        let mut scratch = BScratch::<W::BElem>::default();
        let dimensions: &[usize] = if cfg!(miri) {
            &[1, 2, 3, 4, 5, 15, 16, 17]
        } else {
            &[
                1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 250, 255, 256, 257, 511,
                512, 513,
            ]
        };
        for &dim in dimensions {
            let k = DimK::new(NonZeroUsize::new(dim).unwrap());
            let a = random_minmax_matrix::<8>(MR + 1, dim, &mut rng);
            let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
            let stride = crate::minmax::Data::<B_BITS>::canonical_bytes(dim);
            for offset in [0, 1, 3, 15] {
                let mut storage = vec![0xff; offset + 5 * stride];
                let bytes = &mut storage[offset..];
                for (i, raw) in bytes.chunks_exact_mut(stride).enumerate() {
                    let mut row =
                        crate::minmax::DataMutRef::<B_BITS>::from_canonical_front_mut(raw, dim)
                            .unwrap();
                    let mut sum = 0;
                    for j in 0..dim {
                        // Cover every code value, and every packed byte for MinMax4.
                        let code = if B_BITS == 4 {
                            let byte = (i * 37 + j / 2) & 255;
                            (byte >> (4 * (j % 2))) & 15
                        } else {
                            (i * 37 + j) & ((1 << B_BITS) - 1)
                        } as i64;
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
                let b = MatRef::new(MinMaxMeta::<B_BITS>::new(5, dim), bytes).unwrap();
                for row in b.rows() {
                    let codes = row.vector();
                    if B_BITS == 4 && dim % 2 != 0 {
                        assert_eq!(codes.as_slice().last().unwrap() >> 4, 15);
                    }
                    let mut guarded = vec![sentinel; dim + 2];
                    decode_b::<W, B_BITS, MR, PACK>(codes, &mut guarded[1..=dim]);
                    for j in 0..dim {
                        assert_eq!(
                            guarded[j + 1],
                            W::convert_b(codes.get(j).unwrap() as u8),
                            "row decode: K={dim}, offset={offset}, column={j}, MR={MR}, PACK={PACK}"
                        );
                    }
                    assert_eq!(guarded[0], sentinel);
                    assert_eq!(guarded[dim + 1], sentinel);
                }
                for (start, count) in [(0, 1), (1, 3), (4, 1), (0, 5)] {
                    scratch.codes.fill(sentinel);
                    scratch.meta.fill(MinMaxCompensation::default());
                    let tile = scratch.prepare::<W, B_BITS, MR, PACK>(
                        b,
                        start,
                        NonZeroUsize::new(count).unwrap(),
                        k,
                    );
                    // SAFETY: k is the dimension used to prepare this tile.
                    let codes = unsafe { tile.codes.as_std_slice(k) };
                    assert_eq!(codes.len(), count * dim);
                    assert_eq!(tile.meta.len(), count);
                    for i in 0..count {
                        let row = b.get_row(start + i).unwrap();
                        assert_eq!(tile.meta[i], row.meta());
                        for j in 0..dim {
                            assert_eq!(
                                codes[i * dim + j],
                                W::convert_b(row.vector().get(j).unwrap() as u8),
                                "B tile: K={dim}, offset={offset}, start={start}, \
                                 count={count}, row={i}, column={j}, MR={MR}, PACK={PACK}"
                            );
                        }
                    }
                }
                let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b);
                let mut guarded = vec![654321.0; MR + 3];
                for tile_rows in [3, 4, 5] {
                    let mut driver = checked_driver::<W, 8, B_BITS, MR, NR, PACK>(
                        arch,
                        &packed,
                        b,
                        &mut guarded[1..=MR + 1],
                    );
                    driver.b_rows_per_tile = NonZeroUsize::new(tile_rows).unwrap();
                    driver.c.fill(f32::INFINITY);
                    driver::Drive::drive(&mut driver);
                    assert_scores_close(
                        driver.c,
                        &expected,
                        format_args!(
                            "B decoding: K={dim}, offset={offset}, B tile={tile_rows}, \
                             MR={MR}, NR={NR}, PACK={PACK}"
                        ),
                    );
                    assert_eq!(guarded[0], 654321.0);
                    assert_eq!(guarded[MR + 2], 654321.0);
                }
            }
        }
    }

    #[test]
    fn scalar_b_decoding_preserves_rows_tails_and_scores() {
        assert_b_decoding_boundaries::<_, 4, 16, 2, 4>(Scalar::new(), -1);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_b_decoding_preserves_rows_tails_and_scores() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_b_decoding_boundaries::<_, 4, 16, 4, 4>(arch, -1);
        } else {
            eprintln!("AVX2 unavailable; B decoding regression not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_b_decoding_preserves_rows_tails_and_scores() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_b_decoding_boundaries::<_, 4, 16, 8, 4>(arch, -1);
        } else {
            eprintln!("V4 unavailable; B decoding regression not executed");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_b_decoding_preserves_rows_tails_and_scores() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_b_decoding_boundaries::<_, 4, 16, 4, 4>(arch, u8::MAX);
        } else {
            eprintln!("Neon/dotprod unavailable; B decoding regression not executed");
        }
    }

    #[test]
    #[should_panic]
    fn b_decoding_rejects_a_short_output() {
        let codes = BitSlice::new(&[0x21u8][..], 2).unwrap();
        decode_b::<Scalar, 4, 16, 4>(codes, &mut [0]);
    }

    #[test]
    #[should_panic]
    fn b_decoding_rejects_a_long_output() {
        let codes = BitSlice::new(&[0xf1u8][..], 1).unwrap();
        decode_b::<Scalar, 4, 16, 4>(codes, &mut [0; 2]);
    }

    fn assert_packed_a_reuse<
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
        let k = 17;
        let mut rng = StdRng::seed_from_u64(0xa619e2c5f387b04d);
        let a = random_minmax_matrix::<8>(11, k, &mut rng);
        let b = random_minmax_matrix::<B_BITS>(9, k, &mut rng);
        let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        drop(a);
        let mut actual = vec![0.0; 11];
        for _ in 0..2 {
            actual.fill(f32::INFINITY);
            driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
                arch,
                &packed,
                b.as_view(),
                &mut actual,
            ));
            assert_scores_close(
                &actual,
                &expected,
                format_args!("packed-A reuse: MR={MR}, NR={NR}, PACK={PACK}"),
            );
        }
    }

    #[test]
    fn packed_a_outlives_source_and_supports_repeated_drives() {
        assert_packed_a_reuse::<_, 4, 16, 2, 4>(Scalar::new());
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_packed_a_reuse::<_, 4, 16, 4, 4>(arch);
        }
        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_packed_a_reuse::<_, 4, 16, 8, 4>(arch);
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_packed_a_reuse::<_, 4, 16, 4, 4>(arch);
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
        let dimensions: &[usize] = if cfg!(miri) { &[3] } else { &[3, 256, 257] };
        for m in [1, 16, 17] {
            for n in (1..=NR + 1).chain([16, 17]) {
                for &k in dimensions {
                    let a = random_minmax_matrix::<8>(m, k, &mut rng);
                    let b = random_minmax_matrix::<B_BITS>(n, k, &mut rng);
                    let expected =
                        reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
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
        }
        let a = constant_minmax_matrix::<8>(16, 257, 255, 1.0, 0.0);
        let max_b = (1 << B_BITS) - 1;
        let b = constant_minmax_matrix::<B_BITS>(16, 257, max_b, 1.0, 0.0);
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut c = [0.0; 16];
        driver::Drive::drive(&mut checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
            arch,
            &packed,
            b.as_view(),
            &mut c,
        ));
        assert_eq!(c, [(257 * 255 * max_b) as f32; 16]);
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

    fn assert_nan_candidates_preserve_maxima<
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
        let mut a = constant_minmax_matrix::<8>(MR + 1, 1, 1, 1.0, 0.0);
        // Exercise mixed SIMD lanes as well as the partial A panel.
        for row in [1, MR] {
            a.get_row_mut(row).unwrap().set_meta(MinMaxCompensation {
                dim: 1,
                a: 1.0,
                b: 0.0,
                n: f32::NAN,
                norm_squared: 0.0,
            });
        }
        let packed = PackedA::<_, 8, MR, PACK>::new(arch, a.as_view()).unwrap();
        let mut cases = vec![
            vec![],
            vec![10.0, f32::NAN, 5.0],
            vec![f32::NAN, 10.0, 5.0],
            vec![10.0, 5.0, f32::NAN],
            vec![10.0, f32::NAN, f32::NAN, 5.0],
            vec![-5.0, f32::NAN, -10.0],
            vec![f32::NAN, f32::NAN],
        ];
        let n = 2 * NR + 1;
        for nan_index in [0, NR, n - 1] {
            let mut scores = vec![5.0; n];
            scores[if nan_index == 0 { 1 } else { 0 }] = 10.0;
            scores[nan_index] = f32::NAN;
            cases.push(scores);
        }

        for scores in cases {
            let mut b = constant_minmax_matrix::<B_BITS>(scores.len(), 1, 0, 1.0, 0.0);
            for (mut row, offset) in b.rows_mut().zip(scores.iter().copied()) {
                row.set_meta(MinMaxCompensation {
                    dim: 1,
                    a: 1.0,
                    b: offset,
                    n: 0.0,
                    norm_squared: 0.0,
                });
            }
            let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
            assert_eq!(
                expected[0],
                scores.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
            assert_eq!(expected[1], f32::NEG_INFINITY);
            assert_eq!(expected[MR], f32::NEG_INFINITY);
            for tile_rows in [1, 2, NR, NR + 1, scores.len().max(1)] {
                let mut c = vec![f32::NAN; MR + 1];
                let mut driver = checked_driver::<_, 8, B_BITS, MR, NR, PACK>(
                    arch,
                    &packed,
                    b.as_view(),
                    &mut c,
                );
                driver.b_rows_per_tile = NonZeroUsize::new(tile_rows).unwrap();
                driver::Drive::drive(&mut driver);
                assert_scores_close(
                    &c,
                    &expected,
                    format_args!(
                        "NaN reduction: scores={scores:?}, B tile={tile_rows}, \
                         MR={MR}, NR={NR}, PACK={PACK}"
                    ),
                );
            }
        }
    }

    #[test]
    fn scalar_driver_ignores_nan_candidates() {
        assert_nan_candidates_preserve_maxima::<Scalar, 4, 16, 2, 4>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_driver_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_nan_candidates_preserve_maxima::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("AVX2 unavailable; NaN regression not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_driver_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_nan_candidates_preserve_maxima::<_, 4, 16, 8, 4>(arch);
        } else {
            eprintln!("V4 unavailable; NaN regression not executed");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_ignores_nan_candidates() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_nan_candidates_preserve_maxima::<_, 4, 16, 4, 4>(arch);
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
                let a = W::Wide::splat(arch, W::convert_a(u8::MAX));
                let b = W::Splat::splat(arch, W::convert_b(max_b));
                let dot = arch.dot(W::Acc::default(arch), a, b).simd_cast();
                let mut actual = [0.0; MR];
                assert_eq!(W::Float::LANES, MR);
                // SAFETY: actual has exactly Float::LANES writable elements.
                unsafe { dot.store_simd(actual.as_mut_ptr()) };
                assert_eq!(
                    actual,
                    [(PACK * usize::from(u8::MAX) * usize::from(max_b)) as f32; MR],
                );
            });
        }

        fn assert_bit_pair<
            W,
            const A_BITS: usize,
            const B_BITS: usize,
            const MR: usize,
            const NR: usize,
            const PACK: usize,
        >(
            arch: W,
            rng: &mut StdRng,
        ) where
            W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
            W::AElem: PartialEq + std::fmt::Debug,
            Unsigned: Representation<A_BITS> + Representation<B_BITS>,
        {
            let max_a = (1u64 << A_BITS) - 1;
            let max_b = (1u64 << B_BITS) - 1;
            let limit = max_k::<A_BITS, B_BITS>() as u64;
            assert!(limit * max_a * max_b <= i32::MAX as u64);
            assert!((limit + 1) * max_a * max_b > i32::MAX as u64);
            assert_a_packing::<W, A_BITS, MR, PACK>(arch);

            let dimensions: &[usize] = if cfg!(miri) {
                &[1, 3]
            } else {
                &[1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 33, 255, 256, 257]
            };
            for (m, n) in [
                (1, 0),
                (1, 1),
                (MR - 1, (NR - 1).max(1)),
                (MR, NR),
                (MR + 1, 2 * NR + 1),
            ] {
                for &k in dimensions {
                    let mut a = random_minmax_matrix::<A_BITS>(m, k, rng);
                    let mut b = random_minmax_matrix::<B_BITS>(n, k, rng);
                    if m > 1 {
                        let mut row = a.get_row_mut(m - 1).unwrap();
                        let mut meta = row.meta();
                        meta.n = f32::NAN;
                        row.set_meta(meta);
                    }
                    if n > 1 {
                        let mut row = b.get_row_mut(n - 1).unwrap();
                        let mut meta = row.meta();
                        meta.n = f32::NAN;
                        row.set_meta(meta);
                    }
                    let expected =
                        reference_max_inner_products::<A_BITS, B_BITS>(a.as_view(), b.as_view());
                    let packed = PackedA::<_, A_BITS, MR, PACK>::new(arch, a.as_view()).unwrap();
                    let mut guarded = vec![654321.0; m + 2];
                    let mut kernel = checked_driver::<W, A_BITS, B_BITS, MR, NR, PACK>(
                        arch,
                        &packed,
                        b.as_view(),
                        &mut guarded[1..=m],
                    );
                    for tile_rows in [kernel.b_rows_per_tile, NonZeroUsize::new(NR + 1).unwrap()] {
                        kernel.b_rows_per_tile = tile_rows;
                        kernel.c.fill(f32::INFINITY);
                        driver::Drive::drive(&mut kernel);
                        assert_scores_close(
                            kernel.c,
                            &expected,
                            format_args!(
                                "{A_BITS}x{B_BITS}: M={m}, N={n}, K={k}, B tile={tile_rows}, PACK={PACK}"
                            ),
                        );
                    }
                    assert_eq!(guarded[0], 654321.0);
                    assert_eq!(guarded[m + 1], 654321.0);
                }
            }

            let dimensions: &[usize] = if cfg!(miri) {
                &[PACK, PACK + 1]
            } else {
                &[PACK, PACK + 1, 257]
            };
            for &k in dimensions {
                let a = constant_minmax_matrix::<A_BITS>(MR + 1, k, max_a as i64, 1.0, 0.0);
                let b = constant_minmax_matrix::<B_BITS>(NR + 1, k, max_b as i64, 1.0, 0.0);
                let packed = PackedA::<_, A_BITS, MR, PACK>::new(arch, a.as_view()).unwrap();
                let mut c = vec![0.0; MR + 1];
                driver::Drive::drive(&mut checked_driver::<W, A_BITS, B_BITS, MR, NR, PACK>(
                    arch,
                    &packed,
                    b.as_view(),
                    &mut c,
                ));
                assert_eq!(c, vec![(k as u64 * max_a * max_b) as f32; MR + 1]);
            }
        }

        macro_rules! check_widths {
            ($arch:expr, $sentinel:expr, $nr:literal, $pack:literal; $($b:literal),+ $(,)?) => {{
                let arch = $arch;
                let mut rng = StdRng::seed_from_u64(0x948cd3e02a671fb5);
                assert_backend_code_range::<_, 16, $pack>(arch);
                $(
                    assert_b_decoding_boundaries::<_, $b, 16, $nr, $pack>(arch, $sentinel);
                    assert_bit_pair::<_, 1, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 2, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 3, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 4, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 5, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 6, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 7, $b, 16, $nr, $pack>(arch, &mut rng);
                    assert_bit_pair::<_, 8, $b, 16, $nr, $pack>(arch, &mut rng);
                )+
            }};
        }

        #[test]
        fn scalar_supports_all_bit_pairs_and_byte_layouts() {
            check_widths!(Scalar::new(), -1, 2, 2; 1, 2, 3, 4, 5, 6, 7, 8);
            check_widths!(Scalar::new(), -1, 2, 4; 1, 2, 3, 4, 5, 6, 7);
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_supports_all_bit_pairs_and_byte_layouts() {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_widths!(arch, -1, 4, 2; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, -1, 4, 4; 1, 2, 3, 4, 5, 6);
            } else {
                eprintln!("AVX2 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx512_supports_all_bit_pairs_and_byte_layouts() {
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                check_widths!(arch, -1, 8, 2; 1, 2, 3, 4, 5, 6, 7, 8);
                check_widths!(arch, -1, 8, 4; 1, 2, 3, 4, 5, 6, 7);
            } else {
                eprintln!("V4 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn neon_supports_all_bit_pairs() {
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                check_widths!(arch, u8::MAX, 4, 4; 1, 2, 3, 4, 5, 6, 7, 8);
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
            for k in [1, 3, 17, 257] {
                let a = random_minmax_matrix::<A_BITS>(17, k, &mut rng);
                let b = random_minmax_matrix::<B_BITS>(19, k, &mut rng);
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

        #[test]
        fn packed_a_is_reused_across_b_widths() {
            let arch = Scalar::new();
            let mut rng = StdRng::seed_from_u64(0xac21f49d70b365e8);
            let a = random_minmax_matrix::<3>(17, 33, &mut rng);
            let packed = PackedA::<_, 3, 16, 2>::new(arch, a.as_view()).unwrap();
            let mut c = vec![f32::NAN; 17];
            macro_rules! check {
                ($($b:literal),+) => {$(
                    let b = random_minmax_matrix::<$b>(9, 33, &mut rng);
                    let expected = reference_max_inner_products::<3, $b>(a.as_view(), b.as_view());
                    driver::Drive::drive(&mut checked_driver::<_, 3, $b, 16, 2, 2>(
                        arch, &packed, b.as_view(), &mut c,
                    ));
                    assert_scores_close(&c, &expected, format_args!("reused A3 with B{}", $b));
                )+};
            }
            check!(1, 2, 3, 4, 5, 6, 7, 8);
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
            check::<4, 8>();
            check::<8, 1>();
        }
    }

    mod minmax8 {
        use super::*;

        fn check_backend<W, const NR: usize, const PACK: usize>(arch: W, sentinel: W::BElem)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
            W::AElem: PartialEq + std::fmt::Debug,
            W::BElem: PartialEq + std::fmt::Debug,
        {
            assert_a_packing::<W, 8, 16, PACK>(arch);
            assert_driver_contract::<W, 8, 16, NR, PACK>(arch);
            assert_driver_target_shapes::<W, 8, 16, NR, PACK>(arch);
            assert_nan_candidates_preserve_maxima::<W, 8, 16, NR, PACK>(arch);
            assert_packed_a_reuse::<W, 8, 16, NR, PACK>(arch);
            assert_b_decoding_boundaries::<W, 8, 16, NR, PACK>(arch, sentinel);
            assert_unsigned_byte_products::<W, NR, PACK>(arch);
        }

        fn assert_unsigned_byte_products<W, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
        {
            if cfg!(miri) {
                return;
            }
            for k in [1, 2, 3, 4, 5, 257, 259] {
                let mut a = constant_minmax_matrix::<8>(257, k, 0, 1.0, 0.0);
                for (i, mut row) in a.rows_mut().enumerate() {
                    let code = (i % 256) as i64;
                    for j in 0..k {
                        row.vector_mut().set(j, code).unwrap();
                    }
                    let mut meta = row.meta();
                    meta.n = (k as i64 * code) as f32;
                    row.set_meta(meta);
                }
                let a = PackedA::<_, 8, 16, PACK>::new(arch, a.as_view()).unwrap();
                let mut c = vec![f32::NAN; 257];
                for code in 0..=255 {
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
                            ((i % 256) * code as usize * k) as f32,
                            "A={}, B={code}, K={k}",
                            i % 256,
                        );
                    }
                }
            }
        }

        #[test]
        fn scalar_driver_obeys_maxsim_contract() {
            check_backend::<_, 2, 2>(Scalar::new(), -1);
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                check_backend::<_, 4, 2>(arch, -1);
            } else {
                eprintln!("AVX2 unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx512_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                check_backend::<_, 8, 2>(arch, -1);
            } else {
                eprintln!("V4 unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn neon_driver_obeys_maxsim_contract() {
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                check_backend::<_, 4, 4>(arch, u8::MAX);
            } else {
                eprintln!("Neon/dotprod unavailable; MinMax8 x MinMax8 backend not executed");
            }
        }

        #[cfg(not(miri))]
        #[test]
        #[should_panic(expected = "exceeds the MinMax8 x MinMax8 limit")]
        fn driver_rejects_k_above_accumulator_limit() {
            let k = 33026;
            let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let b = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let a = PackedA::<_, 8, 16, 2>::new(Scalar::new(), a.as_view()).unwrap();
            let mut c = [0.0];
            let _ = checked_driver::<_, 8, 8, 16, 2, 2>(Scalar::new(), &a, b.as_view(), &mut c);
        }

        #[test]
        #[should_panic]
        fn b_decoding_rejects_a_short_output() {
            let codes = BitSlice::new(&[128u8, 255][..], 2).unwrap();
            decode_b::<Scalar, 8, 16, 2>(codes, &mut [0]);
        }

        #[test]
        #[should_panic]
        fn b_decoding_rejects_a_long_output() {
            let codes = BitSlice::new(&[255u8][..], 1).unwrap();
            decode_b::<Scalar, 8, 16, 2>(codes, &mut [0; 2]);
        }
    }
}
