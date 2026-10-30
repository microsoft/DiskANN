/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MaxSim for packed MinMax A and tiled MinMax B.
//!
//! B tiles are decoded once and reused across A panels. Compensation precedes
//! reduction because coefficients differ per B vector. Codes are zero-padded to PACK;
//! compensation and accumulator limits use logical K.
//!
//! Outputs are unnegated maximum inner products. NaN candidates are ignored;
//! empty B or all-NaN comparisons leave negative infinity.

use std::{
    num::NonZeroUsize,
    ops::{Add, Mul},
};

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_wide::{
    Const, SIMDCast, SIMDDotProduct, SIMDMinMax, SIMDReinterpret, SIMDVector, SupportedLaneCount,
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
    minmax::{Coefficients, DataRef, MinMaxMeta},
    multi_vector::{BlockTransposed, MatRef, Overflow},
};

use super::packed_f32_x_unpacked_f32::Params;

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
pub(crate) const fn max_k(a_bits: usize, b_bits: usize) -> usize {
    assert!(a_bits > 0 && a_bits <= 8 && b_bits > 0 && b_bits <= 8);
    (i32::MAX as usize) / (((1 << a_bits) - 1) * ((1 << b_bits) - 1))
}

#[derive(Debug)]
pub(crate) struct PackedA<T: Copy, const MR: usize, const PACK: usize> {
    codes: BlockTransposed<T, MR, PACK>,
    meta: Vec<Coefficients<[f32; MR]>>,
    a_bits: usize,
}

impl<T, const MR: usize, const PACK: usize> PackedA<T, MR, PACK>
where
    T: Copy + Default + From<u8>,
{
    /// Zero-pad partial rows and K packs.
    ///
    /// # Errors
    ///
    /// Returns [`Overflow`] if the packed data or coefficient allocation size overflows.
    #[expect(clippy::expect_used)]
    pub(crate) fn new<const A_BITS: usize>(
        a: MatRef<'_, MinMaxMeta<A_BITS>>,
    ) -> Result<Self, Overflow>
    where
        Unsigned: Representation<A_BITS>,
    {
        let k = a.repr().intrinsic_dim();
        let mut codes = BlockTransposed::try_new(a.num_vectors(), k)?;
        let panels = codes.num_blocks();
        Overflow::check::<Coefficients<[f32; MR]>>(panels, 1)?;
        let mut meta = vec![
            Coefficients {
                a: [0.0; MR],
                b: [0.0; MR],
                n: [0.0; MR],
            };
            panels
        ];
        for (i, row) in a.rows().enumerate() {
            let m = row.meta();
            let coefficients = &mut meta[i / MR];
            coefficients.a[i % MR] = m.a;
            coefficients.b[i % MR] = m.b;
            coefficients.n[i % MR] = m.n;
            let v = row.vector();
            let mut dst = codes
                .get_row_mut(i)
                .expect("A and its packing have equal row counts");
            for j in 0..k {
                // SAFETY: MinMaxMeta exposes exactly k codes for every row.
                dst.set(j, T::from(unsafe { v.get_unchecked(j) } as u8));
            }
        }
        Ok(Self {
            codes,
            meta,
            a_bits: A_BITS,
        })
    }

    fn as_view(&self) -> Option<AView<'_, T, MR, PACK>> {
        Some(AView {
            codes: packed::View::from_block_transposed(self.codes.as_view())?,
            meta: &self.meta,
            k: DimK::new(NonZeroUsize::new(self.codes.ncols())?),
        })
    }

    fn panel_bytes(&self) -> Bytes {
        let codes = Elements::<T>::new(MR * self.codes.padded_ncols()).bytes();
        Bytes::new(codes.value() + size_of::<Coefficients<[f32; MR]>>())
    }
}

#[derive(Clone, Copy)]
struct AView<'a, T, const MR: usize, const PACK: usize> {
    codes: packed::View<'a, T, MR, PACK>,
    meta: &'a [Coefficients<[f32; MR]>],
    k: DimK,
}

impl<T: Copy, const MR: usize, const PACK: usize> AView<'_, T, MR, PACK> {
    #[inline]
    fn visit_panels(&self, mut visitor: impl FnMut(APanel<'_, T, MR, PACK>, usize)) {
        // SAFETY: PackedA constructs codes and coefficients with the same panel count and K.
        unsafe {
            self.codes.visit_panels(self.k, |codes, i| {
                visitor(
                    APanel {
                        codes,
                        meta: &self.meta[i],
                    },
                    i,
                );
            });
        }
    }
}

#[derive(Clone, Copy)]
struct APanel<'a, T, const MR: usize, const PACK: usize> {
    codes: packed::Panel<'a, T, MR, PACK>,
    meta: &'a Coefficients<[f32; MR]>,
}

//--------//
// Driver //
//--------//

/// Each drive overwrites C; results do not accumulate across calls.
pub(crate) struct Driver<
    'a,
    W: ExtraWide<MR, PACK>,
    const MR: usize,
    const NR: usize,
    const PACK: usize,
> {
    arch: W,
    a: AView<'a, W::AElem, MR, PACK>,
    b: rowmajor::Ref<'a, u8>,
    decode_tile: DecodeTileFn<W, W::BElem>,
    c: &'a mut [f32],
    b_rows_per_tile: NonZeroUsize,
}

impl<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    Driver<'a, W, MR, NR, PACK>
{
    /// # Panics
    ///
    /// Panics if K exceeds the accumulator limit for these bit widths or the
    /// canonical document storage size overflows.
    /// Valid integer dot products may still round when converted to f32.
    ///
    /// # Safety
    ///
    /// A must have nonzero row count and dimension, B must have the same dimension,
    /// and C must have exactly A's logical row count.
    #[expect(clippy::expect_used)]
    pub(crate) unsafe fn new<const B_BITS: usize>(
        arch: W,
        a: &'a PackedA<W::AElem, MR, PACK>,
        b: MatRef<'a, MinMaxMeta<B_BITS>>,
        c: &'a mut [f32],
        cache: Cache,
    ) -> Self
    where
        Unsigned: Representation<B_BITS>,
        Self: driver::Drive,
    {
        const {
            assert!(
                B_BITS <= W::MAX_B_BITS,
                "B bit width exceeds the backend's supported range"
            )
        };
        let rows = a.codes.nrows();
        let a_bytes = a.panel_bytes();
        let a_bits = a.a_bits;
        let a = a.as_view().expect("A and K must be nonempty");
        let dim = a.k.value().get();
        bounds::check_eq!(bounds::Bound::new(b.repr().intrinsic_dim()), a.k);
        bounds::check_eq!(bounds::Bound::new(c.len()), rows);
        let max = max_k(a_bits, B_BITS);
        assert!(
            dim <= max,
            "contraction dimension {dim} exceeds the MinMax{a_bits} x MinMax{B_BITS} limit of {max}",
        );
        let b = b
            .as_canonical_view()
            .expect("canonical document storage size must not overflow");
        let b_rows_per_tile = Params::new(
            cache,
            a_bytes,
            BScratch::<W::BElem>::row_bytes::<PACK>(a.k),
            NR,
        )
        .b_cols_in_l1;
        Self {
            arch,
            a,
            b,
            decode_tile: |arch: W, source, k, codes, coefficients| {
                arch.run(
                    #[inline(always)]
                    || decode_tile::<B_BITS, _>(source, k, codes, coefficients, W::convert_b),
                )
            },
            c,
            b_rows_per_tile,
        }
    }
}

impl<W, const MR: usize, const NR: usize, const PACK: usize> driver::Drive
    for Driver<'_, W, MR, NR, PACK>
where
    W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
    for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
{
    fn drive(&mut self) {
        self.arch.run(
            #[inline]
            || {
                self.c.fill(f32::NEG_INFINITY);
                let mut scratch = BScratch::<W::BElem>::default();
                let mut start = 0;
                while let Some(remaining) = NonZeroUsize::new(self.b.nrows() - start) {
                    let count = remaining.min(self.b_rows_per_tile);
                    #[expect(
                        clippy::expect_used,
                        reason = "count is bounded by the remaining rows"
                    )]
                    let source = self
                        .b
                        .subview(start..start + count.get())
                        .expect("B tile in bounds");
                    let b = scratch.prepare::<PACK>(
                        source,
                        self.a.k,
                        |source, k, codes, coefficients| {
                            (self.decode_tile)(self.arch, source, k, codes, coefficients)
                        },
                    );
                    self.a.visit_panels(|a, panel| {
                        let row = panel * MR;
                        let end = (row + MR).min(self.c.len());
                        let c = self.arch.load(&self.c[row..end]);
                        // SAFETY: A and B share K by the driver contract.
                        let kernel =
                            unsafe { PanelKernel::<W, MR, NR, PACK>::new(self.arch, a, b, c) };
                        let c = self.arch.run(kernel);
                        self.arch.store(c, &mut self.c[row..end]);
                    });
                    start += count.get();
                }
            },
        );
    }
}

#[derive(Default)]
struct BScratch<T> {
    codes: Vec<T>,
    meta: Vec<Coefficients<f32>>,
}

// Only decoding is specialized by bit width; allocation, padding, and tile views are shared.
type DecodeTileFn<W, T> = fn(
    arch: W,
    source: rowmajor::Ref<'_, u8>,
    k: DimK,
    codes: rowmajor::Mut<'_, T>,
    coefficients: &mut [Coefficients<f32>],
);

/// Decode canonical rows into the logical columns of an already allocated tile.
/// Inlined into the selected architecture's target-feature entry.
#[inline(always)]
#[expect(
    clippy::expect_used,
    reason = "the driver supplies canonical rows and output dimensions are checked"
)]
fn decode_tile<const B_BITS: usize, T>(
    source: rowmajor::Ref<'_, u8>,
    k: DimK,
    mut codes: rowmajor::Mut<'_, T>,
    coefficients: &mut [Coefficients<f32>],
    mut convert: impl FnMut(u8) -> T,
) where
    Unsigned: Representation<B_BITS>,
{
    let dim = k.value().get();
    assert_eq!(source.nrows(), codes.nrows(), "one output row per document");
    assert_eq!(
        source.nrows(),
        coefficients.len(),
        "one coefficient record per document"
    );
    assert!(
        dim <= codes.ncols(),
        "tile rows must hold the logical dimension"
    );

    for ((bytes, dst), meta) in source.rows().zip(codes.rows_mut()).zip(coefficients) {
        let row =
            DataRef::<B_BITS>::from_canonical_front(bytes, dim).expect("canonical document row");
        *meta = row.meta().into();
        row.vector()
            .decode_into(&mut dst[..dim], &mut convert)
            .expect("B output must match the logical code count");
    }
}

/// Codes and coefficients share row order; code rows are zero-padded to PACK.
#[derive(Clone, Copy)]
struct BTile<'a, T, const PACK: usize> {
    codes: unpacked::View<'a, T>,
    meta: &'a [Coefficients<f32>],
    k: DimK,
    physical_k: DimK,
}

impl<T: Copy, const PACK: usize> BTile<'_, T, PACK> {
    /// # Safety
    ///
    /// `codes` must be the NR-row panel of this tile starting at `start`.
    #[inline(always)]
    unsafe fn bind_panel<'a, const NR: usize>(
        &'a self,
        codes: unpacked::Panel<'a, T, NR>,
        start: usize,
    ) -> BPanel<'a, T, NR, PACK> {
        // SAFETY: Preparation gives each code row one coefficient record in the same order.
        let meta = unsafe {
            &*self
                .meta
                .as_ptr()
                .add(start)
                .cast::<[Coefficients<f32>; NR]>()
        };
        BPanel {
            codes,
            meta,
            k: self.k,
            physical_k: self.physical_k,
        }
    }
}

struct BPanel<'a, T, const NR: usize, const PACK: usize> {
    codes: unpacked::Panel<'a, T, NR>,
    meta: &'a [Coefficients<f32>; NR],
    k: DimK,
    physical_k: DimK,
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
    /// Prepare a nonempty tile. The decoder writes logical columns; padding bypasses it.
    ///
    /// # Panics
    ///
    /// Panics if the source tile is empty or its output storage size overflows.
    #[expect(clippy::expect_used)]
    fn prepare<const PACK: usize>(
        &mut self,
        source: rowmajor::Ref<'_, u8>,
        k: DimK,
        decode: impl FnOnce(rowmajor::Ref<'_, u8>, DimK, rowmajor::Mut<'_, T>, &mut [Coefficients<f32>]),
    ) -> BTile<'_, T, PACK> {
        let count = NonZeroUsize::new(source.nrows()).expect("B tile must be nonempty");
        let dim = k.value().get();
        let physical_k = padded_k::<PACK>(k);
        let stride = physical_k.value().get();
        self.codes.resize(count.get() * stride, T::default());
        self.meta.resize(count.get(), Coefficients::default());
        let mut codes = rowmajor::Mut::try_from_data(&mut self.codes, count.get(), stride)
            .expect("resized tile storage matches its shape");
        // Empty per-row fills can still lower to memset calls.
        if dim < stride {
            for dst in codes.rows_mut() {
                dst[dim..].fill(T::default());
            }
        }
        decode(source, k, codes, &mut self.meta);
        BTile {
            // SAFETY: The buffers contain count rows, each with stride initialized codes.
            codes: unsafe { unpacked::View::new(Slice::new(&self.codes), count, physical_k) },
            meta: &self.meta,
            k,
            physical_k,
        }
    }

    fn row_bytes<const PACK: usize>(k: DimK) -> Bytes {
        let codes = Elements::<T>::new(padded_k::<PACK>(k).value().get()).bytes();
        Bytes::new(codes.value() + size_of::<Coefficients<f32>>())
    }
}

//-------------//
// PanelKernel //
//-------------//

struct PanelKernel<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
{
    arch: W,
    a: APanel<'a, W::AElem, MR, PACK>,
    b: BTile<'a, W::BElem, PACK>,
    c: [f32; MR],
}

impl<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    PanelKernel<'a, W, MR, NR, PACK>
{
    /// # Safety
    ///
    /// A's logical dimension must equal B's.
    unsafe fn new(
        arch: W,
        a: APanel<'a, W::AElem, MR, PACK>,
        b: BTile<'a, W::BElem, PACK>,
        c: [f32; MR],
    ) -> Self {
        bounds::check_eq!(a.codes.k(), b.k);
        Self { arch, a, b, c }
    }
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize, const N: usize>
    unpacked::PanelVisitor<W::BElem, N> for &mut PanelKernel<'_, W, MR, NR, PACK>
{
    #[inline(always)]
    fn visit(&mut self, codes: unpacked::Panel<'_, W::BElem, N>, start: usize) {
        let this = &mut **self;
        // SAFETY: Both full and trailing panels come from this kernel's B tile.
        let b = unsafe { this.b.bind_panel(codes, start) };
        let mut kernel = MicroKernel {
            arch: this.arch,
            a: this.a,
            b,
            c: &mut this.c,
        };
        driver::MicroKernel::micro_kernel(&mut kernel);
    }
}

impl<W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
    Target<W, [f32; MR]> for PanelKernel<'_, W, MR, NR, PACK>
where
    Self: driver::PanelKernel,
{
    #[inline(always)]
    fn run(mut self, _: W) -> [f32; MR] {
        driver::PanelKernel::panel_kernel(&mut self);
        self.c
    }
}

macro_rules! panel_kernel {
    ($nr:literal, [$($tail:literal),*]) => {
        impl<W: ExtraWide<MR, PACK>, const MR: usize, const PACK: usize>
            driver::PanelKernel for PanelKernel<'_, W, MR, $nr, PACK>
        {
            #[inline(always)]
            fn panel_kernel(&mut self) {
                let b = self.b;
                let mut visitor = self;
                // SAFETY: Preparation constructs B with physical_k elements per row.
                let tail = unsafe {
                    b.codes.visit_panels::<$nr>(b.physical_k, &mut *visitor)
                };
                if let Some(tail) = tail {
                    $(
                        if let Some(panel) = tail.try_as_panel::<$tail>() {
                            unpacked::PanelVisitor::visit(&mut visitor, panel, tail.start());
                            return;
                        }
                    )*
                    unreachable!("all remainders below NR are handled");
                }
            }
        }
    };
}

panel_kernel!(2, [1]);
panel_kernel!(4, [1, 2, 3]);
panel_kernel!(8, [1, 2, 3, 4, 5, 6, 7]);

//-------------//
// MicroKernel //
//-------------//

struct MicroKernel<'a, W: ExtraWide<MR, PACK>, const MR: usize, const NR: usize, const PACK: usize>
{
    arch: W,
    a: APanel<'a, W::AElem, MR, PACK>,
    b: BPanel<'a, W::BElem, NR, PACK>,
    c: &'a mut [f32; MR],
}

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
        let physical_k = self.b.physical_k;
        let k = self.b.k.value().get();
        let mut acc = [self.arch.zero_acc(); NR];
        let ap = self.a.codes.as_ptr();
        let bp = self.b.codes.as_ptr();
        let astride = self.a.codes.pack_stride();
        let bstride = self.b.codes.stride(physical_k);

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
        let a_meta = Coefficients {
            a: &self.a.meta.a,
            b: &self.a.meta.b,
            n: &self.a.meta.n,
        };
        self.arch.max_into(acc, a_meta, self.b.meta, k, self.c);
    }
}
//----------//
// Backends //
//----------//

/// Backends accept A codes in 0..=255; B is limited by `MAX_B_BITS`.
pub(crate) trait ExtraWide<const MR: usize, const PACK: usize>: Architecture {
    /// Largest unsigned B width valid for every A code.
    const MAX_B_BITS: usize;

    type AElem: Copy + Default + From<u8>;
    type BElem: Copy + Default;
    type Wide: Copy;
    type Splat: Copy;
    type Acc: Copy;

    /// `code` fits in MAX_B_BITS unsigned bits.
    fn convert_b(code: u8) -> Self::BElem;

    /// # Safety
    ///
    /// `a` must contain at least MR * PACK initialized, readable elements.
    unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide;

    fn zero_acc(self) -> Self::Acc;

    /// Accumulate MR dot products of PACK elements without mixing rows.
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc;

    /// Compensate with logical K and update maxima, ignoring NaNs.
    fn max_into<const NR: usize>(
        self,
        acc: [Self::Acc; NR],
        query: Coefficients<&[f32; MR]>,
        docs: &[Coefficients<f32>; NR],
        k: usize,
        scores: &mut [f32; MR],
    );

    /// # Safety
    ///
    /// B must contain PACK readable elements, including any zero padding.
    unsafe fn splat(self, b: Slice<'_, Self::BElem>) -> Self::Splat;
}

#[inline(always)]
fn compensated_max_into<I, F, const MR: usize, const NR: usize>(
    arch: F::Arch,
    acc: impl IntoIterator<Item = I>,
    query: Coefficients<&[f32; MR]>,
    docs: &[Coefficients<f32>; NR],
    k: usize,
    scores: &mut [f32; MR],
) where
    I: SIMDCast<f32, Cast = F>,
    F: SIMDVector<Scalar = f32, ConstLanes = Const<MR>>
        + SIMDMinMax
        + Add<Output = F>
        + Mul<Output = F>,
    Const<MR>: SupportedLaneCount,
{
    let a = Coefficients {
        a: F::from_array(arch, *query.a),
        b: F::from_array(arch, *query.b),
        n: F::from_array(arch, *query.n),
    };
    let mut best = F::from_array(arch, *scores);
    for (acc, b) in acc.into_iter().zip(docs) {
        let dot = acc.simd_cast();
        let b = Coefficients {
            a: F::splat(arch, b.a),
            b: F::splat(arch, b.b),
            n: F::splat(arch, b.n),
        };
        let dim = F::splat(arch, k as f32);
        let ip = a.compensate_inner_product(b, dot, dim);
        best = best.max_simd_standard(ip);
    }
    *scores = best.to_array();
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

/// # Safety
///
/// B must contain two readable elements.
#[inline(always)]
unsafe fn pack_i16_pair(b: Slice<'_, i16>) -> u32 {
    // SAFETY: The caller provides two readable elements.
    let [lo, hi] = unsafe { load_pack::<_, 2>(b) };
    u32::from(lo as u16) | (u32::from(hi as u16) << 16)
}

impl ExtraWide<16, 4> for Scalar {
    const MAX_B_BITS: usize = 7;

    type AElem = u8;
    type BElem = i8;
    type Wide = u8x64<Scalar>;
    type Splat = i8x64<Scalar>;
    type Acc = i32x16<Scalar>;

    #[inline(always)]
    unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
        // SAFETY: The caller provides 64 initialized elements.
        Self::Wide::from_array(self, unsafe { load_pack::<_, 64>(a) })
    }

    #[inline(always)]
    fn zero_acc(self) -> Self::Acc {
        SIMDVector::default(self)
    }

    #[inline(always)]
    fn max_into<const NR: usize>(
        self,
        acc: [Self::Acc; NR],
        query: Coefficients<&[f32; 16]>,
        docs: &[Coefficients<f32>; NR],
        k: usize,
        scores: &mut [f32; 16],
    ) {
        compensated_max_into(self, acc, query, docs, k, scores);
    }

    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
    }

    fn convert_b(code: u8) -> i8 {
        code as i8
    }

    #[inline(always)]
    unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
        // SAFETY: The caller provides four readable elements.
        let values = unsafe { load_pack::<_, 4>(b) };
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

    #[inline(always)]
    unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
        // SAFETY: The caller provides 32 initialized elements.
        Self::Wide::from_array(self, unsafe { load_pack::<_, 32>(a) })
    }

    #[inline(always)]
    fn zero_acc(self) -> Self::Acc {
        SIMDVector::default(self)
    }

    #[inline(always)]
    fn max_into<const NR: usize>(
        self,
        acc: [Self::Acc; NR],
        query: Coefficients<&[f32; 16]>,
        docs: &[Coefficients<f32>; NR],
        k: usize,
        scores: &mut [f32; 16],
    ) {
        compensated_max_into(self, acc, query, docs, k, scores);
    }

    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
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
    const MAX_B_BITS: usize = 8;
    type AElem = i16;
    type BElem = i16;
    type Wide = i16x16<Scalar>;
    type Splat = Self::Wide;
    type Acc = i32x8<Scalar>;

    #[inline(always)]
    unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
        // SAFETY: The caller provides 16 initialized elements.
        Self::Wide::from_array(self, unsafe { load_pack::<_, 16>(a) })
    }

    #[inline(always)]
    fn zero_acc(self) -> Self::Acc {
        SIMDVector::default(self)
    }

    #[inline(always)]
    fn max_into<const NR: usize>(
        self,
        acc: [Self::Acc; NR],
        query: Coefficients<&[f32; 8]>,
        docs: &[Coefficients<f32>; NR],
        k: usize,
        scores: &mut [f32; 8],
    ) {
        compensated_max_into(self, acc, query, docs, k, scores);
    }

    #[inline(always)]
    fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
        acc.dot_simd(a, b)
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

    #[inline(always)]
    fn compensated_max_into_v3<const NR: usize>(
        arch: V3,
        acc: [i32x16<V3>; NR],
        query: Coefficients<&[f32; 16]>,
        docs: &[Coefficients<f32>; NR],
        k: usize,
        scores: &mut [f32; 16],
    ) {
        let a = query.a.as_chunks::<8>().0;
        let b = query.b.as_chunks::<8>().0;
        let n = query.n.as_chunks::<8>().0;
        let scores = scores.as_chunks_mut::<8>().0;
        // Finish one native-width row group before loading the next group's coefficients.
        compensated_max_into(
            arch,
            acc.iter().map(|v| v.split().lo),
            Coefficients {
                a: &a[0],
                b: &b[0],
                n: &n[0],
            },
            docs,
            k,
            &mut scores[0],
        );
        compensated_max_into(
            arch,
            acc.iter().map(|v| v.split().hi),
            Coefficients {
                a: &a[1],
                b: &b[1],
                n: &n[1],
            },
            docs,
            k,
            &mut scores[1],
        );
    }

    impl ExtraWide<8, 4> for V3 {
        const MAX_B_BITS: usize = 6;
        type AElem = u8;
        type BElem = i8;
        type Wide = u8x32<V3>;
        type Splat = i8x32<V3>;
        type Acc = i32x8<V3>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 32 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 32>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 8]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 8],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
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
        const MAX_B_BITS: usize = 8;
        type AElem = i16;
        type BElem = i16;
        type Wide = i16x16<V3>;
        type Splat = Self::Wide;
        type Acc = i32x8<V3>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 16 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 16>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 8]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 8],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
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
        const MAX_B_BITS: usize = 6;

        type AElem = u8;
        type BElem = i8;
        type Wide = u8x64<V3>;
        type Splat = i8x64<V3>;
        type Acc = i32x16<V3>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 64 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 64>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 16]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 16],
        ) {
            compensated_max_into_v3(self, acc, query, docs, k, scores);
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
            // SAFETY: The caller provides four readable elements.
            let values = unsafe { load_pack::<_, 4>(b) };
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

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 32 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 32>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 16]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 16],
        ) {
            compensated_max_into_v3(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
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
        const MAX_B_BITS: usize = 7;

        type AElem = u8;
        type BElem = i8;
        type Wide = u8x64<V4>;
        type Splat = i8x64<V4>;
        type Acc = i32x16<V4>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 64 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 64>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 16]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 16],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_b(code: u8) -> i8 {
            code as i8
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, i8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements.
            let values = unsafe { load_pack::<_, 4>(b) };
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

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 32 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 32>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 16]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 16],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
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
        const MAX_B_BITS: usize = 8;
        type AElem = u8;
        type BElem = u8;
        type Wide = u8x32<Neon>;
        type Splat = Self::Wide;
        type Acc = u32x8<Neon>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 32 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 32>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 8]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 8],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
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
        const MAX_B_BITS: usize = 8;

        type AElem = u8;
        type BElem = u8;
        type Wide = u8x64<Neon>;
        type Splat = Self::Wide;
        type Acc = u32x16<Neon>;

        #[inline(always)]
        unsafe fn load_a(self, a: Slice<'_, Self::AElem>) -> Self::Wide {
            // SAFETY: The caller provides 64 initialized elements.
            Self::Wide::from_array(self, unsafe { load_pack::<_, 64>(a) })
        }

        #[inline(always)]
        fn zero_acc(self) -> Self::Acc {
            SIMDVector::default(self)
        }

        #[inline(always)]
        fn max_into<const NR: usize>(
            self,
            acc: [Self::Acc; NR],
            query: Coefficients<&[f32; 16]>,
            docs: &[Coefficients<f32>; NR],
            k: usize,
            scores: &mut [f32; 16],
        ) {
            compensated_max_into(self, acc, query, docs, k, scores);
        }

        #[inline(always)]
        fn dot(self, acc: Self::Acc, a: Self::Wide, b: Self::Splat) -> Self::Acc {
            acc.dot_simd(a, b)
        }

        fn convert_b(code: u8) -> u8 {
            code
        }

        #[inline(always)]
        unsafe fn splat(self, b: Slice<'_, u8>) -> Self::Splat {
            // SAFETY: The caller provides four readable elements.
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
        minmax::{DataRef, MinMaxCompensation, MinMaxIP},
        multi_vector::{Defaulted, Mat},
        test_util::Check,
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
            if got == want {
                continue;
            }
            assert!(
                got.is_finite() && want.is_finite(),
                "{context}: row {i}: got {got}, expected {want}"
            );
            Check::absrel(1e-5, 1e-5)
                .check(got, want)
                .unwrap_or_else(|err| panic!("{context}: row {i}: {err}"));
        }
    }

    fn checked_driver<
        'a,
        W,
        const B_BITS: usize,
        const MR: usize,
        const NR: usize,
        const PACK: usize,
    >(
        arch: W,
        a: &'a PackedA<W::AElem, MR, PACK>,
        b: MatRef<'a, MinMaxMeta<B_BITS>>,
        c: &'a mut [f32],
    ) -> Driver<'a, W, MR, NR, PACK>
    where
        W: ExtraWide<MR, PACK> + LoadStore<f32, MR>,
        Unsigned: Representation<B_BITS>,
        for<'b> PanelKernel<'b, W, MR, NR, PACK>: driver::PanelKernel,
    {
        assert!(a.codes.nrows() > 0 && a.codes.ncols() > 0);
        assert_eq!(a.codes.nrows(), c.len());
        assert_eq!(a.codes.ncols(), b.repr().intrinsic_dim());
        // SAFETY: Nonempty A/K, matching dimensions and output length were checked above.
        unsafe { Driver::<W, MR, NR, PACK>::new(arch, a, b, c, Cache::detect()) }
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
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
    {
        let mut rng = StdRng::seed_from_u64(0x63de91a0ba5c8742);
        for shape in packed_x_unpacked_test_dims(MR, NR) {
            for k in [shape.k, shape.k.next_multiple_of(PACK)] {
                let a = random_minmax_matrix::<8>(shape.total_a_rows, k, &mut rng);
                let b = random_minmax_matrix::<B_BITS>(shape.total_b_cols, k, &mut rng);
                let packed = PackedA::<W::AElem, MR, PACK>::new::<8>(a.as_view()).unwrap();
                let expected = reference_max_inner_products::<8, B_BITS>(a.as_view(), b.as_view());
                let mut guarded = vec![654321.0; a.num_vectors() + 2];
                let mut driver = checked_driver::<_, B_BITS, MR, NR, PACK>(
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
        let packed = PackedA::<W::AElem, MR, PACK>::new::<8>(a.as_view()).unwrap();
        let mut c = vec![f32::INFINITY; a.num_vectors()];
        driver::Drive::drive(&mut checked_driver::<_, B_BITS, MR, NR, PACK>(
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

    #[test]
    fn decodes_documents_once_per_tile() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn counting_decode<const B_BITS: usize>(
            _: Scalar,
            source: rowmajor::Ref<'_, u8>,
            k: DimK,
            codes: rowmajor::Mut<'_, i16>,
            coefficients: &mut [Coefficients<f32>],
        ) where
            Unsigned: Representation<B_BITS>,
        {
            CALLS.fetch_add(1, Ordering::Relaxed);
            decode_tile::<B_BITS, _>(source, k, codes, coefficients, i16::from);
        }

        fn check<const B_BITS: usize>()
        where
            Unsigned: Representation<B_BITS>,
        {
            let mut rng = StdRng::seed_from_u64(0xf9380b5db7cb9e41);
            for rows in [1, 17] {
                for docs in [0, 1, 4, 5, 11] {
                    let a = random_minmax_matrix::<B_BITS>(rows, 5, &mut rng);
                    let b = random_minmax_matrix::<B_BITS>(docs, 5, &mut rng);
                    let packed = PackedA::<i16, 8, 2>::new(a.as_view()).unwrap();
                    let expected = reference_max_inner_products(a.as_view(), b.as_view());
                    let mut scores = vec![f32::NAN; rows];
                    let mut driver = checked_driver::<_, B_BITS, 8, 2, 2>(
                        Scalar::new(),
                        &packed,
                        b.as_view(),
                        &mut scores,
                    );
                    driver.b_rows_per_tile = NonZeroUsize::new(4).unwrap();
                    driver.decode_tile = counting_decode::<B_BITS>;
                    CALLS.store(0, Ordering::Relaxed);
                    for iteration in 1..=2 {
                        driver::Drive::drive(&mut driver);
                        assert_eq!(
                            CALLS.load(Ordering::Relaxed),
                            iteration * docs.div_ceil(4),
                            "B_BITS={B_BITS}, rows={rows}, docs={docs}",
                        );
                        assert_scores_close(
                            driver.c,
                            &expected,
                            format_args!("B_BITS={B_BITS}, rows={rows}, docs={docs}"),
                        );
                    }
                }
            }
        }

        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<6>();
        check::<7>();
        check::<8>();
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
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
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
        let packed = PackedA::<W::AElem, MR, PACK>::new::<8>(a.as_view()).unwrap();
        let mut c = vec![0.0; MR + 1];
        let mut driver =
            checked_driver::<_, B_BITS, MR, NR, PACK>(arch, &packed, b.as_view(), &mut c);
        // The first tile contains competing raw dots; its compensated winner must survive the tail.
        driver.b_rows_per_tile = NonZeroUsize::new(NR).unwrap();
        driver::Drive::drive(&mut driver);
        assert_eq!(c, vec![3.0 * winner_scale; MR + 1]);

        let a = constant_minmax_matrix::<8>(MR + 1, 3, 1, 1.0, -2.0);
        let b = constant_minmax_matrix::<B_BITS>((NR - 1).max(1), 3, 1, 1.0, 0.0);
        let packed = PackedA::<W::AElem, MR, PACK>::new::<8>(a.as_view()).unwrap();
        driver::Drive::drive(&mut checked_driver::<_, B_BITS, MR, NR, PACK>(
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
        let k = max_k(8, 8);
        let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let b = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
        let packed = PackedA::<i16, 8, 2>::new(a.as_view()).unwrap();
        let mut c = [0.0];
        driver::Drive::drive(&mut checked_driver::<_, 8, 8, 2, 2>(
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
        assert_driver_target_shapes::<Scalar, 4, 16, 2, 4>(Scalar::new());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
            assert_driver_contract::<_, 4, 16, 4, 4>(arch);
            assert_driver_target_shapes::<_, 4, 16, 4, 4>(arch);
        } else {
            eprintln!("AVX2 unavailable; backend not executed");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
            assert_driver_contract::<_, 4, 16, 8, 4>(arch);
            assert_driver_target_shapes::<_, 4, 16, 8, 4>(arch);
        } else {
            eprintln!("V4 unavailable; backend not executed");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_driver_obeys_maxsim_contract() {
        if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
            assert_driver_contract::<_, 4, 16, 4, 4>(arch);
            assert_driver_target_shapes::<_, 4, 16, 4, 4>(arch);
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
            for<'a> PanelKernel<'a, W, 8, NR, PACK>: driver::PanelKernel,
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

    fn assert_a_packing<T, const A_BITS: usize, const MR: usize, const PACK: usize>()
    where
        T: Copy + Default + From<u8> + PartialEq + std::fmt::Debug,
        Unsigned: Representation<A_BITS>,
    {
        for (m, k) in [(0, 0), (0, 5), (3, 0)] {
            let a = constant_minmax_matrix::<A_BITS>(m, k, 0, 1.0, 0.0);
            let packed = PackedA::<T, MR, PACK>::new(a.as_view()).unwrap();
            assert_eq!(packed.codes.nrows(), m);
            assert_eq!(packed.codes.ncols(), k);
            assert!(packed.as_view().is_none());
        }
        let mut rng = StdRng::seed_from_u64(0x0ae4b97f2c186d35);
        let a = random_minmax_matrix::<A_BITS>(MR + 1, 5, &mut rng);
        let packed = PackedA::<T, MR, PACK>::new(a.as_view()).unwrap();
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
                    T::from(a.get_row(row).unwrap().vector().get(col).unwrap() as u8)
                } else {
                    T::default()
                };
                assert_eq!(packed.codes.as_slice()[index], expected);
            }
            let meta = a.get_row(row).map(|v| v.meta()).unwrap_or_default();
            let panel = &packed.meta[row / MR];
            assert_eq!(panel.a[row % MR], meta.a);
            assert_eq!(panel.b[row % MR], meta.b);
            assert_eq!(panel.n[row % MR], meta.n);
        }
        let mut visited = 0;
        packed.as_view().unwrap().visit_panels(|panel, i| {
            assert_eq!(i, visited);
            assert!(std::ptr::eq(panel.meta, &packed.meta[i]));
            assert_eq!(
                panel.codes.as_ptr().as_ptr(),
                packed
                    .codes
                    .as_slice()
                    .as_ptr()
                    .wrapping_add(i * MR * padded_cols),
            );
            visited += 1;
        });
        assert_eq!(visited, packed.meta.len());
    }

    #[test]
    fn a_packing_preserves_codes_metadata_and_padding() {
        macro_rules! check {
            ($($bits:literal),+) => {$(
                assert_a_packing::<u8, $bits, 16, 4>();
                assert_a_packing::<i16, $bits, 16, 2>();
            )+};
        }
        check!(1, 2, 3, 4, 5, 6, 7, 8);
        assert_a_packing::<i16, 3, 8, 2>();
    }

    #[test]
    fn cache_footprints_match_prepared_layouts() {
        fn check<T: Copy + Default + From<u8>, const PACK: usize>(
            cases: [(usize, usize, usize); 2],
        ) {
            for (k, a_bytes, b_bytes) in cases {
                let dim = DimK::new(NonZeroUsize::new(k).unwrap());
                assert_eq!(BScratch::<T>::row_bytes::<PACK>(dim).value(), b_bytes);
                for rows in [1, 16, 17] {
                    let a = constant_minmax_matrix::<8>(rows, k, 0, 1.0, 0.0);
                    let packed = PackedA::<T, 16, PACK>::new(a.as_view()).unwrap();
                    assert_eq!(packed.panel_bytes().value(), a_bytes);
                }
            }
        }
        check::<u8, 4>([(256, 4288, 268), (257, 4352, 272)]);
        check::<i16, 2>([(256, 8384, 524), (257, 8448, 528)]);
    }

    #[test]
    fn compact_metadata_preserves_scores_across_tile_sizes() {
        let arch = Scalar::new();
        let mut rng = StdRng::seed_from_u64(0x9f23ea1640b187ac);
        let a = random_minmax_matrix::<8>(9, 3, &mut rng);
        let b = random_minmax_matrix::<8>(11, 3, &mut rng);
        let packed = PackedA::<i16, 8, 2>::new(a.as_view()).unwrap();
        // This budget gives different rounded tiles for compact and canonical metadata.
        let cache = Cache::new(
            NonZeroUsize::new(packed.panel_bytes().value() + 81).unwrap(),
            NonZeroUsize::new(4096).unwrap(),
        );
        let expected = reference_max_inner_products(a.as_view(), b.as_view());
        let mut scores = vec![f32::NAN; a.num_vectors()];
        // SAFETY: A is nonempty; both inputs have K=3 and scores covers every A row.
        let mut driver =
            unsafe { Driver::<_, 8, 2, 2>::new(arch, &packed, b.as_view(), &mut scores, cache) };
        assert_eq!(driver.b_rows_per_tile.get(), 6);
        driver::Drive::drive(&mut driver);
        assert_scores_close(driver.c, &expected, format_args!("compact metadata tiling"));

        let previous_row_bytes = Bytes::new(4 * size_of::<i16>() + size_of::<MinMaxCompensation>());
        driver.b_rows_per_tile =
            Params::new(cache, packed.panel_bytes(), previous_row_bytes, 2).b_cols_in_l1;
        assert_eq!(driver.b_rows_per_tile.get(), 4);
        driver::Drive::drive(&mut driver);
        assert_scores_close(
            driver.c,
            &expected,
            format_args!("previous metadata tiling"),
        );
    }

    #[test]
    fn b_preparation_preserves_metadata_and_zero_padding() {
        macro_rules! check {
            ($($bits:literal),+) => {$(
                assert_b_preparation::<i8, $bits, 4>(-1, |code| code as i8);
                assert_b_preparation::<i16, $bits, 2>(-1, i16::from);
                assert_b_preparation::<u8, $bits, 4>(u8::MAX, |code| code);
            )+};
        }
        check!(1, 2, 3, 4, 5, 6, 7);
        assert_b_preparation::<i16, 8, 2>(-1, i16::from);
        assert_b_preparation::<u8, 8, 4>(u8::MAX, |code| code);
        assert_b_preparation::<u16, 3, 4>(u16::MAX, |code| u16::from(code) + 1);
    }

    #[test]
    fn b_panels_bind_coefficients_at_full_and_tail_offsets() {
        let k = DimK::new(NonZeroUsize::new(3).unwrap());
        let mut rng = StdRng::seed_from_u64(0xda7bc35e83012e29);
        let source = random_minmax_matrix::<4>(5, k.value().get(), &mut rng);
        let mut scratch = BScratch::<u8>::default();
        let tile = scratch.prepare::<4>(
            source.as_view().as_canonical_view().unwrap(),
            k,
            |source, k, codes, coefficients| {
                decode_tile::<4, _>(source, k, codes, coefficients, |code| code);
            },
        );
        let mut actual = Vec::new();
        let visit = |codes: unpacked::Panel<'_, u8, 2>, start| {
            // SAFETY: The tile's traversal supplies this panel and its offset.
            let panel = unsafe { tile.bind_panel(codes, start) };
            actual.extend(panel.meta.iter().map(|m| (m.a, m.b, m.n)));
        };
        // SAFETY: Preparation records the view's padded dimension.
        let tail = unsafe { tile.codes.visit_panels::<2>(tile.physical_k, visit) }.unwrap();
        // SAFETY: The remainder comes from the same tile and contains one row.
        let panel = unsafe { tile.bind_panel(tail.try_as_panel::<1>().unwrap(), tail.start()) };
        actual.extend(panel.meta.iter().map(|m| (m.a, m.b, m.n)));
        let expected: Vec<_> = source
            .rows()
            .map(|row| {
                let m = row.meta();
                (m.a, m.b, m.n)
            })
            .collect();
        assert_eq!(actual, expected);
    }

    fn assert_b_preparation<T, const B_BITS: usize, const PACK: usize>(
        sentinel: T,
        convert: impl Fn(u8) -> T,
    ) where
        T: Copy + Default + PartialEq + std::fmt::Debug,
        Unsigned: Representation<B_BITS>,
    {
        let mut scratch = BScratch::<T>::default();
        for dim in [1, PACK - 1, PACK, PACK + 1, 15, 16, 17] {
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
            let source = b.as_canonical_view().unwrap();
            for (start, count) in [(0, 3), (2, 1), (1, 2)] {
                scratch.codes.fill(sentinel);
                scratch.meta.fill(Coefficients::default());
                let mut conversions = 0;
                let tile = scratch.prepare::<PACK>(
                    source.subview(start..start + count).unwrap(),
                    k,
                    |source, k, codes, coefficients| {
                        decode_tile::<B_BITS, _>(source, k, codes, coefficients, |code| {
                            conversions += 1;
                            convert(code)
                        });
                    },
                );
                assert_eq!(conversions, count * dim);
                let padded = padded_k::<PACK>(k);
                let row_len = padded.value().get();
                // SAFETY: Preparation constructs the view with the recorded padded dimension.
                let codes = unsafe { tile.codes.as_std_slice(tile.physical_k) };
                assert_eq!(codes.len(), count * row_len);
                assert_eq!(tile.meta.len(), count);
                for i in 0..count {
                    let row = b.get_row(start + i).unwrap();
                    let expected = row.meta();
                    let actual = tile.meta[i];
                    assert_eq!(
                        (actual.a, actual.b, actual.n),
                        (expected.a, expected.b, expected.n),
                    );
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
        for<'a> PanelKernel<'a, W, MR, NR, PACK>: driver::PanelKernel,
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
            let packed = PackedA::<W::AElem, MR, PACK>::new::<8>(a.as_view()).unwrap();
            let mut c = vec![f32::INFINITY; m];
            driver::Drive::drive(&mut checked_driver::<_, B_BITS, MR, NR, PACK>(
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
                a: [1.0; MR],
                b: [1.0; MR],
                n,
            };
            for (initial, score) in [
                (f32::NEG_INFINITY, f32::NAN),
                (f32::NEG_INFINITY, 5.0),
                (f32::NEG_INFINITY, -5.0),
                (f32::NEG_INFINITY, f32::INFINITY),
                (7.0, f32::NAN),
            ] {
                for column in [0, NR - 1] {
                    let mut b_meta = [Coefficients {
                        a: 1.0,
                        b: 0.0,
                        n: f32::NAN,
                    }; NR];
                    b_meta[column].n = score;
                    let mut c = [initial; MR];
                    let mut kernel = MicroKernel::<W, MR, NR, PACK> {
                        arch,
                        a: APanel {
                            // SAFETY: A is one full MR panel, padded to PACK for logical K=1.
                            codes: unsafe { packed::Panel::new(Slice::new(&a_codes), k) },
                            meta: &a_meta,
                        },
                        b: BPanel {
                            // SAFETY: B contains NR rows of PACK initialized codes.
                            codes: unsafe {
                                unpacked::Panel::new(Slice::new(&b_codes), padded_k::<PACK>(k))
                            },
                            meta: &b_meta,
                            k,
                            physical_k: padded_k::<PACK>(k),
                        },
                        c: &mut c,
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
    fn avx2_half_reduction_matches_whole_rows() {
        use diskann_wide::arch::x86_64::V3;

        fn check<const NR: usize>(arch: V3) {
            arch.run(|| {
                let acc: [i32x16<V3>; NR] = core::array::from_fn(|j| {
                    i32x16::<V3>::from_array(
                        arch,
                        core::array::from_fn(|i| (i * 37 + j * 11) as i32),
                    )
                });
                let a = core::array::from_fn(|i| (i + 1) as f32 * 0.125);
                let b = core::array::from_fn(|i| i as f32 * 0.25 - 2.0);
                let mut n = core::array::from_fn(|i| i as f32 * 0.75);
                n[9] = f32::NAN;
                let query = Coefficients {
                    a: &a,
                    b: &b,
                    n: &n,
                };
                let mut docs: [Coefficients<f32>; NR] = core::array::from_fn(|j| Coefficients {
                    a: (j + 1) as f32 * 0.25,
                    b: 1.0 - j as f32,
                    n: j as f32 * 0.5,
                });
                for special in [0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                    docs[NR - 1].n = special;
                    let initial = core::array::from_fn(|i| {
                        if i.is_multiple_of(2) {
                            f32::NEG_INFINITY
                        } else {
                            i as f32
                        }
                    });
                    let mut expected = initial;
                    compensated_max_into(arch, acc, query, &docs, 31, &mut expected);
                    let mut pair = initial;
                    let mut quad = initial;
                    <V3 as ExtraWide<16, 2>>::max_into(arch, acc, query, &docs, 31, &mut pair);
                    <V3 as ExtraWide<16, 4>>::max_into(arch, acc, query, &docs, 31, &mut quad);
                    assert_eq!(
                        pair.map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "NR={NR}"
                    );
                    assert_eq!(
                        quad.map(f32::to_bits),
                        expected.map(f32::to_bits),
                        "NR={NR}"
                    );
                }
            });
        }

        if let Some(arch) = V3::new_checked() {
            check::<1>(arch);
            check::<2>(arch);
            check::<4>(arch);
            check::<6>(arch);
            check::<8>(arch);
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
                        W::AElem::from(if (i / PACK).is_multiple_of(2) {
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
                    a: &[1.0; MR],
                    b: &[0.0; MR],
                    n: &[0.0; MR],
                };
                let mut actual = [f32::NEG_INFINITY; MR];
                arch.max_into(
                    [acc],
                    coefficients,
                    &[Coefficients {
                        a: 1.0,
                        ..Default::default()
                    }],
                    2 * PACK,
                    &mut actual,
                );
                assert_eq!(
                    actual,
                    core::array::from_fn(|i| {
                        let a = if i.is_multiple_of(2) { 255 } else { 1 };
                        (2 * PACK * a * usize::from(max_b)) as f32
                    }),
                );
            });
        }

        fn assert_dimension_limit<const A_BITS: usize, const B_BITS: usize>() {
            let max_a = (1u64 << A_BITS) - 1;
            let max_b = (1u64 << B_BITS) - 1;
            let limit = max_k(A_BITS, B_BITS) as u64;
            assert!(limit * max_a * max_b <= i32::MAX as u64);
            assert!((limit + 1) * max_a * max_b > i32::MAX as u64);
        }

        #[test]
        fn dimension_limits_cover_all_input_widths() {
            fn check_a<const A_BITS: usize>() {
                assert_dimension_limit::<A_BITS, 1>();
                assert_dimension_limit::<A_BITS, 2>();
                assert_dimension_limit::<A_BITS, 3>();
                assert_dimension_limit::<A_BITS, 4>();
                assert_dimension_limit::<A_BITS, 5>();
                assert_dimension_limit::<A_BITS, 6>();
                assert_dimension_limit::<A_BITS, 7>();
                assert_dimension_limit::<A_BITS, 8>();
            }
            check_a::<1>();
            check_a::<2>();
            check_a::<3>();
            check_a::<4>();
            check_a::<5>();
            check_a::<6>();
            check_a::<7>();
            check_a::<8>();
        }

        #[test]
        fn scalar_backends_cover_supported_code_ranges() {
            assert_backend_code_range::<_, 16, 2>(Scalar::new());
            assert_backend_code_range::<_, 16, 4>(Scalar::new());
            assert_backend_code_range::<_, 8, 2>(Scalar::new());
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx2_backends_cover_supported_code_ranges() {
            if let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() {
                assert_backend_code_range::<_, 16, 2>(arch);
                assert_backend_code_range::<_, 16, 4>(arch);
                assert_backend_code_range::<_, 8, 2>(arch);
                assert_backend_code_range::<_, 8, 4>(arch);
            } else {
                eprintln!("AVX2 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn avx512_backends_cover_supported_code_ranges() {
            if let Some(arch) = diskann_wide::arch::x86_64::V4::new_checked() {
                assert_backend_code_range::<_, 16, 2>(arch);
                assert_backend_code_range::<_, 16, 4>(arch);
            } else {
                eprintln!("V4 unavailable; all-width regression not executed");
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[test]
        fn neon_backends_cover_supported_code_ranges() {
            if let Some(arch) = diskann_wide::arch::aarch64::Neon::new_checked() {
                assert_backend_code_range::<_, 16, 4>(arch);
                assert_backend_code_range::<_, 8, 4>(arch);
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
                let k = max_k(8, 8) + 1;
                let max_a = (1 << A_BITS) - 1;
                let max_b = (1 << B_BITS) - 1;
                let a = constant_minmax_matrix::<A_BITS>(1, k, max_a, 1.0, 0.0);
                let b = constant_minmax_matrix::<B_BITS>(1, k, max_b, 1.0, 0.0);
                let packed = PackedA::<i16, 16, 2>::new(a.as_view()).unwrap();
                let mut c = [0.0];
                driver::Drive::drive(&mut checked_driver::<_, B_BITS, 16, 2, 2>(
                    Scalar::new(),
                    &packed,
                    b.as_view(),
                    &mut c,
                ));
                assert_eq!(c[0], (k as i64 * max_a * max_b) as f32);
            }
            check::<1, 1>();
            check::<8, 1>();
        }
    }

    mod minmax8 {
        use super::*;

        fn check_backend<W, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
            for<'a> PanelKernel<'a, W, 16, NR, PACK>: driver::PanelKernel,
        {
            assert_driver_contract::<W, 8, 16, NR, PACK>(arch);
            assert_driver_target_shapes::<W, 8, 16, NR, PACK>(arch);
            assert_unsigned_byte_products::<W, NR, PACK>(arch);
        }

        fn assert_unsigned_byte_products<W, const NR: usize, const PACK: usize>(arch: W)
        where
            W: ExtraWide<16, PACK> + LoadStore<f32, 16>,
            for<'a> PanelKernel<'a, W, 16, NR, PACK>: driver::PanelKernel,
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
                let a = PackedA::<W::AElem, 16, PACK>::new(a.as_view()).unwrap();
                let mut c = vec![f32::NAN; 17];
                for code in boundary_codes {
                    let b = constant_minmax_matrix::<8>(1, k, code, 1.0, 0.0);
                    driver::Drive::drive(&mut checked_driver::<W, 8, 16, NR, PACK>(
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
            let k = max_k(8, 8) + 1;
            let a = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let b = constant_minmax_matrix::<8>(1, k, 255, 1.0, 0.0);
            let a = PackedA::<i16, 16, 2>::new(a.as_view()).unwrap();
            let mut c = [0.0];
            let _ = checked_driver::<_, 8, 16, 2, 2>(Scalar::new(), &a, b.as_view(), &mut c);
        }
    }
}
