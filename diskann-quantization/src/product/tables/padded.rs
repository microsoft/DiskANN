/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::{DenseData, MatrixBase, MatrixView};
use diskann_wide::{
    SIMDFloat, SIMDSumTree, SIMDVector,
    arch::{Architecture, Scalar, Target, Dispatched3, FTarget3, dispatch_no_features},
    lifetime::{self, Ref},
};

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;

use crate::views::{ChunkOffsetsBase, ChunkOffsetsView};

/// A PQ table that stores pivots grouped by chunk in the following dense, row-major form:
/// ```text
///            | -- pivot 0 --    | -- pivot 1 --    | .... | -- pivot K-1 --    |
///            +------------------+------------------+------+--------------------+
///  chunk 0   | c000 c001 ... 00 | c010 c011 ... 00 | .... | c0K0 c0K1 ...  00 |
///  chunk 1   | c100 c101 ... 00 | c110 c111 ... 00 | .... | c1K0 c1K1 ...  00 |
///    ...     |       ...        |       ...        | .... |       ...         |
///  chunk N-1 | cN00 cN01 ... 00 | cN10 cN11 ... 00 | .... | cNK0 cNK1 ...  00 |
/// ```
/// where `cCPD` is dimension `D` of pivot `P` in chunk `C`, and trailing `00`s denote
/// zero-padding to the SIMD-aligned pivot width.
///
/// The member `offsets` describes the number of *unpadded* dimensions of each chunk.
///
/// Importantly, though, the storage for each pivot is rounded up to a multiple of the
/// runtime system's preferred SIMD width and all pivots are padded to the same length.
/// This makes distance computations between pivots very fast for computing distances
/// between two product-quantized vectors.
#[derive(Debug, Clone)]
pub struct PaddedTableBase<T = Box<[f32]>, U = Box<[usize]>>
where
    T: DenseData<Elem = f32>,
    U: DenseData<Elem = usize>,
{
    pivots: MatrixBase<T>,
    offsets: ChunkOffsetsBase<U>,
}

type PaddedTableView<'a> = PaddedTableBase<&'a [f32], &'a [usize]>;

impl<T, U> PaddedTableBase<T, U>
where
    T: DenseData<Elem = f32>,
    U: DenseData<Elem = usize>,
{
    pub fn distance(&self, metric: Metric) -> Distance<'_> {
        let distance = match metric {
            Metric::SquaredL2 => dispatch_no_features(SquaredL2),
            Metric::InnerProduct => dispatch_no_features(InnerProduct),
            Metric::Cosine => dispatch_no_features(Cosine),
        };

        Distance {
            table: self.view(),
            distance,
        }
    }

    pub fn view(&self) -> PaddedTableView<'_> {
        PaddedTableBase {
            pivots: self.pivots.as_view(),
            offsets: self.offsets.as_view(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

#[derive(Debug)]
struct View;

impl lifetime::AddLifetime for View {
    type Of<'a> = PaddedTableBase<&'a [f32], &'a [usize]>;
}

type Dispatched = Dispatched3<f32, View, Ref<[u8]>, Ref<[u8]>>;

#[derive(Debug)]
pub struct Distance<'a> {
    table: PaddedTableBase<&'a [f32], &'a [usize]>,
    distance: Dispatched,
}

//-----------------------//
// Architecture Specific //
//-----------------------//

trait Preferred: Architecture
{
    type f32s: SIMDVector<Scalar = f32, Arch = Self>;
}

impl Preferred for Scalar {
    type f32s = Self::f32x4;
}

#[cfg(target_arch = "x86_64")]
impl Preferred for V3 {
    type f32s = Self::f32x8;
}

#[cfg(target_arch = "x86_64")]
impl Preferred for V4 {
    type f32s = Self::f32x16;
}

#[cfg(target_arch = "aarch64")]
impl Preferred for Neon {
    type f32s = Self::f32x4;
}

/// Detect the preferred SIMD width for `f32` vectors on the current architecture.
#[derive(Debug, Clone, Copy)]
struct DetectSIMDWidth;

impl<A> Target<A, usize> for DetectSIMDWidth
where
    A: Preferred,
{
    #[inline(always)]
    fn run(self, _: A) -> usize {
        A::f32s::LANES
    }
}

trait Op<V>
where
    V: SIMDVector<Scalar = f32>,
{
    type Accum;

    fn init(arch: V::Arch) -> Self::Accum;

    fn accum(acc: Self::Accum, x: V, y: V) -> Self::Accum;

    fn reduce(a: Self::Accum, b: Self::Accum, c: Self::Accum, d: Self::Accum) -> f32;
}

#[derive(Debug)]
struct SquaredL2;

impl<V> Op<V> for SquaredL2
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
{
    type Accum = V;

    fn init(arch: V::Arch) -> Self::Accum {
        V::default(arch)
    }

    fn accum(acc: Self::Accum, x: V, y: V) -> Self::Accum {
        let d = x - y;
        d.mul_add_simd(d, acc)
    }

    fn reduce(a: Self::Accum, b: Self::Accum, c: Self::Accum, d: Self::Accum) -> f32 {
        ((a + b) + (c + d)).sum_tree()
    }
}

#[derive(Debug)]
struct InnerProduct;

impl<V> Op<V> for InnerProduct
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
{
    type Accum = V;

    fn init(arch: V::Arch) -> Self::Accum {
        V::default(arch)
    }

    fn accum(acc: Self::Accum, x: V, y: V) -> Self::Accum {
        x.mul_add_simd(y, acc)
    }

    fn reduce(a: Self::Accum, b: Self::Accum, c: Self::Accum, d: Self::Accum) -> f32 {
        ((a + b) + (c + d)).sum_tree()
    }
}

#[derive(Debug)]
struct Cosine;

#[derive(Debug)]
struct CosineAccumulator<V> {
    xy: V,
    xnorm: V,
    ynorm: V,
}

impl<V> Op<V> for Cosine
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
{
    type Accum = CosineAccumulator<V>;

    fn init(arch: V::Arch) -> Self::Accum {
        CosineAccumulator {
            xy: V::default(arch),
            xnorm: V::default(arch),
            ynorm: V::default(arch),
        }
    }

    fn accum(acc: Self::Accum, x: V, y: V) -> Self::Accum {
        CosineAccumulator {
            xy: x.mul_add_simd(y, acc.xy),
            xnorm: x.mul_add_simd(x, acc.xnorm),
            ynorm: y.mul_add_simd(y, acc.ynorm),
        }
    }

    fn reduce(a: Self::Accum, b: Self::Accum, c: Self::Accum, d: Self::Accum) -> f32 {
        let xy = ((a.xy + b.xy) + (c.xy + d.xy)).sum_tree();
        let xnorm = ((a.xnorm + b.xnorm) + (c.xnorm + d.xnorm)).sum_tree();
        let ynorm = ((a.ynorm + b.ynorm) + (c.ynorm + d.ynorm)).sum_tree();

        if xnorm < f32::MIN_POSITIVE || ynorm < f32::MIN_POSITIVE {
            0.0
        } else {
            let v = xy / (xnorm.sqrt() * ynorm.sqrt());
            (-1.0f32).max(1.0f32.min(v))
        }
    }
}

macro_rules! target {
    ($op:ident) => {
        impl<A> FTarget3<A, f32, PaddedTableView<'_>, &[u8], &[u8]> for $op
        where
            A: Preferred,
            Self: Op<A::f32s>,
        {
            #[inline(always)]
            fn run(arch: A, table: PaddedTableView<'_>, a: &[u8], b: &[u8]) -> f32 {
                invoke::<A::f32s, Self>(arch, table, a, b)
            }
        }

        impl<A> Target<A, Dispatched> for $op
        where
            A: Preferred,
            Self: Op<A::f32s>,
        {
            fn run(self, arch: A) -> Dispatched {
                arch.dispatch3::<Self, f32, View, Ref<[u8]>, Ref<[u8]>>()
            }
        }
    }
}

target!(SquaredL2);
target!(InnerProduct);
target!(Cosine);

#[inline(always)]
fn invoke<V, O>(
    arch: V::Arch,
    table: PaddedTableView<'_>,
    a: &[u8],
    b: &[u8],
) -> f32
where
    V: SIMDVector<Scalar = f32>,
    O: Op<V>,
{
    // TODO: Safety Checks
    unsafe { kernel::<V, O>(arch, table.pivots, 0, a, b) }
}

#[inline(always)]
unsafe fn kernel<V, O>(
    arch: V::Arch,
    pivots: MatrixView<'_, f32>,
    pivots_per_chunk: usize,
    a: &[u8],
    b: &[u8],
) -> f32
where
    V: SIMDVector<Scalar = f32>,
    O: Op<V>,
{
    debug_assert_eq!(a.len(), b.len());

    // The number of SIMD steps to process for each pivot.
    let steps = pivots.ncols() / V::LANES;

    let pivot_stride = pivots.ncols();
    let chunk_stride = pivots_per_chunk * pivot_stride;

    let len = a.len();
    let mut a0 = O::init(arch);
    let mut a1 = O::init(arch);
    let mut a2 = O::init(arch);
    let mut a3 = O::init(arch);

    let mut i = 0;
    let mut p = pivots.as_ptr();

    let load = |ptr: *const f32, indices: &[u8], chunk: usize, lane: usize| -> V {
        let ptr = unsafe {
            ptr.add(pivot_stride * (*indices.get_unchecked(chunk) as usize) + V::LANES * lane)
        };
        unsafe { V::load_simd(arch, ptr) }
    };

    while i + 4 <= len {
        // Pointers to the start of each chunk.
        let c0 = p;
        let c1 = p.add(chunk_stride);
        let c2 = p.add(2 * chunk_stride);
        let c3 = p.add(3 * chunk_stride);

        for j in 0..steps {
            // Unroll 0
            let va = load(c0, a, i, j);
            let vb = load(c0, b, i, j);
            a0 = O::accum(a0, va, vb);

            // Unroll 1
            let va = load(c1, a, i + 1, j);
            let vb = load(c1, b, i + 1, j);
            a1 = O::accum(a1, va, vb);

            // Unroll 2
            let va = load(c2, a, i + 2, j);
            let vb = load(c2, b, i + 2, j);
            a2 = O::accum(a2, va, vb);

            // Unroll 3
            let va = load(c3, a, i + 3, j);
            let vb = load(c3, b, i + 3, j);
            a3 = O::accum(a3, va, vb);
        }

        i += 4;
        p = unsafe { p.add(4 * chunk_stride) };
    }

    while i < len {
        for j in 0..steps {
            let va = load(p, a, i, j);
            let vb = load(p, b, i, j);
            a0 = O::accum(a0, va, vb);
        }

        i += 1;
        p = unsafe { p.add(chunk_stride) };
    }

    O::reduce(a0, a1, a2, a3)
}
