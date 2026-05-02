/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    strided,
    views::{DenseData, Matrix, MatrixBase, MatrixView},
};
use diskann_vector::{DistanceFunction, distance::Metric as VectorMetric};
use diskann_wide::{
    SIMDFloat, SIMDSumTree, SIMDVector,
    arch::{Architecture, Dispatched3, FTarget3, Scalar, Target, dispatch_no_features},
    lifetime::{self, Ref},
};
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;

use crate::views::{ChunkOffsets, ChunkOffsetsBase, ChunkOffsetsView};

/// A PQ table that stores pivots grouped by chunk in the following dense, row-major form:
/// ```text
///            | -- pivot 0 --    | -- pivot 1 --    | .... | -- pivot K-1 --    |
///            +------------------+------------------+------+--------------------+
///  chunk 0   | c000 c001 ... 0X | c010 c011 ... 0X | .... | c0K0 c0K1 ...  0X |
///  chunk 1   | c100 c101 ... 0X | c110 c111 ... 0X | .... | c1K0 c1K1 ...  0X |
///    ...     |       ...        |       ...        | .... |       ...         |
///  chunk N-1 | cN00 cN01 ... 0X | cN10 cN11 ... 0X | .... | cNK0 cNK1 ...  0X |
/// ```
/// where `cCPD` is dimension `D` of pivot `P` in chunk `C`, and trailing `0X`s denote
/// potential zero-padding to the SIMD-aligned pivot width.
///
/// The member `offsets` describes the number of *unpadded* dimensions of each chunk.
///
/// Importantly, though, the storage for each pivot is rounded up to a multiple of the
/// runtime system's preferred SIMD width and all pivots are padded to the same length.
/// This makes distance computations between pivots very fast for computing distances
/// between two product-quantized vectors.
#[derive(Debug, Clone)]
pub struct PaddedTable {
    pivots: Matrix<f32>,
    offsets: ChunkOffsets,
    pivots_per_chunk: usize,
}

impl PaddedTable {
    pub fn from_parts(
        pivots: MatrixView<'_, f32>,
        offsets: ChunkOffsets,
    ) -> Result<Self, PaddedTableError> {
        let pivot_dim = pivots.ncols();
        let offsets_dim = offsets.dim();
        if pivot_dim != offsets_dim {
            return Err(PaddedTableError::DimMismatch {
                pivot_dim,
                offsets_dim,
            });
        }

        let pivots_per_chunk = pivots.nrows();

        // Compute the padded dimension of the pivots.
        //
        // There exists a corner case where `pivots` barely fits within the `isize` limit
        // of an allocation and padding will put us beyond that threshold, but that is
        // exceedingly unlikely for typical data.
        let max_chunk_dim = offsets.max_chunk_dim().get();
        let simd_width = dispatch_no_features(DetectSIMDWidth);
        let padded_dim = max_chunk_dim.next_multiple_of(simd_width);

        let rows = pivots_per_chunk * offsets.len();
        let mut padded = Matrix::new(0.0, rows, padded_dim);
        let mut row = 0;

        // Since we padded, we use a custom "copy_from_slice" that allows `dst` to shrink.
        fn copy_from_slice_subset(dst: &mut [f32], src: &[f32]) {
            dst[..src.len()].copy_from_slice(src)
        }

        // Copy the pivots.
        (0..offsets.len()).for_each(|i| {
            let range = offsets.at(i);

            let view = strided::StridedView::try_shrink_from(
                &(pivots.as_slice()[range.start..]),
                pivots.nrows(),
                range.len(),
                offsets.dim(),
            )
            .expect("the check on `pivot_dim` and `offsets_dim` should cause this to never error");

            view.row_iter().for_each(|src| {
                copy_from_slice_subset(padded.row_mut(row), src);
                row += 1;
            });
        });

        Ok(Self {
            pivots: padded,
            offsets,
            pivots_per_chunk,
        })
    }

    pub fn distance(&self, metric: Metric) -> Distance<'_> {
        Distance {
            table: self,
            distance: dispatch_no_features(metric),
        }
    }
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PaddedTableError {
    #[error("pivots have {pivot_dim} dimensions while the offsets expect {offsets_dim}")]
    DimMismatch {
        pivot_dim: usize,
        offsets_dim: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl From<VectorMetric> for Metric {
    fn from(metric: VectorMetric) -> Self {
        match metric {
            VectorMetric::L2 => Self::SquaredL2,
            VectorMetric::InnerProduct => Self::InnerProduct,
            VectorMetric::Cosine => Self::Cosine,
            VectorMetric::CosineNormalized => Self::Cosine,
        }
    }
}

type Dispatched = Dispatched3<f32, Ref<PaddedTable>, Ref<[u8]>, Ref<[u8]>>;

#[derive(Debug, Clone)]
pub struct Distance<'a> {
    table: &'a PaddedTable,
    distance: Dispatched,
}

impl DistanceFunction<&[u8], &[u8], f32> for Distance<'_> {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        self.distance.call(self.table, a, b)
    }
}

//-----------------------//
// Architecture Specific //
//-----------------------//

trait Preferred: Architecture {
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
        -((a + b) + (c + d)).sum_tree()
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
            1.0 - (-1.0f32).max(1.0f32.min(v))
        }
    }
}

impl<A> Target<A, Dispatched> for Metric
where
    A: Preferred,
    SquaredL2: Op<A::f32s>,
    InnerProduct: Op<A::f32s>,
    Cosine: Op<A::f32s>,
{
    #[inline(always)]
    fn run(self, arch: A) -> Dispatched {
        match self {
            Self::SquaredL2 => {
                arch.dispatch3::<SquaredL2, f32, Ref<PaddedTable>, Ref<[u8]>, Ref<[u8]>>()
            }
            Self::InnerProduct => {
                arch.dispatch3::<InnerProduct, f32, Ref<PaddedTable>, Ref<[u8]>, Ref<[u8]>>()
            }
            Self::Cosine => arch.dispatch3::<Cosine, f32, Ref<PaddedTable>, Ref<[u8]>, Ref<[u8]>>(),
        }
    }
}

macro_rules! target {
    ($op:ident) => {
        impl<A> FTarget3<A, f32, &PaddedTable, &[u8], &[u8]> for $op
        where
            A: Preferred,
            Self: Op<A::f32s>,
        {
            #[inline(always)]
            fn run(arch: A, table: &PaddedTable, a: &[u8], b: &[u8]) -> f32 {
                invoke::<A::f32s, Self>(arch, table, a, b)
            }
        }
    };
}

target!(SquaredL2);
target!(InnerProduct);
target!(Cosine);

#[inline(always)]
fn invoke<V, O>(arch: V::Arch, table: &PaddedTable, a: &[u8], b: &[u8]) -> f32
where
    V: SIMDVector<Scalar = f32>,
    O: Op<V>,
{
    // TODO: Safety Checks
    unsafe { kernel::<V, O>(arch, table.pivots.as_view(), table.pivots_per_chunk, a, b) }
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
