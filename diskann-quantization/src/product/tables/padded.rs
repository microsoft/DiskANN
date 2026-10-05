/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A PQ table optimized for computing distances between quantized vectors.
//!
//! During similarity search index construction, it is not uncommon to compute distances
//! among quantized elements within a table. The [`PaddedTable`] is designed to facilitate
//! such distances.
//!
//! ```
//! use diskann_quantization::{views::ChunkOffsets, product::tables};
//! use diskann_utils::views::rowmajor::{Owned, Matrix, MatrixMut};
//!
//! // We're creating the following pivot table.
//! //
//! // | chunk 0 | chunk 1 |
//! // |   0  0  |   1  1  | pivot 0
//! // |   1  1  |   2  2  | pivot 1
//! // |   2  2  |   3  3  | pivot 2
//!
//! let mut pivots = Owned::from_element(3, 4, 0.0f32);
//! pivots.row_mut(0).copy_from_slice(&[0.0, 0.0, 1.0, 1.0]);
//! pivots.row_mut(1).copy_from_slice(&[1.0, 1.0, 2.0, 2.0]);
//! pivots.row_mut(2).copy_from_slice(&[2.0, 2.0, 3.0, 3.0]);
//!
//! let offsets = ChunkOffsets::new(Box::new([0, 2, 4])).unwrap();
//!
//! let basic = tables::BasicTable::new(pivots, offsets).unwrap();
//! let padded = tables::PaddedTable::from_basic(basic.as_view());
//!
//! // Distances are provided through a v-table.
//! let vtable = padded.vtable(tables::padded::Metric::SquaredL2);
//!
//! // Compute the distance between [1, 1, 1, 1] and the chunk defined by [2, 1], which
//! // should translate to the compressed vector [2, 2, 2, 2].
//! let distance = vtable.distance(&padded, &[1.0, 1.0, 1.0, 1.0], &[2, 1]).unwrap();
//! assert_eq!(distance, 4.0);
//!
//! // Compute the distance between the two compressed vectors encoded by [0, 2] and [2, 0].
//! let distance = vtable.self_distance(&padded, &[0, 2], &[2, 0]).unwrap();
//! assert_eq!(distance, 16.0);
//! ```

use std::{marker::PhantomData, num::NonZeroUsize};

use diskann_utils::{
    strided,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use diskann_vector::distance::Metric as VectorMetric;
use diskann_wide::{
    SIMDFloat, SIMDSumTree, SIMDVector,
    arch::{Architecture, Dispatched3, FTarget3, Scalar, Target, dispatch_no_features},
    lifetime::Ref,
};
use thiserror::Error;

#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;

use crate::{product::tables::BasicTableView, views::ChunkOffsets};

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
/// This makes distance computations between pivots very fast when computing distances
/// between two product-quantized vectors.
#[derive(Debug, Clone)]
pub struct PaddedTable {
    /// Invariants:
    /// * `pivots.ncols()` is at least as large as the largest chunk in [`Self::offsets`]
    ///   and is always a multiple of `simd_width`.
    /// * `pivots.nrows() == offsets.len() * pivots_per_chunk`.
    pivots: rowmajor::Owned<f32>,
    offsets: ChunkOffsets,
    pivots_per_chunk: usize,
    simd_width: SIMDWidth,
    arch: RuntimeArch,
}

impl PaddedTable {
    /// Construct a [`PaddedTable`] with the same contents as `basic`.
    pub fn from_basic(basic: BasicTableView<'_>) -> Self {
        let arch = RuntimeArch::new();
        Self::from_basic_with(basic, arch)
    }

    fn from_basic_with(basic: BasicTableView<'_>, arch: RuntimeArch) -> Self {
        let pivots = basic.view_pivots();
        let offsets = basic.view_offsets();

        let pivots_per_chunk = pivots.nrows();

        // Compute the padded dimension of the pivots.
        //
        // There exists a corner case where `pivots` barely fits within the `isize` limit
        // of an allocation and padding will put us beyond that threshold, but that is
        // exceedingly unlikely for typical data.
        let max_chunk_dim = offsets.max_chunk_dim();
        let simd_width = arch.select_simd_width(max_chunk_dim);
        let padded_dim = max_chunk_dim
            .get()
            .next_multiple_of(simd_width.as_nonzero().get());

        let rows = pivots_per_chunk * offsets.len();
        let mut padded = rowmajor::Owned::from_element(rows, padded_dim, 0.0);
        let mut row = 0;

        // Since we padded, we use a custom "copy_from_slice" that allows `dst` to shrink.
        fn copy_from_slice_subset(dst: &mut [f32], src: &[f32]) {
            dst[..src.len()].copy_from_slice(src)
        }

        // Copy the pivots.
        (0..offsets.len()).for_each(|i| {
            let range = offsets.at(i);

            #[expect(
                clippy::expect_used,
                reason = "the layout should be pre-validated by `BasicTable`"
            )]
            let view = strided::Strided::try_from_data(
                &(pivots.as_slice()[range.start..]),
                pivots.nrows(),
                range.len(),
                offsets.dim(),
            )
            .expect("the check on `pivot_dim` and `offsets_dim` should cause this to never error");

            view.rows().for_each(|src| {
                copy_from_slice_subset(padded.row_mut(row), src);
                row += 1;
            });
        });

        Self {
            pivots: padded,
            offsets: offsets.to_owned(),
            pivots_per_chunk,
            simd_width,
            arch,
        }
    }

    /// Return the distance [`VTable`] for the requested [`Metric`].
    ///
    /// Note that [`VTable`]s are generally specific to the [`PaddedTable`] that generated
    /// them and cannot be reliably shared with different tables.
    ///
    /// While this is not a safety issue, incorrect [`VTable`]s may yield errors due to
    /// mismatched SIMD widths.
    pub fn vtable(&self, metric: Metric) -> VTable {
        self.arch.dispatch(self.simd_width, metric)
    }

    /// Return the number of PQ centers per chunk.
    pub fn ncenters(&self) -> usize {
        self.pivots_per_chunk
    }

    /// Return the number of PQ chunks.
    pub fn nchunks(&self) -> usize {
        self.offsets.len()
    }

    /// Return the full-precision dimension expected by this table.
    pub fn dim(&self) -> usize {
        self.offsets.dim()
    }
}

/// Distance metrics used by [`PaddedTable`].
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

type Distance = Dispatched3<Result<f32, DistanceError>, Ref<PaddedTable>, Ref<[f32]>, Ref<[u8]>>;
type SelfDistance =
    Dispatched3<Result<f32, SelfDistanceError>, Ref<PaddedTable>, Ref<[u8]>, Ref<[u8]>>;

/// A distance [`VTable`] for a [`PaddedTable`].
///
/// See: [`PaddedTable::vtable`].
#[derive(Debug, Clone, Copy)]
pub struct VTable {
    distance: Distance,
    self_distance: SelfDistance,
}

impl VTable {
    fn new<O>(arch: O::Arch) -> Self
    where
        O: Op,
    {
        Self {
            distance: arch.dispatch3::<
                OpTarget<O>,
                Result<f32, DistanceError>,
                Ref<PaddedTable>,
                Ref<[f32]>,
                Ref<[u8]>,
            >(),
            self_distance: arch.dispatch3::<
                OpTarget<O>,
                Result<f32, SelfDistanceError>,
                Ref<PaddedTable>,
                Ref<[u8]>,
                Ref<[u8]>,
            >(),
        }
    }

    /// Compute the distance between `vector` and the PQ vector encoded by `codes`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * `vector.len() != padded.dim()`
    /// * `codes.len() != padded.nchunks()`
    /// * Any element in `codes` is equal to or greater than `padded.ncenters()`.
    ///
    /// In addition, an error may be returned if `self` was created for a different
    /// [`PaddedTable`]. Mixing [`VTable`]s in this way is not a safety issue, but is also
    /// not guaranteed to work.
    #[inline]
    pub fn distance(
        &self,
        padded: &PaddedTable,
        vector: &[f32],
        codes: &[u8],
    ) -> Result<f32, DistanceError> {
        (self.distance).call(padded, vector, codes)
    }

    /// Compute the distance between two compressed vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * `a.len() != padded.nchunks()`
    /// * `b.len() != padded.nchunks()`
    /// * Any element in `a` or `b` is equal to or greater than `padded.ncenters()`.
    ///
    /// In addition, an error may be returned if `self` was created for a different
    /// [`PaddedTable`]. Mixing [`VTable`]s in this way is not a safety issue, but is also
    /// not guaranteed to work.
    #[inline]
    pub fn self_distance(
        &self,
        padded: &PaddedTable,
        a: &[u8],
        b: &[u8],
    ) -> Result<f32, SelfDistanceError> {
        (self.self_distance).call(padded, a, b)
    }
}

//-----------------//
// Width Selection //
//-----------------//

const FOUR: NonZeroUsize = NonZeroUsize::new(4).unwrap();

#[cfg(target_arch = "x86_64")]
const EIGHT: NonZeroUsize = NonZeroUsize::new(8).unwrap();

#[derive(Debug, Clone, Copy)]
enum SIMDWidth {
    Four,
    #[cfg(target_arch = "x86_64")]
    Eight,
}

impl SIMDWidth {
    fn as_nonzero(self) -> NonZeroUsize {
        match self {
            Self::Four => FOUR,
            #[cfg(target_arch = "x86_64")]
            Self::Eight => EIGHT,
        }
    }
}

/// The runtime architecture.
#[derive(Debug, Clone, Copy)]
enum RuntimeArch {
    Scalar(Scalar),
    #[cfg(target_arch = "x86_64")]
    V3(V3),
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
}

impl RuntimeArch {
    fn new() -> Self {
        dispatch_no_features(GetArch)
    }

    #[cfg_attr(
        not(target_arch = "x86_64"),
        expect(
            unused_variables,
            reason = "the same result is returned regardless of the actual chunk size"
        )
    )]
    fn select_simd_width(&self, max_chunk_size: NonZeroUsize) -> SIMDWidth {
        match self {
            Self::Scalar(_) => SIMDWidth::Four,
            // Use a smaller width if available.
            #[cfg(target_arch = "x86_64")]
            Self::V3(_) => match max_chunk_size.get() {
                0..=4 => SIMDWidth::Four,
                _ => SIMDWidth::Eight,
            },
            #[cfg(target_arch = "aarch64")]
            Self::Neon(_) => SIMDWidth::Four,
        }
    }

    fn dispatch(&self, simd_width: SIMDWidth, metric: Metric) -> VTable {
        diskann_wide::alias!(f32x4<A> = f32x4);
        diskann_wide::alias!(f32x8<A> = f32x8);

        match (*self, simd_width, metric) {
            (Self::Scalar(a), _, Metric::SquaredL2) => VTable::new::<SquaredL2<f32x4<Scalar>>>(a),
            (Self::Scalar(a), _, Metric::InnerProduct) => {
                VTable::new::<InnerProduct<f32x4<Scalar>>>(a)
            }
            (Self::Scalar(a), _, Metric::Cosine) => VTable::new::<Cosine<f32x4<Scalar>>>(a),

            // V3 //
            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Four, Metric::SquaredL2) => {
                VTable::new::<SquaredL2<f32x4<V3>>>(a)
            }
            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Four, Metric::InnerProduct) => {
                VTable::new::<InnerProduct<f32x4<V3>>>(a)
            }
            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Four, Metric::Cosine) => VTable::new::<Cosine<f32x4<V3>>>(a),

            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Eight, Metric::SquaredL2) => {
                VTable::new::<SquaredL2<f32x8<V3>>>(a)
            }
            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Eight, Metric::InnerProduct) => {
                VTable::new::<InnerProduct<f32x8<V3>>>(a)
            }
            #[cfg(target_arch = "x86_64")]
            (Self::V3(a), SIMDWidth::Eight, Metric::Cosine) => VTable::new::<Cosine<f32x8<V3>>>(a),

            // Neon //
            #[cfg(target_arch = "aarch64")]
            (Self::Neon(a), _, Metric::SquaredL2) => VTable::new::<SquaredL2<f32x4<Neon>>>(a),
            #[cfg(target_arch = "aarch64")]
            (Self::Neon(a), _, Metric::InnerProduct) => VTable::new::<InnerProduct<f32x4<Neon>>>(a),
            #[cfg(target_arch = "aarch64")]
            (Self::Neon(a), _, Metric::Cosine) => VTable::new::<Cosine<f32x4<Neon>>>(a),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GetArch;

impl Target<Scalar, RuntimeArch> for GetArch {
    #[inline(always)]
    fn run(self, arch: Scalar) -> RuntimeArch {
        RuntimeArch::Scalar(arch)
    }
}

#[cfg(target_arch = "x86_64")]
impl Target<V3, RuntimeArch> for GetArch {
    #[inline(always)]
    fn run(self, arch: V3) -> RuntimeArch {
        RuntimeArch::V3(arch)
    }
}

#[cfg(target_arch = "x86_64")]
impl Target<V4, RuntimeArch> for GetArch {
    #[inline(always)]
    fn run(self, arch: V4) -> RuntimeArch {
        RuntimeArch::V3(arch.retarget())
    }
}

#[cfg(target_arch = "aarch64")]
impl Target<Neon, RuntimeArch> for GetArch {
    #[inline(always)]
    fn run(self, arch: Neon) -> RuntimeArch {
        RuntimeArch::Neon(arch)
    }
}

//----------//
// SIMD Ops //
//----------//

trait Op {
    type Arch: Architecture;
    type Accum;
    type Vector: SIMDVector<Scalar = f32, Arch = Self::Arch>;

    fn init(arch: Self::Arch) -> Self::Accum;
    fn accum(acc: Self::Accum, x: Self::Vector, y: Self::Vector) -> Self::Accum;
    fn reduce_pair(a: Self::Accum, b: Self::Accum) -> f32;
}

#[derive(Debug)]
struct SquaredL2<V>(PhantomData<V>);

impl<V> Op for SquaredL2<V>
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
    V::Arch: Architecture,
{
    type Arch = V::Arch;
    type Accum = V;
    type Vector = V;

    fn init(arch: Self::Arch) -> V {
        V::default(arch)
    }

    fn accum(acc: V, x: V, y: V) -> V {
        let d = x - y;
        d.mul_add_simd(d, acc)
    }

    fn reduce_pair(a: V, b: V) -> f32 {
        (a + b).sum_tree()
    }
}

#[derive(Debug)]
struct InnerProduct<V>(PhantomData<V>);

impl<V> Op for InnerProduct<V>
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
    V::Arch: Architecture,
{
    type Arch = V::Arch;
    type Accum = V;
    type Vector = V;

    fn init(arch: Self::Arch) -> V {
        V::default(arch)
    }

    fn accum(acc: V, x: V, y: V) -> V {
        x.mul_add_simd(y, acc)
    }

    fn reduce_pair(a: V, b: V) -> f32 {
        -(a + b).sum_tree()
    }
}

#[derive(Debug)]
struct Cosine<V>(PhantomData<V>);

#[derive(Debug)]
struct CosineAccumulator<V> {
    xy: V,
    xnorm: V,
    ynorm: V,
}

fn finish_cosine(xy: f32, xnorm: f32, ynorm: f32) -> f32 {
    if xnorm < f32::MIN_POSITIVE || ynorm < f32::MIN_POSITIVE {
        1.0
    } else {
        let v = xy / (xnorm.sqrt() * ynorm.sqrt());
        1.0 - (-1.0f32).max(1.0f32.min(v))
    }
}

impl<V> Op for Cosine<V>
where
    V: SIMDFloat<Scalar = f32> + SIMDSumTree,
    V::Arch: Architecture,
{
    type Arch = V::Arch;
    type Accum = CosineAccumulator<V>;
    type Vector = V;

    fn init(arch: Self::Arch) -> Self::Accum {
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

    fn reduce_pair(a: Self::Accum, b: Self::Accum) -> f32 {
        let xy = (a.xy + b.xy).sum_tree();
        let xnorm = (a.xnorm + b.xnorm).sum_tree();
        let ynorm = (a.ynorm + b.ynorm).sum_tree();
        finish_cosine(xy, xnorm, ynorm)
    }
}

#[derive(Debug, Clone, Copy)]
struct OpTarget<O>(PhantomData<O>);

impl<O> FTarget3<O::Arch, Result<f32, DistanceError>, &PaddedTable, &[f32], &[u8]> for OpTarget<O>
where
    O: Op,
{
    #[inline(always)]
    fn run(arch: O::Arch, table: &PaddedTable, a: &[f32], b: &[u8]) -> Result<f32, DistanceError> {
        distance::<O>(arch, table, a, b)
    }
}

impl<O> FTarget3<O::Arch, Result<f32, SelfDistanceError>, &PaddedTable, &[u8], &[u8]>
    for OpTarget<O>
where
    O: Op,
{
    #[inline(always)]
    fn run(
        arch: O::Arch,
        table: &PaddedTable,
        a: &[u8],
        b: &[u8],
    ) -> Result<f32, SelfDistanceError> {
        self_distance::<O>(arch, table, a, b)
    }
}

//--------------------------------//
// Full Precision-Quant Distances //
//--------------------------------//

/// Errors for [`VTable::distance`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DistanceError {
    #[error("table has dimension {} but full vector has length {}", dim, alen)]
    ALen { dim: usize, alen: usize },
    #[error("table has {} chunks but codes slice has length {}", chunks, blen)]
    BLen { chunks: usize, blen: usize },
    #[error("codes have a value that exceeds the number of pivots {}", ncenters)]
    OutOfBounds { ncenters: usize },
    #[error("An invalid SIMD width was chosen - ensure the correct vtable is used")]
    InvalidSimd,
}

#[inline(always)]
fn distance<O>(
    arch: O::Arch,
    table: &PaddedTable,
    a: &[f32],
    b: &[u8],
) -> Result<f32, DistanceError>
where
    O: Op,
{
    // Check 1
    if !table
        .simd_width
        .as_nonzero()
        .get()
        .is_multiple_of(O::Vector::LANES)
    {
        return Err(DistanceError::InvalidSimd);
    }

    // Check 2
    if a.len() != table.dim() {
        return Err(DistanceError::ALen {
            dim: table.dim(),
            alen: a.len(),
        });
    }

    // Check 3
    if b.len() != table.nchunks() {
        return Err(DistanceError::BLen {
            chunks: table.nchunks(),
            blen: b.len(),
        });
    }

    // Check 4
    if let Ok(ncenters) = u8::try_from(table.ncenters())
        && let Some(max) = b.iter().max()
        && *max >= ncenters
    {
        return Err(DistanceError::OutOfBounds {
            ncenters: table.ncenters(),
        });
    }

    // All checks passed - we're good to go!

    let pivots = &table.pivots;

    let pivot_stride = pivots.ncols();
    let chunk_stride = table.pivots_per_chunk * pivot_stride;

    let nchunks = b.len();
    let mut d0 = O::init(arch);
    let mut d1 = O::init(arch);

    let mut i = 0;
    let mut p = pivots.as_ptr();

    let offsets = table.offsets.as_slice();
    let aptr = a.as_ptr();

    let lanes = O::Vector::LANES;

    // Here is our conundrum. Unrolling is nice, but there is an issue if two adjacent chunks
    // have wildly different lengths.
    //
    // So here, we process two chunks at a time. We first tackle the common full-width prefix.
    // Then we process the tails independently.
    //
    // SAFETY INVARIANTS (SI)
    //
    // 1. `offsets.len() == nchunks + 1`. Offsets are strictly increasing with
    //    `offsets[0] == 0` and `offsets[nchunks] == a.len()` by the table invariant and
    //    Check 2.
    //
    // 2. At the start of each outer iteration, `p` points to the beginning of pivot storage
    //    for chunk `i`.
    //
    // 3. Pivot chunk `i` begins at `pivots.as_ptr().add(i * chunk_stride)`. Within a chunk,
    //    pivot `j` begins at `chunk_base.add(j * pivot_stride)`. Each chunk and pivot row
    //    contains `chunk_stride` and `pivot_stride` initialized elements, respectively.
    //
    // 4. Each code selects an existing pivot. Check 4 establishes this when the number of
    //    pivots fits in `u8`; otherwise every possible `u8` is less than
    //    `pivots_per_chunk`.
    //
    // 5. `pivot_stride` is a multiple of the table's SIMD width, and Check 1 establishes
    //    that width is a multiple of `lanes`. Each pivot row is at least as long as every
    //    unpadded chunk rounded up to `lanes`.
    while i + 2 <= nchunks {
        // SAFETY: `i + 2 <= nchunks`, so offsets `i` through `i + 2` exist by SI-1.
        let (o0, o1, o2) = unsafe {
            (
                *offsets.get_unchecked(i),
                *offsets.get_unchecked(i + 1),
                *offsets.get_unchecked(i + 2),
            )
        };

        // SAFETY: `o0` and `o1` are in-bounds chunk starts in `a` by SI-1.
        let (a0, a1) = unsafe { (aptr.add(o0), aptr.add(o1)) };

        // SAFETY: By SI-(2,3,4), `p` begins chunk `i` and the code selects an in-bounds
        // pivot row in that chunk.
        let b0 = unsafe { p.add(pivot_stride * (*b.get_unchecked(i) as usize)) };

        // SAFETY: `i + 1 < nchunks`; by SI-(2,3,4), this selects an in-bounds pivot row in
        // the following chunk.
        let b1 = unsafe { p.add(chunk_stride + pivot_stride * (*b.get_unchecked(i + 1) as usize)) };

        let full0 = (o1 - o0) / lanes;
        let full1 = (o2 - o1) / lanes;

        let common = full0.min(full1);
        for j in 0..common {
            // SAFETY: `j < full0` and `j < full1`, so both query loads end at or before
            // their respective chunk boundaries. By SI-(3,4,5), the corresponding pivot
            // loads remain within their initialized padded rows.
            unsafe {
                let va = O::Vector::load_simd(arch, a0.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b0.add(lanes * j));
                d0 = O::accum(d0, va, vb);

                let va = O::Vector::load_simd(arch, a1.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b1.add(lanes * j));
                d1 = O::accum(d1, va, vb);
            }
        }

        // Handle whatever happens to be left of `a0`.
        for j in common..full0 {
            // SAFETY: `j < full0`, so the query load remains within chunk 0. By SI-(3,4,5),
            // the corresponding pivot load remains within its padded row.
            unsafe {
                let va = O::Vector::load_simd(arch, a0.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b0.add(lanes * j));
                d0 = O::accum(d0, va, vb);
            }
        }

        let a0_remaining = (o1 - o0) - full0 * lanes;
        if a0_remaining != 0 {
            // SAFETY: `full0 * lanes + a0_remaining == o1 - o0` and `a0_remaining < lanes`,
            // so this reads exactly the initialized remainder.
            let va =
                unsafe { O::Vector::load_simd_first(arch, a0.add(lanes * full0), a0_remaining) };
            // SAFETY: `(full0 + 1) * lanes` is the chunk length rounded up to `lanes`,
            // which is at most `pivot_stride` by SI-5.
            let vb = unsafe { O::Vector::load_simd(arch, b0.add(lanes * full0)) };
            d0 = O::accum(d0, va, vb);
        }

        // Handle whatever happens to be left of `a1`.
        for j in common..full1 {
            // SAFETY: Same proof as the remaining full-vector loop for chunk 0.
            unsafe {
                let va = O::Vector::load_simd(arch, a1.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b1.add(lanes * j));
                d1 = O::accum(d1, va, vb);
            }
        }

        let a1_remaining = (o2 - o1) - full1 * lanes;
        if a1_remaining != 0 {
            // SAFETY: Same remainder proof as chunk 0.
            let va =
                unsafe { O::Vector::load_simd_first(arch, a1.add(lanes * full1), a1_remaining) };
            // SAFETY: Same padded-row proof as chunk 0.
            let vb = unsafe { O::Vector::load_simd(arch, b1.add(lanes * full1)) };
            d1 = O::accum(d1, va, vb);
        }

        i += 2;

        // SAFETY: Before incrementing, `p` began chunk `i - 2` by SI-2. The loop bound
        // established `i <= nchunks`, so advancing two chunk strides reaches chunk `i` or
        // one-past the final chunk, preserving SI-2.
        p = unsafe { p.add(2 * chunk_stride) };
    }

    if i < nchunks {
        debug_assert!(i + 1 == nchunks);

        // SAFETY: `i < nchunks`, so offsets `i` and `i + 1` exist by SI-1.
        let (o0, o1) = unsafe { (*offsets.get_unchecked(i), *offsets.get_unchecked(i + 1)) };

        // SAFETY: `o0` is an in-bounds chunk start in `a` by SI-1.
        let a0 = unsafe { aptr.add(o0) };

        // SAFETY: By SI-(2,3,4), this selects an in-bounds pivot row in chunk `i`.
        let b0 = unsafe { p.add(pivot_stride * (*b.get_unchecked(i) as usize)) };

        // The number of unprocessed elements.
        let full = (o1 - o0) / lanes;
        for j in 0..full {
            // SAFETY: `j < full`, so the query load remains within the chunk. By SI-(3,4,5),
            // the corresponding pivot load remains within its padded row.
            unsafe {
                let va = O::Vector::load_simd(arch, a0.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b0.add(lanes * j));
                d0 = O::accum(d0, va, vb);
            }
        }

        let remaining = (o1 - o0) - full * lanes;
        if remaining != 0 {
            // SAFETY: `full * lanes + remaining == o1 - o0` and `remaining < lanes`, so
            // this reads exactly the initialized remainder.
            let va = unsafe { O::Vector::load_simd_first(arch, a0.add(lanes * full), remaining) };
            // SAFETY: `(full + 1) * lanes` is at most `pivot_stride` by SI-5.
            let vb = unsafe { O::Vector::load_simd(arch, b0.add(lanes * full)) };
            d0 = O::accum(d0, va, vb);
        }
    }

    Ok(O::reduce_pair(d0, d1))
}

//-----------------------//
// Quant-Quant Distances //
//-----------------------//

/// Errors for [`VTable::self_distance`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SelfDistanceError {
    #[error("table has {} chunks but codes slice has length {}", chunks, len)]
    Len { chunks: usize, len: usize },
    #[error("codes have a value that exceeds the number of pivots {}", ncenters)]
    OutOfBounds { ncenters: usize },
    #[error("An invalid SIMD width was chosen - ensure the correct vtable is used")]
    InvalidSimd,
}

#[inline(always)]
fn self_distance<O>(
    arch: O::Arch,
    table: &PaddedTable,
    a: &[u8],
    b: &[u8],
) -> Result<f32, SelfDistanceError>
where
    O: Op,
{
    // Check 1
    if !table
        .simd_width
        .as_nonzero()
        .get()
        .is_multiple_of(O::Vector::LANES)
    {
        return Err(SelfDistanceError::InvalidSimd);
    }

    let chunks = table.nchunks();
    // Check 2
    if a.len() != chunks {
        return Err(SelfDistanceError::Len {
            chunks,
            len: a.len(),
        });
    }

    // Check 3
    if b.len() != chunks {
        return Err(SelfDistanceError::Len {
            chunks,
            len: b.len(),
        });
    }

    // Check 4
    if let Ok(ncenters) = u8::try_from(table.ncenters()) {
        if let Some(max) = a.iter().max()
            && *max >= ncenters
        {
            return Err(SelfDistanceError::OutOfBounds {
                ncenters: ncenters.into(),
            });
        }

        if let Some(max) = b.iter().max()
            && *max >= ncenters
        {
            return Err(SelfDistanceError::OutOfBounds {
                ncenters: ncenters.into(),
            });
        }
    }

    // All checks passed - we're good to go!

    let pivots = &table.pivots;

    // The number of SIMD steps to process for each pivot.
    let steps = pivots.ncols() / O::Vector::LANES;

    let pivot_stride = pivots.ncols();
    let chunk_stride = table.pivots_per_chunk * pivot_stride;

    let len = a.len();
    let mut d0 = O::init(arch);
    let mut d1 = O::init(arch);

    let mut p = pivots.as_ptr();

    let lanes = O::Vector::LANES;
    let mut i = 0;

    // SAFETY INVARIANTS (SI)
    //
    // 1. Both code slices contain exactly `len == table.nchunks()` entries by Checks 2
    //    and 3.
    //
    // 2. At the start of each outer iteration, `p` points to the beginning of pivot storage
    //    for chunk `i`.
    //
    // 3. Pivot chunk `i` begins at `pivots.as_ptr().add(i * chunk_stride)`. Within a chunk,
    //    pivot `j` begins at `chunk_base.add(j * pivot_stride)`. Each chunk and pivot row
    //    contains `chunk_stride` and `pivot_stride` initialized elements, respectively.
    //
    // 4. Every code selects an existing pivot. Check 4 establishes this when the number of
    //    pivots fits in `u8`; otherwise every possible `u8` is in bounds.
    //
    // 5. `pivot_stride` is a multiple of the table's SIMD width, and Check 1 establishes
    //    that width is a multiple of `lanes`. Therefore
    //    `steps * lanes == pivot_stride`.
    while i + 2 <= len {
        // SAFETY: `i + 2 <= len`; by SI-(1,2,3,4), `p` begins chunk `i` and this code
        // selects an in-bounds pivot row in that chunk.
        let a0 = unsafe { p.add(pivot_stride * (*a.get_unchecked(i) as usize)) };

        // SAFETY: By the same invariants, this selects an in-bounds pivot row in chunk
        // `i + 1`.
        let a1 = unsafe { p.add(chunk_stride + pivot_stride * (*a.get_unchecked(i + 1) as usize)) };

        // SAFETY: Same proof as `a0`; SI-1 establishes that `b[i]` exists.
        let b0 = unsafe { p.add(pivot_stride * (*b.get_unchecked(i) as usize)) };
        // SAFETY: Same proof as `a1`.
        let b1 = unsafe { p.add(chunk_stride + pivot_stride * (*b.get_unchecked(i + 1) as usize)) };

        for j in 0..steps {
            // SAFETY: `j < steps` and `steps * lanes == pivot_stride` by SI-5, so every
            // full-vector load remains within its selected initialized pivot row.
            unsafe {
                // Unroll 0
                let va = O::Vector::load_simd(arch, a0.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b0.add(lanes * j));
                d0 = O::accum(d0, va, vb);

                // Unroll 1
                let va = O::Vector::load_simd(arch, a1.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b1.add(lanes * j));
                d1 = O::accum(d1, va, vb);
            }
        }

        i += 2;

        // SAFETY: Before incrementing, `p` began chunk `i - 2` by SI-2. The loop bound
        // established `i <= len`, so advancing two chunk strides reaches chunk `i` or
        // one-past the final chunk, preserving SI-2.
        p = unsafe { p.add(2 * chunk_stride) };
    }

    if i < len {
        debug_assert!(i + 1 == len);

        // SAFETY: By SI-(1,2,3,4), `p` begins chunk `i` and this selects an in-bounds
        // pivot row.
        let a0 = unsafe { p.add(pivot_stride * (*a.get_unchecked(i) as usize)) };
        // SAFETY: Same proof for `b[i]`.
        let b0 = unsafe { p.add(pivot_stride * (*b.get_unchecked(i) as usize)) };

        for j in 0..steps {
            // SAFETY: `j < steps`, so both loads remain within their initialized pivot rows
            // by SI-5.
            unsafe {
                let va = O::Vector::load_simd(arch, a0.add(lanes * j));
                let vb = O::Vector::load_simd(arch, b0.add(lanes * j));
                d0 = O::accum(d0, va, vb);
            }
        }
    }

    Ok(O::reduce_pair(d0, d1))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::assert_contains;

    use crate::{
        product::tables::test::{self as table_test, DistanceTestTable, QueryLike, SelfLike},
        test_util::Check,
    };

    fn cases() -> &'static [Case] {
        const CASES: &[Case] = &[
            // chunk dimensions, pivots, start
            Case::new(&[1], 1, 0.0, Check::exact()),
            Case::new(&[1, 1, 1], 2, -1.0, Check::exact()),
            Case::new(&[3, 2, 2], 15, -7.0, Check::exact()),
            Case::new(&[2, 2, 2, 2], 16, -8.0, Check::exact()),
            Case::new(&[3, 3, 3, 2, 2], 17, -11.0, Check::exact()),
            Case::new(&[8, 8, 8, 8], 32, -16.0, Check::exact()),
            Case::new(&[6, 6, 5, 5, 5, 5, 5], 33, -20.0, Check::exact()),
            Case::new(&[13, 12, 12], 33, -20.0, Check::exact()),
            Case::new(&[4, 4, 3, 3, 3], 256, -128.0, Check::exact()),
            // Exercise sharply imbalanced chunks in either position of an unrolled pair,
            // followed by an odd chunk that must use the remainder loop.
            Case::new(&[1, 23, 23], 17, -9.0, Check::exact()),
            Case::new(&[23, 1, 23], 17, -9.0, Check::exact()),
            Case::new(&[1, 17, 2, 9, 3], 17, -9.0, Check::exact()),
        ];
        CASES
    }

    #[derive(Debug, Clone, Copy)]
    struct Case {
        chunk_dims: &'static [usize],
        pivots: usize,
        start: f32,
        check: Check,
    }

    impl Case {
        const fn new(
            chunk_dims: &'static [usize],
            pivots: usize,
            start: f32,
            check: Check,
        ) -> Self {
            Self {
                chunk_dims,
                pivots,
                start,
                check,
            }
        }
    }

    fn run_test(
        cases: &[Case],
        metric: Metric,
        arch: Option<RuntimeArch>,
        reference: &dyn Fn(&[f32], &[f32]) -> f32,
        ctx: &dyn std::fmt::Display,
    ) {
        let (num_queries, num_trials) = if cfg!(miri) {
            // The driver will run some directed tests even if there are no regular random
            // trials.
            (1, 0)
        } else {
            (10, 10)
        };

        #[derive(Debug)]
        struct Dut<'a> {
            table: &'a PaddedTable,
            query: Vec<f32>,
            vtable: VTable,
        }

        impl QueryLike for Dut<'_> {
            fn preprocess(&mut self, query: &[f32]) {
                self.query.clear();
                self.query.extend_from_slice(query);
            }

            fn evaluate(&mut self, code: &[u8]) -> f32 {
                self.vtable.distance(self.table, &self.query, code).unwrap()
            }
        }

        impl SelfLike for Dut<'_> {
            fn evaluate(&mut self, a: &[u8], b: &[u8]) -> f32 {
                self.vtable.self_distance(self.table, a, b).unwrap()
            }
        }

        for Case {
            chunk_dims,
            pivots,
            start,
            check,
        } in cases.iter().copied()
        {
            let driver = DistanceTestTable::from_chunk_dims(chunk_dims, pivots, start);
            let dim = driver.dim();
            let chunks = driver.chunks();
            let basic = driver.basic_table();

            let table = match arch {
                Some(arch) => PaddedTable::from_basic_with(basic.as_view(), arch),
                None => PaddedTable::from_basic(basic.as_view()),
            };

            let vtable = table.vtable(metric);

            let mut dut = Dut {
                table: &table,
                query: Vec::new(),
                vtable,
            };

            driver.drive_query_like(
                num_queries,
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                reference,
                &mut dut,
                format_args!(
                    "[{}] padded table - dim = {}, chunks = {}, pivots = {}",
                    ctx, dim, chunks, pivots
                ),
            );

            driver.drive_self_like(
                num_trials,
                &mut driver.rng(0xc0ffee),
                check,
                reference,
                &mut dut,
                format_args!(
                    "[{}] padded table - dim = {}, chunks = {}, pivots = {}",
                    ctx, dim, chunks, pivots
                ),
            );
        }
    }

    // L2 - query-like
    #[test]
    fn test_l2_query_like() {
        run_test(
            cases(),
            Metric::SquaredL2,
            None,
            &table_test::squared_l2,
            &"squared l2 full x quant - auto-detect",
        );

        run_test(
            cases(),
            Metric::SquaredL2,
            Some(RuntimeArch::Scalar(Scalar::new())),
            &table_test::squared_l2,
            &"squared l2 full x quant - scalar",
        );

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = V3::new_checked() {
            run_test(
                cases(),
                Metric::SquaredL2,
                Some(RuntimeArch::V3(arch)),
                &table_test::squared_l2,
                &"squared l2 full x quant - V3",
            );
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = Neon::new_checked() {
            run_test(
                cases(),
                Metric::SquaredL2,
                Some(RuntimeArch::Neon(arch)),
                &table_test::squared_l2,
                &"squared l2 full x quant - V3",
            );
        }
    }

    // Inner Product - query-like
    #[test]
    fn test_inner_product_query_like() {
        run_test(
            cases(),
            Metric::InnerProduct,
            None,
            &table_test::inner_product,
            &"inner-product full x quant - auto-detect",
        );

        run_test(
            cases(),
            Metric::InnerProduct,
            Some(RuntimeArch::Scalar(Scalar::new())),
            &table_test::inner_product,
            &"inner-product full x quant - scalar",
        );

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = V3::new_checked() {
            run_test(
                cases(),
                Metric::InnerProduct,
                Some(RuntimeArch::V3(arch)),
                &table_test::inner_product,
                &"inner-product full x quant - V3",
            );
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = Neon::new_checked() {
            run_test(
                cases(),
                Metric::InnerProduct,
                Some(RuntimeArch::Neon(arch)),
                &table_test::inner_product,
                &"inner-product full x quant - V3",
            );
        }
    }

    // Cosine - query-like
    #[test]
    fn test_cosine_query_like() {
        run_test(
            cases(),
            Metric::Cosine,
            None,
            &table_test::cosine,
            &"cosine full x quant - auto-detect",
        );

        run_test(
            cases(),
            Metric::Cosine,
            Some(RuntimeArch::Scalar(Scalar::new())),
            &table_test::cosine,
            &"cosine full x quant - scalar",
        );

        #[cfg(target_arch = "x86_64")]
        if let Some(arch) = V3::new_checked() {
            run_test(
                cases(),
                Metric::Cosine,
                Some(RuntimeArch::V3(arch)),
                &table_test::cosine,
                &"cosine full x quant - V3",
            );
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(arch) = Neon::new_checked() {
            run_test(
                cases(),
                Metric::Cosine,
                Some(RuntimeArch::Neon(arch)),
                &table_test::cosine,
                &"cosine full x quant - V3",
            );
        }
    }

    ////////////
    // Errors //
    ////////////

    /// A table with dimension 7, 3 chunks, and 3 pivots per chunk.
    fn error_table() -> (PaddedTable, VTable) {
        let driver = DistanceTestTable::new(7, 3, 3, 0.0);
        let basic = driver.basic_table();
        let table =
            PaddedTable::from_basic_with(basic.as_view(), RuntimeArch::Scalar(Scalar::new()));
        let vtable = table.vtable(Metric::SquaredL2);
        (table, vtable)
    }

    #[test]
    fn test_distance_errors() {
        let (table, vtable) = error_table();
        let vector = vec![0.0; table.dim()];
        let codes = vec![0; table.nchunks()];

        for len in [table.dim() - 1, table.dim() + 1] {
            let invalid = vec![0.0; len];
            let err = vtable.distance(&table, &invalid, &codes).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!(
                    "table has dimension {} but full vector has length {len}",
                    table.dim()
                )
            );
        }

        for len in [table.nchunks() - 1, table.nchunks() + 1] {
            let invalid = vec![0; len];
            let err = vtable.distance(&table, &vector, &invalid).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!(
                    "table has {} chunks but codes slice has length {len}",
                    table.nchunks()
                )
            );
        }

        let mut invalid = codes;
        invalid[1] = u8::try_from(table.ncenters()).unwrap();
        let err = vtable.distance(&table, &vector, &invalid).unwrap_err();
        assert_contains!(
            err.to_string(),
            format!(
                "codes have a value that exceeds the number of pivots {}",
                table.ncenters()
            )
        );
    }

    #[test]
    fn test_self_distance_errors() {
        let (table, vtable) = error_table();
        let codes = vec![0; table.nchunks()];

        for len in [table.nchunks() - 1, table.nchunks() + 1] {
            let invalid = vec![0; len];

            let err = vtable.self_distance(&table, &invalid, &codes).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!(
                    "table has {} chunks but codes slice has length {len}",
                    table.nchunks()
                )
            );

            let err = vtable.self_distance(&table, &codes, &invalid).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!(
                    "table has {} chunks but codes slice has length {len}",
                    table.nchunks()
                )
            );
        }

        for invalid_operand in 0..2 {
            let mut a = codes.clone();
            let mut b = codes.clone();
            let invalid = if invalid_operand == 0 { &mut a } else { &mut b };
            invalid[1] = u8::try_from(table.ncenters()).unwrap();

            let err = vtable.self_distance(&table, &a, &b).unwrap_err();
            assert_contains!(
                err.to_string(),
                format!(
                    "codes have a value that exceeds the number of pivots {}",
                    table.ncenters()
                )
            );
        }
    }

    #[test]
    fn test_mismatched_simd_width_errors() {
        diskann_wide::alias!(f32x8<A> = f32x8);

        let (table, _) = error_table();
        let invalid_vtable = VTable::new::<SquaredL2<f32x8<Scalar>>>(Scalar::new());
        let vector = vec![0.0; table.dim()];
        let codes = vec![0; table.nchunks()];

        let err = invalid_vtable
            .distance(&table, &vector, &codes)
            .unwrap_err();
        assert_contains!(err.to_string(), "An invalid SIMD width was chosen");

        let err = invalid_vtable
            .self_distance(&table, &codes, &codes)
            .unwrap_err();
        assert_contains!(err.to_string(), "An invalid SIMD width was chosen");
    }
}
