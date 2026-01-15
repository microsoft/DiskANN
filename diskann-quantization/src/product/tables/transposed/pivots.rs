/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt;

use diskann_utils::strided;
use diskann_wide::{SIMDMask, SIMDMulAdd, SIMDPartialOrd, SIMDSelect, SIMDVector};

use crate::{
    algorithms::kmeans::{self, BlockTranspose},
    distances::{InnerProduct, SquaredL2},
};

// The `Wide` type used as the group granularity for `Chunk`.
diskann_wide::alias!(f32s = f32x8);
diskann_wide::alias!(u32s = u32x8);

/// Error types returned by Chunk construction.
#[derive(Debug, Clone)]
pub enum ChunkConstructionError {
    /// A `StridedView` was provided with a dimension of zero.
    DimensionCannotBeZero,
    /// A `StridedView` was provided with a length of zero.
    LengthCannotBeZero,
}

impl fmt::Display for ChunkConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ChunkConstructionError::DimensionCannotBeZero => write!(
                f,
                "cannot construct a Chunk from a source with zero dimensions"
            )?,
            ChunkConstructionError::LengthCannotBeZero => {
                write!(f, "cannot construct a Chunk from a source with zero length")?
            }
        }
        Ok(())
    }
}

impl std::error::Error for ChunkConstructionError {}

/// The index of the closest center found during compression.
///
/// This value has an inlined-error state to indicate that compression could not complete
/// due to the presence of infinities in the input (or pivot) data.
///
/// In the case of this struct, value of `u32::MAX` indicates the error state.
#[derive(Debug, Clone, Copy)]
pub struct CompressionResult(u32);

impl CompressionResult {
    /// Construct a new version of `self` in the error state.
    fn err() -> Self {
        Self(u32::MAX)
    }

    /// Return the inner value directly, regardless of whether it is in an error state or not.
    pub(super) fn into_inner(self) -> u32 {
        self.0
    }

    /// Return whether or not the result is in a valid state.
    pub fn is_okay(&self) -> bool {
        self.0 != u32::MAX
    }

    /// Unpack `self`.
    ///
    /// If `self.is_okay()`, then call `ok` with the interior value. Otherwise, return the
    /// result of `err`.
    ///
    /// Only one of `ok` or `err` will be called.
    pub fn map<F, G, R, E>(self, ok: F, err: G) -> Result<R, E>
    where
        F: FnOnce(u32) -> R,
        G: FnOnce() -> E,
    {
        if self.is_okay() {
            Ok(ok(self.0))
        } else {
            Err(err())
        }
    }

    #[cfg(test)]
    pub(crate) fn unwrap(self) -> u32 {
        assert!(self.is_okay());
        self.0
    }
}

/// A collection of PQ centroids stored in a block-transposed form.
///
/// This block-transposed form is meant to allow fast closest-centroid computations with
/// the corresponding chunk. A sketch of the layout is described below.
///
/// Suppose we have a large number of 3-dimensional centroids:
/// ```ignore
/// a0 a1 a2
/// b0 b1 b2
/// c0 c1 c2
/// d0 d1 d2
/// ...
/// z0 z1 z2
/// ```
/// The block-transpose layout (with blocking factor 4) looks like this:
/// ```ignore
/// a0 b0 c0 d0  | Group 0 |
/// a1 b1 c1 d1  | Group 1 | Block 0
/// a2 b2 c2 d2  | Group 2 |
///
/// e0 f0 g0 h0  | Group 0 |
/// e1 f1 g1 h1  | Group 1 | Block 1
/// e2 f2 g2 h2  | Group 2 |
///
/// i0 j0 k0 l0  | Group 0 |
/// i1 j1 k1 l1  | Group 1 | Block 2
/// i2 j2 k2 l2  | Group 2 |
/// ...             ...
/// ```
/// Where each block is laid out in row-major order.
///
/// The advantage of this layout is that a sub-group of a block can be loaded into SIMD
/// registers to compute the partial distance between a dimension of a query chunk and
/// the corresponding dimension of all elements of a group.
///
/// For example, the L2 distance between a query `q0 q1 q2` and the contents of `block 0`
/// (centroids `a`, `b`, `c`, and `d`) can be computed by:
///
/// 1. Broadcasting `q0` to all lanes of a SIMD vector `A = (q0 q0 q0 q0)`.
/// 2. Loading `Group 0` into a SIMD register `B = (a0 b0 c0 d0)`.
/// 3. Subtract and square registers `A` and `B` so
///    `C = ((q0-a0)^2 (q0-b0)^2 (q0-c0)^2 (q0-d0)^2`
/// 4. Repeat for the remaining dimensions, accumulating the results.
///
/// In the end, we will be left with a SIMD register containing the distance between the
/// query and `a`, `b`, `c`, and `d` without requiring a horizontal summation.
///
/// These partial distances can then be preprocessed efficiently to find the minimum
/// distance.
///
/// Now, the actual groupsize is dependent on the hardware being used, but this explanation
/// describes the gist of what this class is trying to accomplish.
#[derive(Debug)]
pub struct Chunk {
    /// The data actually underlying the blocked representation.
    data: BlockTranspose<16>,
    /// The squared norms of each center.
    square_norms: Vec<f32>,
}

impl Chunk {
    const fn groupsize() -> usize {
        BlockTranspose::<16>::const_group_size()
    }

    /// The number of queries that can be processed at a time in an efficient manner.
    pub(super) const fn batchsize() -> usize {
        4
    }

    /// Return the dimensionality of the each center in the Chunk.
    pub(super) fn dimension(&self) -> usize {
        self.data.ncols()
    }

    /// Return the number of centers contained in the Chunk.
    pub(super) fn num_centers(&self) -> usize {
        self.data.nrows()
    }

    /// Return the number of blocks contained in ths Chunk.
    pub(super) fn num_blocks(&self) -> usize {
        self.data.num_blocks()
    }

    /// Return the number of underfilled groups in the last block.
    ///
    /// This is used in the case where the number of centers does not evenly divide the
    /// number of blocks, resulting in the last block containing fewer groups.
    pub(super) fn remainder(&self) -> usize {
        self.data.remainder()
    }

    /// Retrieve the value originally stored in `(row, col)` of the input matrix.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.num_centers()` or `col >= self.dimensions()`.
    #[cfg(test)]
    pub(super) fn get(&self, row: usize, col: usize) -> f32 {
        assert!(
            row < self.num_centers(),
            "row {} must be less than {}",
            row,
            self.num_centers()
        );
        assert!(
            col < self.dimension(),
            "col {} must be less than {}",
            col,
            self.dimension()
        );

        self.data[(row, col)]
    }

    /// Create a new `Chunk` from the contents of the provided window.
    ///
    /// These contents are implicitly identified by their position in `data`, so if
    /// `find_closest` returns the index `5`, then it is referring to the slice that
    /// is obtained by `&data[5]`.
    ///
    /// Returns `ChunkConstructionError` under the following circumstances:
    ///
    /// 1. `data.ncols() == 0`
    /// 2. `data.nrows() == 0`
    pub(super) fn new(data: strided::StridedView<'_, f32>) -> Result<Self, ChunkConstructionError> {
        // Error handling.
        if data.ncols() == 0 {
            return Err(ChunkConstructionError::DimensionCannotBeZero);
        }
        if data.nrows() == 0 {
            return Err(ChunkConstructionError::LengthCannotBeZero);
        }

        let square_norms = data.row_iter().map(kmeans::square_norm).collect();
        let data = BlockTranspose::from_strided(data);
        Ok(Self { data, square_norms })
    }

    /// Return the number of full blocks.
    fn full_blocks(&self) -> usize {
        self.data.full_blocks()
    }

    /// Return the index of the closest center to the query.
    ///
    /// Ties are resolved to the lowest index.
    ///
    /// If the distances between `x` and all centers is not finite, return
    /// `CompressionResult::err()`. Otherwise, the returned `CompressionResult` is valid
    /// with a value less than `self.num_centers()`.
    ///
    /// # Panics
    ///
    /// Panics if `x.len() != self.dimension()`.
    pub(super) fn find_closest<T>(&self, x: &[T]) -> CompressionResult
    where
        T: Copy + Into<f32>,
    {
        // The query vector should have the same length as the pivot chunks.
        assert_eq!(x.len(), self.dimension(), "incorrect query dimension");

        // Vectorize the minimum distance calculation.
        let mut min_distances: f32s = f32s::splat(diskann_wide::ARCH, f32::INFINITY);

        let mut min_indices = u32s::splat(diskann_wide::ARCH, u32::MAX);
        let index_offsets = u32s::from_array(diskann_wide::ARCH, [0, 1, 2, 3, 4, 5, 6, 7]);
        let full_blocks = self.full_blocks();

        // This loop only handles full blocks.
        // The remainder is handles next.
        for block in 0..full_blocks {
            let (d0, d1) = self.compute_in_block::<InnerProductMathematical, T>(x, block);

            // SAFETY: `block < self.full-blocks()`.
            let (norm0, norm1) = unsafe { self.load_norms(block) };

            let d0 = norm0 - (d0 + d0);
            let d1 = norm1 - (d1 + d1);

            // Compute the distance between the query and all elements in the block.
            (min_distances, min_indices) = update_tracking_with(
                min_distances,
                min_indices,
                d0,
                (Self::groupsize() * block) as u32,
                index_offsets,
            );
            (min_distances, min_indices) = update_tracking_with(
                min_distances,
                min_indices,
                d1,
                (Self::groupsize() * block + f32s::LANES) as u32,
                index_offsets,
            )
        }

        // If there are remaining elements - handle those.
        let remainder = self.remainder();
        if remainder != 0 {
            let (d0, d1) = self.compute_in_remainder::<InnerProductMathematical, T>(x);

            // SAFETY: We've verified that the remainder is nonzero.
            let (norm0, norm1) = unsafe { self.load_remainder_norms() };

            let d0 = norm0 - (d0 + d0);
            let d1 = norm1 - (d1 + d1);

            // Compute the distance between the query and all elements in the block.
            (min_distances, min_indices) = update_tracking_with(
                min_distances,
                min_indices,
                d0,
                (Self::groupsize() * full_blocks) as u32,
                index_offsets,
            );

            (min_distances, min_indices) = update_tracking_with(
                min_distances,
                min_indices,
                d1,
                (Self::groupsize() * full_blocks + f32s::LANES) as u32,
                index_offsets,
            );
        }

        // The true minimum index is in one of the lanes of the tracking variable.
        let mut minimum_distance = f32::MAX;
        let mut minimum_index = u32::MAX;
        for (&i, &d) in std::iter::zip(
            min_indices.to_array().iter(),
            min_distances.to_array().iter(),
        ) {
            if d < minimum_distance {
                minimum_distance = d;
                minimum_index = i;
            }
        }

        if minimum_distance.is_finite() {
            CompressionResult(minimum_index)
        } else {
            CompressionResult::err()
        }
    }

    /// Find the closest center to each query in the batch with the same semantics as
    /// `find_closest`.
    ///
    /// This method is generally more efficient to call.
    ///
    /// **IMPORTANT**: The provided `StridedView` must have a length of `Self::batchsize()`.
    ///
    /// Providing a `StridedView` as an argument allows the compiler to infer that each
    /// row in `x` has a strided offset from the base, allowing for better code generation.
    ///
    /// If the distances between a row in `x` and all centers is not finite, return
    /// `CompressionResult::err()`. Otherwise, the returned `CompressionResult` is valid
    /// with a value less than `self.num_centers()`.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `x.len() != Self::batchsize()`
    /// * `x.dim() != self.dimension()`
    ///
    /// # Implementation Details
    ///
    /// This function computes the nearest center to each argument `x` using the identity
    /// ```ignore
    /// L2(x, y) = sum((x[i] - y[i])^2 over i)
    ///          = sum(x[i]^2) + sum(y[i]^2) - 2 * sum(x[i] * y[i])
    ///            -----------   -----------       ----------------
    ///                 |             |                    |
    ///            norm squared       |              inner product
    ///                          norm squared
    /// ```
    /// We're seeking the center `y` that minimizes the L2 distance.
    ///
    /// Of these terms, the square norm of each center was computed upon Chunk construction m
    /// and for each argument `x`, its norm is constant and therefore does not need to be
    /// computed when finding the closest center `y`.
    ///
    /// That means it is sufficient to compute the inner product between `x` and the centers
    /// and apply the correction factor inline when tracking the closest center.
    ///
    /// Internally, BLAS style broadcasting + FMA's are used to compute the batch of
    /// inner products.
    pub(super) fn find_closest_batch<T>(
        &self,
        x: strided::StridedView<'_, T>,
    ) -> [CompressionResult; Self::batchsize()]
    where
        T: Copy + Into<f32>,
    {
        // Assertion 1.
        assert_eq!(
            x.nrows(),
            Self::batchsize(),
            "argument StridedView must have a length of {}",
            Self::batchsize()
        );

        // Assertion 2.
        assert_eq!(x.ncols(), self.dimension(), "incorrect query dimension");
        let dim = self.dimension();

        // Stack allocated tracking variables.
        let mut tracking: [(f32s, u32s); Self::batchsize()] = [
            (
                f32s::splat(diskann_wide::ARCH, f32::INFINITY),
                u32s::splat(diskann_wide::ARCH, u32::MAX),
            ),
            (
                f32s::splat(diskann_wide::ARCH, f32::INFINITY),
                u32s::splat(diskann_wide::ARCH, u32::MAX),
            ),
            (
                f32s::splat(diskann_wide::ARCH, f32::INFINITY),
                u32s::splat(diskann_wide::ARCH, u32::MAX),
            ),
            (
                f32s::splat(diskann_wide::ARCH, f32::INFINITY),
                u32s::splat(diskann_wide::ARCH, u32::MAX),
            ),
        ];

        let index_offsets = u32s::from_array(diskann_wide::ARCH, [0, 1, 2, 3, 4, 5, 6, 7]);

        // Unfortunately - we are pushing the limits of what the compilers is able to do
        // regarding elision of bounds checks.
        //
        // Due to the shear number of loads emitted by the BLAS style micro-kernel, this is
        // one situation where it truly is helpful to manually elide the bounds checks.
        //
        // # Safety
        //
        // The index `k` must be strictly less than `x.dim()` to ensure the load is in-bounds.
        let unsafe_load_x = |k| {
            debug_assert!(k < x.ncols());

            // SAFETY:
            // * `StridedView` indexing: We have checked in Assertion 1 that it is safe
            //   to index `x` at indices `0, 1, 2, and 3`.
            // * Inner slice indexing: It is the caller's responsibility to ensure that
            //   `k < x.dim()` so that indexing the slices return by `x.get_unchecked` are
            //   in-bounds.
            unsafe {
                // We should have already checked this invariant at the top of this
                // routine, but lets be conservative.
                debug_assert!(3 < x.nrows());
                (
                    f32s::splat(
                        diskann_wide::ARCH,
                        <T as Into<f32>>::into(*x.get_row_unchecked(0).get_unchecked(k)),
                    ),
                    f32s::splat(
                        diskann_wide::ARCH,
                        <T as Into<f32>>::into(*x.get_row_unchecked(1).get_unchecked(k)),
                    ),
                    f32s::splat(
                        diskann_wide::ARCH,
                        <T as Into<f32>>::into(*x.get_row_unchecked(2).get_unchecked(k)),
                    ),
                    f32s::splat(
                        diskann_wide::ARCH,
                        <T as Into<f32>>::into(*x.get_row_unchecked(3).get_unchecked(k)),
                    ),
                )
            }
        };
        let remainder = self.remainder();
        let base_ptr = self.data.as_ptr();
        const STRIDE: usize = Chunk::groupsize();

        // Load the data for dimension `dim_offset` for `STRIDE` different pivots in the
        // provided block.
        //
        // This function is unsafe to call.
        //
        // # Safety
        //
        // * `block < self.num_blocks()`: Block must be a valid block index for the pivot
        //   data.
        // * `dim_offset < self.dimensions()`: The requested dimension must be inbounds.
        let unsafe_load_pivots = |block, dim_offset| -> (f32s, f32s) {
            debug_assert!(block < self.num_blocks());
            debug_assert!(dim_offset < self.dimension());
            // Compute the linear offset.
            let index = STRIDE * (block * dim + dim_offset);

            // SAFETY: Assuming the input conditions hold, then `BlockTranpose` assures us
            // that this is inbounds.
            unsafe {
                (
                    f32s::load_simd(diskann_wide::ARCH, base_ptr.add(index)),
                    f32s::load_simd(diskann_wide::ARCH, base_ptr.add(index + f32s::LANES)),
                )
            }
        };

        // Loop A
        for i in 0..self.num_blocks() {
            // Here, we process 16 pivots at a time for the each of the argument queries.
            // The variable `dX_Y` contains the distances for query `x` to the subgroup `Y`.
            //
            // Since the size of each subgroup is 8 (the native vector width on AVX2),
            // `Y` takes on the value 0 and 1.
            let mut d0_0 = f32s::default(diskann_wide::ARCH);
            let mut d0_1 = f32s::default(diskann_wide::ARCH);

            let mut d1_0 = f32s::default(diskann_wide::ARCH);
            let mut d1_1 = f32s::default(diskann_wide::ARCH);

            let mut d2_0 = f32s::default(diskann_wide::ARCH);
            let mut d2_1 = f32s::default(diskann_wide::ARCH);

            let mut d3_0 = f32s::default(diskann_wide::ARCH);
            let mut d3_1 = f32s::default(diskann_wide::ARCH);

            // Unroll by a factor fo 2 along the dimension axis.
            const INNER_UNROLL: usize = 2;
            let unrolled_iterations = dim / INNER_UNROLL;

            // Loop B
            for j in 0..unrolled_iterations {
                let j_linear = INNER_UNROLL * j;

                // SAFETY: The requirements for `unsafe_load_x` are satisfied because
                // * By Assertion 2, we know `dim = x.dimension()`.
                // * The bounds on Loop B ensures that `j_linear < dim`.
                let (a0, a1, a2, a3) = unsafe_load_x(j_linear);

                // SAFETY: The requirements for `unsafe_load_pivots` are satisfied because
                // * By Check A0, `i < self.num_blocks()`
                // * The bounds on Loop B ensures that `j_linear < dim`.
                let (b0, b1) = unsafe_load_pivots(i, j_linear);

                d0_0 = a0.mul_add_simd(b0, d0_0);
                d0_1 = a0.mul_add_simd(b1, d0_1);

                d1_0 = a1.mul_add_simd(b0, d1_0);
                d1_1 = a1.mul_add_simd(b1, d1_1);

                d2_0 = a2.mul_add_simd(b0, d2_0);
                d2_1 = a2.mul_add_simd(b1, d2_1);

                d3_0 = a3.mul_add_simd(b0, d3_0);
                d3_1 = a3.mul_add_simd(b1, d3_1);

                // Unrolled iteration 2

                // SAFETY: The requirements for `unsafe_load_x` are satisfied because
                // * By Assertion 2, we know `dim = x.dimension()`.
                // * The bounds on Loop B ensure that `j_linear + 1 < dim`.
                let (a0, a1, a2, a3) = unsafe_load_x(j_linear + 1);

                // SAFETY: The requirements for `unsafe_load_pivots` are satisfied because
                // * By Check A0, `i < self.num_blocks()`
                // * The bounds on Loop B ensures that `j_linear + 1 < dim`.
                let (b0, b1) = unsafe_load_pivots(i, j_linear + 1);

                d0_0 = a0.mul_add_simd(b0, d0_0);
                d0_1 = a0.mul_add_simd(b1, d0_1);

                d1_0 = a1.mul_add_simd(b0, d1_0);
                d1_1 = a1.mul_add_simd(b1, d1_1);

                d2_0 = a2.mul_add_simd(b0, d2_0);
                d2_1 = a2.mul_add_simd(b1, d2_1);

                d3_0 = a3.mul_add_simd(b0, d3_0);
                d3_1 = a3.mul_add_simd(b1, d3_1);
            }

            // Check if we have one more to perform.
            let last_unrolled = INNER_UNROLL * unrolled_iterations;

            // Check B1
            if last_unrolled != dim {
                debug_assert!(last_unrolled + 1 == dim);
                // SAFETY: The requirements for `unsafe_load_x` are satisfied because
                // * By Assertion 2, we know `dim = x.dimension()`.
                // * The check B1 ensures that `last_unrolled < dim`.
                let (a0, a1, a2, a3) = unsafe_load_x(last_unrolled);

                // SAFETY: The requirements for `unsafe_load_pivots` are satisfied because
                // * By Check A0, `i < self.num_blocks()`
                // * The check B1 ensures that `last_unrolled < dim`.
                let (b0, b1) = unsafe_load_pivots(i, last_unrolled);

                d0_0 = a0.mul_add_simd(b0, d0_0);
                d0_1 = a0.mul_add_simd(b1, d0_1);

                d1_0 = a1.mul_add_simd(b0, d1_0);
                d1_1 = a1.mul_add_simd(b1, d1_1);

                d2_0 = a2.mul_add_simd(b0, d2_0);
                d2_1 = a2.mul_add_simd(b1, d2_1);

                d3_0 = a3.mul_add_simd(b0, d3_0);
                d3_1 = a3.mul_add_simd(b1, d3_1);
            }

            // Make the upper lanes infinity if this is the last block.
            //
            // Note that this block is only reachable if Check A-0 is false, which
            // means that we just finished processing two blocks.
            //
            // Of these, only the second block can be partially full, so that is the
            // only one we need to consider to distance masking.
            let (norm_0, norm_1) = if remainder != 0 && i + 1 == self.num_blocks() {
                // Since we are about to compensate for computing inner products
                // rather than L2, we mask out unused lanes to `-infinity`.
                //
                // Subtraction will turn this into `+infinity`.
                let infinity = f32s::splat(diskann_wide::ARCH, f32::NEG_INFINITY);

                // Generate separate masks for the lower 8 lanes and the upper 8 lanes.
                let lo = remainder.min(f32s::LANES);
                let hi = remainder - lo;

                let mask_lo = <f32s as SIMDVector>::Mask::keep_first(diskann_wide::ARCH, lo);
                d0_0 = mask_lo.select(d0_0, infinity);
                d1_0 = mask_lo.select(d1_0, infinity);
                d2_0 = mask_lo.select(d2_0, infinity);
                d3_0 = mask_lo.select(d3_0, infinity);

                let mask_hi = <f32s as SIMDVector>::Mask::keep_first(diskann_wide::ARCH, hi);
                d0_1 = mask_hi.select(d0_1, infinity);
                d1_1 = mask_hi.select(d1_1, infinity);
                d2_1 = mask_hi.select(d2_1, infinity);
                d3_1 = mask_hi.select(d3_1, infinity);

                // SAFETY: We've checked that the remainder is not zero.
                unsafe { self.load_remainder_norms() }
            } else {
                // SAFETY: From the enclosing conditional, we know that
                // * If `remainder != 0`, then `i + 1 != self.num_blocks()` implies
                //   `i < self.full_blocks()`, satisfying the requirements of
                //   `unsafe_load_norms`.
                // * If `remainder == 0`, then `self.num_blocks() == self.full_blocks()`,
                //   and thus `i < self.full_blocks()`.
                unsafe { self.load_norms(i) }
            };

            // Now that we have computed inner products, we need to add in query norms to
            // resolve the proxy for distance.
            let two = f32s::splat(diskann_wide::ARCH, 2.0f32);

            d0_0 = norm_0 - two * d0_0;
            d0_1 = norm_1 - two * d0_1;

            d1_0 = norm_0 - two * d1_0;
            d1_1 = norm_1 - two * d1_1;

            d2_0 = norm_0 - two * d2_0;
            d2_1 = norm_1 - two * d2_1;

            d3_0 = norm_0 - two * d3_0;
            d3_1 = norm_1 - two * d3_1;

            // The indices corresponding to these entries.
            let ind_0 =
                u32s::splat(diskann_wide::ARCH, (Self::groupsize() * i) as u32) + index_offsets;
            let ind_1 = u32s::splat(
                diskann_wide::ARCH,
                (Self::groupsize() * i + f32s::LANES) as u32,
            ) + index_offsets;

            // Combine with the global best results so far.
            tracking[0] =
                update_tracking(tracking[0], update_tracking((d0_0, ind_0), (d0_1, ind_1)));

            tracking[1] =
                update_tracking(tracking[1], update_tracking((d1_0, ind_0), (d1_1, ind_1)));

            tracking[2] =
                update_tracking(tracking[2], update_tracking((d2_0, ind_0), (d2_1, ind_1)));

            tracking[3] =
                update_tracking(tracking[3], update_tracking((d3_0, ind_0), (d3_1, ind_1)));
        }

        // Perform a final reduction of the partial distances.
        let finish = |(distances, indices): (f32s, u32s)| {
            let mut minimum_distance = f32::INFINITY;
            let mut minimum_index = u32::MAX;
            for (&i, &d) in std::iter::zip(indices.to_array().iter(), distances.to_array().iter()) {
                if d < minimum_distance {
                    minimum_distance = d;
                    minimum_index = i;
                }
            }

            // Make sure we return a sentinel value if any flavor of infinity/NaN is observed.
            if minimum_distance.is_finite() {
                CompressionResult(minimum_index)
            } else {
                CompressionResult::err()
            }
        };

        [
            finish(tracking[0]),
            finish(tracking[1]),
            finish(tracking[2]),
            finish(tracking[3]),
        ]
    }

    /// Load the pivot norms for the full block `block`.
    ///
    /// # Safety
    ///
    /// * `block < self.full_blocks()`.
    #[inline(always)]
    unsafe fn load_norms(&self, block: usize) -> (f32s, f32s) {
        debug_assert!(block < self.full_blocks());
        // SAFETY: Assuming the pre-conditions hold, this struct's constructor ensures
        // that `self.square_norms.len() >= Chunk::groupsize() * self.full_blocks()`.
        //
        // Addition is in-bounds.
        let ptr = unsafe { self.square_norms.as_ptr().add(Chunk::groupsize() * block) };

        // SAFETY: From the above logic, we can read up to `Chunk::groupsize()` elements
        // from this pointer, which is what this is doing.
        unsafe {
            (
                f32s::load_simd(diskann_wide::ARCH, ptr),
                f32s::load_simd(diskann_wide::ARCH, ptr.add(f32s::LANES)),
            )
        }
    }

    /// Load the pivot norms for the remainder block.
    ///
    /// # Safety
    ///
    /// The remainder must be non-zero.
    #[inline(always)]
    unsafe fn load_remainder_norms(&self) -> (f32s, f32s) {
        let remainder = self.remainder();
        debug_assert!(remainder != 0);

        let first = remainder % f32s::LANES;
        // SAFETY: This structs constructor ensures that
        // ```
        // self.square_norms.len() == self.data.nrows()
        // ```
        // In particular, this means that if the remainder is nonzero, then
        // ```
        // self.data.nrows() ==
        //     self.data.full_blocks() * Chunk::groupsize() + self.data.remainder()
        // ```
        // Therefore, it is safe to access `self.data.remainder()` values beginning at
        // the offset `self.data.full_blocks() * Chunk::groupsize()`.
        let ptr = unsafe {
            self.square_norms
                .as_ptr()
                .add(Chunk::groupsize() * self.full_blocks())
        };
        if remainder < f32s::LANES {
            // SAFETY: Exactly `remainder` values are readable from `ptr`.
            //
            // In this case, `remainder < f32s::LANES`, so we only load the lower group
            // of f32s.
            unsafe {
                (
                    f32s::load_simd_first(diskann_wide::ARCH, ptr, first),
                    f32s::default(diskann_wide::ARCH),
                )
            }
        } else {
            // SAFETY: Exactly `remainder` values are readable from `ptr`.
            //
            // In this case, `remainder >= f32s::LANES`, so we can do a full load at
            // `ptr`, and a partial load at `ptr.add(f32s::LANES)`.
            unsafe {
                (
                    f32s::load_simd(diskann_wide::ARCH, ptr),
                    f32s::load_simd_first(diskann_wide::ARCH, ptr.add(f32s::LANES), first),
                )
            }
        }
    }

    fn compute_in_block<Op, T>(&self, x: &[T], block: usize) -> (f32s, f32s)
    where
        Op: ComputeKernel,
        T: Copy + Into<f32>,
    {
        // Check 1
        assert_eq!(x.len(), self.dimension());
        // Check 2
        assert!(block < self.data.num_blocks());

        // SAFETY: Check 2 verifies that `block < self.data.num_blocks()`.
        let ptr = unsafe { self.data.block_ptr_unchecked(block) };

        let acc = (
            f32s::default(diskann_wide::ARCH),
            f32s::default(diskann_wide::ARCH),
        );
        x.iter().enumerate().fold(acc, |acc, (i, x)| {
            // SAFETY: Check 2 ensures that we have a valid block, which contains of
            // `Chunk::groupsize() * self.dimensions()` total elements.
            //
            // Check 1 and the loop bounds ensure that `i < self.dimensions()`.
            let a0 =
                unsafe { f32s::load_simd(diskann_wide::ARCH, ptr.add(Chunk::groupsize() * i)) };

            // SAFETY: From the above, up to 16 elements are readable. The previous line
            // read the first 8. This reads the next 8.
            let a1 = unsafe {
                f32s::load_simd(
                    diskann_wide::ARCH,
                    ptr.add(Chunk::groupsize() * i + f32s::LANES),
                )
            };
            Op::step(*x, (a0, a1), acc)
        })
    }

    // Epliogue handling when the total number of centers is not evenly divided by the
    // group size.
    //
    // In this context, we blend in `infinity` to the upper lanes so they do not get
    // selected.
    fn compute_in_remainder<Op, T>(&self, x: &[T]) -> (f32s, f32s)
    where
        Op: ComputeKernel,
        T: Copy + Into<f32>,
    {
        let d = self.compute_in_block::<Op, T>(x, self.data.full_blocks());
        let remainder = self.remainder();
        let keep = <f32s as SIMDVector>::Mask::keep_first(diskann_wide::ARCH, remainder % 8);
        let padding = f32s::splat(diskann_wide::ARCH, Op::REMAINDER);
        if remainder < f32s::LANES {
            (keep.select(d.0, padding), padding)
        } else {
            (d.0, keep.select(d.1, padding))
        }
    }
}

/// Efficiently perform the following operation outlined in pseudo code.
/// ```ignore
/// for i in 0..min_distances.num_lanes() {
///     if distances[i] < min_distances[i] {
///         min_distances[i] = distances[i];
///         min_indices[i] = base_index + index_offsets[i];
///     }
/// }
/// (min_distances, min_indices)
/// ```
///
/// In other words, update the contents of `min_distances` to contain the element-wise
/// minimum between `min_distances` and `distances`.
///
/// If an entry in `min_indices` is updates, also update the corresponding entry in
/// `min_indices` adding the corresponding offset to the base index.
///
/// THe updated variables are returned as a tuple.
#[inline(always)]
fn update_tracking_with(
    min_distances: f32s,
    min_indices: u32s,
    distances: f32s,
    base_index: u32,
    index_offsets: u32s,
) -> (f32s, u32s) {
    // We need to perform a bit of arithmetic to get the correct index numbers
    update_tracking(
        (min_distances, min_indices),
        (
            distances,
            u32s::splat(diskann_wide::ARCH, base_index) + index_offsets,
        ),
    )
}

/// Efficiently perform the following operation outlined in pseudo code:
/// ```ignore
/// for i in 0..d0.num_lanes() {
///     if d1[i] < d0[i] {
///         d0[i] = d1[i];
///         i0[i] = i1[i];
///     }
/// }
/// (d0, i0)
/// ```
/// In other words, update `d0` to contain the element-wise minimmum of `d0` and `d1` and
/// move the values from `i1` in the updated lanes into `i0`.
///
/// Lanes that evaluate to equal will not be updated.
#[inline(always)]
fn update_tracking((d0, i0): (f32s, u32s), (d1, i1): (f32s, u32s)) -> (f32s, u32s) {
    // Generate a mask with lanes set if a computed distance is less that one of the
    // current minimum distances.
    let mask = d1.lt_simd(d0);
    (
        mask.select(d1, d0),
        <u32s as SIMDVector>::Mask::from(mask).select(i1, i0),
    )
}

/// Helper trait for defining operations in `compute_in_block` and `compute_in_remainder`.
trait ComputeKernel {
    /// The value to use in masked out lanes when processing the remainder block.
    const REMAINDER: f32;

    fn step<T>(x: T, y: (f32s, f32s), accumulator: (f32s, f32s)) -> (f32s, f32s)
    where
        T: Into<f32>;
}

impl ComputeKernel for SquaredL2 {
    const REMAINDER: f32 = f32::INFINITY;

    #[inline(always)]
    fn step<T>(x: T, y: (f32s, f32s), accumulator: (f32s, f32s)) -> (f32s, f32s)
    where
        T: Into<f32>,
    {
        let b = f32s::splat(diskann_wide::ARCH, x.into());
        let d0 = y.0 - b;
        let d1 = y.1 - b;
        (
            d0.mul_add_simd(d0, accumulator.0),
            d1.mul_add_simd(d1, accumulator.1),
        )
    }
}

impl ComputeKernel for InnerProduct {
    const REMAINDER: f32 = f32::INFINITY;

    #[inline(always)]
    fn step<T>(x: T, y: (f32s, f32s), accumulator: (f32s, f32s)) -> (f32s, f32s)
    where
        T: Into<f32>,
    {
        let x: f32 = x.into();
        let b = f32s::splat(diskann_wide::ARCH, -x);
        (
            b.mul_add_simd(y.0, accumulator.0),
            b.mul_add_simd(y.1, accumulator.1),
        )
    }
}

/// An internal helper like `InnerProduct` but returns `f32::NEG_INFINITY` in masked out
/// remainder lanes instead of `f32::INFINITY` and does not negate the result.
struct InnerProductMathematical;
impl ComputeKernel for InnerProductMathematical {
    const REMAINDER: f32 = f32::NEG_INFINITY;

    #[inline(always)]
    fn step<T>(x: T, y: (f32s, f32s), accumulator: (f32s, f32s)) -> (f32s, f32s)
    where
        T: Into<f32>,
    {
        let x: f32 = x.into();
        let b = f32s::splat(diskann_wide::ARCH, x);
        (
            b.mul_add_simd(y.0, accumulator.0),
            b.mul_add_simd(y.1, accumulator.1),
        )
    }
}

//////////////////////
// Block Algorithms //
//////////////////////

/// Apply an operation on the argument `from` and all pivots within `Chunk` and store
/// the results into `into`. The result for the pivot 0 is stored in `into[0]`, the result
/// for pivot 1 is stored in `into[1]` etc.
///
/// In particular, this is used to compute squared euclidean and inner products between
/// `from` and all pivots.
///
/// Example types implementing this trait include:
/// * [`SwuaredL2`]: Compute the squared l2 distance between `from` and all pivots.
/// * [`InnerProduct`]: Compute the inner product (as a [`diskann_vector::SimilarityScore`])
///   between `from` and all pivots.
pub trait ProcessInto {
    /// Do the specified operation.
    ///
    /// # Panics
    ///
    /// Panics under the following conditions:
    /// * `from.len() != chunk.dimension()`: The argument `from` must have the same number
    ///   of dimensions as the pivots stored in `chunk`.
    /// * `into.len() != chunk.num_centers()`: This routine will produce one result per
    ///   pivot and `into` must be sized accordingly.
    fn process_into(chunk: &Chunk, from: &[f32], into: &mut [f32]);
}

impl<T> ProcessInto for T
where
    T: ComputeKernel,
{
    fn process_into(chunk: &Chunk, from: &[f32], into: &mut [f32]) {
        assert_eq!(from.len(), chunk.dimension());

        // Check 1 (used for SAFETY arguments)
        assert_eq!(into.len(), chunk.num_centers());

        // Identity 1 (this is used for SAFETY arguments)
        //
        // Chunk maintains the following invariant:
        // ```
        // chunk.num_centers() == Chunk::groupsize() * chunk.full_blocks() + chunk.remainder()
        // ```
        //
        // Identity 2 (this is used for SAFETY arguments)
        //
        // Chunk maintains the following invariant:
        // ```
        // chunk.num_blocks() - chunk.full_blocks() == 1 if chunk.remainder() != 0
        // chunk.num_blocks() == chunk.full_blocks()     if chunk.remainder() == 0
        // ```

        let ptr = into.as_mut_ptr();
        let full_blocks = chunk.full_blocks();
        let remainder = chunk.remainder();
        for block in 0..chunk.num_blocks() {
            let (lo, hi) = chunk.compute_in_block::<T, f32>(from, block);

            if remainder != 0 && block == full_blocks {
                let keep_lo = remainder.min(f32s::LANES);
                let keep_hi = remainder - keep_lo;

                // SAFETY: Safety basically follows from Check 1, identities 1 and 2, the
                // bounds on the loop, and the enclosing conditional.
                unsafe { lo.store_simd_first(ptr.add(Chunk::groupsize() * full_blocks), keep_lo) };
                if keep_hi != 0 {
                    // SAFETY: Safety basically follows from Check 1, identities 1 and 2,
                    // the bounds on the loop, and the enclosing conditional.
                    unsafe {
                        hi.store_simd_first(
                            ptr.add(Chunk::groupsize() * full_blocks + f32s::LANES),
                            keep_hi,
                        )
                    };
                }
            } else {
                // SAFETY: Safety basically follows from Check 1, identities 1 and 2, the
                // bounds on the loop, and the enclosing conditional.
                unsafe {
                    lo.store_simd(ptr.add(Chunk::groupsize() * block));
                    hi.store_simd(ptr.add(Chunk::groupsize() * block + f32s::LANES));
                }
            }
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, views};
    use diskann_vector::{distance, PureDistanceFunction};
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        SeedableRng,
    };

    use super::*;

    ///////////////////////
    // CompressionResult //
    ///////////////////////

    #[test]
    fn compression_result() {
        let v = CompressionResult::err();
        assert_eq!(v.into_inner(), u32::MAX);
        assert!(!v.is_okay());

        for i in 0u32..1000u32 {
            let v = CompressionResult(i);
            assert!(v.is_okay());
            assert_eq!(v.unwrap(), i);
            assert_eq!(v.into_inner(), i);
        }

        // `map` only invokes one of its closures - "ok" branch.
        {
            let mut called_ok = false;
            let mut called_err = false;
            let x: Result<&str, &str> = CompressionResult(10).map(
                |v| {
                    called_ok = true;
                    assert_eq!(v, 10);
                    "okay!"
                },
                || {
                    called_err = true;
                    "not okay!"
                },
            );
            assert_eq!(x.unwrap(), "okay!");
            assert!(called_ok);
            assert!(!called_err);
        }

        // `map` only invokes one of its closures - "err" branch.
        {
            let mut called_ok = false;
            let mut called_err = false;
            let x: Result<&str, &str> = CompressionResult::err().map(
                |_| {
                    called_ok = true;
                    "okay!"
                },
                || {
                    called_err = true;
                    "not okay!"
                },
            );
            assert_eq!(x.unwrap_err(), "not okay!");
            assert!(!called_ok);
            assert!(called_err);
        }
    }

    #[test]
    #[should_panic]
    fn compression_result_unwrap_panics() {
        CompressionResult::err().unwrap();
    }

    /////////////////
    // Compression //
    /////////////////

    fn flatten(x: &[Vec<f32>]) -> Vec<f32> {
        // Ensure the collection is non-empty and that all entries have the same dimensions.
        assert!(!x.is_empty());
        let dim = x[0].len();
        assert!(x.iter().all(|i| i.len() == dim));

        let mut output = Vec::new();
        x.iter().for_each(|i| {
            output.extend_from_slice(i.as_slice());
        });

        assert_eq!(output.len(), dim * x.len());
        output
    }

    /// Create a test pattern with the requested dimension and total.
    ///
    /// The resulting corpus looks like the following:
    /// ```ignore
    /// 0       1     2       ... dim-1
    /// 1       2     3       ... dim
    /// ...
    /// total-1 total total+1 ... total+dim-1
    /// ```
    fn create_test_pattern(dim: usize, total: usize) -> Vec<Vec<f32>> {
        (0..total)
            .map(|i| (0..dim).map(|j| (i + j) as f32).collect())
            .collect()
    }

    /// Test that each output of batch processing behaves as expected.
    fn test_batch(
        chunk: &Chunk,
        query: &[f32],
        expected_closest: usize,
        zero_matches: usize,
        test_context: &dyn std::fmt::Display,
    ) {
        let dim = query.len();

        // Batch query.
        // The strategy here is to set each lane the provided query independently to
        // ensure that every lane is independent.
        for j in 0..Chunk::batchsize() {
            let copy_query = |k| {
                if k == j {
                    query.to_vec()
                } else {
                    vec![0.0; dim]
                }
            };

            assert_eq!(
                4,
                Chunk::batchsize(),
                "if the lower level batch size changes, update the function below"
            );
            let query_batch =
                flatten(&[copy_query(0), copy_query(1), copy_query(2), copy_query(3)]);

            let view = strided::StridedView::try_from(
                query_batch.as_slice(),
                Chunk::batchsize(),
                dim,
                dim,
            )
            .unwrap();

            // Make sure that the query batch was constructed correctly.
            assert_eq!(view.nrows(), Chunk::batchsize());
            assert_eq!(view.ncols(), dim);
            for k in 0..view.nrows() {
                let row = view.row(k);
                if k == j {
                    assert_eq!(
                        row, query,
                        "expected entry {k} to be the query ({test_context})"
                    );
                } else {
                    assert!(
                        row.iter().all(|&k| k == 0.0),
                        "expected inactive rows to be zero ({test_context})"
                    );
                }
            }

            // Now that we have ensured the query batch has been constructed correctly,
            // launch the query and make sure the result is as expected.
            let closest: [CompressionResult; Chunk::batchsize()] = chunk.find_closest_batch(view);
            for (k, &got) in closest.iter().enumerate() {
                let got = got.unwrap() as usize;
                if k == j {
                    // The active lane should match the expected lane.
                    assert_eq!(
                        got, expected_closest,
                        "failed to match active lane {k} ({test_context})"
                    );
                } else {
                    // Inactive queries should match the first entry.
                    assert_eq!(
                        got, zero_matches,
                        "inactive lane {k} assigned to the wrong center ({test_context})"
                    );
                }
            }

            // Check that we correctly handle invalid configurations.
            let maybe_broadcast = |k, v: f32| {
                if k == j {
                    vec![v; dim]
                } else {
                    query.to_vec()
                }
            };

            // Don't loop over the pathological values because that makes the test run way
            // too long.
            //
            // Instead, assign different cases to the various lanes to achieve coverage.
            let values = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, f32::INFINITY];
            let query_batch = flatten(&[
                maybe_broadcast(0, values[0]),
                maybe_broadcast(1, values[1]),
                maybe_broadcast(2, values[2]),
                maybe_broadcast(3, values[3]),
            ]);

            let view = strided::StridedView::try_from(
                query_batch.as_slice(),
                Chunk::batchsize(),
                dim,
                dim,
            )
            .unwrap();

            let closest = chunk.find_closest_batch(view);
            // Lane `j` should not be okay. All other lanes should return the correct
            // expected closest value.
            for (k, &got) in closest.iter().enumerate() {
                if k == j {
                    assert!(
                        !got.is_okay(),
                        "lane {} should not be okay with value {} ({})",
                        k,
                        values[k],
                        test_context
                    );
                } else {
                    assert_eq!(
                        got.unwrap() as usize,
                        expected_closest,
                        "failed to match active lane {k} ({test_context})"
                    );
                }
            }
        }
    }

    fn test_chunk(dim: usize, total: usize) {
        let test_context = lazy_format!("ndims {}, total {}", dim, total);

        // Initialize the chunk data.
        let mut data_aggregate = create_test_pattern(dim, total);

        let data = flatten(&data_aggregate);
        let sliced = strided::StridedView::try_from(data.as_slice(), total, dim, dim).unwrap();
        let chunk = Chunk::new(sliced).unwrap();

        assert_eq!(chunk.num_centers(), total);
        assert_eq!(chunk.dimension(), dim);

        // Check that indexing works properly.
        for row in 0..sliced.nrows() {
            for col in 0..sliced.ncols() {
                assert_eq!(
                    sliced[(row, col)],
                    chunk.get(row, col),
                    "failed on row {} and col {}",
                    row,
                    col,
                );
            }
        }

        // Sanity check the various error conditions.
        assert!(!chunk.find_closest(&vec![f32::NEG_INFINITY; dim]).is_okay());
        assert!(!chunk.find_closest(&vec![f32::INFINITY; dim]).is_okay());
        assert!(!chunk.find_closest(&vec![f32::NAN; dim]).is_okay());

        // N.B.: We need to make sure that ragged group numbers are handled properly by
        // the implementation of `find_closest`.
        //
        // We can test for this by providing an `all-zero` query which will match the
        // remainder elements in the Chunk (provided the total number of blocks is 1).
        let query: Vec<f32> = vec![0.0; dim];
        assert_eq!(chunk.find_closest(&query).unwrap(), 0);

        // Make sure we can match with every group and that the computed distance is
        // correct.
        for i in 0..total {
            // Single query
            let query: Vec<f32> = (0..dim).map(|j| ((i + j) as f32) + 0.125).collect();
            let closest = chunk.find_closest(&query).unwrap();
            assert_eq!(closest as usize, i);

            test_batch(
                &chunk,
                query.as_slice(),
                i,
                0,
                &lazy_format!("main iteration {}, {}", i, test_context),
            );

            // Safety critical control flow within `Chunk` is not dependent on the data,
            // only on the dimensions.
            //
            // After executing this once, we've tested all the bounding logic on the for
            // loops and can thus exit early when running under `miri`.
            if cfg!(miri) {
                return;
            }
        }

        // Due to the way packing works - ragged remainders will be filled with a constant.
        // If the query is one of these constants, then it will match one of the
        // out-of-bounds values without the special handling logic.
        //
        // Skip this test under `miri`.
        for i in total..=(total + dim) {
            let query: Vec<f32> = vec![i as f32; dim];
            let closest = chunk.find_closest(&query).unwrap();
            assert!((closest as usize) < chunk.num_centers());

            test_batch(
                &chunk,
                query.as_slice(),
                closest as usize,
                0,
                &lazy_format!("tail matching {}, {}", i, test_context),
            );
        }

        // Finally, ensure that ties are resolved to the lowset value.
        //
        // Do this by copying the last element to the first element and using the same vector
        // as the query.
        let last = data_aggregate.last().unwrap().clone();
        data_aggregate[0].clone_from(&last);
        let data = flatten(&data_aggregate);
        let sliced = strided::StridedView::try_from(data.as_slice(), total, dim, dim).unwrap();
        let chunk = Chunk::new(sliced).unwrap();

        assert_eq!(chunk.num_centers(), total);
        assert_eq!(chunk.dimension(), dim);
        assert_eq!(
            chunk.find_closest(&last).unwrap(),
            0,
            "ties must resolve to lower index, {}",
            test_context
        );

        // Repeat for batch processing.
        // Since we over-wrote the first chunk, we expect all zeros to match chunk
        // number 1 (unless there is only one chunk or two total chunks, then we expect it
        // to match the only available center).
        let zero_matches = if chunk.num_centers() <= 2 { 0 } else { 1 };

        test_batch(
            &chunk,
            &last,
            0,
            zero_matches,
            &lazy_format!("ties resolve to first, {}", test_context),
        );
    }

    // Test routines for which we expect nothing to really go wrong.
    #[test]
    fn run_test_happy_path() {
        // Step dimensions by 1 to test all possible residual combinations.
        let dims: Vec<usize> = if cfg!(miri) {
            (1..=8).collect()
        } else {
            (1..=16).collect()
        };

        // Test critical regions of totals:
        let totals: Vec<usize> = if cfg!(miri) {
            // When running with `miri`, we need to be more selective about our totals.
            vec![1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33]
        } else {
            [
                (1..=17),    // Low total numbers
                (64..=103),  // medium numbers
                (255..=257), // high numbers (exceeding our current maximum of 256).
            ]
            .into_iter()
            .flatten()
            .collect()
        };

        for &total in totals.iter() {
            for &dim in dims.iter() {
                println!("on {}, {}", dim, total);
                test_chunk(dim, total);
            }
        }
    }

    // Test error paths.
    #[test]
    fn test_chunk_construction_error() {
        // No dimensions
        let chunk = Chunk::new(strided::StridedView::try_from(&[], 3, 0, 0).unwrap());
        let err = chunk.unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot construct a Chunk from a source with zero dimensions"));

        // No length
        let chunk = Chunk::new(strided::StridedView::try_from(&[], 0, 10, 10).unwrap());
        let err = chunk.unwrap_err();
        assert!(err
            .to_string()
            .contains("cannot construct a Chunk from a source with zero length"));
    }

    // Make sure `find_closest` panics for an incorrect dimension.
    #[test]
    #[should_panic(expected = "incorrect query dimension")]
    fn test_find_closest_panics() {
        let dim = 10;
        let total = 13;
        let data = flatten(&create_test_pattern(dim, total));
        let sliced = strided::StridedView::try_from(data.as_slice(), total, dim, dim).unwrap();
        let chunk = Chunk::new(sliced).unwrap();

        let query: Vec<f32> = vec![0.0; total];

        // PANICS
        chunk.find_closest(query.as_slice());
    }

    // Make sure `find_closest_batch` panics for an incorrect dimension.
    #[test]
    #[should_panic(expected = "incorrect query dimension")]
    fn test_find_closest_batch_panics_on_dim_mismatch() {
        let dim = 10;
        let total = 13;
        let data = flatten(&create_test_pattern(dim, total));
        let sliced = strided::StridedView::try_from(data.as_slice(), total, dim, dim).unwrap();
        let chunk = Chunk::new(sliced).unwrap();

        let query: Vec<f32> = vec![0.0; 4 * total];
        let query_view =
            strided::StridedView::try_from(query.as_slice(), Chunk::batchsize(), total, total)
                .unwrap();

        // PANICS
        chunk.find_closest_batch(query_view);
    }

    // Make sure `find_closest_batch` panics for an incorrect length
    #[test]
    #[should_panic(expected = "argument StridedView must have a length of")]
    fn test_find_closest_batch_panics_on_non_batch_length() {
        let dim = 10;
        let total = 13;
        let data = flatten(&create_test_pattern(dim, total));
        let sliced = strided::StridedView::try_from(data.as_slice(), total, dim, dim).unwrap();
        let chunk = Chunk::new(sliced).unwrap();

        let query: Vec<f32> = vec![0.0; (Chunk::batchsize() + 1) * dim];
        let query_view =
            strided::StridedView::try_from(query.as_slice(), Chunk::batchsize() + 1, dim, dim)
                .unwrap();

        // PANICS
        chunk.find_closest_batch(query_view);
    }

    #[test]
    #[should_panic(expected = "row 5 must be less than 5")]
    fn get_panics_on_row() {
        let data = views::Matrix::new(0.0, 5, 10);
        let chunk = Chunk::new(data.as_view().into()).unwrap();
        chunk.get(5, 1);
    }

    #[test]
    #[should_panic(expected = "col 5 must be less than 5")]
    fn get_panics_on_col() {
        let data = views::Matrix::new(0.0, 10, 5);
        let chunk = Chunk::new(data.as_view().into()).unwrap();
        chunk.get(1, 5);
    }

    //////////////////
    // Process Into //
    //////////////////

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const PROCESS_INTO_TRIALS: usize = 1;
        } else {
            const PROCESS_INTO_TRIALS: usize = 10;
        }
    }

    fn test_process_into_impl(dim: usize, total: usize, rng: &mut StdRng) {
        let distribution = Uniform::<i32>::new(-10, 10).unwrap();
        let base =
            views::Matrix::<f32>::new(views::Init(|| distribution.sample(rng) as f32), total, dim);

        let chunk = Chunk::new(base.as_view().into()).unwrap();
        let mut input = vec![0.0; dim];
        let mut output = vec![0.0; total];

        for _ in 0..PROCESS_INTO_TRIALS {
            input
                .iter_mut()
                .for_each(|i| *i = distribution.sample(rng) as f32);

            // Inner Product
            InnerProduct::process_into(&chunk, &input, &mut output);

            // Check outputs
            std::iter::zip(base.row_iter(), output.iter()).for_each(|(row, got)| {
                let expected: f32 = distance::InnerProduct::evaluate(row, input.as_slice());
                assert_eq!(*got, expected);
            });

            // Squared L2
            SquaredL2::process_into(&chunk, &input, &mut output);

            // Check outputs
            std::iter::zip(base.row_iter(), output.iter()).for_each(|(row, got)| {
                let expected: f32 = distance::SquaredL2::evaluate(row, input.as_slice());
                assert_eq!(*got, expected);
            });
        }
    }

    #[test]
    fn test_process_into() {
        let mut rng = StdRng::seed_from_u64(0x21dfb5f35dfe5639);
        for total in 1..64 {
            for dim in 1..5 {
                println!("on ({}, {})", total, dim);
                test_process_into_impl(dim, total, &mut rng);
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_process_into_panics_on_from() {
        let data = views::Matrix::<f32>::new(0.0, 5, 10);
        let chunk = Chunk::new(data.as_view().into()).unwrap();
        assert_eq!(chunk.dimension(), 10);
        assert_eq!(chunk.num_centers(), 5);

        // Query is too large.
        let query: Vec<f32> = vec![0.0; chunk.dimension() + 1];
        let mut dst = vec![0.0; chunk.num_centers()];
        InnerProduct::process_into(&chunk, query.as_slice(), dst.as_mut_slice());
    }

    #[test]
    #[should_panic]
    fn test_process_into_panics_on_into() {
        let data = views::Matrix::<f32>::new(0.0, 5, 10);
        let chunk = Chunk::new(data.as_view().into()).unwrap();
        assert_eq!(chunk.dimension(), 10);
        assert_eq!(chunk.num_centers(), 5);

        let query: Vec<f32> = vec![0.0; chunk.dimension()];
        // Dst is too big.
        let mut dst = vec![0.0; chunk.num_centers() + 1];
        InnerProduct::process_into(&chunk, query.as_slice(), dst.as_mut_slice());
    }
}
