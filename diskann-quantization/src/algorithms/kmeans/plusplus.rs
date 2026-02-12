/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashSet, fmt};

use diskann_utils::{
    strided::StridedView,
    views::{MatrixView, MutMatrixView},
};
use diskann_wide::{SIMDMulAdd, SIMDPartialOrd, SIMDSelect, SIMDVector};
use rand::{
    distr::{Distribution, Uniform},
    RngCore,
};
use thiserror::Error;

use super::common::{square_norm, BlockTranspose};

/// An internal trait implemented for `BlockTranspose` used to accelerate
///
/// 1. Computation of distances between a newly selected kmeans++ center and all elements
///    in the dataset.
///
/// 2. Updating of the minimum distance between the dataset and all candidates.
///
/// 3. Computing the sum of the squares of all the minimum distances in the dataset so far.
///
/// All of these operations are fused for efficiency.
///
/// This trait is only meant to be implemented by `BlockTranspose`.
pub(crate) trait MicroKernel {
    /// The intermediate value storing inner products.
    type Intermediate;

    /// The type of the rolling sum used to accumulate the sum of all the squared minimum
    /// distances.
    type RollingSum: Default + Copy;

    /// A potentially broadcasted representation of a `f32` number.
    type Splat: Copy;

    /// Return the broadcasted representation of `x`.
    fn splat(x: f32) -> Self::Splat;

    /// Compute the distances between `this` and all the vectors stored in `block`.
    ///
    /// This implementation works for both full blocks and partial blocks.
    ///
    /// # SAFETY
    ///
    /// `block` must be the base pointer of a data block in a `BlockTranspose` and the
    /// block size of this block must have the same length as `this`.
    unsafe fn accum_full(block: *const f32, this: &[f32]) -> Self::Intermediate;

    /// Accumulate intermediate distances and store the result in `mins`.
    fn finish(
        intermediate: Self::Intermediate,
        splat: Self::Splat,
        rolling_sum: Self::RollingSum,
        norms: &[f32],
        mins: &mut [f32],
    ) -> Self::RollingSum;

    /// Accumulate the first `first` intermediate distances and store the result in `mins`.
    fn finish_last(
        intermediate: Self::Intermediate,
        splat: Self::Splat,
        rolling_sum: Self::RollingSum,
        norms: &[f32],
        mins: &mut [f32],
        first: usize,
    ) -> Self::RollingSum;

    /// Turn the rolling sum from its internal representation to a final `f32`.
    fn complete_sum(x: Self::RollingSum) -> f64;
}

diskann_wide::alias!(f32s = f32x8);

impl MicroKernel for BlockTranspose<16> {
    // Process 16-dimensions concurrently, split across two `Wide`s.
    type Intermediate = (f32s, f32s);
    type RollingSum = f64;
    type Splat = f32s;

    fn splat(x: f32) -> Self::Splat {
        Self::Splat::splat(diskann_wide::ARCH, x)
    }

    #[inline(always)]
    unsafe fn accum_full(block_ptr: *const f32, this: &[f32]) -> Self::Intermediate {
        let mut s0 = f32s::default(diskann_wide::ARCH);
        let mut s1 = f32s::default(diskann_wide::ARCH);

        this.iter().enumerate().for_each(|(i, b)| {
            let b = f32s::splat(diskann_wide::ARCH, *b);

            // SAFETY: From the requirement that `self.block_size() == this.len()`, then
            // `this.len() * 16` elements are readible from `block_ptr` and `i < this.len()`.
            let a = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(16 * i)) };
            s0 = a.mul_add_simd(b, s0);

            // SAFETY: From the requirement that `self.block_size() == this.len()`, then
            // `this.len() * 16` elements are readible from `block_ptr` and `i < this.len()`.
            let a = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(16 * i + 8)) };
            s1 = a.mul_add_simd(b, s1);
        });

        // Apply the final -2.0 transformation.
        let negative = f32s::splat(diskann_wide::ARCH, -2.0);
        (s0 * negative, s1 * negative)
    }

    #[inline(always)]
    fn finish(
        intermediate: Self::Intermediate,
        splat: Self::Splat,
        rolling_sum: Self::RollingSum,
        norms: &[f32],
        mins: &mut [f32],
    ) -> Self::RollingSum {
        assert_eq!(norms.len(), 16);
        assert_eq!(mins.len(), 16);

        // SAFETY: `norms` has length 16 - this loads the first 8.
        let norms0 = unsafe { f32s::load_simd(diskann_wide::ARCH, norms.as_ptr()) };
        // SAFETY: `norms` has length 16 - this loads the last 8.
        let norms1 = unsafe { f32s::load_simd(diskann_wide::ARCH, norms.as_ptr().add(8)) };

        let distances0 = norms0 + splat + intermediate.0;
        let distances1 = norms1 + splat + intermediate.1;

        // SAFETY: `mins` has length 16 - this loads the first 8.
        let current_distances0 = unsafe { f32s::load_simd(diskann_wide::ARCH, mins.as_ptr()) };
        // SAFETY: `mins` has length 16 - this loads the last 8.
        let current_distances1 =
            unsafe { f32s::load_simd(diskann_wide::ARCH, mins.as_ptr().add(8)) };

        let mask0 = distances0.lt_simd(current_distances0);
        let mask1 = distances1.lt_simd(current_distances1);

        let current_distances0 = mask0.select(distances0, current_distances0);
        let current_distances1 = mask1.select(distances1, current_distances1);

        // SAFETY: `mins` has length 16 - this stores the first 8.
        unsafe { current_distances0.store_simd(mins.as_mut_ptr()) };
        // SAFETY: `mins` has length 16 - this stores the last 8.
        unsafe { current_distances1.store_simd(mins.as_mut_ptr().add(8)) };

        rolling_sum
            + std::iter::zip(
                current_distances0.to_array().iter(),
                current_distances1.to_array().iter(),
            )
            .map(|(d0, d1)| (*d0 as f64) + (*d1 as f64))
            .sum::<f64>()
    }

    #[inline(always)]
    fn finish_last(
        intermediate: Self::Intermediate,
        splat: Self::Splat,
        rolling_sum: Self::RollingSum,
        norms: &[f32],
        mins: &mut [f32],
        first: usize,
    ) -> Self::RollingSum {
        // Check 1.
        assert_eq!(norms.len(), first);
        // Check 2.
        assert_eq!(mins.len(), first);

        let lo = first.min(8);
        let hi = first - lo;

        // SAFETY: This loads `first.min(8)` elements from `norms`, which is valid
        // by check 1.
        let norms0 = unsafe { f32s::load_simd_first(diskann_wide::ARCH, norms.as_ptr(), lo) };
        let norms1 = if hi == 0 {
            f32s::default(diskann_wide::ARCH)
        } else {
            // SAFETY: This is only called if `first > 8`, which means `norms.len() > 8` by
            // check 1. Therefore, we can load `first - 8` elements from `norms.as_ptr() + 8`.
            unsafe { f32s::load_simd_first(diskann_wide::ARCH, norms.as_ptr().add(8), hi) }
        };

        let distances0 = norms0 + splat + intermediate.0;
        let distances1 = norms1 + splat + intermediate.1;

        // SAFETY: Same logic as the load for `norms0`.
        let current_distances0 =
            unsafe { f32s::load_simd_first(diskann_wide::ARCH, mins.as_ptr(), lo) };
        let current_distances1 = if hi == 0 {
            f32s::default(diskann_wide::ARCH)
        } else {
            // SAFETY: Same logic as the load for `norms1`.
            unsafe { f32s::load_simd_first(diskann_wide::ARCH, mins.as_ptr().add(8), hi) }
        };

        let mask0 = distances0.lt_simd(current_distances0);
        let mask1 = distances1.lt_simd(current_distances1);

        let current_distances0 = mask0.select(distances0, current_distances0);
        let current_distances1 = mask1.select(distances1, current_distances1);

        // SAFETY: As per the logic for the load above, it is safe to store at least `lo`
        // elements from the base pointer.
        unsafe { current_distances0.store_simd_first(mins.as_mut_ptr(), lo) };
        if hi != 0 {
            // SAFETY: If `hi != 0`, then `first` must be at least 9. Therefore, adding 8
            // to the base pointer is valid, as is storing `first - 8` elements to that
            // pointer.
            unsafe { current_distances1.store_simd_first(mins.as_mut_ptr().add(8), hi) };
        }

        rolling_sum
            + std::iter::zip(
                current_distances0.to_array().iter(),
                current_distances1.to_array().iter(),
            )
            .map(|(d0, d1)| (*d0 as f64) + (*d1 as f64))
            .sum::<f64>()
    }

    fn complete_sum(x: Self::RollingSum) -> f64 {
        x
    }
}

/// Update `square_distances` to contain the minimum of its current value and the distance
/// between each element in `transpose` and `this`.
///
/// Return the sum of the new `square_distances`.
fn update_distances<const N: usize>(
    square_distances: &mut [f32],
    transpose: &BlockTranspose<N>,
    norms: &[f32],
    this: &[f32],
    this_square_norm: f32,
) -> f64
where
    BlockTranspose<N>: MicroKernel,
{
    // Establish our safety requirements.
    // Check 1.
    assert_eq!(
        this.len(),
        transpose.ncols(),
        "new point and dataset must have the same dimension",
    );
    // Check 2.
    assert_eq!(
        square_distances.len(),
        transpose.nrows(),
        "distances buffer and dataset must have the same length",
    );
    // Check 3.
    assert_eq!(
        norms.len(),
        transpose.nrows(),
        "norms and dataset must have the same length",
    );

    let splat = BlockTranspose::<N>::splat(this_square_norm);
    let mut rolling_sum = <BlockTranspose<N> as MicroKernel>::RollingSum::default();

    let iter =
        std::iter::zip(norms.chunks_exact(N), square_distances.chunks_exact_mut(N)).enumerate();
    iter.for_each(|(block, (these_norms, these_distances))| {
        debug_assert!(block < transpose.num_blocks());
        // SAFETY: Because `transpose.nrows() == norms.len()`, the number of full blocks in
        // `transpose` is `norms.nrows() / N` and therefore the induction variable `block`
        // is less that `transpose.full_blocks()`.
        let base = unsafe { transpose.block_ptr_unchecked(block) };

        // SAFETY: The pointer `base` does point to a full block and by Check 1,
        // `transpose.ncols() == this.len()`.
        let intermediate = unsafe { BlockTranspose::<N>::accum_full(base, this) };

        rolling_sum = BlockTranspose::<N>::finish(
            intermediate,
            splat,
            rolling_sum,
            these_norms,
            these_distances,
        );
    });

    // Do the last iteration if there is an un-even number of rows.
    let remainder = transpose.remainder();
    if remainder != 0 {
        // SAFETY: We've checked that there is a `remainder` block. Therefore,
        // `transpose.full_blocks() < transpose.num_blocks()`.
        let base = unsafe { transpose.block_ptr_unchecked(transpose.full_blocks()) };

        // A full accumulation is fine because `BlockTranspose` allocates at the granularity
        // of blocks. We will just ignore the extra lanes.
        // SAFETY: The pointer `base` does point to a full block and by Check 1,
        // `transpose.ncols() == this.len()`.
        let intermediate = unsafe { BlockTranspose::<N>::accum_full(base, this) };

        let start = N * transpose.full_blocks();
        rolling_sum = BlockTranspose::<N>::finish_last(
            intermediate,
            splat,
            rolling_sum,
            &norms[start..],
            &mut square_distances[start..],
            remainder,
        );
    }

    BlockTranspose::<N>::complete_sum(rolling_sum)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FailureReason {
    /// This error happens when the dataset contains fewer than the requested number of
    /// centers.
    DatasetTooSmall,
    /// The dataset contains fewer than the requested number of points.
    InsufficientDiversity,
    /// Infinity was observed (this also happens when a NaN is present in the input data)
    SawInfinity,
}

impl FailureReason {
    pub fn is_numerically_recoverable(self) -> bool {
        match self {
            // Datasets being too small is recoverable from a `kmeans` perspective, we can
            // simply proceed with fewer points.
            Self::DatasetTooSmall | Self::InsufficientDiversity => true,

            // If we see Infinity, downstream algorithms will likely just break.
            // Don't expect to recover.
            Self::SawInfinity => false,
        }
    }
}

impl fmt::Display for FailureReason {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let reason: &str = match self {
            Self::DatasetTooSmall => "dataset does not have enough points",
            Self::InsufficientDiversity => "dataset is insufficiently diverse",
            Self::SawInfinity => "a value of infinity or NaN was observed",
        };
        f.write_str(reason)
    }
}

#[derive(Debug, Clone, Copy, Error)]
#[error("only populated {selected} of {expected} points because {reason}")]
pub struct KMeansPlusPlusError {
    /// The number of points that were selected.
    pub selected: usize,
    /// The number of points that were expected.
    pub expected: usize,
    /// A concrete reason for the failure.
    pub reason: FailureReason,
}

impl KMeansPlusPlusError {
    fn new(selected: usize, expected: usize, reason: FailureReason) -> Self {
        Self {
            selected,
            expected,
            reason,
        }
    }

    pub fn is_numerically_recoverable(&self) -> bool {
        self.reason.is_numerically_recoverable() && self.selected > 0
    }
}

pub(crate) fn kmeans_plusplus_into_inner<const N: usize>(
    mut points: MutMatrixView<'_, f32>,
    data: StridedView<'_, f32>,
    transpose: &BlockTranspose<N>,
    norms: &[f32],
    rng: &mut dyn RngCore,
) -> Result<(), KMeansPlusPlusError>
where
    BlockTranspose<N>: MicroKernel,
{
    assert_eq!(norms.len(), data.nrows());
    assert_eq!(transpose.nrows(), data.nrows());
    assert_eq!(transpose.block_size(), data.ncols());
    assert_eq!(points.ncols(), data.ncols());

    // Zero the argument
    points.as_mut_slice().fill(0.0);
    let expected = points.nrows();

    // Is someone trying to operate on an empty dataset?
    //
    // We can determine this by constructing a `Uniform` distribution over the rows and
    // checking if the resulting range is Empty.
    let all_rows = match Uniform::new(0, data.nrows()) {
        Ok(dist) => dist,
        Err(_) => {
            // If they want 0 points, then I guess this is okay.
            return if expected == 0 {
                Ok(())
            } else {
                Err(KMeansPlusPlusError::new(
                    0,
                    expected,
                    FailureReason::DatasetTooSmall,
                ))
            };
        }
    };

    let mut min_distances: Vec<f32> = vec![f32::INFINITY; data.nrows()];
    let mut picked = HashSet::with_capacity(expected);

    // Pick the first point randomly.
    let mut previous_square_norm = {
        let i = all_rows.sample(rng);
        points.row_mut(0).copy_from_slice(data.row(i));
        picked.insert(i);
        norms[i]
    };

    let mut selected = 1;
    for current in 1..expected.min(data.nrows()) {
        let last = points.row(current - 1);
        let s = update_distances(
            &mut min_distances,
            transpose,
            norms,
            last,
            previous_square_norm,
        );

        // Pick a threshold.
        // Due to the way we compute distances, values less than 0.0 are technically
        // possible.
        match Uniform::<f64>::new(0.0, s) {
            Ok(distribution) => {
                let threshold = distribution.sample(rng);
                let mut rolling_sum: f64 = 0.0;
                for (i, d) in min_distances.iter().enumerate() {
                    rolling_sum += <f32 as Into<f64>>::into(*d);
                    if rolling_sum >= threshold && (*d > 0.0) && !picked.contains(&i) {
                        // This point is the winner.
                        // Copy it over and update our scratch variables.
                        points.row_mut(current).clone_from_slice(data.row(i));
                        picked.insert(i);
                        previous_square_norm = norms[i];
                        selected = current + 1;
                        break;
                    }
                }
            }
            // If the range is empty, this implies that `s == 0.0`.
            //
            // In this case, we skip and try again,
            Err(rand::distr::uniform::Error::EmptyRange) => {}
            // The upper bound is infinite - this is an error.
            Err(rand::distr::uniform::Error::NonFinite) => {
                return Err(KMeansPlusPlusError::new(
                    selected,
                    expected,
                    FailureReason::SawInfinity,
                ));
            }
        }

        // If we successfully picked a row, than `selected == current`.
        //
        // If this is not the case, then we failed due to insufficient diversity.
        if selected != (current + 1) {
            return Err(KMeansPlusPlusError::new(
                selected,
                expected,
                FailureReason::InsufficientDiversity,
            ));
        }
    }

    // We may have terminated early due to the dataset being too small.
    if selected != expected {
        Err(KMeansPlusPlusError::new(
            selected,
            expected,
            FailureReason::DatasetTooSmall,
        ))
    } else {
        Ok(())
    }
}

pub fn kmeans_plusplus_into(
    centers: MutMatrixView<'_, f32>,
    data: MatrixView<'_, f32>,
    rng: &mut dyn RngCore,
) -> Result<(), KMeansPlusPlusError> {
    assert_eq!(
        centers.ncols(),
        data.ncols(),
        "centers output matrix should have the same dimensionality as the dataset"
    );

    const GROUPSIZE: usize = 16;
    let mut norms: Vec<f32> = vec![0.0; data.nrows()];

    for (n, d) in std::iter::zip(norms.iter_mut(), data.row_iter()) {
        *n = square_norm(d);
    }

    let transpose = BlockTranspose::<GROUPSIZE>::from_matrix_view(data);
    kmeans_plusplus_into_inner(centers, data.into(), &transpose, &norms, rng)
}

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, views::Matrix};
    use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    use super::*;
    use crate::utils;

    fn is_in(needle: &[f32], haystack: MatrixView<'_, f32>) -> bool {
        assert_eq!(needle.len(), haystack.ncols());
        haystack.row_iter().any(|row| row == needle)
    }

    fn check_post_conditions(
        centers: MatrixView<'_, f32>,
        data: MatrixView<'_, f32>,
        err: &KMeansPlusPlusError,
    ) {
        assert_eq!(err.expected, centers.nrows());
        assert!(err.expected > err.selected);
        for i in 0..err.selected {
            assert!(is_in(centers.row(i), data.as_view()));
        }
        for i in err.selected..centers.nrows() {
            assert!(centers.row(i).iter().all(|j| *j == 0.0));
        }
    }

    ///////////////////
    // Error Display //
    ///////////////////

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", FailureReason::DatasetTooSmall),
            "dataset does not have enough points"
        );

        assert_eq!(
            format!("{}", FailureReason::InsufficientDiversity),
            "dataset is insufficiently diverse"
        );

        assert_eq!(
            format!("{}", FailureReason::SawInfinity),
            "a value of infinity or NaN was observed"
        );
    }

    //////////////////////
    // Update Distances //
    //////////////////////

    /// Seed `x` (a KxN matrix) with the following:
    /// ```text
    /// 0,   1,   2,   3 ...   N-1
    /// 1,   2,   3,   4 ...   N
    /// ...
    /// K-1, K,   K+1, K+3 ... N+K-2
    /// ```
    fn set_default_values(mut x: MutMatrixView<'_, f32>) {
        for (i, row) in x.row_iter_mut().enumerate() {
            for (j, r) in row.iter_mut().enumerate() {
                *r = (i + j) as f32;
            }
        }
    }

    // The implementation of `update_distances` is not particularly unsafe with relation
    // to `dim`, but *is* potentially unsafe with respect to `num_points`.
    //
    // Therefore, we need to be sure we sweep sufficient values of `num_points` to get full
    // coverage with Miri.
    //
    // This works our to our advantage because smaller `dims` mean we can have precise
    // floating point values.
    fn test_update_distances_impl<const N: usize, R>(num_points: usize, dim: usize, rng: &mut R)
    where
        BlockTranspose<N>: MicroKernel,
        R: Rng,
    {
        let context = lazy_format!(
            "setup: N = {}, num_points = {}, dim = {}",
            N,
            num_points,
            dim
        );

        let mut data = Matrix::<f32>::new(0.0, num_points, dim);
        set_default_values(data.as_mut_view());

        let square_norms: Vec<f32> = data.row_iter().map(square_norm).collect();

        // The sample points we are computing the distances against.
        let num_samples = 3;
        let mut samples = Matrix::<f32>::new(0.0, num_samples, dim);
        let mut distances = vec![f32::INFINITY; num_points];
        let distribution = Uniform::<u32>::new(0, (num_points + dim) as u32).unwrap();
        let transpose = BlockTranspose::<N>::from_matrix_view(data.as_view());

        let mut last_residual = f64::INFINITY;
        for i in 0..num_samples {
            // Pick a sample.
            {
                let row = samples.row_mut(i);
                row.iter_mut().for_each(|r| {
                    *r = distribution.sample(rng) as f32;
                });
            }
            let row = samples.row(i);
            let norm = square_norm(row);

            let residual = update_distances(&mut distances, &transpose, &square_norms, row, norm);

            // Make sure all the distances are correct.
            for (n, (d, data)) in std::iter::zip(distances.iter(), data.row_iter()).enumerate() {
                let mut min_distance = f32::INFINITY;
                for j in 0..=i {
                    let distance = SquaredL2::evaluate(samples.row(j), data);
                    min_distance = min_distance.min(distance);
                }
                assert_eq!(
                    min_distance, *d,
                    "failed on row {n} on iteration {i}. {}",
                    context
                );
            }

            // The distances match - ensure that the residual was computed properly.
            assert_eq!(
                residual,
                distances.iter().sum::<f32>() as f64,
                "residual sum failed on iteration {i} - {}",
                context
            );

            // Finally - make sure the residual is dropping.
            assert!(
                residual <= last_residual,
                "residual check failed on iteration {}, last = {}, this = {} - {}",
                i,
                last_residual,
                residual,
                context
            );

            last_residual = residual;
        }
    }

    /// A note on testing methodology:
    ///
    /// This function targets running under Miri to test the indexing logic in
    /// `update_distances`.
    ///
    /// It is appropriate for implementations what are not "unsafe" in `dim` but are
    /// "unsafe" in `num_points`.
    ///
    /// In other words, the SIMD operations we are tracking block along `num_points` and
    /// not `dim`. This lets us run at a much smaller `dim` to help Miri finish more quickly.
    #[test]
    fn test_update_distances() {
        let mut rng = StdRng::seed_from_u64(0x56c94b53c73e4fd9);
        for num_points in 0..48 {
            #[cfg(miri)]
            if num_points % 7 != 0 {
                continue;
            }

            for dim in 1..4 {
                test_update_distances_impl(num_points, dim, &mut rng);
            }
        }
    }

    //////////////
    // Kmeans++ //
    //////////////

    // Kmeans++ sanity checks - if there are only `N` distinct and we want `N` centers,
    // then all `N` should be selected without repeats.
    #[cfg(not(miri))]
    fn sanity_check_impl<R: Rng>(ncenters: usize, dim: usize, rng: &mut R) {
        let repeats_per_center = 3;
        let context = lazy_format!(
            "dim = {}, ncenters = {}, repeats_per_center = {}",
            dim,
            ncenters,
            repeats_per_center
        );

        let ndata = repeats_per_center * ncenters;
        let mut values: Vec<f32> = (0..ncenters)
            .flat_map(|i| (0..repeats_per_center).map(move |_| i as f32))
            .collect();
        assert_eq!(values.len(), ndata);

        values.shuffle(rng);
        let mut data = Matrix::new(0.0, ndata, dim);
        for (r, v) in std::iter::zip(data.row_iter_mut(), values.iter()) {
            r.fill(*v);
        }

        let mut centers = Matrix::new(f32::INFINITY, ncenters, dim);
        kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), rng).unwrap();

        // Make sure that each value was selected for a center.
        let mut seen = HashSet::<usize>::new();
        for c in centers.row_iter() {
            let first = c[0];
            assert!(c.iter().all(|i| *i == first));

            let v: usize = first.round() as usize;
            assert_eq!(v as f32, first, "conversion was not lossless - {}", context);

            if !seen.insert(v) {
                panic!("value {first} seen more than oncex - {}", context);
            }
        }
        assert_eq!(
            seen.len(),
            ncenters,
            "not all points were seen - {}",
            context
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn sanity_check() {
        let dims = [1, 16];
        let ncenters = [1, 5, 20, 255];
        let mut rng = StdRng::seed_from_u64(0x68c2080f2ea36f5a);

        for ncenters in ncenters {
            for dim in dims {
                sanity_check_impl(ncenters, dim, &mut rng);
            }
        }
    }

    // This test is like the sanity check - but instead of exact repeats, we use slightly
    // perturbed values to test that the proportionality is of distances is respected.
    #[cfg(not(miri))]
    fn fuzzy_sanity_check_impl<R: Rng>(ncenters: usize, dim: usize, rng: &mut R) {
        let repeats_per_center = 3;

        // A spreading coefficient to space-out points.
        let spreading_multiplier: usize = 16;
        // Purturbation distribution to apply to the input data.
        let perturbation_distribution = Uniform::new(-0.125, 0.125).unwrap();

        let context = lazy_format!(
            "dim = {}, ncenters = {}, repeats_per_center = {}, multiplier = {}",
            dim,
            ncenters,
            repeats_per_center,
            spreading_multiplier,
        );

        let ndata = repeats_per_center * ncenters;
        let mut values: Vec<f32> = (0..ncenters)
            .flat_map(|i| {
                // We need to bounce through a vec to avoid borrowing issues.
                let v: Vec<f32> = (0..repeats_per_center)
                    .map(|_| {
                        (spreading_multiplier * i) as f32 + perturbation_distribution.sample(rng)
                    })
                    .collect();

                v.into_iter()
            })
            .collect();
        assert_eq!(values.len(), ndata);

        values.shuffle(rng);
        let mut data = Matrix::new(0.0, ndata, dim);
        for (r, v) in std::iter::zip(data.row_iter_mut(), values.iter()) {
            r.fill(*v);
        }

        let mut centers = Matrix::new(f32::INFINITY, ncenters, dim);
        kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), rng).unwrap();

        // Make sure that each value was selected for a center.
        let mut seen = HashSet::<usize>::new();
        for (i, c) in centers.row_iter().enumerate() {
            let first = c[0];
            let v: usize = first.round() as usize;
            assert_eq!(
                v % spreading_multiplier,
                0,
                "expected row value to be close to a multiple of the spreading multiplier, \
                 instead got {} - {}",
                v,
                context
            );
            seen.insert(v);

            // Make sure the center is equal to one of the data points.
            let mut found = false;
            for r in data.row_iter() {
                if r == c {
                    found = true;
                    break;
                }
            }
            if !found {
                panic!(
                    "center {} was not found in the original dataset - {}",
                    i, context,
                );
            }
        }
        assert!(
            seen.len() as f32 >= 0.95 * (ncenters as f32),
            "expected the distribution of centers to be wide, \
             instead {} unique values were found - {}",
            seen.len(),
            context
        );
    }

    #[test]
    #[cfg(not(miri))]
    fn fuzzy_sanity_check() {
        let dims = [1, 16];
        // Apparently passing in `0` for `ncenters` works in a well-defined way.
        let ncenters = [0, 1, 5, 20, 255];
        let mut rng = StdRng::seed_from_u64(0x68c2080f2ea36f5a);

        for ncenters in ncenters {
            for dim in dims {
                fuzzy_sanity_check_impl(ncenters, dim, &mut rng);
            }
        }
    }

    // Failure modes
    #[test]
    fn fail_empty_dataset() {
        let data = Matrix::new(0.0, 0, 5);
        let mut centers = Matrix::new(0.0, 10, data.ncols());

        let mut rng = StdRng::seed_from_u64(0xa9eae150d30845a1);

        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(
            result.is_err(),
            "kmeans++ on an empty dataset with non-empty centers should be an error"
        );
        let err = result.unwrap_err();
        assert_eq!(err.selected, 0);
        assert_eq!(err.expected, centers.nrows());
        assert_eq!(err.reason, FailureReason::DatasetTooSmall);
        assert!(!err.is_numerically_recoverable());

        check_post_conditions(centers.as_view(), data.as_view(), &err);
    }

    #[test]
    fn both_empty_is_okay() {
        let data = Matrix::new(0.0, 0, 5);
        let mut centers = Matrix::new(0.0, 0, data.ncols());
        let mut rng = StdRng::seed_from_u64(0x6f7031afd9b5aa18);
        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(
            result.is_ok(),
            "selecting 0 points from an empty dataset is okay"
        );
    }

    #[test]
    fn fail_dataset_not_big_enough() {
        let ndata = 5;
        let ncenters = 10;
        let dim = 5;

        let mut data = Matrix::new(0.0, ndata, dim);
        set_default_values(data.as_mut_view());
        let mut centers = Matrix::new(f32::INFINITY, ncenters, data.ncols());

        let mut rng = StdRng::seed_from_u64(0xa9eae150d30845a1);

        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(
            result.is_err(),
            "kmeans++ on an empty dataset with non-empty centers should be an error"
        );
        let err = result.unwrap_err();
        assert_eq!(err.selected, data.nrows());
        assert_eq!(err.expected, centers.nrows());
        assert_eq!(err.reason, FailureReason::DatasetTooSmall);
        assert!(err.is_numerically_recoverable());

        check_post_conditions(centers.as_view(), data.as_view(), &err);
    }

    // In this test - we ensure that we process as much of a non-diverse dataset as we can
    // before returning an error.
    #[test]
    fn fail_diversity_check() {
        let ncenters = 10;
        let ndata = 50;
        let dim = 3;
        let mut rng = StdRng::seed_from_u64(0xca57b032c21bf4bb);

        // Make sure the dataset only contains 5 unique values.
        let repeats_per_center = 10;
        assert!(ncenters * repeats_per_center > ndata);
        let mut values: Vec<f32> = (0..utils::div_round_up(ndata, repeats_per_center))
            .flat_map(|i| (0..repeats_per_center).map(move |_| i as f32))
            .collect();
        assert!(values.len() >= ndata);

        values.shuffle(&mut rng);
        let mut data = Matrix::new(0.0, ndata, dim);
        for (r, v) in std::iter::zip(data.row_iter_mut(), values.iter()) {
            r.fill(*v);
        }

        let mut centers = Matrix::new(f32::INFINITY, ncenters, dim);
        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(
            result.is_err(),
            "dataset should not have enough unique points"
        );
        let err = result.unwrap_err();
        assert_eq!(err.selected, utils::div_round_up(ndata, repeats_per_center));
        assert_eq!(err.expected, centers.nrows());
        assert_eq!(err.reason, FailureReason::InsufficientDiversity);
        assert!(err.is_numerically_recoverable());

        check_post_conditions(centers.as_view(), data.as_view(), &err);
    }

    #[test]
    fn fail_intinity_check() {
        let mut data = Matrix::new(0.0, 10, 1);
        set_default_values(data.as_mut_view());

        // A very large value that will overflow to infinity when computing the norm.
        data[(6, 0)] = -3.4028235e38;
        let mut centers = Matrix::new(0.0, 2, 1);

        let mut rng = StdRng::seed_from_u64(0xc0449b2aa4e12f05);

        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(result.is_err(), "result should complain about infinity");
        let err = result.unwrap_err();
        assert_eq!(err.selected, 1);
        assert_eq!(err.expected, centers.nrows());
        assert_eq!(err.reason, FailureReason::SawInfinity);
        assert!(!err.is_numerically_recoverable());

        check_post_conditions(centers.as_view(), data.as_view(), &err);
    }

    #[test]
    fn fail_nan_check() {
        let mut data = Matrix::new(0.0, 10, 1);
        set_default_values(data.as_mut_view());

        // A very large value that will overflow to infinity when computing the norm.
        data[(6, 0)] = f32::NAN;
        let mut centers = Matrix::new(0.0, 2, 1);

        let mut rng = StdRng::seed_from_u64(0x55808c6c728c8473);

        let result = kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng);
        assert!(result.is_err(), "result should complain about NaN");
        let err = result.unwrap_err();
        assert_eq!(err.selected, 1);
        assert_eq!(err.expected, centers.nrows());
        assert_eq!(err.reason, FailureReason::SawInfinity);
        assert!(!err.is_numerically_recoverable());

        check_post_conditions(centers.as_view(), data.as_view(), &err);
    }

    ///////////////////////////////
    // Panics - update_distances //
    ///////////////////////////////

    #[test]
    #[should_panic(expected = "new point and dataset must have the same dimension")]
    fn update_distances_panics_dim_mismatch() {
        let npoints = 5;
        let dim = 8;
        let mut square_distances = vec![0.0; npoints];
        let data = Matrix::new(0.0, npoints, dim);
        let norms = vec![0.0; npoints];
        let this = vec![0.0; dim + 1]; // Incorrect
        let this_square_norm = 0.0;
        update_distances::<16>(
            &mut square_distances,
            &BlockTranspose::from_matrix_view(data.as_view()),
            &norms,
            &this,
            this_square_norm,
        );
    }

    #[test]
    #[should_panic(expected = "distances buffer and dataset must have the same length")]
    fn update_distances_panics_distances_length_mismatch() {
        let npoints = 5;
        let dim = 8;
        let mut square_distances = vec![0.0; npoints + 1]; // Incorrect
        let data = Matrix::new(0.0, npoints, dim);
        let norms = vec![0.0; npoints];
        let this = vec![0.0; dim];
        let this_square_norm = 0.0;
        update_distances::<16>(
            &mut square_distances,
            &BlockTranspose::from_matrix_view(data.as_view()),
            &norms,
            &this,
            this_square_norm,
        );
    }

    #[test]
    #[should_panic(expected = "norms and dataset must have the same length")]
    fn update_distances_panics_norms_length_mismatch() {
        let npoints = 5;
        let dim = 8;
        let mut square_distances = vec![0.0; npoints];
        let data = Matrix::new(0.0, npoints, dim);
        let norms = vec![0.0; npoints + 1]; // Incorrect
        let this = vec![0.0; dim];
        let this_square_norm = 0.0;
        update_distances::<16>(
            &mut square_distances,
            &BlockTranspose::from_matrix_view(data.as_view()),
            &norms,
            &this,
            this_square_norm,
        );
    }

    ///////////////////////////////////
    // Panics - kmeans_plusplus_into //
    ///////////////////////////////////

    #[test]
    #[should_panic(
        expected = "centers output matrix should have the same dimensionality as the dataset"
    )]
    fn kmeans_plusplus_into_panics_dim_mismatch() {
        let mut centers = Matrix::new(0.0, 2, 10);
        let data = Matrix::new(0.0, 2, 9);
        kmeans_plusplus_into(
            centers.as_mut_view(),
            data.as_view(),
            &mut rand::rngs::ThreadRng::default(),
        )
        .unwrap();
    }
}
