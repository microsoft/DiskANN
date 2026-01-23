/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::{SIMDMask, SIMDMulAdd, SIMDPartialOrd, SIMDSelect, SIMDSumTree, SIMDVector};

use super::common::{square_norm, BlockTranspose};
use diskann_utils::{
    strided::StridedView,
    views::{Matrix, MatrixView, MutMatrixView},
};

////////////////////////////////
// Closest Centers Algorithms //
////////////////////////////////

diskann_wide::alias!(f32s = f32x8);
diskann_wide::alias!(u32s = u32x8);

// A computation strategy where the final distance values are accumulated in-place.
//
// This is sufficient for low-dimensional clusterings but suffers when the dimensionality
// increases.
//
// Return the residual distance.
pub fn distances_in_place(
    dataset: &BlockTranspose<16>,
    data_norms: &[f32],
    centers: MatrixView<'_, f32>,
    center_norms: &[f32],
    nearest: &mut [u32],
) -> f32 {
    // Safety Checks!
    // Our unchecked-loads rely on these invariants holding.

    // Check 1: Same number of norms as dataset elements.
    assert_eq!(
        dataset.nrows(),
        data_norms.len(),
        "dataset and data norms should have the same length"
    );
    // Check 2: Datasets have the same dimension.
    assert_eq!(
        centers.ncols(),
        dataset.ncols(),
        "dataset and centers should have the same dimensions"
    );
    // Check 3: Same number of center norms as centers.
    assert_eq!(
        centers.nrows(),
        center_norms.len(),
        "centers and center norms should have the same length"
    );
    // Check 4: The `nearest` output's length matches the input dataset.
    assert_eq!(
        nearest.len(),
        dataset.nrows(),
        "dataset and nearest-buffer should have the same length"
    );

    const N: usize = 16;
    const N2: usize = N / 2;

    diskann_wide::alias!(m32s = mask_f32x8);

    let mut residual = f32s::default(diskann_wide::ARCH);

    // Compute the distances between all vectors in the block with index `block` and
    // two consecutive centers starting at `center_row_start`.
    //
    // SAFETY: The following must hold:
    // * `block < transpose.num_blocks()` (this is safe to call on the remainder block).
    // * `center_row_start + 1 < centers.nrows()`: This unrolls by a factor of 2, so reading
    //    two rows must be valid.
    let process_block_unroll_2 = |block: usize, center_row_start: usize| {
        debug_assert!(block < dataset.num_blocks());
        debug_assert!(center_row_start + 1 < centers.nrows());

        let mut s00 = f32s::default(diskann_wide::ARCH);
        let mut s01 = f32s::default(diskann_wide::ARCH);
        let mut s10 = f32s::default(diskann_wide::ARCH);
        let mut s11 = f32s::default(diskann_wide::ARCH);

        // SAFETY: Closure pre-conditions mean that this access is in-bounds.
        let block_ptr = unsafe { dataset.block_ptr_unchecked(block) };
        for dim in 0..dataset.ncols() {
            // SAFETY: For all rows in this block, 16 reads are valid and by construction,
            // `dim < dataset.block_size()`. This loads the first 8.
            let d0 = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)) };
            // SAFETY: For all rows in this block, 16 reads are valid and by construction,
            // `dim < dataset.block_size()`. This loads the last 8.
            let d1 = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)) };

            // SAFETY: Closure pre-conditions and Check 2 make this a valid access.
            let c0 = f32s::splat(diskann_wide::ARCH, unsafe {
                *centers.get_unchecked(center_row_start, dim)
            });
            // SAFETY: Closure pre-conditions and Check 2 make this a valid access.
            let c1 = f32s::splat(diskann_wide::ARCH, unsafe {
                *centers.get_unchecked(center_row_start + 1, dim)
            });

            s00 = c0.mul_add_simd(d0, s00);
            s01 = c0.mul_add_simd(d1, s01);
            s10 = c1.mul_add_simd(d0, s10);
            s11 = c1.mul_add_simd(d1, s11);
        }
        (s00, s01, s10, s11)
    };

    // Compute the distances between all vectors in the block with index `block` and one
    // center starting at `center_row_start`.
    //
    // SAFETY: The following must hold:
    // * `block < transpose.num_blocks()` (this is safe to call on the remainder block).
    // * `center_row_start < centers.nrows()`: This unrolls by a factor of 2, so reading
    //    two rows must be valid.
    let process_block_no_unroll = |block: usize, center_row_start: usize| {
        debug_assert!(block < dataset.num_blocks());
        debug_assert!(center_row_start + 1 == centers.nrows());

        let mut s00 = f32s::default(diskann_wide::ARCH);
        let mut s01 = f32s::default(diskann_wide::ARCH);

        // SAFETY: Closure pre-conditions mean that this access is in-bounds.
        let block_ptr = unsafe { dataset.block_ptr_unchecked(block) };
        for dim in 0..dataset.ncols() {
            // SAFETY: For all rows in this block, 16 reads are valid and by construction,
            // `dim < dataset.block_size()`. This loads the first 8.
            let d0 = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim)) };
            // SAFETY: For all rows in this block, 16 reads are valid and by construction,
            // `dim < dataset.block_size()`. This loads the last 8.
            let d1 = unsafe { f32s::load_simd(diskann_wide::ARCH, block_ptr.add(N * dim + N2)) };

            // SAFETY: Closure pre-conditions and Check 2 make this a valid access.
            let c0 = f32s::splat(diskann_wide::ARCH, unsafe {
                *centers.get_unchecked(center_row_start, dim)
            });

            s00 = c0.mul_add_simd(d0, s00);
            s01 = c0.mul_add_simd(d1, s01);
        }
        (s00, s01)
    };

    // Figure out if the number of centers to process is even or not.
    // If it's even, we can work on centers two rows at a time.
    //
    // Otherwise, we need to deal with the last `centers` row independently.
    let last_pair = if centers.nrows().is_multiple_of(2) {
        centers.nrows()
    } else {
        centers.nrows() - 1
    };

    for i in 0..dataset.full_blocks() {
        let mut t0 = (
            f32s::splat(diskann_wide::ARCH, f32::INFINITY),
            u32s::splat(diskann_wide::ARCH, u32::MAX),
        );
        let mut t1 = (
            f32s::splat(diskann_wide::ARCH, f32::INFINITY),
            u32s::splat(diskann_wide::ARCH, u32::MAX),
        );

        // SAFETY: Check 1 means this access is in-bounds.
        let data_norm_ptr = unsafe { data_norms.as_ptr().add(N * i) };

        // SAFETY: By Check 1 and by being in a full-block, this implies that up to 16
        // values are safe to read from `data_norm_ptr`.
        let d0 = unsafe { f32s::load_simd(diskann_wide::ARCH, data_norm_ptr) };

        // SAFETY: By Check 1 and by being in a full-block, this implies that up to 16
        // values are safe to read from `data_norm_ptr`.
        let d1 = unsafe { f32s::load_simd(diskann_wide::ARCH, data_norm_ptr.add(N2)) };
        for row_start in (0..last_pair).step_by(2) {
            // SAFETY: By construction, `i < transpose.num_blocks()` and
            // `row_start + 1 < centers.nrows()`.
            let (s00, s01, s10, s11) = process_block_unroll_2(i, row_start);

            // Compensate for the inner-product calculation.
            // SAFETY: By Check 3, this access is in-bounds.
            let n0 = f32s::splat(diskann_wide::ARCH, *unsafe {
                center_norms.get_unchecked(row_start)
            });
            // SAFETY: By Check 3 and loop construction, this access is in-bounds.
            let n1 = f32s::splat(diskann_wide::ARCH, *unsafe {
                center_norms.get_unchecked(row_start + 1)
            });

            let s00 = n0 - s00 - s00 + d0;
            let s01 = n0 - s01 - s01 + d1;
            let s10 = n1 - s10 - s10 + d0;
            let s11 = n1 - s11 - s11 + d1;

            let r0 = u32s::splat(diskann_wide::ARCH, row_start as u32);
            let r1 = u32s::splat(diskann_wide::ARCH, (row_start + 1) as u32);
            t0 = update(update(t0, (s00, r0)), (s10, r1));
            t1 = update(update(t1, (s01, r0)), (s11, r1));
        }

        // If there is an odd-number of centers, we need to handle that individually.
        if !centers.nrows().is_multiple_of(2) {
            // SAFETY: By construction, `i < transpose.num_blocks()` and
            // `last_pair < centers.nrows()`.
            let (s00, s01) = process_block_no_unroll(i, last_pair);
            // SAFETY: by Check 3, this access is in-bounds.
            let n0 = f32s::splat(diskann_wide::ARCH, unsafe {
                *center_norms.get_unchecked(last_pair)
            });

            let s00 = n0 - s00 - s00 + d0;
            let s01 = n0 - s01 - s01 + d1;

            let r = u32s::splat(diskann_wide::ARCH, last_pair as u32);
            t0 = update(t0, (s00, r));
            t1 = update(t1, (s01, r));
        }

        // Write back.
        // SAFETY: By Check 4, at least 16 elements are valid and mutable beginning at the
        // offset `N * i`. This writes the first 8.
        unsafe { t0.1.store_simd(nearest.as_mut_ptr().add(N * i)) }
        // SAFETY: By Check 4, at least 16 elements are valid and mutable beginning at the
        // offset `N * i`. This writes the last 8.
        unsafe { t1.1.store_simd(nearest.as_mut_ptr().add(N * i + N2)) }

        // Update the residual.
        residual = residual + t0.0 + t1.0;
    }

    // IF there is a remainder block - we can do pretty much exactly the same thing we did
    // for the full blocks. We just need to be a bit more careful when writing back the
    // results.
    let remainder = dataset.remainder();
    if remainder != 0 {
        let i = dataset.full_blocks();
        let lo = remainder.min(N2);
        let hi = remainder - lo;

        let mut t0 = (
            f32s::splat(diskann_wide::ARCH, f32::INFINITY),
            u32s::splat(diskann_wide::ARCH, u32::MAX),
        );
        let mut t1 = (
            f32s::splat(diskann_wide::ARCH, f32::INFINITY),
            u32s::splat(diskann_wide::ARCH, u32::MAX),
        );

        // SAFETY: Check 1 means this access is in-bounds.
        let data_norm_ptr = unsafe { data_norms.as_ptr().add(N * i) };

        // SAFETY: By Check 1 and by being in a partial block means that up to `remainder`
        // elements are valid. This loads up to the first 8.
        let d0 = unsafe { f32s::load_simd_first(diskann_wide::ARCH, data_norm_ptr, lo) };
        let d1 = if hi == 0 {
            f32s::default(diskann_wide::ARCH)
        } else {
            // SAFETY: By Check 1 and by being in a partial block means that up to `remainder`
            // elements are valid. By taking this branch, we know that `remainder` is
            // at least 9. So it's okay to add 8 to `data_norm_pointer` and load `hi` elements.
            unsafe { f32s::load_simd_first(diskann_wide::ARCH, data_norm_ptr.add(N2), hi) }
        };

        for row_start in (0..last_pair).step_by(2) {
            // SAFETY: By construction, `i < transpose.num_blocks()` and
            // `row_start + 1 < centers.nrows()`.
            let (s00, s01, s10, s11) = process_block_unroll_2(i, row_start);

            // Compensate for the inner-product calculation.
            // SAFETY: By Check 3, this access is in-bounds.
            let n0 = f32s::splat(diskann_wide::ARCH, *unsafe {
                center_norms.get_unchecked(row_start)
            });
            // SAFETY: By Check 3 and loop construction, this access is in-bounds.
            let n1 = f32s::splat(diskann_wide::ARCH, *unsafe {
                center_norms.get_unchecked(row_start + 1)
            });

            let s00 = n0 - s00 - s00 + d0;
            let s01 = n0 - s01 - s01 + d1;
            let s10 = n1 - s10 - s10 + d0;
            let s11 = n1 - s11 - s11 + d1;

            let r0 = u32s::splat(diskann_wide::ARCH, row_start as u32);
            let r1 = u32s::splat(diskann_wide::ARCH, (row_start + 1) as u32);
            t0 = update(update(t0, (s00, r0)), (s10, r1));
            t1 = update(update(t1, (s01, r0)), (s11, r1));
        }

        if !centers.nrows().is_multiple_of(2) {
            // SAFETY: By construction, `i < transpose.num_blocks()` and
            // `last_pair < centers.nrows()`.
            let (s00, s01) = process_block_no_unroll(i, last_pair);
            // SAFETY: by Check 3, this access is in-bounds.
            let n0 = f32s::splat(diskann_wide::ARCH, unsafe {
                *center_norms.get_unchecked(last_pair)
            });

            let s00 = n0 - s00 - s00 + d0;
            let s01 = n0 - s01 - s01 + d1;

            let r = u32s::splat(diskann_wide::ARCH, last_pair as u32);
            t0 = update(t0, (s00, r));
            t1 = update(t1, (s01, r));
        }

        // Write back.
        // SAFETY: By Check 4, at least 1 and up to 16 elements are valid and mutable
        // beginning at the offset `N * i`. This writes the first `min(8, remainder)`.
        unsafe { t0.1.store_simd_first(nearest.as_mut_ptr().add(N * i), lo) };
        if hi != 0 {
            // SAFETY: By Check 4, at least 1 and up to 16 elements are valid and mutable
            // beginning at the offset `N * i`. If `hi != 0`, then `remainder` is at
            // least 9. So it's okay to add `8` to `nearest.as_mut_ptr()` and store `hi`
            // elements.
            unsafe {
                t1.1.store_simd_first(nearest.as_mut_ptr().add(N * i + N2), hi)
            };
        }

        // Update the residual
        // Use a masked select to only accumulate lanes that are in-bounds.
        residual = m32s::keep_first(diskann_wide::ARCH, lo).select(residual + t0.0, residual);
        residual = m32s::keep_first(diskann_wide::ARCH, hi).select(residual + t1.0, residual);
    }
    residual.sum_tree()
}

#[inline(always)]
fn update((d0, i0): (f32s, u32s), (d1, i1): (f32s, u32s)) -> (f32s, u32s) {
    // Generate a mask with lanes set if a computed distance is less that one of theH
    // current minimum distances.
    let mask = d1.lt_simd(d0);
    (
        mask.select(d1, d0),
        <u32s as SIMDVector>::Mask::from(mask).select(i1, i0),
    )
}

/////////////////
// Update Step //
/////////////////

fn update_centroids(mut centers: MutMatrixView<'_, f32>, data: StridedView<'_, f32>, map: &[u32]) {
    let mut sums = Matrix::<f64>::new(0.0, centers.nrows(), centers.ncols());
    let mut counts: Vec<u32> = vec![0; centers.nrows()];
    data.row_iter().zip(map.iter()).for_each(|(row, &center)| {
        counts[center as usize] += 1;
        let sum = sums.row_mut(center as usize);
        std::iter::zip(sum.iter_mut(), row.iter()).for_each(|(s, r)| {
            *s += <f32 as Into<f64>>::into(*r);
        });
    });

    std::iter::zip(counts.iter(), sums.row_iter())
        .zip(centers.row_iter_mut())
        .for_each(|((count, sum), center)| {
            // If the count is zero - we do not want to divide by it because that will
            // result in `NaN`.
            let count = (*count).max(1);
            std::iter::zip(sum.iter(), center.iter_mut()).for_each(|(s, c)| {
                *c = (*s / (count as f64)) as f32;
            });
        });
}

////////////
// Lloyds //
////////////

pub(crate) fn lloyds_inner(
    data: StridedView<'_, f32>,
    square_norms: &[f32],
    transpose: &BlockTranspose<16>,
    mut centers: MutMatrixView<'_, f32>,
    max_reps: usize,
) -> (Vec<u32>, f32) {
    // Check our requirements.
    let num_data = data.nrows();
    assert_eq!(
        num_data,
        square_norms.len(),
        "data and norms should have the same length"
    );
    assert_eq!(
        num_data,
        transpose.nrows(),
        "data and transpose should have the same length"
    );

    let dim = data.ncols();
    assert_eq!(
        dim,
        transpose.block_size(),
        "data and transpose should have the same dimensions"
    );
    assert_eq!(
        dim,
        centers.ncols(),
        "data and centers should have the same dimensions"
    );

    let mut center_square_norms: Vec<f32> = centers.row_iter().map(square_norm).collect();
    let mut assignments: Vec<u32> = vec![0; num_data];
    let mut residual = 0.0;

    for i in 0..max_reps {
        residual = distances_in_place(
            transpose,
            square_norms,
            centers.as_view(),
            &center_square_norms,
            &mut assignments,
        );
        update_centroids(centers.as_mut_view(), data, &assignments);
        if i != max_reps - 1 {
            std::iter::zip(center_square_norms.iter_mut(), centers.row_iter()).for_each(
                |(c, center)| {
                    *c = square_norm(center);
                },
            );
        }
    }
    (assignments, residual)
}

/// Run `max_reps` of Lloyd's algorithm over `data` and `centers`, updating the `centers`
/// argument with the result.
///
/// # Returns
///
/// Returns a tuple `x = (Vec<u32>, f32)` where
/// * `x.0` is the position-wise assignments of each data rows nearest center.
/// * `x.1` is the final squared-l2 residual of the clustered dataset.
///
/// # Panics
///
/// Panics if `data.ncols() != centers.ncols()`. The data and centers must have the same
/// dimension.
pub fn lloyds(
    data: MatrixView<'_, f32>,
    centers: MutMatrixView<'_, f32>,
    max_reps: usize,
) -> (Vec<u32>, f32) {
    assert_eq!(
        data.ncols(),
        centers.ncols(),
        "data and centers must have the same dimension",
    );

    let transpose = BlockTranspose::<16>::from_matrix_view(data);
    let square_norms: Vec<f32> = data.row_iter().map(square_norm).collect();
    lloyds_inner(data.into(), &square_norms, &transpose, centers, max_reps)
}

#[cfg(test)]
mod tests {
    use diskann_utils::{lazy_format, views::Matrix};
    use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
    use rand::{
        distr::{Distribution, Uniform},
        rngs::StdRng,
        seq::{IndexedRandom, SliceRandom},
        Rng, SeedableRng,
    };

    use super::*;

    ////////////////////////
    // Distances in Place //
    ////////////////////////

    // The strategy here is we need to test a wide range of dimensions, dataset sizes,
    // and nubmer of centers ... and have the dimensions be small enough that this can run
    // relatively quickly.
    //
    // Outside of rare validations, Miri tests go through a different path for speed purposes.
    fn test_distances_in_place_impl<R: Rng>(
        ndata: usize,
        ncenters: usize,
        dim: usize,
        trials: usize,
        rng: &mut R,
    ) {
        let context = lazy_format!("ncenters = {}, ndata = {}, dim = {}", ncenters, ndata, dim,);

        let mut centers = Matrix::new(0.0, ncenters, dim);
        let mut data = Matrix::new(0.0, ndata, dim);

        // A list of random "nice" offsets that get applied to each center and data point
        // to ensure proper visitation during computation.
        let offsets = [-0.125, -0.0625, -0.03125, 0.03125, 0.0625, 0.125];

        // Initialize `centers` uniformly but with random offsets applied to each dimension.
        for (i, row) in centers.row_iter_mut().enumerate() {
            for c in row {
                *c = (i as f32) + *offsets.choose(rng).unwrap();
            }
        }

        let center_norms: Vec<f32> = centers.row_iter().map(square_norm).collect();

        // This is the distribution of how we assign data points to centers.
        let assignment_distribution = Uniform::<usize>::new(0, centers.nrows()).unwrap();
        let mut nearest: Vec<u32> = vec![0; ndata];
        for trial in 0..trials {
            let assignments: Vec<_> = (0..ndata)
                .map(|_| assignment_distribution.sample(rng))
                .collect();

            for (assignment, row) in std::iter::zip(assignments.iter(), data.row_iter_mut()) {
                for c in row.iter_mut() {
                    *c = (*assignment as f32) + offsets.choose(rng).unwrap()
                }
            }

            let data_norms: Vec<f32> = data.row_iter().map(square_norm).collect();

            let residual = distances_in_place(
                &(BlockTranspose::from_matrix_view(data.as_view())),
                &data_norms,
                centers.as_view(),
                &center_norms,
                &mut nearest,
            );

            // Check that the assignments are correct.
            for (i, (got, expected)) in
                std::iter::zip(nearest.iter(), assignments.iter()).enumerate()
            {
                assert_eq!(
                    *got as usize,
                    *expected,
                    "failed for data index {} on trial {} -- {}\n\
                     row = {:?}\n\
                     expected = {:?}\n\
                     got = {:?}",
                    i,
                    trial,
                    context,
                    data.row(i),
                    centers.row(*expected),
                    centers.row(*got as usize),
                );
            }

            // Check that the residual computation is correct.
            let mut sum: f32 = 0.0;
            for (a, row) in std::iter::zip(assignments.iter(), data.row_iter()) {
                let distance: f32 = SquaredL2::evaluate(row, centers.row(*a));
                sum += distance;
            }
            assert_eq!(sum, residual, "failed on trial {} -- {}", trial, context);
        }
    }

    cfg_if::cfg_if! {
        if #[cfg(miri)] {
            const TRIALS: usize = 1;
        } else {
            const TRIALS: usize = 100;
        }
    }

    #[test]
    fn test_distances_in_place() {
        let mut rng = StdRng::seed_from_u64(0xece88a9c6cd86a8a);
        #[cfg(not(miri))]
        for ndata in 1..=31 {
            for ncenters in 1..=5 {
                for dim in 1..=4 {
                    test_distances_in_place_impl(ndata, ncenters, dim, TRIALS, &mut rng);
                }
            }
        }

        #[cfg(miri)]
        for ndata in 31..=31 {
            for ncenters in 5..=5 {
                for dim in 4..=4 {
                    test_distances_in_place_impl(ndata, ncenters, dim, TRIALS, &mut rng);
                }
            }
        }
    }

    // We do not perform any value-dependent control-flow for memory accesses.
    // Therefore, the miri tests don't require any setup (this helps everything run faseter).
    fn test_miri_distances_in_place_impl(ndata: usize, ncenters: usize, dim: usize) {
        let centers = Matrix::new(0.0, ncenters, dim);
        let data = Matrix::new(0.0, ndata, dim);
        let data_norms = vec![0.0; ndata];
        let center_norms = vec![0.0; ncenters];
        let mut nearest = vec![0; ndata];

        let _ = distances_in_place(
            &(BlockTranspose::from_matrix_view(data.as_view())),
            &data_norms,
            centers.as_view(),
            &center_norms,
            &mut nearest,
        );
    }

    #[test]
    fn test_miri_distances_in_place() {
        // We need to hit each dimension between 0 and a low-multiple of the tiling size
        // of 16.
        //
        // Set the upper-bound to 32.
        //
        // The implementation is not sensitive to the dimension, so we can keep that low.
        //
        // Similarly, we need to ensure we have both an even and odd number of centers,
        // so bound this up to 5.
        for ndata in 1..=35 {
            for ncenters in 1..=5 {
                for dim in 1..=4 {
                    test_miri_distances_in_place_impl(ndata, ncenters, dim);
                }
            }
        }
    }

    // End-to-end test.
    // The strategy is to initialize a dataset as a shuffled version of the following:
    // ```test
    //   0   0   0   0 ...
    //   1   1   1   1 ...
    //   2   2   2   2 ...
    //
    // 100 100 100 100 ...
    // 101 101 101 101 ...
    // 102 102 102 102 ...
    //
    // 200 200 200 200 ...
    // ...
    // ```
    // And to initialize centers as
    // ```
    //  -1  -1  -1  -1 ...
    //  99  99  99  99 ...
    // 199 199 199 199 ...
    // ```
    // After one round of Lloyds algorithm, the centers should be updated to be the
    // center of their respective cluster.
    #[derive(Debug)]
    struct EndToEndSetup {
        ncenters: usize,
        ndim: usize,
        data_per_center: usize,
        step_between_clusters: usize,
        ntrials: usize,
    }

    fn end_to_end_test_impl<R: Rng>(setup: &EndToEndSetup, rng: &mut R) {
        // How far apart each cluster is.
        let mut values: Vec<usize> = (0..setup.ncenters)
            .flat_map(|i| {
                (0..setup.data_per_center).map(move |j| setup.step_between_clusters * i + j)
            })
            .collect();

        let mut center_order: Vec<usize> = (0..setup.ncenters).collect();
        let mut data = Matrix::new(0.0, setup.ncenters * setup.data_per_center, setup.ndim);
        let mut centers = Matrix::new(0.0, setup.ncenters, setup.ndim);

        for trial in 0..setup.ntrials {
            values.shuffle(rng);
            center_order.shuffle(rng);

            // Populate centers
            assert_eq!(center_order.len(), centers.nrows());
            for (c, row) in std::iter::zip(center_order.iter(), centers.row_iter_mut()) {
                row.fill((setup.step_between_clusters * c) as f32 - 1.0);
            }

            // Populate data.
            assert_eq!(values.len(), data.nrows());
            for (d, row) in std::iter::zip(values.iter(), data.row_iter_mut()) {
                row.fill(*d as f32);
            }

            // Run 2 iteration of lloyds.
            // The second iteration ensures that we recompute norms properly.
            let lloyds_iter = 2;
            let (assignments, loss) = lloyds(data.as_view(), centers.as_mut_view(), lloyds_iter);

            // Make sure all the assignments are returned correctly.
            assert_eq!(assignments.len(), values.len());
            for (i, (&got, v)) in std::iter::zip(assignments.iter(), values.iter()).enumerate() {
                let expected: usize = v / setup.step_between_clusters;
                assert_eq!(
                    center_order[got as usize], expected,
                    "failed at position {} in trial {} - prevalue: {} -- {:?}",
                    i, trial, v, setup
                );
            }

            // Make sure `centers` were properly set to their mean value.
            let triangle_sum = setup.data_per_center * (setup.data_per_center - 1) / 2;
            center_order.iter().enumerate().for_each(|(i, o)| {
                let expected = (setup.step_between_clusters * setup.data_per_center * o
                    + triangle_sum) as f32
                    / setup.data_per_center as f32;
                assert!(
                    centers.row(i).iter().all(|v| *v == expected),
                    "at index {}, expected {}, got {:?} -- {:?}",
                    i,
                    expected,
                    centers.row(i),
                    setup,
                );
            });

            // Verify the loss is correct.
            let expected_loss: f32 = std::iter::zip(assignments.iter(), data.row_iter())
                .map(|(a, row)| -> f32 {
                    let c = centers.row(*a as usize);
                    SquaredL2::evaluate(row, c)
                })
                .sum::<f32>();
            assert_eq!(loss, expected_loss);
        }
    }

    #[test]
    fn end_to_end_test() {
        let mut rng = StdRng::seed_from_u64(0xff22c38d0f0531bf);
        let setup = EndToEndSetup {
            ncenters: 11,
            ndim: 4,
            data_per_center: 8,
            step_between_clusters: 20,
            ntrials: 10,
        };
        end_to_end_test_impl(&setup, &mut rng);
    }

    /////////////////////////////////
    // Panics - distances_in_place //
    /////////////////////////////////

    // Verify that our panic safety-checks are in-place.
    #[test]
    #[should_panic(expected = "dataset and data norms should have the same length")]
    fn distances_in_place_panics_data_norms() {
        let data = Matrix::new(0.0, 5, 8);
        let data_norms = vec![0.0; data.nrows() + 1]; // Incorrect
        let centers = Matrix::new(0.0, 2, 8);
        let center_norms = vec![0.0; centers.nrows()];
        let mut nearest = vec![0; data.nrows()];
        distances_in_place(
            &BlockTranspose::from_matrix_view(data.as_view()),
            &data_norms,
            centers.as_view(),
            &center_norms,
            &mut nearest,
        );
    }

    #[test]
    #[should_panic(expected = "dataset and centers should have the same dimension")]
    fn distances_in_place_panics_different_dim() {
        let data = Matrix::new(0.0, 5, 8);
        let data_norms = vec![0.0; data.nrows()];
        let centers = Matrix::new(0.0, 2, 9); // Incorrect
        let center_norms = vec![0.0; centers.nrows()];
        let mut nearest = vec![0; data.nrows()];
        distances_in_place(
            &BlockTranspose::from_matrix_view(data.as_view()),
            &data_norms,
            centers.as_view(),
            &center_norms,
            &mut nearest,
        );
    }

    #[test]
    #[should_panic(expected = "centers and center norms should have the same length")]
    fn distances_in_place_panics_center_norms() {
        let data = Matrix::new(0.0, 5, 8);
        let data_norms = vec![0.0; data.nrows()];
        let centers = Matrix::new(0.0, 2, 8);
        let center_norms = vec![0.0; centers.nrows() + 1]; // Incorrect
        let mut nearest = vec![0; data.nrows()];
        distances_in_place(
            &BlockTranspose::from_matrix_view(data.as_view()),
            &data_norms,
            centers.as_view(),
            &center_norms,
            &mut nearest,
        );
    }

    #[test]
    #[should_panic(expected = "dataset and nearest-buffer should have the same length")]
    fn distances_in_place_panics_nearest() {
        let data = Matrix::new(0.0, 5, 8);
        let data_norms = vec![0.0; data.nrows()];
        let centers = Matrix::new(0.0, 2, 8);
        let center_norms = vec![0.0; centers.nrows()];
        let mut nearest = vec![0; data.nrows() + 1]; // Incorrect
        distances_in_place(
            &BlockTranspose::from_matrix_view(data.as_view()),
            &data_norms,
            centers.as_view(),
            &center_norms,
            &mut nearest,
        );
    }

    ///////////////////////////
    // Panics - lloyds_inner //
    ///////////////////////////

    #[test]
    #[should_panic(expected = "data and norms should have the same length")]
    fn lloyds_inner_panics_norms_length() {
        let data = Matrix::new(0.0, 5, 8);
        let square_norms = vec![0.0; data.nrows() + 1]; // Incorrect
        let mut centers = Matrix::new(0.0, 2, 8);
        lloyds_inner(
            data.as_view().into(),
            &square_norms,
            &BlockTranspose::from_matrix_view(data.as_view()),
            centers.as_mut_view(),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "data and transpose should have the same length")]
    fn lloyds_inner_panics_transpose_length() {
        let data = Matrix::new(0.0, 5, 8);
        let data_incorrect = Matrix::new(0.0, 5 + 1, 8); // Incorrect
        let square_norms = vec![0.0; data.nrows()];
        let mut centers = Matrix::new(0.0, 2, 8);
        lloyds_inner(
            data.as_view().into(),
            &square_norms,
            &BlockTranspose::from_matrix_view(data_incorrect.as_view()),
            centers.as_mut_view(),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "data and transpose should have the same dimensions")]
    fn lloyds_inner_panics_transpose_dim() {
        let data = Matrix::new(0.0, 5, 8);
        let data_incorrect = Matrix::new(0.0, 5, 8 + 1); // Incorrect
        let square_norms = vec![0.0; data.nrows()];
        let mut centers = Matrix::new(0.0, 2, 8);
        lloyds_inner(
            data.as_view().into(),
            &square_norms,
            &BlockTranspose::from_matrix_view(data_incorrect.as_view()), // Incorrect
            centers.as_mut_view(),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "data and centers should have the same dimensions")]
    fn lloyds_inner_panics_centers_dim() {
        let data = Matrix::new(0.0, 5, 8);
        let square_norms = vec![0.0; data.nrows()];
        let mut centers = Matrix::new(0.0, 2, 8 + 1); // Incorrect
        lloyds_inner(
            data.as_view().into(),
            &square_norms,
            &BlockTranspose::from_matrix_view(data.as_view()),
            centers.as_mut_view(),
            1,
        );
    }

    ////////////////////
    // Panics - lloyds//
    ////////////////////

    #[test]
    #[should_panic(expected = "data and centers must have the same dimension")]
    fn lloyds_panics_dim_mismatch() {
        let data = Matrix::new(0.0, 5, 8);
        let mut centers = Matrix::new(0.0, 5, 8 + 1); // Incorrect
        lloyds(data.as_view(), centers.as_mut_view(), 1);
    }
}
