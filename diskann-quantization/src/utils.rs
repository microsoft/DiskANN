/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::ptr::NonNull;

use thiserror::Error;

use diskann_utils::views::MatrixView;

/// Specify featres and config flags that will be propagated to `docsrs` config.
macro_rules! features {
    (
        #![$meta:meta]
        $($item:item)*
    ) => {
        $(
            #[cfg($meta)]
            #[cfg_attr(docsrs, doc(cfg($meta)))]
            $item
        )*
    }
}

pub(crate) use features;

/// Return a `NonNull<T>` pointer to the first element of `slice`.
pub(crate) fn as_nonnull<T>(slice: &[T]) -> NonNull<T> {
    // SAFETY: Slices guarantee non-null pointers.
    unsafe { NonNull::new_unchecked(slice.as_ptr().cast_mut()) }
}

/// Return a `NonNull<T>` pointer to the first element of `slice`.
pub(crate) fn as_nonnull_mut<T>(slice: &mut [T]) -> NonNull<T> {
    // SAFETY: Slices guarantee non-null pointers.
    unsafe { NonNull::new_unchecked(slice.as_mut_ptr()) }
}

/// Perform the computation `ceil(x / y)` where `x`, `y`, and the returned value are all
/// integers.
///
/// # Panics
///
/// Probably panics if `y == T::default()` as Rust's default `x / y` operator is used.
/// For integers, this operation panics upon division by zero.
pub(crate) fn div_round_up<T>(x: T, y: T) -> T
where
    T: std::ops::Div<Output = T>
        + std::ops::Rem<Output = T>
        + std::ops::Add<Output = T>
        + Default
        + std::cmp::Eq
        + From<u8>
        + Copy,
{
    (x / y)
        + if (x % y) != T::default() {
            T::from(1)
        } else {
            T::default()
        }
}

/// Return `true` if the iterator contains elements that are strictly increasing.
///
/// Otherwise, return false.
pub(crate) fn is_strictly_monotonic<I>(mut itr: I) -> bool
where
    I: Iterator,
    I::Item: Ord,
{
    let mut x = match itr.next() {
        // Empty iterators are monotonic.
        None => {
            return true;
        }
        Some(x) => x,
    };

    for y in itr {
        if y <= x {
            return false;
        }
        x = y;
    }
    true
}

#[derive(Debug, Clone, Copy, Error)]
#[non_exhaustive]
#[error("data cannot be empty")]
pub(crate) struct CannotBeEmpty;

/// Compute the mean of each column in `data` as well as the average norm.
pub(crate) fn compute_means_and_average_norm<T>(data: MatrixView<T>) -> (Vec<f64>, f64)
where
    T: Into<f64> + Copy,
{
    // Compute the centroid of the dataset as well as the sums of the norms of every
    // element in the dataset.
    let mut means: Vec<f64> = vec![0.0; data.ncols()];
    let norm_sum = data.row_iter().fold(0.0f64, |accum: f64, row| {
        // Accumulate this row into the means.
        std::iter::zip(means.iter_mut(), row.iter()).for_each(|(m, r)| {
            let r: f64 = (*r).into();
            *m += r;
        });

        // Accumulate this vector's norm.
        let norm = row
            .iter()
            .map(|&x| {
                let x: f64 = x.into();
                let y = x;
                y * y
            })
            .sum::<f64>()
            .sqrt();

        accum + norm
    });
    let mean_norm: f64 = norm_sum / (data.nrows() as f64);

    // Divide each mean accumulator by the number of rows to obtain the true mean.
    means.iter_mut().for_each(|m| *m /= data.nrows() as f64);
    (means, mean_norm)
}

/// Compute the mean of each column in `data` as well as the average norm.
pub(crate) fn compute_normalized_means<T>(data: MatrixView<T>) -> Result<Vec<f64>, CannotBeEmpty>
where
    T: Into<f64> + Copy,
{
    if data.nrows() == 0 {
        return Err(CannotBeEmpty);
    }
    if data.ncols() == 0 {
        return Ok(Vec::new());
    }

    // Compute the centroid of the dataset as well as the sums of the norms of every
    // element in the dataset.
    let mut means: Vec<f64> = vec![0.0; data.ncols()];

    let square = |x: &T| -> f64 {
        let x: f64 = (*x).into();
        x * x
    };

    data.row_iter().for_each(|row| {
        let norm = row.iter().map(square).sum::<f64>().sqrt();
        let inv_norm = if norm == 0.0 { 1.0 } else { 1.0 / norm };

        // Accumulate this row into the means.
        std::iter::zip(means.iter_mut(), row.iter()).for_each(|(m, r)| {
            let r: f64 = (*r).into() * inv_norm;
            *m += r;
        });
    });

    // Divide each mean accumulator by the number of rows to obtain the true mean.
    means.iter_mut().for_each(|m| *m /= data.nrows() as f64);
    Ok(means)
}

pub(crate) fn compute_variances<T>(data: MatrixView<T>, means: &[f64]) -> Vec<f64>
where
    T: Into<f64> + Copy,
{
    assert_eq!(data.ncols(), means.len());

    let mut variances: Vec<f64> = vec![0.0; data.ncols()];
    data.row_iter().for_each(|row| {
        variances
            .iter_mut()
            .zip(std::iter::zip(row.iter(), means.iter()))
            .for_each(|(v, (r, c))| {
                let r: f64 = (*r).into();
                let d: f64 = r - *c;
                *v += d * d;
            });
    });

    variances.iter_mut().for_each(|v| *v /= data.nrows() as f64);
    variances
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use diskann_utils::views::Matrix;
    use diskann_vector::{norm::FastL2Norm, Norm};
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::test_util::create_test_problem;

    fn normalize(x: &mut [f32]) {
        let norm: f32 = (FastL2Norm).evaluate(&*x);
        x.iter_mut().for_each(|i| *i /= norm);
    }

    // Test the `means` part of `compute_means_and_average_norm`.
    //
    // The strategy here is to generate a small corpus with a known mean for each dimension.
    fn check_on_test_problem(nrows: usize, ncols: usize, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let test_problem = create_test_problem(nrows, ncols, &mut rng);
        let (means, mean_norm) = compute_means_and_average_norm(test_problem.data.as_view());

        for (i, (&expected, &got)) in
            std::iter::zip(test_problem.means.iter(), means.iter()).enumerate()
        {
            assert_eq!(
                expected, got,
                "computed mean failed for dim {} (nrows = {}, ncols = {})",
                i, nrows, ncols
            );
        }

        // Make sure the mean norm is computed properly.
        assert_eq!(
            test_problem.mean_norm, mean_norm,
            "mean norm computation failed (nrows = {}, ncols = {})",
            nrows, ncols
        );

        // Now check that variances are computed properly.
        let variances = compute_variances(test_problem.data.as_view(), &means);
        println!("expected = {:?}", test_problem.variances);
        println!("got = {:?}", variances);

        for (i, (&expected, &got)) in
            std::iter::zip(test_problem.variances.iter(), variances.iter()).enumerate()
        {
            assert_eq!(
                expected, got,
                "computed variance failed for dim {} (nrows = {}, ncols = {})",
                i, nrows, ncols
            );
        }
    }

    #[test]
    fn test_mean_and_norm() {
        check_on_test_problem(10, 16, 0x1d66d1dbf1cfedf4);
        check_on_test_problem(11, 7, 0x571fdc0150fedc24);
    }

    fn check_normalized_on_test_problem(nrows: usize, ncols: usize, seed: u64) {
        const ERROR_BOUND: f64 = 1.0e-6;

        let mut rng = StdRng::seed_from_u64(seed);
        let test_problem = create_test_problem(nrows, ncols, &mut rng);

        let mut normalized_data = test_problem.data.clone();
        normalized_data.row_iter_mut().for_each(normalize);

        // Compute the means and mean norm of the normalized data.
        //
        // The expected `mean_norm` should be close to 1.0
        let (expected_means, expected_mean_norm) =
            compute_means_and_average_norm(normalized_data.as_view());

        let norm_error = (expected_mean_norm - 1.0).abs();
        assert!(norm_error < 1e-7, "got a norm error of {}", norm_error);

        let means = compute_normalized_means(test_problem.data.as_view()).unwrap();

        for (i, (&expected, &got)) in
            std::iter::zip(expected_means.iter(), means.iter()).enumerate()
        {
            let error = if got == 0.0 && expected == 0.0 {
                0.0
            } else {
                (got - expected).abs() / expected.abs()
            };

            assert!(
                error < ERROR_BOUND,
                "got {}, expected {}, error = {}, bound = {} for dim {} (nrows = {}, ncols = {})",
                got,
                expected,
                error,
                ERROR_BOUND,
                i,
                nrows,
                ncols
            );
        }
    }

    #[test]
    fn test_normalized_mean_and_norm() {
        check_normalized_on_test_problem(1, 20, 0x81daae2ac2d06d7c);
        check_normalized_on_test_problem(10, 16, 0x1d66d1dbf1cfedf4);
        check_normalized_on_test_problem(11, 7, 0x571fdc0150fedc24);
    }

    #[test]
    fn test_normalized_means_corner_cases() {
        // If the input data has no columns, the returned vector should be empty.
        let data = Matrix::new(1.0f32, 10, 0);
        let means = compute_normalized_means(data.as_view()).unwrap();
        assert!(means.is_empty());

        // If the data has no rows, an error should be returned.
        let data = Matrix::new(1.0f32, 0, 10);
        let _: CannotBeEmpty = compute_normalized_means(data.as_view()).unwrap_err();
    }

    #[test]
    fn test_div_round_up() {
        assert_eq!(div_round_up(10, 2), 5);
        assert_eq!(div_round_up(10, 3), 4);
        assert_eq!(div_round_up(10, 4), 3);
        assert_eq!(div_round_up(10, 5), 2);
        assert_eq!(div_round_up(10, 6), 2);
        assert_eq!(div_round_up(10, 7), 2);
        assert_eq!(div_round_up(10, 8), 2);
        assert_eq!(div_round_up(10, 9), 2);
        assert_eq!(div_round_up(10, 10), 1);
        assert_eq!(div_round_up(10, 11), 1);
    }

    #[test]
    fn test_is_strictly_monotonic() {
        // Success Cases

        let x: &[usize] = &[];
        assert!(
            is_strictly_monotonic(x.iter()),
            "empty ranges are monotonic"
        );

        let x: &[usize] = &[100];
        assert!(
            is_strictly_monotonic(x.iter()),
            "ranges of length 1 are monotonic"
        );

        let x: &[usize] = &[100, 101];
        assert!(is_strictly_monotonic(x.iter()));

        let x: &[usize] = &[100, 102, 104, 105];
        assert!(is_strictly_monotonic(x.iter()));

        // Failure Cases
        let x: &[usize] = &[100, 100, 105];
        assert!(!is_strictly_monotonic(x.iter()));

        let x: &[usize] = &[100, 90, 105];
        assert!(!is_strictly_monotonic(x.iter()));

        let x: &[usize] = &[100, 105, 105];
        assert!(!is_strictly_monotonic(x.iter()));

        let x: &[usize] = &[100, 105, 100];
        assert!(!is_strictly_monotonic(x.iter()));
    }
}
