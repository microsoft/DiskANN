/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix};
use thiserror::Error;

/// Policy for processing entries in [`lookup`].
///
/// The lookup operation may be performed by using multiple independent accumulators, each
/// processing a subset of the total lookup operation.
pub trait Lookup<T> {
    /// The type of the accumulator.
    ///
    /// Multiple such accumulators may be instantiated.
    type Accumulator;

    /// The type of the final result.
    type Output;

    /// Create a default accumulator.
    fn default(&self) -> Self::Accumulator;

    /// Accumulate the element `v` with the current accumulator.
    fn accumulate(&self, v: &T, acc: Self::Accumulator) -> Self::Accumulator;

    /// Combine the results of two independent accumulators.
    fn reduce(&self, a: Self::Accumulator, b: Self::Accumulator) -> Self::Accumulator;

    /// Process final accumulator, returning the result.
    fn finish(self, acc: Self::Accumulator) -> Self::Output;
}

/// Use `indices` to retrieve one item from each row of `data` and accumulate the results
/// across all rows using `policy`.
/// ```
/// use diskann_quantization::product::tables::lookup;
/// use diskann_utils::views::rowmajor::{Owned, Matrix};
///
/// // Make the following matrix:
/// // 0 1
/// // 2 3
/// // 4 5
/// let data = Owned::from_fn(3, 2, |rc| 2 * rc.row + rc.col);
/// let sum = lookup::lookup_single(
///     lookup::Sum,
///     data.as_view(),
///     &[0, 1, 0]
/// ).unwrap();
///
/// assert_eq!(sum, 0 + 3 + 4);
/// ```
///
/// # Errors
///
/// Returns an error if `indices.len() != data.nrows()` (there must be a one-to-one
/// correspondence between indices and rows) or if any index in `indices` exceeds `data.ncols()`.
pub fn lookup_single<P, T>(
    policy: P,
    data: rowmajor::Ref<'_, T>,
    indices: &[u8],
) -> Result<P::Output, LookupError>
where
    P: Lookup<T>,
{
    if indices.len() != data.nrows() {
        return Err(LookupError::InvalidLength);
    }

    // Conversion fails if `data.ncols()` is 256 or greater.
    //
    // In this case, all indices will be in-bounds anyways.
    if let Ok(ncols) = u8::try_from(data.ncols())
        && let Some(max) = indices.iter().max()
        && *max >= ncols
    {
        return Err(LookupError::OutOfBounds);
    }

    const UNROLL: usize = 4;

    let mut i = 0;
    let mut a = if data.nrows() >= UNROLL {
        let mut a0 = policy.default();
        let mut a1 = policy.default();
        let mut a2 = policy.default();
        let mut a3 = policy.default();

        while i + 4 <= data.nrows() {
            let v0 = unsafe { data.element_unchecked(i, (*indices.get_unchecked(i)).into()) };
            a0 = policy.accumulate(v0, a0);

            let v1 =
                unsafe { data.element_unchecked(i + 1, (*indices.get_unchecked(i + 1)).into()) };
            a1 = policy.accumulate(v1, a1);

            let v2 =
                unsafe { data.element_unchecked(i + 2, (*indices.get_unchecked(i + 2)).into()) };
            a2 = policy.accumulate(v2, a2);

            let v3 =
                unsafe { data.element_unchecked(i + 3, (*indices.get_unchecked(i + 3)).into()) };
            a3 = policy.accumulate(v3, a3);

            i += UNROLL;
        }

        policy.reduce(policy.reduce(a0, a1), policy.reduce(a2, a3))
    } else {
        policy.default()
    };

    if let Some(remainder) = data.nrows().checked_sub(i) {
        // Hint to LLVM that the loop below is bounded.
        let remainder = remainder.min(UNROLL - 1);
        for j in 0..remainder {
            let k = i + j;
            let v = unsafe { data.element_unchecked(k, (*indices.get_unchecked(k)).into()) };
            a = policy.accumulate(v, a);
        }
    }

    Ok(policy.finish(a))
}

/// Errors from [`lookup`].
#[derive(Debug, Error, Clone, Copy)]
#[non_exhaustive]
pub enum LookupError {
    #[error("number of lookup indices does not match the number of data rows")]
    InvalidLength,
    #[error("at least one of the lookup indices exceeds the number of data columns")]
    OutOfBounds,
}

/// A simple [`Lookup`] that uses `std::ops::Add` to accumulat results.
#[derive(Debug, Clone, Copy)]
pub struct Sum;

impl<T> Lookup<T> for Sum
where
    T: Default + std::ops::Add<Output = T> + Copy,
{
    type Accumulator = T;
    type Output = T;

    fn default(&self) -> T {
        T::default()
    }

    fn accumulate(&self, v: &T, acc: T) -> T {
        *v + acc
    }

    fn reduce(&self, a: T, b: T) -> T {
        a + b
    }

    fn finish(self, acc: T) -> T {
        acc
    }
}

/// An element for [`lookup`] that is used for computing cosine similarity.
///
/// Each [`DotAndNorm`] consists of a partial dot-product (e.g. the dot-product between a
/// query chunks and a PQ center) as well as the PQ center's squared norm.
///
/// After the lookup operation, the final [`DotAndNorm`] consists of the dot-product between
/// the query and the effective data vector as well as the total squared norm of the effective
/// data vector.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DotAndNorm {
    dot: f32,
    square_norm: f32,
}

impl DotAndNorm {
    /// Construct a new [`DotAndNorm`].
    pub const fn new(dot: f32, square_norm: f32) -> Self {
        Self { dot, square_norm }
    }

    /// Return the current value of the dot-product.
    pub fn dot(&self) -> f32 {
        self.dot
    }

    /// Return the current value of the squared norm.
    pub fn square_norm(&self) -> f32 {
        self.square_norm
    }

    /// Finish a cosine computation, using the `query_norm`. This computes:
    /// ```math
    /// 1.0 - (self.dot) / (self.square_norm.sqrt() * query_norm)
    /// ```
    /// taking care to avoid division by zero.
    ///
    /// Note that this returns a [`diskann_vector::SimilarityScore`] for use in similarity
    /// reranking.
    pub fn finish_cosine(&self, query_norm: f32) -> diskann_vector::SimilarityScore<f32> {
        use diskann_vector::SimilarityScore;

        if self.square_norm < f32::MIN_POSITIVE || query_norm < f32::MIN_POSITIVE {
            SimilarityScore::new(1.0)
        } else {
            let v = self.dot / (self.square_norm.sqrt() * query_norm);
            SimilarityScore::new(1.0 - (-1.0f32).max(1.0f32.min(v)))
        }
    }
}

impl std::ops::Add for DotAndNorm {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            dot: self.dot + rhs.dot,
            square_norm: self.square_norm + rhs.square_norm,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_utils::assert_contains;
    use rand::{
        SeedableRng,
        distr::{Distribution, Uniform},
        rngs::StdRng,
    };

    fn expected_sum(x: &[u8]) -> f32 {
        let mut sum = 0.0;
        for (i, v) in x.iter().enumerate() {
            sum += (i as f32) + (*v as f32)
        }
        sum
    }

    #[test]
    fn test_lookup_sum() {
        let ntrials = if cfg!(miri) { 1 } else { 10 };

        let mut rng = StdRng::seed_from_u64(0xd0cc501bde4c9ddd);
        for nrows in 0..12 {
            let mut codes = vec![0u8; nrows];
            for ncols in [1, 2, 255, 256] {
                let dist = if ncols == 0 {
                    Uniform::new(0, 1).unwrap()
                } else {
                    Uniform::new(0, ncols).unwrap()
                };

                let table = rowmajor::Owned::from_fn(nrows, ncols, |rc| (rc.row + rc.col) as f32);

                // If it's possible for a value to be out-of-bounds, make sure we return an
                // error if anything *is* out-of-bounds.
                if ncols < 256 {
                    let nc = u8::try_from(ncols).unwrap();
                    codes.fill(0);
                    for r in 0..nrows {
                        codes[r] = nc;
                        let err = lookup_single(Sum, table.as_view(), &codes).unwrap_err();
                        assert_contains!(
                            err.to_string(),
                            "at least one of the lookup indices exceeds the number of data columns"
                        );
                        codes[r] = 0;
                    }
                }

                // Check that too long and too short codes are detected.
                if nrows > 0 {
                    let too_short = vec![0; nrows - 1];
                    let err = lookup_single(Sum, table.as_view(), &too_short).unwrap_err();
                    assert_contains!(err.to_string(), "number of lookup indices does not match");
                }

                {
                    let too_long = vec![0; nrows + 1];
                    let err = lookup_single(Sum, table.as_view(), &too_long).unwrap_err();
                    assert_contains!(err.to_string(), "number of lookup indices does not match");
                }

                // Test all zeros
                codes.iter_mut().for_each(|c| *c = 0);
                assert_eq!(
                    lookup_single(Sum, table.as_view(), &codes).unwrap(),
                    expected_sum(&codes),
                    "all zeros - nrows = {nrows}, ncols = {ncols}",
                );

                // Test all max
                codes
                    .iter_mut()
                    .for_each(|c| *c = (ncols - 1).try_into().unwrap());
                assert_eq!(
                    lookup_single(Sum, table.as_view(), &codes).unwrap(),
                    expected_sum(&codes),
                    "all max - nrows = {nrows}, ncols = {ncols}",
                );

                for trial in 0..ntrials {
                    codes
                        .iter_mut()
                        .for_each(|c| *c = u8::try_from(dist.sample(&mut rng)).unwrap());

                    println!("codes = {:?}, ncols = {}", codes, ncols);

                    assert_eq!(
                        lookup_single(Sum, table.as_view(), &codes).unwrap(),
                        expected_sum(&codes),
                        "nrows = {nrows}, ncols = {ncols}, trial = {} of {}",
                        trial + 1,
                        ntrials,
                    );
                }
            }
        }
    }
}
