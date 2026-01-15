/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::quantizer::ScalarQuantizer;
use crate::{
    num::Positive,
    utils::{compute_means_and_average_norm, compute_variances},
};
use diskann_utils::views;

/// Parameters controlling the generation of the scalar quantization Quantizer.
///
/// When performing scalar quantization, the mean of each dimension will be calculated and
/// the dataset will be shifted around this mean.
///
/// Next, the standard deviation of each dimension will be computed and the maximum `m` found.
///
/// The dynamic range of the final compressed encoding will then span
/// `2 * standard_deviations * m` for each dimension symmetrically about the mean for each
/// dimension. Values outside the spanned dynamic range will be clamped.
pub struct ScalarQuantizationParameters {
    standard_deviations: Positive<f64>,
}

impl ScalarQuantizationParameters {
    /// Construct a new quantizer with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `standard_deviations`: The number of maximal standard deviations to use for the
    ///   encoding's dynamic range. This number **must** be positive, and generally should
    ///   be greater than 1.0.
    ///
    ///   A good starting value is generally 2.0.
    pub fn new(standard_deviations: Positive<f64>) -> Self {
        Self {
            standard_deviations,
        }
    }

    /// Return the current number of standard deviations being used to set the dynamic range.
    pub fn standard_deviations(&self) -> Positive<f64> {
        self.standard_deviations
    }

    /// Train a new [`ScalarQuantizer`] on the input training data.
    ///
    /// The training algoritm works as follows:
    ///
    /// 1. The medoid of the training data is computed.
    ///
    /// 2. The standard deviation for each dimension is then calculated across all rows
    ///    of the training set.
    ///
    /// 3. The maximum standard deviation `s` is computed and the dynamic range `dyn` of the
    ///    quantizer is computed as `dyn = 2.0 * self.standard_deviations() * s`.
    ///
    /// 4. The quantizer is then constructed with `scale = dyn / (2.pow(NBITS) - 1)`.
    ///
    /// # Complexity
    ///
    /// This method is linear in the number of rows and columns in `data`.
    ///
    /// # Allocates
    ///
    /// This method allocated memory on the order of `data.ncols()` (the dimensionality of
    /// the data).
    ///
    /// # Parallelism
    ///
    /// This function is single threaded.
    pub fn train<T>(&self, data: views::MatrixView<T>) -> ScalarQuantizer
    where
        T: Copy + Into<f64> + Into<f32>,
    {
        let (means, mean_norm) = compute_means_and_average_norm(data);
        let variances = compute_variances(data, &means);

        // Take the maximum variance - that will set our global scaling parameter.
        let max_std = variances.iter().fold(0.0f64, |max, &x| max.max(x)).sqrt();
        let p = max_std * self.standard_deviations.into_inner();

        let scale = 2.0 * p;
        let shift = means.into_iter().map(|i| (i - p) as f32).collect();

        ScalarQuantizer::new(scale as f32, shift, Some(mean_norm as f32))
    }
}

// 2.0 seems to be good starting point for scalar quantization.
//
// SAFETY: 2.0 is greater than 0.0.
const DEFAULT_STDEV: Positive<f64> = unsafe { Positive::new_unchecked(2.0) };

impl Default for ScalarQuantizationParameters {
    fn default() -> Self {
        Self::new(DEFAULT_STDEV)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::test_util::create_test_problem;

    fn test_train_impl(nrows: usize, ncols: usize, seed: u64) {
        // Test Default
        let default = ScalarQuantizationParameters::default();
        assert_eq!(default.standard_deviations(), DEFAULT_STDEV);

        let mut rng = StdRng::seed_from_u64(seed);
        let problem = create_test_problem(nrows, ncols, &mut rng);

        // Compute the maximum standard deviation from the expected variances.
        let problem_std_max = problem
            .variances
            .iter()
            .copied()
            .reduce(|a, b| a.max(b))
            .unwrap()
            .sqrt();

        // Provide a range of standard deviation requests to the training algoritm.
        let standard_deviations: [f64; 3] = [1.0, 1.5, 2.0];
        for std in standard_deviations {
            let parameters = ScalarQuantizationParameters::new(Positive::new(std).unwrap());

            let quantizer = parameters.train(problem.data.as_view());
            assert_eq!(quantizer.dim(), ncols);

            let expected_scale = std * 2.0 * problem_std_max;
            let got_scale = quantizer.scale();

            let relative_diff = (got_scale as f64 - expected_scale) / expected_scale;

            assert!(
                relative_diff < 1.0e-7,
                "Relative difference in scaling of {}. Got {}, expected {} \
                 (nrows = {}, ncols = {})",
                relative_diff,
                got_scale,
                expected_scale,
                nrows,
                ncols
            );

            assert_eq!(quantizer.mean_norm().unwrap(), problem.mean_norm as f32);

            // The quantizer shift should be the dataset mean shifted by the appropriate
            // amount for the unsigned quantization.
            let shift = std * problem_std_max;
            let quantizer_shift = quantizer.shift();
            for (i, (&got, &expected)) in
                std::iter::zip(quantizer_shift.iter(), problem.means.iter()).enumerate()
            {
                let expected = expected - shift;

                assert_eq!(
                    got, expected as f32,
                    "Mismatch in shift amount at index {}, (nrows = {}, ncols = {})",
                    i, nrows, ncols,
                );
            }
        }
    }

    #[test]
    fn test_train() {
        test_train_impl(10, 16, 0x0b1d3ccb952d3079);
        test_train_impl(7, 8, 0xda9a5c0a672f43cd);
    }
}
