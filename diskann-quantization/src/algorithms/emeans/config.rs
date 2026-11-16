/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::EmeansError;

// Versioned empirical recipe, not parameters fitted by DiskANN:
// https://github.com/facebookresearch/faiss/blob/65a4e3880ece1494c9d79e06f886c89810a1f85a/contrib/e_means/base.py
// Faiss reports full-pass batch fits on 100K-10M rows; the step fit also includes
// 100M rows and was fitted with a zero-ending cosine. These shipped coefficients
// differ from Table 2 of arXiv:2611.09501v1; execution uses the nonzero floor below.

// B = C_B * k^b_k * n^b_n * T^b_T: k centers, n samples, T epochs.
// The quality recipe divides the raw batch by eight for more frequent updates;
// rounding and the distinct caps can change the ratio of the final batch sizes.
const BATCH_SIZE_THROUGHPUT_COEFFICIENT: f64 = 0.035134;
const BATCH_SIZE_QUALITY_COEFFICIENT: f64 = BATCH_SIZE_THROUGHPUT_COEFFICIENT / 8.0;
const BATCH_SIZE_CLUSTER_EXPONENT: f64 = 0.134749;
const BATCH_SIZE_SAMPLE_EXPONENT: f64 = 0.876473;
const BATCH_SIZE_EPOCH_EXPONENT: f64 = 0.813344;

// alpha0 = C_alpha * k^a_k * B^a_B * n^a_n * T^a_T, using the resolved B.
// alpha weights incoming EMA counts/sums; it is not the centroid interpolation weight.
const INITIAL_STEP_COEFFICIENT: f64 = 0.8244;
const INITIAL_STEP_CLUSTER_EXPONENT: f64 = -0.4617;
const INITIAL_STEP_BATCH_EXPONENT: f64 = 0.9003;
const INITIAL_STEP_SAMPLE_EXPONENT: f64 = -0.4498;
const INITIAL_STEP_EPOCH_EXPONENT: f64 = -0.1057;

// Reference API budget, independently overridable by the caller.
const DEFAULT_EPOCHS: usize = 5;
// Faiss's low-dimensional ablations favor more frequent updates at d <= 16.
const AUTO_QUALITY_MAX_DIMENSION: usize = 16;
// Preserve Faiss's two caps based on half the dataset: quality floors this value,
// throughput rounds it to a power of two (so its cap can exceed n/2).
const AUTO_BATCH_CAP_DIVISOR: usize = 2;
// The lower clamp belongs to the automatic recipe only. The upper bound keeps
// both EMA weights, alpha and 1-alpha, nonnegative, including explicit overrides.
const MIN_AUTO_STEP_SIZE: f64 = 1e-4;
const MAX_STEP_SIZE: f64 = 1.0;

// Equation 5 anneals toward one tenth of the initial step. The half-range cosine
// weight is derived from that floor rather than an independent fitted parameter.
const COSINE_MIN_STEP_FACTOR: f64 = 0.1;
const COSINE_STEP_WEIGHT: f64 = (1.0 - COSINE_MIN_STEP_FACTOR) / 2.0;

/// Scaling law for automatic batch-size selection from the dataset size, cluster count
/// and epoch budget. Ignored when [EmeansConfig::batch_size] is supplied explicitly.
/// The resolved batch size also determines the default [EmeansConfig::step_size].
#[derive(Debug, Clone, Copy, Default)]
pub enum BatchSizePolicy {
    /// Quality for dimensions <= 16, throughput otherwise, as in Faiss.
    #[default]
    Auto,
    /// Favor larger batches to amortize centroid-update overhead.
    Throughput,
    /// Use one eighth of the throughput law's coefficient for more frequent updates.
    /// Rounding and caps mean the final batch is not necessarily eight times smaller.
    Quality,
}

/// Training budget and optional overrides of the automatic recipe.
#[derive(Debug, Clone)]
pub struct EmeansConfig {
    /// Positive number of complete passes; default: 5.
    pub epochs: usize,
    /// Positive batch size, capped at the dataset size. None uses a scaling law.
    pub batch_size: Option<usize>,
    /// Initial EMA step in (0, 1]. None uses the step scaling law.
    pub step_size: Option<f64>,
    /// Policy for the batch scaling law.
    pub batch_size_policy: BatchSizePolicy,
    /// Enable mass-proportional splitting every five updates (only when d > 2).
    pub revive: bool,
}

impl Default for EmeansConfig {
    fn default() -> Self {
        Self {
            epochs: DEFAULT_EPOCHS,
            batch_size: None,
            step_size: None,
            batch_size_policy: BatchSizePolicy::Auto,
            revive: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ResolvedParameters {
    pub(super) batch_size: usize,
    pub(super) initial_step_size: f64,
    pub(super) total_updates: usize,
}

impl EmeansConfig {
    pub(super) fn resolve_parameters(
        &self,
        n: usize,
        k: usize,
        d: usize,
    ) -> Result<ResolvedParameters, EmeansError> {
        if self.epochs == 0 {
            return Err(EmeansError::InvalidConfig("epochs must be positive"));
        }
        if self.batch_size == Some(0) {
            return Err(EmeansError::InvalidConfig("batch_size must be positive"));
        }
        if self
            .step_size
            .is_some_and(|x| !x.is_finite() || x <= 0.0 || x > MAX_STEP_SIZE)
        {
            return Err(EmeansError::InvalidConfig("step_size must be in (0, 1]"));
        }
        let quality = match self.batch_size_policy {
            BatchSizePolicy::Auto => d <= AUTO_QUALITY_MAX_DIMENSION,
            BatchSizePolicy::Quality => true,
            BatchSizePolicy::Throughput => false,
        };
        let batch_size = self.batch_size.map_or_else(
            || {
                let coefficient = if quality {
                    BATCH_SIZE_QUALITY_COEFFICIENT
                } else {
                    BATCH_SIZE_THROUGHPUT_COEFFICIENT
                };
                let raw = coefficient
                    * (k as f64).powf(BATCH_SIZE_CLUSTER_EXPONENT)
                    * (n as f64).powf(BATCH_SIZE_SAMPLE_EXPONENT)
                    * (self.epochs as f64).powf(BATCH_SIZE_EPOCH_EXPONENT);
                let cap = if quality {
                    n / AUTO_BATCH_CAP_DIVISOR
                } else {
                    round_power_of_two(n as f64 / AUTO_BATCH_CAP_DIVISOR as f64)
                };
                round_power_of_two(raw).min(cap).clamp(1, n)
            },
            |b| b.min(n),
        );
        let initial_step_size = self.step_size.unwrap_or_else(|| {
            (INITIAL_STEP_COEFFICIENT
                * (k as f64).powf(INITIAL_STEP_CLUSTER_EXPONENT)
                * (batch_size as f64).powf(INITIAL_STEP_BATCH_EXPONENT)
                * (n as f64).powf(INITIAL_STEP_SAMPLE_EXPONENT)
                * (self.epochs as f64).powf(INITIAL_STEP_EPOCH_EXPONENT))
            .clamp(MIN_AUTO_STEP_SIZE, MAX_STEP_SIZE)
        });
        let total_updates = n
            .div_ceil(batch_size)
            .checked_mul(self.epochs)
            .ok_or(EmeansError::SizeOverflow)?;
        Ok(ResolvedParameters {
            batch_size,
            initial_step_size,
            total_updates,
        })
    }
}

fn round_power_of_two(value: f64) -> usize {
    2.0_f64.powf(value.max(1.0).log2().round()) as usize
}

pub(super) fn scheduled_step_size(
    initial_step_size: f64,
    update_index: usize,
    total_updates: usize,
) -> f64 {
    initial_step_size
        * (COSINE_MIN_STEP_FACTOR
            + COSINE_STEP_WEIGHT
                * (1.0 + (std::f64::consts::PI * update_index as f64 / total_updates as f64).cos()))
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn assert_near(got: f64, expected: f64, tolerance: f64) {
        assert!(
            (got - expected).abs() <= tolerance,
            "expected {expected}, got {got}, tolerance {tolerance}"
        );
    }

    #[test]
    fn published_recipe_and_cosine_schedule() -> TestResult {
        let throughput = EmeansConfig {
            epochs: 25,
            batch_size_policy: BatchSizePolicy::Throughput,
            ..Default::default()
        };
        let quality = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Quality,
            ..throughput.clone()
        };
        let recipe = throughput.resolve_parameters(1_000_000, 16_384, 128)?;
        assert_eq!(recipe.batch_size, 262_144);
        assert_eq!(recipe.total_updates, 100);
        assert_eq!(recipe.initial_step_size, 1.0);
        let recipe = quality.resolve_parameters(1_000_000, 16_384, 128)?;
        assert_eq!(recipe.batch_size, 32_768);
        assert_near(recipe.initial_step_size, 0.1545388281304627, 1e-14);
        let recipe = quality.resolve_parameters(100_000_000, 262_144, 128)?;
        assert_eq!(recipe.batch_size, 4_194_304);
        assert_near(recipe.initial_step_size, 0.42718555053293517, 1e-14);
        assert_near(scheduled_step_size(0.4, 0, 10), 0.4, 1e-15);
        assert_near(scheduled_step_size(0.4, 9, 10), 0.04880982706687237, 1e-15);
        for step in 1..10 {
            assert!(scheduled_step_size(0.4, step, 10) < scheduled_step_size(0.4, step - 1, 10));
        }
        Ok(())
    }
    #[test]
    fn automatic_policy_caps_rounding_and_explicit_overrides() -> TestResult {
        let config = EmeansConfig::default();
        let quality = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Quality,
            ..config.clone()
        };
        let throughput = EmeansConfig {
            batch_size_policy: BatchSizePolicy::Throughput,
            ..config.clone()
        };
        assert_eq!(
            config.resolve_parameters(100_000, 256, 16)?.batch_size,
            quality.resolve_parameters(100_000, 256, 16)?.batch_size
        );
        assert_eq!(
            config.resolve_parameters(100_000, 256, 17)?.batch_size,
            throughput.resolve_parameters(100_000, 256, 17)?.batch_size
        );
        assert_eq!(round_power_of_two(5.0), 4);
        assert_eq!(round_power_of_two(6.0), 8);
        assert_eq!(round_power_of_two(0.5), 1);
        let capped = EmeansConfig {
            epochs: 100_000,
            ..quality
        };
        assert_eq!(capped.resolve_parameters(11, 3, 5)?.batch_size, 5);
        let capped = EmeansConfig {
            epochs: 100_000,
            ..throughput
        };
        assert_eq!(capped.resolve_parameters(11, 3, 5)?.batch_size, 4);
        let explicit = EmeansConfig {
            epochs: 3,
            batch_size: Some(usize::MAX),
            step_size: Some(0.75),
            ..config
        }
        .resolve_parameters(11, 3, 5)?;
        assert_eq!(explicit.batch_size, 11);
        assert_eq!(explicit.initial_step_size, 0.75);
        assert_eq!(explicit.total_updates, 3);
        Ok(())
    }
}
