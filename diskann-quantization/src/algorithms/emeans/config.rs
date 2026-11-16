/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::EmeansError;

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
            epochs: 5,
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
            .is_some_and(|x| !x.is_finite() || x <= 0.0 || x > 1.0)
        {
            return Err(EmeansError::InvalidConfig("step_size must be in (0, 1]"));
        }
        let quality = match self.batch_size_policy {
            BatchSizePolicy::Auto => d <= 16,
            BatchSizePolicy::Quality => true,
            BatchSizePolicy::Throughput => false,
        };
        let batch_size = self.batch_size.map_or_else(
            || {
                let coefficient = if quality { 0.00439175 } else { 0.035134 };
                let raw = coefficient
                    * (k as f64).powf(0.134749)
                    * (n as f64).powf(0.876473)
                    * (self.epochs as f64).powf(0.813344);
                let cap = if quality {
                    n / 2
                } else {
                    round_power_of_two(n as f64 / 2.0)
                };
                round_power_of_two(raw).min(cap).clamp(1, n)
            },
            |b| b.min(n),
        );
        let initial_step_size = self.step_size.unwrap_or_else(|| {
            (0.8244
                * (k as f64).powf(-0.4617)
                * (batch_size as f64).powf(0.9003)
                * (n as f64).powf(-0.4498)
                * (self.epochs as f64).powf(-0.1057))
            .clamp(1e-4, 1.0)
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
        * (0.1
            + 0.45
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
