/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{CheckDeserialization, Checker};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum TopkPostProcessor {
    DeterminantDiversity {
        #[serde(default = "default_det_div_power")]
        power: f32,
        #[serde(default = "default_det_div_eta")]
        eta: f32,
    },
}

const fn default_det_div_power() -> f32 {
    2.0
}

const fn default_det_div_eta() -> f32 {
    0.01
}

impl CheckDeserialization for TopkPostProcessor {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            TopkPostProcessor::DeterminantDiversity { power, eta } => {
                if *power <= 0.0 {
                    anyhow::bail!(
                        "determinant-diversity power must be > 0.0, got: {}",
                        power
                    );
                }
                if *eta < 0.0 {
                    anyhow::bail!(
                        "determinant-diversity eta must be >= 0.0, got: {}",
                        eta
                    );
                }
                Ok(())
            }
        }
    }
}
