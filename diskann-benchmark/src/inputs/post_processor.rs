/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{CheckDeserialization, Checker};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum TopkPostProcessor {
    DeterminantDiversity { power: f32, eta: f32 },
}

impl CheckDeserialization for TopkPostProcessor {
    fn check_deserialization(&mut self, _checker: &mut Checker) -> Result<(), anyhow::Error> {
        match self {
            TopkPostProcessor::DeterminantDiversity { power, eta } => {
                if *power <= 0.0 {
                    anyhow::bail!("determinant-diversity power must be > 0.0, got: {}", power);
                }
                if *eta < 0.0 {
                    anyhow::bail!("determinant-diversity eta must be >= 0.0, got: {}", eta);
                }
                Ok(())
            }
        }
    }
}
