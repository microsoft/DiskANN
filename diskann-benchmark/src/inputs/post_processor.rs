/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{CheckDeserialization, Checker};
use diskann_providers::post_processor::DeterminantDiversityParams;
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
                DeterminantDiversityParams::new(*power, *eta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok(())
            }
        }
    }
}
