/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt;

use diskann_benchmark_runner::Checker;
use diskann_providers::model::graph::provider::DeterminantDiversityParams;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "kebab-case")]
enum RawTopkPostProcessor {
    DeterminantDiversity { power: f32, eta: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "RawTopkPostProcessor", into = "RawTopkPostProcessor")]
pub(crate) enum TopkPostProcessor {
    DeterminantDiversity(DeterminantDiversityParams),
}

impl schemars::JsonSchema for TopkPostProcessor {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        RawTopkPostProcessor::schema_name()
    }

    fn json_schema(generator: &mut schemars::generate::SchemaGenerator) -> schemars::Schema {
        RawTopkPostProcessor::json_schema(generator)
    }
}

impl TryFrom<RawTopkPostProcessor> for TopkPostProcessor {
    type Error = String;

    fn try_from(raw: RawTopkPostProcessor) -> Result<Self, Self::Error> {
        match raw {
            RawTopkPostProcessor::DeterminantDiversity { power, eta } => {
                let params =
                    DeterminantDiversityParams::new(power, eta).map_err(|e| e.to_string())?;
                Ok(Self::DeterminantDiversity(params))
            }
        }
    }
}

impl From<TopkPostProcessor> for RawTopkPostProcessor {
    fn from(value: TopkPostProcessor) -> Self {
        match value {
            TopkPostProcessor::DeterminantDiversity(params) => {
                RawTopkPostProcessor::DeterminantDiversity {
                    power: params.power(),
                    eta: params.eta(),
                }
            }
        }
    }
}

impl TopkPostProcessor {
    pub(crate) fn validate(&mut self, _checker: &mut Checker) -> anyhow::Result<()> {
        Ok(())
    }
}

impl fmt::Display for TopkPostProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopkPostProcessor::DeterminantDiversity(params) => {
                write!(
                    f,
                    "determinant-diversity (power={}, eta={})",
                    params.power(),
                    params.eta()
                )
            }
        }
    }
}
