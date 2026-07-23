/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{Checker, Input};
use serde::{Deserialize, Serialize};

/// A tolerance [`Input`] for [`diskann_benchmark_runner::benchmark::Regression`]s that
/// do not need any external tolerances.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct Empty;

impl Input for Empty {
    type Raw = Self;

    fn tag() -> &'static str {
        "empty-tolerance"
    }

    fn from_raw(raw: Self::Raw, _: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        Self
    }
}
