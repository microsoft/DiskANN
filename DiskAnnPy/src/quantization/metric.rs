/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::fmt;

use diskann_vector::distance::Metric;
use pyo3::{exceptions::PyValueError, PyResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMetric {
    L2,
    InnerProduct,
    Cosine,
    CosineNormalized,
}

impl QuantizationMetric {
    pub fn parse(name: &str) -> PyResult<Self> {
        match name.to_ascii_lowercase().as_str() {
            "l2" | "l2_squared" | "squared_l2" => Ok(Self::L2),
            "ip" | "inner_product" => Ok(Self::InnerProduct),
            "cosine" => Ok(Self::Cosine),
            "cosine_normalized" | "cosine_norm" => Ok(Self::CosineNormalized),
            other => Err(PyValueError::new_err(format!(
                "unsupported metric '{other}'"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::InnerProduct => "inner_product",
            Self::Cosine => "cosine",
            Self::CosineNormalized => "cosine_normalized",
        }
    }

    pub fn to_vector_metric(self) -> Metric {
        match self {
            Self::L2 => Metric::L2,
            Self::InnerProduct => Metric::InnerProduct,
            Self::Cosine => Metric::Cosine,
            Self::CosineNormalized => Metric::CosineNormalized,
        }
    }
}

impl fmt::Display for QuantizationMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
