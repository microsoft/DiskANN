/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::str::FromStr;

use diskann_vector::distance::Metric;
use pyo3::{exceptions, prelude::*};

#[derive(Debug, Clone, Copy, PartialEq)]
#[pyclass(name = "Metric")]
pub enum MetricPy {
    /// Squared Euclidean (L2-Squared)
    L2,

    /// Cosine similarity
    Cosine,

    /// Normalized Cosine Similarity
    CosineNormalized,

    // InnerProduct,
    InnerProduct,
}

#[pymethods]
impl MetricPy {
    #[new]
    pub fn new(metric: String) -> PyResult<Self> {
        Self::from_str(&metric)
            .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e.to_string()))
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ParseMetricError {
    #[error("Invalid format for Metric: {0}")]
    InvalidFormat(String),
}

impl FromStr for MetricPy {
    type Err = ParseMetricError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" => Ok(MetricPy::L2),
            "cosine" => Ok(MetricPy::Cosine),
            "cosinenormalized" => Ok(MetricPy::CosineNormalized),
            "innerproduct" => Ok(MetricPy::InnerProduct),
            _ => Err(ParseMetricError::InvalidFormat(String::from(s))),
        }
    }
}

impl From<Metric> for MetricPy {
    fn from(metric: Metric) -> Self {
        match metric {
            Metric::L2 => MetricPy::L2,
            Metric::Cosine => MetricPy::Cosine,
            Metric::CosineNormalized => MetricPy::CosineNormalized,
            Metric::InnerProduct => MetricPy::InnerProduct,
        }
    }
}

impl From<MetricPy> for Metric {
    fn from(metric: MetricPy) -> Self {
        match metric {
            MetricPy::L2 => Metric::L2,
            MetricPy::Cosine => Metric::Cosine,
            MetricPy::CosineNormalized => Metric::CosineNormalized,
            MetricPy::InnerProduct => Metric::InnerProduct,
        }
    }
}
