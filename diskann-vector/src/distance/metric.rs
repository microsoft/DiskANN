/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]
use std::str::FromStr;

#[repr(C)]
/// Distance metric
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Metric {
    /// Cosine similarity
    Cosine,
    /// Inner product
    InnerProduct,
    /// Squared Euclidean (L2-Squared)
    L2,
    /// Normalized Cosine Similarity
    CosineNormalized,
}

impl Metric {
    /// Returns the string representation of the metric.
    pub const fn as_str(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::InnerProduct => "innerproduct",
            Metric::L2 => "l2",
            Metric::CosineNormalized => "cosinenormalized",
        }
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug)]
pub enum ParseMetricError {
    InvalidFormat(String),
}

impl std::fmt::Display for ParseMetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(str) => write!(f, "Invalid format for Metric: {}", str),
        }
    }
}

impl std::error::Error for ParseMetricError {}

impl FromStr for Metric {
    type Err = ParseMetricError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            x if x == Metric::L2.as_str() => Ok(Metric::L2),
            x if x == Metric::Cosine.as_str() => Ok(Metric::Cosine),
            x if x == Metric::InnerProduct.as_str() => Ok(Metric::InnerProduct),
            x if x == Metric::CosineNormalized.as_str() => Ok(Metric::CosineNormalized),
            _ => Err(ParseMetricError::InvalidFormat(String::from(s))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{Metric, ParseMetricError};

    #[test]
    fn test_metric_from_str() {
        assert_eq!(Metric::from_str("cosine").unwrap(), Metric::Cosine);
        assert_eq!(Metric::from_str("l2").unwrap(), Metric::L2);
        assert_eq!(
            Metric::from_str("innerproduct").unwrap(),
            Metric::InnerProduct
        );
        assert_eq!(
            Metric::from_str("cosinenormalized").unwrap(),
            Metric::CosineNormalized
        );
        assert_eq!(
            Metric::from_str("invalid").unwrap_err().to_string(),
            ParseMetricError::InvalidFormat(String::from("invalid")).to_string()
        );
    }
}
