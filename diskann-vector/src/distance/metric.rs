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

impl From<Metric> for i32 {
    fn from(metric: Metric) -> Self {
        metric as i32
    }
}

impl TryFrom<i32> for Metric {
    type Error = TryFromMetricError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            x if x == i32::from(Metric::Cosine) => Ok(Metric::Cosine),
            x if x == i32::from(Metric::InnerProduct) => Ok(Metric::InnerProduct),
            x if x == i32::from(Metric::L2) => Ok(Metric::L2),
            x if x == i32::from(Metric::CosineNormalized) => Ok(Metric::CosineNormalized),
            _ => Err(TryFromMetricError(value)),
        }
    }
}

/// Error returned when an `i32` value does not correspond to a valid [`Metric`] variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TryFromMetricError(pub i32);

impl std::fmt::Display for TryFromMetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid Metric discriminant: {}", self.0)
    }
}

impl std::error::Error for TryFromMetricError {}

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

////////////////////////
// diskann-record I/O //
////////////////////////

/// Stable wire names for [`Metric`] variants. Renaming a Rust variant must not change
/// these strings without bumping the saved version, or old manifests will fail to load.
const METRIC_VARIANT_COSINE: &str = "Cosine";
const METRIC_VARIANT_INNER_PRODUCT: &str = "InnerProduct";
const METRIC_VARIANT_L2: &str = "L2";
const METRIC_VARIANT_COSINE_NORMALIZED: &str = "CosineNormalized";

impl diskann_record::save::Save for Metric {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);

    fn save(
        &self,
        _context: diskann_record::save::Context<'_>,
    ) -> diskann_record::save::Result<diskann_record::save::Record<'_>> {
        Ok(diskann_record::save::Record::empty())
    }

    fn variant(&self) -> Option<std::borrow::Cow<'_, str>> {
        Some(
            match self {
                Self::Cosine => METRIC_VARIANT_COSINE,
                Self::InnerProduct => METRIC_VARIANT_INNER_PRODUCT,
                Self::L2 => METRIC_VARIANT_L2,
                Self::CosineNormalized => METRIC_VARIANT_COSINE_NORMALIZED,
            }
            .into(),
        )
    }
}

impl diskann_record::load::Load<'_> for Metric {
    const VERSION: diskann_record::Version = diskann_record::Version::new(0, 0, 0);
    const IS_ENUM: bool = true;

    fn load(
        object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        let variant = object
            .variant()
            .ok_or(diskann_record::load::error::Kind::MissingVariant)?;
        match variant {
            METRIC_VARIANT_COSINE => Ok(Self::Cosine),
            METRIC_VARIANT_INNER_PRODUCT => Ok(Self::InnerProduct),
            METRIC_VARIANT_L2 => Ok(Self::L2),
            METRIC_VARIANT_COSINE_NORMALIZED => Ok(Self::CosineNormalized),
            other => Err(diskann_record::load::Error::message(format!(
                "unknown Metric variant: {other:?}"
            ))),
        }
    }

    fn load_legacy(
        _object: diskann_record::load::Object<'_>,
    ) -> diskann_record::load::Result<Self> {
        Err(diskann_record::load::error::Kind::UnknownVersion.into())
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::{Metric, ParseMetricError, TryFromMetricError};

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

    #[test]
    fn test_metric_to_i32() {
        assert_eq!(i32::from(Metric::Cosine), 0);
        assert_eq!(i32::from(Metric::InnerProduct), 1);
        assert_eq!(i32::from(Metric::L2), 2);
        assert_eq!(i32::from(Metric::CosineNormalized), 3);
    }

    #[test]
    fn test_metric_try_from_i32() {
        assert_eq!(Metric::try_from(0), Ok(Metric::Cosine));
        assert_eq!(Metric::try_from(1), Ok(Metric::InnerProduct));
        assert_eq!(Metric::try_from(2), Ok(Metric::L2));
        assert_eq!(Metric::try_from(3), Ok(Metric::CosineNormalized));
        assert_eq!(Metric::try_from(-1), Err(TryFromMetricError(-1)));
        assert_eq!(Metric::try_from(4), Err(TryFromMetricError(4)));
    }

    #[test]
    fn metric_round_trips_through_record() {
        for metric in [
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::L2,
            Metric::CosineNormalized,
        ] {
            let dir = tempfile::tempdir().expect("tempdir");
            let manifest = dir.path().join("metric.json");
            diskann_record::save::save_to_disk(&metric, dir.path(), &manifest)
                .expect("save_to_disk");
            let restored: Metric =
                diskann_record::load::load_from_disk(&manifest, dir.path())
                    .expect("load_from_disk");
            assert_eq!(metric, restored);
        }
    }

    #[test]
    fn loading_metric_from_unknown_variant_is_rejected() {
        // Save a valid metric, hand-edit the JSON to use an unrecognised variant,
        // and confirm we get a load error mentioning the bogus name.
        let dir = tempfile::tempdir().expect("tempdir");
        let manifest = dir.path().join("metric.json");
        diskann_record::save::save_to_disk(&Metric::L2, dir.path(), &manifest)
            .expect("save_to_disk");

        let raw = std::fs::read_to_string(&manifest).expect("read manifest");
        let tampered = raw.replace("\"L2\"", "\"Bogus\"");
        std::fs::write(&manifest, &tampered).expect("write manifest");

        let err = diskann_record::load::load_from_disk::<Metric>(&manifest, dir.path())
            .expect_err("load should fail for unknown variant");
        let msg = format!("{err}");
        assert!(
            msg.contains("Bogus"),
            "error should embed the unknown variant name, got: {msg}",
        );
    }
}
