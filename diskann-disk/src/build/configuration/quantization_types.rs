/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, fmt::Display, str::FromStr};

use diskann_quantization::num::Positive;
use serde::{
    de::{self, Visitor},
    Deserializer, Serializer,
};
use thiserror::Error;

const EXPECTED_QUANTIZATION_TEMPLATE: &str =
    "Expected 'FP', 'PQ_N', 'SQ_NBITS'(STDDEV=2.0 by default), 'SQ_NBITS_STDDEV', got: ";

/// - `PQ`: Product Quantization with a specified number of chunks.
/// - `SQ`: Scalar Quantization with a specified number of bits per dimension and a standard deviation.
///   The `Default` implementation returns `FP` (Full Precision).
#[derive(Debug, Copy, Default, PartialEq, Clone)]
pub enum QuantizationType {
    /// No quantization - uses full precision
    #[default]
    FP,

    /// Product Quantization (PQ)
    PQ {
        /// Number of PQ chunks
        num_chunks: usize,
    },

    SQ {
        /// Number of bits per dimension for
        nbits: usize,
        /// The number of maximal standard deviations to use for the
        /// encoding's dynamic range. This number **must** be positive, and generally should
        /// be greater than 1.0.
        standard_deviation: Option<Positive<f64>>,
    },
}

#[derive(Debug, Error)]
#[error("Invalid quantization type: {0:?}")]
pub struct QuantizationTypeParseError(String);

impl serde::Serialize for QuantizationType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for QuantizationType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct QuantizationTypeVisitor;

        impl Visitor<'_> for QuantizationTypeVisitor {
            type Value = QuantizationType;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a string like \"PQ_192\" or \"FP\"")
            }

            fn visit_str<E>(self, value: &str) -> Result<QuantizationType, E>
            where
                E: de::Error,
            {
                QuantizationType::from_str(value).map_err(E::custom)
            }
        }

        deserializer.deserialize_str(QuantizationTypeVisitor)
    }
}

impl FromStr for QuantizationType {
    type Err = QuantizationTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let clean_s = s.trim().to_lowercase();
        let parts = clean_s.split('_').collect::<Vec<_>>();

        match parts[0] {
            "fp" => {
                if parts.len() != 1 {
                    return Err(QuantizationTypeParseError(format!(
                        "{} {}",
                        EXPECTED_QUANTIZATION_TEMPLATE, s
                    )));
                };
                Ok(QuantizationType::FP)
            }
            "pq" => {
                if parts.len() != 2 {
                    return Err(QuantizationTypeParseError(format!(
                        "{} {}",
                        EXPECTED_QUANTIZATION_TEMPLATE, s
                    )));
                }
                let num_chunks = parts[1].parse::<usize>().map_err(|_| {
                    QuantizationTypeParseError(format!("{} {}", EXPECTED_QUANTIZATION_TEMPLATE, s))
                })?;
                if num_chunks == 0 {
                    return Err(QuantizationTypeParseError(
                        "The num_chunks should be more than 0 for PQ".to_string(),
                    ));
                }
                Ok(QuantizationType::PQ { num_chunks })
            }
            "sq" => {
                if parts.len() < 2 || parts.len() > 3 {
                    return Err(QuantizationTypeParseError(format!(
                        "{} {}",
                        EXPECTED_QUANTIZATION_TEMPLATE, s
                    )));
                }

                let nbits = parts[1].parse::<usize>().map_err(|_| {
                    QuantizationTypeParseError(format!("{} {}", EXPECTED_QUANTIZATION_TEMPLATE, s))
                })?;
                let standard_deviation = if parts.len() == 2 {
                    None
                } else {
                    let value = parts[2].parse::<f64>().map_err(|_| {
                        QuantizationTypeParseError(format!(
                            "{} {}",
                            EXPECTED_QUANTIZATION_TEMPLATE, s
                        ))
                    })?;
                    Some(Positive::new(value).map_err(|_| {
                        QuantizationTypeParseError(format!(
                            "{} {}",
                            EXPECTED_QUANTIZATION_TEMPLATE, s
                        ))
                    })?)
                };

                Ok(QuantizationType::SQ {
                    nbits,
                    standard_deviation,
                })
            }
            _ => Err(QuantizationTypeParseError(format!(
                "{} {}",
                EXPECTED_QUANTIZATION_TEMPLATE, s
            ))),
        }
    }
}

impl Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationType::PQ { num_chunks } => write!(f, "PQ_{}", num_chunks),
            QuantizationType::FP => write!(f, "FP"),
            QuantizationType::SQ {
                nbits,
                standard_deviation,
            } => {
                let standard_deviation = match standard_deviation {
                    Some(sd) => sd.into_inner().to_string(),
                    None => "None".to_string(),
                };
                write!(f, "SQ_{}_{}", nbits, standard_deviation)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("PQ_128", Ok(QuantizationType::PQ { num_chunks: 128 }))]
    #[case("FP", Ok(QuantizationType::FP))]
    #[case("pq_64", Ok(QuantizationType::PQ { num_chunks: 64 }))]
    #[case("SQ_1_2.0", Ok(QuantizationType::SQ {
        nbits: 1,
        standard_deviation: Some(Positive::new(2.0).unwrap())
    }))]
    #[case("SQ_1", Ok(QuantizationType::SQ {
        nbits: 1,
        standard_deviation: None
    }))]
    #[case("", Err(()))]
    #[case("FP_1", Err(()))]
    #[case("invalid", Err(()))]
    #[case("PQ_abc", Err(()))]
    #[case("PQ128", Err(()))]
    #[case("PQ_4_", Err(()))]
    #[case("PQ_0", Err(()))]
    #[case("SQ_2.0", Err(()))]
    #[case("SQ_1_-1", Err(()))]
    #[case("SQ_1_2.0_3", Err(()))]
    #[case("SQ_1_abc", Err(()))]
    fn test_parse_quantization_type(
        #[case] input: &str,
        #[case] expected: Result<QuantizationType, ()>,
    ) {
        let result = input.parse::<QuantizationType>();

        match expected {
            Ok(expected_value) => assert_eq!(result.unwrap(), expected_value),
            Err(_) => assert!(result.is_err(), "Expected parse error for input: {}", input),
        }
    }

    #[test]
    fn test_default_implementation() {
        // Test the Default implementation
        let default_quant = QuantizationType::default();
        assert_eq!(default_quant, QuantizationType::FP);
    }

    #[rstest]
    #[case(QuantizationType::PQ { num_chunks: 16 }, "PQ_16")]
    #[case(QuantizationType::FP, "FP")]
    #[case(
        QuantizationType::SQ { nbits: 8, standard_deviation: None },
        "SQ_8_None"
    )]
    #[case(
        QuantizationType::SQ {
            nbits: 8,
            standard_deviation: Some(Positive::new(2.98).unwrap())
        },
        "SQ_8_2.98"
    )]
    fn fmt_quantization_type(#[case] quantization: QuantizationType, #[case] expected: &str) {
        assert_eq!(quantization.to_string(), expected);
    }
}
