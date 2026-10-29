/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{self as dbr};
use diskann_inmem::{self as di};
use diskann_quantization::{self as dq};
use diskann_utils::views::rowmajor;
use diskann_vector::{self as dv};

use crate::index::inmem2;

mod dto {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case", deny_unknown_fields)]
    pub(super) enum Bits {
        One,
        Two,
        Four,
        Eight,
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2::quantized) struct Scalar {
        pub(super) bits: Bits,
        pub(super) std: f64,
    }

    impl Scalar {
        pub(super) fn example() -> Self {
            Self {
                bits: Bits::Eight,
                std: 1.5,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Bits {
    One,
    Two,
    Four,
    Eight,
}

impl Bits {
    fn from_raw(raw: &dto::Bits) -> Self {
        match raw {
            dto::Bits::One => Self::One,
            dto::Bits::Two => Self::Two,
            dto::Bits::Four => Self::Four,
            dto::Bits::Eight => Self::Eight,
        }
    }
}

impl std::fmt::Display for Bits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::One => "1",
            Self::Two => "2",
            Self::Four => "4",
            Self::Eight => "8",
        };

        f.write_str(s)
    }
}

#[derive(Debug)]
pub(super) struct Scalar {
    bits: Bits,
    std: dq::num::Positive<f64>,
    metric: di::repr::quantized::scalar::Metric,
}

impl inmem2::quantized::Quantizer for Scalar {
    const KIND: &'static str = "scalar";
    const TAG: &'static str = "inmem2-scq-static";

    type Raw = dto::Scalar;

    fn from_raw(raw: &dto::Scalar, metric: dv::distance::Metric) -> anyhow::Result<Self> {
        let dto::Scalar { bits, std } = raw;

        Ok(Self {
            bits: Bits::from_raw(bits),
            std: dq::num::Positive::new(*std)?,
            metric: metric.try_into()?,
        })
    }

    fn example() -> dto::Scalar {
        dto::Scalar::example()
    }

    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend> {
        use di::repr::quantized::{scalar::Scalar, Backend};

        let Self { bits, std, metric } = self;

        let trainer = dq::scalar::train::ScalarQuantizationParameters::new(*std);
        let quantizer = trainer.train(data);

        let backend = match bits {
            Bits::One => Backend::new(Scalar::<1>::new(quantizer, *metric)),
            Bits::Two => Backend::new(Scalar::<2>::new(quantizer, *metric)),
            Bits::Four => Backend::new(Scalar::<4>::new(quantizer, *metric)),
            Bits::Eight => Backend::new(Scalar::<8>::new(quantizer, *metric)),
        };

        Ok(backend)
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { bits, std, metric } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("bits", bits);

        let std = std.into_inner();
        kv.push("std", &std);
        kv.push_debug("metric", metric);
        kv.fmt(f)
    }
}

///////////
// tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_parses() {
        assert!(<Scalar as inmem2::quantized::Quantizer>::from_raw(
            &dto::Scalar::example(),
            dv::distance::Metric::L2
        )
        .is_ok());
    }
}
