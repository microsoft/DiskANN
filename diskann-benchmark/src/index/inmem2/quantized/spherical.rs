/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{self as dbr};
use diskann_inmem::{self as di};
use diskann_quantization::{self as dq};
use diskann_utils::views::rowmajor;
use diskann_vector::{self as dv};
use rand::{rngs::StdRng, SeedableRng};

use crate::index::inmem2;

mod dto {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) enum Bits {
        One,
        Two,
        Four,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(in crate::index::inmem2::quantized) struct Spherical {
        pub(super) bits: Bits,
        pub(super) seed: u64,
    }

    impl Spherical {
        pub(super) fn example() -> Self {
            Self {
                bits: Bits::One,
                seed: 0xdeadbeef,
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Bits {
    One,
    Two,
    Four,
}

impl Bits {
    fn from_raw(raw: &dto::Bits) -> Self {
        match raw {
            dto::Bits::One => Self::One,
            dto::Bits::Two => Self::Two,
            dto::Bits::Four => Self::Four,
        }
    }
}

impl std::fmt::Display for Bits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::One => "1",
            Self::Two => "2",
            Self::Four => "4",
        };

        f.write_str(s)
    }
}

#[derive(Debug)]
pub(super) struct Spherical {
    bits: Bits,
    seed: u64,
    metric: dq::spherical::SupportedMetric,
}

impl inmem2::quantized::Quantizer for Spherical {
    const KIND: &'static str = "spherical";
    const TAG: &'static str = "inmem2-sphq-build";

    type Raw = dto::Spherical;

    fn from_raw(raw: &dto::Spherical, metric: dv::distance::Metric) -> anyhow::Result<Self> {
        let dto::Spherical { bits, seed } = raw;

        Ok(Self {
            bits: Bits::from_raw(bits),
            seed: *seed,
            metric: metric.try_into()?,
        })
    }

    fn example() -> dto::Spherical {
        dto::Spherical::example()
    }

    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend> {
        use di::repr::quantized::{spherical::Spherical, Backend};

        let Self { bits, seed, metric } = self;

        let mut rng = StdRng::seed_from_u64(*seed);
        let quantizer = dq::spherical::SphericalQuantizer::train(
            data,
            dq::algorithms::transforms::TransformKind::DoubleHadamard {
                target_dim: dq::algorithms::transforms::TargetDim::Same,
            },
            *metric,
            dq::spherical::PreScale::ReciprocalMeanNorm,
            &mut rng,
            dq::alloc::GlobalAllocator,
        )?;

        let quantizer = match bits {
            Bits::One => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<1>()?)),
            Bits::Two => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<2>()?)),
            Bits::Four => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<4>()?)),
        };

        Ok(quantizer)
    }
}

impl std::fmt::Display for Spherical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { bits, seed, metric } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("bits", bits);
        kv.push("seed", seed);
        kv.push_debug("metric", metric);
        kv.fmt(f)
    }
}
