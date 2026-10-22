/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write};

use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_benchmark_runner::{self as dbr, utils::{fmt::Quote, datatype::AsDataType}};
use diskann_quantization::{self as dq};
use diskann_vector::{self as dv};
use rand::{SeedableRng, rngs::StdRng};

use crate::index::inmem2::{self, FromRawWith, reborrow};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register("inmem2-spherical", RunBuildAndSearch)?;
    Ok(())
}

/////////////////////////////
// Static Build And Search //
/////////////////////////////

pub(super) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) enum Bits {
        One,
        Two,
        Four,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Spherical {
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

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BuildAndSearch {
        pub(super) data: inmem2::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::KnnSearch,
        pub(super) spherical: Spherical,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::KnnSearch::example(),
                spherical: Spherical::example(),
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
struct Spherical {
    bits: Bits,
    seed: u64,
    metric: dq::spherical::SupportedMetric,
}

impl Spherical {
    fn from_raw(raw: &dto::Spherical, metric: dv::distance::Metric) -> anyhow::Result<Self> {
        let dto::Spherical {
            bits,
            seed,
        } = raw;

        Ok(Self {
            bits: Bits::from_raw(bits),
            seed: *seed,
            metric: metric.try_into()?,
        })
    }

    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend> {
        use di::repr::quantized::{Backend, spherical::Spherical};

        let Self {
            bits,
            seed,
            metric,
        } = self;

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
            Bits::Four => {
                Backend::new(Spherical::new(quantizer.as_quantizer_impl::<4>()?))
            }
        };

        Ok(quantizer)
    }
}

impl std::fmt::Display for Spherical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            bits,
            seed,
            metric,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("bits", bits);
        kv.push("seed", seed);
        kv.push_debug("metric", metric);
        kv.fmt(f)
    }
}

#[derive(Debug)]
struct BuildAndSearch {
    data: inmem2::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::KnnSearch,
    spherical: Spherical,
}

impl FromRawWith for BuildAndSearch {
    type Raw = dto::BuildAndSearch;

    fn from_raw_with(
        raw: &dto::BuildAndSearch,
        mut checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let dto::BuildAndSearch {
            data,
            build,
            search,
            spherical,
        } = raw;

        let data = inmem2::Data::from_raw_with(data, reborrow(&mut checker))?;
        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance)?;
        let search = inmem2::benchmark::knn::KnnSearch::from_raw_with(search, checker)?;
        let spherical = Spherical::from_raw(spherical, data.distance)?;

        Ok(Self {
            data,
            build,
            search,
            spherical,
        })
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
            spherical,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();

        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        kv.push("spherical", &spherical);
        kv.fmt(f)
    }
}

impl dbr::Input for BuildAndSearch {
    type Raw = dto::BuildAndSearch;
    fn tag() -> &'static str {
        "inmem2-spherical-quantization-build"
    }

    fn from_raw(raw: Self::Raw, checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        Self::from_raw_with(&raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }

    fn example() -> Self::Raw {
        Self::Raw::example()
    }
}

#[derive(Debug)]
struct RunBuildAndSearch;

impl dbr::Benchmark for RunBuildAndSearch {
    type Input = BuildAndSearch;
    type Output = ();

    fn try_match(
        &self,
        input: &BuildAndSearch,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        let mut score = context.success(0);

        let inmem2::DataDispatchParams {
            data_type,
        } = input.data.dispatch_params();

        if !f32::is_match(data_type) {
            score.fail(
                1000,
                &format_args!(
                    "expected data-type {}, instead got {}",
                    Quote(f32::DATA_TYPE),
                    Quote(data_type)
                ),
            )
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spherical-quantized static build+search",
        )?;

        Ok(())
    }

    fn run(
        &self,
        input: &BuildAndSearch,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<()> {
        writeln!(output, "{}\n", input)?;

        // Load data.
        let data = input.data.load_arc::<f32>(Some(output))?;
        let quantizer = input.spherical.train(data.as_view())?;

        // Compute the medoid of the dataset as the single start point.
        let start = diskann::graph::StartPointStrategy::Medoid.compute(data.as_view())?;
        let config = di::repr::Quantized::config(
            quantizer,
            di::num::Capacity::new(data.nrows()),
            di::num::MaxDegree::new(input.build.max_degree()),
            start,
            di::repr::quantized::Rerank::F16 {
                metric: input.data.distance.into(),
            },
        )?;

        let provider = di::Provider::<_, u32>::new(config)?;
        let engine = input.build.engine(provider, di::Strategy);
        let build = engine.build(data, output)?;
        let search = input
            .search
            .search::<f32, _>(engine.index().clone(), Some(output))?;

        Ok(())
    }
}
