/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_benchmark_runner::{self as dbr};
use diskann_inmem::{self as di};
use diskann_quantization::{self as dq};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{self as dv};

use crate::{
    index::inmem2::{self, common::FromRawWith},
};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register("inmem2-scalar", RunBuildAndSearch)?;
    Ok(())
}

pub(super) mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) enum Bits {
        One,
        Two,
        Four,
        Eight,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Scalar {
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

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BuildAndSearch {
        pub(super) data: inmem2::common::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::KnnSearch,
        pub(super) scalar: Scalar,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::common::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::KnnSearch::example(),
                scalar: Scalar::example(),
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
struct Scalar {
    bits: Bits,
    std: dq::num::Positive<f64>,
    metric: di::repr::quantized::scalar::Metric,
}

impl Scalar {
    fn from_raw(raw: &dto::Scalar, metric: dv::distance::Metric) -> anyhow::Result<Self> {
        let dto::Scalar { bits, std } = raw;

        Ok(Self {
            bits: Bits::from_raw(bits),
            std: dq::num::Positive::new(*std)?,
            metric: metric.try_into()?,
        })
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

#[derive(Debug)]
struct BuildAndSearch {
    data: inmem2::common::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::KnnSearch,
    scalar: Scalar,
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
            scalar,
        } = raw;

        let data = inmem2::common::Data::from_raw_with(data, checker.as_deref_mut())?;
        if data.data_type() != dbr::utils::datatype::DataType::Float32 {
            anyhow::bail!("quantization only supports f32 data");
        }

        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance())?;
        let search = inmem2::benchmark::knn::KnnSearch::from_raw_with(search, checker)?;
        let scalar = Scalar::from_raw(scalar, data.distance())?;

        Ok(Self {
            data,
            build,
            search,
            scalar,
        })
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
            scalar,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();

        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        kv.push("scalar", &scalar);
        kv.fmt(f)
    }
}

impl dbr::Input for BuildAndSearch {
    type Raw = dto::BuildAndSearch;
    fn tag() -> &'static str {
        "inmem2-scalar-quantization-build"
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
    type Output = inmem2::benchmark::BuildAndSearchStats;

    fn try_match(
        &self,
        _input: &BuildAndSearch,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "scalar-quantized static build+search",)?;

        Ok(())
    }

    fn run(
        &self,
        input: &BuildAndSearch,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}\n", input)?;

        let data = input.data.load_arc::<f32>(Some(output))?;
        let quantizer = input.scalar.train(data.as_view())?;
        let provider = inmem2::benchmark::quantized(
            quantizer,
            data.as_view(),
            input.build.max_degree(),
            input.data.distance(),
        )?;
        let engine = input.build.engine(provider);
        let build = engine.build(data, Some(output))?;
        let search = input
            .search
            .search::<f32, _>(engine.index().clone(), Some(output))?;

        Ok(Self::Output { build, search })
    }
}
