/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use anyhow::Context;
use diskann::graph::DiskANNIndex;
use diskann_benchmark_runner::{
    Checker, Checkpoint, Output, Registry, RegistryError,
    benchmark::{FailureScore, MatchScore},
    files::InputFile,
};
use diskann_vector::distance::Metric;
use diskann_utils::views::Matrix;
use half::f16;

use diskann_inmem::{Provider, layers};

use crate::{
    index::Index,
    support::{datatype::DataType, io::load_and_convert},
};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    registry.register("full-precision-integration-test", FullPrecision)?;
    Ok(())
}

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) enum SerdeMetric {
        L2,
        InnerProduct,
        Cosine,
    }

    impl From<SerdeMetric> for Metric {
        fn from(m: SerdeMetric) -> Self {
            match m {
                SerdeMetric::L2 => Metric::L2,
                SerdeMetric::InnerProduct => Metric::InnerProduct,
                SerdeMetric::Cosine => Metric::Cosine,
            }
        }
    }

    impl TryFrom<Metric> for SerdeMetric {
        type Error = anyhow::Error;
        fn try_from(m: Metric) -> anyhow::Result<Self> {
            match m {
                Metric::L2 => Ok(SerdeMetric::L2),
                Metric::InnerProduct => Ok(SerdeMetric::InnerProduct),
                Metric::Cosine => Ok(SerdeMetric::Cosine),
                Metric::CosineNormalized => anyhow::bail!("cosine normalized is not supported"),
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Data {
        pub(super) data: InputFile,
        pub(super) queries: InputFile,
        pub(super) groundtruth: InputFile,
        pub(super) metric: SerdeMetric,
        pub(super) data_type: DataType,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) enum Layer {
        FullPrecision { data_type: DataType },
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Build {
        pub(super) pruned_degree: usize,
        pub(super) max_degree: usize,
        pub(super) l_build: usize,
        pub(super) alpha: f32,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Test {
        pub(super) data: Data,
        pub(super) layer: Layer,
        pub(super) build: Build,
    }
}

#[derive(Debug)]
struct Data {
    data: InputFile,
    queries: InputFile,
    groundtruth: InputFile,
    metric: Metric,
    data_type: DataType,
}

impl Data {
    fn from_raw(mut raw: dto::Data, checker: &mut Checker) -> anyhow::Result<Self> {
        let dto::Data {
            mut data,
            mut queries,
            mut groundtruth,
            metric,
            data_type,
        } = raw;
        data.resolve(checker)?;
        queries.resolve(checker)?;
        groundtruth.resolve(checker)?;
        Ok(Self {
            data,
            queries,
            groundtruth,
            metric: metric.into(),
            data_type,
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Data> {
        Ok(dto::Data {
            data: self.data.clone(),
            queries: self.queries.clone(),
            groundtruth: self.groundtruth.clone(),
            metric: self.metric.try_into()?,
            data_type: self.data_type,
        })
    }
}

#[derive(Debug)]
enum Layer {
    FullPrecision { data_type: DataType },
}

impl Layer {
    fn from_raw(raw: dto::Layer) -> Self {
        match raw {
            dto::Layer::FullPrecision { data_type } => Self::FullPrecision { data_type },
        }
    }

    fn as_raw(&self) -> dto::Layer {
        match self {
            Self::FullPrecision { data_type } => dto::Layer::FullPrecision {
                data_type: *data_type,
            },
        }
    }
}

#[derive(Debug)]
struct Build {
    config: diskann::graph::Config,
}

impl Build {
    fn from_raw(raw: dto::Build, metric: Metric) -> anyhow::Result<Self> {
        let dto::Build {
            pruned_degree,
            max_degree,
            l_build,
            alpha,
        } = raw;
        let config = diskann::graph::config::Builder::new_with(
            pruned_degree,
            diskann::graph::config::MaxDegree::new(max_degree),
            l_build,
            metric.into(),
            |b| {
                b.alpha(alpha);
            },
        )
        .build()?;

        Ok(Self { config })
    }

    fn as_raw(&self) -> dto::Build {
        dto::Build {
            pruned_degree: self.config.pruned_degree().get(),
            max_degree: self.config.max_degree().get(),
            l_build: self.config.l_build().get(),
            alpha: self.config.alpha(),
        }
    }
}

#[derive(Debug)]
struct Test {
    data: Data,
    layer: Layer,
    build: Build,
}

impl Test {
    fn from_raw(raw: dto::Test, checker: &mut Checker) -> anyhow::Result<Self> {
        let data = Data::from_raw(raw.data, checker)?;
        let layer = Layer::from_raw(raw.layer);
        let build = Build::from_raw(raw.build, data.metric)?;

        Ok(Self { data, layer, build })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Test> {
        Ok(dto::Test {
            data: self.data.as_raw()?,
            layer: self.layer.as_raw(),
            build: self.build.as_raw(),
        })
    }
}

///////////////
// Benchmark //
///////////////

impl diskann_benchmark_runner::Input for Test {
    type Raw = dto::Test;

    fn tag() -> &'static str {
        "integration-test"
    }

    fn from_raw(raw: dto::Test, checker: &mut Checker) -> anyhow::Result<Self> {
        <Test>::from_raw(raw, checker)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        let raw = self.as_raw()?;
        Ok(serde_json::to_value(raw)?)
    }

    fn example() -> dto::Test {
        dto::Test {
            data: dto::Data {
                data: InputFile::new("path/to/data"),
                queries: InputFile::new("path/to/queries"),
                groundtruth: InputFile::new("path/to/groundtruth"),
                metric: dto::SerdeMetric::L2,
                data_type: DataType::F32,
            },
            layer: dto::Layer::FullPrecision {
                data_type: DataType::F32,
            },
            build: dto::Build {
                pruned_degree: 16,
                max_degree: 20,
                l_build: 50,
                alpha: 1.2,
            },
        }
    }
}

#[derive(Debug)]
struct FullPrecision;

impl diskann_benchmark_runner::Benchmark for FullPrecision {
    type Input = Test;
    type Output = ();

    fn try_match(&self, input: &Test) -> Result<MatchScore, FailureScore> {
        if let Layer::FullPrecision { .. } = input.layer {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(1))
        }
    }

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&Test>,
    ) -> std::fmt::Result {
        write!(f, "nop")
    }

    fn run(
        &self,
        input: &Test,
        checkpoint: Checkpoint<'_>,
        output: &mut dyn Output,
    ) -> anyhow::Result<()> {
        let Layer::FullPrecision { data_type } = input.layer else {
            anyhow::bail!("oops");
        };

        // Load the data and perform any necessary data conversions.
        let data = {
            let mut io = std::fs::File::open(&*input.data.data)
                .with_context(|| format!("could not open {}", input.data.data.display()))?;

            load_and_convert(&mut io, input.data.data_type, data_type)?
        };

        let dim = data.nrows();

        let config = diskann_inmem::provider::Config::new(
            data.nrows(),
            input.build.config.max_degree().get()
        );

        fn finish<DP>(provider: DP, config: diskann::graph::Config) -> Arc<dyn Index>
        where
            DP: diskann::provider::DataProvider,
            DiskANNIndex<DP>: Index,
        {
            Arc::new(DiskANNIndex::new(config, provider, None))
        }

        let index_config = input.build.config.clone();
        let index: Arc<dyn Index> = match data_type {
            DataType::F32 => {
                let start = Matrix::new(0.0f32, dim, 1);
                let provider = Provider::new(
                    layers::Full::<f32>::new(dim, input.data.metric),
                    config,
                    start.row_iter(),
                );

                finish(provider, index_config)
            },
            DataType::F16 => {
                let start = Matrix::new(f16::from_f32(0.0f32), dim, 1);
                let provider = Provider::new(
                    layers::Full::<f16>::new(dim, input.data.metric),
                    config,
                    start.row_iter(),
                );

                finish(provider, index_config)
            },
            DataType::U8 => {
                let start = Matrix::new(0u8, dim, 1);
                let provider = Provider::new(
                    layers::Full::<u8>::new(dim, input.data.metric),
                    config,
                    start.row_iter(),
                );
                finish(provider, index_config)
            },
            DataType::I8 => {
                let start = Matrix::new(0i8, dim, 1);
                let provider = Provider::new(
                    layers::Full::<i8>::new(dim, input.data.metric),
                    config,
                    start.row_iter(),
                );

                finish(provider, index_config)
            },
        };

        let rt = diskann_benchmark_core::tokio::runtime(1)?;

        super::tests::insert(
            &*index,
            data.as_view(),
            rt.handle(),
        )?;

        Ok(())
    }
}
