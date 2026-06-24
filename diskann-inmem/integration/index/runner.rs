/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use anyhow::Context;
use diskann::graph::{DiskANNIndex, search::Knn};
use diskann_benchmark_runner::{
    Checker, Checkpoint, Output, Registry, RegistryError,
    benchmark::{FailureScore, MatchScore},
    files::InputFile,
    utils::fmt::Indent,
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;
use half::f16;
use serde::{Deserialize, Serialize};

use diskann_inmem::{Provider, layers};

use crate::{
    index::{Counters, Index},
    support::{
        datatype::{DataType, Dataset, DatasetView},
        io::load_and_convert,
    },
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
    pub(super) struct KnnSearch {
        pub(super) knn: usize,
        pub(super) search_l: usize,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        pub(super) beam_width: Option<usize>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Search {
        pub(super) knn: Vec<KnnSearch>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Test {
        pub(super) data: Data,
        pub(super) layer: Layer,
        pub(super) build: Build,
        pub(super) search: Search,
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

    fn load_as(&self, data_type: DataType) -> anyhow::Result<Bundle> {
        let data = {
            let mut io = std::fs::File::open(&*self.data)
                .with_context(|| format!("could not open {}", self.data.display()))?;

            load_and_convert(&mut io, self.data_type, data_type)?
        };

        let queries = {
            let mut io = std::fs::File::open(&*self.queries)
                .with_context(|| format!("could not open {}", self.queries.display()))?;

            load_and_convert(&mut io, self.data_type, data_type)?
        };

        let groundtruth = {
            let mut io = std::fs::File::open(&*self.groundtruth)
                .with_context(|| format!("could not open {}", self.queries.display()))?;

            let raw = diskann_utils::io::read_bin::<u32>(&mut io)?;
            raw.map(|&x| u64::from(x))
        };

        Ok(Bundle {
            data,
            queries,
            groundtruth,
        })
    }
}

#[derive(Debug)]
struct Bundle {
    data: Dataset,
    queries: Dataset,
    groundtruth: Matrix<u64>,
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
struct Search {
    knn: Vec<Knn>,
}

impl Search {
    fn from_raw(raw: dto::Search) -> anyhow::Result<Self> {
        fn make_knn(raw: &dto::KnnSearch) -> anyhow::Result<Knn> {
            Ok(Knn::new(raw.knn, raw.search_l, raw.beam_width)?)
        }

        Ok(Self {
            knn: raw
                .knn
                .iter()
                .map(make_knn)
                .collect::<anyhow::Result<Vec<_>>>()?,
        })
    }

    fn as_raw(&self) -> dto::Search {
        fn make_knn(knn: &Knn) -> dto::KnnSearch {
            dto::KnnSearch {
                knn: knn.k_value().get(),
                search_l: knn.l_value().get(),
                beam_width: Some(knn.beam_width().get()),
            }
        }

        dto::Search {
            knn: self.knn.iter().map(make_knn).collect(),
        }
    }
}

#[derive(Debug)]
struct Test {
    data: Data,
    layer: Layer,
    build: Build,
    search: Search,
}

impl Test {
    fn from_raw(raw: dto::Test, checker: &mut Checker) -> anyhow::Result<Self> {
        let data = Data::from_raw(raw.data, checker)?;
        let layer = Layer::from_raw(raw.layer);
        let build = Build::from_raw(raw.build, data.metric)?;
        let search = Search::from_raw(raw.search)?;

        Ok(Self {
            data,
            layer,
            build,
            search,
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Test> {
        Ok(dto::Test {
            data: self.data.as_raw()?,
            layer: self.layer.as_raw(),
            build: self.build.as_raw(),
            search: self.search.as_raw(),
        })
    }

    fn index(
        &self,
        capacity: usize,
        start_points: DatasetView<'_>,
    ) -> anyhow::Result<Arc<dyn Index>> {
        match self.layer {
            Layer::FullPrecision { data_type } => {
                if start_points.data_type() != data_type {
                    anyhow::bail!(
                        "mismatched data types for start point - expected {}, got {}",
                        data_type,
                        start_points.data_type(),
                    );
                }

                let dim = start_points.ncols();
                let metric = self.data.metric;
                let config = diskann_inmem::provider::Config::new(
                    capacity,
                    self.build.config.max_degree().get(),
                );

                let index_config = self.build.config.clone();

                let index = match start_points {
                    DatasetView::F32(v) => finish(
                        Provider::new(layers::Full::<f32>::new(dim, metric), config, v.row_iter()),
                        index_config,
                    ),
                    DatasetView::F16(v) => finish(
                        Provider::new(layers::Full::<f16>::new(dim, metric), config, v.row_iter()),
                        index_config,
                    ),
                    DatasetView::U8(v) => finish(
                        Provider::new(layers::Full::<u8>::new(dim, metric), config, v.row_iter()),
                        index_config,
                    ),
                    DatasetView::I8(v) => finish(
                        Provider::new(layers::Full::<i8>::new(dim, metric), config, v.row_iter()),
                        index_config,
                    ),
                };

                Ok(index)
            }
        }
    }
}

fn finish<DP>(provider: DP, config: diskann::graph::Config) -> Arc<dyn Index>
where
    DP: diskann::provider::DataProvider,
    DiskANNIndex<DP>: Index,
{
    Arc::new(DiskANNIndex::new(config, provider, None))
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
            search: dto::Search {
                knn: vec![
                    dto::KnnSearch {
                        knn: 10,
                        search_l: 50,
                        beam_width: None,
                    },
                    dto::KnnSearch {
                        knn: 10,
                        search_l: 50,
                        beam_width: Some(3),
                    },
                    dto::KnnSearch {
                        knn: 20,
                        search_l: 100,
                        beam_width: Some(3),
                    },
                ],
            },
        }
    }
}

////////////////
// Benchmarks //
////////////////

#[derive(Debug)]
struct FullPrecision;

impl diskann_benchmark_runner::Benchmark for FullPrecision {
    type Input = Test;
    type Output = BuildAndSearch;

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
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let Layer::FullPrecision { data_type } = input.layer else {
            anyhow::bail!("expected full-precision");
        };

        // Load the data and perform any necessary data conversions.
        let Bundle {
            data,
            queries,
            groundtruth,
        } = input.data.load_as(data_type)?;

        let index = input.index(data.nrows(), data.medoid().as_view())?;
        let rt = diskann_benchmark_core::tokio::runtime(1)?;
        let build = super::tests::insert(&*index, data.as_view(), rt.handle())?;

        let mut knn = Vec::new();
        for param in input.search.knn.iter() {
            let stats = super::tests::knn(
                &*index,
                param.clone(),
                queries.as_view(),
                &groundtruth.as_view(),
                rt.handle(),
            )?;

            knn.push(stats);
        }

        let build_and_search = BuildAndSearch { build, knn };

        writeln!(output, "{}", build_and_search)?;

        Ok(build_and_search)
    }
}

////////////
// Output //
////////////

#[derive(Debug, Serialize, Deserialize)]
struct BuildAndSearch {
    build: Counters,
    knn: Vec<super::tests::KnnStats>,
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "build stats")?;
        writeln!(f, "{}", Indent::new(&self.build.to_string(), 4))?;
        writeln!(f, "knn stats")?;
        for k in self.knn.iter() {
            writeln!(f, "{}\n", k)?;
        }

        Ok(())
    }
}
