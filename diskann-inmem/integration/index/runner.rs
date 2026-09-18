/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, sync::Arc};

use anyhow::Context;
use diskann::graph::{DiskANNIndex, search::Knn};
use diskann_benchmark_runner::{
    Checker, Checkpoint, Output, Registry, RegistryError,
    benchmark::{MatchContext, PassFail, Regression, Score},
    files::InputFile,
    utils::fmt::Indent,
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;
use serde::{Deserialize, Serialize};

use diskann_inmem::{
    self as inmem, Provider,
    num::{Capacity, MaxDegree},
    repr::Full,
};

use crate::{
    index::{Counters, Index},
    support::{
        check::{CheckMatch, Match, check_all_fields},
        datatype::{self, DataType, Dataset, DatasetView},
        io::load_and_convert,
        tolerance,
    },
};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    registry.register_regression("full-precision-integration-test", FullPrecision)?;
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
    #[serde(rename_all = "kebab-case")]
    pub(super) enum Preprocess {
        Halve,
        Floor,
    }

    impl From<Preprocess> for datatype::Preprocess {
        fn from(op: Preprocess) -> Self {
            match op {
                Preprocess::Halve => datatype::Preprocess::Halve,
                Preprocess::Floor => datatype::Preprocess::Floor,
            }
        }
    }

    impl From<&datatype::Preprocess> for Preprocess {
        fn from(op: &datatype::Preprocess) -> Self {
            match op {
                datatype::Preprocess::Halve => Preprocess::Halve,
                datatype::Preprocess::Floor => Preprocess::Floor,
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
        pub(super) preprocess: Vec<Preprocess>,
    }

    //-------------------------//
    // Quantization Parameters //
    //-------------------------//

    pub(super) mod spherical {
        use super::*;

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case")]
        pub(in crate::index::runner) enum Bits {
            One,
            Two,
            Four,
        }

        #[derive(Debug, Serialize, Deserialize)]
        #[serde(rename_all = "kebab-case")]
        pub(in crate::index::runner) enum Rerank {
            None,
            F16,
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(rename_all = "kebab-case")]
    pub(super) enum Representation {
        FullPrecision {
            data_type: DataType,
        },
        Spherical {
            bits: spherical::Bits,
            rerank: spherical::Rerank,
        },
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
        pub(super) representation: Representation,
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
    preprocess: Vec<datatype::Preprocess>,
}

impl Data {
    fn from_raw(raw: dto::Data, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let dto::Data {
            mut data,
            mut queries,
            mut groundtruth,
            metric,
            data_type,
            preprocess,
        } = raw;

        if let Some(checker) = checker {
            data.resolve(checker)?;
            queries.resolve(checker)?;
            groundtruth.resolve(checker)?;
        }

        Ok(Self {
            data,
            queries,
            groundtruth,
            metric: metric.into(),
            data_type,
            preprocess: preprocess.into_iter().map(From::from).collect(),
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Data> {
        Ok(dto::Data {
            data: self.data.clone(),
            queries: self.queries.clone(),
            groundtruth: self.groundtruth.clone(),
            metric: self.metric.try_into()?,
            data_type: self.data_type,
            preprocess: self.preprocess.iter().map(From::from).collect(),
        })
    }

    fn load_as(&self, data_type: DataType) -> anyhow::Result<Bundle> {
        let data = {
            let mut io = std::fs::File::open(&*self.data)
                .with_context(|| format!("could not open {}", self.data.display()))?;

            load_and_convert(&mut io, self.data_type, data_type, &self.preprocess)?
        };

        let queries = {
            let mut io = std::fs::File::open(&*self.queries)
                .with_context(|| format!("could not open {}", self.queries.display()))?;

            load_and_convert(&mut io, self.data_type, data_type, &self.preprocess)?
        };

        let groundtruth = {
            let mut io = std::fs::File::open(&*self.groundtruth)
                .with_context(|| format!("could not open {}", self.groundtruth.display()))?;

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

mod spherical {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub(super) enum Bits {
        One,
        Two,
        Four,
    }

    impl Bits {
        pub(super) fn from_raw(raw: dto::spherical::Bits) -> Self {
            match raw {
                dto::spherical::Bits::One => Self::One,
                dto::spherical::Bits::Two => Self::Two,
                dto::spherical::Bits::Four => Self::Four,
            }
        }

        pub(super) fn as_raw(&self) -> dto::spherical::Bits {
            match self {
                Self::One => dto::spherical::Bits::One,
                Self::Two => dto::spherical::Bits::Two,
                Self::Four => dto::spherical::Bits::Four,
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) enum Rerank {
        None,
        F16,
    }

    impl Rerank {
        pub(super) fn from_raw(raw: dto::spherical::Rerank) -> Self {
            match raw {
                dto::spherical::Rerank::None => Self::None,
                dto::spherical::Rerank::F16 => Self::F16,
            }
        }

        pub(super) fn as_raw(&self) -> dto::spherical::Rerank {
            match self {
                Self::None => dto::spherical::Rerank::None,
                Self::F16 => dto::spherical::Rerank::F16,
            }
        }
    }

    impl From<Rerank> for inmem::repr::spherical::Rerank {
        fn from(rerank: Rerank) -> Self {
            match rerank {
                Rerank::None => inmem::repr::spherical::Rerank::None,
                Rerank::F16 => inmem::repr::spherical::Rerank::F16,
            }
        }
    }
}

#[derive(Debug)]
enum Representation {
    FullPrecision {
        data_type: DataType,
    },
    Spherical {
        bits: spherical::Bits,
        rerank: spherical::Rerank,
    },
}

impl Representation {
    fn from_raw(raw: dto::Representation) -> Self {
        match raw {
            dto::Representation::FullPrecision { data_type } => Self::FullPrecision { data_type },
            dto::Representation::Spherical { bits, rerank } => Self::Spherical {
                bits: spherical::Bits::from_raw(bits),
                rerank: spherical::Rerank::from_raw(rerank),
            },
        }
    }

    fn as_raw(&self) -> dto::Representation {
        match self {
            Self::FullPrecision { data_type } => dto::Representation::FullPrecision {
                data_type: *data_type,
            },
            Self::Spherical { bits, rerank } => dto::Representation::Spherical {
                bits: bits.as_raw(),
                rerank: rerank.as_raw(),
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
    knn: Vec<(usize, Knn)>,
}

impl Search {
    fn from_raw(raw: dto::Search) -> anyhow::Result<Self> {
        fn make_knn(raw: &dto::KnnSearch) -> anyhow::Result<(usize, Knn)> {
            Ok((raw.knn, Knn::new(raw.search_l, raw.beam_width)?))
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
        fn make_knn((k, knn): &(usize, Knn)) -> dto::KnnSearch {
            dto::KnnSearch {
                knn: *k,
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
    representation: Representation,
    build: Build,
    search: Search,
}

impl Test {
    fn from_raw(raw: dto::Test, checker: Option<&mut Checker>) -> anyhow::Result<Self> {
        let data = Data::from_raw(raw.data, checker)?;
        let representation = Representation::from_raw(raw.representation);
        let build = Build::from_raw(raw.build, data.metric)?;
        let search = Search::from_raw(raw.search)?;

        Ok(Self {
            data,
            representation,
            build,
            search,
        })
    }

    fn as_raw(&self) -> anyhow::Result<dto::Test> {
        Ok(dto::Test {
            data: self.data.as_raw()?,
            representation: self.representation.as_raw(),
            build: self.build.as_raw(),
            search: self.search.as_raw(),
        })
    }

    fn index(&self, data: DatasetView<'_>) -> anyhow::Result<Arc<dyn Index>> {
        match &self.representation {
            Representation::FullPrecision { data_type } => {
                if data.data_type() != *data_type {
                    anyhow::bail!(
                        "mismatched data types for start point - expected {}, got {}",
                        data_type,
                        data.data_type(),
                    );
                }

                let start_points = data.medoid();
                let metric = self.data.metric;
                let capacity = Capacity::new(data.nrows());
                let max_degree = self.build.config.max_degree().get();
                let index_config = self.build.config.clone();

                let index = match start_points {
                    Dataset::F32(v) => finish(
                        Provider::new(Full::config(
                            capacity,
                            MaxDegree::new(max_degree),
                            metric,
                            v,
                        )?)?,
                        index_config,
                    ),
                    Dataset::F16(v) => finish(
                        Provider::new(Full::config(
                            capacity,
                            MaxDegree::new(max_degree),
                            metric,
                            v,
                        )?)?,
                        index_config,
                    ),
                    Dataset::U8(v) => finish(
                        Provider::new(Full::config(
                            capacity,
                            MaxDegree::new(max_degree),
                            metric,
                            v,
                        )?)?,
                        index_config,
                    ),
                    Dataset::I8(v) => finish(
                        Provider::new(Full::config(
                            capacity,
                            MaxDegree::new(max_degree),
                            metric,
                            v,
                        )?)?,
                        index_config,
                    ),
                };

                Ok(index)
            }
            Representation::Spherical { bits, rerank } => {
                self.create_spherical(data, *bits, *rerank)
            }
        }
    }

    fn create_spherical(
        &self,
        data: DatasetView<'_>,
        bits: spherical::Bits,
        rerank: spherical::Rerank,
    ) -> anyhow::Result<Arc<dyn Index>> {
        use diskann_quantization::{
            algorithms::transforms,
            alloc::GlobalAllocator,
            spherical::{PreScale, SphericalQuantizer},
        };
        use rand::SeedableRng;

        let DatasetView::F32(data) = data else {
            anyhow::bail!("spherical quantization only supports f32 data");
        };

        // Step 1: Train a generic quantizer.
        let quantizer = SphericalQuantizer::train(
            data,
            transforms::TransformKind::DoubleHadamard {
                target_dim: transforms::TargetDim::Same,
            },
            self.data.metric.try_into().map_err(|_| {
                anyhow::anyhow!(
                    "metric {} not supported for spherical quatnization",
                    self.data.metric
                )
            })?,
            PreScale::ReciprocalMeanNorm,
            &mut rand::rngs::StdRng::seed_from_u64(0xc0ff33),
            GlobalAllocator,
        )?;

        // Step 2: Associate it with the target bit-width.
        let quantizer = match bits {
            spherical::Bits::One => quantizer.as_quantizer::<1>()?,
            spherical::Bits::Two => quantizer.as_quantizer::<2>()?,
            spherical::Bits::Four => quantizer.as_quantizer::<4>()?,
        };

        let start_point = Matrix::row_vector(Box::from(
            <f32 as diskann_utils::sampling::medoid::ComputeMedoid>::compute_medoid(data),
        ));

        // Step 3: Create the config.
        let config = diskann_inmem::repr::Spherical::config(
            quantizer,
            Capacity::new(data.nrows()),
            MaxDegree::new(self.build.config.max_degree().get()),
            start_point,
            rerank.into(),
        )?;

        let index_config = self.build.config.clone();

        Ok(finish(Provider::new(config)?, index_config))
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
        <Test>::from_raw(raw, Some(checker))
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
                preprocess: vec![],
            },
            representation: dto::Representation::FullPrecision {
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

    fn try_match(&self, input: &Test, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        match input.representation {
            Representation::FullPrecision { .. } => {
                // We match all valid data-types
            }
            Representation::Spherical { .. } => {
                let data_type = input.data.data_type;
                // Ensure that the data type if `f32`.
                if data_type != DataType::F32 {
                    score.fail(
                        1,
                        &format_args!(
                            "spherical-quantization requires f32 data, not {}",
                            data_type
                        ),
                    );
                }
            }
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nop")
    }

    fn run(
        &self,
        input: &Test,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let data_type = match input.representation {
            Representation::FullPrecision { data_type } => data_type,
            Representation::Spherical { .. } => input.data.data_type,
        };

        // Load the data and perform any necessary data conversions.
        let Bundle {
            data,
            queries,
            groundtruth,
        } = input.data.load_as(data_type)?;

        let index = input.index(data.as_view())?;
        let rt = diskann_benchmark_core::tokio::runtime(1)?;
        let build = super::tests::insert(&*index, data.as_view(), rt.handle())?;

        let mut knn = Vec::new();
        for (k, param) in input.search.knn.iter() {
            let stats = super::tests::knn(
                &*index,
                *k,
                *param,
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

impl Regression for FullPrecision {
    type Tolerances = tolerance::Empty;
    type Pass = Match;
    type Fail = Match;

    fn check(
        &self,
        _tolerances: &Self::Tolerances,
        _input: &Self::Input,
        before: &Self::Output,
        after: &Self::Output,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        Ok(after.check_match(before).pass_fail())
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

impl CheckMatch for BuildAndSearch {
    fn check_match(&self, previous: &Self) -> Match {
        let builder = check_all_fields!(
            self,
            previous,
            { build, knn },
        );
        builder.finish()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_parses() {
        let _ = Test::from_raw(<Test as diskann_benchmark_runner::Input>::example(), None).unwrap();
    }
}
