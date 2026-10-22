/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{io::Write, num::NonZeroUsize};

use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_benchmark_runner::{self as dbr, utils::{fmt::Quote, datatype::AsDataType}};
use diskann_vector::{self as dv};

use crate::index::inmem2::{self, FromRawWith, reborrow, nz};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register("inmem2-pq", RunBuildAndSearch)?;
    Ok(())
}

/////////////////////////////
// Static Build And Search //
/////////////////////////////

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct Product {
        pub(super) chunks: NonZeroUsize,
        pub(super) centers: NonZeroUsize,
        pub(super) seed: u64,
        pub(super) num_threads: NonZeroUsize,
    }

    impl Product {
        pub(super) fn example() -> Self {
            nz!(CHUNKS = 16);
            nz!(CENTERS = 16);
            nz!(NUM_THREADS = 8);

            Self {
                chunks: CHUNKS,
                centers: CENTERS,
                seed: 0xc0ffee,
                num_threads: NUM_THREADS,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BuildAndSearch {
        pub(super) data: inmem2::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::KnnSearch,
        pub(super) pq: Product,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::KnnSearch::example(),
                pq: Product::example(),
            }
        }
    }
}

#[derive(Debug)]
struct Product {
    chunks: NonZeroUsize,
    centers: NonZeroUsize,
    seed: u64,
    metric: di::repr::quantized::product::Metric,
    num_threads: NonZeroUsize,
}

impl Product {
    fn from_raw(raw: &dto::Product, metric: dv::distance::Metric) -> Self {
        let dto::Product {
            chunks,
            centers,
            seed,
            num_threads,
        } = raw;

        Self {
            chunks: *chunks,
            centers: *centers,
            seed: *seed,
            metric: metric.into(),
            num_threads: *num_threads,
        }
    }

    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend> {
        use diskann_quantization::{
            cancel::DontCancel,
            product::{self, train::TrainQuantizer},
            random,
            views::ChunkOffsets,
            Parallelism,
        };

        let Self {
            chunks,
            centers,
            seed,
            metric,
            num_threads,
        } = self;

        let trainer = product::train::LightPQTrainingParameters::new(centers.get(), 5);
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads.get())
            .build()?;

        let Some(dim) = NonZeroUsize::new(data.ncols()) else {
            anyhow::bail!("zero dimensional data is not supported");
        };

        let table = threadpool.install(|| -> anyhow::Result<_> {
            let table = trainer.train(
                data,
                ChunkOffsets::partition(dim, *chunks)?.as_view(),
                Parallelism::Rayon,
                &random::StdRngBuilder::new(*seed),
                &DontCancel,
            )?;

            Ok(table)
        })?;

        let quantizer = di::repr::quantized::product::Product::new(
            table,
            *metric,
            Some(*num_threads),
        )?;

        Ok(di::repr::quantized::Backend::new(quantizer))
    }
}

impl std::fmt::Display for Product {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            chunks,
            centers,
            seed,
            metric,
            num_threads,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("chunks", chunks);
        kv.push("centers", centers);
        kv.push("seed", seed);
        kv.push_eager("metric", format_args!("{:?}", metric));
        kv.push("num_threads", num_threads);
        write!(f, "{}", kv)
    }
}

#[derive(Debug)]
struct BuildAndSearch {
    data: inmem2::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::KnnSearch,
    pq: Product,
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
            pq,
        } = raw;

        let data = inmem2::Data::from_raw_with(data, reborrow(&mut checker))?;
        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance)?;
        let search = inmem2::benchmark::knn::KnnSearch::from_raw_with(search, checker)?;
        let pq = Product::from_raw(pq, data.distance);

        Ok(Self {
            data,
            build,
            search,
            pq,
        })
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
            pq,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();

        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        kv.push("pq", &pq);

        write!(f, "{}", kv)
    }
}

impl dbr::Input for BuildAndSearch {
    type Raw = dto::BuildAndSearch;
    fn tag() -> &'static str {
        "inmem2-product-quantization-build"
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
            "product-quantized static build+search",
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
        let quantizer = input.pq.train(data.as_view())?;

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
