/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_benchmark_runner::{
    self as dbr,
    utils::{datatype::AsDataType, fmt::Quote},
};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::Matrix;

use crate::index::inmem2::{self, reborrow, FromRawWith};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register("inmem2-f32", RunBuildAndSearch::<f32>::new())?;
    Ok(())
}

/////////////////////////////
// Static Build And Search //
/////////////////////////////

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct BuildAndSearch {
        pub(super) data: inmem2::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::KnnSearch,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::KnnSearch::example(),
            }
        }
    }
}

#[derive(Debug)]
struct BuildAndSearch {
    data: inmem2::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::KnnSearch,
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
        } = raw;

        let data = inmem2::Data::from_raw_with(data, reborrow(&mut checker))?;
        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance)?;
        let search = inmem2::benchmark::knn::KnnSearch::from_raw_with(search, checker)?;

        Ok(Self {
            data,
            build,
            search,
        })
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
        } = self;
        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        write!(f, "{}", kv)
    }
}

impl dbr::Input for BuildAndSearch {
    type Raw = dto::BuildAndSearch;
    fn tag() -> &'static str {
        "inmem2-full-build"
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
struct RunBuildAndSearch<T>(std::marker::PhantomData<T>);

impl<T> RunBuildAndSearch<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> dbr::Benchmark for RunBuildAndSearch<T>
where
    T: di::repr::FullPrecision + diskann::graph::SampleableForStart + AsDataType,
{
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

        if !T::is_match(data_type) {
            score.fail(
                1000,
                &format_args!(
                    "expected data-type {}, instead got {}",
                    Quote(T::DATA_TYPE),
                    Quote(data_type)
                ),
            )
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "full-precision static build+search with data type {}",
            Quote(T::DATA_TYPE)
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
        let data = input.data.load_arc::<T>(Some(output))?;

        // Compute the medoid of the dataset as the single start point.
        let start = diskann::graph::StartPointStrategy::Medoid.compute(data.as_view())?;
        let config = di::repr::Full::config(
            di::num::Capacity::new(data.nrows()),
            di::num::MaxDegree::new(input.build.max_degree()),
            input.data.distance,
            start,
        )?;

        let provider = di::Provider::<_, u32>::new(config)?;
        let engine = input.build.engine(provider, di::Strategy);
        let build = engine.build(data, output)?;
        let search = input
            .search
            .search::<T, _>(engine.index().clone(), Some(output))?;

        Ok(())
    }
}
