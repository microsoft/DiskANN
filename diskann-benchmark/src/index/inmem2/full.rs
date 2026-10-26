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

use crate::{
    index::inmem2::{self, common::FromRawWith},
    utils::match_data_type,
};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register("inmem2-f32", RunBuildAndSearch::<f32>::new())?;
    registry.register("inmem2-f16", RunBuildAndSearch::<half::f16>::new())?;
    registry.register("inmem2-u8", RunBuildAndSearch::<u8>::new())?;
    registry.register("inmem2-f32-stream", RunStream::<f32>::new())?;
    Ok(())
}

//-----//
// DTO //
//-----//

mod dto {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct BuildAndSearch {
        pub(super) data: inmem2::common::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::Knn,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::common::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::Knn::example(),
            }
        }
    }
}

//--------------//
// Static Build //
//--------------//

#[derive(Debug)]
struct BuildAndSearch {
    data: inmem2::common::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::Knn,
    // The raw input.
    raw: serde_json::Value,
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

        let data = inmem2::common::Data::from_raw_with(data, checker.as_deref_mut())?;
        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance())?;
        let search = inmem2::benchmark::knn::Knn::from_raw_with(search, checker)?;

        Ok(Self {
            data,
            build,
            search,
            raw: serde_json::to_value(raw)?,
        })
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
            raw: _,
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
        "inmem2-fp-static"
    }

    fn from_raw(raw: Self::Raw, checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        Self::from_raw_with(&raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(self.raw.clone())
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
    type Output = inmem2::benchmark::BuildAndSearchStats;

    fn try_match(
        &self,
        input: &BuildAndSearch,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        let mut score = context.success(0);
        match_data_type::<T>(&mut score, input.data.data_type());
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
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}\n", input)?;

        // Load data.
        let data = input.data.load_arc::<T>(Some(output))?;

        // Compute the medoid of the dataset as the single start point.
        let start = diskann::graph::StartPointStrategy::Medoid.compute(data.as_view())?;
        let config = di::repr::Full::config(
            di::num::Capacity::new(data.nrows()),
            di::num::MaxDegree::new(input.build.max_degree()),
            input.data.distance(),
            start,
        )?;

        let provider = di::Provider::<_, u32>::new(config)?;
        let engine = input.build.engine(provider)?;
        let build = engine.build(data, Some(output))?;
        let search = input
            .search
            .search::<T, _>(engine.index().clone(), Some(output))?;

        Ok(Self::Output { build, search })
    }
}

//--------//
// BigANN //
//--------//

#[derive(Debug)]
struct Stream {
    stream: inmem2::benchmark::bigann::BigANN,
    // The raw input.
    raw: serde_json::Value,
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stream.fmt(f)
    }
}

impl dbr::Input for Stream {
    type Raw = inmem2::benchmark::bigann::dto::BigANN;
    fn tag() -> &'static str {
        "inmem2-fp-bigann-stream"
    }

    fn from_raw(raw: Self::Raw, checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        let stream = inmem2::benchmark::bigann::BigANN::from_raw_with(&raw, Some(checker))?;
        Ok(Self {
            stream,
            raw: serde_json::to_value(raw)?,
        })
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(self.raw.clone())
    }

    fn example() -> Self::Raw {
        Self::Raw::example()
    }
}

#[derive(Debug)]
struct RunStream<T>(std::marker::PhantomData<T>);

impl<T> RunStream<T> {
    fn new() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T> dbr::Benchmark for RunStream<T>
where
    T: di::repr::FullPrecision + diskann::graph::SampleableForStart + AsDataType,
{
    type Input = Stream;
    type Output = inmem2::benchmark::bigann::Stats;

    fn try_match(
        &self,
        input: &Stream,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        let mut score = context.success(0);
        match_data_type::<T>(&mut score, input.stream.data_type());
        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "full-precision streaming with data type {}",
            Quote(T::DATA_TYPE)
        )?;

        Ok(())
    }

    fn run(
        &self,
        input: &Stream,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}\n", input)?;

        let Stream { stream, raw: _ } = input;

        // Load data.
        let data = stream.load_data::<T>(Some(output))?;

        // Compute the medoid of the dataset as the single start point.
        let start = diskann::graph::StartPointStrategy::Medoid.compute(data.as_view())?;
        let config = di::repr::Full::config(
            di::num::Capacity::new(stream.max_points()),
            di::num::MaxDegree::new(stream.max_degree()),
            stream.distance(),
            start,
        )?;

        let provider = di::Provider::<_, u32>::new(config)?;
        let engine = stream.engine(provider)?;
        let runner = stream.runner(data)?;
        let stats = runner.run(engine, output)?;

        writeln!(output, "{}", stats)?;
        Ok(stats)
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
        assert!(BuildAndSearch::from_raw_with(&dto::BuildAndSearch::example(), None,).is_ok());
    }
}
