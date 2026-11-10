/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
};

use diskann_benchmark_runner::{
    self as dbr,
    report::Finished,
    utils::{datatype::AsDataType, fmt::Quote},
};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::Matrix;

use crate::{
    index::inmem2::{self, common::FromRawWith},
    utils::match_data_type,
};

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register_regression(
        "inmem2-f32",
        RunBuildAndSearch::<f32>::new(Some(save_provider)),
    )?;
    registry.register_regression("inmem2-f16", RunBuildAndSearch::<half::f16>::new(None))?;
    registry.register_regression("inmem2-u8", RunBuildAndSearch::<u8>::new(None))?;

    registry.register("inmem2-f32-search", RunLoadAndSearch)?;
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub(super) save_path: Option<PathBuf>,
    }

    impl BuildAndSearch {
        pub(super) fn example() -> Self {
            Self {
                data: inmem2::common::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::Knn::example(),
                save_path: None,
            }
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(super) struct LoadAndSearch {
        pub(super) load_path: dbr::files::InputFile,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub(super) capacity: Option<usize>,
        pub(super) search: inmem2::benchmark::knn::dto::Knn,
    }

    impl LoadAndSearch {
        pub(super) fn example() -> Self {
            Self {
                load_path: dbr::files::InputFile::new("path/to/metadata.json"),
                capacity: None,
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
    save_path: Option<PathBuf>,
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
            save_path,
        } = raw;

        let data = inmem2::common::Data::from_raw_with(data, checker.as_deref_mut())?;
        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance())?;
        let save_path = if let Some(path) = save_path {
            anyhow::ensure!(
                data.data_type() == dbr::utils::datatype::DataType::Float32,
                "inmem2 persistence currently supports only float32 vectors",
            );
            let filename = path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("save_path must name a manifest file"))?;
            Some(match checker.as_deref_mut() {
                Some(checker) => checker.register_output(path.parent())?.join(filename),
                None => path.clone(),
            })
        } else {
            None
        };
        let search = inmem2::benchmark::knn::Knn::from_raw_with(search, checker)?;

        Ok(Self {
            data,
            build,
            search,
            save_path,
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
            save_path,
            raw: _,
        } = self;
        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        let save_path = save_path.as_ref().map(|path| path.display());
        if let Some(save_path) = &save_path {
            kv.push("save_path", save_path);
        }
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

type SaveProvider<T> = fn(&inmem2::Provider<di::repr::Full<T>>, &Path) -> anyhow::Result<()>;

fn manifest_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn save_provider(
    provider: &inmem2::Provider<di::repr::Full<f32>>,
    path: &Path,
) -> anyhow::Result<()> {
    diskann_record::save::save_to_disk(provider, manifest_dir(path), path)?;
    Ok(())
}

#[derive(Debug)]
struct RunBuildAndSearch<T: 'static> {
    save: Option<SaveProvider<T>>,
}

impl<T: 'static> RunBuildAndSearch<T> {
    fn new(save: Option<SaveProvider<T>>) -> Self {
        Self { save }
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
        if let Some(path) = &input.save_path {
            let save = self
                .save
                .ok_or_else(|| anyhow::anyhow!("saving this data type is not supported"))?;
            save(engine.index().provider(), path)?;
            writeln!(output, "Saved provider to {}", path.display())?;
        }
        let search = input
            .search
            .search::<T, _>(engine.index().clone(), Some(output))?;

        Ok(Self::Output { build, search })
    }
}

impl<T> dbr::benchmark::Regression for RunBuildAndSearch<T>
where
    T: di::repr::FullPrecision + diskann::graph::SampleableForStart + AsDataType,
{
    type Tolerances = inmem2::benchmark::Tolerance;
    type Pass = Finished;
    type Fail = Finished;

    fn check(
        &self,
        tolerances: &Self::Tolerances,
        _input: &Self::Input,
        before: &Self::Output,
        after: &Self::Output,
    ) -> anyhow::Result<dbr::benchmark::PassFail<Finished, Finished>> {
        Ok(tolerances.check(before, after).pass_fail())
    }
}

//-------------//
// Load/Search //
//-------------//

#[derive(Debug)]
struct LoadAndSearch {
    load_path: dbr::files::InputFile,
    capacity: Option<usize>,
    search: inmem2::benchmark::knn::Knn,
    raw: serde_json::Value,
}

impl FromRawWith for LoadAndSearch {
    type Raw = dto::LoadAndSearch;

    fn from_raw_with(
        raw: &Self::Raw,
        mut checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let mut load_path = raw.load_path.clone();
        if let Some(checker) = checker.as_deref_mut() {
            load_path.resolve(checker)?;
        }
        let search = inmem2::benchmark::knn::Knn::from_raw_with(&raw.search, checker)?;
        Ok(Self {
            load_path,
            capacity: raw.capacity,
            search,
            raw: serde_json::to_value(raw)?,
        })
    }
}

impl std::fmt::Display for LoadAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut kv = dbr::utils::fmt::KeyValue::new();
        kv.push("load_path", &self.load_path);
        if let Some(capacity) = &self.capacity {
            kv.push("capacity", capacity);
        }
        kv.push("search", &self.search);
        write!(f, "{}", kv)
    }
}

impl dbr::Input for LoadAndSearch {
    type Raw = dto::LoadAndSearch;

    fn tag() -> &'static str {
        "inmem2-fp-search"
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

struct LoadIndex {
    capacity: Option<usize>,
}

impl<'a> diskann_record::load::Loader<'a> for LoadIndex {
    type Value = inmem2::Index<di::repr::Full<f32>>;

    fn load(
        self,
        context: diskann_record::load::Context<'a>,
    ) -> diskann_record::load::Result<Self::Value> {
        use diskann_record::load::{error::Kind, Error};

        let start = std::time::Instant::now();
        let mut loader = di::FullProviderLoader::new();
        if let Some(capacity) = self.capacity {
            loader = loader.capacity(di::num::Capacity::new(capacity));
        }
        let provider = context.load_with(loader)?;
        eprintln!(
            "t={:.3} Finished loading provider.",
            start.elapsed().as_secs_f64()
        );
        let object = context.as_object().ok_or(Kind::TypeMismatch)?;
        let metric: diskann_vector::distance::Metric = object
            .field::<&str>("metric")?
            .parse()
            .map_err(Error::new)?;
        let max_degree = provider.max_degree().value().max(1);
        let config = diskann::graph::config::Builder::new(
            max_degree,
            diskann::graph::config::MaxDegree::same(),
            1,
            metric.into(),
        )
        .build()
        .map_err(Error::new)?;
        eprintln!(
            "t={:.3} Finished loading graph.",
            start.elapsed().as_secs_f64()
        );
        let index = diskann::graph::DiskANNIndex::new(config, provider, None);
        eprintln!(
            "t={:.3} Finished creating index.",
            start.elapsed().as_secs_f64()
        );
        Ok(index)
    }
}

#[derive(Debug)]
struct RunLoadAndSearch;

impl dbr::Benchmark for RunLoadAndSearch {
    type Input = LoadAndSearch;
    type Output = inmem2::benchmark::knn::Stats;

    fn try_match(
        &self,
        _input: &Self::Input,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "load and search a saved full-precision float32 provider")
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}\n", input)?;
        let start = std::time::Instant::now();
        let index = diskann_record::load::load_from_disk_with(
            &input.load_path,
            manifest_dir(&input.load_path),
            LoadIndex {
                capacity: input.capacity,
            },
        )?;
        writeln!(
            output,
            "Finished loading index in {:.3} seconds\n",
            start.elapsed().as_secs_f64(),
        )?;
        input.search.search::<f32, _>(Arc::new(index), Some(output))
    }
}

//-----------//
// Streaming //
//-----------//

#[derive(Debug)]
struct Stream {
    stream: inmem2::benchmark::bigann::Streaming,
    // The raw input.
    raw: serde_json::Value,
}

impl std::fmt::Display for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stream.fmt(f)
    }
}

impl dbr::Input for Stream {
    type Raw = inmem2::benchmark::bigann::dto::Streaming;
    fn tag() -> &'static str {
        "inmem2-fp-bigann-stream"
    }

    fn from_raw(raw: Self::Raw, checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        let stream = inmem2::benchmark::bigann::Streaming::from_raw_with(&raw, Some(checker))?;
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
        assert!(LoadAndSearch::from_raw_with(&dto::LoadAndSearch::example(), None).is_ok());
        let example: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/example/inmem2-search.json",
        )))
        .unwrap();
        let raw = serde_json::from_value(example["jobs"][0]["content"].clone()).unwrap();
        assert!(LoadAndSearch::from_raw_with(&raw, None).is_ok());
    }

    #[test]
    fn test_save_path_is_optional_and_float32_only() {
        let mut raw = dto::BuildAndSearch::example();
        let json = serde_json::to_value(&raw).unwrap();
        assert!(json.get("save_path").is_none());
        assert!(serde_json::from_value::<dto::BuildAndSearch>(json)
            .unwrap()
            .save_path
            .is_none());
        raw.save_path = Some("index.json".into());
        assert!(BuildAndSearch::from_raw_with(&raw, None).is_ok());
        for data_type in [
            dbr::utils::datatype::DataType::Float16,
            dbr::utils::datatype::DataType::UInt8,
        ] {
            let mut json = serde_json::to_value(&raw).unwrap();
            json["data"]["data_type"] = serde_json::to_value(data_type).unwrap();
            let raw = serde_json::from_value(json).unwrap();
            assert!(BuildAndSearch::from_raw_with(&raw, None).is_err());
        }
    }
}
