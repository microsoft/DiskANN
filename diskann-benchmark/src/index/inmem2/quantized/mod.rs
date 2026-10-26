/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! [`diskann_inmem`]'s quantizer interface is rather uniform. This structure is reflected
//! here with common [`dbr::Input`] and [`dbr::Benchmark`] infrastructure with slight
//! variations to support the different quantizer's training needs.

use std::{io::Write, marker::PhantomData, sync::Arc};

use diskann_benchmark_runner::{self as dbr};
use diskann_inmem::{self as di};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;
use serde::{de::DeserializeOwned, Serialize};

use crate::index::inmem2::{self, common::FromRawWith};

mod product;
mod scalar;
mod spherical;

pub(super) fn register_benchmarks(registry: &mut dbr::Registry) -> anyhow::Result<()> {
    registry.register(
        "inmem2-sphq-static",
        RunBuildAndSearch::<spherical::Spherical>::new(),
    )?;
    registry.register(
        "inmem2-pq-static",
        RunBuildAndSearch::<product::Product>::new(),
    )?;
    registry.register(
        "inmem2-sq-static",
        RunBuildAndSearch::<scalar::Scalar>::new(),
    )?;
    Ok(())
}

/// The quantizer trait used by the various backends.
trait Quantizer: std::fmt::Debug + std::fmt::Display + Sized + 'static {
    /// The [`dbr::Input::tag`] to use for static build and search benchmarks.
    const TAG: &'static str;

    /// A short tag (e.g. "product", "scalar", or "spherical") used for pretty formatting.
    const KIND: &'static str;

    /// The raw JSON configuration for the quantizer.
    type Raw: Serialize + DeserializeOwned;

    /// Construct the quantizer from the raw representation and dataset metric.
    fn from_raw(raw: &Self::Raw, metric: Metric) -> anyhow::Result<Self>;

    /// Emit the example raw struct for the UI.
    fn example() -> Self::Raw;

    /// Train the quantizer from raw data.
    fn train(&self, data: rowmajor::Ref<'_, f32>) -> anyhow::Result<di::repr::quantized::Backend>;
}

pub(super) mod dto {
    use super::*;

    use serde::Deserialize;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct BuildAndSearch<Q> {
        pub(super) data: inmem2::common::dto::Data,
        pub(super) build: inmem2::benchmark::build::dto::Config,
        pub(super) search: inmem2::benchmark::knn::dto::Knn,
        pub(super) quantizer: Q,
    }

    impl<Q> BuildAndSearch<Q> {
        pub(super) fn example(quantizer: Q) -> Self {
            Self {
                data: inmem2::common::dto::Data::example(),
                build: inmem2::benchmark::build::dto::Config::example(),
                search: inmem2::benchmark::knn::dto::Knn::example(),
                quantizer,
            }
        }
    }
}

#[derive(Debug)]
struct BuildAndSearch<Q> {
    data: inmem2::common::Data,
    build: inmem2::benchmark::build::Config,
    search: inmem2::benchmark::knn::Knn,
    quantizer: Q,
    // The raw input.
    raw: serde_json::Value,
}

impl<Q> FromRawWith for BuildAndSearch<Q>
where
    Q: Quantizer,
{
    type Raw = dto::BuildAndSearch<Q::Raw>;

    fn from_raw_with(
        raw: &dto::BuildAndSearch<Q::Raw>,
        mut checker: Option<&mut dbr::Checker>,
    ) -> anyhow::Result<Self> {
        let dto::BuildAndSearch {
            data,
            build,
            search,
            quantizer,
        } = raw;

        let data = inmem2::common::Data::from_raw_with(data, checker.as_deref_mut())?;
        if data.data_type() != dbr::utils::datatype::DataType::Float32 {
            anyhow::bail!("quantization only supports f32 data");
        }

        let build = inmem2::benchmark::build::Config::from_raw(build, data.distance())?;
        let search = inmem2::benchmark::knn::Knn::from_raw_with(search, checker)?;
        let quantizer = Q::from_raw(quantizer, data.distance())?;

        Ok(Self {
            data,
            build,
            search,
            quantizer,
            raw: serde_json::to_value(raw)?,
        })
    }
}

impl<Q> std::fmt::Display for BuildAndSearch<Q>
where
    Q: Quantizer,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            data,
            build,
            search,
            quantizer,
            raw: _,
        } = self;

        let mut kv = dbr::utils::fmt::KeyValue::new();

        kv.push("data", &data);
        kv.push("build", &build);
        kv.push("search", &search);
        let quantizer_key = format!("quantizer ({})", Q::KIND);
        kv.push(&quantizer_key, &quantizer);
        kv.fmt(f)
    }
}

impl<Q> dbr::Input for BuildAndSearch<Q>
where
    Q: Quantizer,
{
    type Raw = dto::BuildAndSearch<Q::Raw>;
    fn tag() -> &'static str {
        Q::TAG
    }

    fn from_raw(raw: Self::Raw, checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        Self::from_raw_with(&raw, Some(checker))
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(self.raw.clone())
    }

    fn example() -> Self::Raw {
        Self::Raw::example(Q::example())
    }
}

#[derive(Debug)]
struct RunBuildAndSearch<Q>(PhantomData<Q>);

impl<Q> RunBuildAndSearch<Q> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<Q> dbr::Benchmark for RunBuildAndSearch<Q>
where
    Q: Quantizer,
{
    type Input = BuildAndSearch<Q>;
    type Output = inmem2::benchmark::BuildAndSearchStats;

    fn try_match(
        &self,
        _input: &BuildAndSearch<Q>,
        context: &dbr::benchmark::MatchContext,
    ) -> dbr::benchmark::Score {
        // Distance compatibility and data-types are validated during deserialization, so
        // there's nothing more to do here.
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-quantized static build+search", Q::KIND)?;
        Ok(())
    }

    fn run(
        &self,
        input: &BuildAndSearch<Q>,
        _checkpoint: dbr::Checkpoint<'_>,
        mut output: &mut dyn dbr::Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}\n", input)?;

        // Load Data
        let data = input.data.load_arc::<f32>(Some(output))?;
        let backend = input.quantizer.train(data.as_view())?;

        run_quantized(
            backend,
            data,
            &input.data,
            &input.build,
            &input.search,
            output,
        )
    }
}

// An outlined helper for [`RunBuildAndSearch::run`] with concrete types.
fn run_quantized(
    backend: di::repr::quantized::Backend,
    dataset: Arc<rowmajor::Owned<f32>>,
    data: &inmem2::common::Data,
    build: &inmem2::benchmark::build::Config,
    search: &inmem2::benchmark::knn::Knn,
    output: &mut dyn dbr::Output,
) -> anyhow::Result<inmem2::benchmark::BuildAndSearchStats> {
    let start = diskann::graph::StartPointStrategy::Medoid.compute(dataset.as_view())?;
    let config = di::repr::Quantized::config(
        backend,
        di::num::Capacity::new(dataset.nrows()),
        di::num::MaxDegree::new(build.max_degree()),
        start,
        di::repr::quantized::Rerank::F16 {
            metric: data.distance().into(),
        },
    )?;

    let provider = di::Provider::<di::repr::Quantized, u32>::new(config)?;
    let engine = build.engine(provider)?;

    // Run benchmarks.
    let build = engine.build(dataset, Some(output))?;
    let search = search.search::<f32, _>(engine.index().clone(), Some(output))?;

    Ok(inmem2::benchmark::BuildAndSearchStats { build, search })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_parse() {
        assert!(BuildAndSearch::<product::Product>::from_raw_with(
            &dto::BuildAndSearch::example(product::Product::example()),
            None
        )
        .is_ok());

        assert!(BuildAndSearch::<spherical::Spherical>::from_raw_with(
            &dto::BuildAndSearch::example(spherical::Spherical::example()),
            None
        )
        .is_ok());

        assert!(BuildAndSearch::<scalar::Scalar>::from_raw_with(
            &dto::BuildAndSearch::example(scalar::Scalar::example()),
            None
        )
        .is_ok());
    }
}
