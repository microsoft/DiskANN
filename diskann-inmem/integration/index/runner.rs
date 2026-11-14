/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::io::Write;

use diskann_benchmark_runner::{
    Checker, Checkpoint, Output, Registry, RegistryError,
    benchmark::{MatchContext, PassFail, Regression, Score},
    report::{Finished, Report},
    utils::fmt::Indent,
};
use serde::{Deserialize, Serialize};

use super::{
    factory, ops, parse,
    stats::{CheckCounters, CheckKnnStats, Counters, KnnStats},
};
use crate::support::{check_all, datatype::DataType};

pub(super) fn register(registry: &mut Registry) -> Result<(), RegistryError> {
    registry.register_regression("full-integration-test", FullTest)?;
    registry.register_regression("partial-integration-test", PartialTest)?;
    Ok(())
}

////////////////
// Benchmarks //
////////////////

//----------//
// FullTest //
//----------//

/// Integration tests capable of insert, multi-insert, and all the supported search variations.
#[derive(Debug)]
struct FullTest;

impl diskann_benchmark_runner::Benchmark for FullTest {
    type Input = parse::Test;
    type Output = BuildAndSearch;

    fn try_match(&self, input: &parse::Test, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        // Match data-type.
        match input.representation {
            parse::Representation::FullPrecision { .. } => {
                // We match all valid data-types
            }
            parse::Representation::Spherical { .. }
            | parse::Representation::Product { .. }
            | parse::Representation::Scalar { .. } => {
                let data_type = input.data.data_type;
                // Ensure that the data type if `f32`.
                if data_type != DataType::F32 {
                    score.fail(
                        1,
                        &format_args!("quantization requires f32 data, not {}", data_type),
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
        input: &parse::Test,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let parse::Test {
            data,
            representation,
            build,
            search,
        } = input;

        let data_type = match representation {
            parse::Representation::FullPrecision { data_type } => *data_type,
            parse::Representation::Spherical { .. }
            | parse::Representation::Product { .. }
            | parse::Representation::Scalar { .. } => data.data_type,
        };

        // Load the data and perform any necessary data conversions.
        let ops::Bundle {
            data,
            labels,
            queries,
        } = ops::load_data(data, data_type)?;

        let index = factory::create(
            representation,
            input.data.metric,
            &build.config,
            data.as_view(),
        )?;
        let build = match build.insert {
            parse::Insert::Single => {
                let rt = diskann_benchmark_core::tokio::runtime(1)?;
                ops::insert(&*index, data.as_view(), rt.handle())?
            }
            parse::Insert::Multi { batch_size } => {
                let rt = diskann_benchmark_core::tokio::runtime(
                    build.config.max_minibatch_par().into(),
                )?;
                ops::multi_insert(&index, data.as_view(), batch_size, rt.handle())?
            }
        };

        let rt = diskann_benchmark_core::tokio::runtime(1)?;

        let parse::Search {
            knn,
            inline_filter,
            multihop_filter,
        } = search;

        let knn = ops::knn(&*index, knn, queries.as_view(), rt.handle())?;
        let inline_filter = match inline_filter {
            Some(search) => {
                ops::filter_search(&*index, &labels, search.common(), queries.as_view())?
            }
            None => Vec::new(),
        };
        let multihop_filter = match multihop_filter {
            Some(search) => {
                ops::filter_search(&*index, &labels, search.common(), queries.as_view())?
            }
            None => Vec::new(),
        };

        let build_and_search = BuildAndSearch {
            build,
            knn,
            inline_filter,
            multihop_filter,
        };

        writeln!(output, "{}", build_and_search)?;

        Ok(build_and_search)
    }
}

impl Regression for FullTest {
    type Tolerances = CheckBuildAndSearch;
    type Pass = Finished;
    type Fail = Finished;

    fn check(
        &self,
        tolerances: &Self::Tolerances,
        _input: &Self::Input,
        before: &Self::Output,
        after: &Self::Output,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        Ok(tolerances.check(before, after).pass_fail())
    }
}

//-------------//
// PartialTest //
//-------------//

/// Partial integration test supporting only insert and regular KNN search.
///
/// This is used to lightly test the static build path without needing to support the whole
/// range of build and search modes. Done to save compilation time and to keep the static
/// build support limited to just what is needed.
#[derive(Debug)]
struct PartialTest;

impl diskann_benchmark_runner::Benchmark for PartialTest {
    type Input = parse::PartialTest;
    type Output = BuildAndSearch;

    fn try_match(&self, input: &parse::PartialTest, context: &MatchContext) -> Score {
        let mut score = context.success(0);

        // Match data-type.
        let data_type = input.data.data_type;
        if data_type != DataType::F32 {
            score.fail(
                1,
                &format_args!("static-build requires f32 data, not {}", data_type),
            );
        }

        // Don't run multi-insert for static-builds.
        //
        // This should be kept in-sync with backend capabilities.
        match input.build.insert {
            parse::Insert::Single => {}
            parse::Insert::Multi { .. } => {
                score.fail(
                    1,
                    &"multi-insert is not supported for static builds to save compilation time",
                );
            }
        }

        score
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nop")
    }

    fn run(
        &self,
        input: &parse::PartialTest,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        let parse::PartialTest {
            data,
            build,
            search,
        } = input;

        let data_type = data.data_type;

        // Load the data and perform any necessary data conversions.
        let ops::Bundle {
            data,
            labels: _, // unused for partial-tests.
            queries,
        } = ops::load_data(data, data_type)?;

        let index = factory::create_partial(input.data.metric, &build.config, data.as_view())?;

        let rt = diskann_benchmark_core::tokio::runtime(1)?;
        let build = match build.insert {
            parse::Insert::Single => ops::insert(&*index, data.as_view(), rt.handle())?,
            parse::Insert::Multi { .. } => {
                anyhow::bail!("partial-tests do not support multi-insert");
            }
        };

        let rt = diskann_benchmark_core::tokio::runtime(1)?;

        let knn = ops::knn(&*index, search, queries.as_view(), rt.handle())?;

        let build_and_search = BuildAndSearch {
            build,
            knn,
            inline_filter: Vec::new(),
            multihop_filter: Vec::new(),
        };

        writeln!(output, "{}", build_and_search)?;
        Ok(build_and_search)
    }
}

impl Regression for PartialTest {
    type Tolerances = CheckBuildAndSearch;
    type Pass = Finished;
    type Fail = Finished;

    fn check(
        &self,
        tolerances: &Self::Tolerances,
        _input: &Self::Input,
        before: &Self::Output,
        after: &Self::Output,
    ) -> anyhow::Result<PassFail<Self::Pass, Self::Fail>> {
        Ok(tolerances.check(before, after).pass_fail())
    }
}

////////////
// Output //
////////////

#[derive(Debug, Serialize, Deserialize)]
struct BuildAndSearch {
    build: Counters,
    knn: Vec<KnnStats>,
    inline_filter: Vec<KnnStats>,
    multihop_filter: Vec<KnnStats>,
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "build stats")?;
        writeln!(f, "{}", Indent::new(&self.build.to_string(), 4))?;

        writeln!(f, "knn stats")?;
        for k in self.knn.iter() {
            writeln!(f, "{}\n", k)?;
        }

        writeln!(f, "inline filter")?;
        for k in self.inline_filter.iter() {
            writeln!(f, "{}\n", k)?;
        }

        writeln!(f, "multihop filter")?;
        for k in self.multihop_filter.iter() {
            writeln!(f, "{}\n", k)?;
        }

        Ok(())
    }
}

///////////////
// Tolerance //
///////////////

/// A tolerance [`diskann_benchmark_runner::Input`] for
/// [`diskann_benchmark_runner::benchmark::Regression`]s.
///
/// This allows for minor cross-architecture (`aarch64` vs `x86`) due to different choices
/// with floating point SIMD width/association.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CheckBuildAndSearch {
    build: CheckCounters,
    knn: CheckKnnStats,
}

impl CheckBuildAndSearch {
    fn example() -> Self {
        Self {
            build: CheckCounters::example(),
            knn: CheckKnnStats::example(),
        }
    }

    fn check(&self, old: &BuildAndSearch, new: &BuildAndSearch) -> Report {
        let Self {
            build: check_build,
            knn: check_knn,
        } = self;

        let BuildAndSearch {
            build,
            knn,
            inline_filter,
            multihop_filter,
        } = old;

        let mut builder = Report::builder();
        builder.push("build", check_build.check(build, &new.build));
        builder.push(
            "knn",
            check_all(knn, &new.knn, "KNN", |o, n| check_knn.check(o, n)),
        );
        builder.push(
            "inline_filter",
            check_all(
                inline_filter,
                &new.inline_filter,
                "Inline Filter",
                |o, n| check_knn.check(o, n),
            ),
        );
        builder.push(
            "multihop_filter",
            check_all(
                multihop_filter,
                &new.multihop_filter,
                "Multihop Filter",
                |o, n| check_knn.check(o, n),
            ),
        );

        builder.finish()
    }
}

impl diskann_benchmark_runner::Input for CheckBuildAndSearch {
    type Raw = Self;

    fn tag() -> &'static str {
        "tolerance-build-and-search"
    }

    fn from_raw(raw: Self::Raw, _: &mut Checker) -> anyhow::Result<Self> {
        Ok(raw)
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(self)?)
    }

    fn example() -> Self::Raw {
        <Self>::example()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use diskann_benchmark_runner::report::Status;

    #[test]
    fn check_build_and_search_rejects_knn_length_mismatch() {
        let tolerance = CheckBuildAndSearch::example();
        let output = |len| BuildAndSearch {
            build: Counters::test_fixture(),
            knn: (0..len).map(|_| KnnStats::test_fixture()).collect(),
            inline_filter: vec![],
            multihop_filter: vec![],
        };
        for (old_len, new_len, expected) in [
            (1, 1, Status::Pass),
            (1, 0, Status::Fail),
            (0, 1, Status::Fail),
        ] {
            assert_eq!(
                tolerance.check(&output(old_len), &output(new_len)).status(),
                expected,
                "old_len = {old_len}, new_len = {new_len}"
            );
        }
    }
}
