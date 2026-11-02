/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_benchmark_runner::{self as dbr, report::Report};

pub(in crate::index::inmem2) mod bigann;
pub(in crate::index::inmem2) mod build;
pub(in crate::index::inmem2) mod knn;

/// Measurements shared by the static build-and-search benchmarks.
///
/// Build statistics cover insertion, excluding data loading and quantizer training.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::index::inmem2) struct BuildAndSearchStats {
    pub(in crate::index::inmem2) build: build::Stats,
    pub(in crate::index::inmem2) search: knn::Stats,
}

mod dto {
    use super::*;

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    pub(in crate::index::inmem2) struct Tolerance {
        pub(super) build: build::dto::Tolerance,
        pub(super) search: knn::dto::Tolerance,
    }

    impl Tolerance {
        pub(in crate::index::inmem2) fn example() -> Self {
            Self {
                build: build::dto::Tolerance::example(),
                search: knn::dto::Tolerance::example(),
            }
        }
    }
}

#[derive(Debug)]
pub(in crate::index::inmem2) struct Tolerance {
    build: build::Tolerance,
    search: knn::Tolerance,
    // Used for serialization.
    raw: dto::Tolerance,
}

impl Tolerance {
    pub(in crate::index::inmem2) fn check(
        &self,
        old: &BuildAndSearchStats,
        new: &BuildAndSearchStats,
    ) -> Report {
        let mut builder = Report::builder();
        builder.push("build", self.build.check(&old.build, &new.build));
        builder.push("search", self.search.check(&old.search, &new.search));
        builder.finish()
    }
}

impl dbr::Input for Tolerance {
    type Raw = dto::Tolerance;

    fn tag() -> &'static str {
        "inmem2-build-and-search-tolerance"
    }

    fn from_raw(raw: dto::Tolerance, _checker: &mut dbr::Checker) -> anyhow::Result<Self> {
        let dto::Tolerance { build, search } = &raw;
        Ok(Self {
            build: build::Tolerance::from_raw(build),
            search: knn::Tolerance::from_raw(search),
            raw,
        })
    }

    fn serialize(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(&self.raw)?)
    }

    fn example() -> Self::Raw {
        dto::Tolerance::example()
    }
}
