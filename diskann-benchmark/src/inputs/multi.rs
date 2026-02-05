/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZero;

use serde::{Deserialize, Serialize};

use diskann::graph::{config, StartPointStrategy};
use diskann_benchmark_runner::{files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker};
use diskann_providers::model::graph::provider::async_::inmem::DefaultProviderParameters;

use super::async_::{StartPointStrategyRef, TopkSearchPhase};
use crate::inputs::{as_input, Input, Example};

const METRIC: diskann_vector::distance::Metric = diskann_vector::distance::Metric::InnerProduct;

as_input!(BuildAndSearch);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<BuildAndSearch>::new())?;
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Build {
    pub(crate) data_type: DataType,
    pub(crate) data: InputFile,
    pub(crate) pruned_degree: usize,
    pub(crate) max_degree: usize,
    pub(crate) l_build: usize,
    #[serde(with = "StartPointStrategyRef")]
    pub(crate) start_point_strategy: StartPointStrategy,
    pub(crate) alpha: f32,
    pub(crate) backedge_ratio: f32,
    pub(crate) num_threads: usize,
}

impl Build {
    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        Ok(config::Builder::new_with(
            self.pruned_degree,
            config::MaxDegree::new(self.max_degree),
            self.l_build,
            (METRIC).into(),
            |b| {
                b
                    .alpha(self.alpha)
                    .backedge_ratio(self.backedge_ratio);
            }
        ))
    }

    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> DefaultProviderParameters {
        DefaultProviderParameters {
            max_points: num_points,
            frozen_points: NonZero::new(self.start_point_strategy.count()).unwrap(),
            metric: METRIC,
            dim,
            max_degree: self.max_degree as u32,
            prefetch_lookahead: None,
            prefetch_cache_line_level: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BuildAndSearch {
    pub(crate) build: Build,
    pub(crate) search: TopkSearchPhase,
}

// Constant for aligning field descriptions in Display implementations.
const PRINT_WIDTH: usize = 18;

macro_rules! write_field {
    ($f:ident, $field:tt, $($expr:tt)*) => {
        writeln!($f, "{:>PRINT_WIDTH$}: {}", $field, $($expr)*)
    }
}

impl Build {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-build"
    }
}

impl Example for Build {
    fn example() -> Self {
        Self {
            data_type: DataType::Float32,
            data: InputFile::new("path/to/data"),
            pruned_degree: 32,
            max_degree: 64,
            l_build: 50,
            start_point_strategy: StartPointStrategy::Medoid,
            alpha: 1.2,
            backedge_ratio: 1.0,
            num_threads: 1,
        }
    }
}

impl CheckDeserialization for Build {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.check_deserialization(checker)?;
        if self.pruned_degree > self.max_degree {
            anyhow::bail!(
                "Pruned degree ({}) must be less than max degree ({})",
                self.pruned_degree,
                self.max_degree,
            );
        }

        Ok(())
    }
}

impl std::fmt::Display for Build {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Index Build\n")?;
        write_field!(f, "tag", Self::tag())?;
        write_field!(f, "file", self.data.display())?;
        write_field!(f, "data_type", self.data_type)?;
        write_field!(f, "pruned degree", self.pruned_degree)?;
        write_field!(f, "max degree", self.max_degree)?;
        write_field!(f, "L-build", self.l_build)?;
        write_field!(f, "start point strategy", self.start_point_strategy)?;
        write_field!(f, "alpha", self.alpha)?;
        write_field!(f, "backedge ratio", self.backedge_ratio)?;
        write_field!(f, "build threads", self.num_threads)?;
        Ok(())
    }
}

impl BuildAndSearch {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-build-and-search"
    }
}

impl Example for BuildAndSearch {
    fn example() -> Self {
        Self {
            build: Build::example(),
            search: TopkSearchPhase::example(),
        }
    }
}

impl CheckDeserialization for BuildAndSearch {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.build.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;
        Ok(())
    }
}

impl std::fmt::Display for BuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Multi-Vector Index Build and Search\n")?;
        write_field!(f, "tag", Self::tag())?;
        write!(f, "{}", self.build)?;
        Ok(())
    }
}
