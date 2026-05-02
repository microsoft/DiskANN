/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use diskann::graph::{config, StartPointStrategy};
use diskann_benchmark_runner::{
    files::InputFile, utils::datatype::DataType, CheckDeserialization, Checker,
};
use diskann_providers::model::graph::provider::async_::inmem::DefaultProviderParameters;

use super::async_::{StartPointStrategyRef, TopkSearchPhase};
use super::exhaustive;
use crate::inputs::{as_input, Example, Input};

const METRIC: diskann_vector::distance::Metric = diskann_vector::distance::Metric::InnerProduct;

as_input!(BuildAndSearch);
as_input!(ExhaustiveSearch);
as_input!(SphericalRerankBuildAndSearch);

pub(super) fn register_inputs(
    registry: &mut diskann_benchmark_runner::registry::Inputs,
) -> anyhow::Result<()> {
    registry.register(Input::<BuildAndSearch>::new())?;
    registry.register(Input::<ExhaustiveSearch>::new())?;
    registry.register(Input::<SphericalRerankBuildAndSearch>::new())?;
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
    pub(crate) num_threads: NonZeroUsize,
    /// Distance method for graph construction. Defaults to `mean_pool`.
    #[serde(default)]
    pub(crate) method: DistanceMethod,
    /// Query layout for spherical distance computation (only used when
    /// `method` is `spherical_chamfer`).
    #[serde(default)]
    pub(crate) query_layout: Option<exhaustive::SphericalQuery>,
}

impl Build {
    pub(crate) fn as_config(&self) -> config::Builder {
        config::Builder::new_with(
            self.pruned_degree,
            config::MaxDegree::new(self.max_degree),
            self.l_build,
            (METRIC).into(),
            |b| {
                b.alpha(self.alpha).backedge_ratio(self.backedge_ratio);
            },
        )
    }

    pub(crate) fn inmem_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> DefaultProviderParameters {
        DefaultProviderParameters {
            max_points: num_points,
            frozen_points: NonZeroUsize::new(self.start_point_strategy.count()).unwrap(),
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
            num_threads: diskann::utils::ONE,
            method: DistanceMethod::MeanPool,
            query_layout: None,
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
        write_field!(f, "method", self.method)?;
        if let Some(ql) = &self.query_layout {
            write_field!(f, "query_layout", ql)?;
        }
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

///////////////////////
// Exhaustive Search //
///////////////////////

/// Input for exhaustive (brute-force) multi-vector KNN search using Chamfer distance.
///
/// This computes exact K-nearest neighbors for each query and writes the results
/// to a binary groundtruth file.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExhaustiveSearch {
    /// Path to the multi-vector dataset file.
    pub(crate) data: InputFile,
    /// Path to the multi-vector queries file.
    pub(crate) queries: InputFile,
    /// Path for the output groundtruth file (.bin format).
    pub(crate) output: String,
    /// Number of nearest neighbors to find per query.
    pub(crate) num_nearest_neighbors: usize,
    /// Number of threads for parallel search.
    pub(crate) num_threads: NonZeroUsize,
}

impl ExhaustiveSearch {
    pub(crate) const fn tag() -> &'static str {
        "exhaustive-multi-vector"
    }
}

impl Example for ExhaustiveSearch {
    fn example() -> Self {
        Self {
            data: InputFile::new("path/to/data.mvbin"),
            queries: InputFile::new("path/to/queries.mvbin"),
            output: "groundtruth.bin".to_string(),
            num_nearest_neighbors: 100,
            num_threads: NonZeroUsize::new(8).unwrap(),
        }
    }
}

impl CheckDeserialization for ExhaustiveSearch {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.data.check_deserialization(checker)?;
        self.queries.check_deserialization(checker)?;

        if self.num_nearest_neighbors == 0 {
            anyhow::bail!("num_nearest_neighbors must be greater than 0");
        }

        // Resolve output path relative to output directory if set
        let output_path = std::path::Path::new(&self.output);
        let output_filename = output_path
            .file_name()
            .unwrap_or_else(|| output_path.as_os_str());
        let resolved_path = checker.register_output(output_path.parent())?;
        let full_path = resolved_path.join(output_filename);
        self.output = full_path.to_string_lossy().to_string();

        Ok(())
    }
}

impl std::fmt::Display for ExhaustiveSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Exhaustive Multi-Vector Search (Chamfer Distance)\n")?;
        write_field!(f, "tag", Self::tag())?;
        write_field!(f, "data", self.data.display())?;
        write_field!(f, "queries", self.queries.display())?;
        write_field!(f, "output", self.output)?;
        write_field!(f, "number of nearest neighbors", self.num_nearest_neighbors)?;
        write_field!(f, "number of threads", self.num_threads)?;
        Ok(())
    }
}

//////////////////////////////////////////////
// Configurable Multi-Vector Build & Search //
//////////////////////////////////////////////

/// Spherical quantization configuration for multi-vector search.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SphericalConfig {
    pub(crate) num_bits: NonZeroUsize,
    pub(crate) seed: u64,
    pub(crate) transform_kind: exhaustive::TransformKind,
    #[serde(default)]
    pub(crate) pre_scale: Option<exhaustive::PreScale>,
}

impl std::fmt::Display for SphericalConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "num_bits", self.num_bits)?;
        write_field!(f, "seed", self.seed)?;
        write_field!(f, "transform_kind", self.transform_kind)?;
        write_field!(
            f,
            "pre_scale",
            self.pre_scale
                .as_ref()
                .unwrap_or(&exhaustive::PreScale::None)
        )?;
        Ok(())
    }
}

/// Reranking method configuration for multi-vector search.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub(crate) enum RerankMethod {
    /// Full-precision Chamfer distance using the index's stored multi-vectors.
    Chamfer,
    /// Full-precision Chamfer distance using a separate (side-loaded) multi-vector dataset.
    SideloadedChamfer {
        /// Path to the multi-vector data for reranking.
        data: InputFile,
        /// Optional separate query multi-vectors for reranking.
        #[serde(default)]
        query_data: Option<InputFile>,
    },
    /// Spherical-quantized Chamfer distance (requires `spherical` config).
    SphericalChamfer {
        /// Multi-vector data to quantize for reranking.
        /// If absent, uses the build data.
        #[serde(default)]
        data: Option<InputFile>,
        /// Optional separate query multi-vectors for reranking.
        #[serde(default)]
        query_data: Option<InputFile>,
        /// Query layouts to evaluate.
        query_layouts: Vec<exhaustive::SphericalQuery>,
    },
    /// No reranking — pass through first-stage candidate ordering.
    None,
}

/// Inner-loop (first-stage graph search) method configuration.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub(crate) enum InnerLoopMethod {
    /// Mean-pooled inner product distance (default).
    MeanPool,
    /// Spherical-quantized Chamfer distance over compressed sub-vectors.
    SphericalChamfer {
        /// Multi-vector data for the inner loop. If absent, uses the build data.
        #[serde(default)]
        data: Option<InputFile>,
        /// Optional separate query multi-vectors for the inner loop.
        #[serde(default)]
        query_data: Option<InputFile>,
        /// Query layout to use for spherical distance computation.
        query_layout: exhaustive::SphericalQuery,
    },
}

impl Default for InnerLoopMethod {
    fn default() -> Self {
        Self::MeanPool
    }
}

impl std::fmt::Display for InnerLoopMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeanPool => write!(f, "mean_pool"),
            Self::SphericalChamfer {
                data,
                query_data,
                query_layout,
            } => {
                write!(f, "spherical_chamfer (layout: {})", query_layout)?;
                if let Some(d) = data {
                    write!(f, " (data: {})", d.display())?;
                }
                if let Some(qd) = query_data {
                    write!(f, " (queries: {})", qd.display())?;
                }
                Ok(())
            }
        }
    }
}

/// Distance method selector used by the build stage.
///
/// - `mean_pool`: computes a mean-pooled vector and uses inner product.
/// - `spherical_chamfer`: trains a spherical quantizer and uses Chamfer distance
///   over compressed sub-vectors. Requires `query_layout` and a `spherical` config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DistanceMethod {
    MeanPool,
    SphericalChamfer,
}

impl Default for DistanceMethod {
    fn default() -> Self {
        Self::MeanPool
    }
}

impl std::fmt::Display for DistanceMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MeanPool => write!(f, "mean_pool"),
            Self::SphericalChamfer => write!(f, "spherical_chamfer"),
        }
    }
}

impl std::fmt::Display for RerankMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chamfer => write!(f, "chamfer (in-index)"),
            Self::SideloadedChamfer { data, query_data } => {
                write!(f, "chamfer (side-loaded: {})", data.display())?;
                if let Some(qd) = query_data {
                    write!(f, ", queries: {}", qd.display())?;
                }
                Ok(())
            }
            Self::SphericalChamfer {
                data,
                query_data,
                query_layouts,
            } => {
                write!(f, "spherical chamfer")?;
                if let Some(d) = data {
                    write!(f, " (data: {})", d.display())?;
                }
                if let Some(qd) = query_data {
                    write!(f, " (queries: {})", qd.display())?;
                }
                write!(f, " layouts: {:?}", query_layouts)?;
                Ok(())
            }
            Self::None => write!(f, "none (pass-through)"),
        }
    }
}

/// Input for a multi-vector index build and search with independently configurable
/// build, first-stage search (inner loop), and reranking (outer loop) strategies.
///
/// Each stage can use mean-pooled vectors or spherical-quantized Chamfer distance,
/// potentially with different data sources (e.g., reduced vs. full multi-vector sets).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SphericalRerankBuildAndSearch {
    /// Build parameters (data, graph config, distance method).
    pub(crate) build: Build,
    /// Search parameters (queries, groundtruth, threads, runs).
    pub(crate) search: TopkSearchPhase,
    /// Inner-loop (first-stage search) method. Defaults to mean-pooled IP.
    #[serde(default)]
    pub(crate) inner_loop: InnerLoopMethod,
    /// Reranking method and data configuration.
    pub(crate) rerank: RerankMethod,
    /// Spherical quantization config (required when using spherical methods).
    #[serde(default)]
    pub(crate) spherical: Option<SphericalConfig>,
}

impl SphericalRerankBuildAndSearch {
    pub(crate) const fn tag() -> &'static str {
        "multi-vector-build-and-search-spherical-rerank"
    }
}

impl Example for SphericalRerankBuildAndSearch {
    fn example() -> Self {
        Self {
            build: Build::example(),
            search: TopkSearchPhase::example(),
            inner_loop: InnerLoopMethod::MeanPool,
            rerank: RerankMethod::SphericalChamfer {
                data: None,
                query_data: None,
                query_layouts: vec![exhaustive::SphericalQuery::FullPrecision],
            },
            spherical: Some(SphericalConfig {
                num_bits: NonZeroUsize::new(1).unwrap(),
                seed: 0xc0ffee,
                transform_kind: exhaustive::TransformKind::PaddingHadamard(
                    exhaustive::TargetDim::Same,
                ),
                pre_scale: None,
            }),
        }
    }
}

impl CheckDeserialization for SphericalRerankBuildAndSearch {
    fn check_deserialization(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.build.check_deserialization(checker)?;
        self.search.check_deserialization(checker)?;

        // Validate build method requires spherical config if using spherical_chamfer.
        if self.build.method == DistanceMethod::SphericalChamfer {
            let query_layout = self.build.query_layout.ok_or_else(|| {
                anyhow::anyhow!(
                    "spherical_chamfer build method requires a `query_layout` field in `build`"
                )
            })?;
            let spherical = self.spherical.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "spherical_chamfer build method requires a `spherical` configuration block"
                )
            })?;
            exhaustive::check_compatibility(spherical.num_bits.get(), query_layout)
                .context("while validating build method query layout")?;
        }

        // Validate inner-loop config.
        match &mut self.inner_loop {
            InnerLoopMethod::MeanPool => {}
            InnerLoopMethod::SphericalChamfer {
                data,
                query_data,
                query_layout,
            } => {
                if let Some(d) = data {
                    d.check_deserialization(checker)?;
                }
                if let Some(qd) = query_data {
                    qd.check_deserialization(checker)?;
                }
                let spherical = self.spherical.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "spherical_chamfer inner loop requires a `spherical` configuration block"
                    )
                })?;
                exhaustive::check_compatibility(spherical.num_bits.get(), *query_layout)
                    .context("while validating inner loop query layout")?;
            }
        }

        // Validate rerank-specific paths.
        match &mut self.rerank {
            RerankMethod::Chamfer | RerankMethod::None => {}
            RerankMethod::SideloadedChamfer { data, query_data } => {
                data.check_deserialization(checker)?;
                if let Some(qd) = query_data {
                    qd.check_deserialization(checker)?;
                }
            }
            RerankMethod::SphericalChamfer {
                data,
                query_data,
                query_layouts,
            } => {
                if let Some(d) = data {
                    d.check_deserialization(checker)?;
                }
                if let Some(qd) = query_data {
                    qd.check_deserialization(checker)?;
                }
                let spherical = self.spherical.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "spherical_chamfer rerank requires a `spherical` configuration block"
                    )
                })?;
                for (i, layout) in query_layouts.iter().enumerate() {
                    exhaustive::check_compatibility(spherical.num_bits.get(), *layout)
                        .with_context(|| {
                            format!(
                                "while validating rerank query layout {} of {}",
                                i + 1,
                                query_layouts.len()
                            )
                        })?;
                }
            }
        }

        if let Some(ref mut spherical) = self.spherical {
            if let Some(ref mut pre_scale) = spherical.pre_scale {
                pre_scale.check_deserialization(checker)?;
            }
        }

        Ok(())
    }
}

impl std::fmt::Display for SphericalRerankBuildAndSearch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Multi-Vector Index Build and Search (Configurable Rerank)\n"
        )?;
        write_field!(f, "tag", Self::tag())?;
        write!(f, "{}", self.build)?;
        write_field!(f, "inner_loop", self.inner_loop)?;
        write_field!(f, "rerank", self.rerank)?;
        if let Some(ref spherical) = self.spherical {
            writeln!(f)?;
            writeln!(f, "  Spherical Quantization:")?;
            write!(f, "{}", spherical)?;
        }
        Ok(())
    }
}
