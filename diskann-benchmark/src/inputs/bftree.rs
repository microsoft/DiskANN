/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::num::{NonZero, NonZeroUsize};

use crate::inputs::{
    as_input, exhaustive,
    graph_index::{IndexBuild, SearchPhase, StreamingRunbookParams, TopkSearchPhase},
    write_field, Example, PRINT_WIDTH,
};
use diskann::graph::config;
use diskann_benchmark_runner::{utils::datatype::DataType, Checker};
use diskann_bftree::BfTreeProviderParameters;
use serde::{Deserialize, Serialize};

// ─── BfTree Store Configuration ───────────────────────────────────────────────

/// Configuration for a single bf_tree store instance.
///
/// Required fields control memory sizing before data spills to disk.
/// Optional fields tune internal behavior and default to bf_tree's defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BfTreeStoreConfig {
    /// Size of the circular buffer (in-memory write cache) in bytes.
    pub(crate) cb_size_byte: usize,

    /// Size of leaf pages in bytes.
    pub(crate) leaf_page_size: usize,

    /// Maximum record size that can be stored in the circular buffer.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) cb_max_record_size: Option<usize>,

    /// Minimum record size for the circular buffer.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) cb_min_record_size: Option<usize>,

    /// Probability (0-100) of promoting a read record to the front of the buffer.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) read_promotion_rate: Option<usize>,

    /// Probability (0-100) of promoting a scanned record to the front of the buffer.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) scan_promotion_rate: Option<usize>,

    /// Ratio of buffer used before copy-on-access kicks in.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) cb_copy_on_access_ratio: Option<f64>,

    /// Whether to cache full pages on read.
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) read_record_cache: Option<bool>,

    /// If true, only use the in-memory circular buffer (no disk pages).
    #[serde(deserialize_with = "Deserialize::deserialize")]
    pub(crate) cache_only: Option<bool>,
}

impl BfTreeStoreConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        // bf-tree requires:
        //   cache-only mode:     cb_size_byte >= 4 * leaf_page_size
        //   non cache-only mode: cb_size_byte >= 2 * leaf_page_size
        let multiplier = if self.cache_only.unwrap_or(false) {
            4
        } else {
            2
        };
        let min_cb = multiplier * self.leaf_page_size;
        anyhow::ensure!(
            self.cb_size_byte >= min_cb,
            "cb_size_byte ({}) must be at least {} * leaf_page_size ({}) = {} bytes",
            self.cb_size_byte,
            multiplier,
            self.leaf_page_size,
            min_cb,
        );
        Ok(())
    }

    pub(crate) fn into_config(self) -> bf_tree::Config {
        let mut c = bf_tree::Config::default();
        c.cb_size_byte(self.cb_size_byte);
        c.leaf_page_size(self.leaf_page_size);
        if let Some(v) = self.cb_max_record_size {
            c.cb_max_record_size(v);
        }
        if let Some(v) = self.cb_min_record_size {
            c.cb_min_record_size(v);
        }
        if let Some(v) = self.read_promotion_rate {
            c.read_promotion_rate(v);
        }
        if let Some(v) = self.scan_promotion_rate {
            c.scan_promotion_rate(v);
        }
        if let Some(v) = self.cb_copy_on_access_ratio {
            c.cb_copy_on_access_ratio(v);
        }
        if let Some(v) = self.read_record_cache {
            c.read_record_cache(v);
        }
        if let Some(v) = self.cache_only {
            c.cache_only(v);
        }
        c
    }

    /// Backfills fields with "None" from the config json with the actual defaults so the display
    /// function will print helpful information in the json.
    /// Some of the fields do not have accessors, so their values will be manually populated with
    /// the publicly listed defaults for bf-tree v0.4.9.
    pub(crate) fn fill_defaults(&mut self) {
        let defaults = bf_tree::Config::default();
        // fill optional field with available config field accessors
        self.cb_max_record_size
            .get_or_insert(defaults.get_cb_max_record_size());

        // these are the fields without accessors from the defaults, fill with known defaults from
        // bf-tree (v0.4.9)
        self.cb_min_record_size.get_or_insert(4);
        self.read_promotion_rate
            .get_or_insert(if cfg!(debug_assertions) { 50 } else { 30 });
        self.scan_promotion_rate
            .get_or_insert(if cfg!(debug_assertions) { 50 } else { 30 });
        self.cb_copy_on_access_ratio.get_or_insert(0.1);
        self.read_record_cache.get_or_insert(true);
        self.cache_only.get_or_insert(false);
    }
}

impl Default for BfTreeStoreConfig {
    fn default() -> Self {
        Self {
            cb_size_byte: 32 * 1024 * 1024, // 32MB
            leaf_page_size: 4096,
            cb_max_record_size: None,
            cb_min_record_size: None,
            read_promotion_rate: None,
            scan_promotion_rate: None,
            cb_copy_on_access_ratio: None,
            read_record_cache: None,
            cache_only: None,
        }
    }
}

impl Example for BfTreeStoreConfig {
    fn example() -> Self {
        Self::default()
    }
}

impl std::fmt::Display for BfTreeStoreConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "cb_size_byte", self.cb_size_byte)?;
        write_field!(f, "leaf_page_size", self.leaf_page_size)?;
        if let Some(v) = self.cb_max_record_size {
            write_field!(f, "cb_max_record_size", v)?;
        }
        if let Some(v) = self.cb_min_record_size {
            write_field!(f, "cb_min_record_size", v)?;
        }
        if let Some(v) = self.read_promotion_rate {
            write_field!(f, "read_promotion_rate", v)?;
        }
        if let Some(v) = self.scan_promotion_rate {
            write_field!(f, "scan_promotion_rate", v)?;
        }
        if let Some(v) = self.cb_copy_on_access_ratio {
            write_field!(f, "cb_copy_on_access_ratio", v)?;
        }
        if let Some(v) = self.read_record_cache {
            write_field!(f, "read_record_cache", v)?;
        }
        if let Some(v) = self.cache_only {
            write_field!(f, "cache_only", v)?;
        }
        Ok(())
    }
}

/// Shared helper to construct [`BfTreeProviderParameters`] from common fields.
fn bftree_parameters_from(
    build: &IndexBuild,
    num_points: usize,
    dim: usize,
    vector_store_config: &Option<BfTreeStoreConfig>,
    neighbor_store_config: &Option<BfTreeStoreConfig>,
    quant_store_config: &Option<BfTreeStoreConfig>,
) -> BfTreeProviderParameters {
    BfTreeProviderParameters {
        max_points: num_points,
        max_degree: build.exact_max_degree() as u32,
        num_start_points: NonZero::new(build.start_point_strategy().count()).unwrap(),
        dim,
        metric: build.distance().into(),
        vector_provider_config: vector_store_config
            .clone()
            .unwrap_or_default()
            .into_config(),
        neighbor_list_provider_config: neighbor_store_config
            .clone()
            .unwrap_or_default()
            .into_config(),
        quant_vector_provider_config: quant_store_config.clone().unwrap_or_default().into_config(),
        graph_params: None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub(crate) enum QuantConfig {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "spherical")]
    Spherical {
        seed: u64,
        transform_kind: exhaustive::TransformKind,
        num_bits: NonZeroUsize,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        pre_scale: Option<exhaustive::PreScale>,
        #[serde(deserialize_with = "Deserialize::deserialize")]
        quant_store_config: Option<BfTreeStoreConfig>,
    },
}

impl QuantConfig {
    pub(crate) fn validate(&mut self) -> anyhow::Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Spherical {
                num_bits,
                quant_store_config,
                ..
            } => {
                match num_bits.get() {
                    1 | 2 | 4 => {}
                    n => anyhow::bail!("{n} bits are not supported for spherical quantization"),
                }
                if let Some(cfg) = quant_store_config {
                    cfg.fill_defaults();
                    cfg.validate()?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BfTreeBuild {
    build: IndexBuild,
    search_phase: SearchPhase,
    quantization: QuantConfig,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    vector_store_config: Option<BfTreeStoreConfig>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    neighbor_store_config: Option<BfTreeStoreConfig>,
}
impl BfTreeBuild {
    pub(crate) const fn tag() -> &'static str {
        "graph-index-bftree"
    }

    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        // Delegate to IndexBuild's try_as_config which uses the default
        // MaxDegree::Value(exact_max_degree) with 1.3x slack. The bf_tree
        // neighbor pages are sized to exact_max_degree to accommodate this.
        self.build.try_as_config()
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.build.data_type()
    }

    pub(crate) fn search_phase(&self) -> &SearchPhase {
        &self.search_phase
    }

    pub(crate) fn build(&self) -> &IndexBuild {
        &self.build
    }

    pub(crate) fn bftree_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> BfTreeProviderParameters {
        let quant_store_config = match &self.quantization {
            QuantConfig::None => &None,
            QuantConfig::Spherical {
                quant_store_config, ..
            } => quant_store_config,
        };
        bftree_parameters_from(
            &self.build,
            num_points,
            dim,
            &self.vector_store_config,
            &self.neighbor_store_config,
            quant_store_config,
        )
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> Result<(), anyhow::Error> {
        self.build.validate(checker)?;
        self.search_phase.validate(checker)?;
        if let Some(cfg) = &mut self.neighbor_store_config {
            cfg.fill_defaults();
            cfg.validate()?;
        }
        if let Some(cfg) = &mut self.vector_store_config {
            cfg.fill_defaults();
            cfg.validate()?;
        }
        self.quantization.validate()?;
        Ok(())
    }

    pub(crate) fn quantization(&self) -> &QuantConfig {
        &self.quantization
    }
}

impl Example for BfTreeBuild {
    fn example() -> Self {
        Self {
            build: IndexBuild::example(),
            search_phase: SearchPhase::Topk(TopkSearchPhase::example()),
            quantization: QuantConfig::None,
            vector_store_config: None,
            neighbor_store_config: None,
        }
    }
}

impl std::fmt::Display for BfTreeBuild {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph Index Bf_Tree Build")?;
        if cfg!(not(feature = "bftree")) {
            writeln!(f, "Requires the `bftree` feature")?;
        }
        write_field!(f, "tag", Self::tag())?;

        if let QuantConfig::Spherical {
            seed,
            transform_kind,
            num_bits,
            ..
        } = &self.quantization
        {
            write_field!(f, "quantization", "spherical")?;
            write_field!(f, "num_bits", num_bits)?;
            write_field!(f, "seed", seed)?;
            write_field!(f, "transform_kind", transform_kind)?;
        } else {
            write_field!(f, "quantization", "none")?;
        }

        writeln!(f)?;
        self.build.summarize_fields(f)?;

        if let Some(ref cfg) = self.vector_store_config {
            writeln!(f, "\n  Vector Store:")?;
            write!(f, "{}", cfg)?;
        }
        if let Some(ref cfg) = self.neighbor_store_config {
            writeln!(f, "\n  Neighbor Store:")?;
            write!(f, "{}", cfg)?;
        }
        if let QuantConfig::Spherical {
            quant_store_config: Some(ref cfg),
            ..
        } = self.quantization
        {
            writeln!(f, "\n  Quant Store:")?;
            write!(f, "{}", cfg)?;
        }

        Ok(())
    }
}

as_input!(BfTreeBuild);
as_input!(BfTreeStreamingRun);

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BfTreeStreamingRun {
    build: IndexBuild,
    search_phase: SearchPhase,
    runbook_params: StreamingRunbookParams,
    quantization: QuantConfig,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    vector_store_config: Option<BfTreeStoreConfig>,
    #[serde(deserialize_with = "Deserialize::deserialize")]
    neighbor_store_config: Option<BfTreeStoreConfig>,
}

impl BfTreeStreamingRun {
    pub(crate) const fn tag() -> &'static str {
        "graph-index-stream-bftree"
    }

    pub(crate) fn try_as_config(&self) -> anyhow::Result<config::Builder> {
        self.build.try_as_config()
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.build.data_type()
    }

    pub(crate) fn search_phase(&self) -> &SearchPhase {
        &self.search_phase
    }

    pub(crate) fn build(&self) -> &IndexBuild {
        &self.build
    }

    pub(crate) fn runbook_params(&self) -> &StreamingRunbookParams {
        &self.runbook_params
    }

    pub(crate) fn quantization(&self) -> &QuantConfig {
        &self.quantization
    }

    pub(crate) fn bftree_parameters(
        &self,
        num_points: usize,
        dim: usize,
    ) -> BfTreeProviderParameters {
        let quant_store_config = match &self.quantization {
            QuantConfig::None => &None,
            QuantConfig::Spherical {
                quant_store_config, ..
            } => quant_store_config,
        };
        bftree_parameters_from(
            &self.build,
            num_points,
            dim,
            &self.vector_store_config,
            &self.neighbor_store_config,
            quant_store_config,
        )
    }

    pub(crate) fn validate(&mut self, checker: &mut Checker) -> anyhow::Result<()> {
        self.build.validate(checker)?;
        self.search_phase.validate(checker)?;
        self.runbook_params.validate(checker)?;
        if let Some(cfg) = &mut self.vector_store_config {
            cfg.fill_defaults();
            cfg.validate()?;
        }
        if let Some(cfg) = &mut self.neighbor_store_config {
            cfg.fill_defaults();
            cfg.validate()?;
        }
        self.quantization.validate()?;
        Ok(())
    }
}

impl Example for BfTreeStreamingRun {
    fn example() -> Self {
        Self {
            build: IndexBuild::example(),
            search_phase: SearchPhase::Topk(TopkSearchPhase::example()),
            runbook_params: StreamingRunbookParams::example_immediate(),
            quantization: QuantConfig::None,
            vector_store_config: None,
            neighbor_store_config: None,
        }
    }
}

impl std::fmt::Display for BfTreeStreamingRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Graph Index Bf_Tree Streaming")?;
        if cfg!(not(feature = "bftree")) {
            writeln!(f, "Requires the `bftree` feature")?;
        }
        write_field!(f, "tag", Self::tag())?;

        if let QuantConfig::Spherical {
            seed,
            transform_kind,
            num_bits,
            ..
        } = &self.quantization
        {
            write_field!(f, "quantization", "spherical")?;
            write_field!(f, "num_bits", num_bits)?;
            write_field!(f, "seed", seed)?;
            write_field!(f, "transform_kind", transform_kind)?;
        } else {
            write_field!(f, "quantization", "none")?;
        }

        writeln!(f)?;
        self.build.summarize_fields(f)?;

        if let Some(ref cfg) = self.vector_store_config {
            writeln!(f, "\n  Vector Store:")?;
            write!(f, "{}", cfg)?;
        }
        if let Some(ref cfg) = self.neighbor_store_config {
            writeln!(f, "\n  Neighbor Store:")?;
            write!(f, "{}", cfg)?;
        }
        if let QuantConfig::Spherical {
            quant_store_config: Some(ref cfg),
            ..
        } = self.quantization
        {
            writeln!(f, "\n  Quant Store:")?;
            write!(f, "{}", cfg)?;
        }

        Ok(())
    }
}

