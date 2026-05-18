/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Search plugins let each benchmark define exactly which search flavors it supports while
//! keeping benchmark matching and reporting consistent.
//!
//! The core abstraction is split across the [`Plugin`] trait and the [`Plugins`] helper.
//! [`Plugin`] is dyn-compatible and generic over three things:
//!
//! * `DP`: the concrete index/data provider being searched.
//! * `Kind`: the value used for matching and pre-validation.
//! * `Params`: any additional execution context needed once a plugin has been selected.
//!
//! Keeping `Kind` and `Params` separate is intentional. Matching usually wants a narrow,
//! user-facing notion of "what kind of search was requested?", while execution often needs
//! additional benchmark-specific state. This keeps diagnostics precise without forcing the
//! matching type to absorb every runtime detail.
//!
//! Benchmarks own a [`Plugins`] collection and register only the plugin types they want to
//! support. The helper methods on [`Plugins`] then integrate with
//! [`diskann_benchmark_runner::Registry`]:
//!
//! * [`Plugins::format_kinds`]: format the registered plugin labels for diagnostics.
//! * [`Plugins::is_match`]: check whether any registered plugin accepts a requested `Kind`.
//! * [`Plugins::run`]: dispatch to the first registered plugin matching `Kind`.
//!
//! The built-in ZST plugins in this module (`Topk`, `Range`, `BetaFilter`, and
//! `MultihopFilter`) target the async benchmark inputs and fold their outputs into the closed
//! [`AggregatedSearchResults`] families. That closed result boundary is deliberate: plugins are
//! open for new search flavors, while result aggregation remains a curated
//! reporting/evaluation boundary.

use std::sync::Arc;

use diskann::{graph::DiskANNIndex, provider::DataProvider};
use diskann_benchmark_runner::utils::fmt::{Delimit, Quote};
use diskann_providers::post_processor::DeterminantDiversityParams;

use crate::{
    backend::index::result::AggregatedSearchResults,
    inputs::{
        graph_index::{SearchPhase, TopkSearchPhase},
        post_processor::TopkPostProcessor,
    },
};

/// A dyn-compatible search plugin for `DP`.
///
/// `Kind` is the matching surface used for benchmark selection and diagnostics. `Params`
/// contains any additional execution context needed after a plugin has been selected.
pub(crate) trait Plugin<DP, Kind, Params>: std::fmt::Debug
where
    DP: DataProvider,
{
    /// Return `true` if this plugin can accept `kind`.
    fn is_match(&self, kind: &Kind) -> bool;

    /// Return a human-readable label for the flavors of `Kind` supported by this plugin.
    ///
    /// This is used for informational diagnostics and benchmark descriptions.
    fn kind(&self) -> &'static str;

    /// Run the search.
    ///
    /// The user can assume that `kind` passes [`Self::is_match`] and may return an error
    /// if this is not the case.
    fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        kind: &Kind,
        parameters: &Params,
    ) -> anyhow::Result<AggregatedSearchResults>;
}

/// A collection of dynamically registered [`Plugin`]s.
#[derive(Debug)]
pub(crate) struct Plugins<DP, Kind, Params>
where
    DP: DataProvider,
{
    plugins: Vec<Box<dyn Plugin<DP, Kind, Params>>>,
}

impl<DP, Kind, Params> Plugins<DP, Kind, Params>
where
    DP: DataProvider,
{
    /// Create a new empty [`Plugins`].
    pub(crate) fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Register `plugin` in the managed collection.
    pub(crate) fn register<T>(&mut self, plugin: T)
    where
        T: Plugin<DP, Kind, Params> + 'static,
    {
        self.plugins.push(Box::new(plugin));
    }

    /// Return an iterator over the labels of all currently registered plugins.
    pub(crate) fn kinds(
        &self,
    ) -> impl ExactSizeIterator<Item = &'static str> + use<'_, DP, Kind, Params> {
        self.plugins.iter().map(|p| p.kind())
    }

    /// Return whether any registered [`Plugin`] matches `kind`.
    pub(crate) fn is_match(&self, kind: &Kind) -> bool {
        self.plugins.iter().any(|p| p.is_match(kind))
    }

    /// Return a human readable, formatted list of the registered plugin labels.
    pub(crate) fn format_kinds(&self) -> impl std::fmt::Display + use<'_, DP, Kind, Params> {
        Delimit::new(self.kinds().map(Quote), ", ")
            .with_last(", and ")
            .with_pair(" and ")
    }

    /// Try to run a search plugin for `kind`.
    ///
    /// If no such plugin exists, an "INTERNAL ERROR:" is returned.
    /// Within the `diskann-benchmark` crate, pre-validation with [`Self::is_match`] should
    /// be used before calling this method.
    pub(crate) fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        kind: &Kind,
        parameters: &Params,
    ) -> anyhow::Result<AggregatedSearchResults>
    where
        Kind: std::fmt::Debug,
    {
        match self.plugins.iter().find(|p| p.is_match(kind)) {
            Some(plugin) => plugin.run(index, kind, parameters),
            None => Err(anyhow::anyhow!(
                "INTERNAL ERROR: Could not find a suitable search plugin for {:?}",
                kind
            )),
        }
    }
}

/// A search plugin for vanilla top-k search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Topk;

impl Topk {
    pub(crate) fn is_match(phase: &SearchPhase) -> bool {
        phase
            .as_topk()
            .ok()
            .is_some_and(|topk| topk.post_processor.is_none())
    }

    pub(crate) const fn as_str() -> &'static str {
        "topk"
    }
}

/// A search plugin for determinant-diversity top-k post-processing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DeterminantDiversity;

impl DeterminantDiversity {
    pub(crate) fn is_match(phase: &SearchPhase) -> bool {
        phase
            .as_topk()
            .ok()
            .and_then(|topk| topk.post_processor.as_ref())
            .is_some_and(|pp| matches!(pp, TopkPostProcessor::DeterminantDiversity { .. }))
    }

    pub(crate) const fn as_str() -> &'static str {
        "topk + determinant-diversity"
    }

    pub(crate) fn get(
        phase: &SearchPhase,
    ) -> anyhow::Result<(&TopkSearchPhase, DeterminantDiversityParams)> {
        let topk = phase.as_topk()?;
        match topk.post_processor.as_ref() {
            Some(TopkPostProcessor::DeterminantDiversity { power, eta }) => {
                let params = DeterminantDiversityParams::new(*power, *eta)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                Ok((topk, params))
            }
            _ => Err(anyhow::anyhow!(
                "determinant-diversity plugin selected for non determinant-diversity input",
            )),
        }
    }
}

/// A search plugin for range search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Range;

impl Range {
    pub(crate) fn is_match(phase: &SearchPhase) -> bool {
        phase.as_range().is_ok()
    }

    pub(crate) const fn as_str() -> &'static str {
        "range"
    }
}

/// A search plugin for beta-filtered search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TopkBetaFilter;

impl TopkBetaFilter {
    pub(crate) fn is_match(phase: &SearchPhase) -> bool {
        phase.as_topk_beta_filter().is_ok()
    }

    pub(crate) const fn as_str() -> &'static str {
        "topk + beta filter"
    }
}

/// A search plugin for multi-hop filtered search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TopkMultihopFilter;

impl TopkMultihopFilter {
    pub(crate) fn is_match(phase: &SearchPhase) -> bool {
        phase.as_topk_multihop_filter().is_ok()
    }

    pub(crate) const fn as_str() -> &'static str {
        "topk + multihop filter"
    }
}
