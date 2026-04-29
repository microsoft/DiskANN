/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Search plugins are the solution the following benchmarking problem:
//!
//! The [`SearchPhase`] enum contains a list of available search kinds. Adding a new variant
//! either requires updating **all** users to implement that related search (harming compile
//! times) or requires users to explicitly opt-out. Unfortunately, the latter is difficult
//! to maintain with benchmark matching (i.e., the desire to catch configuration mismatches
//! such as requesting an unsupported search early, rather than reaching an error late in
//! a benchmark run). Additionally, if only a subset of search kinds are supported, it
//! is user-friendly to document which variants are actually supported and to make it simple
//! to add or remove flavors.
//!
//! The solution is the [`Plugin`] trait and the [`Plugins`] helper. The trait is a
//! dyn-compatible wrapper for a search and the [`Plugins`] struct simply collects a list
//! of [`Plugin`]s.
//!
//! Implementations of [`Plugin`] declare which type of search they support, which is aggregated
//! in the [`Plugins`] helper.
//!
//! Benchmarks can then contain a [`Plugins`] field, dynamically register plugin types, and
//! then get registered in [`diskann_benchmark_runner::Benchmarks`]. The follow methods then
//! support proper reporting in the benchmark infrastructure:
//!
//! * [`Plugins::format_kinds`]: Format the registered plugins.
//! * [`Plugins::is_match`]: Return whether a [`Plugin`] is registered matching a phase.
//! * [`Plugins::run`]: Run the first matching plugin.
//!
//! Concrete plugins maintain a one-to-one relationship with variants in [`SearchPhase`] and
//! [`SearchPhaseKind`] and are simple ZSTs.

use std::sync::Arc;

use diskann::{graph::DiskANNIndex, provider::DataProvider};
use diskann_benchmark_runner::utils::fmt::{Delimit, Quote};

use crate::{
    backend::index::result::AggregatedSearchResults,
    inputs::async_::{SearchPhase, SearchPhaseKind},
};

/// A search plugin for `DP`. The generic `P` is for any additional parameters needed by
/// a benchmark.
pub(crate) trait Plugin<DP, P>: std::fmt::Debug
where
    DP: DataProvider,
{
    /// The flavor of `SearchPhase` this plugin is compiled for.
    fn kind(&self) -> SearchPhaseKind;

    /// Run the search.
    ///
    /// The user can assume that `phase` has the same [`SearchPhaseKind`] as [`Self::kind`]
    /// and may return an error if this is not the case.
    fn search(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        parameters: &P,
        phase: &SearchPhase,
    ) -> anyhow::Result<AggregatedSearchResults>;
}

/// A collection of dynamically registered [`Plugins`].
#[derive(Debug)]
pub(crate) struct Plugins<DP, P>
where
    DP: DataProvider,
{
    plugins: Vec<Box<dyn Plugin<DP, P>>>,
}

impl<DP, P> Plugins<DP, P>
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
        T: Plugin<DP, P> + 'static,
    {
        self.plugins.push(Box::new(plugin));
    }

    /// Return an iterator over all [`SearchPhaseKind`]s currently registered.
    pub(crate) fn kinds(&self) -> impl ExactSizeIterator<Item = SearchPhaseKind> + use<'_, DP, P> {
        self.plugins.iter().map(|p| p.kind())
    }

    /// Return whether a [`Plugin`] is registered matching `phase`.
    pub(crate) fn is_match(&self, phase: SearchPhaseKind) -> bool {
        self.plugins.iter().any(|p| p.kind() == phase)
    }

    /// Return a human readable, formatted list of the registered [`SearchPhaseKind`]s.
    pub(crate) fn format_kinds(&self) -> impl std::fmt::Display + use<'_, DP, P> {
        Delimit::new(self.kinds().map(Quote), ", ", Some(", and "))
    }

    /// Try to run a search plugin for `phase`.
    ///
    /// If no such plugin exists, an "INTERNAL ERROR:" is returned.
    /// Within the `diskann-benchmark` crate, pre-validation with [`Self::is_match`] should
    /// be used before calling this method.
    pub(crate) fn run(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        parameters: &P,
        phase: &SearchPhase,
    ) -> anyhow::Result<AggregatedSearchResults> {
        match self.plugins.iter().find(|p| p.kind() == phase.kind()) {
            Some(plugin) => plugin.search(index, parameters, phase),
            None => Err(anyhow::anyhow!(
                "INTERNAL ERROR: Could not find a search plugin for {}",
                phase.kind()
            )),
        }
    }
}

/// A search plugin for vanilla top-k search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Topk;

impl Topk {
    /// Returns [`SearchPhaseKind::Topk`].
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::Topk
    }
}

/// A search plugin for range search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Range;

impl Range {
    /// Returns [`SearchPhaseKind::Range`].
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::Range
    }
}

/// A search plugin for beta-filtered search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BetaFilter;

impl BetaFilter {
    /// Returns [`SearchPhaseKind::TopkBetaFilter`].
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::TopkBetaFilter
    }
}

/// A search plugin for multi-hop filtered search.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MultihopFilter;

impl MultihopFilter {
    /// Returns [`SearchPhaseKind::TopkMultihopFilter`].
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::TopkMultihopFilter
    }
}
