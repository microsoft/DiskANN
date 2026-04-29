/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{any::Any, sync::Arc};

use diskann::{graph::DiskANNIndex, provider::DataProvider};

use crate::{
    backend::index::result::AggregatedSearchResults,
    inputs::async_::{SearchPhase, SearchPhaseKind},
};

pub(crate) trait Plugin<DP, P>: std::fmt::Debug
where
    DP: DataProvider,
{
    /// The flavor of `SearchPhase` this plugin is compiled for.
    fn kind(&self) -> SearchPhaseKind;

    fn search(
        &self,
        index: Arc<DiskANNIndex<DP>>,
        parameters: &P,
        phase: &SearchPhase,
    ) -> anyhow::Result<AggregatedSearchResults>;
}

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
    pub(crate) fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    pub(crate) fn register<T>(&mut self, plugin: T)
    where
        T: Plugin<DP, P> + 'static,
    {
        self.plugins.push(Box::new(plugin));
    }

    pub(crate) fn kinds(&self) -> Vec<SearchPhaseKind> {
        self.plugins.iter().map(|p| p.kind()).collect()
    }

    pub(crate) fn is_match(&self, phase: &SearchPhase) -> bool {
        self.plugins.iter().any(|p| p.kind() == phase.kind())
    }

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

#[derive(Debug, Clone, Copy)]
pub(crate) struct Topk;

impl Topk {
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::Topk
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Range;

impl Range {
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::Range
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BetaFilter;

impl BetaFilter {
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::TopkBetaFilter
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MultihopFilter;

impl MultihopFilter {
    pub(crate) fn kind() -> SearchPhaseKind {
        SearchPhaseKind::TopkMultihopFilter
    }
}
