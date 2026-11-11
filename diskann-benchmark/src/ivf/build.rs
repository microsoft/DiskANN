/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use diskann::{
    ivf::{Config, IVFIndex, ReassignCount},
    provider::DefaultContext,
};
use serde::Serialize;

use super::{
    data::Rows,
    provider::{InMemProvider, Store},
};
use crate::inputs::ivf::Build;

pub(super) struct BuiltIndex {
    pub(super) index: IVFIndex,
    pub(super) provider: Arc<InMemProvider>,
}

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) inserted: usize,
    pub(super) lists: usize,
    pub(super) centers: String,
    pub(super) membership: String,
}

pub(super) fn layout(store: &Store) -> Layout {
    let mut centers = DefaultHasher::new();
    let mut membership = DefaultHasher::new();
    for (&id, list) in &store.lists {
        id.hash(&mut centers);
        for &value in &list.centroid {
            value.to_bits().hash(&mut centers);
        }
        id.hash(&mut membership);
        list.members.hash(&mut membership);
    }
    Layout {
        inserted: store.inserted,
        lists: store.lists.len(),
        centers: format!("{:016x}", centers.finish()),
        membership: format!("{:016x}", membership.finish()),
    }
}

#[derive(Debug, Serialize)]
pub(super) struct BuildResult {
    pub(super) base_count: usize,
    pub(super) dim: usize,
    /// Sum of initialization and insert calls; excludes loading, checkpoints and validation.
    pub(super) build_ms: f64,
    pub(super) insert_ms: f64,
    /// Insertion batches that split at least one parent.
    pub(super) split_calls: usize,
    pub(super) split_parents: usize,
    pub(super) lists: usize,
    pub(super) max_list_size: usize,
    pub(super) empty_lists: usize,
    pub(super) layout: Layout,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) trajectory: Vec<Layout>,
}

pub(super) async fn build(
    base: Arc<Rows>,
    config: &Build,
    checkpoint: &mut impl FnMut() -> Result<()>,
) -> Result<(BuiltIndex, BuildResult)> {
    config.validate()?;
    let dim = base.first().context("empty IVF dataset")?.len();
    let count = base.len();
    let start = Instant::now();
    let provider = Arc::new(InMemProvider::new(base)?);
    let mut index = IVFIndex::new(Config {
        dim: std::num::NonZeroUsize::new(dim).context("empty vectors")?,
        split_threshold: config.split_threshold,
        split_k: config.split_k,
        reassign_count: std::num::NonZeroUsize::new(config.reassign_count)
            .map_or(ReassignCount::None, ReassignCount::Count),
        split_iterations: config.iterations,
    })?
    .with_reassignment_mode(config.reassignment_mode.into());
    let initialization_time = start.elapsed();
    let mut insert_time = Duration::ZERO;
    let mut split_calls = 0;
    let mut split_parents = 0;
    let mut trajectory = Vec::new();
    for offset in (0..count).step_by(config.batch_size.get()) {
        let start = Instant::now();
        let stats = index
            .insert_batch(
                provider.as_ref(),
                &DefaultContext,
                offset..(offset + config.batch_size.get()).min(count),
            )
            .await?;
        insert_time += start.elapsed();
        split_calls += usize::from(stats.splits > 0);
        split_parents += stats.splits;
        if config.record_layout {
            trajectory.push(layout(&*provider.read()?));
        }
        checkpoint()?;
    }
    let store = provider.read()?;
    let mut seen = vec![false; count];
    let mut members = 0;
    for &id in store.lists.values().flat_map(|list| &list.members) {
        ensure!(
            (id as usize) < count && !std::mem::replace(&mut seen[id as usize], true),
            "duplicate or invalid IVF member"
        );
        members += 1;
    }
    ensure!(
        members == count && store.inserted == count,
        "missing IVF members"
    );
    let result = BuildResult {
        base_count: count,
        dim,
        build_ms: (initialization_time + insert_time).as_secs_f64() * 1000.0,
        insert_ms: insert_time.as_secs_f64() * 1000.0,
        split_calls,
        split_parents,
        lists: store.lists.len(),
        max_list_size: store
            .lists
            .values()
            .map(|l| l.members.len())
            .max()
            .unwrap_or(0),
        empty_lists: store
            .lists
            .values()
            .filter(|l| l.members.is_empty())
            .count(),
        layout: layout(&store),
        trajectory,
    };
    drop(store);
    Ok((BuiltIndex { index, provider }, result))
}
