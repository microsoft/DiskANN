/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

use anyhow::{ensure, Result};
use diskann::{
    ivf::{Config, IVFIndex, ReassignmentMode, TwoMeansMode},
    provider::DefaultContext,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use serde::Serialize;

use super::provider::{InMemProvider, Strategy};
use crate::inputs::ivf::{Build, SplitMode};

#[derive(Debug, Serialize)]
pub(super) struct SplitEvent {
    batch: usize,
    parents: usize,
    points: usize,
    split_ms: f64,
    fit_ms: f64,
    assignment_ms: f64,
    gemm_calls: usize,
    gemm_parallel_jobs: usize,
    gemm_points: usize,
    projection_points: usize,
    numeric_fallbacks: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct BuildResult {
    pub(super) base_count: usize,
    pub(super) dim: usize,
    pub(super) build_ms: f64,
    pub(super) insert_ms: f64,
    pub(super) split_ms: f64,
    pub(super) fit_ms: f64,
    pub(super) assignment_ms: f64,
    pub(super) split_calls: usize,
    pub(super) split_parents: usize,
    pub(super) split_points: usize,
    pub(super) gemm_calls: usize,
    pub(super) gemm_parallel_jobs: usize,
    pub(super) gemm_points: usize,
    pub(super) projection_points: usize,
    pub(super) numeric_fallbacks: usize,
    pub(super) lists: usize,
    pub(super) max_list_size: usize,
    pub(super) empty_lists: usize,
    pub(super) splits: Vec<SplitEvent>,
}

pub(super) async fn build(
    base: &[Vec<f32>],
    config: &Build,
    checkpoint: &mut impl FnMut() -> Result<()>,
) -> Result<(IVFIndex<InMemProvider>, BuildResult)> {
    let dim = base[0].len();
    let mut index = IVFIndex::new(
        InMemProvider::new(dim),
        Config {
            split_threshold: config.split_threshold,
            two_means_iterations: config.iterations,
            seed: config.seed,
        },
    )?;
    if config.split_mode == SplitMode::ProjectionGemm {
        index.set_two_means_mode(TwoMeansMode::Projection)?;
        index.set_reassignment_mode(ReassignmentMode::Gemm)?;
    }
    // Keep the baseline initialization and input order.
    let initial = rowmajor::Owned::try_from_data(base[0].clone().into_boxed_slice(), 1, dim)?;
    let mut checkpoint_time = Duration::ZERO;
    let start = Instant::now();
    index
        .initialize(&Strategy, &DefaultContext, initial.as_view())
        .await?;
    let mut events = Vec::new();
    for (batch, rows) in base.chunks(config.batch_size.get()).enumerate() {
        let points: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(i, row)| ((batch * config.batch_size.get() + i) as u32, row.as_slice()))
            .collect();
        let before = *index.build_metrics();
        index
            .insert_batch(&Strategy, &DefaultContext, &points)
            .await?;
        let after = index.build_metrics();
        if after.split_calls > before.split_calls {
            events.push((batch, before, *after));
        }
        let check_start = Instant::now();
        checkpoint()?;
        checkpoint_time += check_start.elapsed();
    }
    let elapsed = start.elapsed() - checkpoint_time;
    let ids: Vec<_> = index
        .provider()
        .lists
        .values()
        .flat_map(|list| list.members.iter().copied())
        .collect();
    let unique: BTreeSet<_> = ids.iter().copied().collect();
    ensure!(
        ids.len() == base.len()
            && unique.len() == base.len()
            && unique.iter().all(|&id| (id as usize) < base.len()),
        "invalid index membership"
    );
    let m = index.build_metrics();
    let result = BuildResult {
        base_count: base.len(),
        dim,
        build_ms: elapsed.as_secs_f64() * 1000.0,
        insert_ms: m.insert_time.as_secs_f64() * 1000.0,
        split_ms: m.split_time.as_secs_f64() * 1000.0,
        fit_ms: m.fit_time.as_secs_f64() * 1000.0,
        assignment_ms: m.assignment_time.as_secs_f64() * 1000.0,
        split_calls: m.split_calls,
        split_parents: m.split_parents,
        split_points: m.split_points,
        gemm_calls: m.gemm_calls,
        gemm_parallel_jobs: m.gemm_parallel_jobs,
        gemm_points: m.gemm_points,
        projection_points: m.projection_points,
        numeric_fallbacks: m.numeric_fallbacks,
        lists: index.provider().lists.len(),
        max_list_size: index
            .provider()
            .lists
            .values()
            .map(|l| l.members.len())
            .max()
            .unwrap_or(0),
        empty_lists: index
            .provider()
            .lists
            .values()
            .filter(|l| l.members.is_empty())
            .count(),
        splits: events
            .into_iter()
            .map(|(batch, before, after)| SplitEvent {
                batch,
                parents: after.split_parents - before.split_parents,
                points: after.split_points - before.split_points,
                split_ms: (after.split_time - before.split_time).as_secs_f64() * 1000.0,
                fit_ms: (after.fit_time - before.fit_time).as_secs_f64() * 1000.0,
                assignment_ms: (after.assignment_time - before.assignment_time).as_secs_f64()
                    * 1000.0,
                gemm_calls: after.gemm_calls - before.gemm_calls,
                gemm_parallel_jobs: after.gemm_parallel_jobs - before.gemm_parallel_jobs,
                gemm_points: after.gemm_points - before.gemm_points,
                projection_points: after.projection_points - before.projection_points,
                numeric_fallbacks: after.numeric_fallbacks - before.numeric_fallbacks,
            })
            .collect(),
    };
    Ok((index, result))
}
