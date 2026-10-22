/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use diskann::{
    ivf::{Config, IVFIndex, InsertEvent, InsertPhase},
    provider::DefaultContext,
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::{rngs::StdRng, seq::index, SeedableRng};

use super::{
    provider::{Store, Strategy},
    result::BuildResult,
    search::CentroidGraph,
};
use crate::inputs::graph_ivf::GraphIvfBuild;

pub(super) async fn build(
    input: &GraphIvfBuild,
    data: rowmajor::Ref<'_, f32>,
) -> anyhow::Result<(IVFIndex<Store>, CentroidGraph, BuildResult)> {
    let start = Instant::now();
    let settings = &input.build;
    let mut rng = StdRng::seed_from_u64(settings.seed);
    let sampled = index::sample(&mut rng, data.nrows(), settings.initial_centroids.get());
    let mut centroids = rowmajor::Owned::try_from_element(sampled.len(), data.ncols(), 0.0)?;
    for (row, index) in centroids.rows_mut().zip(sampled) {
        row.copy_from_slice(data.row(index));
    }
    let mut index = IVFIndex::new(
        Store::new(data.ncols()),
        Config {
            split_threshold: settings.split_threshold,
            two_means_iterations: settings.two_means_iterations.get(),
            seed: settings.seed,
        },
    )?;
    index
        .initialize(&Strategy, &DefaultContext, centroids.as_view())
        .await?;
    let mut inserted = 0;
    let mut splits = 0;
    let mut split_points = 0;
    let mut two_means = Duration::ZERO;
    let mut reassignment = Duration::ZERO;
    let mut split_parent_histogram = BTreeMap::new();
    for first in (0..data.nrows()).step_by(settings.batch_size.get()) {
        let end = first
            .saturating_add(settings.batch_size.get())
            .min(data.nrows());
        let points: Vec<_> = (first..end).map(|i| (i as u32, data.row(i))).collect();
        let mut timer = SplitTimer::default();
        let stats = index
            .insert_batch_observed(&Strategy, &DefaultContext, &points, |event| {
                timer.observe(event);
            })
            .await?;
        inserted += stats.inserted;
        splits += stats.splits;
        split_points += stats.split_points;
        two_means += timer.two_means;
        reassignment += timer.reassignment;
        *split_parent_histogram.entry(stats.splits).or_insert(0) += 1;
    }
    let ivf = start.elapsed();
    let graph_start = Instant::now();
    let graph = CentroidGraph::build(index.provider(), &input.graph).await?;
    let centroid_graph_seconds = graph_start.elapsed().as_secs_f64();
    let total_seconds = start.elapsed().as_secs_f64();
    let store = index.provider();
    let result = BuildResult {
        repetition: 0,
        ivf_seconds: ivf.as_secs_f64(),
        two_means_seconds: two_means.as_secs_f64(),
        reassignment_seconds: reassignment.as_secs_f64(),
        // Check the partition instead of clamping a broken measurement to zero.
        other_ivf_seconds: ivf
            .checked_sub(two_means + reassignment)
            .ok_or_else(|| anyhow::anyhow!("split timings exceed the IVF build interval"))?
            .as_secs_f64(),
        centroid_graph_seconds,
        total_seconds,
        vectors_per_second: data.nrows() as f64 / total_seconds,
        inserted,
        splits,
        split_points,
        split_parent_histogram,
        lists: store.lists.len(),
        empty_lists: store
            .lists
            .values()
            .filter(|l| l.members.is_empty())
            .count(),
        max_list_size: store
            .lists
            .values()
            .map(|l| l.members.len())
            .max()
            .unwrap_or(0),
    };
    Ok((index, graph, result))
}

#[derive(Default)]
struct SplitTimer {
    started: Option<(InsertPhase, Instant)>,
    two_means: Duration,
    reassignment: Duration,
}

impl SplitTimer {
    fn observe(&mut self, event: InsertEvent) {
        match event {
            InsertEvent::PhaseStarted(phase) => {
                debug_assert!(self.started.is_none());
                self.started = Some((phase, Instant::now()));
            }
            InsertEvent::PhaseFinished(phase) => {
                let Some((started_phase, started)) = self.started.take() else {
                    debug_assert!(false, "phase finish must follow a phase start");
                    return;
                };
                debug_assert_eq!(started_phase, phase);
                if started_phase != phase {
                    return;
                }
                match phase {
                    InsertPhase::TwoMeans => self.two_means += started.elapsed(),
                    InsertPhase::Reassignment => self.reassignment += started.elapsed(),
                    _ => {}
                }
            }
            _ => {}
        }
    }
}
