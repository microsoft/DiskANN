/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Immutable search snapshot; prepare centroids once without changing the build's means.

use std::{
    collections::BTreeMap,
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use diskann::{
    ivf::traits::{Accumulator, HasIds, SearchAccessor, SearchStrategy},
    provider::DefaultContext,
    ANNError, ANNResult,
};
use diskann_utils::future::SendFuture;

use super::{
    data::Rows,
    metric,
    provider::{self, List, Store},
};
use crate::utils::SimilarityMeasure;

pub(super) struct SearchProvider {
    points: Arc<Rows>,
    lists: BTreeMap<u32, List>,
    inserted: usize,
    distance: SimilarityMeasure,
}

impl SearchProvider {
    pub(super) fn new(store: &Store, distance: SimilarityMeasure) -> Self {
        let lists = store
            .lists
            .iter()
            .map(|(&id, list)| {
                (
                    id,
                    List {
                        centroid: metric::prepare_centroid(distance, &list.centroid),
                        members: list.members.clone(),
                    },
                )
            })
            .collect();
        Self {
            points: store.points.clone(),
            lists,
            inserted: store.inserted,
            distance,
        }
    }

    fn select(&self, query: &[f32], nprobe: usize) -> Vec<(u32, f32)> {
        provider::select(
            self.lists.iter().map(|(&id, list)| (id, list)),
            query,
            nprobe,
            |query, centroid| metric::evaluate(self.distance, query, centroid),
        )
    }

    /// Choose the closest achievable scan budgets, sharing centroid rankings across targets.
    pub(super) fn calibrate(&self, queries: &[Vec<f32>], ratios: &[f64]) -> ANNResult<Vec<usize>> {
        if ratios.is_empty() {
            return Ok(Vec::new());
        }
        if queries.is_empty() || self.inserted == 0 {
            return Err(ANNError::message(
                "scan calibration needs an index and queries",
            ));
        }
        let mut counts = vec![0usize; self.lists.len()];
        for query in queries {
            let selected = self.select(query, self.lists.len());
            for (count, (id, _)) in counts.iter_mut().zip(selected) {
                *count += self.lists[&id].members.len();
            }
        }
        let targets: Vec<_> = ratios
            .iter()
            .map(|ratio| ratio * self.inserted as f64 * queries.len() as f64)
            .collect();
        let mut cumulative = 0usize;
        let mut choices = vec![(1, f64::INFINITY); targets.len()];
        for (rank, count) in counts.into_iter().enumerate() {
            cumulative += count;
            for (target, best) in targets.iter().zip(&mut choices) {
                let error = (cumulative as f64 - target).abs();
                if error < best.1 {
                    *best = (rank + 1, error);
                }
            }
        }
        Ok(choices.into_iter().map(|(nprobe, _)| nprobe).collect())
    }
}

impl HasIds for SearchProvider {
    type Id = u32;
    type ListId = u32;
}

#[derive(Default)]
pub(super) struct ScanStats {
    pub(super) candidates: AtomicUsize,
    pub(super) lists: AtomicUsize,
}

#[derive(Clone, Copy)]
pub(super) struct Query<'a> {
    pub(super) vector: &'a [f32],
    pub(super) stats: &'a ScanStats,
}

pub(super) struct Search<'a> {
    provider: &'a SearchProvider,
    query: Query<'a>,
}

impl HasIds for Search<'_> {
    type Id = u32;
    type ListId = u32;
}

impl<'a> SearchStrategy<'a, Query<'a>, DefaultContext> for SearchProvider {
    type Accessor = Search<'a>;
    type Error = ANNError;

    fn search_accessor(&'a self, _: &'a DefaultContext, query: Query<'a>) -> ANNResult<Search<'a>> {
        if query.vector.len() != self.points[0].len() || !query.vector.iter().all(|x| x.is_finite())
        {
            return Err(ANNError::message("invalid IVF query"));
        }
        Ok(Search {
            provider: self,
            query,
        })
    }
}

impl SearchAccessor for Search<'_> {
    fn scan<A>(&self, nprobe: NonZeroUsize, results: &mut A) -> impl SendFuture<ANNResult<()>>
    where
        A: Accumulator<u32>,
    {
        async move {
            let selected = self.provider.select(self.query.vector, nprobe.get());
            let mut candidates = 0;
            for &(id, _) in &selected {
                for &member in &self.provider.lists[&id].members {
                    results.push(
                        member,
                        metric::evaluate(
                            self.provider.distance,
                            self.query.vector,
                            self.provider.points[member as usize].as_slice(),
                        ),
                    );
                    candidates += 1;
                }
            }
            self.query
                .stats
                .candidates
                .store(candidates, Ordering::Relaxed);
            self.query
                .stats
                .lists
                .store(selected.len(), Ordering::Relaxed);
            Ok(())
        }
    }
}
