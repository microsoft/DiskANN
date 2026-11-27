/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Shared IVF fixtures and snapshot-based search oracles.
//!
//! Fixtures use the build API and reset counters before use. Oracles read published state
//! without calling the index or using its candidate queue.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Debug, Display},
    num::NonZeroUsize,
    sync::Arc,
};

use diskann_quantization::Parallelism;
use diskann_utils::assert_contains;
use diskann_vector::{PureDistanceFunction, distance::SquaredL2};

use crate::{
    ANNResult,
    graph::SearchOutputBuffer,
    ivf::{
        Config, IVFIndex, InsertStats, ReassignCount,
        region::Region as _,
        test::provider::{Build, Metrics, Provider, Region, Search, Snapshot},
        traits::BuildAccessor as _,
    },
    neighbor::{Neighbor, ord::fast_distance_total},
    test::cmp::assert_eq_verbose,
    utils::ONE,
};

pub(super) const PARALLELISMS: [Parallelism; 2] = [Parallelism::Sequential, Parallelism::Rayon];

/// One-dimensional configuration with ten Lloyd iterations; zero `neighbors` skips reassignment.
pub(super) fn config(split_threshold: usize, split_k: usize, neighbors: usize) -> Config {
    Config {
        dim: ONE,
        split_threshold: NonZeroUsize::new(split_threshold).unwrap(),
        split_k: NonZeroUsize::new(split_k).unwrap(),
        reassign_count: NonZeroUsize::new(neighbors)
            .map_or(ReassignCount::None, ReassignCount::Count),
        split_iterations: NonZeroUsize::new(10).unwrap(),
    }
}

/// Seed one-dimensional `(centroid, points)` lists with exact memberships and zero counters.
///
/// List IDs follow slice order; point IDs follow flattened member order, both from zero.
pub(super) async fn line(lists: &[(f32, &[f32])]) -> Provider {
    let seeding = Provider::new(Snapshot::new(ONE));

    let mut build = Build::new(&seeding).unwrap();
    let mut region = build.fill(std::iter::empty()).await.unwrap();
    let ids: Vec<u32> = lists
        .iter()
        .map(|&(centroid, _)| region.create(&[centroid], &[]).unwrap())
        .collect();
    region.finish().await.unwrap();
    build.finish().await.unwrap();

    let xs = lists.iter().flat_map(|&(_, xs)| xs.iter().copied());
    let points: Vec<(u32, f32)> = (0..).zip(xs).collect();
    if !points.is_empty() {
        let mut build = Build::new(&seeding).unwrap();
        build.route(&batch(&points)[..], |_, _| {}).await.unwrap();

        // Memberships are prescribed by the fixture, not by nearest-centroid routing.
        let mut region = build.fill(ids.iter().copied()).await.unwrap();
        let mut point_ids = points.iter().map(|&(id, _)| id);
        for (&id, &(_, xs)) in ids.iter().zip(lists) {
            let routed = region.get_list(id).unwrap().members.to_vec();
            let own: Vec<u32> = point_ids.by_ref().take(xs.len()).collect();
            region.remove_members(id, &routed).unwrap();
            region.append_members(id, &own).unwrap();
        }
        region.finish().await.unwrap();
        build.finish().await.unwrap();
    }

    Provider::new(Snapshot::clone(&seeding.snapshot()))
}

/// Commit a region operation as one build; any error abandons it.
pub(super) async fn stage<T>(
    provider: &Provider,
    lists: &[u32],
    operation: impl FnOnce(&mut Region<'_>) -> ANNResult<T>,
) -> ANNResult<T> {
    let mut build = Build::new(provider)?;
    let mut region = build.fill(lists.iter().copied()).await?;
    let result = operation(&mut region)?;
    region.finish().await?;
    build.finish().await?;
    Ok(result)
}

/// Return one-dimensional `points`, given as `(id, x)`, as a batch to route.
pub(super) fn batch(points: &[(u32, f32)]) -> Vec<(u32, &[f32])> {
    points
        .iter()
        .map(|(id, x)| (*id, std::slice::from_ref(x)))
        .collect()
}

/// Insert `points` through a fresh build.
pub(super) async fn insert(
    index: &mut IVFIndex,
    provider: &Provider,
    points: &[(u32, &[f32])],
    parallelism: Parallelism,
) -> ANNResult<InsertStats> {
    index
        .insert_batch(Build::new(provider)?, points, parallelism)
        .await
}

/// Check that search does not publish, reports appended counts on success, and leaves
/// output length unchanged on failure.
pub(super) async fn search_into(
    index: &IVFIndex,
    provider: &Provider,
    query: &[f32],
    k: usize,
    nprobe: usize,
    output: &mut (impl SearchOutputBuffer<u32> + Send),
) -> ANNResult<usize> {
    let before = output.current_len();
    let published = provider.snapshot();
    let result = index
        .search(
            Search::new(provider, query),
            NonZeroUsize::new(k).unwrap(),
            NonZeroUsize::new(nprobe).unwrap(),
            output,
        )
        .await;
    assert!(
        Arc::ptr_eq(&provider.snapshot(), &published),
        "search must not publish a new snapshot"
    );
    match result {
        Ok(count) => {
            assert_eq!(output.current_len(), before + count);
            Ok(count)
        }
        Err(err) => {
            assert_eq!(
                output.current_len(),
                before,
                "a failed search must not append"
            );
            Err(err)
        }
    }
}

pub(super) async fn search(
    index: &IVFIndex,
    provider: &Provider,
    query: &[f32],
    k: usize,
    nprobe: usize,
) -> ANNResult<Vec<Neighbor<u32>>> {
    let mut output = Vec::new();
    search_into(index, provider, query, k, nprobe, &mut output).await?;
    Ok(output)
}

/// Exhaustively rank accepted centroids by distance and then list ID.
pub(super) fn nearest_lists(
    snapshot: &Snapshot,
    query: &[f32],
    nprobe: usize,
    accept: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let mut lists: Vec<Neighbor<u32>> = snapshot
        .lists()
        .iter()
        .filter(|&(&id, _)| accept(id))
        .map(|(&id, list)| Neighbor::new(id, SquaredL2::evaluate(query, list.centroid())))
        .collect();
    lists.sort_by(fast_distance_total);
    lists.into_iter().take(nprobe).map(|n| *n.id()).collect()
}

fn rank_members(
    snapshot: &Snapshot,
    query: &[f32],
    lists: impl Iterator<Item = u32>,
) -> Vec<Neighbor<u32>> {
    let mut neighbors: Vec<_> = lists
        .flat_map(|id| snapshot.lists()[&id].members().iter().copied())
        .map(|id| Neighbor::new(id, SquaredL2::evaluate(query, snapshot.vector(id).unwrap())))
        .collect();
    neighbors.sort_by(fast_distance_total);
    neighbors
}

/// Rank every published point by distance and then point ID.
pub(super) fn brute_force(snapshot: &Snapshot, query: &[f32]) -> Vec<Neighbor<u32>> {
    rank_members(snapshot, query, snapshot.lists().keys().copied())
}

/// Rank every member of the `nprobe` nearest lists.
pub(super) fn scanned_brute_force(
    snapshot: &Snapshot,
    query: &[f32],
    nprobe: usize,
) -> Vec<Neighbor<u32>> {
    let lists = nearest_lists(snapshot, query, nprobe, |_| true);
    rank_members(snapshot, query, lists.into_iter())
}

/// Check top-k distances and distinct, correctly scored IDs; tied IDs may differ.
///
/// `candidates` must contain all allowed points in distance order, including cutoff ties.
#[track_caller]
pub(super) fn assert_neighbors(actual: &[Neighbor<u32>], candidates: &[Neighbor<u32>], k: usize) {
    let distances =
        |neighbors: &[Neighbor<u32>]| neighbors.iter().map(|n| *n.distance()).collect::<Vec<_>>();
    assert_eq!(
        distances(actual),
        distances(&candidates[..k.min(candidates.len())])
    );

    let scores: BTreeMap<_, _> = candidates
        .iter()
        .map(|n| (*n.id(), *n.distance()))
        .collect();
    let mut seen = BTreeSet::new();
    for neighbor in actual {
        let id = *neighbor.id();
        assert!(seen.insert(id), "point {id} was returned twice");
        assert_eq!(
            scores.get(&id),
            Some(neighbor.distance()),
            "point {id} must be a correctly scored candidate"
        );
    }
}

/// Check that every point retains its vector and belongs to exactly one published list.
#[track_caller]
pub(super) fn assert_points<'a>(
    snapshot: &Snapshot,
    points: impl ExactSizeIterator<Item = (u32, &'a [f32])>,
) {
    snapshot.is_consistent().unwrap();
    assert_eq!(
        snapshot
            .lists()
            .values()
            .map(|list| list.members().len())
            .sum::<usize>(),
        points.len()
    );
    for (id, vector) in points {
        assert_eq!(snapshot.vector(id), Some(vector), "vector of point {id}");
    }
}

/// Check publication was abandoned and the build lease was released.
#[track_caller]
pub(super) fn assert_abandoned(provider: &Provider, before: &Arc<Snapshot>) {
    assert!(
        Arc::ptr_eq(&provider.snapshot(), before),
        "an abandoned build must not publish"
    );
    drop(Build::new(provider).unwrap());
}

/// Return the calls made between two metric snapshots.
pub(super) fn delta(before: Metrics, after: Metrics) -> Metrics {
    Metrics {
        route: after.route - before.route,
        find: after.find - before.find,
        fill: after.fill - before.fill,
        append_members: after.append_members - before.append_members,
        remove_members: after.remove_members - before.remove_members,
        retire: after.retire - before.retire,
        create: after.create - before.create,
        region_finish: after.region_finish - before.region_finish,
        finish: after.finish - before.finish,
        scan: after.scan - before.scan,
        distances: after.distances - before.distances,
    }
}

/// Check scan/distance counts and require every non-search counter to be zero.
#[track_caller]
pub(super) fn assert_scans(metrics: Metrics, lists: usize, points: usize) {
    assert_eq_verbose!(
        metrics,
        Metrics {
            scan: lists,
            distances: points,
            ..Metrics::default()
        }
    );
}

/// Check the calls required by the insertion protocol, including the distinct lists
/// selected by the post-split, non-sibling neighborhood queries.
#[track_caller]
pub(super) fn assert_protocol(
    before: &Snapshot,
    after: &Snapshot,
    config: Config,
    stats: &InsertStats,
    metrics: Metrics,
) {
    let children = stats.splits * config.split_k.get();
    let neighbors = match config.reassign_count {
        ReassignCount::Count(n) if stats.splits != 0 => n.get(),
        ReassignCount::None | ReassignCount::Count(_) => 0,
    };
    let stages = usize::from(stats.splits != 0) + usize::from(neighbors != 0);
    let linked = if neighbors == 0 {
        0
    } else {
        let first = before.next_list();
        let family = |id: u32| {
            id.checked_sub(first)
                .map(|offset| offset as usize / config.split_k.get())
        };
        let mut linked: BTreeSet<u32> = after.lists().range(first..).map(|(&id, _)| id).collect();
        assert_eq!(linked.len(), children);
        for child in linked.clone() {
            linked.extend(nearest_lists(
                after,
                after.lists()[&child].centroid(),
                neighbors,
                |id| family(id) != family(child),
            ));
        }
        linked.len()
    };

    assert_eq_verbose!(
        metrics,
        Metrics {
            route: 1,
            finish: 1,
            fill: stages,
            region_finish: stages,
            retire: stats.splits,
            create: children,
            find: if neighbors == 0 { 0 } else { children },
            remove_members: linked,
            append_members: linked,
            ..Metrics::default()
        }
    );
}

/// Return `(id, centroid, points)` in list-ID order, retaining member order.
/// Requires a consistent one-dimensional snapshot.
pub(super) fn layout(snapshot: &Snapshot) -> Vec<(u32, f32, Vec<f32>)> {
    assert_eq!(
        snapshot.dim(),
        1,
        "only one-dimensional snapshots have a layout"
    );
    snapshot.is_consistent().unwrap();

    snapshot
        .lists()
        .iter()
        .map(|(&id, list)| {
            let points = list.members().iter();
            let points = points.map(|&point| snapshot.vector(point).unwrap()[0]);
            (id, list.centroid()[0], points.collect())
        })
        .collect()
}

/// Assert that `result` is an error whose message contains `expected`.
#[track_caller]
pub(super) fn assert_fails<T: Debug, E: Display>(result: Result<T, E>, expected: &str) {
    match result {
        Ok(value) => panic!("expected an error containing {expected:?}, got {value:?}"),
        Err(err) => assert_contains!(err.to_string(), expected),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn line_puts_each_point_in_its_own_list() {
        // Routing would put point 0 at 6 in list 1, and point 2 at 1 in list 0.
        let provider = line(&[(0.0, &[6.0, -1.0]), (10.0, &[]), (20.0, &[1.0])]).await;
        let snapshot = provider.snapshot();

        assert_eq!(
            layout(&snapshot),
            [
                (0, 0.0, vec![6.0, -1.0]),
                (1, 10.0, vec![]),
                (2, 20.0, vec![1.0])
            ]
        );
        for (id, x) in [(0, 6.0), (1, -1.0), (2, 1.0)] {
            assert_eq!(
                snapshot.vector(id),
                Some(&[x][..]),
                "points are numbered in order"
            );
        }
        assert_eq!(snapshot.next_list(), 3);
        assert_eq!(
            provider.metrics(),
            Metrics::default(),
            "seeding should not count"
        );

        assert!(layout(&line(&[]).await.snapshot()).is_empty());
    }
}
