/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Static build + search benchmark for the new generic IVF implementation.
//! Search composition and storage are benchmark-local, pending the library APIs.

mod provider;
mod search;

use std::{collections::HashSet, hint::black_box, io::Write, time::Instant};

use diskann::{
    ivf::{Config, IVFIndex},
    provider::DefaultContext,
};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::percentiles::{self, Percentiles},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
use rand::{rngs::StdRng, seq::index, Rng, SeedableRng};
use serde::Serialize;

use crate::{
    inputs::graph_ivf::{Dataset, GraphIvfBuild, Search},
    utils::datafiles::{self, BinFile},
};
use provider::{Store, Strategy};
use search::CentroidGraph;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("graph-ivf-static-f32", Static)?;
    Ok(())
}

struct Static;

impl Benchmark for Static {
    type Input = GraphIvfBuild;
    type Output = Results;

    fn try_match(&self, _: &Self::Input, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Static FP32/L2 IVF build + centroid-graph search; benchmark-local in-memory provider"
        )
    }

    fn run(
        &self,
        input: &Self::Input,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Results> {
        writeln!(output, "Graph-IVF static benchmark: exhaustive build routing; DiskANN centroid graph + posting scan search; single thread")?;
        let (data, queries) = load_data(input)?;
        input.build.validate_rows(data.nrows())?;
        validate_vectors(data.as_view(), "dataset")?;
        let groundtruth = match (&input.search, &queries) {
            (Some(search), Some(queries)) => {
                validate_vectors(queries.as_view(), "queries")?;
                anyhow::ensure!(
                    queries.ncols() == data.ncols(),
                    "query dimension does not match dataset"
                );
                anyhow::ensure!(search.k.get() <= data.nrows(), "k exceeds dataset size");
                let truth =
                    load_groundtruth(input, data.as_view(), queries.as_view(), search.k.get())?;
                validate_groundtruth(
                    truth.as_view(),
                    queries.nrows(),
                    data.nrows(),
                    search.k.get(),
                )?;
                Some(truth)
            }
            (None, _) => None,
            _ => anyhow::bail!("search requires queries"),
        };
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut builds = Vec::new();
        let mut searches = Vec::new();
        for repetition in 0..input.build.repetitions.get() {
            let (index, graph, mut result) = runtime.block_on(build(input, data.as_view()))?;
            result.repetition = repetition;
            // Membership validation and groundtruth never contribute to build/search timing.
            index.provider().validate_membership(data.nrows())?;
            writeln!(
                output,
                "Build {}: {:.3}s, {} lists, {} splits",
                repetition, result.total_seconds, result.lists, result.splits
            )?;
            if let (Some(settings), Some(queries), Some(truth)) =
                (&input.search, &queries, &groundtruth)
            {
                let runs = runtime.block_on(measure_search(
                    index.provider(),
                    &graph,
                    queries.as_view(),
                    truth.as_view(),
                    settings,
                    repetition,
                ))?;
                for run in &runs {
                    writeln!(
                        output,
                        "  nprobe={}: {:.1} QPS, recall@{}={:.4}, scanned={:.1}",
                        run.requested_nprobe,
                        run.qps,
                        settings.k,
                        run.recall_at_k,
                        run.mean_points_scanned
                    )?;
                }
                searches.extend(runs);
            }
            builds.push(result);
        }
        Ok(Results {
            implementation: "diskann::ivf + benchmark-local static centroid-graph adapter",
            provider: "in-memory-fp32",
            build_routing: "exhaustive-centroids",
            search_routing: "diskann-centroid-graph",
            distance: "squared_l2",
            threads: 1,
            vectors: data.nrows(),
            dimensions: data.ncols(),
            builds,
            searches,
        })
    }
}

#[derive(Serialize)]
struct Results {
    implementation: &'static str,
    provider: &'static str,
    build_routing: &'static str,
    search_routing: &'static str,
    distance: &'static str,
    threads: usize,
    vectors: usize,
    dimensions: usize,
    builds: Vec<BuildResult>,
    searches: Vec<SearchResult>,
}

#[derive(Serialize)]
struct BuildResult {
    repetition: usize,
    ivf_seconds: f64,
    centroid_graph_seconds: f64,
    total_seconds: f64,
    vectors_per_second: f64,
    inserted: usize,
    splits: usize,
    lists: usize,
    empty_lists: usize,
    max_list_size: usize,
}

#[derive(Serialize)]
struct SearchResult {
    build_repetition: usize,
    repetition: usize,
    k: usize,
    requested_nprobe: usize,
    effective_nprobe: usize,
    queries: usize,
    total_seconds: f64,
    qps: f64,
    latency_ns: Percentiles<u64>,
    recall_at_k: f64,
    mean_points_scanned: f64,
    scanned_fraction: f64,
    mean_lists_scanned: f64,
    mean_centroid_comparisons: f64,
    mean_results_returned: f64,
}

async fn build(
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
    for first in (0..data.nrows()).step_by(settings.batch_size.get()) {
        let end = first
            .saturating_add(settings.batch_size.get())
            .min(data.nrows());
        let points: Vec<_> = (first..end).map(|i| (i as u32, data.row(i))).collect();
        let stats = index
            .insert_batch(&Strategy, &DefaultContext, &points)
            .await?;
        inserted += stats.inserted;
        splits += stats.splits;
    }
    let ivf_seconds = start.elapsed().as_secs_f64();
    let graph_start = Instant::now();
    let graph = CentroidGraph::build(index.provider(), &input.graph).await?;
    let centroid_graph_seconds = graph_start.elapsed().as_secs_f64();
    let total_seconds = start.elapsed().as_secs_f64();
    let store = index.provider();
    let result = BuildResult {
        repetition: 0,
        ivf_seconds,
        centroid_graph_seconds,
        total_seconds,
        vectors_per_second: data.nrows() as f64 / total_seconds,
        inserted,
        splits,
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

async fn measure_search(
    store: &Store,
    graph: &CentroidGraph,
    queries: rowmajor::Ref<'_, f32>,
    truth: rowmajor::Ref<'_, u32>,
    settings: &Search,
    build_repetition: usize,
) -> anyhow::Result<Vec<SearchResult>> {
    let mut results = Vec::new();
    for nprobe in &settings.nprobe {
        for _ in 0..settings.warmup_repetitions {
            for query in queries.rows() {
                black_box(
                    graph
                        .search(
                            store,
                            query,
                            settings.k,
                            nprobe.get(),
                            settings.l_search.get(),
                        )
                        .await?,
                );
            }
        }
        for repetition in 0..settings.repetitions.get() {
            let mut samples = Vec::with_capacity(queries.nrows());
            let start = Instant::now();
            for query in queries.rows() {
                let query_start = Instant::now();
                let result = graph
                    .search(
                        store,
                        query,
                        settings.k,
                        nprobe.get(),
                        settings.l_search.get(),
                    )
                    .await?;
                let latency = query_start.elapsed().as_nanos() as u64;
                samples.push((result, latency));
            }
            let total_seconds = start.elapsed().as_secs_f64();
            let mut hits = 0;
            let mut scanned = 0;
            let mut lists = 0;
            let mut centroid_comparisons = 0;
            let mut returned = 0;
            let mut latencies = Vec::with_capacity(samples.len());
            for ((result, latency), expected) in samples.into_iter().zip(truth.rows()) {
                hits += recall_hits(&result.ids, expected, settings.k.get());
                scanned += result.scanned;
                lists += result.lists;
                centroid_comparisons += result.centroid_comparisons;
                returned += result.ids.len();
                latencies.push(latency);
            }
            let count = queries.nrows() as f64;
            results.push(SearchResult {
                build_repetition,
                repetition,
                k: settings.k.get(),
                requested_nprobe: nprobe.get(),
                effective_nprobe: nprobe.get().min(store.lists.len()),
                queries: queries.nrows(),
                total_seconds,
                qps: count / total_seconds,
                latency_ns: percentiles::compute_percentiles(&mut latencies)?,
                recall_at_k: hits as f64 / (count * settings.k.get() as f64),
                mean_points_scanned: scanned as f64 / count,
                scanned_fraction: scanned as f64 / (count * store.len() as f64),
                mean_lists_scanned: lists as f64 / count,
                mean_centroid_comparisons: centroid_comparisons as f64 / count,
                mean_results_returned: returned as f64 / count,
            });
        }
    }
    Ok(results)
}

fn recall_hits(actual: &[u32], expected: &[u32], k: usize) -> usize {
    // Duplicates and short result sets must never improve the score.
    actual
        .iter()
        .take(k)
        .copied()
        .collect::<HashSet<_>>()
        .iter()
        .filter(|id| expected[..k].contains(id))
        .count()
}

fn load_data(
    input: &GraphIvfBuild,
) -> anyhow::Result<(rowmajor::Owned<f32>, Option<rowmajor::Owned<f32>>)> {
    match &input.dataset {
        Dataset::Synthetic {
            vectors,
            queries,
            dimensions,
            seed,
        } => {
            let mut rng = StdRng::seed_from_u64(*seed);
            let data = random_matrix(vectors.get(), dimensions.get(), &mut rng)?;
            let queries = input
                .search
                .as_ref()
                .map(|_| random_matrix(queries.get(), dimensions.get(), &mut rng))
                .transpose()?;
            Ok((data, queries))
        }
        Dataset::Files { data, queries, .. } => {
            let data = datafiles::load_dataset(BinFile(data))?;
            let queries = if input.search.is_some() {
                Some(datafiles::load_dataset(BinFile(
                    queries
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("missing queries file"))?,
                ))?)
            } else {
                None
            };
            Ok((data, queries))
        }
    }
}

fn random_matrix(
    rows: usize,
    cols: usize,
    rng: &mut StdRng,
) -> anyhow::Result<rowmajor::Owned<f32>> {
    let mut matrix = rowmajor::Owned::try_from_element(rows, cols, 0.0)?;
    for x in matrix.as_mut_slice() {
        *x = rng.random_range(-1.0..1.0);
    }
    Ok(matrix)
}

fn validate_vectors(data: rowmajor::Ref<'_, f32>, name: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        data.nrows() > 0 && data.ncols() > 0,
        "{name} must not be empty"
    );
    // Also reject finite coordinates whose squared L2 distances could overflow.
    let limit = (f32::MAX as f64 / (8.0 * data.ncols() as f64)).sqrt();
    anyhow::ensure!(
        data.as_slice()
            .iter()
            .all(|x| x.is_finite() && (f64::from(*x)).abs() <= limit),
        "{name} contains non-finite or excessively large coordinates"
    );
    Ok(())
}

fn load_groundtruth(
    input: &GraphIvfBuild,
    data: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: usize,
) -> anyhow::Result<rowmajor::Owned<u32>> {
    if let Dataset::Files {
        groundtruth: Some(path),
        ..
    } = &input.dataset
    {
        return datafiles::load_groundtruth(BinFile(path), Some(k));
    }
    let mut truth = rowmajor::Owned::try_from_element(queries.nrows(), k, 0)?;
    for (query, out) in queries.rows().zip(truth.rows_mut()) {
        // Independent exhaustive oracle; never consult the constructed index.
        let mut distances: Vec<(u32, f32)> = data
            .rows()
            .enumerate()
            .map(|(id, vector)| (id as u32, SquaredL2::evaluate(query, vector)))
            .collect();
        let compare =
            |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
        if k < distances.len() {
            distances.select_nth_unstable_by(k, compare);
            distances.truncate(k);
        }
        distances.sort_unstable_by(compare);
        for (out, (id, _)) in out.iter_mut().zip(distances) {
            *out = id;
        }
    }
    Ok(truth)
}

fn validate_groundtruth(
    truth: rowmajor::Ref<'_, u32>,
    queries: usize,
    vectors: usize,
    k: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        truth.nrows() == queries && truth.ncols() >= k,
        "groundtruth shape does not match queries/k"
    );
    for row in truth.rows() {
        let ids = &row[..k];
        anyhow::ensure!(
            ids.iter().all(|&id| (id as usize) < vectors),
            "groundtruth ID is outside dataset"
        );
        anyhow::ensure!(
            ids.iter().copied().collect::<HashSet<_>>().len() == k,
            "groundtruth contains duplicate IDs"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
