/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::Write,
};

use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_utils::views::rowmajor::Matrix;

use super::{
    build::build,
    data::{load_data, load_groundtruth, validate_groundtruth, validate_vectors},
    result::Results,
    search::measure_search,
};
use crate::inputs::graph_ivf::GraphIvfBuild;

pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("graph-ivf-static-f32", Static)?;
    #[cfg(feature = "ivf-two-means")]
    registry.register(
        "graph-ivf-two-means-strategies",
        super::two_means::Strategies,
    )?;
    #[cfg(not(feature = "ivf-two-means"))]
    registry.register_gated(
        "graph-ivf-two-means",
        "graph-ivf-two-means-strategies",
        diskann_benchmark_runner::Features::new("ivf-two-means"),
        "Complete two-means strategy comparison",
    )?;
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
            result.index_fingerprint = index_fingerprint(index.provider());
            writeln!(
                output,
                "Build {}: {:.3}s, {} lists, {} splits; two-means {:.6}s, reassignment {:.6}s, other IVF {:.6}s",
                repetition, result.total_seconds, result.lists, result.splits,
                result.two_means_seconds, result.reassignment_seconds, result.other_ivf_seconds
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

fn index_fingerprint(store: &super::provider::Store) -> String {
    let mut digest = DefaultHasher::new();
    for (id, list) in &store.lists {
        id.hash(&mut digest);
        list.centroid.len().hash(&mut digest);
        for value in &list.centroid {
            value.to_bits().hash(&mut digest);
        }
        list.members.hash(&mut digest);
    }
    format!("{:016x}", digest.finish())
}
