/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, path::Path, time::Instant};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_graphivf::{
    BuildParams, BuildProfile, GraphIvfIndex, GraphParams, Metric as GraphIvfMetric,
};
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

use crate::{
    inputs::graph_ivf::GraphIvfBuild,
    utils::{datafiles, SimilarityMeasure},
};

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct GraphIvfBuildStats {
    /// Time to load the on-disk `T` corpus and widen it to `f32` (the
    /// benchmark's own I/O, not part of the index build itself).
    corpus_load: MicroSeconds,
    /// End-to-end index build wall-clock.
    build_time: MicroSeconds,
    /// Per-stage build latency breakdown.
    profile: BuildProfile,
    num_points: usize,
    dim: usize,
}

impl fmt::Display for GraphIvfBuildStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Build time: {:.3}s ({} points, dim {})",
            self.build_time.as_seconds(),
            self.num_points,
            self.dim
        )?;
        writeln!(f, "  corpus_load: {:.3}s", self.corpus_load.as_seconds())?;
        write!(f, "{}", self.profile)
    }
}

/// Map the benchmark's distance measure onto the graph-IVF metric.
///
/// Graph-IVF only supports squared-L2 and cosine. Inner product is unsupported.
/// Already-normalized cosine is treated as plain L2 (it is the same ranking on
/// unit vectors and avoids a redundant re-normalization pass).
pub(super) fn to_graphivf_metric(distance: SimilarityMeasure) -> anyhow::Result<GraphIvfMetric> {
    match distance {
        SimilarityMeasure::SquaredL2 | SimilarityMeasure::CosineNormalized => {
            Ok(GraphIvfMetric::L2)
        }
        SimilarityMeasure::Cosine => Ok(GraphIvfMetric::Cosine),
        SimilarityMeasure::InnerProduct => {
            anyhow::bail!("graph-ivf does not support the inner-product metric")
        }
    }
}

/// Load a `.bin` corpus stored as `T` into a contiguous `Matrix<f32>`.
///
/// Graph-IVF consumes an `f32` corpus and re-encodes it into the stored element
/// type `T` internally, so the on-disk `T` representation is widened here.
pub(super) fn load_corpus_as_f32<T: VectorRepr>(path: &Path) -> anyhow::Result<Matrix<f32>> {
    let corpus: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(path))?;
    let (nrows, ncols) = (corpus.nrows(), corpus.ncols());
    let widened = T::as_f32(corpus.as_slice())
        .map_err(|e| anyhow::anyhow!("failed to widen corpus to f32: {e}"))?;
    let mut out = Matrix::<f32>::new(0.0, nrows, ncols);
    out.as_mut_slice().copy_from_slice(&widened);
    Ok(out)
}

pub(super) fn build_graph_ivf<T>(params: &GraphIvfBuild) -> anyhow::Result<GraphIvfBuildStats>
where
    T: VectorRepr,
{
    let data_path = params.data.to_string_lossy().to_string();
    let corpus_load_start = Instant::now();
    let corpus = load_corpus_as_f32::<T>(Path::new(&data_path))?;
    let corpus_load: MicroSeconds = corpus_load_start.elapsed().into();
    let num_points = corpus.nrows();
    let dim = corpus.ncols();

    let build_params = BuildParams {
        num_clusters: params.num_clusters,
        metric: to_graphivf_metric(params.distance)?,
        sample_size: params.sample_size,
        kmeans_iters: params.kmeans_iters,
        assign_l: params.assign_l,
        graph: GraphParams {
            degree: params.graph_degree,
            slack: params.graph_slack,
            l_build: params.graph_l_build,
            alpha: params.graph_alpha,
        },
        num_threads: params.num_threads,
        seed: params.seed,
    };

    let save_prefix = Path::new(&params.save_path);

    let start = Instant::now();
    let (_index, profile) =
        GraphIvfIndex::<T>::build_profiled(corpus.as_view(), &build_params, save_prefix)?;
    let build_time: MicroSeconds = start.elapsed().into();

    Ok(GraphIvfBuildStats {
        corpus_load,
        build_time,
        profile,
        num_points,
        dim,
    })
}
