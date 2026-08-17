/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{fmt, path::Path, time::Instant};

use diskann::utils::VectorRepr;
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_graphivf::{
    AssignMethod, BuildParams, BuildProfile, CentroidInit, CentroidRouting, CentroidSearch,
    EmptyClusterPolicy, GraphIvfIndex, GraphParams, Metric as GraphIvfMetric,
};
use diskann_utils::views::Matrix;
use serde::{Deserialize, Serialize};

use crate::{
    backend::graph_ivf::element::GraphIvfElement,
    inputs::graph_ivf::{
        AssignMethodConfig, CentroidSearchConfig, EmptyClusterConfig, GraphIvfStaticBuild,
        StaticRoutingConfig,
    },
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
/// Graph-IVF supports squared-L2, cosine, and a hybrid inner-product metric
/// (build clusters under L2, search scores by inner product). Already-
/// normalized cosine is treated as plain L2 (it is the same ranking on unit
/// vectors and avoids a redundant re-normalization pass).
pub(super) fn to_graphivf_metric(distance: SimilarityMeasure) -> anyhow::Result<GraphIvfMetric> {
    match distance {
        SimilarityMeasure::SquaredL2 | SimilarityMeasure::CosineNormalized => {
            Ok(GraphIvfMetric::L2)
        }
        SimilarityMeasure::Cosine => Ok(GraphIvfMetric::Cosine),
        SimilarityMeasure::InnerProduct => Ok(GraphIvfMetric::InnerProduct),
    }
}

/// Map the config mirror onto the library's centroid-search mode.
pub(super) fn to_centroid_search(mode: CentroidSearchConfig) -> CentroidSearch {
    match mode {
        CentroidSearchConfig::Graph => CentroidSearch::Graph,
        CentroidSearchConfig::Exact => CentroidSearch::Exact,
    }
}

/// Map the static routing config onto the library's routing parameters.
pub(super) fn to_routing(routing: StaticRoutingConfig) -> CentroidRouting {
    match routing {
        StaticRoutingConfig::Graph {
            assign_l,
            graph_degree,
            graph_slack,
            graph_l_build,
            graph_alpha,
        } => CentroidRouting::Graph {
            graph: GraphParams {
                degree: graph_degree,
                slack: graph_slack,
                l_build: graph_l_build,
                alpha: graph_alpha,
            },
            assign_l,
        },
        StaticRoutingConfig::Exact => CentroidRouting::Exact,
    }
}

/// Load a `.bin` corpus stored as `T`, returning it with its logical dimension.
///
/// The row width on disk is not the vector's logical dimension for every element type —
/// a quantized row carries per-vector metadata alongside its codes — so the dimension
/// comes from [`VectorRepr::full_dimension`] rather than the stored width.
pub(super) fn load_stored_corpus<T: VectorRepr>(path: &Path) -> anyhow::Result<(Matrix<T>, usize)> {
    let corpus: Matrix<T> = datafiles::load_dataset(datafiles::BinFile(path))?;
    anyhow::ensure!(corpus.nrows() > 0, "corpus {} is empty", path.display());
    let dim = T::full_dimension(corpus.row(0))
        .map_err(|e| anyhow::anyhow!("failed to determine corpus dimension: {e}"))?;
    Ok((corpus, dim))
}

/// Widen a stored corpus into a contiguous `dim`-wide `f32` matrix.
pub(super) fn decompress_to_f32<T: VectorRepr>(
    corpus: &Matrix<T>,
    dim: usize,
) -> anyhow::Result<Matrix<f32>> {
    let nrows = corpus.nrows();
    let mut buf = vec![0.0f32; nrows * dim];
    for (src, dst) in corpus
        .as_slice()
        .chunks(corpus.ncols())
        .zip(buf.chunks_mut(dim))
    {
        T::as_f32_into(src, dst)
            .map_err(|e| anyhow::anyhow!("failed to widen corpus to f32: {e}"))?;
    }
    Ok(Matrix::try_from(buf.into_boxed_slice(), nrows, dim)?)
}

/// A corpus loaded in whichever form graph-IVF needs for element type `T`.
///
/// The variant is chosen by [`GraphIvfElement::STORED_VERBATIM`], not by the job config.
enum LoadedCorpus<T: VectorRepr> {
    /// Widened to `f32`; graph-IVF encodes each row to `T` while writing the lists.
    Plain(Matrix<f32>),
    /// Still encoded as `T`; graph-IVF stores the rows verbatim and decodes its own
    /// clustering copy. `dim` is the logical dimension, narrower than the stored width.
    Stored { corpus: Matrix<T>, dim: usize },
}

impl<T: GraphIvfElement> LoadedCorpus<T> {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let (corpus, dim) = load_stored_corpus::<T>(path)?;
        if T::STORED_VERBATIM {
            Ok(Self::Stored { corpus, dim })
        } else {
            Ok(Self::Plain(decompress_to_f32(&corpus, dim)?))
        }
    }

    fn num_points(&self) -> usize {
        match self {
            Self::Plain(corpus) => corpus.nrows(),
            Self::Stored { corpus, .. } => corpus.nrows(),
        }
    }

    /// The logical vector dimension, which for [`Self::Stored`] is narrower than the
    /// stored row width the index reports as its dimension.
    fn dim(&self) -> usize {
        match self {
            Self::Plain(corpus) => corpus.ncols(),
            Self::Stored { dim, .. } => *dim,
        }
    }

    fn build(&self, params: &BuildParams, prefix: &Path) -> anyhow::Result<BuildProfile> {
        let (_index, profile) = match self {
            Self::Plain(corpus) => {
                GraphIvfIndex::<T>::build_profiled(corpus.as_view(), params, prefix)?
            }
            Self::Stored { corpus, .. } => GraphIvfIndex::<T>::build_compressed_profiled(
                corpus.as_view(),
                CentroidInit::Forgy {
                    samples: params.effective_sample_size(corpus.nrows()),
                },
                params,
                prefix,
            )?,
        };
        Ok(profile)
    }
}

pub(super) fn build_graph_ivf<T>(params: &GraphIvfStaticBuild) -> anyhow::Result<GraphIvfBuildStats>
where
    T: GraphIvfElement,
{
    let data_path = params.data.to_string_lossy().to_string();
    let corpus_load_start = Instant::now();
    let corpus = LoadedCorpus::<T>::load(Path::new(&data_path))?;
    let corpus_load: MicroSeconds = corpus_load_start.elapsed().into();
    let num_points = corpus.num_points();
    let dim = corpus.dim();

    let build_params = BuildParams {
        num_clusters: params.num_clusters,
        metric: to_graphivf_metric(params.distance)?,
        sample_size: params.sample_size,
        kmeans_iters: params.kmeans_iters,
        routing: to_routing(params.routing),
        num_threads: params.num_threads,
        seed: params.seed,
        assign_method: match params.assign_method {
            AssignMethodConfig::Exact => AssignMethod::Exact,
            AssignMethodConfig::Graph {
                rebuild_every,
                rerank,
                assign_l,
                graph_degree,
                graph_slack,
                graph_l_build,
                graph_alpha,
            } => AssignMethod::Graph {
                rebuild_every,
                rerank,
                assign_l,
                graph: GraphParams {
                    degree: graph_degree,
                    slack: graph_slack,
                    l_build: graph_l_build,
                    alpha: graph_alpha,
                },
            },
        },
        empty_clusters: match params.empty_clusters {
            EmptyClusterConfig::Zero => EmptyClusterPolicy::Zero,
            EmptyClusterConfig::PreserveOld => EmptyClusterPolicy::PreserveOld,
            EmptyClusterConfig::ReseedFarthest => EmptyClusterPolicy::ReseedFarthest,
        },
        normalize_centroids: false,
    };

    let save_prefix = Path::new(&params.save_path);

    let start = Instant::now();
    let profile = corpus.build(&build_params, save_prefix)?;
    let build_time: MicroSeconds = start.elapsed().into();

    Ok(GraphIvfBuildStats {
        corpus_load,
        build_time,
        profile,
        num_points,
        dim,
    })
}
