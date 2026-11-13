/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroUsize, sync::Arc};

use diskann::graph::{Config, DiskANNIndex};
use diskann_inmem::{
    self as inmem, Provider,
    num::{Capacity, MaxDegree, Range},
    repr::Full,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::distance::Metric;

use super::{
    object::{Index, PartialIndex},
    parse::{Representation, quantization, scalar, spherical},
};
use crate::support::datatype::{DataType, Dataset, DatasetView};

pub(super) fn create(
    representation: &Representation,
    metric: Metric,
    graph_config: &Config,
    data: DatasetView<'_>,
) -> anyhow::Result<Arc<dyn Index>> {
    match representation {
        Representation::FullPrecision { data_type } => {
            create_full(data, *data_type, metric, graph_config)
        }
        Representation::Spherical { bits, rerank } => {
            create_spherical(data, *bits, *rerank, metric, graph_config)
        }
        Representation::Product { chunks, rerank } => {
            create_product(data, *chunks, *rerank, metric, graph_config)
        }
        Representation::Scalar {
            bits,
            override_to_cosine_normalized,
            rerank,
        } => create_scalar(
            data,
            *bits,
            *override_to_cosine_normalized,
            *rerank,
            metric,
            graph_config,
        ),
    }
}

fn create_full(
    data: DatasetView<'_>,
    data_type: DataType,
    metric: Metric,
    graph_config: &Config,
) -> anyhow::Result<Arc<dyn Index>> {
    if data.data_type() != data_type {
        anyhow::bail!(
            "mismatched data types for start point - expected {}, got {}",
            data_type,
            data.data_type(),
        );
    }

    let start_points = data.medoid();
    let capacity = Capacity::new(data.nrows());
    let max_degree = graph_config.max_degree().get();
    let index_config = graph_config.clone();

    let index = match start_points {
        Dataset::F32(v) => finish(
            Provider::new(Full::config(
                capacity,
                MaxDegree::new(max_degree),
                metric,
                v,
            )?)?,
            index_config,
        ),
        Dataset::F16(v) => finish(
            Provider::new(Full::config(
                capacity,
                MaxDegree::new(max_degree),
                metric,
                v,
            )?)?,
            index_config,
        ),
        Dataset::U8(v) => finish(
            Provider::new(Full::config(
                capacity,
                MaxDegree::new(max_degree),
                metric,
                v,
            )?)?,
            index_config,
        ),
        Dataset::I8(v) => finish(
            Provider::new(Full::config(
                capacity,
                MaxDegree::new(max_degree),
                metric,
                v,
            )?)?,
            index_config,
        ),
    };

    Ok(index)
}

fn create_spherical(
    data: DatasetView<'_>,
    bits: spherical::Bits,
    rerank: quantization::Rerank,
    metric: Metric,
    graph_config: &Config,
) -> anyhow::Result<Arc<dyn Index>> {
    use diskann_quantization::{
        algorithms::transforms,
        alloc::GlobalAllocator,
        spherical::{PreScale, SphericalQuantizer},
    };
    use inmem::repr::quantized::{Backend, Config, spherical::Spherical};
    use rand::SeedableRng;

    let DatasetView::F32(data) = data else {
        anyhow::bail!("spherical quantization only supports f32 data");
    };

    // Step 1: Train a generic quantizer.
    let quantizer = SphericalQuantizer::train(
        data,
        transforms::TransformKind::DoubleHadamard {
            target_dim: transforms::TargetDim::Same,
        },
        metric.try_into().map_err(|_| {
            anyhow::anyhow!("metric {} not supported for spherical quantization", metric)
        })?,
        PreScale::ReciprocalMeanNorm,
        &mut rand::rngs::StdRng::seed_from_u64(0xc0ff33),
        GlobalAllocator,
    )?;

    // Step 2: Associate it with the target bit-width.
    let capacity = Capacity::new(data.nrows());
    let max_degree = MaxDegree::new(graph_config.max_degree().get());
    let start_point = rowmajor::Owned::row_vector(Box::from(
        <f32 as diskann_utils::sampling::medoid::ComputeMedoid>::compute_medoid(data),
    ));
    let rerank = quantizer_rerank(rerank, metric);

    let backend = match bits {
        spherical::Bits::One => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<1>()?)),
        spherical::Bits::Two => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<2>()?)),
        spherical::Bits::Four => Backend::new(Spherical::new(quantizer.as_quantizer_impl::<4>()?)),
    };

    let config = Config::new(backend, capacity, max_degree, start_point, rerank)?;
    let index_config = graph_config.clone();

    Ok(finish(Provider::new(config)?, index_config))
}

fn create_product(
    data: DatasetView<'_>,
    chunks: NonZeroUsize,
    rerank: quantization::Rerank,
    metric: Metric,
    graph_config: &Config,
) -> anyhow::Result<Arc<dyn Index>> {
    use diskann_quantization::{
        Parallelism,
        cancel::DontCancel,
        product::{self, train::TrainQuantizer},
        random,
        views::ChunkOffsets,
    };

    let DatasetView::F32(data) = data else {
        anyhow::bail!("product quantization only supports f32 data");
    };

    // Step 1: Train a generic quantizer.
    let trainer = product::train::LightPQTrainingParameters::new(256, 2);
    let Some(dim) = NonZeroUsize::new(data.ncols()) else {
        anyhow::bail!("cannot compress a zero-dimensional dataset");
    };

    let table = trainer.train(
        data,
        ChunkOffsets::partition(dim, chunks)?.as_view(),
        Parallelism::Sequential,
        &random::StdRngBuilder::new(0xc0ff33),
        &DontCancel,
    )?;

    let start_point = rowmajor::Owned::row_vector(Box::from(
        <f32 as diskann_utils::sampling::medoid::ComputeMedoid>::compute_medoid(data),
    ));

    // Step 2: Create the quantizer.
    let product = inmem::repr::quantized::product::Product::new(table, metric.into(), None)?;

    // Step 3: Create the config.
    let config = inmem::repr::quantized::Quantized::config(
        product,
        Capacity::new(data.nrows()),
        MaxDegree::new(graph_config.max_degree().get()),
        start_point,
        quantizer_rerank(rerank, metric),
    )?;

    let index_config = graph_config.clone();
    Ok(finish(Provider::new(config)?, index_config))
}

fn create_scalar(
    data: DatasetView<'_>,
    bits: scalar::Bits,
    override_to_cosine_normalized: bool,
    rerank: quantization::Rerank,
    metric: Metric,
    graph_config: &Config,
) -> anyhow::Result<Arc<dyn Index>> {
    use diskann_quantization::{num::Positive, scalar as qscalar};
    use inmem::repr::quantized::{Backend, Quantized, scalar::Scalar};

    let DatasetView::F32(data) = data else {
        anyhow::bail!("scalar quantization only supports f32 data");
    };

    // Step 1: Train a generic quantizer.
    let std = Positive::<f64>::new(1.5)?;
    let trainer = qscalar::train::ScalarQuantizationParameters::new(std);
    let quantizer = trainer.train(data);

    let start_point = rowmajor::Owned::row_vector(Box::from(
        <f32 as diskann_utils::sampling::medoid::ComputeMedoid>::compute_medoid(data),
    ));

    // Step 2: Create the config.
    let scalar_metric = if override_to_cosine_normalized {
        inmem::repr::quantized::scalar::Metric::CosineNormalized
    } else {
        inmem::repr::quantized::scalar::Metric::try_from(metric)?
    };

    let backend = match bits {
        scalar::Bits::One => Backend::new(Scalar::<1>::new(quantizer, scalar_metric)),
        scalar::Bits::Two => Backend::new(Scalar::<2>::new(quantizer, scalar_metric)),
        scalar::Bits::Four => Backend::new(Scalar::<4>::new(quantizer, scalar_metric)),
        scalar::Bits::Eight => Backend::new(Scalar::<8>::new(quantizer, scalar_metric)),
    };

    let capacity = Capacity::new(data.nrows());
    let max_degree = MaxDegree::new(graph_config.max_degree().get());
    let rerank = quantizer_rerank(rerank, metric);

    let config = Quantized::config(backend, capacity, max_degree, start_point, rerank)?;

    let index_config = graph_config.clone();
    Ok(finish(Provider::new(config)?, index_config))
}

pub(super) fn create_partial(
    metric: Metric,
    graph_config: &Config,
    data: DatasetView<'_>,
) -> anyhow::Result<Arc<dyn PartialIndex>> {
    let DatasetView::F32(data) = data else {
        anyhow::bail!("static build tests only support f32 data");
    };

    let start_point = rowmajor::Owned::row_vector(Box::from(
        <f32 as diskann_utils::sampling::medoid::ComputeMedoid>::compute_medoid(data),
    ));
    let num_start_points = start_point.nrows();

    let capacity = Capacity::new(data.nrows());
    let max_degree = graph_config.max_degree().get();
    let index_config = graph_config.clone();

    let end_range = Range::start_plus_usize(data.nrows(), num_start_points)
        .ok_or_else(|| anyhow::anyhow!("cannot move start points to the end"))?;

    let full_config = Full::config(capacity, MaxDegree::new(max_degree), metric, start_point)?
        .locate_start_point_ids_at(end_range)?;

    let provider = inmem::provider::buildonly::Provider::new(full_config)?;
    Ok(Arc::new(DiskANNIndex::new(index_config, provider, None)))
}

fn finish<DP>(provider: DP, config: diskann::graph::Config) -> Arc<dyn Index>
where
    DP: diskann::provider::DataProvider,
    DiskANNIndex<DP>: Index,
{
    Arc::new(DiskANNIndex::new(config, provider, None))
}

fn quantizer_rerank(
    rerank: quantization::Rerank,
    metric: Metric,
) -> inmem::repr::quantized::Rerank {
    match rerank {
        quantization::Rerank::None => inmem::repr::quantized::Rerank::None,
        quantization::Rerank::F16 => inmem::repr::quantized::Rerank::F16 {
            metric: metric.into(),
        },
    }
}
