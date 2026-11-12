/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Disk index quantizer implementation.

use std::num::NonZeroUsize;

use diskann::{ANNError, ANNResult, utils::VectorRepr};
use diskann_inmem::{self as di, repr::quantized};
use diskann_providers::storage::{StorageReadProvider, StorageWriteProvider};
use diskann_providers::{
    common::MinMaxElement,
    index::diskann_async::train_pq,
    model::{IndexConfiguration, MAX_PQ_TRAINING_SET_SIZE},
    storage::{PQStorage, SQStorage},
    utils::{BridgeErr, PQPathNames, create_thread_pool, gen_random_slice},
};
use diskann_quantization::{
    algorithms::transforms::{TargetDim, TransformKind},
    alloc::GlobalAllocator,
    scalar::train::ScalarQuantizationParameters,
    spherical::{PreScale, SphericalQuantizer, SupportedMetric},
};
use diskann_utils::views::rowmajor;
use tracing::info;

use crate::{
    QuantizationType, SphericalBits,
    build::builder::inmem_builder::{self, CreateBuilder},
    error::{ErrorKind, diskann_error},
};

/// Backend build compatibility for "full precision" element types.
///
/// # Support Matrix
///
/// * The types [`f32`], [`half::f16`], [`u8`], and [`i8`] support all variants of
///   [`QuantizationType`].
///
/// * [`MinMaxElement`] supports all variants of [`QuantizationType`] except for
///   [`QuantizationType::FP`]. This makes [`MinMaxElement`] suitable as a **source** for
///   index builds, but not a native type for index builds.
///
/// # Side Effects
///
/// Training data will be pulled from `data_path` and interpreted with `Self` as the element
/// type.
///
/// There are occasionaly side-effects as a result of training depending on `quantization`.
///
/// * [`QuantizationType::PQ`]: The quantizer is saved according to [`PQPathNames`].
/// * [`QuantizationType::SQ`]: The quantizer is saved according to [`SQStorage`].
/// * All other types: No current side-effects.
pub trait BuildElement: std::fmt::Debug + Sized + Send + Sync + 'static {
    fn backend<P>(
        quantization: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        data_path: &str,
        storage_provider: &P,
    ) -> ANNResult<Backend<Self>>
    where
        P: StorageReadProvider + StorageWriteProvider;
}

/// An opaque backend for creating in-memory index builders.
#[derive(Debug)]
pub struct Backend<T> {
    factory: Box<dyn CreateBuilder<T>>,
}

impl<T> Backend<T> {
    /// Construct a new [`Backend`] using [`BuildElement`].
    pub fn new<P>(
        quantization: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        data_path: &str,
        storage_provider: &P,
    ) -> ANNResult<Self>
    where
        T: BuildElement,
        P: StorageReadProvider + StorageWriteProvider,
    {
        T::backend(
            quantization,
            index_path_prefix,
            index_configuration,
            data_path,
            storage_provider,
        )
    }

    /// Construct an [`inmem_builder::InmemIndexBuilder`] from the configuration with the
    /// requested start point.
    pub(super) fn builder(
        &self,
        graph: diskann::graph::Config,
        provider: inmem_builder::Config,
        start: inmem_builder::StartPoint<T>,
    ) -> ANNResult<std::sync::Arc<dyn inmem_builder::InmemIndexBuilder<T>>>
    where
        T: 'static,
    {
        self.factory.create_builder(graph, provider, start)
    }

    fn from_factory<U>(factory: U) -> Self
    where
        U: CreateBuilder<T>,
    {
        Self {
            factory: Box::new(factory),
        }
    }
}

//----------------//
// Full Precision //
//----------------//

macro_rules! full_precision {
    ($T:ty) => {
        impl BuildElement for $T {
            fn backend<P>(
                quantization: &QuantizationType,
                index_path_prefix: &str,
                index_configuration: &IndexConfiguration,
                data_path: &str,
                storage_provider: &P,
            ) -> ANNResult<Backend<$T>>
            where
                P: StorageReadProvider + StorageWriteProvider {
                full_precision::<$T, _>(
                    quantization,
                    index_path_prefix,
                    index_configuration,
                    data_path,
                    storage_provider,
                )
            }
        }
    };
    ($($Ts:ty),+ $(,)?) => {
        $(full_precision!($Ts);)+
    };
}

full_precision!(f32, half::f16, u8, i8);

fn full_precision<T, P>(
    quantization: &QuantizationType,
    index_path_prefix: &str,
    index_configuration: &IndexConfiguration,
    data_path: &str,
    storage_provider: &P,
) -> ANNResult<Backend<T>>
where
    T: VectorRepr + di::repr::FullPrecision,
    P: StorageReadProvider + StorageWriteProvider,
{
    match train_quantizer::<T, _>(
        quantization,
        index_path_prefix,
        index_configuration,
        data_path,
        storage_provider,
    )? {
        Some(backend) => Ok(backend),
        None => Ok(Backend::from_factory(inmem_builder::FullPrecision::new(
            index_configuration.dist_metric,
        ))),
    }
}

//--------//
// MinMax //
//--------//

impl<const N: usize> BuildElement for MinMaxElement<N>
where
    Self: VectorRepr,
{
    fn backend<P>(
        quantization: &QuantizationType,
        index_path_prefix: &str,
        index_configuration: &IndexConfiguration,
        data_path: &str,
        storage_provider: &P,
    ) -> ANNResult<Backend<Self>>
    where
        P: StorageReadProvider + StorageWriteProvider,
    {
        match train_quantizer::<Self, _>(
            quantization,
            index_path_prefix,
            index_configuration,
            data_path,
            storage_provider,
        )? {
            Some(backend) => Ok(backend),
            None => Err(diskann_error!(
                ErrorKind::IndexError,
                "full-precision builds are not supported for minmax-{}",
                N
            )),
        }
    }
}

//----------//
// Adaptors //
//----------//

// Here, we take advantage of [`quantized::spherical::Spherical`] being cloneable to create
// a new [`quantized::Backend`].
impl<Q> inmem_builder::QuantizedBackend for quantized::spherical::Spherical<Q>
where
    quantized::spherical::Spherical<Q>:
        quantized::IntoBackend + std::fmt::Debug + Send + Sync + 'static,
{
    fn quantized_backend(&self) -> quantized::Backend {
        <Self as quantized::IntoBackend>::into_backend(self.clone())
    }
}

// Here, we take advantage of [`quantized::scalar::Scalar`] being cloneable to create
// a new [`quantized::Backend`].
impl<const N: usize> inmem_builder::QuantizedBackend for quantized::scalar::Scalar<N>
where
    quantized::scalar::Scalar<N>: quantized::IntoBackend,
{
    fn quantized_backend(&self) -> quantized::Backend {
        <Self as quantized::IntoBackend>::into_backend(self.clone())
    }
}

/// A custom wrapper for [`quantized::product::Product`] that externally tracks the cache
/// capacity to avoid the initial allocation in the other-wise unused base
/// [`inmem_builder::QuantizedBackend`].
#[derive(Debug)]
struct Product {
    quantizer: quantized::product::Product,
    // The number of distance tables to reserve space for.
    //
    // Should be sized to the expected number of threads.
    cache_capacity: Option<NonZeroUsize>,
}

impl Product {
    fn new(quantizer: quantized::product::Product, cache_capacity: Option<NonZeroUsize>) -> Self {
        Self {
            quantizer,
            cache_capacity,
        }
    }
}

// Create a new [`quantized::product::Product`] with its own distance table cache.
//
// The size of the cache is derived from the original struct.
impl inmem_builder::QuantizedBackend for Product {
    fn quantized_backend(&self) -> quantized::Backend {
        <_ as quantized::IntoBackend>::into_backend(self.quantizer.share(self.cache_capacity))
    }
}

///////////
// Train //
///////////

/// Train the requested quantizer and package into a [`quantized::Backend`].
///
/// Returns `None` if `quantization` is [`QuantizationType::FP`].
fn train_quantizer<T, StorageProvider>(
    quantization: &QuantizationType,
    index_path_prefix: &str,
    index_configuration: &IndexConfiguration,
    data_path: &str,
    storage_provider: &StorageProvider,
) -> ANNResult<Option<Backend<T>>>
where
    T: VectorRepr,
    StorageProvider: StorageReadProvider + StorageWriteProvider,
{
    let num_points = index_configuration.max_points;
    let p_val = MAX_PQ_TRAINING_SET_SIZE / (num_points as f64);
    match *quantization {
        QuantizationType::FP => Ok(None),
        QuantizationType::PQ { num_chunks } => {
            let table = {
                //generate pq pivots.
                let seed = index_configuration.random_seed;
                let mut rnd =
                    diskann_providers::utils::create_rnd_provider_from_optional_seed(seed)
                        .create_rnd();
                let (train_data, train_size, train_dim) =
                    gen_random_slice::<T, _>(data_path, p_val, storage_provider, &mut rnd)?;
                train_pq(
                    rowmajor::Ref::try_from_data(&train_data, train_size, train_dim)
                        .bridge_err()?,
                    num_chunks,
                    &mut rnd,
                    create_thread_pool(index_configuration.num_threads)?.as_ref(),
                )?
            };
            // The compressed data path and pivots path are derived from the index prefix.
            // are different than the ones used in quant vector generation.
            let pq_paths = PQPathNames::new(index_path_prefix);
            let pq_build_storage =
                PQStorage::new(&pq_paths.pivots, &pq_paths.compressed_data, None);
            pq_build_storage.write_pivot_data(
                table.get_pq_table(),
                None,
                table.get_chunk_offsets(),
                table.get_num_centers(),
                table.get_dim(),
                storage_provider,
            )?;

            let metric = index_configuration.dist_metric;
            let quantizer =
                quantized::product::Product::new(table.into_table(), metric.into(), None)?;

            let backend = Product::new(
                quantizer,
                NonZeroUsize::new(index_configuration.num_threads),
            );

            Ok(Some(Backend::from_factory(inmem_builder::Quantized::new(
                backend,
            ))))
        }
        QuantizationType::SQ {
            nbits,
            standard_deviation,
        } => {
            if nbits != 1 {
                return Err(diskann_error!(
                    ErrorKind::IndexConfigError("quantization"),
                    "SQ quantization is only supported for 1 bit",
                ));
            }
            let rng = diskann_providers::utils::create_rnd_provider_from_optional_seed(
                index_configuration.random_seed,
            );
            let (train_data_vector, train_size, train_dim) = gen_random_slice::<T, _>(
                data_path,
                p_val,
                storage_provider,
                &mut rng.create_rnd(),
            )?;

            let quantizer_params = if let Some(std_dev) = standard_deviation {
                ScalarQuantizationParameters::new(std_dev)
            } else {
                ScalarQuantizationParameters::default()
            };

            let quantizer = quantizer_params.train(
                rowmajor::Ref::try_from_data(&train_data_vector, train_size, train_dim)
                    .bridge_err()?,
            );

            info!("Now quantizer is trained and saving to file");
            let sq_storage = SQStorage::new(index_path_prefix);
            sq_storage.save_quantizer(&quantizer, storage_provider)?;

            let quantizer = quantized::scalar::Scalar::<1>::new(
                quantizer,
                index_configuration.dist_metric.try_into()?,
            );

            Ok(Some(Backend::from_factory(inmem_builder::Quantized::new(
                quantizer,
            ))))
        }
        QuantizationType::Spherical(SphericalBits::One) => {
            let metric: SupportedMetric =
                index_configuration.dist_metric.try_into().bridge_err()?;
            let rng = diskann_providers::utils::create_rnd_provider_from_optional_seed(
                index_configuration.random_seed,
            );
            let mut rnd = rng.create_rnd();
            let (train_data, train_size, train_dim) =
                gen_random_slice::<T, _>(data_path, p_val, storage_provider, &mut rnd)?;
            let train_data =
                rowmajor::Ref::try_from_data(&train_data, train_size, train_dim).bridge_err()?;
            let quantizer = SphericalQuantizer::train(
                train_data,
                TransformKind::DoubleHadamard {
                    target_dim: TargetDim::Natural,
                },
                metric,
                PreScale::ReciprocalMeanNorm,
                &mut rnd,
                GlobalAllocator,
            )
            .map_err(|err| ANNError::new(err).context("Failed to train spherical quantizer"))?;

            let quantizer = quantized::spherical::Spherical::new(
                quantizer.as_quantizer_impl::<1>().map_err(ANNError::new)?,
            );

            Ok(Some(Backend::from_factory(inmem_builder::Quantized::new(
                quantizer,
            ))))
        }
    }
}
