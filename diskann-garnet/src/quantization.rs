use std::{num::NonZero, sync::RwLock};

use diskann::utils::VectorRepr;
use diskann_quantization::{
    CompressInto,
    algorithms::{Transform, TransformKind, transforms::NewTransformError},
    alloc::{GlobalAllocator, ScopedAllocator},
    minmax,
    num::Positive,
    spherical::{
        self, Data, PreScale, SphericalQuantizer, SupportedMetric,
        iface::{self, Opaque, OpaqueMut, Quantizer},
    },
};
use diskann_utils::views::MatrixView;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::provider::{GarnetDistanceComputer, GarnetQueryComputer};

#[derive(Debug, Error)]
pub enum GarnetQuantizerError {
    #[error("Quantization training error: {0}")]
    Training(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Quantization alloc error: {0}")]
    Alloc(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Query computer error: {0}")]
    QueryComputer(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("Binary quantization error: {0}")]
    Compression(Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("No quantizer found")]
    NoQuantizer,
    #[error("Got zero dimension")]
    ZeroDim,
    #[error("Transform error: {0}")]
    BadTransform(#[from] NewTransformError),
}

/// Quantizer trait that all diskann-garnet quantizers must implement
pub trait GarnetQuantizer: Send + Sync {
    /// Check whether the quantizer is ready to be used
    fn is_prepared(&self) -> bool;
    /// Returns the number of vectors needed before the quantizer can be trained
    fn required_vectors(&self) -> usize;
    /// Returns the size of a quantized vector
    fn canonical_bytes(&self) -> usize;
    /// Train the quantizer.
    /// Each row of the matrix will be a vector
    fn train(&self, metric: Metric, data: MatrixView<f32>) -> Result<(), GarnetQuantizerError>;
    /// Quantize a vector
    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError>;
    /// Returns a distance computer for comparing quantized vectors
    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError>;
    /// Returns a query computer for comparing distances to a particular query
    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError>;
}

/// Type-erased distance computer
pub trait DynDistanceComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32;
}

/// Type-erased query computer
pub trait DynQueryComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8]) -> f32;
}

/// Spherical 1-bit quantization.
///
/// This quantizer corresponds to `BIN` quantizer in the Redis protocol. It requires hundreds of
/// vectors (but not thousands) for training. Quantized vectors have 1 bit per dimension plus up
/// to 6 bytes of overhead.
pub struct Spherical1Bit {
    dim: usize,
    inner: RwLock<Option<spherical::iface::Impl<1, GlobalAllocator>>>,
}

impl Spherical1Bit {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            inner: RwLock::new(None),
        }
    }
}

impl GarnetQuantizer for Spherical1Bit {
    fn is_prepared(&self) -> bool {
        self.inner.read().unwrap().is_some()
    }

    fn required_vectors(&self) -> usize {
        1000
    }

    fn canonical_bytes(&self) -> usize {
        Data::<1, GlobalAllocator>::canonical_bytes(self.dim)
    }

    fn train(
        &self,
        metric_type: Metric,
        data: MatrixView<f32>,
    ) -> Result<(), GarnetQuantizerError> {
        let mut rng = rand::rng();
        let quantizer = SphericalQuantizer::train(
            data.as_view(),
            TransformKind::DoubleHadamard {
                target_dim: diskann_quantization::algorithms::transforms::TargetDim::Same,
            },
            SupportedMetric::try_from(metric_type)
                .map_err(|e| GarnetQuantizerError::Training(Box::new(e)))?,
            PreScale::ReciprocalMeanNorm,
            &mut rng,
            GlobalAllocator,
        )
        .map_err(|e| GarnetQuantizerError::Training(Box::new(e)))?;

        let mut inner = self.inner.write().unwrap();
        *inner = Some(
            spherical::iface::Impl::<1>::new(quantizer)
                .map_err(|e| GarnetQuantizerError::Alloc(Box::new(e)))?,
        );

        Ok(())
    }

    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            spherical::iface::Quantizer::<GlobalAllocator>::compress(
                quantizer,
                v,
                OpaqueMut::new(into),
                ScopedAllocator::global(),
            )
            .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
            Ok(())
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }

    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            let computer = quantizer
                .distance_computer(GlobalAllocator)
                .map_err(|e| GarnetQuantizerError::Alloc(Box::new(e)))?;
            Ok(GarnetDistanceComputer::new(computer))
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }

    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            let computer = quantizer
                .fused_query_computer(
                    query,
                    iface::QueryLayout::FullPrecision,
                    true,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .map_err(|e| GarnetQuantizerError::QueryComputer(Box::new(e)))?;
            Ok(GarnetQueryComputer::new(computer))
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }
}

impl DynDistanceComputer for iface::DistanceComputer {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        <Self as DistanceFunction<Opaque<'_>, Opaque<'_>, _>>::evaluate_similarity(
            self,
            Opaque::new(a),
            Opaque::new(b),
        )
        .unwrap()
    }
}

impl DynQueryComputer for iface::QueryComputer {
    fn evaluate_similarity(&self, a: &[u8]) -> f32 {
        <Self as PreprocessedDistanceFunction<Opaque<'_>, _>>::evaluate_similarity(
            self,
            Opaque::new(a),
        )
        .unwrap()
    }
}

/// 8-bit scalar quantizer using MinMax
///
/// This quantizer requires no training at all and is usable immediately on the first first. Each
/// quantized vector has 8 bits per dimension and 20 bytes of overhead.
pub struct MinMax8Bit {
    dim: usize,
    metric: Metric,
    inner: minmax::MinMaxQuantizer,
}

impl MinMax8Bit {
    pub fn new(dim: usize, metric: Metric) -> Result<Self, GarnetQuantizerError> {
        let dim = match NonZero::new(dim) {
            Some(d) => d,
            None => return Err(GarnetQuantizerError::ZeroDim),
        };
        let mut rng = rand::rng();
        let transform = Transform::new(
            TransformKind::DoubleHadamard {
                target_dim: diskann_quantization::algorithms::transforms::TargetDim::Same,
            },
            dim,
            Some(&mut rng),
            GlobalAllocator,
        )?;
        let grid_scale = Positive::new(1.0).unwrap();

        Ok(Self {
            dim: dim.get(),
            metric,
            inner: minmax::MinMaxQuantizer::new(transform, grid_scale),
        })
    }
}

impl GarnetQuantizer for MinMax8Bit {
    fn is_prepared(&self) -> bool {
        true
    }

    fn required_vectors(&self) -> usize {
        0
    }

    fn canonical_bytes(&self) -> usize {
        minmax::Data::<8>::canonical_bytes(self.dim)
    }

    fn train(&self, _metric: Metric, _data: MatrixView<f32>) -> Result<(), GarnetQuantizerError> {
        Ok(())
    }

    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError> {
        let into = minmax::DataMutRef::<8>::from_canonical_front_mut(into, self.dim)
            .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
        self.inner
            .compress_into(v, into)
            .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
        Ok(())
    }

    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError> {
        let computer = GarnetDistanceComputer::new(
            <diskann_providers::common::MinMax8 as VectorRepr>::distance(
                self.metric,
                Some(self.dim),
            ),
        );
        Ok(computer)
    }

    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError> {
        let computer = GarnetQueryComputer::new(MinMax8BitQueryComputer::new(
            &self.inner,
            query,
            self.dim,
            self.metric,
        )?);
        Ok(computer)
    }
}

impl DynDistanceComputer for diskann_providers::common::FnPtr<diskann_providers::common::MinMax8> {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        let a = diskann_providers::common::MinMax8::from_bytes(a);
        let b = diskann_providers::common::MinMax8::from_bytes(b);
        <Self as DistanceFunction<_, _>>::evaluate_similarity(self, a, b)
    }
}
struct MinMax8BitQueryComputer(
    diskann_providers::common::BufferedFnPtr<diskann_providers::common::MinMax8>,
);

impl MinMax8BitQueryComputer {
    fn new(
        quantizer: &minmax::MinMaxQuantizer,
        query: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Self, GarnetQuantizerError> {
        let mut v = vec![Default::default(); minmax::Data::<8>::canonical_bytes(dim)];
        quantizer
            .compress_into(
                query,
                minmax::DataMutRef::<8>::from_canonical_front_mut(&mut v, dim)
                    .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?,
            )
            .map_err(|e| GarnetQuantizerError::Compression(Box::new(e)))?;
        let inner = diskann_providers::common::MinMax8::query_distance(
            diskann_providers::common::MinMax8::from_bytes(&v),
            metric,
        );
        Ok(Self(inner))
    }
}

impl DynQueryComputer for MinMax8BitQueryComputer {
    fn evaluate_similarity(&self, a: &[u8]) -> f32 {
        let a = diskann_providers::common::MinMax8::from_bytes(a);
        self.0.evaluate_similarity(a)
    }
}
