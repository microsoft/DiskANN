use std::{num::NonZero, sync::RwLock};

use diskann::utils::VectorRepr;
use diskann_quantization::{
    CompressInto,
    algorithms::{Transform, TransformKind, transforms::NewTransformError},
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    minmax,
    num::POSITIVE_ONE_F32,
    spherical::{
        self, Data, PreScale, SphericalQuantizer, SupportedMetric,
        iface::{self, Opaque, OpaqueMut, Quantizer},
    },
};
use diskann_utils::MatrixView;
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};
use thiserror::Error;

use crate::provider::{GarnetDistanceComputer, GarnetQueryComputer};

#[derive(Debug, Error)]
pub(crate) enum GarnetQuantizerError {
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
    #[error("Unsupported serialization/deserialization")]
    UnsupportedSerialization,
    #[error("Quantizer deserialization error: {0}")]
    Deserialization(Box<dyn std::error::Error + Send + Sync + 'static>),
}

/// Quantizer trait that all diskann-garnet quantizers must implement
pub(crate) trait GarnetQuantizer: Send + Sync {
    /// Returns the number of vectors needed before the quantizer can be trained
    fn required_vectors(&self) -> usize;
    /// Returns the size of a quantized vector
    fn bytes(&self) -> usize;
    /// Return whether the quantizer is trained.
    fn is_trained(&self) -> bool;
    /// Train the quantizer.
    /// Each row of the matrix will be a vector.
    /// Returns a lock guard for purposes of synchronization; after the guard is released, the
    /// quantizer will be accessible to all threads.
    fn train(&self, metric: Metric, data: MatrixView<f32>) -> Result<(), GarnetQuantizerError>;
    /// Quantize a vector
    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError>;
    /// Returns a distance computer for comparing quantized vectors
    fn distance_computer(&self) -> Result<GarnetDistanceComputer, GarnetQuantizerError>;
    /// Returns a query computer for comparing distances to a particular query
    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError>;
    // Serialize the quantizer state.
    fn serialize(&self) -> Result<Poly<[u8], GlobalAllocator>, GarnetQuantizerError>;
    // Deserialize the quantizer state.
    fn deserialize(&self, state: &[u8]) -> Result<(), GarnetQuantizerError>;
}

/// Type-erased distance computer
pub(crate) trait DynDistanceComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8], b: &[u8]) -> f32;
}

/// Type-erased query computer
pub(crate) trait DynQueryComputer: Send + Sync {
    fn evaluate_similarity(&self, a: &[u8]) -> f32;
}

/// Spherical 1-bit quantization.
///
/// This quantizer corresponds to `BIN` quantizer in the Redis protocol. It requires hundreds of
/// vectors (but not thousands) for training. Quantized vectors have 1 bit per dimension plus up
/// to 6 bytes of overhead.
pub(crate) struct Spherical1Bit {
    dim: usize,
    inner: RwLock<Option<spherical::iface::Impl<1, GlobalAllocator>>>,
}

impl Spherical1Bit {
    pub(crate) fn new(dim: usize) -> Self {
        Self {
            dim,
            inner: RwLock::new(None),
        }
    }
}

impl GarnetQuantizer for Spherical1Bit {
    fn required_vectors(&self) -> usize {
        1000
    }

    fn bytes(&self) -> usize {
        Data::<1, GlobalAllocator>::canonical_bytes(self.dim)
    }

    fn is_trained(&self) -> bool {
        self.inner.read().unwrap().is_some()
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

    fn serialize(&self) -> Result<Poly<[u8], GlobalAllocator>, GarnetQuantizerError> {
        let guard = self.inner.read().unwrap();
        if let Some(quantizer) = &*guard {
            quantizer
                .serialize(GlobalAllocator)
                .map_err(|e| GarnetQuantizerError::Alloc(Box::new(e)))
        } else {
            Err(GarnetQuantizerError::NoQuantizer)
        }
    }

    fn deserialize(&self, state: &[u8]) -> Result<(), GarnetQuantizerError> {
        let mut guard = self.inner.write().unwrap();
        if guard.is_some() {
            Err(GarnetQuantizerError::UnsupportedSerialization)
        } else {
            let q = spherical::iface::Impl::<1>::try_deserialize(state, GlobalAllocator)
                .map_err(|e| GarnetQuantizerError::Deserialization(Box::new(e)))?;
            *guard = Some(q);
            Ok(())
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
/// This quantizer requires no training at all and is usable immediately on the first vector. Each
/// quantized vector has 8 bits per dimension and 20 bytes of overhead.
pub(crate) struct MinMax8Bit {
    metric: Metric,
    inner: minmax::MinMaxQuantizer,
}

impl MinMax8Bit {
    pub(crate) fn new(dim: usize, metric: Metric) -> Result<Self, GarnetQuantizerError> {
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
        let grid_scale = POSITIVE_ONE_F32;

        Ok(Self {
            metric,
            inner: minmax::MinMaxQuantizer::new(transform, grid_scale),
        })
    }
}

impl GarnetQuantizer for MinMax8Bit {
    fn required_vectors(&self) -> usize {
        0
    }

    fn bytes(&self) -> usize {
        minmax::Data::<8>::canonical_bytes(self.inner.dim())
    }

    fn is_trained(&self) -> bool {
        true
    }

    fn train(&self, _metric: Metric, _data: MatrixView<f32>) -> Result<(), GarnetQuantizerError> {
        Ok(())
    }

    fn compress(&self, v: &[f32], into: &mut [u8]) -> Result<(), GarnetQuantizerError> {
        let into = minmax::DataMutRef::<8>::from_canonical_front_mut(into, self.inner.dim())
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
                Some(self.inner.dim()),
            ),
        );
        Ok(computer)
    }

    fn query_computer(&self, query: &[f32]) -> Result<GarnetQueryComputer, GarnetQuantizerError> {
        let computer = GarnetQueryComputer::new(MinMax8BitQueryComputer::new(
            &self.inner,
            query,
            self.inner.dim(),
            self.metric,
        )?);
        Ok(computer)
    }

    fn serialize(&self) -> Result<Poly<[u8], GlobalAllocator>, GarnetQuantizerError> {
        Err(GarnetQuantizerError::UnsupportedSerialization)
    }

    fn deserialize(&self, _state: &[u8]) -> Result<(), GarnetQuantizerError> {
        Err(GarnetQuantizerError::UnsupportedSerialization)
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

#[cfg(test)]
mod tests {
    use diskann_utils::Matrix;
    use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction, distance::Metric};

    use crate::quantization::{GarnetQuantizer, GarnetQuantizerError, MinMax8Bit, Spherical1Bit};

    #[test]
    fn basic_spherical_1bit() {
        let quantizer = Spherical1Bit::new(2);

        assert_eq!(quantizer.required_vectors(), 1000);
        assert_eq!(quantizer.bytes(), 1 + 6);
        assert!(!quantizer.is_trained());

        let test_v = [0.5f32, 0.5];
        let mut test_q = vec![0u8; quantizer.bytes()];

        assert!(matches!(
            quantizer.compress(&test_v, &mut test_q),
            Err(GarnetQuantizerError::NoQuantizer)
        ));
        assert!(matches!(
            quantizer.distance_computer(),
            Err(GarnetQuantizerError::NoQuantizer)
        ));
        assert!(matches!(
            quantizer.query_computer(&test_v),
            Err(GarnetQuantizerError::NoQuantizer)
        ));

        let mut test_data = Matrix::new(0.0f32, 1000, 2);
        for i in 0..1000 {
            test_data
                .row_mut(i)
                .copy_from_slice(&[(i + 1) as f32, (i + 1) as f32]);
        }
        quantizer.train(Metric::L2, test_data.as_view()).unwrap();

        assert!(quantizer.is_trained());

        quantizer.compress(&test_v, &mut test_q).unwrap();
        assert!(!test_q.iter().all(|&b| b == 0));

        let dist_comp = quantizer.distance_computer().unwrap();
        let full_a = [0.0f32, 0.0];
        let mut quant_a = vec![0u8; quantizer.bytes()];
        quantizer.compress(&full_a, &mut quant_a).unwrap();

        let d = dist_comp.evaluate_similarity(&quant_a, &test_q);
        assert_ne!(d, 0.0);

        let query_comp = quantizer.query_computer(&test_v).unwrap();
        let d = query_comp.evaluate_similarity(&quant_a);
        assert_ne!(d, 0.0);
    }

    #[test]
    fn basic_minmax_8bit() {
        let quantizer = MinMax8Bit::new(2, Metric::L2).unwrap();

        assert_eq!(quantizer.required_vectors(), 0);
        assert_eq!(quantizer.bytes(), 22);
        // MinMax8Bit starts trained
        assert!(quantizer.is_trained());

        let test_v = [0.5f32, 0.5];
        let mut test_q = vec![0u8; quantizer.bytes()];

        let mut test_data = Matrix::new(0.0f32, 1, 2);
        test_data.row_mut(0).copy_from_slice(&[1.0f32, 1.0]);

        // Training is a no-op, but succeeds.
        quantizer.train(Metric::L2, test_data.as_view()).unwrap();

        quantizer.compress(&test_v, &mut test_q).unwrap();
        assert!(!test_q.iter().all(|&b| b == 0));

        let dist_comp = quantizer.distance_computer().unwrap();
        let full_a = [0.0f32, 0.0];
        let mut quant_a = vec![0u8; quantizer.bytes()];
        quantizer.compress(&full_a, &mut quant_a).unwrap();

        let d = dist_comp.evaluate_similarity(&quant_a, &test_q);
        assert_ne!(d, 0.0);

        let query_comp = quantizer.query_computer(&test_v).unwrap();
        let d = query_comp.evaluate_similarity(&quant_a);
        assert_ne!(d, 0.0);
    }
}
