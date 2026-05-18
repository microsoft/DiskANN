/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Bf-Tree quant vector provider.

use crate::{AccessError, AsKey, VectorError, VectorUnavailable};
use bf_tree::{BfTree, Config};
use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_quantization::{
    alloc::{GlobalAllocator, Poly, ScopedAllocator},
    spherical::iface::{
        DistanceComputer, Opaque, OpaqueMut, Quantizer, QueryComputer, QueryLayout,
    },
};
use diskann_vector::PreprocessedDistanceFunction;

use super::ConfigError;
use crate::TestCallCount;

pub struct QuantQueryComputer(QueryComputer<GlobalAllocator>);

impl QuantQueryComputer {
    pub(crate) fn into_inner(self) -> QueryComputer<GlobalAllocator> {
        self.0
    }
}

impl PreprocessedDistanceFunction<&[u8], f32> for QuantQueryComputer {
    fn evaluate_similarity(&self, x: &[u8]) -> f32 {
        self.0
            .evaluate_similarity(Opaque::new(x))
            .expect("spherical query distance failed")
    }
}

pub struct QuantVectorProvider {
    quant_vector_index: BfTree,
    pub(crate) quantizer: Poly<dyn Quantizer>,
    pub(super) num_get_calls: TestCallCount,
}

impl QuantVectorProvider {
    pub fn new_with_config(quantizer: Poly<dyn Quantizer>, config: Config) -> ANNResult<Self> {
        let quant_vector_index = BfTree::with_config(config, None).map_err(ConfigError)?;

        Ok(Self {
            quant_vector_index,
            quantizer,
            num_get_calls: TestCallCount::default(),
        })
    }

    /// Access the BfTree config
    pub(crate) fn config(&self) -> &Config {
        self.quant_vector_index.config()
    }

    /// Access the underlying BfTree
    pub(crate) fn bftree(&self) -> &BfTree {
        &self.quant_vector_index
    }

    /// Create a new instance from an existing BfTree (for loading from snapshot)
    ///
    pub(crate) fn new_from_bftree(
        quantizer: Poly<dyn Quantizer>,
        quant_vector_index: BfTree,
    ) -> Self {
        Self {
            quant_vector_index,
            quantizer,
            num_get_calls: TestCallCount::default(),
        }
    }

    /// Return the dimension of the full-precision data associated with this provider
    pub fn full_dim(&self) -> usize {
        self.quantizer.full_dim()
    }

    /// Create a query computer for the provided query vector
    pub fn query_computer<T>(&self, query: &[T]) -> ANNResult<QuantQueryComputer>
    where
        T: VectorRepr,
    {
        let query_f32 = T::as_f32(query).into_ann_result()?;
        let inner = self
            .quantizer
            .fused_query_computer(
                &query_f32,
                QueryLayout::FullPrecision,
                true,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .map_err(|e| ANNError::log_sq_error(e))?;
        Ok(QuantQueryComputer(inner))
    }

    /// Create a distance computer for the underlying schema
    pub fn distance_computer(&self) -> ANNResult<DistanceComputer> {
        self.quantizer
            .distance_computer(GlobalAllocator)
            .map_err(|e| ANNError::log_sq_error(e))
    }

    pub(crate) fn get_vector_into(&self, i: usize, buffer: &mut [u8]) -> Result<(), AccessError> {
        use diskann::ANNErrorKind;
        use thiserror::Error;

        let expected = self.quantizer.bytes();
        if buffer.len() != expected {
            #[derive(Debug, Error)]
            #[error("expected a buffer with dim {0}, instead got {1}")]
            struct WrongDim(usize, usize);

            return Err(AccessError::Error(ANNError::new(
                ANNErrorKind::IndexError,
                WrongDim(expected, buffer.len()),
            )));
        }

        self.num_get_calls.increment();
        match self.quant_vector_index.read(i.as_key(), buffer) {
            bf_tree::LeafReadResult::Found(read_size) => {
                if read_size as usize != expected {
                    return Err(AccessError::Error(ANNError::log_index_error(format!(
                        "The bf-tree entry for vector id {} is marked as found but has size {} instead of the expected size {}",
                        i, read_size, expected,
                    ))));
                }
            }
            bf_tree::LeafReadResult::Deleted => {
                return Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::Deleted,
                }));
            }
            bf_tree::LeafReadResult::InvalidKey => {
                return Err(AccessError::Error(ANNError::log_index_error(format!(
                    "The bf-tree entry for vector id {} is marked as invalid",
                    i,
                ))));
            }
            bf_tree::LeafReadResult::NotFound => {
                return Err(AccessError::Transient(VectorUnavailable {
                    id: i,
                    err: VectorError::NotFound,
                }));
            }
        };

        Ok(())
    }

    /// Return the quant vector at index `i`.
    pub(crate) fn get_vector_sync(&self, i: usize) -> Result<Vec<u8>, AccessError> {
        let mut value = vec![0u8; self.quantizer.bytes()];
        self.get_vector_into(i, &mut value)?;
        Ok(value)
    }

    /// Compress the vector, `v`, and set the compressed quant vector with Id, `i`, to it
    ///
    /// Errors if:
    ///
    /// * `v.dim() != self.full_dim()`: The slice must have the proper length.
    /// * PQ compression encounters an error (such as the presence of `NaN`s).
    pub(crate) fn set_vector_sync<T>(&self, i: usize, v: &[T]) -> ANNResult<()>
    where
        T: Copy + VectorRepr,
    {
        let vf32: &[f32] = &T::as_f32(v).into_ann_result()?;

        if vf32.len() != self.full_dim() {
            return Err(ANNError::log_dimension_mismatch_error(
                "Vector f32 dimension is not equal to the expected dimension.".to_string(),
            ));
        }

        // Serialize the key into a byte string, &[u8]
        let key = i.as_key();

        let dim = self.quantizer.bytes();
        let quant_vector = &mut vec![0u8; dim];
        self.quantizer
            .compress(
                vf32,
                OpaqueMut::new(quant_vector),
                ScopedAllocator::global(),
            )
            .map_err(|e| ANNError::log_sq_error(e))?;

        self.quant_vector_index.insert(key, quant_vector);

        Ok(())
    }

    /// Set the quant vector with Id, `i`, to `v`
    ///
    /// Errors if:
    ///
    /// * `v.len() != self.pq_chunks()`: `v` must have the right length.
    #[cfg(test)]
    pub(crate) fn set_quant_vector(&self, i: usize, v: &[u8]) -> ANNResult<()> {
        if v.len() != self.quantizer.bytes() {
            return Err(ANNError::log_index_error(
                "Vector dimension is not equal to the expected dimension.",
            ));
        }

        // Update pq vector with id = i to v
        let key = i.as_key();

        self.quant_vector_index.insert(key, v);

        Ok(())
    }

    pub(crate) fn delete_vector(&self, i: usize) {
        let key = i.as_key();
        self.quant_vector_index.delete(key);
    }
}

/// Train a spherical quantizer on simple data and return it as a `Poly<dyn Quantizer>`.
#[cfg(test)]
pub(crate) fn create_test_quantizer(dim: usize) -> Poly<dyn Quantizer> {
    use diskann_quantization::{
        algorithms::TransformKind,
        alloc::poly,
        spherical::{iface, PreScale, SphericalQuantizer, SupportedMetric},
    };
    use diskann_utils::views::Init;
    use diskann_utils::views::Matrix;
    use rand::{rngs::StdRng, SeedableRng};

    // Create training data with spread-out values.
    let nrows = 8;
    let mut counter = 0.0f32;
    let data = Matrix::new(
        Init(move || {
            counter += 0.5;
            counter
        }),
        nrows,
        dim,
    );

    let mut rng = StdRng::seed_from_u64(42);
    let quantizer = SphericalQuantizer::train(
        data.as_view(),
        TransformKind::Null,
        SupportedMetric::SquaredL2,
        PreScale::None,
        &mut rng,
        GlobalAllocator,
    )
    .unwrap();

    let imp = iface::Impl::<1>::new(quantizer).unwrap();
    poly!(Quantizer, imp, GlobalAllocator).unwrap()
}

///////////
// Tests //
///////////
/// These unit tests target the functionality of Bf-Tree quant vector provider alone
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use diskann::ANNErrorKind;
    use diskann_quantization::spherical::iface::Opaque;
    use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};
    use tokio::task::JoinSet;

    use super::*;

    /// Test edge cases of the Bf-Tree quant vector provider
    #[tokio::test]
    async fn common_errors() {
        let dim = 5;
        let quantizer = create_test_quantizer(dim);
        let quant_bytes = quantizer.bytes();

        let bf_tree_config = Config::default();
        let provider = QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap();

        // try to set an out of bounds vector
        let result = provider.set_quant_vector(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // try to set an out of bounds vector via set_vector_sync
        let result = provider.set_vector_sync::<f32>(20, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::DimensionMismatchError);

        // try to set a quant vector with the wrong dimension
        let result = provider.set_quant_vector(0, &[]).unwrap_err();
        assert_eq!(result.kind(), ANNErrorKind::IndexError);

        // verify expected quant vector byte count
        assert_eq!(quant_bytes, provider.quantizer.bytes());
    }

    fn create_test_provider() -> QuantVectorProvider {
        let dim = 2;

        let quantizer = create_test_quantizer(dim);

        let bf_tree_config = Config::default();
        let provider = QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap();

        assert_eq!(provider.full_dim(), dim);

        // Set vectors.
        provider.set_vector_sync(0, &[-1.5, -1.5]).unwrap();
        provider.set_vector_sync(1, &[-0.5, -0.5]).unwrap();
        provider.set_vector_sync(2, &[0.5, 0.5]).unwrap();
        provider.set_vector_sync(3, &[1.5, 1.5]).unwrap();
        provider.set_vector_sync(4, &[2.5, 2.5]).unwrap();
        provider
    }

    /// Test the distance computation functions of the provider
    #[tokio::test]
    async fn test_similarity_function() {
        let provider = create_test_provider();
        let quant_bytes = provider.quantizer.bytes();

        // Verify compressed vectors are the expected size.
        for i in 0..5 {
            let v = provider.get_vector_sync(i).unwrap();
            assert_eq!(v.len(), quant_bytes);
        }

        // Error checking.
        assert!(provider.set_vector_sync(2, &[0.0]).is_err());

        // Query Computer — verify it returns finite distances.
        let c = provider.query_computer(&[-0.5f32, -0.5]).unwrap();
        let dist = c.evaluate_similarity(&provider.get_vector_sync(3).unwrap());
        assert!(dist.is_finite(), "query distance should be finite");

        // Distance Computer — verify distances between compressed vectors are finite
        // and that identical vectors produce zero distance.
        let d = provider.distance_computer().unwrap();
        let v0 = provider.get_vector_sync(0).unwrap();
        let v3 = provider.get_vector_sync(3).unwrap();
        let dist = d
            .evaluate_similarity(Opaque::new(&v0), Opaque::new(&v3))
            .unwrap();
        assert!(dist.is_finite(), "distance should be finite");

        // Same vector should have small self-distance (may not be exactly zero
        // due to quantization loss, especially at low bit-widths).
        let self_dist = d
            .evaluate_similarity(Opaque::new(&v0), Opaque::new(&v0))
            .unwrap();
        assert!(
            self_dist.abs() < 1.0,
            "self-distance should be small, got {}",
            self_dist
        );
    }

    /// Test the interleaved and parallel traversal of the Bf-Tree
    /// by invoking the async accessors of the quant vector provider
    #[tokio::test(flavor = "multi_thread", worker_threads = 5)]
    async fn test_parallel_tree_traversal() {
        let dim = 2;
        let quantizer = create_test_quantizer(dim);

        let bf_tree_config = Config::default();
        let provider =
            Arc::new(QuantVectorProvider::new_with_config(quantizer, bf_tree_config).unwrap());
        let mut set = JoinSet::new();
        for i in 0..11 {
            let vector = vec![i as f32, (i + 1) as f32];
            let provider_clone = Arc::clone(&provider);
            set.spawn(async move { provider_clone.set_vector_sync(i as usize, &vector).unwrap() });
        }

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        // Verify that each vector was stored and can be retrieved with the correct size.
        let quant_bytes = provider.quantizer.bytes();
        let mut expected_buf = vec![0u8; quant_bytes];

        for i in 0..11 {
            let stored = provider.get_vector_sync(i).unwrap();
            assert_eq!(stored.len(), quant_bytes);

            // Compress the same input again and verify we get the same output
            // (spherical compression is deterministic).
            provider
                .quantizer
                .compress(
                    &[i as f32, (i + 1) as f32],
                    OpaqueMut::new(&mut expected_buf),
                    ScopedAllocator::global(),
                )
                .unwrap();
            assert_eq!(stored, expected_buf);
        }
    }
}
