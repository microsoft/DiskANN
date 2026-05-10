/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Product Quantizer for learned codebook-based vector compression.
//!
//! Product quantization partitions vectors into chunks and learns a codebook
//! for each chunk via k-means clustering. This is particularly effective for
//! high-dimensional vectors, achieving high compression while preserving
//! distance relationships.
//!
//! # Example
//!
//! ```python
//! from diskannpy import ProductQuantizer
//! import numpy as np
//!
//! # Training data (should be representative of your dataset)
//! training_data = np.random.randn(10000, 128).astype(np.float32)
//!
//! # Train the quantizer
//! quantizer = ProductQuantizer(
//!     training=training_data,
//!     num_chunks=16,       # Number of sub-vector partitions
//!     num_centers=256,     # Codebook size per chunk (max 256)
//!     lloyds_iters=10,     # K-means iterations
//!     seed=42              # For reproducibility
//! )
//!
//! # Compress vectors
//! vectors = np.random.randn(1000, 128).astype(np.float32)
//! compressed = quantizer.compress_batch(vectors)  # (1000, num_chunks) uint8
//!
//! # Search: preprocess query with metric, then compute distances
//! query = np.random.randn(1, 128).astype(np.float32)
//! preprocessed = quantizer.preprocess(query, metric="l2")
//! distances = quantizer.distances_batch(preprocessed, compressed)
//!
//! # Find k nearest neighbors
//! nearest_indices = np.argsort(distances)[:10]
//! ```

use std::{borrow::Cow, sync::Arc};

use numpy::{
    ndarray::{ArrayView1, ArrayViewMut1},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use diskann_providers::model::pq::{
    calculate_chunk_offsets_auto, distance::QueryComputer, FixedChunkPQTable,
};
use diskann_quantization::{
    cancel::DontCancel,
    product::train::{LightPQTrainingParameters, TrainQuantizer},
    random::StdRngBuilder,
    views::ChunkOffsetsView,
    CompressInto, Parallelism,
};
use diskann_utils::views::Matrix;
use diskann_vector::PreprocessedDistanceFunction;

use crate::quantization::{QuantizationMetric, QuantizerBase};

#[pyclass]
pub struct ProductPreprocessedQuery {
    metric: QuantizationMetric,
    num_chunks: usize,
    computer: QueryComputer<Arc<FixedChunkPQTable>>,
}

#[pymethods]
impl ProductPreprocessedQuery {
    #[getter]
    pub fn metric(&self) -> String {
        self.metric.to_string()
    }

    #[getter]
    pub fn len(&self) -> usize {
        self.num_chunks
    }
}

/// Product quantizer that learns codebooks via k-means clustering.
///
/// Partitions vectors into `num_chunks` sub-vectors and learns a codebook
/// of `num_centers` centroids for each chunk. Each vector is then encoded
/// as `num_chunks` bytes (one centroid index per chunk).
///
/// # Properties
///
/// - `num_chunks`: Number of sub-vector partitions
/// - `num_centers`: Codebook size per chunk (max 256)
/// - `bytes_per_vector`: Size of compressed vector (equals num_chunks)
/// - `dim`: Input vector dimension
#[pyclass(extends = QuantizerBase)]
pub struct ProductQuantizer {
    table: Arc<FixedChunkPQTable>,
}

#[pymethods]
impl ProductQuantizer {
    /// Train and create a new ProductQuantizer.
    ///
    /// Training uses k-means clustering to learn codebooks from representative data.
    ///
    /// # Arguments
    ///
    /// * `training` - Training vectors as a 2D float32 array of shape (N, dim)
    /// * `num_chunks` - Number of sub-vector partitions (must be > 0 and <= dim)
    /// * `num_centers` - Codebook size per chunk (1 to 256, typically 256)
    /// * `lloyds_iters` - Number of k-means iterations (default: 5)
    /// * `seed` - Random seed for reproducibility (default: 7)
    /// * `parallel` - Parallelism: "rayon" (default) or "sequential"
    ///
    /// # Example
    ///
    /// ```python
    /// from diskannpy import ProductQuantizer
    /// import numpy as np
    ///
    /// training_data = np.random.randn(10000, 128).astype(np.float32)
    /// quantizer = ProductQuantizer(
    ///     training=training_data,
    ///     num_chunks=16,
    ///     num_centers=256,
    ///     lloyds_iters=10,
    ///     seed=42
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (training, num_chunks, num_centers, lloyds_iters=5, seed=7, sequential_parallelism=false))]
    pub fn new(
        training: PyReadonlyArray2<'_, f32>,
        num_chunks: usize,
        num_centers: usize,
        lloyds_iters: usize,
        seed: u64,
        sequential_parallelism: bool,
    ) -> PyResult<(Self, QuantizerBase)> {
        let training_view = training.as_array();
        let (rows, cols) = training_view.dim();

        if num_chunks == 0 || num_chunks > cols {
            return Err(PyValueError::new_err(format!(
                "num_chunks must be > 0 and < {cols}"
            )));
        }
        if num_centers == 0 || num_centers > 256 {
            return Err(PyValueError::new_err(
                "num_centers must be in the range [1, 256]",
            ));
        }

        if cols == 0 {
            return Err(PyValueError::new_err(
                "training data must have at least one dimension",
            ));
        }
        if rows == 0 {
            return Err(PyValueError::new_err(
                "training data must contain at least one sample",
            ));
        }

        let data: Vec<f32> = training_view.iter().copied().collect();
        let matrix = Matrix::try_from(data.into(), rows, cols)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let offsets = calculate_chunk_offsets_auto(cols, num_chunks);
        let offsets_view = ChunkOffsetsView::new(offsets.as_slice())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let parallelism = if sequential_parallelism {
            Parallelism::Sequential
        } else {
            Parallelism::Rayon
        };

        let trainer = LightPQTrainingParameters::new(num_centers, lloyds_iters);
        let pivots = trainer
            .train(
                matrix.as_view(),
                offsets_view,
                parallelism,
                &StdRngBuilder::new(seed),
                &DontCancel,
            )
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let table = FixedChunkPQTable::new(
            cols,
            pivots.flatten().into(),
            vec![0.0f32; cols].into(),
            offsets.clone().into(),
            None,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let base = QuantizerBase::new(cols, cols, 8, "product");
        Ok((
            Self {
                table: Arc::new(table),
            },
            base,
        ))
    }

    /// Number of sub-vector partitions.
    #[getter]
    pub fn num_chunks(&self) -> usize {
        self.table.get_num_chunks()
    }

    /// Codebook size per chunk (number of centroids).
    #[getter]
    pub fn num_centers(&self) -> usize {
        self.table.get_num_centers()
    }

    /// Size of a compressed vector in bytes (equals num_chunks).
    #[getter]
    pub fn bytes_per_vector(&self) -> usize {
        self.table.get_num_chunks()
    }

    /// Compress a batch of vectors into quantized byte representations.
    ///
    /// Each vector is encoded as `num_chunks` bytes, where each byte is
    /// the index of the nearest centroid in that chunk's codebook.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Input vectors as a 2D float32 array of shape (N, dim)
    ///
    /// # Returns
    ///
    /// A 2D uint8 numpy array of shape (N, num_chunks) containing the
    /// compressed representations.
    ///
    /// # Example
    ///
    /// ```python
    /// vectors = np.random.randn(1000, 128).astype(np.float32)
    /// compressed = quantizer.compress_batch(vectors)
    /// print(compressed.shape)  # (1000, num_chunks)
    /// print(compressed.dtype)  # uint8
    /// ```
    pub fn compress_batch<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let array = vectors.as_array();
        if array.ncols() != self.table.get_dim() {
            return Err(PyValueError::new_err(format!(
                "expected input dimension {} but received {}",
                self.table.get_dim(),
                array.ncols()
            )));
        }

        let nrows = array.nrows();
        let bytes_per_vec = self.bytes_per_vector();
        let out = PyArray2::<u8>::zeros(py, [nrows, bytes_per_vec], false);

        // Safety: we have exclusive access to the output array we just created
        unsafe {
            let mut out_array = out.as_array_mut();
            for (i, row) in array.rows().into_iter().enumerate() {
                let mut out_row = out_array.row_mut(i);
                self.compress_into_slice(row, &mut out_row)?;
            }
        }

        Ok(out)
    }

    /// Preprocess a query vector for distance computation.
    ///
    /// Unlike MinMaxQuantizer, the metric must be specified during preprocessing
    /// because it affects how the distance lookup tables are computed.
    ///
    /// # Arguments
    ///
    /// * `vector` - Query vector as a 2D float32 array of shape (1, dim)
    /// * `metric` - Distance metric: "l2", "inner_product", "cosine", or "cosine_normalized"
    ///
    /// # Returns
    ///
    /// A `ProductPreprocessedQuery` object to pass to `distances_batch`.
    ///
    /// # Example
    ///
    /// ```python
    /// query = np.random.randn(1, 128).astype(np.float32)
    /// preprocessed = quantizer.preprocess(query, metric="l2")
    /// ```
    pub fn preprocess(
        &self,
        vector: PyReadonlyArray2<'_, f32>,
        metric: &str,
    ) -> PyResult<ProductPreprocessedQuery> {
        let array = vector.as_array();
        if array.nrows() != 1 {
            return Err(PyValueError::new_err("expected a (1, dim) shaped array"));
        }
        let row = array.row(0);
        if row.len() != self.table.get_dim() {
            return Err(PyValueError::new_err(format!(
                "expected input dimension {} but received {}",
                self.table.get_dim(),
                row.len()
            )));
        }

        let metric = QuantizationMetric::parse(metric)?;
        let query_values = match row.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(row.to_vec()),
        };

        let computer = QueryComputer::new(
            self.table.clone(),
            metric.to_vector_metric(),
            query_values.as_ref(),
            None,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(ProductPreprocessedQuery {
            metric,
            num_chunks: self.table.get_num_chunks(),
            computer,
        })
    }

    /// Decompress a single vector back to float32 (approximate reconstruction).
    ///
    /// Reconstructs the vector by looking up and concatenating centroids.
    /// Useful for debugging. The reconstructed vector is an approximation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Compressed vector as a list of bytes (length = num_chunks)
    ///
    /// # Example
    ///
    /// ```python
    /// first_compressed = list(compressed[0])
    /// reconstructed = quantizer.decompress(first_compressed)
    /// ```
    pub fn decompress(&self, vector: Vec<u8>) -> PyResult<Vec<f32>> {
        if vector.len() != self.table.get_num_chunks() {
            return Err(PyValueError::new_err(
                "compressed vector length does not match quantizer",
            ));
        }
        Ok(self.table.inflate_vector(&vector))
    }

    /// Compute distances between a preprocessed query and compressed vectors.
    ///
    /// Uses precomputed lookup tables for efficient distance computation.
    ///
    /// # Arguments
    ///
    /// * `preprocessed` - Preprocessed query from `preprocess()`
    /// * `compressed` - Compressed vectors as a 2D uint8 array of shape (N, num_chunks)
    ///
    /// # Returns
    ///
    /// A 1D float32 numpy array of shape (N,) containing distances.
    ///
    /// # Example
    ///
    /// ```python
    /// # Preprocess query with metric
    /// query = np.random.randn(1, 128).astype(np.float32)
    /// preprocessed = quantizer.preprocess(query, metric="l2")
    ///
    /// # Compute distances to all compressed vectors
    /// distances = quantizer.distances_batch(preprocessed, compressed)
    ///
    /// # Find k nearest neighbors
    /// k = 10
    /// nearest_indices = np.argsort(distances)[:k]
    /// ```
    pub fn distances_batch<'py>(
        &self,
        py: Python<'py>,
        preprocessed: &ProductPreprocessedQuery,
        compressed: PyReadonlyArray2<'_, u8>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let array = compressed.as_array();
        let expected_cols = self.bytes_per_vector();

        if array.ncols() != expected_cols {
            return Err(PyValueError::new_err(format!(
                "expected {} bytes per vector but received {}",
                expected_cols,
                array.ncols()
            )));
        }

        let nrows = array.nrows();
        let out = PyArray1::<f32>::zeros(py, nrows, false);

        // Safety: we have exclusive access to the output array we just created
        unsafe {
            let mut out_array = out.as_array_mut();
            for (i, row) in array.rows().into_iter().enumerate() {
                let slice = row.as_slice().ok_or_else(|| {
                    PyValueError::new_err("compressed array must be C-contiguous")
                })?;
                out_array[i] = preprocessed.computer.evaluate_similarity(slice);
            }
        }

        Ok(out)
    }
}

impl ProductQuantizer {
    fn compress_into_slice(
        &self,
        row: ArrayView1<'_, f32>,
        out: &mut ArrayViewMut1<'_, u8>,
    ) -> PyResult<()> {
        let values = match row.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(row.to_vec()),
        };
        let out_slice = out
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("output array must be C-contiguous"))?;
        self.table
            .compress_into(values.as_ref(), out_slice)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }
}
