/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! MinMax Quantizer for scalar quantization of high-dimensional vectors.
//!
//! MinMax quantization maps each dimension to a fixed grid, achieving compression
//! ratios of 4x to 32x depending on the bit width (1, 2, 4, or 8 bits per dimension).
//!
//! # Example
//!
//! ```python
//! from diskannpy import MinMaxQuantizer
//! import numpy as np
//!
//! # Create quantizer: bit_width, grid_scale, dimension
//! quantizer = MinMaxQuantizer(bit_width=8, grid_scale=1.0, dim=128)
//!
//! # Compress vectors
//! vectors = np.random.randn(1000, 128).astype(np.float32)
//! compressed = quantizer.compress_batch(vectors)  # (1000, bytes_per_vector) uint8
//!
//! # Search: preprocess query, then compute distances
//! query = np.random.randn(1, 128).astype(np.float32)
//! preprocessed = quantizer.preprocess(query)
//! distances = quantizer.distances_batch(preprocessed, compressed, metric="l2")
//!
//! # Find k nearest neighbors
//! nearest_indices = np.argsort(distances)[:10]
//! ```
//!
//! # Optional Transforms
//!
//! Transforms can improve quantization quality by spreading information across dimensions:
//!
//! ```python
//! quantizer = MinMaxQuantizer(
//!     bit_width=4,
//!     grid_scale=1.0,
//!     dim=128,
//!     transform="double_hadamard",  # "null", "padding_hadamard", or "double_hadamard"
//!     target_behavior="natural",    # "same" or "natural"
//!     rng_seed=42                   # for reproducibility
//! )
//! ```

use std::{borrow::Cow, num::NonZeroUsize};

use numpy::{
    ndarray::{ArrayView1, ArrayViewMut1},
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2,
};
use pyo3::{exceptions::PyValueError, prelude::*};

use diskann_utils::{Reborrow, ReborrowMut};

use diskann_quantization::{
    algorithms::transforms::{NullTransform, TargetDim, Transform, TransformKind},
    alloc::GlobalAllocator,
    bits::{BitSlice, Representation, Unsigned},
    distances::{InnerProduct, MathematicalResult, Result as DistanceResult},
    meta::NotCanonical,
    minmax::FullQuery,
    minmax::{
        DataMutRef, DataRef, DecompressError, MinMaxCosine, MinMaxCosineNormalized, MinMaxIP,
        MinMaxL2Squared, MinMaxQuantizer as InnerMinMaxQuantizer,
    },
    num::Positive,
    CompressInto,
};
use diskann_vector::PureDistanceFunction;
use rand::{rngs::StdRng, SeedableRng};

use crate::quantization::{QuantizationMetric, QuantizerBase};

#[pyclass]
pub struct MinMaxPreprocessedQuery {
    query: FullQuery,
}

#[pymethods]
impl MinMaxPreprocessedQuery {
    #[getter]
    pub fn dim(&self) -> usize {
        self.query.len()
    }
}

fn parse_transform(
    transform: Option<&str>,
    target_behavior: Option<&str>,
    target_override: Option<usize>,
    dim: NonZeroUsize,
    rng_seed: Option<u64>,
) -> PyResult<Transform<GlobalAllocator>> {
    let normalized = transform.and_then(|s| {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_ascii_lowercase())
        }
    });

    let transform = match normalized.as_deref() {
        None | Some("null") => {
            return Ok(Transform::Null(NullTransform::new(dim)));
        }
        Some(name) => name,
    };

    let target_dim = if let Some(override_dim) = target_override {
        let nz = NonZeroUsize::new(override_dim)
            .ok_or_else(|| PyValueError::new_err("target_override must be greater than zero"))?;
        TargetDim::Override(nz)
    } else {
        match target_behavior
            .unwrap_or("same")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "same" => TargetDim::Same,
            "natural" | "" => TargetDim::Natural,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unsupported target_behavior '{other}'. expected 'same' or 'natural'"
                )));
            }
        }
    };

    let kind = match transform {
        "padding_hadamard" => TransformKind::PaddingHadamard { target_dim },
        "double_hadamard" => TransformKind::DoubleHadamard { target_dim },
        other => {
            return Err(PyValueError::new_err(format!(
                "unsupported transform '{other}'. expected one of 'null', 'padding_hadamard', 'double_hadamard'"
            )));
        }
    };

    let mut rng = StdRng::seed_from_u64(rng_seed.unwrap_or(42));
    Transform::new(kind, dim, Some(&mut rng), GlobalAllocator)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Scalar quantizer that maps each dimension to a fixed grid.
///
/// Supports bit widths of 1, 2, 4, or 8 bits per dimension, achieving
/// compression ratios of 32x, 16x, 8x, and 4x respectively.
///
/// # Properties
///
/// - `bit_width`: Number of bits per dimension (1, 2, 4, or 8)
/// - `bytes_per_vector`: Size of compressed vector in bytes
/// - `dim`: Input vector dimension
/// - `output_dim`: Output dimension (may differ if transform is applied)
#[pyclass(extends = QuantizerBase)]
pub struct MinMaxQuantizer {
    inner: InnerMinMaxQuantizer,
    bit_width: usize,
    output_dim: usize,
}

#[pymethods]
impl MinMaxQuantizer {
    /// Create a new MinMaxQuantizer.
    ///
    /// # Arguments
    ///
    /// * `bit_width` - Bits per dimension: 1, 2, 4, or 8
    /// * `grid_scale` - Scale factor for the quantization grid (must be positive)
    /// * `dim` - Input vector dimension
    /// * `transform` - Optional transform: "null", "padding_hadamard", or "double_hadamard"
    /// * `target_behavior` - Transform target: "same" or "natural" (default)
    /// * `target_override` - Override output dimension (optional)
    /// * `rng_seed` - Random seed for transform initialization (default: 42)
    ///
    /// # Example
    ///
    /// ```python
    /// from diskannpy import MinMaxQuantizer
    ///
    /// # Basic usage
    /// quantizer = MinMaxQuantizer(bit_width=8, grid_scale=1.0, dim=128)
    ///
    /// # With transform for better quantization
    /// quantizer = MinMaxQuantizer(
    ///     bit_width=4,
    ///     grid_scale=1.0,
    ///     dim=128,
    ///     transform="double_hadamard"
    /// )
    /// ```
    #[new]
    #[pyo3(signature = (bit_width, grid_scale, dim, *, transform=None, target_behavior=None, target_override=None, rng_seed=None))]
    pub fn new(
        bit_width: usize,
        grid_scale: f32,
        dim: usize,
        transform: Option<&str>,
        target_behavior: Option<&str>,
        target_override: Option<usize>,
        rng_seed: Option<u64>,
    ) -> PyResult<(Self, QuantizerBase)> {
        if !matches!(bit_width, 1 | 2 | 4 | 8) {
            return Err(PyValueError::new_err(
                "minmax quantizer supports bit widths 1, 2, 4, or 8",
            ));
        }
        let dim = NonZeroUsize::new(dim)
            .ok_or_else(|| PyValueError::new_err("dim must be greater than zero"))?;
        let transform =
            parse_transform(transform, target_behavior, target_override, dim, rng_seed)?;
        let scale = Positive::new(grid_scale)
            .map_err(|_| PyValueError::new_err("grid_scale must be positive"))?;
        let quantizer = InnerMinMaxQuantizer::new(transform, scale);
        let output_dim = quantizer.output_dim();
        let base = QuantizerBase::new(quantizer.dim(), output_dim, bit_width, "minmax");
        Ok((
            Self {
                inner: quantizer,
                bit_width,
                output_dim,
            },
            base,
        ))
    }

    /// Number of bits used per dimension (1, 2, 4, or 8).
    #[getter]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Size of a compressed vector in bytes.
    #[getter]
    pub fn bytes_per_vector(&self) -> usize {
        match self.bit_width {
            1 => DataRef::<1>::canonical_bytes(self.output_dim),
            2 => DataRef::<2>::canonical_bytes(self.output_dim),
            4 => DataRef::<4>::canonical_bytes(self.output_dim),
            8 => DataRef::<8>::canonical_bytes(self.output_dim),
            _ => unreachable!(),
        }
    }

    /// Compress a batch of vectors into quantized byte representations.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Input vectors as a 2D float32 array of shape (N, dim)
    ///
    /// # Returns
    ///
    /// A 2D uint8 numpy array of shape (N, bytes_per_vector) containing the
    /// compressed representations.
    ///
    /// # Example
    ///
    /// ```python
    /// import numpy as np
    /// vectors = np.random.randn(1000, 128).astype(np.float32)
    /// compressed = quantizer.compress_batch(vectors)
    /// print(compressed.shape)  # (1000, bytes_per_vector)
    /// print(compressed.dtype)  # uint8
    /// ```
    pub fn compress_batch<'py>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        match self.bit_width() {
            1 => self.compress_batch_::<1>(py, vectors),
            2 => self.compress_batch_::<2>(py, vectors),
            4 => self.compress_batch_::<4>(py, vectors),
            8 => self.compress_batch_::<8>(py, vectors),
            _ => unreachable!(),
        }
    }

    /// Preprocess a query vector for distance computation.
    ///
    /// The preprocessed query can be reused to compute distances against
    /// multiple compressed vectors efficiently.
    ///
    /// # Arguments
    ///
    /// * `vector` - Query vector as a 2D float32 array of shape (1, dim)
    ///
    /// # Returns
    ///
    /// A `MinMaxPreprocessedQuery` object to pass to `distances_batch`.
    ///
    /// # Example
    ///
    /// ```python
    /// query = np.random.randn(1, 128).astype(np.float32)
    /// preprocessed = quantizer.preprocess(query)
    /// ```
    pub fn preprocess(
        &self,
        vector: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<MinMaxPreprocessedQuery> {
        let array = vector.as_array();
        if array.nrows() != 1 {
            return Err(PyValueError::new_err("expected a (1, dim) shaped array"));
        }
        let row = array.row(0);
        if row.len() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected input dimension {} but received {}",
                self.inner.dim(),
                row.len()
            )));
        }

        let row = row
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("query array must be C-contiguous"))?;

        let mut query = FullQuery::new_in(self.output_dim, GlobalAllocator)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.inner
            .compress_into(row, query.reborrow_mut())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        Ok(MinMaxPreprocessedQuery { query })
    }

    /// Compute distances between a preprocessed query and compressed vectors.
    ///
    /// # Arguments
    ///
    /// * `query` - Preprocessed query from `preprocess()`
    /// * `compressed` - Compressed vectors as a 2D uint8 array of shape (N, bytes_per_vector)
    /// * `metric` - Distance metric: "l2", "inner_product", "cosine", or "cosine_normalized"
    ///
    /// # Returns
    ///
    /// A 1D float32 numpy array of shape (N,) containing distances.
    ///
    /// # Example
    ///
    /// ```python
    /// # Preprocess query
    /// query = np.random.randn(1, 128).astype(np.float32)
    /// preprocessed = quantizer.preprocess(query)
    ///
    /// # Compute distances to all compressed vectors
    /// distances = quantizer.distances_batch(preprocessed, compressed, metric="l2")
    ///
    /// # Find k nearest neighbors
    /// k = 10
    /// nearest_indices = np.argsort(distances)[:k]
    /// ```
    pub fn distances_batch<'py>(
        &self,
        py: Python<'py>,
        query: &MinMaxPreprocessedQuery,
        compressed: PyReadonlyArray2<'_, u8>,
        metric: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        match self.bit_width() {
            1 => self.distances_batch_::<1>(py, query, compressed, metric),
            2 => self.distances_batch_::<2>(py, query, compressed, metric),
            4 => self.distances_batch_::<4>(py, query, compressed, metric),
            8 => self.distances_batch_::<8>(py, query, compressed, metric),
            _ => unreachable!(),
        }
    }

    /// Decompress a single vector back to float32 (approximate reconstruction).
    ///
    /// Useful for debugging. The reconstructed vector is an approximation.
    ///
    /// # Arguments
    ///
    /// * `compressed` - Compressed vector as a list of bytes
    ///
    /// # Example
    ///
    /// ```python
    /// first_compressed = list(compressed[0])
    /// reconstructed = quantizer.decompress(first_compressed)
    /// ```
    pub fn decompress(&self, compressed: Vec<u8>) -> PyResult<Vec<f32>> {
        let expected = self.bytes_per_vector();
        if compressed.len() != expected {
            return Err(PyValueError::new_err(format!(
                "compressed vector length {} does not match expected {}",
                compressed.len(),
                expected
            )));
        }
        match self.bit_width {
            1 => decompress_single::<1>(&compressed, self.output_dim),
            2 => decompress_single::<2>(&compressed, self.output_dim),
            4 => decompress_single::<4>(&compressed, self.output_dim),
            8 => decompress_single::<8>(&compressed, self.output_dim),
            _ => unreachable!(),
        }
    }
}

impl MinMaxQuantizer {
    fn compress_batch_<'py, const NBITS: usize>(
        &self,
        py: Python<'py>,
        vectors: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>>
    where
        Unsigned: Representation<NBITS>,
    {
        let array = vectors.as_array();
        if array.ncols() != self.inner.dim() {
            return Err(PyValueError::new_err(format!(
                "expected input dimension {} but received {}",
                self.inner.dim(),
                array.ncols()
            )));
        }

        let nrows = array.nrows();
        let out = PyArray2::<u8>::zeros(py, [nrows, self.bytes_per_vector()], false);

        // Safety: we have exclusive access to the output array we just created
        unsafe {
            let mut out_array = out.as_array_mut();
            for (i, row) in array.rows().into_iter().enumerate() {
                self.compress_slice_::<NBITS>(row, &mut out_array.row_mut(i))?
            }
        }

        Ok(out)
    }

    fn compress_slice_<const NBITS: usize>(
        &self,
        row: ArrayView1<'_, f32>,
        out: &mut ArrayViewMut1<'_, u8>,
    ) -> PyResult<()>
    where
        Unsigned: Representation<NBITS>,
    {
        let values = match row.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(row.to_vec()),
        };
        let out_slice = out
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("output array must be C-contiguous"))?;
        let view = DataMutRef::<NBITS>::from_canonical_front_mut(out_slice, self.output_dim)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.inner
            .compress_into(values.as_ref(), view)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    fn distances_batch_<'py, const NBITS: usize>(
        &self,
        py: Python<'py>,
        query: &MinMaxPreprocessedQuery,
        compressed: PyReadonlyArray2<'_, u8>,
        metric: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>>
    where
        Unsigned: Representation<NBITS>,
        InnerProduct: for<'lhs, 'rhs> PureDistanceFunction<
            &'lhs [f32],
            BitSlice<'rhs, NBITS, Unsigned>,
            MathematicalResult<f32>,
        >,
    {
        let array = compressed.as_array();

        if array.ncols() != self.bytes_per_vector() {
            return Err(PyValueError::new_err(format!(
                "expected {} bytes per vector but received {}",
                self.bytes_per_vector(),
                array.ncols()
            )));
        }

        let metric = QuantizationMetric::parse(metric)?;
        let out = PyArray1::<f32>::zeros(py, array.nrows(), false);

        // Safety: we have exclusive access to the output array we just created
        unsafe {
            let mut out_array = out.as_array_mut();
            for (i, row) in array.rows().into_iter().enumerate() {
                let slice = row.as_slice().ok_or_else(|| {
                    PyValueError::new_err("compressed array must be C-contiguous")
                })?;

                out_array[i] = self.distance_single_::<NBITS>(&query.query, slice, metric)?;
            }
        }

        Ok(out)
    }

    fn distance_single_<const NBITS: usize>(
        &self,
        query: &FullQuery,
        storage: &[u8],
        metric: QuantizationMetric,
    ) -> PyResult<f32>
    where
        Unsigned: Representation<NBITS>,
        InnerProduct: for<'lhs, 'rhs> PureDistanceFunction<
            &'lhs [f32],
            BitSlice<'rhs, NBITS, Unsigned>,
            MathematicalResult<f32>,
        >,
    {
        let rhs = data_ref::<NBITS>(storage, self.output_dim)?;
        let rhs_view = rhs.reborrow();
        let query_ref = query.reborrow();
        let distance: DistanceResult<f32> = match metric {
            QuantizationMetric::L2 => MinMaxL2Squared::evaluate(query_ref, rhs_view),
            QuantizationMetric::InnerProduct => MinMaxIP::evaluate(query_ref, rhs_view),
            QuantizationMetric::Cosine => MinMaxCosine::evaluate(query_ref, rhs_view),
            QuantizationMetric::CosineNormalized => {
                MinMaxCosineNormalized::evaluate(query_ref, rhs_view)
            }
        };
        distance.map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

fn data_ref<const NBITS: usize>(storage: &[u8], dim: usize) -> PyResult<DataRef<'_, NBITS>>
where
    Unsigned: Representation<NBITS>,
{
    DataRef::from_canonical_front(storage, dim)
        .map_err(|err: NotCanonical| PyValueError::new_err(err.to_string()))
}

fn decompress_single<const NBITS: usize>(storage: &[u8], dim: usize) -> PyResult<Vec<f32>>
where
    Unsigned: Representation<NBITS>,
{
    let data = data_ref::<NBITS>(storage, dim)?;
    let mut output = vec![0.0f32; data.len()];
    data.decompress_into(&mut output)
        .map_err(|err: DecompressError| PyValueError::new_err(err.to_string()))?;
    Ok(output)
}
