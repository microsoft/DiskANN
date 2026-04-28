/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use pyo3::prelude::*;

#[pyclass(subclass)]
pub struct QuantizerBase {
    dim: usize,
    output_dim: usize,
    bit_width: usize,
    algorithm: &'static str,
}

impl QuantizerBase {
    pub(crate) fn new(
        dim: usize,
        output_dim: usize,
        bit_width: usize,
        algorithm: &'static str,
    ) -> Self {
        Self {
            dim,
            output_dim,
            bit_width,
            algorithm,
        }
    }
}

#[pymethods]
impl QuantizerBase {
    /// Dimensionality of the full-precision training data.
    #[getter]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Dimensionality after applying the quantizer transform.
    #[getter]
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Bit width used by this quantizer.
    #[getter]
    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// Descriptive name of the quantization algorithm.
    #[getter]
    pub fn algorithm(&self) -> &str {
        self.algorithm
    }
}
