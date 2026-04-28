/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::ANNError;
use pyo3::{exceptions::PyException, prelude::*};

pub type ANNResultPy<T> = Result<T, ANNErrorPy>;

#[derive(Debug)]
#[pyclass]
pub struct ANNErrorPy {
    #[pyo3(get, set)]
    pub message: String,
}

impl ANNErrorPy {
    pub fn new(err: ANNError) -> ANNErrorPy {
        ANNErrorPy {
            message: err.to_string(),
        }
    }
}

impl std::error::Error for ANNErrorPy {}

impl std::fmt::Display for ANNErrorPy {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ANNError: {}", self.message)
    }
}

impl From<ANNError> for ANNErrorPy {
    fn from(err: ANNError) -> ANNErrorPy {
        ANNErrorPy {
            message: err.to_string(),
        }
    }
}

impl From<std::io::Error> for ANNErrorPy {
    fn from(err: std::io::Error) -> ANNErrorPy {
        ANNErrorPy {
            message: err.to_string(),
        }
    }
}

impl From<ANNErrorPy> for PyErr {
    fn from(err: ANNErrorPy) -> PyErr {
        PyErr::new::<PyException, _>(err.message)
    }
}
