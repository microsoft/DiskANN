/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use numpy::{ndarray, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

//converts a PyArray2 to a vector of vectors where each vector represents one row
pub fn pyarray2_to_vec_row_decomp<T: numpy::Element + Clone>(
    pyarray: &Bound<PyArray2<T>>,
) -> Vec<Vec<T>> {
    let converted_pyarray = pyarray
        .readonly()
        .as_array()
        .axis_iter(ndarray::Axis(0)) // Iterate over rows
        .map(|row| row.to_vec())
        .collect();
    converted_pyarray
}

#[cfg(test)]
mod pyarray_test {
    use numpy::PyArray2;
    use pyo3::Python;

    use crate::utils::pyarray2_to_vec_row_decomp;

    #[test]
    fn test_convert_pyarray2_to_vec() {
        Python::initialize();
        Python::attach(|py| {
            //create pyArray, convert to vector, check that vectors are the same
            let vec2 = vec![vec![1, 2, 3], vec![4, 5, 6]];
            let pyarray_to_convert = &PyArray2::from_vec2(py, &vec2).unwrap();
            let vec_of_vecs = pyarray2_to_vec_row_decomp(pyarray_to_convert);
            assert!(vec_of_vecs.eq(&vec2));
        });
    }
}
