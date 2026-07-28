/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_linalg::{sgemm_aat_lower, MatrixName, SgemmError};

#[test]
fn computes_lower_triangle_and_preserves_upper_triangle() {
    #[rustfmt::skip]
    let a = [
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ];
    let untouched = -123.0;
    let mut c = [untouched; 9];

    sgemm_aat_lower(&a, 3, 2, &mut c).unwrap();

    #[rustfmt::skip]
    assert_eq!(c, [
         5.0, untouched, untouched,
        11.0,     25.0, untouched,
        17.0,     39.0,      61.0,
    ]);
}

#[test]
fn accepts_a_matrix_with_no_rows() {
    sgemm_aat_lower(&[], 0, 3, &mut []).unwrap();
}

#[test]
fn zero_inner_dimension_zeros_only_the_lower_triangle() {
    let untouched = -123.0;
    let mut c = [untouched; 9];

    sgemm_aat_lower(&[], 3, 0, &mut c).unwrap();

    #[rustfmt::skip]
    assert_eq!(c, [
        0.0, untouched, untouched,
        0.0,       0.0, untouched,
        0.0,       0.0,       0.0,
    ]);
}

#[test]
fn rejects_invalid_input_dimensions() {
    let mut c = [0.0; 4];

    let error = sgemm_aat_lower(&[0.0; 3], 2, 2, &mut c).unwrap_err();

    assert_eq!(
        error,
        SgemmError::InvalidMatrixDimensions {
            matrix_name: MatrixName::A,
            expected_rows: 2,
            expected_cols: 2,
            actual_len: 3,
        }
    );
}

#[test]
fn rejects_invalid_output_dimensions() {
    let mut c = [0.0; 3];

    let error = sgemm_aat_lower(&[0.0; 4], 2, 2, &mut c).unwrap_err();

    assert_eq!(
        error,
        SgemmError::InvalidMatrixDimensions {
            matrix_name: MatrixName::C,
            expected_rows: 2,
            expected_cols: 2,
            actual_len: 3,
        }
    );
}

#[test]
fn rejects_input_size_overflow() {
    let error = sgemm_aat_lower(&[], usize::MAX, 2, &mut []).unwrap_err();

    assert_eq!(
        error,
        SgemmError::DimensionOverflow {
            matrix_name: MatrixName::A,
            rows: usize::MAX,
            cols: 2,
        }
    );
}

#[test]
fn rejects_output_size_overflow() {
    let error = sgemm_aat_lower(&[], usize::MAX, 0, &mut []).unwrap_err();

    assert_eq!(
        error,
        SgemmError::DimensionOverflow {
            matrix_name: MatrixName::C,
            rows: usize::MAX,
            cols: usize::MAX,
        }
    );
}
