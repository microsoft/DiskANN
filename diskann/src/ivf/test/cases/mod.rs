/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Test cases for the IVF module, grouped by concern.
//!
//! * [`correctness`] -- the central invariant (IVF == brute force restricted to probed
//!   lists) and its corollaries across `k`/`nprobe`.
//! * [`concurrency`] -- the sequential and concurrent fine-scan paths agree, and the
//!   concurrent path actually fans out.
//! * [`insert`] -- inserted points land in the right list and become findable.
//! * [`errors`] -- accessor/strategy failures escalate out of the index API.
//!
//! All cases share the [`fixture`] below: a clean 8x8 integer grid partitioned into four
//! quadrant lists. The geometry is deliberately simple so the expected list assignments are
//! obvious by inspection, but every assertion is still validated against the exact oracle in
//! [`super::harness`] rather than hand-computed answers.

mod concurrency;
mod correctness;
mod errors;
mod insert;

use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;

use crate::ivf::{IvfIndex, test::provider::Provider};

/// Vector/centroid dimension for the shared fixture.
pub(super) const DIM: usize = 2;

/// Side length of the integer grid used as the dataset (`SIDE * SIDE` points).
pub(super) const SIDE: usize = 8;

/// Number of inverted lists (quadrant centroids) in the shared fixture.
pub(super) const N_LISTS: usize = 4;

/// Metric used throughout the IVF tests.
pub(super) const METRIC: Metric = Metric::L2;

/// The four quadrant centroids of the 8x8 grid, one per inverted list.
fn centroids() -> Matrix<f32> {
    // Quadrant centers of the [0, 7]^2 grid.
    #[rustfmt::skip]
    let data = vec![
        1.5, 1.5, // list 0: lower-left
        1.5, 5.5, // list 1: upper-left
        5.5, 1.5, // list 2: lower-right
        5.5, 5.5, // list 3: upper-right
    ];
    Matrix::try_from(data.into_boxed_slice(), N_LISTS, DIM).expect("centroid matrix is well-formed")
}

/// Build the shared fixture: an [`IvfIndex`] over an 8x8 grid split into four quadrant
/// lists.
pub(super) fn grid_index() -> IvfIndex<Provider> {
    let vectors = crate::graph::test::synthetic::Grid::Two.data(SIDE);
    debug_assert_eq!(vectors.ncols(), DIM);
    debug_assert_eq!(vectors.nrows(), SIDE * SIDE);
    let provider =
        Provider::build(vectors, centroids(), METRIC).expect("fixture provider is well-formed");
    IvfIndex::new(provider)
}

/// A spread of query points: grid corners, the center, and off-grid coordinates.
pub(super) fn queries() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![7.0, 7.0],
        vec![3.5, 3.5],
        vec![2.0, 6.0],
        vec![6.3, 1.1],
        vec![4.0, 0.0],
    ]
}
