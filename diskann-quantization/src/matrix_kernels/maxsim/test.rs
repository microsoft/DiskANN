/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::Matrix;

use crate::matrix_kernels::test_util::TestDistr;

/// Generate a test MaxSim problem `[M x K] . [K x N]` where both matrices are row-major.
pub(super) fn generate(
    m: usize,
    k: usize,
    n: usize,
    rng: &mut impl rand::Rng,
) -> (Matrix<f32>, Matrix<f32>, Vec<f32>) {
    let ref_a = TestDistr::matrix::<f32>(m, k, rng);
    let ref_b = TestDistr::matrix::<f32>(k, n, rng);

    let ref_c: Vec<f32> = ref_a
        .row_iter()
        .map(|a_row| {
            let mut max_ip = f32::NEG_INFINITY;
            for b_col in 0..n {
                let mut ip = 0.0;
                for (k, a) in a_row.iter().enumerate() {
                    ip = a.mul_add(ref_b[(k, b_col)], ip);
                }
                max_ip = max_ip.max(ip);
            }

            max_ip
        })
        .collect();

    (ref_a, ref_b, ref_c)
}

#[derive(Debug, Clone)]
pub(super) struct TestDims {
    pub(super) a_panels_per_tile: usize,
    pub(super) total_a_rows: usize,
    pub(super) b_cols_per_tile: usize,
    pub(super) total_b_cols: usize,
    pub(super) k: usize,
}

impl TestDims {
    fn from_tuple(
        [
            a_panels_per_tile,
            total_a_rows,
            b_cols_per_tile,
            total_b_cols,
            k,
        ]: [usize; 5],
    ) -> Self {
        Self {
            a_panels_per_tile,
            total_a_rows,
            b_cols_per_tile,
            total_b_cols,
            k,
        }
    }
}

/// Test dimensions for `packed_x_unpacked` kernels.
pub(super) fn packed_x_unpacked_test_dims(mr: usize, nr: usize) -> Vec<TestDims> {
    [
        [1, 1, 1, 1, 1],                        // Smallest logical output
        [1, mr / 2, 1, 1, 1],                   // Partial first A panel
        [1, mr - 1, 2 * nr, nr, 3],             // Nearly full first A panel
        [2, mr + 1, 2 * nr, nr, 3],             // Partial second A panel
        [2, 2 * mr - 1, 2 * nr, 2 * nr + 1, 5], // Partial A panel and split B
        [2, 2 * mr + 1, 2 * nr, 2 * nr + 1, 5], // Split A and B with a partial panel
        [1, mr * 3, 1, 3, 1],                   // Unit advancement, no reuse.
        [2, mr * 2, 2 * nr, 2 * nr, 3],         // Values a direct multiple of the blocking.
        [2, mr * 3, 2 * nr, nr, 3],
        [2, mr, 2 * nr, 2 * nr + 1, 3],
        [2, mr * 3, 2 * nr, 2 * nr + 1, 5],
        [2, mr * 5, 2 * nr, 4 * nr + 1, 1],
    ]
    .map(TestDims::from_tuple)
    .into_iter()
    .collect()
}
