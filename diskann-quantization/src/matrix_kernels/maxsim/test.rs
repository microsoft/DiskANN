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
