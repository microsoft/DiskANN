/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use thiserror::Error;

use super::common::Transpose;

/// Computes a matrix-matrix product with general matrices.
/// This implementation is used for miri testing.
/// Miri doesn't support cblas::sgemm().
#[allow(clippy::too_many_arguments)]
pub(super) fn sgemm_impl(
    atranspose: Transpose,
    btranspose: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: Option<f32>,
    c: &mut [f32],
) {
    let beta: f32 = beta.unwrap_or(0.0);

    for i in 0..m {
        for j in 0..n {
            let mut temp = 0.0;
            for l in 0..k {
                let a_val = match atranspose {
                    Transpose::None => a[(i * k) + l],
                    Transpose::Ordinary => a[(l * m) + i],
                };
                let b_val = match btranspose {
                    Transpose::None => b[(n * l) + j],
                    Transpose::Ordinary => b[(j * k) + l],
                };
                temp += a_val * b_val;
            }
            c[i * n + j] = alpha * temp + beta * c[i * n + j];
        }
    }
}

/// A test-problem for GEMM.
#[derive(Debug)]
pub(crate) struct TestProblem {
    atranspose: Transpose,
    btranspose: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: Vec<f32>,
    b: Vec<f32>,
    beta: Option<f32>,
    c: Vec<f32>,
    expected: Vec<f32>,
}

#[derive(Debug, Error)]
#[error("mismatch in test problem. got {:?}, expected {:?}", got, expected)]
pub(crate) struct ReferenceError {
    got: Vec<f32>,
    expected: Vec<f32>,
}

pub(crate) trait GemmFunction:
    Fn(Transpose, Transpose, usize, usize, usize, f32, &[f32], &[f32], Option<f32>, &mut [f32])
{
}
impl<F> GemmFunction for F where
    F: Fn(Transpose, Transpose, usize, usize, usize, f32, &[f32], &[f32], Option<f32>, &mut [f32])
{
}

impl TestProblem {
    // We're in a world with too many arguments unfortunately.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        atranspose: Transpose,
        btranspose: Transpose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: Vec<f32>,
        b: Vec<f32>,
        beta: Option<f32>,
        c: Vec<f32>,
        expected: Vec<f32>,
    ) -> Self {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), n * k);
        assert_eq!(c.len(), m * n);
        assert_eq!(expected.len(), m * n);
        Self {
            atranspose,
            btranspose,
            m,
            n,
            k,
            alpha,
            a,
            b,
            beta,
            c,
            expected,
        }
    }

    pub(crate) fn check<F: GemmFunction>(&self, f: F) -> Result<(), ReferenceError> {
        let mut result = self.c.clone();
        f(
            self.atranspose,
            self.btranspose,
            self.m,
            self.n,
            self.k,
            self.alpha,
            &self.a,
            &self.b,
            self.beta,
            &mut result,
        );

        if result == self.expected {
            Ok(())
        } else {
            Err(ReferenceError {
                got: result,
                expected: self.expected.clone(),
            })
        }
    }
}

/// Return a basic set of test-problems to check that a GEMM implementation passes a rough
/// sanity check of the API.
pub(crate) fn test_sgemm_problems() -> Vec<TestProblem> {
    let m = 2;
    let n = 3;
    let k = 4;

    // Matrix A:
    //  7  1  6  8
    //  6  2  6  1
    let a = vec![7.0, 1.0, 6.0, 8.0, 6.0, 2.0, 6.0, 1.0];
    let at = vec![7.0, 6.0, 1.0, 2.0, 6.0, 6.0, 8.0, 1.0];

    // Matrix B:
    //  1  9  6
    //  8  7  5
    //  6  4  3
    //  4  7  6
    let b = vec![1.0, 9.0, 6.0, 8.0, 7.0, 5.0, 6.0, 4.0, 3.0, 4.0, 7.0, 6.0];
    let bt = vec![1.0, 8.0, 6.0, 4.0, 9.0, 7.0, 4.0, 7.0, 6.0, 5.0, 3.0, 6.0];

    // Matrix C:
    //  3  3  3
    //  0  1  9
    let c = vec![3.0, 3.0, 3.0, 0.0, 1.0, 9.0];

    // None * C + 1 * A * B
    let none_1ab = vec![83.0, 150.0, 113.0, 62.0, 99.0, 70.0];
    // None * C + 2 * A * B
    let none_2ab = vec![166.0, 300.0, 226.0, 124.0, 198.0, 140.0];

    // 2 * C + 1 * A * B
    let c2_1ab = vec![89.0, 156.0, 119.0, 62.0, 101.0, 88.0];
    // 2 * C + 2 * A * B
    let c2_2ab = vec![172.0, 306.0, 232.0, 124.0, 200.0, 158.0];

    // 1 * C + 1 * A * B
    let c1_1ab = vec![86.0, 153.0, 116.0, 62.0, 100.0, 79.0];
    // 1 * C + 2 * A * B
    let c1_2ab = vec![169.0, 303.0, 229.0, 124.0, 199.0, 149.0];

    let mut problems = Vec::new();
    let make_problem = |atranspose: Transpose,
                        btranspose: Transpose,
                        alpha: f32,
                        beta: Option<f32>,
                        expected: Vec<f32>| {
        let a_ = atranspose.call(|| a.clone(), || at.clone());
        let b_ = btranspose.call(|| b.clone(), || bt.clone());
        TestProblem::new(
            atranspose,
            btranspose,
            m,
            n,
            k,
            alpha,
            a_,
            b_,
            beta,
            c.clone(),
            expected,
        )
    };

    let mut on_transpose_combinations = |alpha: f32, beta: Option<f32>, expected: Vec<f32>| {
        problems.push(make_problem(
            Transpose::None,
            Transpose::None,
            alpha,
            beta,
            expected.clone(),
        ));
        problems.push(make_problem(
            Transpose::None,
            Transpose::Ordinary,
            alpha,
            beta,
            expected.clone(),
        ));
        problems.push(make_problem(
            Transpose::Ordinary,
            Transpose::None,
            alpha,
            beta,
            expected.clone(),
        ));
        problems.push(make_problem(
            Transpose::Ordinary,
            Transpose::Ordinary,
            alpha,
            beta,
            expected,
        ));
    };

    on_transpose_combinations(1.0, None, none_1ab.clone());
    on_transpose_combinations(2.0, None, none_2ab.clone());

    on_transpose_combinations(1.0, Some(0.0), none_1ab.clone());
    on_transpose_combinations(2.0, Some(0.0), none_2ab.clone());

    on_transpose_combinations(1.0, Some(2.0), c2_1ab.clone());
    on_transpose_combinations(2.0, Some(2.0), c2_2ab.clone());

    on_transpose_combinations(1.0, Some(1.0), c1_1ab.clone());
    on_transpose_combinations(2.0, Some(1.0), c1_2ab.clone());

    assert_eq!(problems.len(), 32);
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_implementation() {
        let problems = test_sgemm_problems();
        for (i, problem) in problems.iter().enumerate() {
            let result = problem.check(sgemm_impl);
            if let Err(err) = result {
                panic!("{} on iteration {}. Problem: {:?}", err, i, problem);
            }
        }
    }
}
