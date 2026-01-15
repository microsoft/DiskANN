/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use diskann_utils::views::Matrix;
use rand::{
    distr::{Distribution, Uniform},
    rngs::StdRng,
    seq::SliceRandom,
};

use crate::alloc::{AllocatorCore, AllocatorError, GlobalAllocator};

/// An allocator that always fails.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlwaysFails;

// SAFETY: This always fails.
unsafe impl AllocatorCore for AlwaysFails {
    fn allocate(
        &self,
        _layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, AllocatorError> {
        Err(AllocatorError)
    }

    unsafe fn deallocate(&self, _ptr: std::ptr::NonNull<[u8]>, _layout: std::alloc::Layout) {}
}

/// An allocator that can only perform a limited number of allocations.
///
/// Used to test interfaces for allocation reliability.
#[derive(Debug, Clone)]
pub(crate) struct LimitedAllocator {
    remaining: Arc<AtomicUsize>,
}

impl LimitedAllocator {
    pub(crate) fn new(allocations: usize) -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(allocations)),
        }
    }
}

/// SAFETY: This either forwards to the global allocator, or failed.
unsafe impl AllocatorCore for LimitedAllocator {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, AllocatorError> {
        let mut remaining = self.remaining.load(Ordering::Relaxed);
        if remaining == 0 {
            return Err(AllocatorError);
        }

        while let Err(actual) = self.remaining.compare_exchange(
            remaining,
            remaining - 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            if actual == 0 {
                return Err(AllocatorError);
            }
            remaining = actual;
        }

        // The loop succeeded - we can return an al.location.
        (GlobalAllocator).allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<[u8]>, layout: std::alloc::Layout) {
        (GlobalAllocator).deallocate(ptr, layout)
    }
}

/// Compute the relative error between `got` and `expected`.
pub(crate) fn compute_relative_error(got: f32, expected: f32) -> f32 {
    if got == expected {
        0.0
    } else {
        assert!(expected != 0.0);
        (got - expected).abs() / expected.abs()
    }
}

/// Compute the absolute error between `got` and `expected`.
pub(crate) fn compute_absolute_error(got: f32, expected: f32) -> f32 {
    (got - expected).abs()
}

pub(crate) struct TestProblem {
    pub(crate) data: Matrix<f32>,
    pub(crate) means: Vec<f64>,
    pub(crate) variances: Vec<f64>,
    pub(crate) mean_norm: f64,
}

pub(crate) fn create_test_problem(nrows: usize, ncols: usize, rng: &mut StdRng) -> TestProblem {
    let distribution = Uniform::new_inclusive::<i64, i64>(-10, 10).unwrap();

    // The expected mean for each dimension.
    let means: Vec<f32> = (0..ncols)
        .map(|_| distribution.sample(rng) as f32)
        .collect();

    // How much we are going to scale the offsets for each dimension to change the
    // variance.
    let scales: Vec<f32> = (0..ncols)
        .map(|_| distribution.sample(rng).abs() as f32)
        .collect();

    // We add an equal number of components above and below the target mean and shuffle
    // the order in which they are applied.
    let mut offsets: Vec<f32> = (0i64..(nrows as i64))
        .map(|row: i64| {
            let nrows: i64 = nrows as i64;
            // If we have an even number of rows, then we want an equal number above and
            // below zero.
            let offset = if nrows % 2 == 0 {
                2 * row - nrows + 1
            } else {
                row - (nrows - 1) / 2
            };
            offset as f32
        })
        .collect();

    // There is a closed for solution for doing the sum of squares like we are here,
    // but we're just going to go ahead and compute it manually.
    let variances = scales
        .iter()
        .map(|&scale| {
            let scale = scale as usize;

            let variance: usize = if nrows.is_multiple_of(2) {
                let half: usize = (1..=nrows / 2)
                    .map(|i| {
                        let j = scale * (2 * i - 1);
                        j * j
                    })
                    .sum();
                2 * half
            } else {
                let half: usize = (1..=nrows / 2)
                    .map(|i| {
                        let j = scale * i;
                        j * j
                    })
                    .sum();
                2 * half
            };
            variance as f64 / nrows as f64
        })
        .collect();

    let mut data = Matrix::<f32>::new(0.0, nrows, ncols);
    for col in 0..ncols {
        offsets.shuffle(rng);
        for row in 0..nrows {
            data[(row, col)] = means[col] + scales[col] * offsets[row];
        }
    }

    // Compute the mean norm directly.
    let mean_norm = data
        .row_iter()
        .map(|row| {
            row.iter()
                .map(|&i| {
                    let i: f64 = i.into();
                    i * i
                })
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>()
        / (nrows as f64);

    TestProblem {
        data,
        means: means.into_iter().map(|i| i as f64).collect(),
        variances,
        mean_norm,
    }
}
