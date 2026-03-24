/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[cfg(not(miri))]
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use diskann_utils::views::Matrix;
use rand::{
    distr::{Distribution, Uniform},
    rngs::StdRng,
    seq::SliceRandom,
};
use thiserror::Error;

#[cfg(not(miri))]
use crate::alloc::GlobalAllocator;
use crate::alloc::{AllocatorCore, AllocatorError};

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
#[cfg(not(miri))]
#[derive(Debug, Clone)]
pub(crate) struct LimitedAllocator {
    remaining: Arc<AtomicUsize>,
}

#[cfg(not(miri))]
impl LimitedAllocator {
    pub(crate) fn new(allocations: usize) -> Self {
        Self {
            remaining: Arc::new(AtomicUsize::new(allocations)),
        }
    }
}

#[cfg(not(miri))]
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
        // SAFETY: Inherited from caller.
        unsafe { (GlobalAllocator).deallocate(ptr, layout) }
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

/// A utility to help check fuzzy numerical bounds.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Check {
    /// Assert two floating point numbers are within a set number of ULPs oeachother.
    Ulp(usize),

    /// Assert two floating point numbers `x` and `y` satisfy:
    /// ```text
    /// | x - y | <= abs OR | x - y | / max(|x|, |y|) <= rel
    /// ```
    AbsRel { abs: f32, rel: f32 },

    /// Skip the check entirely.
    #[cfg(not(miri))]
    Skip,
}

impl Check {
    pub(crate) const fn ulp(ulp: usize) -> Self {
        Self::Ulp(ulp)
    }

    pub(crate) const fn absrel(abs: f32, rel: f32) -> Self {
        Self::AbsRel { abs, rel }
    }

    #[cfg(not(miri))]
    pub(crate) const fn skip() -> Self {
        Self::Skip
    }

    /// Return `Ok` if the two arguments satisfy the check.
    ///
    /// Otherwise, return an error with a suitably formed `Display` implementation to provide
    /// a useful diagnostic.
    pub(crate) fn check(&self, got: f32, expected: f32) -> Result<(), CheckFailed> {
        match self {
            Self::Ulp(ulp) => {
                if within_ulp(got, expected, *ulp) {
                    Ok(())
                } else {
                    Err(CheckFailed::Ulp {
                        ulp: *ulp,
                        got,
                        expected,
                    })
                }
            }
            Self::AbsRel { abs, rel } => {
                let abs_got = (got - expected).abs();
                let max_magnitude = got.abs().max(expected.abs());

                // When both values are zero (or very near), the relative error
                // is undefined. Fall back to the absolute check only.
                let (rel_ok, rel_got) = if max_magnitude == 0.0 {
                    (false, f32::INFINITY)
                } else {
                    let rel_got = abs_got / max_magnitude;
                    (rel_got <= *rel, rel_got)
                };

                if abs_got <= *abs || rel_ok {
                    Ok(())
                } else {
                    Err(CheckFailed::AbsRel {
                        abs_limit: *abs,
                        rel_limit: *rel,
                        abs_got,
                        rel_got,
                        got,
                        expected,
                    })
                }
            }
            #[cfg(not(miri))]
            Self::Skip => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub(crate) enum CheckFailed {
    #[error("not within {ulp} ulp - got {got}, expected {expected}")]
    Ulp { ulp: usize, got: f32, expected: f32 },
    #[error(
        "not within {abs_limit}/{rel_limit} - errors {abs_got}/{rel_got} - \
            got {got}, expected {expected}"
    )]
    AbsRel {
        abs_limit: f32,
        rel_limit: f32,
        abs_got: f32,
        rel_got: f32,
        got: f32,
        expected: f32,
    },
}

fn within_ulp(mut got: f32, expected: f32, ulp: usize) -> bool {
    if got == expected {
        true
    } else if got < expected {
        for _ in 0..ulp {
            got = got.next_up();
            if got >= expected {
                return true;
            }
        }
        false
    } else {
        for _ in 0..ulp {
            got = got.next_down();
            if got <= expected {
                return true;
            }
        }
        false
    }
}
