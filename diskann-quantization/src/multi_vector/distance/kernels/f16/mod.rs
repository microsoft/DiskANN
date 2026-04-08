// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 × f16 micro-kernel family for block-transposed multi-vector distance.
//!
//! Implements the [`Kernel<A>`](super::Kernel) trait with `AElem = BElem = f16`,
//! lazily unpacking both sides to f32 via the `prepare_a` / `prepare_b` hooks.
//! The micro-kernel body itself is the same f32 SIMD arithmetic used by the
//! f32 kernel family — zero additional micro-kernel code is required.
//!
//! # Architecture-specific implementations
//!
//! | Module   | Arch      | Geometry | Conversion       |
//! |----------|-----------|----------|------------------|
//! | `v3`     | V3        | 16 × 4   | [`CastFromSlice`]|
//! | `scalar` | Scalar    | 8 × 2    | [`CastFromSlice`]|
//!
//! [`CastFromSlice`] dispatches via `diskann_wide::ARCH` at compile time, so
//! the conversion is SIMD-accelerated on every platform that `diskann-wide`
//! supports (e.g. Neon on aarch64), regardless of the kernel's architecture
//! parameter.
//!
//! [`CastFromSlice`]: diskann_vector::conversion::CastFromSlice

use diskann_wide::{
    Architecture,
    arch::{
        Scalar, Target, Target1, Target2, Target3,
        x86_64::{V3, V4},
    },
};

use diskann_utils::Reborrow;

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};

use super::Kernel;
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::distance::{Chamfer, MaxSim};
use crate::multi_vector::{BlockTransposed, BlockTransposedRef, MatRef, Standard};

mod scalar;
#[cfg(target_arch = "x86_64")]
mod v3;

// ── F16 kernel ───────────────────────────────────────────────────

/// Kernel type for f16 micro-kernels with block size `GROUP`.
///
/// Both A and B sides store [`half::f16`] values. The
/// [`prepare_a`](Kernel::prepare_a) / [`prepare_b`](Kernel::prepare_b)
/// hooks convert to f32 into caller-provided staging buffers allocated via
/// [`Kernel::new_buffers`]. The micro-kernel itself is the same f32 SIMD
/// arithmetic used by [`F32Kernel`](super::f32::F32Kernel).
pub(crate) struct F16Kernel<const GROUP: usize>;

// ── Public entry point ───────────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn chamfer_kernel_f16_panic() {
    panic!(
        "chamfer_kernel_f16: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed f16 A matrix
/// and a row-major f16 B matrix, writing per-A-row max similarities into
/// `scratch`.
///
/// Both A and B sides are lazily unpacked to f32 via the `prepare_a` /
/// `prepare_b` hooks. The micro-kernel itself operates in f32.
///
/// # Arguments
///
/// * `arch` - Architecture token (must satisfy the `Kernel` where-bound).
/// * `a` - Block-transposed f16 matrix view with block size `GROUP`.
/// * `b` - Row-major f16 matrix view.
/// * `scratch` - Mutable buffer of length [`BlockTransposedRef::available_rows()`].
///   Must be initialized to `f32::MIN` before the first call.
///
/// # Panics
///
/// Panics if `scratch.len() != a.available_rows()` or `a.ncols() != b.vector_dim()`.
pub(crate) fn chamfer_kernel_f16<A: Architecture, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, half::f16, GROUP>,
    b: MatRef<'_, Standard<half::f16>>,
    scratch: &mut [f32],
) where
    F16Kernel<GROUP>:
        Kernel<A, AElem = half::f16, BElem = half::f16, APrepared = f32, BPrepared = f32>,
{
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        chamfer_kernel_f16_panic();
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    const {
        assert!(
            <F16Kernel<GROUP> as Kernel<A>>::A_PANEL == GROUP,
            "F16Kernel A_PANEL must equal GROUP for layout correctness"
        )
    };

    // SAFETY:
    // - a.as_ptr() is valid for a.available_rows() * k elements of f16.
    // - MatRef<Standard<f16>> stores nrows * ncols contiguous f16 elements.
    // - scratch.len() == a.available_rows() (checked above).
    // - a.available_rows() is always a multiple of GROUP, and the debug_assert
    //   above verifies A_PANEL == GROUP.
    unsafe {
        tiled_reduce::<A, F16Kernel<GROUP>>(
            arch,
            a.as_ptr(),
            a.available_rows(),
            b.as_slice().as_ptr(),
            b_nrows,
            k,
            scratch,
        );
    }
}

impl<A, const GROUP: usize>
    Target3<
        A,
        (),
        BlockTransposedRef<'_, half::f16, GROUP>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for F16Kernel<GROUP>
where
    A: Architecture,
    Self: Kernel<A, AElem = half::f16, BElem = half::f16, APrepared = f32, BPrepared = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, half::f16, GROUP>,
        rhs: MatRef<'_, Standard<half::f16>>,
        scratch: &mut [f32],
    ) {
        chamfer_kernel_f16(arch, lhs, rhs, scratch)
    }
}

// ── MaxSim / Chamfer trait implementations (runtime-dispatched) ──

use crate::multi_vector::transposed_query::TransposedQueryRef;

impl DistanceFunctionMut<TransposedQueryRef<'_, half::f16>, MatRef<'_, Standard<half::f16>>>
    for MaxSim<'_>
{
    fn evaluate(
        &mut self,
        query: TransposedQueryRef<'_, half::f16>,
        doc: MatRef<'_, Standard<half::f16>>,
    ) {
        assert!(
            self.size() == query.nrows(),
            "scores buffer not right size: {} != {}",
            self.size(),
            query.nrows()
        );

        if doc.num_vectors() == 0 {
            self.scores_mut().fill(f32::MAX);
            return;
        }

        let available = query.available_rows();
        let nq = query.nrows();

        let mut padded_scratch = vec![f32::MIN; available];

        diskann_wide::arch::dispatch_no_features(super::ChamferKernelRunnerF16 {
            tq: query,
            doc,
            scratch: &mut padded_scratch,
        });

        let scratch = self.scores_mut();
        scratch.copy_from_slice(&padded_scratch[..nq]);

        for s in scratch.iter_mut() {
            *s = -*s;
        }
    }
}

impl<A, const GROUP: usize>
    Target2<A, f32, BlockTransposedRef<'_, half::f16, GROUP>, MatRef<'_, Standard<half::f16>>>
    for Chamfer
where
    A: Architecture,
    F16Kernel<GROUP>:
        Kernel<A, AElem = half::f16, BElem = half::f16, APrepared = f32, BPrepared = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        query: BlockTransposedRef<'_, half::f16, GROUP>,
        doc: MatRef<'_, Standard<half::f16>>,
    ) -> f32 {
        if doc.num_vectors() == 0 {
            return 0.0;
        }

        let available = query.available_rows();
        let nq = query.nrows();
        let mut scratch = vec![f32::MIN; available];

        arch.run3(F16Kernel::<GROUP>, query, doc, scratch.as_mut_slice());

        scratch.iter().take(nq).map(|&s| -s).sum()
    }
}

pub struct QueryComputer<T> {
    inner: Box<dyn DynQueryComputer<T>>,
}

impl QueryComputer<half::f16> {
    pub fn new(query: MatRef<'_, Standard<half::f16>>) -> Self {
        diskann_wide::arch::dispatch1_no_features(Build, query)
    }
}

trait DynQueryComputer<T: Copy> {
    fn evaluate(&self, rhs: MatRef<'_, Standard<T>>) -> f32;
}

struct Impl<A, T> {
    prepared: T,
    arch: A,
}

impl<A, T> DynQueryComputer<half::f16> for Impl<A, T>
where
    A: Architecture,
    T: for<'a> Reborrow<'a>,
    Chamfer: for<'a> Target2<A, f32, <T as Reborrow<'a>>::Target, MatRef<'a, Standard<half::f16>>>,
{
    fn evaluate(&self, rhs: MatRef<'_, Standard<half::f16>>) -> f32 {
        self.arch.run2(Chamfer, self.prepared.reborrow(), rhs)
    }
}

struct Build;

impl Target1<Scalar, QueryComputer<half::f16>, MatRef<'_, Standard<half::f16>>> for Build {
    #[inline(always)]
    fn run(self, arch: Scalar, m: MatRef<'_, Standard<half::f16>>) -> QueryComputer<half::f16> {
        let prepared = BlockTransposed::<_, 8>::from_matrix_view(m.as_matrix_view());
        let inner = Box::new(Impl { prepared, arch });
        QueryComputer { inner }
    }
}

impl Target1<V3, QueryComputer<half::f16>, MatRef<'_, Standard<half::f16>>> for Build {
    #[inline(always)]
    fn run(self, arch: V3, m: MatRef<'_, Standard<half::f16>>) -> QueryComputer<half::f16> {
        let prepared = BlockTransposed::<_, 16>::from_matrix_view(m.as_matrix_view());
        let inner = Box::new(Impl { prepared, arch });
        QueryComputer { inner }
    }
}

impl Target1<V4, QueryComputer<half::f16>, MatRef<'_, Standard<half::f16>>> for Build {
    #[inline(always)]
    fn run(self, arch: V4, m: MatRef<'_, Standard<half::f16>>) -> QueryComputer<half::f16> {
        let prepared = BlockTransposed::<_, 16>::from_matrix_view(m.as_matrix_view());
        let arch: V3 = arch.retarget();
        let inner = Box::new(Impl { prepared, arch });
        QueryComputer { inner }
    }
}

// impl PureDistanceFunction<TransposedQueryRef<'_, half::f16>, MatRef<'_, Standard<half::f16>>, f32>
//     for Chamfer
// {
//     fn evaluate(
//         query: TransposedQueryRef<'_, half::f16>,
//         doc: MatRef<'_, Standard<half::f16>>,
//     ) -> f32 {
//         if doc.num_vectors() == 0 {
//             return 0.0;
//         }
//
//         let available = query.available_rows();
//         let nq = query.nrows();
//
//         let mut scratch = vec![f32::MIN; available];
//
//         diskann_wide::arch::dispatch_no_features(super::ChamferKernelRunnerF16 {
//             tq: query,
//             doc,
//             scratch: &mut scratch,
//         });
//
//         scratch.iter().take(nq).map(|&s| -s).sum()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::distance::QueryMatRef;
    use crate::multi_vector::transposed_query::transpose_query_f16;

    /// Helper to create a MatRef from raw f16 data.
    fn make_query_mat(
        data: &[half::f16],
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, Standard<half::f16>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    /// Generate deterministic test data as f16.
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<half::f16> {
        (0..len)
            .map(|v| diskann_wide::cast_f32_to_f16(((v + shift) % ceil) as f32))
            .collect()
    }

    /// Test cases: (num_queries, num_docs, dim).
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 1),   // Degenerate single-element
        (1, 1, 2),   // Minimal non-trivial
        (1, 1, 4),   // Single query, single doc
        (1, 5, 8),   // Single query, multiple docs
        (5, 1, 8),   // Multiple queries, single doc
        (3, 2, 3),   // Prime k
        (3, 4, 16),  // General case
        (5, 3, 5),   // Prime k, A-remainder on aarch64
        (7, 7, 32),  // Square case
        (2, 3, 7),   // k not divisible by SIMD lanes
        (2, 3, 128), // Larger dimension
        (16, 4, 64), // Exact A_PANEL on x86_64; two panels on aarch64
        (17, 4, 64), // One more than A_PANEL (remainder)
        (32, 5, 16), // Multiple full A-panels, remainder B-rows (5 % 4 = 1)
        (48, 3, 16), // 3 A-tiles on x86_64; 6 on aarch64
        (16, 6, 32), // Remainder B-rows (6 % 4 = 2)
        (16, 7, 32), // Remainder B-rows (7 % 4 = 3)
        (16, 8, 32), // No remainder B-rows (8 % 4 = 0)
    ];

    #[test]
    fn chamfer_matches_fallback() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: fallback Chamfer (works on f16 via InnerProduct)
            let simple_query: QueryMatRef<_> = query_mat.into();
            let expected = Chamfer::evaluate(simple_query, doc);

            // Runtime-dispatched transposed query
            let tq = transpose_query_f16(query_mat.as_matrix_view());
            let actual = Chamfer::evaluate(tq.as_ref(), doc);

            assert!(
                (actual - expected).abs() < 1e-1,
                "Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn max_sim_matches_fallback() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: fallback MaxSim
            let mut expected_scores = vec![0.0f32; nq];
            let simple_query: QueryMatRef<_> = query_mat.into();
            let _ = MaxSim::new(&mut expected_scores)
                .unwrap()
                .evaluate(simple_query, doc);

            // Runtime-dispatched transposed query
            let tq = transpose_query_f16(query_mat.as_matrix_view());
            let mut actual_scores = vec![0.0f32; nq];
            MaxSim::new(&mut actual_scores)
                .unwrap()
                .evaluate(tq.as_ref(), doc);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < 1e-1,
                    "MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i]
                );
            }
        }
    }

    #[test]
    fn chamfer_with_zero_docs_returns_zero() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let tq = transpose_query_f16(query_mat.as_matrix_view());

        let doc = make_query_mat(&[], 0, 2);
        let result = Chamfer::evaluate(tq.as_ref(), doc);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(2.0),
            diskann_wide::cast_f32_to_f16(3.0),
            diskann_wide::cast_f32_to_f16(4.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let tq = transpose_query_f16(query_mat.as_matrix_view());

        let doc_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let doc = make_query_mat(&doc_data, 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        MaxSim::new(&mut scores).unwrap().evaluate(tq.as_ref(), doc);
    }

    #[test]
    fn negative_values_propagate() {
        // Hand-crafted negative vectors: query = [[-1, -2], [-3, -4]], doc = [[-1, 0]]
        let query_data = [
            diskann_wide::cast_f32_to_f16(-1.0),
            diskann_wide::cast_f32_to_f16(-2.0),
            diskann_wide::cast_f32_to_f16(-3.0),
            diskann_wide::cast_f32_to_f16(-4.0),
        ];
        let doc_data = [
            diskann_wide::cast_f32_to_f16(-1.0),
            diskann_wide::cast_f32_to_f16(0.0),
        ];

        let query_mat = make_query_mat(&query_data, 2, 2);
        let doc = make_query_mat(&doc_data, 1, 2);

        let simple_query: QueryMatRef<_> = query_mat.into();
        let expected = Chamfer::evaluate(simple_query, doc);

        let tq = transpose_query_f16(query_mat.as_matrix_view());
        let actual = Chamfer::evaluate(tq.as_ref(), doc);

        assert!(
            (actual - expected).abs() < 1e-2,
            "Chamfer mismatch with negative values: actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn max_sim_with_zero_docs() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(0.0),
            diskann_wide::cast_f32_to_f16(1.0),
        ];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let tq = transpose_query_f16(query_mat.as_matrix_view());

        let doc = make_query_mat(&[], 0, 2);
        let mut scores = vec![0.0f32; 2];
        MaxSim::new(&mut scores).unwrap().evaluate(tq.as_ref(), doc);

        // With zero docs the kernel fills f32::MIN then negates → f32::MAX.
        for &s in &scores {
            assert_eq!(s, f32::MAX, "zero-doc MaxSim should produce f32::MAX");
        }
    }

    /// Verify f16 kernel precision with known exact inner products.
    ///
    /// Query: [[1, 2]], Doc: [[3, 4]]
    /// IP = 1*3 + 2*4 = 11. All values are exactly representable in f16.
    /// Chamfer returns negated distance: sum of (-IP) = -11.
    #[test]
    fn f16_known_exact_result() {
        let query_data = [
            diskann_wide::cast_f32_to_f16(1.0),
            diskann_wide::cast_f32_to_f16(2.0),
        ];
        let doc_data = [
            diskann_wide::cast_f32_to_f16(3.0),
            diskann_wide::cast_f32_to_f16(4.0),
        ];

        let query_mat = make_query_mat(&query_data, 1, 2);
        let doc = make_query_mat(&doc_data, 1, 2);

        let tq = transpose_query_f16(query_mat.as_matrix_view());
        let actual = Chamfer::evaluate(tq.as_ref(), doc);

        // IP = 1*3 + 2*4 = 11. Chamfer returns the negated-IP distance = -11.
        // All intermediate values (1, 2, 3, 4, 11) are exactly representable in f16.
        assert!(
            (actual - (-11.0)).abs() < 1e-3,
            "f16 precision test: expected -11.0, got {actual}"
        );
    }
}
