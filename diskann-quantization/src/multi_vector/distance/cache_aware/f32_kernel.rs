// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 × f32 cache-aware micro-kernel using FMA and max-reduce.
//!
//! Provides:
//!
//! - `F32Kernel` — SIMD micro-kernel (16×4 via FMA + `max_simd`).
//! - [`cache_aware_chamfer`] — public entry point for the reducing max-IP GEMM.
//! - [`QueryBlockTransposedRef`] — newtype wrapper for f32 block-transposed queries.
//! - [`MaxSim`](crate::multi_vector::distance::MaxSim) /
//!   [`Chamfer`](crate::multi_vector::distance::Chamfer) trait implementations for
//!   `QueryBlockTransposedRef` × `MatRef<Standard<f32>>`.

use std::ops::Deref;

use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::CacheAwareKernel;
use super::kernel::{Reduce, tiled_reduce};
use crate::multi_vector::distance::{Chamfer, MaxSim};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

diskann_wide::alias!(f32s = f32x8);

// ── QueryBlockTransposedRef ──────────────────────────────────────

/// A query wrapper for block-transposed multi-vector views.
///
/// This wrapper distinguishes query matrices from document matrices
/// at compile time, preventing accidental argument swapping in asymmetric
/// distance computations like [`MaxSim`](crate::multi_vector::distance::MaxSim) and
/// [`Chamfer`](crate::multi_vector::distance::Chamfer).
///
/// Analogous to [`QueryMatRef`](crate::multi_vector::distance::QueryMatRef) but for
/// [`BlockTransposedRef`] queries rather than row-major
/// [`MatRef`](crate::multi_vector::MatRef) queries.
///
/// # Example
///
/// ```
/// use diskann_quantization::multi_vector::{BlockTransposed, MatRef, Standard};
/// use diskann_quantization::multi_vector::distance::QueryBlockTransposedRef;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let mat_ref = MatRef::new(Standard::new(2, 3).unwrap(), &data).unwrap();
/// let bt = BlockTransposed::<f32, 16>::from(mat_ref);
/// let query = QueryBlockTransposedRef::from(bt.as_view());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QueryBlockTransposedRef<'a>(pub BlockTransposedRef<'a, f32, 16>);

impl<'a> From<BlockTransposedRef<'a, f32, 16>> for QueryBlockTransposedRef<'a> {
    fn from(view: BlockTransposedRef<'a, f32, 16>) -> Self {
        Self(view)
    }
}

impl<'a> Deref for QueryBlockTransposedRef<'a> {
    type Target = BlockTransposedRef<'a, f32, 16>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// ── F32 kernel ───────────────────────────────────────────────────

/// Cache-aware micro-kernel for f32 queries and f32 documents.
///
/// Uses FMA (`mul_add_simd`) to accumulate inner products and `max_simd` to
/// reduce across document vectors. The micro-panel geometry is 16 × 4
/// (2 × f32x8 lanes × 4 broadcast unrolls).
pub(crate) struct F32Kernel;

// SAFETY: F32Kernel's `full_panel` and `remainder_dispatch` only access
// A_PANEL(16) * k query elements, UNROLL * k doc elements, and A_PANEL(16)
// scratch elements — all within the bounds guaranteed by `tiled_reduce`.
unsafe impl CacheAwareKernel for F32Kernel {
    type QueryElem = f32;
    type DocElem = f32;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

    #[inline(always)]
    unsafe fn full_panel(
        arch: diskann_wide::arch::Current,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // SAFETY: Caller guarantees pointer validity per CacheAwareKernel contract.
        unsafe { f32_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, r) }
    }

    #[inline(always)]
    unsafe fn remainder_dispatch(
        arch: diskann_wide::arch::Current,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        // SAFETY: Caller guarantees pointer validity per CacheAwareKernel contract.
        unsafe {
            match remainder {
                1 => f32_microkernel::<1>(arch, a, b, k, r),
                2 => f32_microkernel::<2>(arch, a, b, k, r),
                3 => f32_microkernel::<3>(arch, a, b, k, r),
                _ => {
                    debug_assert!(
                        false,
                        "unexpected remainder {remainder} for B_PANEL={}",
                        Self::B_PANEL
                    )
                }
            }
        }
    }
}

// ── f32 micro-kernel ─────────────────────────────────────────────

/// SIMD micro-kernel: processes 16 query rows × `UNROLL` document rows.
///
/// Accumulates inner products via FMA (`mul_add_simd`) into two `f32x8` register
/// tiles (covering 16 query rows), then reduces across the `UNROLL` document
/// lanes with `max_simd` and merges into the scratch buffer `r`.
///
/// # Safety
///
/// * `a_packed` must point to `A_PANEL(16) × k` contiguous `f32` values.
/// * `b` must point to `UNROLL` rows of `k` contiguous `f32` values.
/// * `r` must point to at least `A_PANEL(16)` writable `f32` values.
#[inline(always)]
unsafe fn f32_microkernel<const UNROLL: usize>(
    arch: diskann_wide::arch::Current,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
) where
    [f32s; UNROLL]: Reduce<Element = f32s>,
{
    let op = |x: f32s, y: f32s| x.max_simd(y);

    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32s::LANES;
    let a_stride_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: a_packed points to A_PANEL * k contiguous f32s (one micro-panel).
        // b points to UNROLL rows of k contiguous f32s each. All reads are in-bounds.
        unsafe {
            let a0 = f32s::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32s::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32s::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    // SAFETY: r points to at least A_PANEL = 16 writable f32s (2 × f32x8).
    let mut r0 = unsafe { f32s::load_simd(arch, r) };
    // SAFETY: r + f32s::LANES is within the same A_PANEL-sized scratch region.
    let mut r1 = unsafe { f32s::load_simd(arch, r.add(f32s::LANES)) };

    r0 = op(r0, p0.reduce(&op));
    r1 = op(r1, p1.reduce(&op));

    // SAFETY: r points to at least A_PANEL = 16 writable f32s (2 × f32x8).
    unsafe { r0.store_simd(r) };
    // SAFETY: r + f32s::LANES is within the same A_PANEL-sized scratch region.
    unsafe { r1.store_simd(r.add(f32s::LANES)) };
}

// ── Public f32 entry point ───────────────────────────────────────

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn cache_aware_chamfer_panic() {
    panic!(
        "cache_aware_chamfer: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed query (`a`) and
/// a row-major document matrix (`b`), writing per-query max similarities into `scratch`.
///
/// This is a thin wrapper over the generic [`tiled_reduce`] loop using the
/// [`F32Kernel`] micro-kernel.
///
/// # Arguments
///
/// * `arch` - The SIMD architecture to use (from [`diskann_wide::ARCH`]).
/// * `a` - Block-transposed query matrix view (GROUP=16, PACK=1).
/// * `b` - Row-major document matrix view.
/// * `scratch` - Mutable buffer of length [`BlockTransposedRef::available_rows()`].
///   Must be initialized to `f32::MIN` before the first call. On return, `scratch[i]`
///   contains the maximum inner product between query vector `i` and any document vector.
///
/// # Panics
///
/// Panics if `scratch.len() != a.available_rows()` or `a.ncols() != b.vector_dim()`.
pub fn cache_aware_chamfer(
    arch: diskann_wide::arch::Current,
    a: BlockTransposedRef<'_, f32, 16>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) {
    if scratch.len() != a.available_rows() || a.ncols() != b.vector_dim() {
        cache_aware_chamfer_panic();
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // SAFETY:
    // - a.as_ptr() is valid for a.available_rows() * k elements of f32.
    // - MatRef<Standard<f32>> stores nrows * ncols contiguous f32 elements at as_raw_ptr().
    // - scratch.len() == a.available_rows() (checked above).
    // - a.available_rows() is always a multiple of F32Kernel::A_PANEL (= 16 = GROUP).
    unsafe {
        tiled_reduce::<F32Kernel>(
            arch,
            a.as_ptr(),
            a.available_rows(),
            b.as_raw_ptr().cast::<f32>(),
            b_nrows,
            k,
            scratch,
        );
    }
}

// ── MaxSim / Chamfer trait implementations ───────────────────────

impl DistanceFunctionMut<QueryBlockTransposedRef<'_>, MatRef<'_, Standard<f32>>> for MaxSim<'_> {
    #[inline(always)]
    fn evaluate(&mut self, query: QueryBlockTransposedRef<'_>, doc: MatRef<'_, Standard<f32>>) {
        assert!(
            self.size() == query.nrows(),
            "scores buffer not right size: {} != {}",
            self.size(),
            query.nrows()
        );

        if doc.num_vectors() == 0 {
            // No document vectors — fill with MAX (no similarity found).
            self.scores_mut().fill(f32::MAX);
            return;
        }

        let scratch = self.scores_mut();
        scratch.fill(f32::MIN);

        // Extend scratch to available_rows if needed (scratch may be smaller
        // than available_rows due to padding).
        let available = query.available_rows();
        let nq = query.nrows();

        if available == nq {
            // No padding — scratch is exactly the right size.
            cache_aware_chamfer(diskann_wide::ARCH, *query, doc, scratch);
        } else {
            // Padding rows exist — need a larger scratch buffer.
            let mut padded_scratch = vec![f32::MIN; available];
            cache_aware_chamfer(diskann_wide::ARCH, *query, doc, &mut padded_scratch);
            scratch.copy_from_slice(&padded_scratch[..nq]);
        }

        // The kernel wrote max inner products (positive = more similar).
        // DiskANN convention: negate so that lower = better (distance semantics).
        for s in scratch.iter_mut() {
            *s = -*s;
        }
    }
}

impl PureDistanceFunction<QueryBlockTransposedRef<'_>, MatRef<'_, Standard<f32>>, f32> for Chamfer {
    #[inline(always)]
    fn evaluate(query: QueryBlockTransposedRef<'_>, doc: MatRef<'_, Standard<f32>>) -> f32 {
        if doc.num_vectors() == 0 {
            return 0.0;
        }

        let available = query.available_rows();
        let nq = query.nrows();

        let mut scratch = vec![f32::MIN; available];
        cache_aware_chamfer(diskann_wide::ARCH, *query, doc, &mut scratch);

        // Sum negated max similarities to get Chamfer distance.
        scratch.iter().take(nq).map(|&s| -s).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::BlockTransposed;
    use crate::multi_vector::distance::QueryMatRef;

    /// Helper to create a MatRef from raw data.
    fn make_query_mat(data: &[f32], nrows: usize, ncols: usize) -> MatRef<'_, Standard<f32>> {
        MatRef::new(Standard::new(nrows, ncols).unwrap(), data).unwrap()
    }

    /// Generate deterministic test data.
    fn make_test_data(len: usize, ceil: usize, shift: usize) -> Vec<f32> {
        (0..len).map(|v| ((v + shift) % ceil) as f32).collect()
    }

    /// Test cases: (num_queries, num_docs, dim).
    const TEST_CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),   // Single query, single doc
        (1, 5, 8),   // Single query, multiple docs
        (5, 1, 8),   // Multiple queries, single doc
        (3, 4, 16),  // General case
        (7, 7, 32),  // Square case
        (2, 3, 128), // Larger dimension
        (16, 4, 64), // Exact A_PANEL width
        (17, 4, 64), // One more than A_PANEL (remainder)
        (32, 5, 16), // Multiple full A-panels, remainder B-rows (5 % 4 = 1)
        (16, 6, 32), // Remainder B-rows (6 % 4 = 2)
        (16, 7, 32), // Remainder B-rows (7 % 4 = 3)
        (16, 8, 32), // No remainder B-rows (8 % 4 = 0)
    ];

    #[test]
    fn chamfer_matches_simple_kernel() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: simple kernel Chamfer
            let simple_query: QueryMatRef<_> = query_mat.into();
            let expected = Chamfer::evaluate(simple_query, doc);

            // Cache-aware: block-transposed query
            let bt = BlockTransposed::<f32, 16>::from(query_mat);
            let bt_query = QueryBlockTransposedRef::from(bt.as_view());
            let actual = Chamfer::evaluate(bt_query, doc);

            assert!(
                (actual - expected).abs() < 1e-2,
                "Chamfer mismatch for ({nq},{nd},{dim}): actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn max_sim_matches_simple_kernel() {
        for &(nq, nd, dim) in TEST_CASES {
            let query_data = make_test_data(nq * dim, dim, dim / 2);
            let doc_data = make_test_data(nd * dim, dim, dim);

            let query_mat = make_query_mat(&query_data, nq, dim);
            let doc = make_query_mat(&doc_data, nd, dim);

            // Reference: simple kernel MaxSim
            let mut expected_scores = vec![0.0f32; nq];
            let simple_query: QueryMatRef<_> = query_mat.into();
            let _ = MaxSim::new(&mut expected_scores)
                .unwrap()
                .evaluate(simple_query, doc);

            // Cache-aware: block-transposed query
            let bt = BlockTransposed::<f32, 16>::from(query_mat);
            let bt_query = QueryBlockTransposedRef::from(bt.as_view());
            let mut actual_scores = vec![0.0f32; nq];
            MaxSim::new(&mut actual_scores)
                .unwrap()
                .evaluate(bt_query, doc);

            for i in 0..nq {
                assert!(
                    (actual_scores[i] - expected_scores[i]).abs() < 1e-2,
                    "MaxSim[{i}] mismatch for ({nq},{nd},{dim}): actual={}, expected={}",
                    actual_scores[i],
                    expected_scores[i]
                );
            }
        }
    }

    #[test]
    fn chamfer_with_zero_docs_returns_zero() {
        let query_data = [1.0f32, 0.0, 0.0, 1.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let bt = BlockTransposed::<f32, 16>::from(query_mat);
        let bt_query = QueryBlockTransposedRef::from(bt.as_view());

        let doc = make_query_mat(&[], 0, 2);
        let result = Chamfer::evaluate(bt_query, doc);
        assert_eq!(result, 0.0);
    }

    #[test]
    #[should_panic(expected = "scores buffer not right size")]
    fn max_sim_panics_on_size_mismatch() {
        let query_data = [1.0f32, 2.0, 3.0, 4.0];
        let query_mat = make_query_mat(&query_data, 2, 2);
        let bt = BlockTransposed::<f32, 16>::from(query_mat);
        let bt_query = QueryBlockTransposedRef::from(bt.as_view());

        let doc = make_query_mat(&[1.0, 1.0], 1, 2);
        let mut scores = vec![0.0f32; 3]; // Wrong size
        MaxSim::new(&mut scores).unwrap().evaluate(bt_query, doc);
    }
}
