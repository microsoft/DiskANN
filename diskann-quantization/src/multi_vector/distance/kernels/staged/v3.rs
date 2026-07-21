// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 (AVX2+FMA) staged path: the store-out micro-kernel (Stage A), the
//! register-resident `max_simd` reducer (Stage C), and the f32 entry point.
//!
//! The Stage-A k-loop is byte-identical to the fused `super::super::f32::v3`
//! 16×4 kernel, so per-(query, doc) inner products are bit-identical. Only the
//! epilogue differs: instead of reducing the `UNROLL` accumulators and merging
//! into the score scratch, it **stores** them into `partial_buf` (A-major),
//! deferring the reduction to Stage C.
//!
//! The whole module is V3-specific; the experiment targets V3 only.

use diskann_wide::arch::Target3;
use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector};

use super::super::TileBudget;
use super::super::layouts::{self, DescribeLayout, Layout};
use super::driver::tiled_reduce_staged;
use super::maxsim::{Identity, MaxReducer, StagedRun};
use super::{Reducer, StagedConvert, StagedKernel};
use crate::alloc::ScopedAllocator;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

diskann_wide::alias!(f32s = <V3>::f32x8);

/// Zero-sized Stage-A kernel marker for the f32 staged path with block size
/// `GROUP`.
pub(crate) struct StagedF32Kernel<const GROUP: usize>;

// SAFETY: `full_panel`/`partial_panel` read A_PANEL(16) * k A elements and
// UNROLL * k B elements, and write UNROLL columns of A_PANEL(16) f32 into
// `partial` at stride `partial_b_stride` — all within the bounds the
// `StagedKernel` contract guarantees.
unsafe impl StagedKernel<V3> for StagedF32Kernel<16> {
    type Left = layouts::BlockTransposed<f32, 16>;
    type Right = layouts::RowMajor<f32>;
    type Acc = f32;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

    #[inline(always)]
    unsafe fn full_panel(
        arch: V3,
        a: *const f32,
        b: *const f32,
        k: usize,
        partial: *mut f32,
        partial_b_stride: usize,
    ) {
        // SAFETY: pointer validity per the `StagedKernel<V3>` contract.
        unsafe { store_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, partial, partial_b_stride) }
    }

    #[inline(always)]
    unsafe fn partial_panel(
        arch: V3,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        partial: *mut f32,
        partial_b_stride: usize,
    ) {
        // SAFETY: pointer validity per the `StagedKernel<V3>` contract.
        unsafe {
            match remainder {
                1 => store_microkernel::<1>(arch, a, b, k, partial, partial_b_stride),
                2 => store_microkernel::<2>(arch, a, b, k, partial, partial_b_stride),
                3 => store_microkernel::<3>(arch, a, b, k, partial, partial_b_stride),
                _ => unreachable!(
                    "unexpected remainder {remainder} for B_PANEL={}",
                    Self::B_PANEL
                ),
            }
        }
    }
}

/// V3 store-out micro-kernel: 16 A-rows × `UNROLL` B-rows.
///
/// The accumulation loop matches `super::super::f32::v3::f32_microkernel`
/// exactly (two `f32x8` register tiles, FMA, same splat/stride/unroll order).
/// The epilogue stores each B-column's 16 A-row accumulators into `partial`
/// (A-major: column `j` at `partial + j*b_stride`, as two `f32x8` halves).
///
/// # Safety
///
/// 1. `a_packed` points to `16 * k` contiguous `f32`.
/// 2. `b` points to `UNROLL` rows of `k` contiguous `f32`.
/// 3. `partial` is valid for `UNROLL` columns of 16 `f32` at stride `b_stride`.
#[inline(always)]
unsafe fn store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    partial: *mut f32,
    b_stride: usize,
) {
    let mut p0 = [f32s::default(arch); UNROLL];
    let mut p1 = [f32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32s::LANES;
    let a_stride_half = f32s::LANES;

    for i in 0..k {
        // SAFETY: preconditions 1 and 2; i < k and j < UNROLL.
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

    for j in 0..UNROLL {
        // SAFETY: precondition 3; column j occupies [j*b_stride, j*b_stride+16).
        unsafe {
            p0[j].store_simd(partial.add(j * b_stride));
            p1[j].store_simd(partial.add(j * b_stride + a_stride_half));
        }
    }
}

// ── Stage C: V3 SIMD max reducer ─────────────────────────────────

impl Reducer<V3> for MaxReducer {
    type Score = f32;
    type State = f32;

    #[inline(always)]
    fn init() -> f32 {
        f32::MIN
    }

    #[inline(always)]
    unsafe fn fold_block(
        arch: V3,
        state: *mut f32,
        scores: *const f32,
        a_panel: usize,
        valid_b_cols: usize,
        b_stride: usize,
    ) {
        let lanes = f32s::LANES;

        // The V3 staged kernel always folds a full A_PANEL = 16 = 2*LANES block:
        // two register-resident accumulators sweep the valid B-columns of
        // `partial_buf` in a single pass — the same access pattern the fused
        // kernel uses, just hoisted out of the inner B-loop.
        debug_assert_eq!(
            a_panel,
            2 * lanes,
            "V3 MaxReducer expects A_PANEL == 2*LANES"
        );

        // SAFETY: `state` is writable for 16; `scores` is valid for `valid_b_cols`
        // columns of 16 f32 at `b_stride`; only the valid columns are read (never
        // the stale padded remainder).
        unsafe {
            let mut a0 = f32s::load_simd(arch, state);
            let mut a1 = f32s::load_simd(arch, state.add(lanes));
            for c in 0..valid_b_cols {
                let col = scores.add(c * b_stride);
                a0 = a0.max_simd(f32s::load_simd(arch, col));
                a1 = a1.max_simd(f32s::load_simd(arch, col.add(lanes)));
            }
            a0.store_simd(state);
            a1.store_simd(state.add(lanes));
        }
    }
}

// ── Entry point ──────────────────────────────────────────────────

/// Compute per-A-row max inner product (block-transposed A query, row-major B
/// doc) into `state` via the staged pipeline. `state` (len ≥ `padded_nrows`) is
/// the caller's output, left holding the raw max-IP (the caller negates).
/// Transient scratch (`partial`, Stage-B region) is allocated internally from
/// `alloc` — the caller sizes nothing.
///
/// # Panics
///
/// Panics if `state.len() < a.padded_nrows()` or `a.ncols() != b.vector_dim()`.
pub(crate) fn max_ip_kernel_staged<const GROUP: usize>(
    arch: V3,
    a: BlockTransposedRef<'_, f32, GROUP>,
    b: MatRef<'_, Standard<f32>>,
    state: &mut [f32],
    alloc: ScopedAllocator<'_>,
    budget: TileBudget,
) where
    StagedF32Kernel<GROUP>: StagedKernel<V3, Acc = f32>,
    layouts::BlockTransposed<f32, GROUP>: StagedConvert<V3, <StagedF32Kernel<GROUP> as StagedKernel<V3>>::Left>
        + Layout<Element = f32>,
    layouts::RowMajor<f32>: StagedConvert<V3, <StagedF32Kernel<GROUP> as StagedKernel<V3>>::Right>
        + Layout<Element = f32>,
{
    let padded = a.padded_nrows();
    if state.len() < padded || a.ncols() != b.vector_dim() {
        max_ip_kernel_staged_panic(state.len(), padded, a.ncols(), b.vector_dim());
    }

    // A_PANEL must equal GROUP for block-transposed layout correctness.
    const { assert!(<StagedF32Kernel<GROUP> as StagedKernel<V3>>::A_PANEL == GROUP) }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // Empty contraction: every IP is 0 ⇒ max-IP is 0. Callers guarantee
    // b_nrows > 0 (the zero-doc case is short-circuited before reaching here).
    if k == 0 {
        state[..padded].fill(0.0);
        return;
    }

    let ca = a.layout();
    let cb = b.layout();
    let post = Identity::<f32>::new();

    // SAFETY:
    // - a.as_ptr() is valid for padded * k f32, and padded is a multiple of
    //   GROUP == A_PANEL (const-asserted above).
    // - b.as_slice() is num_vectors * vector_dim contiguous f32.
    // - state.len() >= padded (checked); the driver allocates its scratch from
    //   `alloc`.
    unsafe {
        tiled_reduce_staged::<V3, StagedF32Kernel<GROUP>, Identity<f32>, MaxReducer, _, _>(
            arch,
            &ca,
            &cb,
            &post,
            a.as_ptr(),
            padded,
            b.as_slice().as_ptr(),
            b_nrows,
            k,
            &mut state[..padded],
            alloc,
            budget,
        );
    }
}

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn max_ip_kernel_staged_panic(state_len: usize, padded: usize, a_ncols: usize, b_dim: usize) {
    panic!(
        "max_ip_kernel_staged: precondition failed: \
         state.len()={state_len} (expected >= {padded}), \
         a.ncols()={a_ncols}, b.vector_dim()={b_dim}"
    );
}

// ── Dispatch glue ────────────────────────────────────────────────

impl<const GROUP: usize>
    Target3<V3, (), BlockTransposedRef<'_, f32, GROUP>, MatRef<'_, Standard<f32>>, StagedRun<'_>>
    for StagedF32Kernel<GROUP>
where
    StagedF32Kernel<GROUP>: StagedKernel<V3, Acc = f32>,
    layouts::BlockTransposed<f32, GROUP>:
        StagedConvert<V3, <Self as StagedKernel<V3>>::Left> + Layout<Element = f32>,
    layouts::RowMajor<f32>:
        StagedConvert<V3, <Self as StagedKernel<V3>>::Right> + Layout<Element = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: V3,
        lhs: BlockTransposedRef<'_, f32, GROUP>,
        rhs: MatRef<'_, Standard<f32>>,
        scratch: StagedRun<'_>,
    ) {
        max_ip_kernel_staged(
            arch,
            lhs,
            rhs,
            scratch.state,
            scratch.alloc,
            TileBudget::default(),
        );
    }
}

#[cfg(test)]
mod tests {
    use diskann_wide::arch::x86_64::V3;

    use super::super::super::TileBudget;
    use super::super::super::f32::max_ip_kernel;
    use super::max_ip_kernel_staged;
    use crate::alloc::ScopedAllocator;
    use crate::multi_vector::{BlockTransposed, MatRef, Standard};

    // (a_nrows, b_nrows, dim): degenerate, zero-dim, zero-doc, prime k, A/B-panel
    // boundaries and every B-remainder class for V3 (B_PANEL=4), plus multi-tile.
    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 4),
        (1, 5, 8),
        (5, 1, 8),
        (5, 3, 5),
        (3, 2, 0), // zero dim
        (3, 0, 4), // zero docs
        (7, 7, 32),
        (2, 3, 128),
        (16, 4, 64), // one A-panel, no B remainder
        (17, 4, 64), // A-panel remainder
        (16, 5, 8),  // B remainder = 1
        (16, 6, 32), // B remainder = 2
        (16, 7, 32), // B remainder = 3
        (16, 8, 32),
        (32, 5, 16),
        (48, 3, 16),
        (8, 32, 128),
        (64, 32, 128),
        (32, 16, 256),
        (64, 1250, 512), // multi-tile B
    ];

    fn naive(a: &[f32], a_nrows: usize, b: &[f32], b_nrows: usize, k: usize) -> Vec<f32> {
        (0..a_nrows)
            .map(|i| {
                (0..b_nrows)
                    .map(|j| (0..k).map(|d| a[i * k + d] * b[j * k + d]).sum::<f32>())
                    .fold(f32::MIN, f32::max)
            })
            .collect()
    }

    /// The staged kernel must be **bit-identical** to the fused V3 kernel (same
    /// k-loop ⇒ same per-pair IP; `max` is order-independent), and within
    /// tolerance of the naive reference.
    #[test]
    fn staged_matches_fused_v3() {
        let Some(arch) = V3::new_checked() else {
            // No AVX2/FMA on this host; the staged V3 path cannot run.
            return;
        };

        for &(a_nrows, b_nrows, dim) in CASES {
            let a_data: Vec<f32> = (0..a_nrows * dim).map(|i| (i % 13 + 1) as f32).collect();
            let b_data: Vec<f32> = (0..b_nrows * dim).map(|i| (i % 7 + 1) as f32).collect();

            let a_mat = MatRef::new(Standard::new(a_nrows, dim).unwrap(), &a_data).unwrap();
            let a_bt = BlockTransposed::<f32, 16>::from_matrix_view(a_mat.as_matrix_view());
            let b_mat = MatRef::new(Standard::new(b_nrows, dim).unwrap(), &b_data).unwrap();

            let mut fused = vec![f32::MIN; a_bt.padded_nrows()];
            max_ip_kernel::<V3, f32, 16>(
                arch,
                a_bt.as_view(),
                b_mat,
                &mut fused,
                TileBudget::default(),
            );

            let mut state = vec![f32::MIN; a_bt.padded_nrows()];
            max_ip_kernel_staged::<16>(
                arch,
                a_bt.as_view(),
                b_mat,
                &mut state,
                ScopedAllocator::global(),
                TileBudget::default(),
            );

            let expected = naive(&a_data, a_nrows, &b_data, b_nrows, dim);
            for i in 0..a_nrows {
                assert_eq!(
                    state[i].to_bits(),
                    fused[i].to_bits(),
                    "staged != fused at row {i} for ({a_nrows},{b_nrows},{dim}): staged={}, fused={}",
                    state[i],
                    fused[i],
                );
                assert!(
                    (state[i] - expected[i]).abs() < 1e-6 * expected[i].abs().max(1.0),
                    "staged != naive at row {i} for ({a_nrows},{b_nrows},{dim}): staged={}, naive={}",
                    state[i],
                    expected[i],
                );
            }
        }
    }
}
