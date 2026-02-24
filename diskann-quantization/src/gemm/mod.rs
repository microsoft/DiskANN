/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Strategy
//!
//! We use a cache-aware loop tiling strategy to keep data resident in CPU caches as much
//! as possible. This section outlines the terminology used through the calculations here.
//!
//! For the packed matrix A - we have the following
//! ```text
//!    Memory Order
//! +----------------->
//!
//! +---------------------+
//! | A[0  ,0] A[1  ,0] A[2  ,0] ... A[P-1,0] |
//! | A[0  ,1] A[1  ,1] A[2  ,1] ... A[P-1,1] |
//! | A[0  ,2] A[1  ,2] A[2  ,2] ... A[P-1,2] |
//! |  ...        ... ...                     |
//! | A[0  ,K] A[1  ,K] A[2  ,K] ... A[P-1,K] |
//! +---------------------+
//! | A[P  ,0] A[P+1,0] A[P+2,0] ... A[2P-1,0] |
//! | A[P  ,1] A[P+1,1] A[P+2,1] ... A[2P-1,1] |
//! | A[P  ,2] A[P+1,2] A[P+2,2] ... A[2P-1,2] |
//! |  ...        ... ...                     |
//! | A[0  ,K] A[1  ,K] A[2  ,K] ... A[P  ,K] |
//! ```
//!

use crate::{
    algorithms::kmeans::BlockTranspose,
    multi_vector::{MatRef, Standard},
};
use diskann_wide::{SIMDMinMax, SIMDMulAdd, SIMDVector, arch::x86_64::V3};

diskann_wide::alias!(f32x8 = <V3>::f32x8);

// pub unsafe fn test_function(
//     arch: V3,
//     a_packed: *const f32,
//     b: *const f32,
//     k: usize,
//     r0: &mut f32x8,
//     r1: &mut f32x8,
// ) {
//     let op = |x: f32x8, y: f32x8| x.max_simd(y);
//     unsafe { microkernel(arch, a_packed, b, k, r0, r1, op) }
// }

#[inline(never)]
#[cold]
fn test_function_panic() {
    // assert_eq!(scratch.len(), a.nrows())
    //     || !a.nrows().is_multiple_of(32)
    //     || !b.nrows().is_multiple_of(4)
    //     || a.ncols() != b.ncols() {
    //     test_function_panic();
}

#[derive(Debug, Clone)]
struct Budgets {
    l1: usize,
    l2: usize,
}

impl Budgets {
    const fn new() -> Self {
        Self {
            l1: L1_B_TILE_BUDGET,
            l2: L2_A_TILE_BUDGET,
        }
    }
}

/// Approximate usable L1 data cache in bytes (conservative estimate).
const L1_CACHE: usize = 48_000;

/// Approximate usable L2 cache in bytes (conservative estimate).
const L2_CACHE: usize = 1_250_000;

/// Fraction of L2 reserved for the A tile. The remainder accommodates B streaming
/// traffic, partial_c (Strategy B), and incidental cache pollution.
const L2_A_TILE_BUDGET: usize = L2_CACHE / 2;

/// Fraction of L1 available for the B tile. The A micro-panel is subtracted at
/// runtime since it depends on K; this is the total L1 budget before that subtraction.
const L1_B_TILE_BUDGET: usize = L1_CACHE * 3 / 4;

/// Number of A-rows processed per micro-kernel invocation (SIMD width: 2 × f32x8 = 16 lanes).
const A_PANEL: usize = 16;

/// Number of B-rows processed per micro-kernel invocation (broadcast unroll factor).
const B_PANEL: usize = 4;

/// A plan for performing the reducing GEMM that is used when the contraction dimension `k`
/// is small enough that a micro-panel of `A` fits comfortably in L1 cache with room for
/// multiple micro-panels of `B`.
#[derive(Debug, Clone, Copy)]
struct FullReduce {
    /// The number of micro panels of `A` that make up a tile.
    ///
    /// NOTE: This value must be multiplied by the `A` micro-panel size to obtain the
    /// number of rows!
    a_panels: usize,

    /// The number of micro panels of `B` that make up a tile.
    ///
    /// NOTE: This value must be multiplied by the `B` micro-panel size to obtain the
    /// number of rows!
    b_panels: usize,
}

impl FullReduce {
    // Construct a new `Plan` for the given contraction dimension.
    //
    // The tile sizes will be such that:
    //
    // * One `A` tile fits withing the L2 budget.
    // * One tile of `B` plus one micro-panel of `A` fit within the L1 budget.
    // * The tile sizes computed are guaranteed to be multiples of the
    //
    // Budget overruns may occur for sufficiently large `k`. In these situations, a micro
    // panel of `A` on its own can overrun the L1 cache and a different strategy should
    // be used.
    fn new<T>(k: usize, budgets: Budgets) -> Self {
        let row_bytes = (k * std::mem::size_of::<T>()).max(std::mem::size_of::<T>());

        let a_panels = (budgets.l2 / (row_bytes * A_PANEL)).max(1);

        let a_panel_bytes = A_PANEL * row_bytes;
        let b_tile_budget = budgets.l1.saturating_sub(a_panel_bytes);
        let b_panels = (b_tile_budget / (row_bytes * B_PANEL)).max(1);

        Self {
            a_panels,
            b_panels,
        }
    }
}

// /// Compute MC (A tile height): sized so the packed A tile (MC × K × sizeof(T)) fits
// /// within [`L2_A_TILE_BUDGET`]. Result is rounded down to a multiple of [`A_PANEL`].
// fn compute_mc<T>(k: usize) -> usize {
//     let mc_raw = L2_A_TILE_BUDGET / (k * size_of::<T>());
//     let mc = (mc_raw / A_PANEL) * A_PANEL;
//     mc.max(A_PANEL)
// }
//
// /// Compute NC (B tile height): sized so the B tile (NC × K × sizeof(T)) plus the A
// /// micro-panel (A_PANEL × K × sizeof(T)) fit within [`L1_B_TILE_BUDGET`].
// /// Result is rounded down to a multiple of [`B_PANEL`].
// fn compute_nc<T>(k: usize) -> usize {
//     let a_panel_bytes = A_PANEL * k * size_of::<T>();
//     let budget = L1_B_TILE_BUDGET.saturating_sub(a_panel_bytes);
//     let nc_raw = budget / (k * size_of::<T>());
//     let nc = (nc_raw / B_PANEL) * B_PANEL;
//     nc.max(B_PANEL)
// }

pub fn test_function(
    arch: V3,
    a: &BlockTranspose<A_PANEL>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) {
    // Let's get this out of the way.
    if scratch.len() != a.available_rows()
        || a.ncols() != b.ncols()
    {
        test_function_panic();
    }

    let k = a.ncols();
    let plan = FullReduce::new::<f32>(k, Budgets::new());

    let op = |x: f32x8, y: f32x8| x.max_simd(y);

    // Precompute strides (in elements, not bytes).
    let a_panel_stride = A_PANEL * k;
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    // SAFETY: We trust the caller.
    let pa_end = unsafe { a.as_ptr().add(a.nrows() * k) };
    // SAFETY: We trust the caller.
    let pb_end = unsafe { b.as_ptr().add(b.nrows() * k) };

    // Compute once how many remainder B-rows there are after all full panels.
    let remainder = b.nrows() % B_PANEL;
    // Pointer to the start of remainder B-rows (past all full panels).
    let pb_full_end = unsafe { pb_end.sub(remainder * k) };

    unsafe {
        let mut pa_tile = a.as_ptr();
        let mut pr_tile = scratch.as_mut_ptr();

        // Loop 1: Tiles of `A`.
        while pa_tile < pa_end {
            let pa_tile_end = pa_tile.add(a_tile_stride.min(pa_end.offset_from_unsigned(pa_tile)));

            let mut pb_tile = b.as_ptr();

            // Loop 2: Full B-tiles (every panel in the tile is complete).
            while pb_tile + b_tile_stride <= pb_full_end {
                let pb_tile_end = pb_tile.add(b_tile_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
                pb_tile = pb_tile.add(b_tile_stride);
            }

            // Peeled last B-tile: contains remaining full panels + remainder rows.
            if pb_tile < pb_end {
                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3 (peeled): Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4 (peeled): Full B-panels in the last tile.
                    while pb_panel < pb_full_end {
                        microkernel::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1–3 leftover B-rows.
                    if remainder == 1 {
                        microkernel::<1, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 2 {
                        microkernel::<2, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 3 {
                        microkernel::<3, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
            }

            pa_tile = pa_tile.add(a_tile_stride);
            pr_tile = pr_tile.add(A_PANEL * plan.a_panels);
        }
    }
}

// TODO: Unroll loops.
#[inline(always)]
unsafe fn microkernel<const UNROLL: usize, Op>(
    arch: V3,
    a_packed: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
    reduce: Op,
) where
    Op: Fn(f32x8, f32x8) -> f32x8,
    [f32x8; UNROLL]: Reduce<Element = f32x8>,
{
    let mut p0 = [f32x8::default(arch); UNROLL];
    let mut p1 = [f32x8::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * f32x8::LANES;
    let a_stride_half = f32x8::LANES;

    for i in 0..k {
        unsafe {
            let a0 = f32x8::load_simd(arch, a_packed.add(a_stride * i));
            let a1 = f32x8::load_simd(arch, a_packed.add(a_stride * i + a_stride_half));

            for j in 0..UNROLL {
                let bj = f32x8::splat(arch, b.add(i + offsets[j]).read_unaligned());
                p0[j] = a0.mul_add_simd(bj, p0[j]);
                p1[j] = a1.mul_add_simd(bj, p1[j]);
            }
        }
    }

    let mut r0 = unsafe { f32x8::load_simd(arch, r) };
    let mut r1 = unsafe { f32x8::load_simd(arch, r.add(f32x8::LANES)) };

    r0 = reduce(r0, p0.reduce(&reduce));
    r1 = reduce(r1, p1.reduce(&reduce));

    unsafe { r0.store_simd(r) };
    unsafe { r1.store_simd(r.add(f32x8::LANES)) };
}

trait Reduce {
    type Element;
    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;
}

impl<T> Reduce for [T; 1]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, _f: F) -> T
    where
        F: Fn(T, T) ->  T{
        self[0]
    }
}

impl<T> Reduce for [T; 2]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(self[0], self[1])
    }
}

impl<T> Reduce for [T; 3]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(f(self[0], self[1]), self[2])
    }
}

impl<T> Reduce for [T; 4]
where
    T: Copy
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) ->  T{
        f(f(self[0], self[1]), f(self[2], self[3]))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::Mat;
    use crate::multi_vector::distance::{MaxSim, QueryMatRef};
    use diskann_utils::{ReborrowMut, lazy_format};
    use diskann_vector::DistanceFunctionMut;

    /// (M, N, K) test dimensions.
    const TEST_CASES: &[(usize, usize, usize)] = &[
        // Exact tile boundaries
        (32, 4, 16),
        (32, 8, 32),
        // Multiple A panels
        (64, 4, 32),
        // Multiple B panels
        (32, 16, 32),
        // Multiple tiles in both dimensions
        (256, 256, 64),
        // Large K
        (32, 8, 512),
        // Minimal K
        (32, 4, 1),
        // Many B rows
        (32, 128, 48),
        // Square
        (128, 128, 128),
        // N remainder = 1
        (32, 5, 32),
        // N remainder = 2
        (32, 6, 32),
        // N remainder = 3
        (32, 7, 32),
        // N = 1 (pure remainder, no full panels)
        (32, 1, 16),
        // N = 2
        (32, 2, 16),
        // N = 3
        (32, 3, 16),
        // M not multiple of A_PANEL (tests BlockTranspose padding)
        (17, 4, 32),
        (33, 8, 16),
        (1, 4, 8),
        // Both M and N non-aligned
        (17, 5, 32),
        (33, 7, 48),
        (1, 1, 16),
        // Larger with remainder
        (100, 100, 64),
    ];

    /// Fill a matrix with deterministic data: value = ((i * ncols + j + offset) % modulus) * scale
    fn fill_matrix(mat: &mut Mat<Standard<f32>>, modulus: usize, offset: usize, scale: f32) {
        let ncols = mat.as_view().ncols();
        for (i, row) in mat.reborrow_mut().rows_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((i * ncols + j + offset) % modulus) as f32 * scale;
            }
        }
    }

    /// Run the GEMM kernel and compare against MaxSim reference.
    ///
    /// `test_function` computes: scratch[i] = max_j(IP(a[i], b[j]))
    /// `MaxSim` computes:        scores[i]  = min_j(-IP(a[i], b[j])) = -max_j(IP(a[i], b[j]))
    ///
    /// So we expect: scratch[i] == -scores[i]
    fn run_test(m: usize, n: usize, k: usize) {
        let mut a = Mat::new(Standard::new(m, k).unwrap(), 0.0f32).unwrap();
        let mut b = Mat::new(Standard::new(n, k).unwrap(), 0.0f32).unwrap();
        fill_matrix(&mut a, 97, 0, 0.01);
        fill_matrix(&mut b, 79, 13, 0.01);

        // Reference: MaxSim scores
        let query: QueryMatRef<_> = a.as_view().into();
        let mut ref_scores = vec![0.0f32; m];
        MaxSim::new(&mut ref_scores)
            .unwrap()
            .evaluate(query, b.as_view())
            .unwrap();

        // Our kernel
        let a_packed = BlockTranspose::<16>::from_matrix_view(a.as_view().as_legacy_view());
        let mut scratch = vec![f32::NEG_INFINITY; a_packed.nrows()];
        test_function(diskann_wide::ARCH, &a_packed, b.as_view(), &mut scratch);

        // Compare: scratch[i] should equal -ref_scores[i]
        for i in 0..m {
            let expected = -ref_scores[i];
            let got = scratch[i];
            let diff = (got - expected).abs();
            let tolerance = 1e-3 * expected.abs().max(1.0);
            assert!(
                diff <= tolerance,
                "{}",
                lazy_format!(
                    "row {i}: got {got}, expected {expected}, diff {diff} \
                     [m={m}, n={n}, k={k}, tol={tolerance}]"
                ),
            );
        }
    }

    #[test]
    fn matches_max_sim_reference() {
        for &(m, n, k) in TEST_CASES {
            run_test(m, n, k);
        }
    }
}
