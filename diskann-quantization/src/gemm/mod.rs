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
use diskann_utils::views::{Matrix, MatrixView};
use diskann_wide::{SIMDMinMax, SIMDDotProduct, SIMDReinterpret, SIMDMulAdd, SIMDVector, arch::x86_64::V3};

// Aliases
diskann_wide::alias!(f32x8 = <V3>::f32x8);
diskann_wide::alias!(i32x8 = <V3>::i32x8);
diskann_wide::alias!(i16x16 = <V3>::i16x16);
diskann_wide::alias!(i16x8 = <V3>::i16x8);
diskann_wide::alias!(u8x16 = <V3>::u8x16);

#[inline(never)]
#[cold]
fn test_function_panic() {
    panic!(
        "test_function: precondition failed (scratch.len != available_rows or dimension mismatch)"
    );
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

/// A "super-packed" block-transpose layout for u8 data.
///
/// Unlike [`BlockTranspose`], which stores one K-column of `N` rows per 16-byte group,
/// `SuperPacked` interleaves **K-pairs** so that adjacent bytes in memory belong to
/// consecutive K-values for the same logical row. This layout feeds directly into
/// `vpmaddwd` which multiplies and horizontally adds adjacent i16 pairs.
///
/// ## Memory layout
///
/// Within each panel of `N` rows, the data is arranged as:
///
/// ```text
/// K-pair 0, rows 0–7:   [r0_k0, r0_k1, r1_k0, r1_k1, ..., r7_k0, r7_k1]   (16 bytes)
/// K-pair 0, rows 8–15:  [r8_k0, r8_k1, r9_k0, r9_k1, ..., r15_k0, r15_k1]  (16 bytes)
/// K-pair 1, rows 0–7:   [r0_k2, r0_k3, r1_k2, r1_k3, ..., r7_k2, r7_k3]    (16 bytes)
/// K-pair 1, rows 8–15:  [r8_k2, r8_k3, ...]                                  (16 bytes)
/// ...
/// ```
///
/// When K is odd, the last pair is zero-padded: `[r_i_k_last, 0, ...]`.
///
/// Each K-pair group is `N * 2` bytes. The total panel size is `N * 2 * ceil(K/2)` bytes,
/// or equivalently `N * K` bytes rounded up to the next even K.
#[derive(Debug)]
pub struct SuperPacked<const N: usize> {
    data: Box<[u8]>,
    /// Number of logical rows (before rounding up to N).
    nrows: usize,
    /// Number of logical columns (K dimension).
    ncols: usize,
}

impl<const N: usize> SuperPacked<N> {
    /// Number of K-pair groups per panel.
    fn k_pairs(&self) -> usize {
        self.ncols.div_ceil(2)
    }

    /// Bytes per panel: `N * 2 * ceil(K/2)`.
    pub fn panel_bytes(&self) -> usize {
        N * 2 * self.k_pairs()
    }

    /// Number of logical rows rounded up to the panel size.
    pub fn available_rows(&self) -> usize {
        self.nrows.next_multiple_of(N)
    }

    /// Number of logical rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of logical columns (K dimension).
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Pointer to the start of the packed data.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Pack a row-major `Matrix<u8>` into super-packed layout.
    pub fn from_matrix(src: &MatrixView<'_, u8>) -> Self {
        let nrows = src.nrows();
        let ncols = src.ncols();
        let k_pairs = ncols.div_ceil(2);
        let num_panels = nrows.div_ceil(N);
        let panel_bytes = N * 2 * k_pairs;

        let mut data = vec![0u8; num_panels * panel_bytes];

        for panel in 0..num_panels {
            let panel_base = panel * panel_bytes;
            let row_base = panel * N;

            for r in 0..N {
                let global_row = row_base + r;
                if global_row >= nrows {
                    break;
                }

                for kp in 0..k_pairs {
                    let k0 = kp * 2;
                    let byte_offset = panel_base + kp * (N * 2) + r * 2;

                    data[byte_offset] = src[(global_row, k0)];
                    if k0 + 1 < ncols {
                        data[byte_offset + 1] = src[(global_row, k0 + 1)];
                    }
                }
            }
        }

        Self { data: data.into_boxed_slice(), nrows, ncols }
    }
}

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

        Self { a_panels, b_panels }
    }
}

pub fn test_function(
    arch: V3,
    a: &BlockTranspose<A_PANEL>,
    b: MatRef<'_, Standard<f32>>,
    scratch: &mut [f32],
) {
    // Let's get this out of the way.
    if scratch.len() != a.available_rows() || a.ncols() != b.ncols() {
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
            while pb_tile.wrapping_add(b_tile_stride) <= pb_full_end {
                let pb_tile_end = pb_tile.add(b_tile_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        microkernel_f32::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
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
                        microkernel_f32::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1–3 leftover B-rows.
                    if remainder == 1 {
                        microkernel_f32::<1, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 2 {
                        microkernel_f32::<2, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 3 {
                        microkernel_f32::<3, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
            }

            // NOTE: Use `wrapping_add` so we can still do this on the last iteration.
            pa_tile = pa_tile.wrapping_add(a_tile_stride);
            pr_tile = pr_tile.wrapping_add(A_PANEL * plan.a_panels);
        }
    }
}

pub fn test_function_u8(
    arch: V3,
    a: &SuperPacked<A_PANEL>,
    b: MatRef<'_, Standard<u8>>,
    scratch: &mut [i32],
) {
    // Let's get this out of the way.
    if scratch.len() != a.available_rows() || a.ncols() != b.ncols() {
        test_function_panic();
    }

    let k = a.ncols();
    let plan = FullReduce::new::<u8>(k, Budgets::new());

    let op = |x: i32x8, y: i32x8| {
        let x = x.to_array();
        let y = y.to_array();
        let z = core::array::from_fn(|i| x[i].max(y[i]));
        i32x8::from_array(arch, z)
    };

    // Precompute strides in bytes.
    // NOTE: For SuperPacked, a_panel_stride accounts for K-pair padding (odd k → +1 byte/row).
    let a_panel_stride = a.panel_bytes();
    let a_tile_stride = a_panel_stride * plan.a_panels;
    let b_panel_stride = B_PANEL * k;
    let b_tile_stride = b_panel_stride * plan.b_panels;

    // SAFETY: We trust the caller.
    let pa_end = unsafe { a.as_ptr().add(a.nrows().div_ceil(A_PANEL) * a_panel_stride) };
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
            while pb_tile.wrapping_add(b_tile_stride) <= pb_full_end {
                let pb_tile_end = pb_tile.add(b_tile_stride);

                let mut pa_panel = pa_tile;
                let mut pr_panel = pr_tile;

                // Loop 3: Micro-panels of `A`.
                while pa_panel < pa_tile_end {
                    let mut pb_panel = pb_tile;

                    // Loop 4: Micro-panels of `B` (all full, no remainder check).
                    while pb_panel < pb_tile_end {
                        microkernel_u8::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
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
                        microkernel_u8::<B_PANEL, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                        pb_panel = pb_panel.add(b_panel_stride);
                    }

                    // Remainder dispatch: 1–3 leftover B-rows.
                    if remainder == 1 {
                        microkernel_u8::<1, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 2 {
                        microkernel_u8::<2, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    } else if remainder == 3 {
                        microkernel_u8::<3, _>(arch, pa_panel, pb_panel, k, pr_panel, op);
                    }

                    pa_panel = pa_panel.add(a_panel_stride);
                    pr_panel = pr_panel.add(A_PANEL);
                }
            }

            // NOTE: Use `wrapping_add` so we can still do this on the last iteration.
            pa_tile = pa_tile.wrapping_add(a_tile_stride);
            pr_tile = pr_tile.wrapping_add(A_PANEL * plan.a_panels);
        }
    }
}

#[inline(always)]
unsafe fn microkernel_f32<const UNROLL: usize, Op>(
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

// NOTE: `a_packed` is actually SUPER packed.
//
// Expected layout for `a`:
//
// ```text
// |------------------------------------------------------------------------|--------|
// | a00 a01  a10 a11  a20 a21  a30 a31  a40 a41  a50 a51  a60 a61  a70 a71 |  next  |
// |------------------------------------------------------------------------|--------|
// | a02 a03  a12 a13  a22 a23  a32 a33  a42 a43  a52 a53  a62 a63  a72 a73 |  next  |
//                                    ...
// | a0K   0  a1K   0  a2K   0  a3K   0  a4K   0  a5K   0  a6K   0  a7K   0 |  next  |
// |------------------------------------------------------------------------|--------|
// ```
//
// * All `aij` are 8-bit values.
// * Each transposed "row" contains 32 bytes (treated as two blocks of 16-bytes).
// * Within each block, we group together two consecutive values for each logical row of `A`.
//   This is because our reduction kernel `SIMDDotProduct` adds together adjacent pairs.
// * When `K` is odd, the second byte of each pair is padded with zero.
//
// The matrix `b` may be simply a row-major byte array.
#[inline(always)]
unsafe fn microkernel_u8<const UNROLL: usize, Op>(
    arch: V3,
    a_packed: *const u8,
    b: *const u8,
    k: usize,
    r: *mut i32,
    reduce: Op,
) where
    Op: Fn(i32x8, i32x8) -> i32x8,
    [i32x8; UNROLL]: Reduce<Element = i32x8>,
{
    let mut p0 = [i32x8::default(arch); UNROLL];
    let mut p1 = [i32x8::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|i| k * i);

    let a_stride = 2 * i16x16::LANES;
    let a_stride_half = i16x16::LANES;

    // TODO: Peel last iteration.
    for i in 0..k / 2 {
        unsafe {
            // NOTE: this is loading 2 columns of 8 different rows of `A`.
            let a0: i16x16 = u8x16::load_simd(arch, a_packed.add(a_stride * i)).into();
            let a1: i16x16 =
                u8x16::load_simd(arch, a_packed.add(a_stride * i + a_stride_half)).into();

            for j in 0..UNROLL {
                // **CAREFUL**: We are really loading 2 element of `b` at a time, with a
                // 16-bit load. The dance with reinterpreting to `u8` and then casting ensures
                // that we get a final `i16x16` containing alternating elements of `b`.
                //
                // This ensures that the horizontal reduction step of `dot_simd` contains
                // proper elements of `b`.
                let bj: u8x16 = i16x8::splat(
                    arch,
                    b.add(2 * i + offsets[j]).cast::<i16>().read_unaligned(),
                )
                .reinterpret_simd();
                let bj: i16x16 = bj.into();

                p0[j] = p0[j].dot_simd(a0, bj);
                p1[j] = p1[j].dot_simd(a1, bj);
            }
        }
    }

    // If `k` is odd, then the super-packing of `a` is fine (zeros disappear), but we need
    // to ensure we only load one bytes of `b`.
    if !k.is_multiple_of(2) {
        unsafe {
            let last = k - 1;

            // NOTE: this is loading 2 columns of 8 different rows of `A`.
            let a0: i16x16 = u8x16::load_simd(arch, a_packed.add(a_stride * (k / 2))).into();
            let a1: i16x16 =
                u8x16::load_simd(arch, a_packed.add(a_stride * (k / 2) + a_stride_half)).into();

            for j in 0..UNROLL {
                let bj = i16x16::splat(
                    arch,
                    b.add(last + offsets[j]).read_unaligned().into(),
                );

                p0[j] = p0[j].dot_simd(a0, bj);
                p1[j] = p1[j].dot_simd(a1, bj);
            }
        }
    }

    let mut r0 = unsafe { i32x8::load_simd(arch, r) };
    let mut r1 = unsafe { i32x8::load_simd(arch, r.add(i32x8::LANES)) };

    r0 = reduce(r0, p0.reduce(&reduce));
    r1 = reduce(r1, p1.reduce(&reduce));

    unsafe { r0.store_simd(r) };
    unsafe { r1.store_simd(r.add(i32x8::LANES)) };
}

trait Reduce {
    type Element;
    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;
}

impl<T> Reduce for [T; 1]
where
    T: Copy,
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, _f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        self[0]
    }
}

impl<T> Reduce for [T; 2]
where
    T: Copy,
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(self[0], self[1])
    }
}

impl<T> Reduce for [T; 3]
where
    T: Copy,
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
        f(f(self[0], self[1]), self[2])
    }
}

impl<T> Reduce for [T; 4]
where
    T: Copy,
{
    type Element = T;

    #[inline(always)]
    fn reduce<F>(self, f: F) -> T
    where
        F: Fn(T, T) -> T,
    {
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
        // Minimal K
        (32, 4, 1),
        // Many B rows
        (32, 128, 48),
        // // Square
        // (128, 128, 128),
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
        let mut scratch = vec![f32::NEG_INFINITY; a_packed.available_rows()];
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
            println!("processing {m}, {n}, {k}");
            run_test(m, n, k);
        }
    }

    /// Scalar reference: for each row i of A, compute max over all rows j of B of dot(A[i], B[j]).
    fn reference_max_dot_u8(a: &MatrixView<'_, u8>, b: &MatrixView<'_, u8>) -> Vec<i32> {
        let m = a.nrows();
        let n = b.nrows();
        let k = a.ncols();
        assert_eq!(k, b.ncols());

        (0..m)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..k).map(|c| a[(i, c)] as i32 * b[(j, c)] as i32).sum::<i32>()
                    })
                    .max()
                    .unwrap_or(i32::MIN)
            })
            .collect()
    }

    fn fill_matrix_u8(mat: &mut Matrix<u8>, modulus: u8, offset: u8) {
        let ncols = mat.ncols();
        for i in 0..mat.nrows() {
            for j in 0..ncols {
                mat[(i, j)] = ((i * ncols + j) as u8).wrapping_add(offset) % modulus;
            }
        }
    }

    fn run_test_u8(m: usize, n: usize, k: usize) {
        let mut a = Matrix::new(0u8, m, k);
        let mut b = Matrix::new(0u8, n, k);
        fill_matrix_u8(&mut a, 97, 0);
        fill_matrix_u8(&mut b, 79, 13);

        // Scalar reference
        let ref_scores = reference_max_dot_u8(&a.as_view(), &b.as_view());

        // Our kernel
        let a_packed = SuperPacked::<16>::from_matrix(&a.as_view());
        let mut scratch = vec![i32::MIN; a_packed.available_rows()];

        let b_view = MatRef::new(Standard::new(n, k).unwrap(), b.as_slice()).unwrap();
        test_function_u8(diskann_wide::ARCH, &a_packed, b_view, &mut scratch);

        for i in 0..m {
            let expected = ref_scores[i];
            let got = scratch[i];
            assert_eq!(
                got, expected,
                "{}",
                lazy_format!(
                    "row {i}: got {got}, expected {expected} \
                     [m={m}, n={n}, k={k}]"
                ),
            );
        }
    }

    #[test]
    fn u8_matches_scalar_reference() {
        // Reuse shared cases plus u8-specific ones (odd K, K=1).
        let u8_extra: &[(usize, usize, usize)] = &[
            // Odd K (exercises K-pair padding in SuperPacked)
            (32, 4, 3),
            (32, 4, 7),
            (32, 8, 15),
            (16, 4, 1),
            // Odd K + M remainder
            (17, 4, 3),
            (33, 7, 5),
            // Odd K + N remainder
            (32, 5, 7),
            (32, 3, 9),
            // All three non-aligned
            (17, 5, 7),
            (1, 1, 1),
        ];

        for &(m, n, k) in TEST_CASES.iter().chain(u8_extra.iter()) {
            println!("processing u8 {m}, {n}, {k}");
            run_test_u8(m, n, k);
        }
    }
}
