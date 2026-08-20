/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! f32 MaxSim: the accumulator is already the score, so the drain is a bare reduction.

use core::num::NonZeroUsize;
use core::ops::Range;

use diskann_wide::arch::Scalar;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::V3;

use super::leaves::scalar as scalar_leaf;
use super::leaves::scalar::{A_PANEL as SC_A, B_PANEL as SC_B};
#[cfg(target_arch = "x86_64")]
use super::leaves::v3 as v3_leaf;
#[cfg(target_arch = "x86_64")]
use super::leaves::v3::{A_PANEL as V3_A, B_PANEL as V3_B};
use super::strip::{Slot, Strip};
use super::tiles::{
    BlockTransposedPanel, BlockTransposedTile, BlockTransposedWalk, RowMajorPanel, RowMajorTile,
    RowMajorWalk,
};
use super::{Accumulate, Drain, Plan, TileAt, TileBudget, TileWalk, drive};
use crate::bits::{Dynamic, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

/// Selects the f32 leaf for whichever architecture is in play.
pub(super) struct Kernel;

/// Reduces a strip into the running per-A-row maxima, one entry per A row.
///
/// This is the module's single [`Drain`], which makes it the one place the accumulator's
/// axes acquire meaning. A strip column holds one B row's inner products against every A
/// row. A B row therefore arrives as a *row* of the input and lands as a *column* of the
/// accumulator.
///
/// Carries `nd` because a strip's trailing columns belong to B rows past the end of the
/// matrix, and nothing else here knows where that end is.
pub(super) struct RawMax<'o> {
    out: &'o mut [f32],
    nd: usize,
}

////////
// V3 //
////////

#[cfg(target_arch = "x86_64")]
impl<'a, 'b, 's>
    Accumulate<
        V3,
        BlockTransposedPanel<'a, f32, V3_A>,
        RowMajorPanel<'b, f32, V3_B, Static<V3_B>>,
        Slot<'s, f32, V3_A, V3_B>,
    > for Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: BlockTransposedPanel<'a, f32, V3_A>,
        b: RowMajorPanel<'b, f32, V3_B, Static<V3_B>>,
        out: Slot<'s, f32, V3_A, V3_B>,
    ) {
        v3_leaf::f32_store_microkernel::<V3_B, _>(arch, a, b, out);
    }
}

#[cfg(target_arch = "x86_64")]
impl<'a, 'b, 's>
    Accumulate<
        V3,
        BlockTransposedPanel<'a, f32, V3_A>,
        RowMajorPanel<'b, f32, V3_B, Dynamic>,
        Slot<'s, f32, V3_A, V3_B>,
    > for Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: BlockTransposedPanel<'a, f32, V3_A>,
        b: RowMajorPanel<'b, f32, V3_B, Dynamic>,
        out: Slot<'s, f32, V3_A, V3_B>,
    ) {
        // Dispatch the runtime width onto a const the leaf can unroll for.
        match b.rows() {
            3 => v3_leaf::f32_store_microkernel::<3, _>(arch, a, b, out),
            2 => v3_leaf::f32_store_microkernel::<2, _>(arch, a, b, out),
            1 => v3_leaf::f32_store_microkernel::<1, _>(arch, a, b, out),
            other => unreachable!("tail width {other} outside 1..{V3_B}"),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl Drain<V3, Strip<'_, f32, V3_A, V3_B>> for RawMax<'_> {
    #[inline(always)]
    fn drain(
        &mut self,
        arch: V3,
        scratch: &Strip<'_, f32, V3_A, V3_B>,
        a_panel: usize,
        b_panels: Range<usize>,
    ) {
        // A is padded to whole panels and its output cut never clamps. B is not, so the
        // live width does.
        let live = (b_panels.end * V3_B).min(self.nd) - b_panels.start * V3_B;
        let rows = &mut self.out.as_chunks_mut::<V3_A>().0[a_panel];
        v3_leaf::max_into_rows(arch, scratch.columns(live), rows);
    }
}

//////////////
// Emulated //
//////////////

impl<'a, 'b, 's>
    Accumulate<
        Scalar,
        BlockTransposedPanel<'a, f32, SC_A>,
        RowMajorPanel<'b, f32, SC_B, Static<SC_B>>,
        Slot<'s, f32, SC_A, SC_B>,
    > for Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: Scalar,
        a: BlockTransposedPanel<'a, f32, SC_A>,
        b: RowMajorPanel<'b, f32, SC_B, Static<SC_B>>,
        out: Slot<'s, f32, SC_A, SC_B>,
    ) {
        scalar_leaf::f32_store_microkernel::<SC_B, _>(arch, a, b, out);
    }
}

impl<'a, 'b, 's>
    Accumulate<
        Scalar,
        BlockTransposedPanel<'a, f32, SC_A>,
        RowMajorPanel<'b, f32, SC_B, Dynamic>,
        Slot<'s, f32, SC_A, SC_B>,
    > for Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: Scalar,
        a: BlockTransposedPanel<'a, f32, SC_A>,
        b: RowMajorPanel<'b, f32, SC_B, Dynamic>,
        out: Slot<'s, f32, SC_A, SC_B>,
    ) {
        match b.rows() {
            1 => scalar_leaf::f32_store_microkernel::<1, _>(arch, a, b, out),
            other => unreachable!("tail width {other} outside 1..{SC_B}"),
        }
    }
}

impl Drain<Scalar, Strip<'_, f32, SC_A, SC_B>> for RawMax<'_> {
    #[inline(always)]
    fn drain(
        &mut self,
        arch: Scalar,
        scratch: &Strip<'_, f32, SC_A, SC_B>,
        a_panel: usize,
        b_panels: Range<usize>,
    ) {
        let live = (b_panels.end * SC_B).min(self.nd) - b_panels.start * SC_B;
        let rows = &mut self.out.as_chunks_mut::<SC_A>().0[a_panel];
        scalar_leaf::max_into_rows(arch, scratch.columns(live), rows);
    }
}

///////////
// Entry //
///////////

/// Plan, allocate the strip, and drive.
///
/// `k` is the *physical* row length both walks stride by, so it must be A's padded column
/// count, not its logical one.
///
/// `walks` is built from the plan instead of being passed in. That way the empty-contraction
/// guard runs before any walk exists. A zero-length row would give a walk a zero stride.
///
/// On return `state` holds the per-A-row maximum inner product, one entry per padded A
/// row. Its incoming contents are ignored: seeding the max is this function's job,
/// not the caller's, because a caller that got it wrong would silently clamp the result
/// instead of failing.
///
/// # Panics
///
/// Panics if `state` is shorter than A's padded row count: the drain indexes it one
/// A-panel at a time.
pub(super) fn run<Arch, AW, BW, const AR: usize, const BR: usize>(
    arch: Arch,
    nd: usize,
    k: usize,
    budget: TileBudget,
    state: &mut [f32],
    walks: impl FnOnce(Plan<AR, BR>) -> (AW, BW),
) where
    Arch: Copy,
    AW: TileWalk + for<'a> TileAt<'a, Tile = BlockTransposedTile<'a, f32, AR>>,
    BW: TileWalk + for<'b> TileAt<'b, Tile = RowMajorTile<'b, f32, BR>>,
    Kernel: for<'a, 'b, 's> Accumulate<
            Arch,
            BlockTransposedPanel<'a, f32, AR>,
            RowMajorPanel<'b, f32, BR, Static<BR>>,
            Slot<'s, f32, AR, BR>,
        > + for<'a, 'b, 's> Accumulate<
            Arch,
            BlockTransposedPanel<'a, f32, AR>,
            RowMajorPanel<'b, f32, BR, Dynamic>,
            Slot<'s, f32, AR, BR>,
        >,
    for<'o, 'x> RawMax<'o>: Drain<Arch, Strip<'x, f32, AR, BR>>,
{
    // The identity for max.
    state.fill(f32::MIN);

    let Some(k) = NonZeroUsize::new(k) else {
        // Every inner product is the empty sum.
        if nd > 0 {
            state.fill(0.0);
        }
        return;
    };

    // Both sides are f32 rows of `k`: whatever the source element type, the leaves
    // consume f32. The clamp is an unreachable backstop, as in `tile_stride`.
    let row_bytes = NonZeroUsize::new(k.get() * size_of::<f32>()).unwrap_or(NonZeroUsize::MIN);
    let plan = Plan::<AR, BR>::new(row_bytes, row_bytes, nd, size_of::<f32>(), budget);
    let (a_walk, b_walk) = walks(plan);
    let mut buf = vec![0.0f32; plan.strip_len()];

    drive(
        arch,
        a_walk,
        b_walk,
        &mut Strip::new(&mut buf),
        &Kernel,
        &mut RawMax { out: state, nd },
    );
}

/// The f32 MaxSim entry, and with [`MaxIpF16`](super::MaxIpF16) one of the two places that
/// name the operands: the block-transposed A side is the query, the row-major B side the
/// documents.
///
/// Which leaf geometry applies follows from the architecture, and the block size of the
/// query must match the [`Target3`](diskann_wide::arch::Target3) impl selected.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MaxIp;

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target3<
        V3,
        (),
        BlockTransposedRef<'_, f32, V3_A>,
        MatRef<'_, Standard<f32>>,
        &mut [f32],
    > for MaxIp
{
    #[inline(always)]
    fn run(
        self,
        arch: V3,
        query: BlockTransposedRef<'_, f32, V3_A>,
        docs: MatRef<'_, Standard<f32>>,
        state: &mut [f32],
    ) {
        run(
            arch,
            docs.num_vectors(),
            query.padded_ncols(),
            TileBudget::default(),
            state,
            |plan: Plan<V3_A, V3_B>| {
                (
                    BlockTransposedWalk::new(query, plan.a_panels),
                    RowMajorWalk::new(docs, plan.b_panels),
                )
            },
        );
    }
}

impl
    diskann_wide::arch::Target3<
        Scalar,
        (),
        BlockTransposedRef<'_, f32, SC_A>,
        MatRef<'_, Standard<f32>>,
        &mut [f32],
    > for MaxIp
{
    #[inline(always)]
    fn run(
        self,
        arch: Scalar,
        query: BlockTransposedRef<'_, f32, SC_A>,
        docs: MatRef<'_, Standard<f32>>,
        state: &mut [f32],
    ) {
        run(
            arch,
            docs.num_vectors(),
            query.padded_ncols(),
            TileBudget::default(),
            state,
            |plan: Plan<SC_A, SC_B>| {
                (
                    BlockTransposedWalk::new(query, plan.a_panels),
                    RowMajorWalk::new(docs, plan.b_panels),
                )
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::BlockTransposed;

    fn sample(len: usize, phase: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7 + phase) % 23) as f32 - 11.0) / 4.0)
            .collect()
    }

    /// Reference in f64 so the comparison measures the kernel, not the reference.
    fn naive(a: &[f32], nq: usize, b: &[f32], nd: usize, k: usize) -> Vec<f64> {
        (0..nq)
            .map(|i| {
                (0..nd)
                    .map(|j| {
                        (0..k)
                            .map(|t| a[i * k + t] as f64 * b[j * k + t] as f64)
                            .sum::<f64>()
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }

    fn check<Arch, const AR: usize, const BR: usize>(
        arch: Arch,
        label: &str,
        (nq, nd, k): (usize, usize, usize),
        budget: TileBudget,
    ) where
        Arch: Copy,
        Kernel: for<'a, 'b, 's> Accumulate<
                Arch,
                BlockTransposedPanel<'a, f32, AR>,
                RowMajorPanel<'b, f32, BR, Static<BR>>,
                Slot<'s, f32, AR, BR>,
            > + for<'a, 'b, 's> Accumulate<
                Arch,
                BlockTransposedPanel<'a, f32, AR>,
                RowMajorPanel<'b, f32, BR, Dynamic>,
                Slot<'s, f32, AR, BR>,
            >,
        for<'o, 'x> RawMax<'o>: Drain<Arch, Strip<'x, f32, AR, BR>>,
    {
        let a = sample(nq * k, 0);
        let b = sample(nd * k, 5);
        let query = MatRef::new(Standard::new(nq, k).unwrap(), &a).unwrap();
        let docs = MatRef::new(Standard::new(nd, k).unwrap(), &b).unwrap();
        let bt = BlockTransposed::<f32, AR>::from_matrix_view(query.as_matrix_view());

        // Poisoned, not seeded: `run` owes the caller a full initialization, and every case
        // below would fail loudly if it skipped one.
        let mut state = vec![f32::MAX; bt.padded_nrows()];
        run(arch, nd, k, budget, &mut state, |plan: Plan<AR, BR>| {
            (
                BlockTransposedWalk::new(bt.as_view(), plan.a_panels),
                RowMajorWalk::new(docs, plan.b_panels),
            )
        });

        for (i, &expected) in naive(&a, nq, &b, nd, k).iter().enumerate() {
            let actual = state[i] as f64;
            let tol = 1e-5 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() < tol,
                "[{label}] row {i} of ({nq},{nd},{k}): actual={actual}, expected={expected}",
            );
        }
    }

    /// Shapes chosen to cross every A-panel boundary (8 and 16), every B-panel remainder
    /// class (mod 2 and mod 4), and both degenerate and prime contraction lengths.
    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (1, 1, 64),
        (5, 3, 5),
        (8, 2, 8),
        (8, 33, 127),
        (9, 4, 3),
        (15, 5, 16),
        (16, 4, 64),
        (16, 5, 128),
        (16, 6, 64),
        (16, 7, 256),
        (17, 9, 65),
        (32, 16, 256),
        (33, 1, 2),
        (64, 250, 64),
    ];

    #[test]
    fn scalar_matches_naive() {
        for &case in CASES {
            check::<_, SC_A, SC_B>(Scalar::new(), "scalar", case, TileBudget::default());
        }
    }

    /// Forces `a_panels == b_panels == 1`, so every panel is its own tile and the driver's
    /// cross-tile ordinal carry is exercised on every shape.
    #[test]
    fn scalar_matches_naive_one_panel_per_tile() {
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &case in CASES {
            check::<_, SC_A, SC_B>(Scalar::new(), "scalar/tiny", case, budget);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_matches_naive() {
        let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() else {
            return;
        };
        for &case in CASES {
            check::<_, V3_A, V3_B>(arch, "x86-64-v3", case, TileBudget::default());
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_matches_naive_one_panel_per_tile() {
        let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() else {
            return;
        };
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &case in CASES {
            check::<_, V3_A, V3_B>(arch, "x86-64-v3/tiny", case, budget);
        }
    }

    /// A zero-length contraction makes every inner product the empty sum, which the entry
    /// has to answer without building a zero-stride walk.
    #[test]
    fn empty_contraction() {
        let mut state = vec![f32::MAX; SC_A];
        run(
            Scalar::new(),
            3,
            0,
            TileBudget::default(),
            &mut state,
            |_: Plan<SC_A, SC_B>| -> (
                BlockTransposedWalk<'_, f32, SC_A>,
                RowMajorWalk<'_, f32, SC_B>,
            ) { unreachable!("no walk is built for an empty contraction") },
        );
        assert_eq!(state, vec![0.0; SC_A]);

        // With no B rows there is no maximum, so the identity stands and the caller
        // negates it into its empty-input sentinel.
        let mut state = vec![f32::MAX; SC_A];
        run(
            Scalar::new(),
            0,
            0,
            TileBudget::default(),
            &mut state,
            |_: Plan<SC_A, SC_B>| -> (
                BlockTransposedWalk<'_, f32, SC_A>,
                RowMajorWalk<'_, f32, SC_B>,
            ) { unreachable!("no walk is built for an empty contraction") },
        );
        assert_eq!(state, vec![f32::MIN; SC_A]);
    }
}
