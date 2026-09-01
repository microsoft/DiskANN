/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! f32 MaxSim: each completed panel is folded directly into the running row maxima.

use core::num::NonZeroUsize;

use diskann_wide::arch::Scalar;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::V3;

use super::leaves::scalar as scalar_leaf;
use super::leaves::scalar::{A_PANEL as SC_A, B_PANEL as SC_B};
#[cfg(target_arch = "x86_64")]
use super::leaves::v3 as v3_leaf;
#[cfg(target_arch = "x86_64")]
use super::leaves::v3::{A_PANEL as V3_A, B_PANEL as V3_B};
use super::tiles::{
    BlockTransposedPanel, BlockTransposedTile, BlockTransposedWalk, RowMajorPanel, RowMajorTile,
    RowMajorWalk,
};
use super::{PanelOp, Plan, TileAt, TileBudget, TileWalk, drive};
use crate::bits::{Dynamic, Static};
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

/// Selects the f32 leaf for the active architecture and owns the running maxima.
pub(super) struct MaxOp<'o> {
    out: &'o mut [f32],
}

////////
// V3 //
////////

#[cfg(target_arch = "x86_64")]
impl<'a, 'b>
    PanelOp<V3, BlockTransposedPanel<'a, f32, V3_A>, RowMajorPanel<'b, f32, V3_B, Static<V3_B>>>
    for MaxOp<'_>
{
    #[inline(always)]
    fn process(
        &mut self,
        arch: V3,
        a: BlockTransposedPanel<'a, f32, V3_A>,
        b: RowMajorPanel<'b, f32, V3_B, Static<V3_B>>,
        a_panel: usize,
        _: usize,
    ) {
        let rows = &mut self.out.as_chunks_mut::<V3_A>().0[a_panel];
        v3_leaf::f32_max_microkernel::<V3_B, _>(arch, a, b, rows);
    }
}

#[cfg(target_arch = "x86_64")]
impl<'a, 'b> PanelOp<V3, BlockTransposedPanel<'a, f32, V3_A>, RowMajorPanel<'b, f32, V3_B, Dynamic>>
    for MaxOp<'_>
{
    #[inline(always)]
    fn process(
        &mut self,
        arch: V3,
        a: BlockTransposedPanel<'a, f32, V3_A>,
        b: RowMajorPanel<'b, f32, V3_B, Dynamic>,
        a_panel: usize,
        _: usize,
    ) {
        let rows = &mut self.out.as_chunks_mut::<V3_A>().0[a_panel];
        // Dispatch the runtime width onto a const the leaf can unroll for.
        match b.rows() {
            3 => v3_leaf::f32_max_microkernel::<3, _>(arch, a, b, rows),
            2 => v3_leaf::f32_max_microkernel::<2, _>(arch, a, b, rows),
            1 => v3_leaf::f32_max_microkernel::<1, _>(arch, a, b, rows),
            other => unreachable!("tail width {other} outside 1..{V3_B}"),
        }
    }
}

//////////////
// Emulated //
//////////////

impl<'a, 'b>
    PanelOp<Scalar, BlockTransposedPanel<'a, f32, SC_A>, RowMajorPanel<'b, f32, SC_B, Static<SC_B>>>
    for MaxOp<'_>
{
    #[inline(always)]
    fn process(
        &mut self,
        arch: Scalar,
        a: BlockTransposedPanel<'a, f32, SC_A>,
        b: RowMajorPanel<'b, f32, SC_B, Static<SC_B>>,
        a_panel: usize,
        _: usize,
    ) {
        let rows = &mut self.out.as_chunks_mut::<SC_A>().0[a_panel];
        scalar_leaf::f32_max_microkernel::<SC_B, _>(arch, a, b, rows);
    }
}

impl<'a, 'b>
    PanelOp<Scalar, BlockTransposedPanel<'a, f32, SC_A>, RowMajorPanel<'b, f32, SC_B, Dynamic>>
    for MaxOp<'_>
{
    #[inline(always)]
    fn process(
        &mut self,
        arch: Scalar,
        a: BlockTransposedPanel<'a, f32, SC_A>,
        b: RowMajorPanel<'b, f32, SC_B, Dynamic>,
        a_panel: usize,
        _: usize,
    ) {
        let rows = &mut self.out.as_chunks_mut::<SC_A>().0[a_panel];
        match b.rows() {
            1 => scalar_leaf::f32_max_microkernel::<1, _>(arch, a, b, rows),
            other => unreachable!("tail width {other} outside 1..{SC_B}"),
        }
    }
}

///////////
// Entry //
///////////

/// Plan and drive the streaming reduction.
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
/// Panics if `state` is shorter than A's padded row count: the operation indexes it one
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
    for<'o, 'a, 'b> MaxOp<'o>: PanelOp<Arch, BlockTransposedPanel<'a, f32, AR>, RowMajorPanel<'b, f32, BR, Static<BR>>>
        + PanelOp<Arch, BlockTransposedPanel<'a, f32, AR>, RowMajorPanel<'b, f32, BR, Dynamic>>,
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
    let plan = Plan::<AR, BR>::new(row_bytes, row_bytes, nd, 0, budget);
    let (a_walk, b_walk) = walks(plan);
    drive(arch, a_walk, b_walk, &mut MaxOp { out: state });
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
        for<'o, 'a, 'b> MaxOp<'o>: PanelOp<Arch, BlockTransposedPanel<'a, f32, AR>, RowMajorPanel<'b, f32, BR, Static<BR>>>
            + PanelOp<Arch, BlockTransposedPanel<'a, f32, AR>, RowMajorPanel<'b, f32, BR, Dynamic>>,
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_panel_reduction_ignores_nan_when_a_finite_maximum_exists() {
        let Some(arch) = diskann_wide::arch::x86_64::V3::new_checked() else {
            return;
        };

        let a = [1.0];
        let b = [0.0, 0.0, 1.0, f32::NAN, 0.0, 0.0, 0.0, 0.0];
        let query = MatRef::new(Standard::new(1, 1).unwrap(), &a).unwrap();
        let docs = MatRef::new(Standard::new(b.len(), 1).unwrap(), &b).unwrap();
        let bt = BlockTransposed::<f32, V3_A>::from_matrix_view(query.as_matrix_view());
        let mut state = vec![f32::MAX; bt.padded_nrows()];

        run(
            arch,
            docs.num_vectors(),
            1,
            TileBudget::default(),
            &mut state,
            |plan: Plan<V3_A, V3_B>| {
                (
                    BlockTransposedWalk::new(bt.as_view(), plan.a_panels),
                    RowMajorWalk::new(docs, plan.b_panels),
                )
            },
        );

        assert_eq!(state[0], 1.0);
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
