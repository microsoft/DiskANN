// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 instantiation: block-transposed A, row-major B, running max. The accumulator is
//! already the score, so [`RawMax`] is a bare reduction — the degenerate [`Drain`].

use core::mem::size_of;

use diskann_wide::arch::x86_64::V3;

use super::arena::ResettableArena;
use super::leaves::{A_PANEL, B_PANEL};
use super::views::{DPanel, DocWalk, QPanel, QueryWalk};
use super::{Accumulate, At, Block, Drain, Plan, Strip, StripRef, TileBudget, drive, leaves};
use crate::alloc::{Poly, ScopedAllocator};
use crate::bits::{Dynamic, Static};
use crate::multi_vector::{BlockTransposed, Mat, MatRef, Standard};

// ── Kernel ───────────────────────────────────────────────────────

pub(crate) struct F32Kernel;

impl<'a, 'b, 'x>
    Accumulate<
        V3,
        QPanel<'a, f32, A_PANEL>,
        DPanel<'b, f32, B_PANEL, Static<B_PANEL>>,
        Block<'x, f32, A_PANEL, B_PANEL, Static<B_PANEL>>,
    > for F32Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, f32, A_PANEL>,
        b: DPanel<'b, f32, B_PANEL, Static<B_PANEL>>,
        mut out: Block<'x, f32, A_PANEL, B_PANEL, Static<B_PANEL>>,
    ) {
        // SAFETY: `a` is an A_PANEL×k block-transposed f32 block; `b` is B_PANEL rows
        // of k f32; `out` is B_PANEL columns of A_PANEL f32 at stride A_PANEL.
        unsafe {
            leaves::f32_store_microkernel::<B_PANEL>(
                arch,
                a.as_ptr(),
                b.as_ptr(),
                a.k(),
                out.as_mut_ptr(),
            );
        }
    }
}

impl<'a, 'b, 'x>
    Accumulate<
        V3,
        QPanel<'a, f32, A_PANEL>,
        DPanel<'b, f32, B_PANEL, Dynamic>,
        Block<'x, f32, A_PANEL, B_PANEL, Dynamic>,
    > for F32Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, f32, A_PANEL>,
        b: DPanel<'b, f32, B_PANEL, Dynamic>,
        mut out: Block<'x, f32, A_PANEL, B_PANEL, Dynamic>,
    ) {
        debug_assert_eq!(out.cols(), b.rows());
        debug_assert!(b.rows() < B_PANEL);
        let (ap, bp, op, k) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.k());
        // SAFETY: as the full-width impl, with a runtime width in 1..B_PANEL.
        unsafe {
            match b.rows() {
                3 => leaves::f32_store_microkernel::<3>(arch, ap, bp, k, op),
                2 => leaves::f32_store_microkernel::<2>(arch, ap, bp, k, op),
                1 => leaves::f32_store_microkernel::<1>(arch, ap, bp, k, op),
                other => unreachable!("tail width {other} out of 1..{B_PANEL}"),
            }
        }
    }
}

// ── Drain ────────────────────────────────────────────────────────

/// Running max over an output it owns, padded to whole A-panels by the caller.
pub(crate) struct RawMax<'o> {
    out: &'o mut [f32],
}

impl<'o> RawMax<'o> {
    fn new(out: &'o mut [f32]) -> Self {
        out.fill(f32::MIN);
        Self { out }
    }
}

impl Drain<V3, Strip<'_, f32, A_PANEL, B_PANEL>> for RawMax<'_> {
    #[inline(always)]
    fn drain(&mut self, arch: V3, acc: StripRef<'_, f32, A_PANEL>, at: At) {
        let out = &mut self.out[at.a_panel * A_PANEL..][..A_PANEL];
        // SAFETY: `out` is A_PANEL f32; `acc` is `cols` columns of A_PANEL f32.
        unsafe { leaves::fold_strip(arch, out.as_mut_ptr(), acc.as_ptr(), acc.cols()) }
    }
}

// ── Public entry ─────────────────────────────────────────────────

/// A prepared f32 query set for the paneled driver (V3/AVX2).
pub struct PaneledF32Query {
    query: BlockTransposed<f32, A_PANEL>,
    dim: usize,
    arch: V3,
    state: Vec<f32>,
    arena: ResettableArena,
}

impl PaneledF32Query {
    /// `None` if AVX2 (V3) is unavailable.
    #[allow(clippy::expect_used)]
    pub fn build(query: MatRef<'_, Standard<f32>>) -> Option<Self> {
        let arch = V3::new_checked()?;
        let dim = query.vector_dim();
        let query = BlockTransposed::<f32, A_PANEL>::from_matrix_view(query.as_matrix_view());
        let padded = query.padded_nrows();

        // The planner keeps `A_PANEL · acc_bytes · b_tile_rows` inside `l1_b`, so one
        // page of headroom bounds the single strip for any k.
        let arena = ResettableArena::with_capacity(TileBudget::default().l1_b + 4096)
            .expect("arena allocation");

        Some(Self {
            query,
            dim,
            arch,
            state: vec![f32::MIN; padded],
            arena,
        })
    }

    pub fn is_supported() -> bool {
        V3::new_checked().is_some()
    }

    pub fn num_vectors(&self) -> usize {
        self.query.nrows()
    }

    /// Per-query max inner product (the MaxSim similarity) against `docs`.
    ///
    /// # Panics
    ///
    /// If `scores.len() != self.num_vectors()` or the logical dims differ.
    pub fn compute_max_sim(&mut self, docs: &PaneledF32Docs, scores: &mut [f32]) {
        self.compute(docs, scores, TileBudget::default());
    }

    #[allow(clippy::expect_used)]
    fn compute(&mut self, docs: &PaneledF32Docs, scores: &mut [f32], budget: TileBudget) {
        let nq = self.query.nrows();
        assert_eq!(scores.len(), nq, "scores length must equal query count");
        assert_eq!(self.dim, docs.data.vector_dim(), "query dim != doc dim");

        let k = self.query.padded_ncols();
        let padded = self.query.padded_nrows();
        assert_eq!(docs.data.vector_dim(), k, "doc row stride must equal k");

        self.arena.reset();
        let row_bytes = k * size_of::<f32>();
        let plan = Plan::<A_PANEL, B_PANEL>::new(row_bytes, row_bytes, size_of::<f32>(), budget);
        let strip_len = plan.strip_len();
        let mut buf =
            Poly::<[f32], _>::new_uninit_slice(strip_len, ScopedAllocator::new(&self.arena))
                .expect("strip fits the arena");
        let mut scratch = Strip::<f32, A_PANEL, B_PANEL>::from_uninit(&mut buf, strip_len);
        drive::<_, _, _, _, Strip<'_, f32, A_PANEL, B_PANEL>, _>(
            self.arch,
            QueryWalk::new(self.query.as_view(), plan.a_panels),
            DocWalk::<f32, B_PANEL>::new(docs.data.as_view(), plan.b_panels),
            &F32Kernel,
            &mut scratch,
            &mut RawMax::new(&mut self.state[..padded]),
        );

        scores.copy_from_slice(&self.state[..nq]);
    }
}

/// A prepared f32 document set, kept row-major.
pub struct PaneledF32Docs {
    data: Mat<Standard<f32>>,
}

impl PaneledF32Docs {
    pub fn build(docs: MatRef<'_, Standard<f32>>) -> Self {
        let mut src = docs.as_slice().iter().copied();
        Self {
            data: Mat::from_fn(*docs.repr(), || src.next().unwrap_or_default()),
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.data.num_vectors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: u64, idx: usize) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(idx as u64)
            .wrapping_mul(1442695040888963407);
        ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn reference(q: &[f32], nq: usize, d: &[f32], nd: usize, dim: usize) -> Vec<f32> {
        (0..nq)
            .map(|i| {
                (0..nd)
                    .map(|j| {
                        (0..dim)
                            .map(|c| q[i * dim + c] * d[j * dim + c])
                            .sum::<f32>()
                    })
                    .fold(f32::MIN, f32::max)
            })
            .collect()
    }

    /// (nq, nd, dim): every B-remainder class, an A-panel remainder (17), a
    /// multi-B-tile case, and odd dims.
    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 64),
        (5, 3, 5),
        (16, 4, 64),
        (16, 5, 128),
        (16, 6, 64),
        (16, 7, 256),
        (17, 9, 65),
        (32, 16, 256),
        (64, 1250, 64),
        (8, 33, 127),
    ];

    #[allow(clippy::expect_used)]
    fn run(nq: usize, nd: usize, dim: usize, seed: u64, budget: TileBudget) {
        let q: Vec<f32> = (0..nq * dim).map(|i| rnd(seed, i)).collect();
        let d: Vec<f32> = (0..nd * dim).map(|i| rnd(seed + 1, i)).collect();

        let q_mat = MatRef::new(Standard::<f32>::new(nq, dim).expect("nq×dim"), q.as_slice())
            .expect("query slice");
        let d_mat = MatRef::new(Standard::<f32>::new(nd, dim).expect("nd×dim"), d.as_slice())
            .expect("doc slice");

        let mut query = PaneledF32Query::build(q_mat).expect("V3 checked by caller");
        let docs = PaneledF32Docs::build(d_mat);
        let mut got = vec![0.0f32; nq];
        query.compute(&docs, &mut got, budget);

        let want = reference(&q, nq, &d, nd, dim);
        for i in 0..nq {
            assert!(
                (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                "({nq},{nd},{dim}) row {i}: paneled-f32 {} != reference {}",
                got[i],
                want[i],
            );
        }
    }

    #[test]
    fn paneled_f32_matches_reference() {
        if V3::new_checked().is_none() {
            return;
        }
        for &(nq, nd, dim) in CASES {
            run(nq, nd, dim, 1, TileBudget::default());
        }
    }

    /// A tiny budget clamps the planner to one panel per tile, forcing multiple A- and
    /// B-tiles — the cross-tile offset carry the default budget never reaches.
    #[test]
    fn paneled_f32_multi_tile_tiny_budget() {
        if V3::new_checked().is_none() {
            return;
        }
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &(nq, nd, dim) in &[(48usize, 22usize, 64usize), (33, 37, 128), (35, 19, 65)] {
            run(nq, nd, dim, 3, budget);
        }
    }
}
