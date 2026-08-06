// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! 4-bit MinMax instantiation. The interesting half is [`MinMaxMax`]: dequant needs
//! per-vector metadata indexed by [`Region`] and the reduction needs the dequantized
//! score, so both ride in one [`Drain`] and the score never reaches memory.

use core::mem::size_of;
use std::num::NonZeroUsize;

use diskann_utils::ReborrowMut;
use diskann_wide::arch::x86_64::V3;

use super::arena::ResettableArena;
use super::leaves::{A_PANEL, B_PANEL};
use super::views::{DPanel, DocWalk, QPanel, QueryWalk};
use super::{Accumulate, Block, Drain, Plan, Region, Strip, TileBudget, drive, leaves};
use crate::CompressInto;
use crate::algorithms::Transform;
use crate::algorithms::transforms::NullTransform;
use crate::alloc::{Poly, ScopedAllocator};
use crate::bits::{Dynamic, Static};
use crate::minmax::{MinMaxCompensation, MinMaxMeta, MinMaxQuantizer};
use crate::multi_vector::{BlockTransposed, Defaulted, Mat, MatRef, Standard};
use crate::num::Positive;

// ── Kernel ───────────────────────────────────────────────────────

pub(crate) struct I8Kernel;

impl<'a, 'b, 'x>
    Accumulate<
        V3,
        QPanel<'a, i16, A_PANEL>,
        DPanel<'b, u8, B_PANEL, Static<B_PANEL>>,
        Block<'x, i32, A_PANEL, B_PANEL>,
    > for I8Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, i16, A_PANEL>,
        b: DPanel<'b, u8, B_PANEL, Static<B_PANEL>>,
        out: Block<'x, i32, A_PANEL, B_PANEL>,
    ) {
        leaves::int_store_microkernel::<B_PANEL, _>(arch, a, b, out);
    }
}

impl<'a, 'b, 'x>
    Accumulate<
        V3,
        QPanel<'a, i16, A_PANEL>,
        DPanel<'b, u8, B_PANEL, Dynamic>,
        Block<'x, i32, A_PANEL, B_PANEL>,
    > for I8Kernel
{
    #[inline(always)]
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, i16, A_PANEL>,
        b: DPanel<'b, u8, B_PANEL, Dynamic>,
        out: Block<'x, i32, A_PANEL, B_PANEL>,
    ) {
        // The leaf checks that the width it unrolls for is the width `b` actually has.
        match b.rows() {
            3 => leaves::int_store_microkernel::<3, _>(arch, a, b, out),
            2 => leaves::int_store_microkernel::<2, _>(arch, a, b, out),
            1 => leaves::int_store_microkernel::<1, _>(arch, a, b, out),
            other => unreachable!("tail width {other} out of 1..{B_PANEL}"),
        }
    }
}

// ── Drain ────────────────────────────────────────────────────────

/// Fused 4-bit MinMax dequant + running max: rewrites each raw integer dot into the
/// MinMax inner product using per-vector `a`/`b`/`n` metadata, then folds it straight
/// into the output — the score never reaches memory.
pub(crate) struct MinMaxMax<'m, 'o> {
    query_meta: &'m [MinMaxCompensation],
    doc_meta: &'m [MinMaxCompensation],
    out: &'o mut [f32],
    dim: f32,
}

impl<'m, 'o> MinMaxMax<'m, 'o> {
    fn new(
        query_meta: &'m [MinMaxCompensation],
        doc_meta: &'m [MinMaxCompensation],
        out: &'o mut [f32],
        dim: f32,
    ) -> Self {
        out.fill(f32::MIN);
        Self {
            query_meta,
            doc_meta,
            out,
            dim,
        }
    }
}

impl Drain<V3, Strip<'_, i32, A_PANEL, B_PANEL>> for MinMaxMax<'_, '_> {
    #[inline(always)]
    fn drain(&mut self, arch: V3, acc: &Strip<'_, i32, A_PANEL, B_PANEL>, region: Region) {
        let (a, b) = (region.a, region.b);
        // A-indexed buffers here are padded to whole panels, so the stride is ours to
        // state — which is what makes the leaf's one-compensation-per-row check hold.
        let q = &self.query_meta[a.start..][..A_PANEL];
        let d = &self.doc_meta[b.range()];
        let dim = self.dim;
        let out = &mut self.out[a.start..][..A_PANEL];
        leaves::score_fold_strip(arch, acc, out, b.len(), q, d, dim);
    }
}

// ── Public entry ─────────────────────────────────────────────────

/// Quantize an f32 multi-vector to 4-bit MinMax (Null transform, scale 1.0).
#[allow(clippy::expect_used)]
fn quantize(input: MatRef<'_, Standard<f32>>) -> Mat<MinMaxMeta<4>> {
    let (n, dim) = (input.num_vectors(), input.vector_dim());
    let q = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(
            NonZeroUsize::new(dim).expect("dimension must be non-zero"),
        )),
        Positive::new(1.0).expect("1.0 is positive"),
    );
    let mut out: Mat<MinMaxMeta<4>> =
        Mat::new(MinMaxMeta::new(n, dim), Defaulted).expect("MinMaxMeta allocation");
    q.compress_into(input, out.reborrow_mut())
        .expect("input must be finite");
    out
}

/// A prepared 4-bit MinMax query set for the paneled driver (V3/AVX2).
pub struct PaneledQuantQuery {
    query: BlockTransposed<i16, A_PANEL, 2>,
    meta: Vec<MinMaxCompensation>,
    dim: usize,
    arch: V3,
    state: Vec<f32>,
    arena: ResettableArena,
}

impl PaneledQuantQuery {
    /// `None` if AVX2 (V3) is unavailable.
    #[allow(clippy::expect_used)]
    pub fn build(query: MatRef<'_, Standard<f32>>) -> Option<Self> {
        let arch = V3::new_checked()?;
        let (nq, dim) = (query.num_vectors(), query.vector_dim());
        let q_mat = quantize(query);

        let mut codes = vec![0i16; nq * dim];
        for r in 0..nq {
            let row = q_mat.get_row(r).expect("row < nq");
            for j in 0..dim {
                codes[r * dim + j] = i16::from(row.vector().get(j).expect("col < dim") as u8);
            }
        }
        let view = MatRef::new(Standard::<i16>::new(nq, dim).expect("nq×dim"), &codes)
            .expect("code slice");
        let query = BlockTransposed::<i16, A_PANEL, 2>::from_matrix_view(view.as_matrix_view());

        let padded = query.padded_nrows();
        let mut meta = vec![MinMaxCompensation::default(); padded];
        for (r, m) in meta.iter_mut().enumerate().take(nq) {
            *m = q_mat.get_row(r).expect("row < nq").meta();
        }

        // The planner keeps `A_PANEL · acc_bytes · b_tile_rows` inside `l1_b`, so one
        // page of headroom bounds the single strip for any k.
        let arena = ResettableArena::with_capacity(TileBudget::default().l1_b + 4096)
            .expect("arena allocation");

        Some(Self {
            query,
            meta,
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

    /// Per-query min distance (`= -max_d IP`) against `docs`.
    ///
    /// # Panics
    ///
    /// If `scores.len() != self.num_vectors()` or the logical dims differ.
    pub fn compute_max_sim(&mut self, docs: &PaneledQuantDocs, scores: &mut [f32]) {
        self.compute(docs, scores, TileBudget::default());
    }

    #[allow(clippy::expect_used)]
    fn compute(&mut self, docs: &PaneledQuantDocs, scores: &mut [f32], budget: TileBudget) {
        let nq = self.query.nrows();
        assert_eq!(scores.len(), nq, "scores length must equal query count");
        assert_eq!(self.dim, docs.dim, "query dim != doc dim");

        let k = self.query.padded_ncols();
        let padded = self.query.padded_nrows();
        assert_eq!(docs.codes.vector_dim(), k, "doc row stride must equal k");

        self.arena.reset();
        let plan = Plan::<A_PANEL, B_PANEL>::new(
            k * size_of::<i16>(),
            k * size_of::<u8>(),
            size_of::<i32>(),
            budget,
        );
        let mut drain = MinMaxMax::new(
            &self.meta,
            &docs.meta,
            &mut self.state[..padded],
            self.dim as f32,
        );
        let strip_len = plan.strip_len();
        let mut buf =
            Poly::<[i32], _>::new_uninit_slice(strip_len, ScopedAllocator::new(&self.arena))
                .expect("strip fits the arena");
        let mut scratch = Strip::<i32, A_PANEL, B_PANEL>::from_uninit(&mut buf, strip_len);
        drive::<_, _, _, _, Strip<'_, i32, A_PANEL, B_PANEL>, _>(
            self.arch,
            QueryWalk::new(self.query.as_view(), plan.a_panels),
            DocWalk::<u8, B_PANEL>::new(docs.codes.as_view(), plan.b_panels),
            &I8Kernel,
            &mut scratch,
            &mut drain,
        );

        for (s, &raw) in scores.iter_mut().zip(self.state.iter()) {
            *s = -raw;
        }
    }
}

/// A prepared 4-bit MinMax document set. Codes are row-major with rows padded to an
/// even length, which the integer microkernel requires; metadata is kept alongside.
pub struct PaneledQuantDocs {
    codes: Mat<Standard<u8>>,
    meta: Vec<MinMaxCompensation>,
    dim: usize,
}

impl PaneledQuantDocs {
    #[allow(clippy::expect_used)]
    pub fn build(docs: MatRef<'_, Standard<f32>>) -> Self {
        let (nv, dim) = (docs.num_vectors(), docs.vector_dim());
        let repr = Standard::<u8>::new(nv, dim.next_multiple_of(2)).expect("codes fit in memory");
        let d_mat = quantize(docs);

        let mut codes = Mat::from_fn(repr, || 0u8);
        let mut meta = Vec::with_capacity(nv);
        for r in 0..nv {
            let row = d_mat.get_row(r).expect("row < nv");
            let dst = codes.get_row_mut(r).expect("row < nv");
            for (j, d) in dst.iter_mut().take(dim).enumerate() {
                *d = row.vector().get(j).expect("col < dim") as u8;
            }
            meta.push(row.meta());
        }
        Self { codes, meta, dim }
    }

    pub fn num_vectors(&self) -> usize {
        self.codes.num_vectors()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_vector::distance::{MaxSim, QueryMatRef};
    use diskann_vector::DistanceFunctionMut;

    fn rnd(seed: u64, idx: usize) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(idx as u64)
            .wrapping_mul(1442695040888963407);
        ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    #[allow(clippy::expect_used)]
    fn reference(q: &[f32], nq: usize, d: &[f32], nd: usize, dim: usize) -> Vec<f32> {
        let quantize = |data: &[f32], n: usize| -> Mat<MinMaxMeta<4>> {
            let input =
                MatRef::new(Standard::<f32>::new(n, dim).expect("n×dim"), data).expect("slice");
            super::quantize(input)
        };
        let q_mat = quantize(q, nq);
        let d_mat = quantize(d, nd);
        let query: QueryMatRef<_> = q_mat.as_view().into();
        let mut out = vec![0.0f32; nq];
        MaxSim::new(&mut out).evaluate(query, d_mat.as_view());
        out
    }

    /// (nq, nd, dim): every B-remainder class, an A-panel remainder (17), a
    /// multi-B-tile case, and the odd-dim even-K contract.
    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 64),
        (5, 1, 128),
        (16, 4, 64),
        (16, 5, 128),
        (16, 6, 64),
        (16, 7, 256),
        (17, 9, 64),
        (32, 16, 256),
        (64, 1250, 64),
        (5, 3, 63),
        (17, 9, 65),
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

        let mut query = PaneledQuantQuery::build(q_mat).expect("V3 checked by caller");
        let docs = PaneledQuantDocs::build(d_mat);
        let mut got = vec![0.0f32; nq];
        query.compute(&docs, &mut got, budget);

        let want = reference(&q, nq, &d, nd, dim);
        for i in 0..nq {
            assert!(
                (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                "({nq},{nd},{dim}) row {i}: paneled-i8 {} != reference {}",
                got[i],
                want[i],
            );
        }
    }

    #[test]
    fn paneled_i8_matches_minmax_reference() {
        if V3::new_checked().is_none() {
            return;
        }
        for &(nq, nd, dim) in CASES {
            run(nq, nd, dim, 1, TileBudget::default());
        }
    }

    /// A tiny budget clamps the planner to one panel per tile, forcing multiple A- and
    /// B-tiles — the cross-tile metadata offsets the default budget never reaches.
    #[test]
    fn paneled_i8_multi_tile_tiny_budget() {
        if V3::new_checked().is_none() {
            return;
        }
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &(nq, nd, dim) in &[(48usize, 22usize, 64usize), (33, 37, 128), (35, 19, 65)] {
            run(nq, nd, dim, 3, budget);
        }
    }

    /// Arena reuse across differently-sized doc sets stays correct.
    #[test]
    #[allow(clippy::expect_used)]
    fn paneled_i8_arena_reuse() {
        if V3::new_checked().is_none() {
            return;
        }
        const NQ: usize = 17;
        const DIM: usize = 128;
        let q: Vec<f32> = (0..NQ * DIM).map(|i| rnd(5, i)).collect();
        let q_mat = MatRef::new(Standard::<f32>::new(NQ, DIM).expect("nq×dim"), q.as_slice())
            .expect("query slice");
        let mut query = PaneledQuantQuery::build(q_mat).expect("V3 checked by caller");

        for (call, &nd) in [251usize, 3, 64, 1].iter().enumerate() {
            let d: Vec<f32> = (0..nd * DIM).map(|i| rnd(6 + call as u64, i)).collect();
            let d_mat = MatRef::new(Standard::<f32>::new(nd, DIM).expect("nd×dim"), d.as_slice())
                .expect("doc slice");
            let docs = PaneledQuantDocs::build(d_mat);
            let mut got = vec![0.0f32; NQ];
            query.compute_max_sim(&docs, &mut got);

            let want = reference(&q, NQ, &d, nd, DIM);
            for i in 0..NQ {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "call {call} (nd={nd}) row {i}: paneled-i8 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }
}
