// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! 4-bit MinMax instantiation of the coarse driver: identity [`QueryWalk`] /
//! [`DocWalk`] (panels borrow the source), the `I8Kernel` (Stage A), `MinMax`
//! postprocess (Stage B), `Max` reducer (Stage C), and the standalone
//! `QuantTiled{Query,Docs}` entry. Everything else is generic in [`super`].

use core::mem::size_of;
use std::num::NonZeroUsize;

use diskann_utils::ReborrowMut;
use diskann_wide::arch::x86_64::V3;

use super::arena::ResettableArena;
use super::tilers::{A_PANEL, B_PANEL, DPanel, DocWalk, QPanel, QueryWalk};
use super::{
    Accumulate, BlockCtx, Kernel, Plan, Postprocess, Reducer, Strip, StripMut, TileBudget, drive,
    leaves,
};
use crate::CompressInto;
use crate::algorithms::Transform;
use crate::algorithms::transforms::NullTransform;
use crate::alloc::ScopedAllocator;
use crate::minmax::{MinMaxCompensation, MinMaxMeta, MinMaxQuantizer};
use crate::multi_vector::{BlockTransposed, Defaulted, Mat, MatRef, Standard};
use crate::num::Positive;

// ── Stage A ──────────────────────────────────────────────────────

pub(crate) struct I8Kernel;

impl Kernel<V3> for I8Kernel {
    type Acc = i32;
    const A_PANEL: usize = A_PANEL;
    const B_PANEL: usize = B_PANEL;
}

impl<'a, 'b> Accumulate<V3, QPanel<'a, i16>, DPanel<'b, u8>> for I8Kernel {
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, i16>,
        b: DPanel<'b, u8>,
        mut out: StripMut<'_, i32>,
    ) {
        // SAFETY: `a` is a 16×k block-transposed i16 block; `b` is B_PANEL rows of k u8;
        // `out` is B_PANEL columns of 16 i32 at stride 16 (`k` even).
        unsafe {
            leaves::int_store_microkernel::<B_PANEL>(
                arch,
                a.as_ptr(),
                b.as_ptr(),
                a.k(),
                out.as_mut_ptr(),
                A_PANEL,
            );
        }
    }

    fn accumulate_tail(
        &self,
        arch: V3,
        a: QPanel<'a, i16>,
        b: DPanel<'b, u8>,
        mut out: StripMut<'_, i32>,
    ) {
        let (ap, bp, op) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        // SAFETY: as `accumulate`, with a runtime width `b.rows()` in 1..B_PANEL.
        unsafe {
            match b.rows() {
                3 => leaves::int_store_microkernel::<3>(arch, ap, bp, a.k(), op, A_PANEL),
                2 => leaves::int_store_microkernel::<2>(arch, ap, bp, a.k(), op, A_PANEL),
                1 => leaves::int_store_microkernel::<1>(arch, ap, bp, a.k(), op, A_PANEL),
                other => unreachable!("tail width {other} out of 1..{B_PANEL}"),
            }
        }
    }
}

// ── Stage B ──────────────────────────────────────────────────────

/// 4-bit MinMax dequant. Rewrites each raw integer dot into the MinMax inner product
/// using per-vector `a`/`b`/`n` metadata, indexed by the block offsets.
pub(crate) struct MinMax<'m> {
    query_meta: &'m [MinMaxCompensation],
    doc_meta: &'m [MinMaxCompensation],
    dim: f32,
}

impl Postprocess<V3, i32> for MinMax<'_> {
    type Score = f32;

    fn scratch_len(&self, cols: usize) -> usize {
        A_PANEL * cols
    }

    fn apply<'s>(
        &self,
        arch: V3,
        acc: Strip<'s, i32>,
        scratch: StripMut<'s, f32>,
        ctx: BlockCtx,
    ) -> Strip<'s, f32> {
        let cols = acc.cols();
        let q = &self.query_meta[ctx.a_row_offset..ctx.a_row_offset + A_PANEL];
        let d = &self.doc_meta[ctx.b_row_offset..ctx.b_row_offset + cols];
        let StripMut { data, rows } = scratch;
        // SAFETY: `acc` is `cols` cols of 16 i32; `data` is writable for `A_PANEL*cols`
        // f32; `q.len() == 16`, `d.len() == cols`.
        unsafe {
            leaves::score_strip(arch, acc.as_ptr(), data.as_mut_ptr(), cols, q, d, self.dim);
        }
        Strip {
            data: &data[..rows * cols],
            rows,
        }
    }
}

// ── Stage C ──────────────────────────────────────────────────────

pub(crate) struct Max;

impl Reducer<V3, f32> for Max {
    type State = f32;
    const A_PANEL: usize = A_PANEL;

    fn init() -> f32 {
        f32::MIN
    }

    fn fold(&self, arch: V3, state: &mut [f32], scores: Strip<f32>, _first_col: usize) {
        // SAFETY: `state` is A_PANEL=16 f32; `scores` is `cols` cols of 16 f32.
        unsafe { leaves::fold_strip(arch, state.as_mut_ptr(), scores.as_ptr(), scores.cols()) }
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

/// A prepared 4-bit MinMax query set for the coarse tiled driver (V3/AVX2).
/// Standalone POC entry, mirroring `QuantStagedQuery` but built on `drive`.
pub struct QuantTiledQuery {
    query: BlockTransposed<i16, 16, 2>,
    meta: Vec<MinMaxCompensation>,
    dim: usize,
    arch: V3,
    state: Vec<f32>,
    arena: ResettableArena,
}

impl QuantTiledQuery {
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
        let query = BlockTransposed::<i16, 16, 2>::from_matrix_view(view.as_matrix_view());

        let padded = query.padded_nrows();
        let mut meta = vec![MinMaxCompensation::default(); padded];
        for (r, m) in meta.iter_mut().enumerate().take(nq) {
            *m = q_mat.get_row(r).expect("row < nq").meta();
        }

        // `partial` and `scored` each fit `l1_b`, so `2·l1_b` bounds the arena for any
        // k; a page of headroom covers alignment.
        let arena = ResettableArena::with_capacity(2 * TileBudget::default().l1_b + 4096)
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
    pub fn compute_max_sim(&mut self, docs: &QuantTiledDocs, scores: &mut [f32]) {
        self.compute(docs, scores, TileBudget::default());
    }

    fn compute(&mut self, docs: &QuantTiledDocs, scores: &mut [f32], budget: TileBudget) {
        let nq = self.query.nrows();
        assert_eq!(scores.len(), nq, "scores length must equal query count");
        assert_eq!(self.dim, docs.dim, "query dim != doc dim");

        let k = self.query.padded_ncols();
        let padded = self.query.padded_nrows();

        self.arena.reset();
        let plan = Plan::new(
            k * size_of::<i16>(),
            k * size_of::<u8>(),
            A_PANEL,
            B_PANEL,
            size_of::<i32>(),
            budget,
        );
        let a_walk = QueryWalk::new(self.query.as_slice(), k, plan.a_panels);
        let b_walk = DocWalk::new(&docs.codes, k, plan.b_panels);
        let post = MinMax {
            query_meta: &self.meta,
            doc_meta: &docs.meta,
            dim: self.dim as f32,
        };
        drive(
            self.arch,
            a_walk,
            b_walk,
            &I8Kernel,
            &post,
            &Max,
            &mut self.state[..padded],
            ScopedAllocator::new(&self.arena),
        );

        for (s, &raw) in scores.iter_mut().zip(self.state.iter()) {
            *s = -raw;
        }
    }
}

/// A prepared 4-bit MinMax document set (codes-together / metadata-together SoA).
pub struct QuantTiledDocs {
    codes: Vec<u8>,
    meta: Vec<MinMaxCompensation>,
    dim: usize,
    nv: usize,
}

impl QuantTiledDocs {
    #[allow(clippy::expect_used)]
    pub fn build(docs: MatRef<'_, Standard<f32>>) -> Self {
        let (nv, dim) = (docs.num_vectors(), docs.vector_dim());
        let padded_dim = dim.next_multiple_of(2);
        let d_mat = quantize(docs);

        let mut codes = vec![0u8; nv * padded_dim];
        let mut meta = Vec::with_capacity(nv);
        for r in 0..nv {
            let row = d_mat.get_row(r).expect("row < nv");
            for j in 0..dim {
                codes[r * padded_dim + j] = row.vector().get(j).expect("col < dim") as u8;
            }
            meta.push(row.meta());
        }
        Self {
            codes,
            meta,
            dim,
            nv,
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.nv
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
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );
        let quantize = |data: &[f32], n: usize| -> Mat<MinMaxMeta<4>> {
            let input = MatRef::new(Standard::<f32>::new(n, dim).unwrap(), data).unwrap();
            let mut out: Mat<MinMaxMeta<4>> = Mat::new(MinMaxMeta::new(n, dim), Defaulted).unwrap();
            quantizer.compress_into(input, out.reborrow_mut()).unwrap();
            out
        };
        let q_mat = quantize(q, nq);
        let d_mat = quantize(d, nd);
        let query: QueryMatRef<_> = q_mat.as_view().into();
        let mut out = vec![0.0f32; nq];
        MaxSim::new(&mut out).evaluate(query, d_mat.as_view());
        out
    }

    /// (nq, nd, dim): B-remainder classes, A-panel remainder (17), multi-tile B, and
    /// the odd-dim even-K contract.
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

    #[test]
    fn tiled_i8_matches_minmax_reference() {
        if V3::new_checked().is_none() {
            return;
        }
        for &(nq, nd, dim) in CASES {
            let q_data: Vec<f32> = (0..nq * dim).map(|i| rnd(1, i)).collect();
            let d_data: Vec<f32> = (0..nd * dim).map(|i| rnd(2, i)).collect();

            let q_f32 = MatRef::new(Standard::<f32>::new(nq, dim).unwrap(), &q_data).unwrap();
            let d_f32 = MatRef::new(Standard::<f32>::new(nd, dim).unwrap(), &d_data).unwrap();
            let mut query = QuantTiledQuery::build(q_f32).unwrap();
            let docs = QuantTiledDocs::build(d_f32);
            let mut got = vec![0.0f32; nq];
            query.compute_max_sim(&docs, &mut got);

            let want = reference(&q_data, nq, &d_data, nd, dim);
            for i in 0..nq {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: tiled-i8 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }

    /// Tiny cache budget clamps the planner to one A-panel and one B-panel per tile,
    /// forcing multiple A-tiles and B-tiles — exercising the cross-tile offset carry
    /// the default-budget cases (one A-tile) never reach.
    #[test]
    fn tiled_i8_multi_tile_tiny_budget() {
        if V3::new_checked().is_none() {
            return;
        }
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &(nq, nd, dim) in &[(48usize, 22usize, 64usize), (33, 37, 128), (35, 19, 65)] {
            let q_data: Vec<f32> = (0..nq * dim).map(|i| rnd(3, i)).collect();
            let d_data: Vec<f32> = (0..nd * dim).map(|i| rnd(4, i)).collect();

            let q_f32 = MatRef::new(Standard::<f32>::new(nq, dim).unwrap(), &q_data).unwrap();
            let d_f32 = MatRef::new(Standard::<f32>::new(nd, dim).unwrap(), &d_data).unwrap();
            let mut query = QuantTiledQuery::build(q_f32).unwrap();
            let docs = QuantTiledDocs::build(d_f32);
            let mut got = vec![0.0f32; nq];
            query.compute(&docs, &mut got, budget);

            let want = reference(&q_data, nq, &d_data, nd, dim);
            for i in 0..nq {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: tiny-budget tiled-i8 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }

    /// Arena reuse across differently-sized doc sets stays correct.
    #[test]
    fn tiled_i8_arena_reuse() {
        if V3::new_checked().is_none() {
            return;
        }
        const NQ: usize = 17;
        const DIM: usize = 128;
        let q_data: Vec<f32> = (0..NQ * DIM).map(|i| rnd(5, i)).collect();
        let q_f32 = MatRef::new(Standard::<f32>::new(NQ, DIM).unwrap(), &q_data).unwrap();
        let mut query = QuantTiledQuery::build(q_f32).unwrap();

        for (call, &nd) in [251usize, 3, 64, 1].iter().enumerate() {
            let d_data: Vec<f32> = (0..nd * DIM).map(|i| rnd(6 + call as u64, i)).collect();
            let d_f32 = MatRef::new(Standard::<f32>::new(nd, DIM).unwrap(), &d_data).unwrap();
            let docs = QuantTiledDocs::build(d_f32);
            let mut got = vec![0.0f32; NQ];
            query.compute_max_sim(&docs, &mut got);

            let want = reference(&q_data, NQ, &d_data, nd, DIM);
            for i in 0..NQ {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "call {call} (nd={nd}) row {i}: tiled-i8 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }
}
