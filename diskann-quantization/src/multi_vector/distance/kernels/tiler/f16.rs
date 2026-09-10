// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 entry for the coarse driver, via **per-tile** f16→f32 conversion.
//!
//! The query is stored block-transposed *as f16*; docs stay row-major f16. Each
//! tile is widened f16→f32 into a small buffer the walk **reuses** across tiles
//! (allocated once from the arena) — see [`QueryConvertWalk`]/[`DocConvertWalk`].
//! The widened f32 tiles feed the same f32 store kernel → [`Identity`] postprocess →
//! `Max` reducer as a native-f32 walk would. No 2× whole-matrix f32 copy; the
//! lending [`TileWalk`] contract makes the buffer reuse sound.

use core::mem::size_of;

use diskann_wide::arch::x86_64::V3;

use super::arena::ResettableArena;
use super::minmax::Max;
use super::tilers::{A_PANEL, B_PANEL, DPanel, DocConvertWalk, QPanel, QueryConvertWalk};
use super::{Accumulate, Identity, Kernel, Plan, StripMut, TileBudget, drive, leaves, zeroed};
use crate::alloc::{Poly, ScopedAllocator};
use crate::multi_vector::{BlockTransposed, MatRef, Standard};

// ── Stage A (f32 store kernel) ───────────────────────────────────

pub(crate) struct F32Kernel;

impl Kernel<V3> for F32Kernel {
    type Acc = f32;
    const A_PANEL: usize = A_PANEL;
    const B_PANEL: usize = B_PANEL;
}

impl<'a, 'b> Accumulate<V3, QPanel<'a, f32>, DPanel<'b, f32>> for F32Kernel {
    fn accumulate(
        &self,
        arch: V3,
        a: QPanel<'a, f32>,
        b: DPanel<'b, f32>,
        mut out: StripMut<'_, f32>,
    ) {
        // SAFETY: `a` is a 16×k block-transposed f32 block; `b` is B_PANEL rows of k f32;
        // `out` is B_PANEL columns of 16 f32 at stride 16.
        unsafe {
            leaves::f32_store_microkernel::<B_PANEL>(
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
        a: QPanel<'a, f32>,
        b: DPanel<'b, f32>,
        mut out: StripMut<'_, f32>,
    ) {
        let (ap, bp, op) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        // SAFETY: as `accumulate`, with a runtime width `b.rows()` in 1..B_PANEL.
        unsafe {
            match b.rows() {
                3 => leaves::f32_store_microkernel::<3>(arch, ap, bp, a.k(), op, A_PANEL),
                2 => leaves::f32_store_microkernel::<2>(arch, ap, bp, a.k(), op, A_PANEL),
                1 => leaves::f32_store_microkernel::<1>(arch, ap, bp, a.k(), op, A_PANEL),
                other => unreachable!("tail width {other} out of 1..{B_PANEL}"),
            }
        }
    }
}

// ── Public entry ─────────────────────────────────────────────────

/// A prepared f16 query set, stored block-transposed as f16 (widened per tile).
pub struct QuantTiledF16Query {
    query: BlockTransposed<half::f16, 16>,
    dim: usize,
    arch: V3,
    state: Vec<f32>,
    arena: ResettableArena,
}

impl QuantTiledF16Query {
    /// `None` if AVX2 (V3) is unavailable.
    #[allow(clippy::expect_used)]
    pub fn build(query: MatRef<'_, Standard<half::f16>>) -> Option<Self> {
        let arch = V3::new_checked()?;
        let dim = query.vector_dim();
        let query = BlockTransposed::<half::f16, 16>::from_matrix_view(query.as_matrix_view());

        let k = query.padded_ncols();
        let padded = query.padded_nrows();

        // Size the arena for the reused convert buffers (one A-tile = the whole query
        // at most; one B-tile) plus the driver's `partial` strip, at the default
        // budget. `Identity` postprocess needs no `scored` scratch.
        let plan = Plan::new(
            k * size_of::<f32>(),
            k * size_of::<f32>(),
            A_PANEL,
            B_PANEL,
            size_of::<f32>(),
            TileBudget::default(),
        );
        let b_tile_rows = B_PANEL * plan.b_panels;
        let f32s = padded * k + b_tile_rows * k + A_PANEL * b_tile_rows;
        let arena = ResettableArena::with_capacity(f32s * size_of::<f32>() + 8192)
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
    pub fn compute_max_sim(&mut self, docs: &QuantTiledF16Docs, scores: &mut [f32]) {
        self.compute(docs, scores, TileBudget::default());
    }

    #[allow(clippy::expect_used)]
    fn compute(&mut self, docs: &QuantTiledF16Docs, scores: &mut [f32], budget: TileBudget) {
        let nq = self.query.nrows();
        assert_eq!(scores.len(), nq, "scores length must equal query count");
        assert_eq!(self.dim, docs.dim, "query dim != doc dim");

        let k = self.query.padded_ncols();
        let padded = self.query.padded_nrows();

        self.arena.reset();
        let plan = Plan::new(
            k * size_of::<f32>(),
            k * size_of::<f32>(),
            A_PANEL,
            B_PANEL,
            size_of::<f32>(),
            budget,
        );

        // Reused per-tile convert buffers, each sized to its largest single tile.
        let q_src = self.query.as_slice();
        let qbuf_len = q_src.len().min(A_PANEL * plan.a_panels * k).max(1);
        let dbuf_len = docs.codes.len().min(B_PANEL * plan.b_panels * k).max(1);
        let alloc = ScopedAllocator::new(&self.arena);
        let mut qbuf_poly = Poly::<[f32], _>::new_uninit_slice(qbuf_len, alloc).expect("q convert");
        let mut dbuf_poly = Poly::<[f32], _>::new_uninit_slice(dbuf_len, alloc).expect("d convert");
        let qbuf = zeroed(&mut qbuf_poly, qbuf_len);
        let dbuf = zeroed(&mut dbuf_poly, dbuf_len);

        let a_walk = QueryConvertWalk::new(q_src, k, plan.a_panels, qbuf);
        let b_walk = DocConvertWalk::new(&docs.codes, k, plan.b_panels, dbuf);
        drive(
            self.arch,
            a_walk,
            b_walk,
            &F32Kernel,
            &Identity,
            &Max,
            &mut self.state[..padded],
            alloc,
        );

        scores.copy_from_slice(&self.state[..nq]);
    }
}

/// A prepared f16 document set, kept row-major as f16 (widened per tile).
pub struct QuantTiledF16Docs {
    codes: Vec<half::f16>,
    dim: usize,
    nv: usize,
}

impl QuantTiledF16Docs {
    pub fn build(docs: MatRef<'_, Standard<half::f16>>) -> Self {
        let (nv, dim) = (docs.num_vectors(), docs.vector_dim());
        let codes = docs.as_slice().to_vec();
        Self { codes, dim, nv }
    }

    pub fn num_vectors(&self) -> usize {
        self.nv
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diskann_wide::{cast_f16_to_f32, cast_f32_to_f16};

    fn rnd(seed: u64, idx: usize) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(idx as u64)
            .wrapping_mul(1442695040888963407);
        ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// Naive f16 max-IP reference: widen every f16 to f32, dot, take the max doc.
    fn reference(q: &[half::f16], nq: usize, d: &[half::f16], nd: usize, dim: usize) -> Vec<f32> {
        (0..nq)
            .map(|i| {
                (0..nd)
                    .map(|j| {
                        (0..dim)
                            .map(|c| {
                                cast_f16_to_f32(q[i * dim + c]) * cast_f16_to_f32(d[j * dim + c])
                            })
                            .sum::<f32>()
                    })
                    .fold(f32::MIN, f32::max)
            })
            .collect()
    }

    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 64),
        (5, 3, 5),
        (16, 4, 64),
        (16, 5, 128),
        (16, 7, 256),
        (17, 9, 65),
        (32, 16, 256),
        (64, 1250, 64),
        (8, 33, 127),
    ];

    #[test]
    fn tiled_f16_matches_reference() {
        if V3::new_checked().is_none() {
            return;
        }
        for &(nq, nd, dim) in CASES {
            let q: Vec<half::f16> = (0..nq * dim).map(|i| cast_f32_to_f16(rnd(1, i))).collect();
            let d: Vec<half::f16> = (0..nd * dim).map(|i| cast_f32_to_f16(rnd(2, i))).collect();

            let q_mat = MatRef::new(Standard::<half::f16>::new(nq, dim).unwrap(), &q).unwrap();
            let d_mat = MatRef::new(Standard::<half::f16>::new(nd, dim).unwrap(), &d).unwrap();
            let mut query = QuantTiledF16Query::build(q_mat).unwrap();
            let docs = QuantTiledF16Docs::build(d_mat);
            let mut got = vec![0.0f32; nq];
            query.compute_max_sim(&docs, &mut got);

            let want = reference(&q, nq, &d, nd, dim);
            for i in 0..nq {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-3 * want[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: tiled-f16 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }

    /// Tiny cache budget forces multiple A- and B-tiles.
    #[test]
    fn tiled_f16_multi_tile_tiny_budget() {
        if V3::new_checked().is_none() {
            return;
        }
        let budget = TileBudget { l2_a: 1, l1_b: 1 };
        for &(nq, nd, dim) in &[(48usize, 22usize, 64usize), (33, 37, 128), (35, 19, 65)] {
            let q: Vec<half::f16> = (0..nq * dim).map(|i| cast_f32_to_f16(rnd(3, i))).collect();
            let d: Vec<half::f16> = (0..nd * dim).map(|i| cast_f32_to_f16(rnd(4, i))).collect();

            let q_mat = MatRef::new(Standard::<half::f16>::new(nq, dim).unwrap(), &q).unwrap();
            let d_mat = MatRef::new(Standard::<half::f16>::new(nd, dim).unwrap(), &d).unwrap();
            let mut query = QuantTiledF16Query::build(q_mat).unwrap();
            let docs = QuantTiledF16Docs::build(d_mat);
            let mut got = vec![0.0f32; nq];
            query.compute(&docs, &mut got, budget);

            let want = reference(&q, nq, &d, nd, dim);
            for i in 0..nq {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-3 * want[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: tiny-budget tiled-f16 {} != reference {}",
                    got[i],
                    want[i],
                );
            }
        }
    }
}
