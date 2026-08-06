// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! V3 (AVX2) **quantized** staged path: 4-bit MinMax MaxSim.
//!
//! This is the payoff the staged framework was built for — the first
//! **non-identity Stage B**. Stage A accumulates a raw *integer* dot product
//! (`Acc = i32`), Stage B ([`MinMaxPostprocess`]) turns each `i32` into the
//! finished MinMax inner product (`Score = f32`) using per-vector
//! scale/center/sum metadata, and Stage C ([`MaxReducer`](super::maxsim::MaxReducer))
//! folds the `f32` scores exactly as in the f32 path — **unchanged**.
//!
//! # Stage A mirrors the f32 kernel (block-transposed + broadcast, no reduction)
//!
//! The integer micro-kernel is structurally the f32 [`store_microkernel`] with
//! `PACK = 2` integer MACs:
//!
//! * **Query** (`Left`): [`BlockTransposed<i16, 16, 2>`] — codes widened `u8→i16`
//!   once at build time, then block-transposed with two K-columns interleaved per
//!   row (`[r0_k0, r0_k1, r1_k0, r1_k1, …]`). One col-pair is `GROUP·PACK = 32`
//!   `i16` = two `i16x16` halves (rows `0..8` / `8..16`).
//! * **Doc** (`Right`): [`RowMajor<u8>`] — codes stream in place; the kernel reads
//!   each doc col's 2-K word, widens it, and broadcasts it as
//!   `[d_k0, d_k1, d_k0, d_k1, …]`.
//! * **MAC**: `i32x8::dot_simd(query_half, doc_broadcast)` — i.e. `vpmaddwd`
//!   (`_mm256_madd_epi16`), which sums each interleaved K-pair into one `i32`
//!   lane. Each lane **is** one A-row's running dot for that doc col, so the
//!   accumulators *are* the outputs — no per-pair horizontal reduction, exactly
//!   the property block-transposition buys in the f32 kernel.
//!
//! 4-bit codes are `u8 ∈ [0, 15]` (not nibble-packed), widened to `i16 ∈ [0, 15]`.
//! A 2-K partial is `≤ 2·15·15 = 450` and the full dot over `dim ≤ 512` is
//! `≤ 15·15·512 ≈ 1.2e5`, so the `i32` accumulation **never overflows** — the
//! integer dot is *exact*. No new `diskann-wide` op is needed: the existing
//! [`SIMDDotProduct<i16x16>`] for `i32x8` is the right shape in broadcast config.
//!
//! # Even-K contract (zero driver change)
//!
//! The shared [driver](super::driver) derives the A-side physical row stride from
//! its `k` argument (`rows_done · k == block_offset`), which only matches a
//! block-transposed query when `k == padded_ncols`. So the entry point passes the
//! query's **padded** (even) column count as the driver `k` and requires the doc
//! to match (zero-padded to even). The padding column holds `0` on *both* sides,
//! so it contributes `0` to every dot — the IP is unchanged, and the kernel walks
//! exactly `k/2` full K-pairs with no odd-K tail branch.

use std::num::NonZeroUsize;

use diskann_utils::ReborrowMut;
use diskann_wide::arch::x86_64::V3;
use diskann_wide::{SIMDCast, SIMDDotProduct, SIMDMulAdd, SIMDReinterpret, SIMDVector};

use super::super::TileBudget;
use super::super::layouts;
use super::arena::ResettableArena;
use super::driver::tiled_reduce_staged;
use super::maxsim::MaxReducer;
use super::{FoldCtx, Postprocess, StagedKernel};
use crate::CompressInto;
use crate::algorithms::Transform;
use crate::algorithms::transforms::NullTransform;
use crate::alloc::ScopedAllocator;
use crate::minmax::{MinMaxCompensation, MinMaxMeta, MinMaxQuantizer};
use crate::multi_vector::{BlockTransposed, BlockTransposedRef, Defaulted, Mat, MatRef, Standard};
use crate::num::Positive;

diskann_wide::alias!(i16s = <V3>::i16x16);
diskann_wide::alias!(i32s = <V3>::i32x8);
diskann_wide::alias!(u32s = <V3>::u32x8);
diskann_wide::alias!(f32s = <V3>::f32x8);

// ── Stage A: integer store-out micro-kernel ──────────────────────

/// Zero-sized Stage-A kernel marker for the quantized (4-bit MinMax) staged path
/// with block size `GROUP`.
pub(crate) struct StagedI8Kernel<const GROUP: usize>;

// SAFETY: `full_panel`/`partial_panel` read A_PANEL(16) i16 query rows × K
// (block-transposed, K padded to even) and UNROLL × K u8 doc elements, and write
// UNROLL columns of A_PANEL(16) i32 into `partial` at stride `partial_b_stride` —
// all within the bounds the `StagedKernel` contract guarantees.
unsafe impl StagedKernel<V3> for StagedI8Kernel<16> {
    type Left = layouts::BlockTransposed<i16, 16, 2>;
    type Right = layouts::RowMajor<u8>;
    type Acc = i32;
    const A_PANEL: usize = 16;
    const B_PANEL: usize = 4;

    #[inline(always)]
    unsafe fn full_panel(
        arch: V3,
        a: *const i16,
        b: *const u8,
        k: usize,
        partial: *mut i32,
        partial_b_stride: usize,
    ) {
        // SAFETY: pointer validity per the `StagedKernel<V3>` contract.
        unsafe {
            int_store_microkernel::<{ Self::B_PANEL }>(arch, a, b, k, partial, partial_b_stride)
        }
    }

    #[inline(always)]
    unsafe fn partial_panel(
        arch: V3,
        remainder: usize,
        a: *const i16,
        b: *const u8,
        k: usize,
        partial: *mut i32,
        partial_b_stride: usize,
    ) {
        // SAFETY: pointer validity per the `StagedKernel<V3>` contract.
        unsafe {
            match remainder {
                1 => int_store_microkernel::<1>(arch, a, b, k, partial, partial_b_stride),
                2 => int_store_microkernel::<2>(arch, a, b, k, partial, partial_b_stride),
                3 => int_store_microkernel::<3>(arch, a, b, k, partial, partial_b_stride),
                _ => unreachable!(
                    "unexpected remainder {remainder} for B_PANEL={}",
                    Self::B_PANEL
                ),
            }
        }
    }
}

/// V3 integer store-out micro-kernel: 16 A-rows × `UNROLL` B-rows.
///
/// Mirrors [`super::v3::store_microkernel`] with `PACK = 2` integer MACs (see the
/// module docs). The epilogue stores each B-column's 16 A-row `i32` accumulators
/// into `partial` (A-major: column `j` at `partial + j*b_stride`, as two `i32x8`
/// halves) — identical contract to the f32 kernel, so Stage B / Stage C are
/// unchanged.
///
/// # Safety
///
/// 1. `a_packed` points to a block-transposed query block of `16 * k` `i16`
///    (`k` even — the padded column count).
/// 2. `b` points to `UNROLL` rows of `k` contiguous `u8` (`k` even).
/// 3. `partial` is valid for `UNROLL` columns of 16 `i32` at stride `b_stride`.
#[inline(always)]
unsafe fn int_store_microkernel<const UNROLL: usize>(
    arch: V3,
    a_packed: *const i16,
    b: *const u8,
    k: usize,
    partial: *mut i32,
    b_stride: usize,
) {
    let mut p0 = [i32s::default(arch); UNROLL];
    let mut p1 = [i32s::default(arch); UNROLL];
    let offsets: [usize; UNROLL] = core::array::from_fn(|j| k * j);

    // One col-pair of the block-transposed query = GROUP·PACK = 32 i16 = two
    // `i16x16` halves (rows 0..8 low, rows 8..16 high).
    let a_pair_stride = 2 * i16s::LANES;
    let a_half = i16s::LANES;
    let pairs = k / 2; // `k` is even (the padded column count) ⇒ exact, no tail.

    for p in 0..pairs {
        // SAFETY: precondition 1 — the query block has `pairs` col-pairs of 32 i16.
        let (a0, a1) = unsafe {
            (
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p)),
                i16s::load_simd(arch, a_packed.add(a_pair_stride * p + a_half)),
            )
        };

        for j in 0..UNROLL {
            // SAFETY: precondition 2 — doc col j is `offsets[j]` in, and
            // `2*p + 1 < k` because `pairs == k/2`.
            let (d0, d1) = unsafe {
                let base = 2 * p + offsets[j];
                (
                    u32::from(b.add(base).read()),
                    u32::from(b.add(base + 1).read()),
                )
            };
            // Broadcast the K-pair [d0, d1] across all 8 i32 lanes, reinterpreted
            // as i16x16 = [d0, d1, d0, d1, …] — the shape `madd_epi16` consumes
            // (pairing each query [k0, k1] with [d0, d1] into one i32 lane).
            let packed = d0 | (d1 << 16);
            let bcast: i16s = u32s::splat(arch, packed).reinterpret_simd();
            p0[j] = p0[j].dot_simd(a0, bcast);
            p1[j] = p1[j].dot_simd(a1, bcast);
        }
    }

    for j in 0..UNROLL {
        // SAFETY: precondition 3 — column j occupies [j*b_stride, j*b_stride+16) i32.
        unsafe {
            p0[j].store_simd(partial.add(j * b_stride));
            p1[j].store_simd(partial.add(j * b_stride + i32s::LANES));
        }
    }
}

// ── Stage B: integer code → MinMax inner product ─────────────────

/// Stage B for 4-bit MinMax: convert each raw integer dot `⟨codes⟩` (the `i32`
/// `Acc` from Stage A) into the finished MinMax inner product
///
/// ```text
/// IP = qm.a·dm.a·⟨codes⟩ + qm.n·dm.b + dm.n·qm.b + qm.b·dm.b·dim
/// ```
///
/// (the linear decomposition in `minmax::vectors`), emitting **+IP** so Stage C
/// folds `max` and the caller negates once at the end (`min distance =
/// max_a(-IP) = -max_a IP`).
///
/// This is the first non-identity [`Postprocess`]: it reports a non-zero
/// [`scratch_len`](Postprocess::scratch_len) (so the driver allocates an f32
/// region) and [`apply`] writes the converted scores into it, indexing its
/// per-vector metadata by the *global* row offsets in [`FoldCtx`].
pub(crate) struct MinMaxPostprocess<'m> {
    /// Per-query-vector metadata, indexed by `ctx.a_row_offset + i`. Length must
    /// be `≥ padded_nrows` (padded rows carry default metadata; their scores are
    /// computed but never read).
    query_meta: &'m [MinMaxCompensation],
    /// Per-doc-vector metadata, indexed by `ctx.b_row_offset + c`. Length `≥ nd`.
    doc_meta: &'m [MinMaxCompensation],
    /// The **logical** dimension (the `dim` term in the IP formula). The integer
    /// dot is taken over the padded columns, but the extra column's codes are `0`
    /// on both sides, so `⟨codes⟩` is unchanged.
    dim: f32,
}

impl<'m> MinMaxPostprocess<'m> {
    pub(crate) fn new(
        query_meta: &'m [MinMaxCompensation],
        doc_meta: &'m [MinMaxCompensation],
        dim: usize,
    ) -> Self {
        Self {
            query_meta,
            doc_meta,
            dim: dim as f32,
        }
    }
}

// SAFETY: `apply` reads exactly `ctx.valid_b_cols` columns of `ctx.a_panel` `i32`
// from `acc` at stride `ctx.b_stride`, writes only the corresponding
// `ctx.a_panel × ctx.valid_b_cols` region of `scratch` (the driver allocates
// `scratch_len(a_panel, max_b_cols) = a_panel · max_b_cols ≥ a_panel ·
// valid_b_cols` `f32`), and returns a pointer into it. Metadata indices
// `a_row_offset + i < padded_nrows ≤ query_meta.len()` and `b_row_offset + c < nd
// ≤ doc_meta.len()` are in bounds by the entry-point's preconditions.
//
// V3-specific: the quantized kernel only runs on V3 (`StagedI8Kernel: StagedKernel<V3>`),
// so Stage B is implemented for V3 only and uses AVX2 for the score conversion.
unsafe impl Postprocess<V3> for MinMaxPostprocess<'_> {
    type Acc = i32;
    type Score = f32;

    #[inline]
    fn scratch_len(&self, a_panel: usize, max_b_cols: usize) -> usize {
        a_panel * max_b_cols
    }

    #[inline]
    unsafe fn apply(
        &self,
        scratch: *mut f32,
        arch: V3,
        acc: *const i32,
        ctx: FoldCtx,
    ) -> *const f32 {
        let out = scratch;

        // Rewrite the per-(row i, col c) IP into a per-row-vector form whose only
        // per-column inputs are three doc scalars (so the 16-row inner loop is a
        // straight SIMD sweep):
        //   ip = dm.a·(qm.a·raw) + dm.b·qm.n + (dm.n + dm.b·dim)·qm.b
        //      = A_c·(qa·raw)    + B_c·qn    + C_c·qb,
        // with A_c=dm.a, B_c=dm.b, C_c=dm.n+dm.b·dim, and qa/qn/qb the per-query-row
        // metadata. This is the reference formula regrouped (within f32 rounding).
        // The quantized kernel always uses A_PANEL = 16 = 2*LANES, so Stage B is a
        // pure SIMD sweep — no scalar fallback for other panel widths.
        let lanes = f32s::LANES;
        debug_assert_eq!(
            ctx.a_panel,
            2 * lanes,
            "quantized Stage B expects A_PANEL == 2*LANES"
        );

        // Gather the per-A-row metadata into contiguous SoA arrays once (the
        // source `MinMaxCompensation` is AoS, so a strided scalar gather), then
        // hold them in registers across the whole B-column sweep.
        let mut qa = [0.0f32; 16];
        let mut qb = [0.0f32; 16];
        let mut qn = [0.0f32; 16];
        for i in 0..16 {
            let qm = self.query_meta[ctx.a_row_offset + i];
            qa[i] = qm.a;
            qb[i] = qm.b;
            qn[i] = qm.n;
        }
        // SAFETY: each array holds exactly 16 = 2·LANES f32.
        let (qa0, qa1, qb0, qb1, qn0, qn1) = unsafe {
            (
                f32s::load_simd(arch, qa.as_ptr()),
                f32s::load_simd(arch, qa.as_ptr().add(lanes)),
                f32s::load_simd(arch, qb.as_ptr()),
                f32s::load_simd(arch, qb.as_ptr().add(lanes)),
                f32s::load_simd(arch, qn.as_ptr()),
                f32s::load_simd(arch, qn.as_ptr().add(lanes)),
            )
        };

        for c in 0..ctx.valid_b_cols {
            let dm = self.doc_meta[ctx.b_row_offset + c];
            let a_c = f32s::splat(arch, dm.a);
            let b_c = f32s::splat(arch, dm.b);
            let c_c = f32s::splat(arch, dm.n + dm.b * self.dim);
            let acc_col = c * ctx.b_stride;
            let out_col = c * ctx.a_panel;
            // SAFETY: `acc_col + 2·LANES ≤ valid_b_cols · b_stride`; the partial
            // block is valid for that many i32, and `out_col + 2·LANES ≤ buf.len()`.
            unsafe {
                let raw0 = i32s::load_simd(arch, acc.add(acc_col)).simd_cast();
                let raw1 = i32s::load_simd(arch, acc.add(acc_col + lanes)).simd_cast();
                // a_c·(qa·raw) + (b_c·qn + c_c·qb)
                let s0 = a_c.mul_add_simd(qa0 * raw0, b_c.mul_add_simd(qn0, c_c * qb0));
                let s1 = a_c.mul_add_simd(qa1 * raw1, b_c.mul_add_simd(qn1, c_c * qb1));
                s0.store_simd(out.add(out_col));
                s1.store_simd(out.add(out_col + lanes));
            }
        }
        scratch.cast_const()
    }
}

// ── Public POC entry: prepared 4-bit MinMax staged MaxSim ────────

/// Quantize an f32 multi-vector to 4-bit MinMax (Null transform, scale 1.0) —
/// the shared quantizer for the public query/doc builders so both sides decode
/// to comparable codes + metadata.
#[allow(clippy::expect_used)] // POC constructor: inputs are pre-validated by the caller.
fn quantize_minmax_4bit(input: MatRef<'_, Standard<f32>>) -> Mat<MinMaxMeta<4>> {
    let dim = input.vector_dim();
    let n = input.num_vectors();
    let q = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(
            NonZeroUsize::new(dim).expect("dimension must be non-zero"),
        )),
        Positive::new(1.0).expect("1.0 is positive"),
    );
    let mut out: Mat<MinMaxMeta<4>> =
        Mat::new(MinMaxMeta::new(n, dim), Defaulted).expect("MinMaxMeta allocation");
    q.compress_into(input, out.reborrow_mut())
        .expect("input must be finite (no NaN)");
    out
}

/// A prepared 4-bit MinMax **query** set for the staged MaxSim kernel (V3/AVX2).
///
/// Built once from an f32 multi-vector; [`compute_max_sim`](Self::compute_max_sim)
/// is the per-document-set hot path. It owns a `ResettableArena` that backs the
/// staged driver's per-call `partial` / Stage-B scratch and a reused `state`
/// buffer, so **steady-state calls perform no heap allocation** (the arena is
/// reset, not reallocated, each call).
///
/// This is a **standalone POC entry**: the quantized path is intentionally *not*
/// yet unified into [`MaxSimIsa`](crate::multi_vector::distance::MaxSimIsa) /
/// `build_max_sim`, which ties to a productized `QuantizedSoa` matrix `Repr` (see
/// [`QuantStagedDocs`] and `docs/staged_multi_vector_kernel.md`).
pub struct QuantStagedQuery {
    /// Codes widened `u8→i16` and block-transposed (`PACK=2`) for Stage A.
    query: BlockTransposed<i16, 16, 2>,
    /// Per-vector metadata, padded to `query.padded_nrows()` (padded rows carry
    /// default metadata; their scores are computed but never read).
    meta: Vec<MinMaxCompensation>,
    /// Logical dimension.
    dim: usize,
    arch: V3,
    /// Reusable running-reduction output (len `query.padded_nrows()`); the driver
    /// re-initialises it each call, so it is reused, not reallocated.
    state: Vec<f32>,
    /// Reusable arena backing the driver's transient `partial` / `scored` scratch;
    /// reset (not reallocated) at the top of every call.
    arena: ResettableArena,
}

impl QuantStagedQuery {
    /// Quantize `query` to 4-bit MinMax and prepare the block-transposed layout +
    /// the reusable scratch arena. Returns `None` if AVX2 (V3) is unavailable on
    /// this host.
    #[allow(clippy::expect_used)] // POC constructor: dims are valid by construction.
    pub fn build(query: MatRef<'_, Standard<f32>>) -> Option<Self> {
        let arch = V3::new_checked()?;
        let dim = query.vector_dim();
        let nq = query.num_vectors();

        let q_mat = quantize_minmax_4bit(query);

        let mut codes = vec![0i16; nq * dim];
        for r in 0..nq {
            let row = q_mat.get_row(r).expect("row r < nq");
            for j in 0..dim {
                codes[r * dim + j] = i16::from(row.vector().get(j).expect("col j < dim") as u8);
            }
        }
        let code_view = MatRef::new(Standard::<i16>::new(nq, dim).expect("nq×dim i16"), &codes)
            .expect("code slice length");
        let bt = BlockTransposed::<i16, 16, 2>::from_matrix_view(code_view.as_matrix_view());

        let padded_nrows = bt.padded_nrows();
        let mut meta = vec![MinMaxCompensation::default(); padded_nrows];
        for (r, m) in meta.iter_mut().enumerate().take(nq) {
            *m = q_mat.get_row(r).expect("row r < nq").meta();
        }

        // The driver allocates only `partial` + `scored` from the arena (the
        // identity conversion buffers are zero-length). `StagedPlan` co-budgets so
        // each is `<= l1_b`, so `2 * l1_b` is a provable upper bound for *any* k
        // (no per-shape sizing needed); add one page of headroom for alignment.
        let arena_bytes = 2 * TileBudget::default().l1_b + 4096;
        let arena = ResettableArena::with_capacity(arena_bytes).expect("staged arena allocation");

        Some(Self {
            query: bt,
            meta,
            dim,
            arch,
            state: vec![f32::MIN; padded_nrows],
            arena,
        })
    }

    /// Whether this host supports the quantized staged kernel (requires AVX2 /
    /// the `V3` ISA). Use this to gate before calling [`build`](Self::build).
    pub fn is_supported() -> bool {
        V3::new_checked().is_some()
    }

    /// Number of (logical) query vectors.
    pub fn num_vectors(&self) -> usize {
        self.query.nrows()
    }

    /// Compute the per-query **min distance** (`= -max_d IP`) against `docs`,
    /// writing one score per query vector into `scores`.
    ///
    /// # Panics
    ///
    /// Panics if `scores.len() != self.num_vectors()` or the query and doc
    /// logical dimensions differ.
    #[allow(clippy::expect_used)] // doc view length is guaranteed by construction.
    pub fn compute_max_sim(&mut self, docs: &QuantStagedDocs, scores: &mut [f32]) {
        let nq = self.query.nrows();
        assert_eq!(
            scores.len(),
            nq,
            "scores length {} must equal query vector count {nq}",
            scores.len()
        );
        assert_eq!(
            self.dim, docs.dim,
            "query dim {} != doc dim {}",
            self.dim, docs.dim
        );

        let doc = MatRef::new(
            Standard::<u8>::new(docs.nv, docs.padded_dim).expect("nv×padded_dim u8"),
            &docs.codes,
        )
        .expect("doc code slice length");
        let post = MinMaxPostprocess::new(&self.meta, &docs.meta, self.dim);

        // Rewind the arena so the driver's `partial` / `scored` reuse last call's
        // storage (`&mut self` proves no prior allocation is still borrowing it),
        // then hand it the arena and the reused `state` output. Steady-state: no
        // heap allocation.
        let padded = self.query.padded_nrows();
        self.arena.reset();
        max_ip_kernel_staged_i8(
            self.arch,
            self.query.as_view(),
            doc,
            &post,
            &mut self.state[..padded],
            ScopedAllocator::new(&self.arena),
            TileBudget::default(),
        );

        for (s, &raw) in scores.iter_mut().zip(self.state.iter()) {
            *s = -raw; // min distance = -(max inner product)
        }
    }
}

/// A prepared 4-bit MinMax **document** set: the minimal "codes-together,
/// metadata-together" SoA the staged kernel streams. Stage A reads the contiguous
/// codes region (row-major `u8`, `padded_dim` per vector); Stage B reads the
/// contiguous metadata region (one [`MinMaxCompensation`] per vector).
///
/// This is the doc-side storage the kernel was designed around — the interleaved
/// `MinMaxMeta` `Repr` (one codes+meta blob per row) *cannot* be streamed by a
/// kernel that needs the codes contiguous for SIMD and the metadata only in the
/// postprocess.
///
/// # Productization (assessed, not built): a `QuantizedSoa<NBITS>` matrix `Repr`
///
/// The natural next step is a matrix `Repr` that owns this layout in one
/// allocation, `[codes region | aligned metadata region]`, with `Row<'a> = (&'a
/// [u8], &'a MinMaxCompensation)`. It fits the existing `Repr`/`ReprOwned`
/// contract (`layout()` = `codes_bytes + pad + meta_bytes`; `get_row` splits the
/// two regions) and needs a `CompressInto` that emits SoA from `MinMaxQuantizer`
/// (today's emits the interleaved blob). That `Repr` is also the prerequisite for
/// unifying the quantized path into `MaxSimIsa`; this owning prototype keeps the
/// POC self-contained without committing to it.
pub struct QuantStagedDocs {
    /// Row-major codes, `nv * padded_dim` `u8` (each `∈ [0, 15]` for 4-bit), with
    /// the trailing (padded) column zeroed when `dim` is odd.
    codes: Vec<u8>,
    /// Per-vector metadata, `nv` entries.
    meta: Vec<MinMaxCompensation>,
    /// Logical dimension (the IP formula's `dim`).
    dim: usize,
    /// Physical (even) column count `next_multiple_of(dim, 2)` — the doc row
    /// stride, which must equal the query's padded column count.
    padded_dim: usize,
    /// Number of vectors.
    nv: usize,
}

impl QuantStagedDocs {
    /// Quantize `docs` to 4-bit MinMax and pack the codes-together /
    /// metadata-together SoA (codes zero-padded to an even `padded_dim`).
    #[allow(clippy::expect_used)] // POC constructor: dims are valid by construction.
    pub fn build(docs: MatRef<'_, Standard<f32>>) -> Self {
        let dim = docs.vector_dim();
        let nv = docs.num_vectors();
        let padded_dim = dim.next_multiple_of(2);

        let d_mat = quantize_minmax_4bit(docs);
        let mut codes = vec![0u8; nv * padded_dim];
        let mut meta = Vec::with_capacity(nv);
        for r in 0..nv {
            let row = d_mat.get_row(r).expect("row r < nv");
            for j in 0..dim {
                codes[r * padded_dim + j] = row.vector().get(j).expect("col j < dim") as u8;
            }
            meta.push(row.meta());
        }
        Self {
            codes,
            meta,
            dim,
            padded_dim,
            nv,
        }
    }

    /// Number of document vectors.
    pub fn num_vectors(&self) -> usize {
        self.nv
    }
}

// ── Entry point ──────────────────────────────────────────────────

/// Compute per-query-vector max MinMax inner product into `state` via the staged
/// quantized pipeline. `state` (len ≥ `query.padded_nrows()`) is the caller's
/// output, left holding the raw max-IP (the caller negates for min-distance).
/// Transient scratch (`partial`, Stage-B region) is allocated internally from
/// `alloc` — the caller sizes nothing.
///
/// `query` is the block-transposed (`i16`, `GROUP=16`, `PACK=2`) widened codes;
/// `doc` is the row-major `u8` codes at stride `query.padded_ncols()` (even); and
/// `post` carries the query/doc metadata + logical `dim`.
///
/// # Panics
///
/// Panics if `state.len() < query.padded_nrows()` or `query.padded_ncols() !=
/// doc.vector_dim()` (the even-K contract — see the module docs).
pub(crate) fn max_ip_kernel_staged_i8(
    arch: V3,
    query: BlockTransposedRef<'_, i16, 16, 2>,
    doc: MatRef<'_, Standard<u8>>,
    post: &MinMaxPostprocess<'_>,
    state: &mut [f32],
    alloc: ScopedAllocator<'_>,
    budget: TileBudget,
) {
    let padded = query.padded_nrows();
    // `k` is the *padded* (even) column count: the driver derives the A-side
    // physical row stride from it, and it must match the doc stride.
    let k = query.padded_ncols();
    if state.len() < padded || k != doc.vector_dim() {
        max_ip_kernel_staged_i8_panic(state.len(), padded, k, doc.vector_dim());
    }

    let b_nrows = doc.num_vectors();

    // Empty contraction: every IP reduces to the metadata-only terms. The POC
    // does not exercise `dim == 0`; fill 0 and bail rather than enter the tiling
    // nest with a zero stride (matches the f32 entry's degenerate guard).
    if k == 0 {
        state[..padded].fill(0.0);
        return;
    }

    let ca = layouts::BlockTransposed::<i16, 16, 2>::new();
    let cb = layouts::RowMajor::<u8>::new();

    // SAFETY:
    // - `query.as_ptr()` is valid for `padded * k` i16 (block-transposed, K padded
    //   to even == k), and `padded` is a multiple of GROUP == A_PANEL == 16.
    // - `doc.as_slice()` is `b_nrows * k` contiguous u8 (k == doc.vector_dim()).
    // - `state.len() >= padded` (checked); the driver allocates its scratch from
    //   `alloc`.
    unsafe {
        tiled_reduce_staged::<V3, StagedI8Kernel<16>, MinMaxPostprocess<'_>, MaxReducer, _, _>(
            arch,
            &ca,
            &cb,
            post,
            query.as_ptr(),
            padded,
            doc.as_slice().as_ptr(),
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
fn max_ip_kernel_staged_i8_panic(state_len: usize, padded: usize, k: usize, doc_dim: usize) {
    panic!(
        "max_ip_kernel_staged_i8: precondition failed: \
         state.len()={state_len} (expected >= {padded}), \
         padded_ncols(k)={k}, doc.vector_dim()={doc_dim} (must be equal — even-K contract)"
    );
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_utils::ReborrowMut;
    use diskann_wide::arch::x86_64::V3;

    use super::super::super::TileBudget;
    use super::{
        MinMaxPostprocess, QuantStagedDocs, QuantStagedQuery, int_store_microkernel,
        max_ip_kernel_staged_i8,
    };
    use crate::CompressInto;
    use crate::algorithms::Transform;
    use crate::algorithms::transforms::NullTransform;
    use crate::alloc::ScopedAllocator;
    use crate::minmax::{MinMaxCompensation, MinMaxMeta, MinMaxQuantizer};
    use crate::multi_vector::distance::{MaxSim, QueryMatRef};
    use crate::multi_vector::{BlockTransposed, Defaulted, Mat, MatRef, Standard};
    use crate::num::Positive;
    use diskann_vector::DistanceFunctionMut;

    const NBITS: usize = 4;

    fn quantizer(dim: usize) -> MinMaxQuantizer {
        MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        )
    }

    /// Pseudo-random f32 in roughly `[-1, 1]`, deterministic per `(seed, idx)`.
    fn rnd(seed: u64, idx: usize) -> f32 {
        let x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(idx as u64)
            .wrapping_mul(1442695040888963407);
        ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn quantize(q: &MinMaxQuantizer, data: &[f32], n: usize, dim: usize) -> Mat<MinMaxMeta<NBITS>> {
        let input = MatRef::new(Standard::<f32>::new(n, dim).unwrap(), data).unwrap();
        let mut out: Mat<MinMaxMeta<NBITS>> = Mat::new(MinMaxMeta::new(n, dim), Defaulted).unwrap();
        q.compress_into(input, out.reborrow_mut()).unwrap();
        out
    }

    /// Extract `(codes_u8 [nv × padded_dim, zero-padded], meta [nv])` from a
    /// quantized doc matrix — the minimal doc-side SoA.
    fn doc_soa(mat: &Mat<MinMaxMeta<NBITS>>, dim: usize) -> (Vec<u8>, Vec<MinMaxCompensation>) {
        let nv = mat.num_vectors();
        let padded_dim = dim.next_multiple_of(2);
        let mut codes = vec![0u8; nv * padded_dim];
        let mut meta = Vec::with_capacity(nv);
        for r in 0..nv {
            let row = mat.get_row(r).unwrap();
            for j in 0..dim {
                codes[r * padded_dim + j] = row.vector().get(j).unwrap() as u8;
            }
            meta.push(row.meta());
        }
        (codes, meta)
    }

    /// Extract `(codes_i16 [nq × dim], meta padded to padded_nrows)` for the query
    /// side — i16 widening + row padding feeds `BlockTransposed`.
    fn query_arrays(
        mat: &Mat<MinMaxMeta<NBITS>>,
        dim: usize,
        padded_nrows: usize,
    ) -> (Vec<i16>, Vec<MinMaxCompensation>) {
        let nq = mat.num_vectors();
        let mut codes = vec![0i16; nq * dim];
        let mut meta = vec![MinMaxCompensation::default(); padded_nrows];
        for r in 0..nq {
            let row = mat.get_row(r).unwrap();
            for j in 0..dim {
                codes[r * dim + j] = i16::from(row.vector().get(j).unwrap() as u8);
            }
            meta[r] = row.meta();
        }
        (codes, meta)
    }

    /// (nq, nd, dim): every B-remainder class (`nd ∈ {1,5,6,7,8}`), A-panel
    /// remainder (`17`), multi-tile B (`1250`), and `dim ∈ {64,128,256}`.
    const CASES: &[(usize, usize, usize)] = &[
        (1, 1, 64),
        (1, 5, 64),
        (5, 1, 128),
        (16, 4, 64),
        (16, 5, 128),
        (16, 6, 64),
        (16, 7, 256),
        (16, 8, 128),
        (17, 9, 64),
        (32, 16, 256),
        (8, 1250, 128),
        (64, 1250, 64),
        // Odd dims exercise the even-K contract (padded_dim = dim + 1, the trailing
        // column zero-padded on both sides), end-to-end against the reference.
        (5, 3, 63),
        (17, 9, 65),
        (8, 33, 127),
        (16, 7, 1),
    ];

    /// The public quantized staged path must match the scalar MinMax `MaxSim`
    /// reference within tolerance (Stage A's integer dot is exact; only Stage B's
    /// f32 accumulation order differs from the reference's). Exercises the full
    /// public API: [`QuantStagedQuery`]/[`QuantStagedDocs`] build + compute.
    #[test]
    fn staged_i8_matches_minmax_reference() {
        if V3::new_checked().is_none() {
            return; // No AVX2 on this host.
        }

        for &(nq, nd, dim) in CASES {
            let q_data: Vec<f32> = (0..nq * dim).map(|i| rnd(1, i)).collect();
            let d_data: Vec<f32> = (0..nd * dim).map(|i| rnd(2, i)).collect();

            // ── Path A: the public quantized staged kernel. ──
            let q_f32 = MatRef::new(Standard::<f32>::new(nq, dim).unwrap(), &q_data).unwrap();
            let d_f32 = MatRef::new(Standard::<f32>::new(nd, dim).unwrap(), &d_data).unwrap();
            let mut query = QuantStagedQuery::build(q_f32).unwrap();
            let docs = QuantStagedDocs::build(d_f32);
            let mut got = vec![0.0f32; nq];
            query.compute_max_sim(&docs, &mut got);

            // ── Path B: scalar MinMax MaxSim reference (identical quantization). ──
            let q = quantizer(dim);
            let q_mat = quantize(&q, &q_data, nq, dim);
            let d_mat = quantize(&q, &d_data, nd, dim);
            let query_ref: QueryMatRef<_> = q_mat.as_view().into();
            let mut ref_scores = vec![0.0f32; nq];
            MaxSim::new(&mut ref_scores).evaluate(query_ref, d_mat.as_view());

            for i in 0..nq {
                assert!(
                    (got[i] - ref_scores[i]).abs() <= 1e-4 * ref_scores[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: staged-i8 min-dist {} != reference {}",
                    got[i],
                    ref_scores[i],
                );
            }
        }
    }

    /// Reusing a single [`QuantStagedQuery`] across multiple `compute_max_sim`
    /// calls (different doc sets / counts) must give correct results every time —
    /// the regression guard for the [`ResettableArena`](super::super::arena::ResettableArena)
    /// reset path: each call rewinds and re-fills the shared scratch, so a stale
    /// or aliased buffer would corrupt the second/third call.
    #[test]
    fn staged_i8_arena_reuse_across_calls() {
        if V3::new_checked().is_none() {
            return; // No AVX2 on this host.
        }

        const NQ: usize = 17; // exercises the A-panel row padding (17 -> 32)
        const DIM: usize = 128;
        let q_data: Vec<f32> = (0..NQ * DIM).map(|i| rnd(5, i)).collect();
        let q_f32 = MatRef::new(Standard::<f32>::new(NQ, DIM).unwrap(), &q_data).unwrap();
        let mut query = QuantStagedQuery::build(q_f32).unwrap();

        let quant = quantizer(DIM);
        let q_mat = quantize(&quant, &q_data, NQ, DIM);

        // Distinct doc counts (multi-tile, single panel, remainder) reusing the
        // same query — the arena is reset, never reallocated, between calls.
        for (call, &nd) in [251usize, 3, 64, 1].iter().enumerate() {
            let d_data: Vec<f32> = (0..nd * DIM).map(|i| rnd(6 + call as u64, i)).collect();
            let d_f32 = MatRef::new(Standard::<f32>::new(nd, DIM).unwrap(), &d_data).unwrap();
            let docs = QuantStagedDocs::build(d_f32);

            let mut got = vec![0.0f32; NQ];
            query.compute_max_sim(&docs, &mut got);

            let d_mat = quantize(&quant, &d_data, nd, DIM);
            let query_ref: QueryMatRef<_> = q_mat.as_view().into();
            let mut ref_scores = vec![0.0f32; NQ];
            MaxSim::new(&mut ref_scores).evaluate(query_ref, d_mat.as_view());

            for i in 0..NQ {
                assert!(
                    (got[i] - ref_scores[i]).abs() <= 1e-4 * ref_scores[i].abs().max(1.0),
                    "call {call} (nd={nd}) row {i}: reused staged-i8 {} != reference {}",
                    got[i],
                    ref_scores[i],
                );
            }
        }
    }

    /// Isolate Stage A: the raw `i32` partial it stores must equal the brute-force
    /// integer code dot `⟨codes_q, codes_d⟩` exactly (no float, no metadata).
    #[test]
    fn stage_a_integer_dot_exact() {
        let Some(arch) = V3::new_checked() else {
            return;
        };

        for &dim in &[64usize, 128, 130, 256] {
            let padded_dim = dim.next_multiple_of(2);
            let q = quantizer(dim);
            // Exactly one A-panel (16 rows) × one B-panel (4 cols).
            let q_data: Vec<f32> = (0..16 * dim).map(|i| rnd(3, i)).collect();
            let d_data: Vec<f32> = (0..4 * dim).map(|i| rnd(4, i)).collect();
            let q_mat = quantize(&q, &q_data, 16, dim);
            let d_mat = quantize(&q, &d_data, 4, dim);

            let (d_codes, _) = doc_soa(&d_mat, dim);
            let q_i16 = {
                let bt = BlockTransposed::<i16, 16, 2>::new(16, dim);
                let (c, _) = query_arrays(&q_mat, dim, bt.padded_nrows());
                c
            };
            let q_mat_view = MatRef::new(Standard::<i16>::new(16, dim).unwrap(), &q_i16).unwrap();
            let bt = BlockTransposed::<i16, 16, 2>::from_matrix_view(q_mat_view.as_matrix_view());

            let mut partial = vec![0i32; 16 * 4];
            // SAFETY: `bt` has exactly one block (16 rows) at `as_ptr()`; `d_codes`
            // is 4 rows × padded_dim u8; `partial` is 4 cols × 16 i32 at stride 16.
            unsafe {
                int_store_microkernel::<4>(
                    arch,
                    bt.as_ptr(),
                    d_codes.as_ptr(),
                    padded_dim,
                    partial.as_mut_ptr(),
                    16,
                );
            }

            // Brute-force ⟨codes⟩ over the logical dim (codes are u8 ∈ [0,15]).
            for i in 0..16 {
                let qr = q_mat.get_row(i).unwrap();
                for jcol in 0..4 {
                    let dr = d_mat.get_row(jcol).unwrap();
                    let expect: i32 = (0..dim)
                        .map(|d| {
                            i32::from(qr.vector().get(d).unwrap() as u8)
                                * i32::from(dr.vector().get(d).unwrap() as u8)
                        })
                        .sum();
                    assert_eq!(
                        partial[jcol * 16 + i],
                        expect,
                        "dim={dim} A-major partial[col {jcol}, row {i}] != brute force"
                    );
                }
            }
        }
    }

    /// Drive the internal entry with a deliberately tiny cache budget so the
    /// planner clamps to one A-panel and one B-panel per tile. With `nq > 16` and
    /// `nd > 4` this forces **multiple A-tiles and multiple B-tiles**, exercising
    /// the cross-tile `a_row_offset`/`b_row_offset` carry that the default-budget
    /// reference test (one giant A-tile) never reaches — yet still matching the
    /// scalar reference.
    #[test]
    fn staged_i8_multi_tile_tiny_budget() {
        let Some(arch) = V3::new_checked() else {
            return;
        };

        // l2_a / l1_b of 1 clamp `a_panels_per_tile` / `b_panels_per_tile` to 1
        // (both `.max(1)` in `StagedPlan::new`): a_tile_rows = 16, b_tile_rows = 4.
        let budget = TileBudget { l2_a: 1, l1_b: 1 };

        for &(nq, nd, dim) in &[(48usize, 22usize, 64usize), (33, 37, 128), (35, 19, 65)] {
            let q = quantizer(dim);
            let q_data: Vec<f32> = (0..nq * dim).map(|i| rnd(5, i)).collect();
            let d_data: Vec<f32> = (0..nd * dim).map(|i| rnd(6, i)).collect();
            let q_mat = quantize(&q, &q_data, nq, dim);
            let d_mat = quantize(&q, &d_data, nd, dim);

            let padded_dim = dim.next_multiple_of(2);
            let (d_codes, d_meta) = doc_soa(&d_mat, dim);
            let doc = MatRef::new(Standard::<u8>::new(nd, padded_dim).unwrap(), &d_codes).unwrap();

            let q_i16 = {
                let probe = BlockTransposed::<i16, 16, 2>::new(nq, dim);
                let (c, _) = query_arrays(&q_mat, dim, probe.padded_nrows());
                c
            };
            let q_view = MatRef::new(Standard::<i16>::new(nq, dim).unwrap(), &q_i16).unwrap();
            let query_bt = BlockTransposed::<i16, 16, 2>::from_matrix_view(q_view.as_matrix_view());
            let (_, q_meta) = query_arrays(&q_mat, dim, query_bt.padded_nrows());
            let post = MinMaxPostprocess::new(&q_meta, &d_meta, dim);

            let mut state = vec![f32::MIN; query_bt.padded_nrows()];
            max_ip_kernel_staged_i8(
                arch,
                query_bt.as_view(),
                doc,
                &post,
                &mut state,
                ScopedAllocator::global(),
                budget,
            );

            let query_ref: QueryMatRef<_> = q_mat.as_view().into();
            let mut ref_scores = vec![0.0f32; nq];
            MaxSim::new(&mut ref_scores).evaluate(query_ref, d_mat.as_view());

            for i in 0..nq {
                let got = -state[i];
                assert!(
                    (got - ref_scores[i]).abs() <= 1e-4 * ref_scores[i].abs().max(1.0),
                    "({nq},{nd},{dim}) row {i}: tiny-budget staged-i8 {got} != reference {}",
                    ref_scores[i],
                );
            }
        }
    }
}
