// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Stage B / Stage C impls for MaxSim, plus the owned reset-arena scratch for the
//! `MaxSimKernel` (f32) path.
//!
//! The Stage-A kernel and the V3 entry point live in [`super::v3`]; the
//! Stage-C reducer's SIMD fold is V3-specific and also lives there.

use core::marker::PhantomData;

use diskann_wide::Architecture;

use super::super::TileBudget;
use super::arena::ResettableArena;
use super::{FoldCtx, Postprocess};
use crate::alloc::ScopedAllocator;

// ── Stage B: identity ────────────────────────────────────────────

/// Identity postprocess: the raw inner product (`Acc`) *is* the `Score`.
///
/// Reports [`scratch_len`](Postprocess::scratch_len) 0 and
/// [`apply`](Postprocess::apply) returns its input pointer unchanged, so the
/// driver folds `partial_buf` directly — no `scored_buf`, no extra pass, no
/// boolean flag. (Mirrors [`ConvertTo`](super::super::layouts::ConvertTo)'s
/// zero-cost identity blanket impl.)
pub(super) struct Identity<T>(PhantomData<T>);

impl<T> Identity<T> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

// SAFETY: identity reads nothing beyond `acc`, writes nothing (`scratch_len`
// is 0), and returns exactly `acc` — `Score == Acc == T`, already A-major at the
// fixed output stride `a_panel`. The returned pointer carries the caller's
// validity for `acc` unchanged.
unsafe impl<A: Architecture, T: Copy> Postprocess<A> for Identity<T> {
    type Acc = T;
    type Score = T;

    #[inline(always)]
    fn scratch_len(&self, _a_panel: usize, _max_b_cols: usize) -> usize {
        0
    }

    #[inline(always)]
    unsafe fn apply(&self, _scratch: *mut T, _arch: A, acc: *const T, _ctx: FoldCtx) -> *const T {
        acc
    }
}

/// MaxSim reducer: per-A-row running maximum of the inner products. The
/// `Reducer` impl (a register-resident `max_simd` sweep) is V3-specific and
/// lives in [`super::v3`].
pub(super) struct MaxReducer;

// ── Dispatch carrier ─────────────────────────────────────────────

/// `state` (the output running-reduction) + `alloc` (the caller's allocator the
/// driver carves its internal `partial`/`scored` scratch from) bundled to cross
/// the [`Target3`](diskann_wide::arch::Target3) dispatch boundary. Only the
/// allocator crosses here — the driver allocates the scratch buffers itself, so
/// the caller hands in nothing but `state` and `alloc`.
pub(crate) struct StagedRun<'a> {
    pub(crate) state: &'a mut [f32],
    pub(crate) alloc: ScopedAllocator<'a>,
}

// ── Owned reset-arena scratch for the f32 `MaxSimKernel` path ─────

/// Per-kernel reusable scratch for the staged f32 [`MaxSimKernel`](super::super::MaxSimKernel)
/// path: the running-reduction `state` output plus a [`ResettableArena`] backing
/// the driver's transient `partial`/`scored`. Owned by `PreparedStaged` (behind a
/// `RefCell`, since `compute_max_sim` is `&self`) so steady-state calls allocate
/// nothing — the arena is reset, not reallocated, each call.
///
/// This is the f32 counterpart to the i8 path's `QuantStagedQuery`-owned arena;
/// it lives here (not in `factory`) because sizing reads [`TileBudget`], which is
/// private to the `kernels` module tree.
#[derive(Debug)]
pub(crate) struct F32StagedScratch {
    state: Vec<f32>,
    arena: ResettableArena,
}

impl F32StagedScratch {
    /// Build scratch for a query of `padded` rows. The arena is sized once to the
    /// provable `2·l1_b` ceiling (`StagedPlan` co-budgets `partial`/`scored` each
    /// `≤ l1_b`), so it never needs per-shape sizing or reallocation.
    #[allow(clippy::expect_used)] // POC: 72 KB arena, OOM is not a recoverable case here.
    pub(crate) fn new(padded: usize) -> Self {
        let arena_bytes = 2 * TileBudget::default().l1_b + 4096;
        Self {
            state: vec![f32::MIN; padded],
            arena: ResettableArena::with_capacity(arena_bytes).expect("f32 staged arena"),
        }
    }

    /// Reset the arena, ensure `state` covers `padded` rows, then run `f` with the
    /// `state` slice (the driver re-initialises it) and a `ScopedAllocator` over
    /// the arena.
    pub(crate) fn run<R>(
        &mut self,
        padded: usize,
        f: impl FnOnce(&mut [f32], ScopedAllocator<'_>) -> R,
    ) -> R {
        if self.state.len() < padded {
            self.state.resize(padded, f32::MIN);
        }
        // Split the borrow so `state` (mut) and `arena` (shared, via the allocator)
        // are simultaneously live as disjoint fields.
        let Self { state, arena } = self;
        arena.reset();
        f(&mut state[..padded], ScopedAllocator::new(arena))
    }
}

#[cfg(test)]
mod tests {
    use diskann_wide::arch::Scalar;

    use super::*;

    /// The identity postprocess returns its input pointer unchanged (the
    /// zero-cost identity), so the driver folds `partial_buf` directly.
    #[test]
    fn identity_returns_source_pointer() {
        let acc = [1.0f32, 2.0, 3.0, 4.0];
        let id = Identity::<f32>::new();
        let ctx = FoldCtx {
            a_panel: 2,
            valid_b_cols: 2,
            b_stride: 2,
            a_row_offset: 0,
            b_row_offset: 0,
        };
        // SAFETY: identity ignores `scratch` (its `scratch_len` is 0) and returns
        // its input pointer unchanged. UFCS pins `A = Scalar` (the impl is blanket
        // over `A`); the real driver pins `A = V3` via turbofish, so production
        // never needs this.
        let out = unsafe {
            <Identity<f32> as Postprocess<Scalar>>::apply(
                &id,
                core::ptr::null_mut(),
                Scalar::new(),
                acc.as_ptr(),
                ctx,
            )
        };
        assert_eq!(out, acc.as_ptr());
    }
}
