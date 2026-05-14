/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! **Template for an experimental multi-vector kernel.**
//!
//! Copy this file (e.g. to `v4_wide.rs`), rename `Template*`, change the
//! `Kernel<A>` impl to your target ISA, and add an `Arch` variant + a
//! `register_regression` call to wire it up.
//!
//! # The 5-step workflow
//!
//! 1. **Add an [`Arch`](crate::inputs::multi_vector::Arch) variant** for your
//!    experimental kernel (e.g. `X86_64_V4_Wide`). The `#[non_exhaustive]`
//!    attribute on `Arch` makes this a non-breaking addition.
//! 2. **Author the micro-kernel.** Implement
//!    [`Kernel<A>`](diskann_quantization::multi_vector::distance::kernels::Kernel)
//!    on your struct (`unsafe impl Kernel<V4>` etc.), filling in
//!    `full_panel` and `partial_panel` with your SIMD intrinsics.
//! 3. **Author the adapter.** Implement
//!    [`DynQueryComputer<T>`](diskann_quantization::multi_vector::distance::DynQueryComputer)
//!    on a struct that owns the prepared query data; in `compute_max_sim`,
//!    call
//!    [`tiled_reduce`](diskann_quantization::multi_vector::distance::kernels::tiled_reduce)
//!    with your kernel.
//! 4. **Add a marker + `DispatchRule<Arch>`.** Mirror the pattern in
//!    `library_kernels.rs` (e.g. `match_arch_x86_64!`) for your new variant.
//! 5. **Add a `RunBenchmark<Marker>` impl + `register_regression` call.** Use
//!    `Kernel::<Marker, T>::new()` as the registered benchmark entry.
//!
//! Then validate under Miri before treating the kernel as correct — see the
//! section below.
//!
//! # Validating under Miri (REQUIRED)
//!
//! Experimental kernels rely on `unsafe fn full_panel` / `partial_panel`
//! with raw-pointer arithmetic. Pointer provenance, alignment, and
//! out-of-bounds bugs are easy to introduce and hard to catch by
//! inspection. **Run your kernel under Miri before assuming it's correct.**
//!
//! Rules:
//!
//! - Inside your `#[cfg(test)]` module, construct arch tokens via the
//!   Miri-friendly variants: `Scalar::new()` (always Miri-safe) or
//!   `V4::new_checked_miri()` (returns a token unconditionally under
//!   `cfg(miri)` using AVX-512 emulation, so tests run even when Miri
//!   can't do real CPU detection). `V3` and `Neon` only expose
//!   `new_checked()` today — if you need them under Miri, follow
//!   `V4::new_checked_miri()`'s pattern in `diskann-wide`.
//! - Any SIMD intrinsic Miri doesn't support must have a scalar fallback
//!   gated by `#[cfg(miri)]`.
//! - Add at least one small-shape correctness test that runs your kernel
//!   against a naive reference and is Miri-friendly.
//! - Run: `cargo +nightly miri test -p diskann-benchmark --features multi-vector
//!   backend::multi_vector::experimental::<your_kernel>`. Reduce
//!   test-sweep size under Miri with `if cfg!(miri) { small } else { full }`
//!   (see this file's test for the pattern).
//!
//! Miri won't catch performance bugs, but it'll catch UB — and UB in an
//! experimental kernel breaks the benchmark binary, not the kernel you're
//! trying to measure.
//!
//! # This template
//!
//! This file defines `TemplateKernel: Kernel<Scalar>` (uses `Scalar` so the
//! template is host-portable + Miri-friendly) and a `TemplateComputer`
//! adapter that pipes it through `tiled_reduce`. It is **not registered** as
//! a benchmark entry — see step 5 in the workflow. The included
//! `#[cfg(test)]` `template_matches_pinned_scalar` test exercises the API
//! surface end-to-end so this file catches public-API drift even though it
//! isn't wired into the benchmark dispatcher.

#![allow(dead_code)]

use diskann_quantization::multi_vector::distance::{
    kernels::{layouts, tiled_reduce, Kernel, TileBudget},
    DynQueryComputer,
};
use diskann_quantization::multi_vector::{BlockTransposed, BlockTransposedRef, MatRef, Standard};
use diskann_wide::arch::Scalar;

/// Step 2: the micro-kernel struct. Rename and implement for your target arch.
pub(super) struct TemplateKernel;

// SAFETY: `full_panel` / `partial_panel` only access `A_PANEL * k` /
// `B_PANEL * k` source elements and write `A_PANEL` destination f32s,
// matching `Kernel<Scalar>`'s safety contract. The simple scalar
// computation here is Miri-clean.
unsafe impl Kernel<Scalar> for TemplateKernel {
    type Left = layouts::BlockTransposedLayout<f32, 8>;
    type Right = layouts::RowMajor<f32>;
    const A_PANEL: usize = 8;
    const B_PANEL: usize = 2;

    unsafe fn full_panel(_arch: Scalar, a: *const f32, b: *const f32, k: usize, r: *mut f32) {
        // SAFETY: a covers A_PANEL * k contiguous block-transposed f32s,
        // b covers B_PANEL * k contiguous row-major f32s, r covers A_PANEL f32s.
        unsafe { panel::<8, 2>(a, b, k, r) }
    }

    unsafe fn partial_panel(
        _arch: Scalar,
        remainder: usize,
        a: *const f32,
        b: *const f32,
        k: usize,
        r: *mut f32,
    ) {
        debug_assert!(remainder == 1);
        // SAFETY: as full_panel but with `b` covering `remainder * k` f32s.
        unsafe { panel::<8, 1>(a, b, k, r) }
    }
}

/// Replace this with your SIMD intrinsics. The block-transposed A layout
/// stores `A_ROWS` contiguous f32s per dimension index `i`, so the q-th
/// query row at dimension i lives at `a[i * A_ROWS + q]`. The row-major B
/// layout stores doc d's k-th element at `b[d * k_dim + k]`. The scratch
/// `r` accumulates max IP per query row (library convention; the
/// `QueryComputer` veneer negates at the end).
///
/// # Safety
/// - `a` covers `A_ROWS * k` block-transposed f32s.
/// - `b` covers `B_ROWS * k` row-major f32s.
/// - `r` covers `A_ROWS` writable f32s.
unsafe fn panel<const A_ROWS: usize, const B_ROWS: usize>(
    a: *const f32,
    b: *const f32,
    k: usize,
    r: *mut f32,
) {
    for q in 0..A_ROWS {
        // SAFETY: q < A_ROWS.
        let mut best = unsafe { *r.add(q) };

        for d in 0..B_ROWS {
            let mut ip: f32 = 0.0;
            for i in 0..k {
                // SAFETY: i < k, q < A_ROWS.
                let a_val = unsafe { *a.add(i * A_ROWS + q) };
                // SAFETY: d < B_ROWS, b covers B_ROWS rows of k f32s each.
                let b_val = unsafe { *b.add(d * k + i) };
                ip += a_val * b_val;
            }
            best = best.max(ip);
        }

        // SAFETY: q < A_ROWS.
        unsafe { *r.add(q) = best };
    }
}

/// Step 3: the `DynQueryComputer<T>` adapter. Owns the prepared query data
/// and routes `compute_max_sim` through `tiled_reduce` with the kernel.
#[derive(Debug)]
pub(super) struct TemplateComputer {
    arch: Scalar,
    prepared: BlockTransposed<f32, 8>,
}

impl TemplateComputer {
    pub(super) fn new(query: MatRef<'_, Standard<f32>>) -> Self {
        let prepared = BlockTransposed::<f32, 8>::from_matrix_view(query.as_matrix_view());
        Self {
            arch: Scalar::new(),
            prepared,
        }
    }
}

impl DynQueryComputer<f32> for TemplateComputer {
    fn nrows(&self) -> usize {
        self.prepared.nrows()
    }

    fn compute_max_sim(&self, doc: MatRef<'_, Standard<f32>>, scores: &mut [f32]) {
        let mut scratch = vec![f32::MIN; self.prepared.padded_nrows()];
        let prepared_ref: BlockTransposedRef<'_, f32, 8> = self.prepared.as_view();
        let ca = <BlockTransposedRef<'_, f32, 8> as layouts::DescribeLayout>::layout(&prepared_ref);
        let cb = <MatRef<'_, Standard<f32>> as layouts::DescribeLayout>::layout(&doc);

        // SAFETY: prepared.as_ptr() covers padded_nrows * ncols block-transposed
        // f32s; doc.as_slice() covers num_vectors * vector_dim row-major f32s;
        // scratch length == padded_nrows; padded_nrows is a multiple of
        // A_PANEL=8 by BlockTransposed construction.
        unsafe {
            tiled_reduce::<Scalar, TemplateKernel, _, _>(
                self.arch,
                &ca,
                &cb,
                self.prepared.as_ptr(),
                self.prepared.padded_nrows(),
                doc.as_slice().as_ptr(),
                doc.num_vectors(),
                doc.vector_dim(),
                &mut scratch,
                TileBudget::default(),
            );
        }

        for (dst, &src) in scores.iter_mut().zip(&scratch[..self.prepared.nrows()]) {
            *dst = -src;
        }
    }
}

#[cfg(test)]
mod tests {
    //! Miri-friendly correctness test for the template kernel.
    //!
    //! Validates that the template's adapter machinery produces the same
    //! per-row scores as `QueryComputer::from_arch(Scalar)`. Iteration
    //! count is reduced under Miri so `cargo +nightly miri test` finishes
    //! in seconds, not minutes.
    use super::*;
    use diskann_quantization::multi_vector::QueryComputer;

    fn make_data(nrows: usize, ncols: usize, shift: usize) -> Vec<f32> {
        (0..nrows * ncols)
            .map(|v| ((v + shift) % ncols) as f32)
            .collect()
    }

    #[test]
    fn template_matches_pinned_scalar() {
        let cases: &[(usize, usize, usize)] = if cfg!(miri) {
            // Single small case under Miri to keep runtime reasonable.
            &[(3, 4, 8)]
        } else {
            &[(1, 1, 4), (3, 5, 8), (8, 4, 16), (10, 6, 32)]
        };

        for &(nq, nd, dim) in cases {
            let qd = make_data(nq, dim, dim / 2);
            let dd = make_data(nd, dim, dim);
            let query = MatRef::new(Standard::<f32>::new(nq, dim).unwrap(), &qd).unwrap();
            let doc = MatRef::new(Standard::<f32>::new(nd, dim).unwrap(), &dd).unwrap();

            let pinned = QueryComputer::<f32>::from_arch(query, Scalar::new());
            let template = QueryComputer::<f32>::from_dyn(Box::new(TemplateComputer::new(query)));

            let mut pinned_scores = vec![0.0f32; nq];
            let mut template_scores = vec![0.0f32; nq];
            pinned.max_sim(doc, &mut pinned_scores);
            template.max_sim(doc, &mut template_scores);

            for (i, (p, t)) in pinned_scores.iter().zip(template_scores.iter()).enumerate() {
                assert!(
                    (p - t).abs() < 1e-10,
                    "shape ({nq},{nd},{dim}) row {i}: pinned={p} template={t}",
                );
            }
        }
    }
}
