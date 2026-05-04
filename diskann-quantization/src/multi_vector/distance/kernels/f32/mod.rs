// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f32 micro-kernel family for block-transposed multi-vector distance.
//!
//! Provides:
//!
//! - `F32Kernel<GROUP>` — zero-sized marker type selecting the f32 micro-kernel
//!   for `BlockTransposed<f32, GROUP>` data.
//! - [`max_ip_kernel`] — architecture-, element-type-, and GROUP-generic entry point
//!   for the reducing max-IP GEMM. Accepts any element type `T` for which
//!   [`ConvertTo`](super::layouts::ConvertTo) impls exist (identity for f32,
//!   SIMD-accelerated f16→f32, etc.).
//!
//! # Architecture-specific micro-kernels
//!
//! - `v3` (x86_64) — V3 (AVX2+FMA) 16×4 micro-kernel (GROUP=16). V4 delegates to V3 at dispatch.
//! - `scalar` — Emulated 8×2 micro-kernel (GROUP=8). Neon delegates to Scalar at dispatch.

use diskann_wide::Architecture;

use super::Kernel;
use super::TileBudget;
use super::layouts::{self, DescribeLayout};
use super::tiled_reduce::tiled_reduce;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

mod scalar;
#[cfg(target_arch = "x86_64")]
mod v3;

/// Zero-sized kernel type for f32 micro-kernels with block size `GROUP`.
pub(crate) struct F32Kernel<const GROUP: usize>;

#[inline(never)]
#[cold]
#[allow(clippy::panic)]
fn max_ip_kernel_panic(scratch_len: usize, padded_nrows: usize, a_ncols: usize, b_dim: usize) {
    panic!(
        "max_ip_kernel: precondition failed: \
         scratch.len()={scratch_len} (expected {padded_nrows}), \
         a.ncols()={a_ncols}, b.vector_dim()={b_dim}"
    );
}

/// Compute the reducing max-IP GEMM between a block-transposed A matrix and
/// a row-major B matrix, writing per-A-row max similarities into `scratch`.
///
/// Thin wrapper over [`tiled_reduce`] using `F32Kernel<GROUP>` for the
/// requested architecture. The element type `T` can be any `Copy` type with
/// matching [`ConvertTo`](super::layouts::ConvertTo) impls (zero-cost for
/// `T = f32`; SIMD f16→f32 conversion once per tile for `T = half::f16`).
///
/// `scratch` must have length [`BlockTransposedRef::padded_nrows()`] and be
/// initialized to `f32::MIN` before the first call. On return, `scratch[i]`
/// holds the maximum inner product between A row `i` and any B row.
///
/// # Panics
///
/// Panics if `scratch.len() != a.padded_nrows()` or `a.ncols() != b.vector_dim()`.
pub(super) fn max_ip_kernel<A: Architecture, T: Copy, const GROUP: usize>(
    arch: A,
    a: BlockTransposedRef<'_, T, GROUP>,
    b: MatRef<'_, Standard<T>>,
    scratch: &mut [f32],
    budget: TileBudget,
) where
    F32Kernel<GROUP>: Kernel<A>,
    layouts::BlockTransposed<T, GROUP>:
        layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Left> + layouts::Layout<Element = T>,
    layouts::RowMajor<T>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Right>
        + layouts::Layout<Element = T>,
{
    if scratch.len() != a.padded_nrows() || a.ncols() != b.vector_dim() {
        max_ip_kernel_panic(scratch.len(), a.padded_nrows(), a.ncols(), b.vector_dim());
    }

    let k = a.ncols();
    let b_nrows = b.num_vectors();

    // Compile-time: A_PANEL must equal GROUP for block-transposed layout correctness.
    const { assert!(<F32Kernel<GROUP> as Kernel<A>>::A_PANEL == GROUP) }

    let ca = a.layout();
    let cb = b.layout();

    // SAFETY:
    // - a.as_ptr() is valid for a.padded_nrows() * k elements of T.
    // - MatRef<Standard<T>> stores nrows * ncols contiguous T elements.
    // - scratch.len() == a.padded_nrows() (checked above).
    // - a.padded_nrows() is always a multiple of GROUP, and the const assert above
    //   verifies A_PANEL == GROUP at compile time.
    unsafe {
        tiled_reduce::<A, F32Kernel<GROUP>, _, _>(
            arch,
            &ca,
            &cb,
            a.as_ptr(),
            a.padded_nrows(),
            b.as_slice().as_ptr(),
            b_nrows,
            k,
            scratch,
            budget,
        );
    }
}

impl<A, const GROUP: usize>
    diskann_wide::arch::Target3<
        A,
        (),
        BlockTransposedRef<'_, f32, GROUP>,
        MatRef<'_, Standard<f32>>,
        &mut [f32],
    > for F32Kernel<GROUP>
where
    A: Architecture,
    Self: Kernel<A>,
    layouts::BlockTransposed<f32, GROUP>:
        layouts::ConvertTo<A, <Self as Kernel<A>>::Left> + layouts::Layout<Element = f32>,
    layouts::RowMajor<f32>:
        layouts::ConvertTo<A, <Self as Kernel<A>>::Right> + layouts::Layout<Element = f32>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, f32, GROUP>,
        rhs: MatRef<'_, Standard<f32>>,
        scratch: &mut [f32],
    ) {
        max_ip_kernel(arch, lhs, rhs, scratch, TileBudget::default());
    }
}
