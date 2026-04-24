// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 dispatch adapter for block-transposed multi-vector distance.
//!
//! Reuses the f32 micro-kernel family with tile-level f16→f32 conversion
//! via [`ConvertTo`](super::layouts::ConvertTo). No f16-specific micro-kernel
//! code is needed — the [`F32Kernel`](super::f32::F32Kernel) does all the
//! SIMD work after conversion.
//!
//! Conversion from f16 to f32 is performed at tile granularity via
//! [`SliceCast`](diskann_vector::conversion::SliceCast), dispatched through
//! the runtime architecture token — the same SIMD level used by the
//! micro-kernel.

use diskann_wide::Architecture;

use super::Kernel;
use super::TileBudget;
use super::f32::{F32Kernel, max_ip_kernel};
use super::layouts;
use crate::multi_vector::{BlockTransposedRef, MatRef, Standard};

pub(crate) struct F16Entry<const GROUP: usize>;

impl<A, const GROUP: usize>
    diskann_wide::arch::Target3<
        A,
        (),
        BlockTransposedRef<'_, half::f16, GROUP>,
        MatRef<'_, Standard<half::f16>>,
        &mut [f32],
    > for F16Entry<GROUP>
where
    A: Architecture,
    F32Kernel<GROUP>: Kernel<A>,
    layouts::BlockTransposed<half::f16, GROUP>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Left>
        + layouts::Layout<Element = half::f16>,
    layouts::RowMajor<half::f16>: layouts::ConvertTo<A, <F32Kernel<GROUP> as Kernel<A>>::Right>
        + layouts::Layout<Element = half::f16>,
{
    #[inline(always)]
    fn run(
        self,
        arch: A,
        lhs: BlockTransposedRef<'_, half::f16, GROUP>,
        rhs: MatRef<'_, Standard<half::f16>>,
        scratch: &mut [f32],
    ) {
        max_ip_kernel(arch, lhs, rhs, scratch, TileBudget::default());
    }
}
