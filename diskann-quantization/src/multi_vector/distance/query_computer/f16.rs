// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! f16 [`DynQueryComputer`](super::DynQueryComputer) implementation and
//! architecture dispatch.

use diskann_wide::Architecture;

use super::super::kernels::f16::F16Entry;
use super::{DynQueryComputer, Prepared, QueryComputer, build_prepared};
use crate::multi_vector::block_transposed::{BlockTransposed, BlockTransposedRef};
use crate::multi_vector::matrix::{MatRef, Standard};
use diskann_utils::Reborrow;

impl QueryComputer<half::f16> {
    /// Build an f16 query computer, selecting the optimal architecture and
    /// GROUP for the current CPU at runtime.
    pub fn new(query: MatRef<'_, Standard<half::f16>>) -> Self {
        diskann_wide::arch::dispatch1_no_features(BuildComputer, query)
    }
}

impl<A, const GROUP: usize> DynQueryComputer<half::f16>
    for Prepared<A, BlockTransposed<half::f16, GROUP>>
where
    A: Architecture,
    F16Entry<GROUP>: for<'a> diskann_wide::arch::Target3<
            A,
            (),
            BlockTransposedRef<'a, half::f16, GROUP>,
            MatRef<'a, Standard<half::f16>>,
            &'a mut [f32],
        >,
{
    fn raw_kernel(&self, doc: MatRef<'_, Standard<half::f16>>, scratch: &mut [f32]) {
        self.arch
            .run3(F16Entry::<GROUP>, self.prepared.reborrow(), doc, scratch);
    }

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn ncols(&self) -> usize {
        self.ncols
    }

    fn padded_nrows(&self) -> usize {
        self.padded_nrows
    }
}

#[derive(Clone, Copy)]
pub(super) struct BuildComputer;

impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::Scalar,
        QueryComputer<half::f16>,
        MatRef<'_, Standard<half::f16>>,
    > for BuildComputer
{
    fn run(
        self,
        arch: diskann_wide::arch::Scalar,
        query: MatRef<'_, Standard<half::f16>>,
    ) -> QueryComputer<half::f16> {
        QueryComputer {
            inner: Box::new(build_prepared::<half::f16, _, 8>(arch, query)),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V3,
        QueryComputer<half::f16>,
        MatRef<'_, Standard<half::f16>>,
    > for BuildComputer
{
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V3,
        query: MatRef<'_, Standard<half::f16>>,
    ) -> QueryComputer<half::f16> {
        QueryComputer {
            inner: Box::new(build_prepared::<half::f16, _, 16>(arch, query)),
        }
    }
}

#[cfg(target_arch = "x86_64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::x86_64::V4,
        QueryComputer<half::f16>,
        MatRef<'_, Standard<half::f16>>,
    > for BuildComputer
{
    fn run(
        self,
        arch: diskann_wide::arch::x86_64::V4,
        query: MatRef<'_, Standard<half::f16>>,
    ) -> QueryComputer<half::f16> {
        let arch = arch.retarget();
        QueryComputer {
            inner: Box::new(build_prepared::<half::f16, _, 16>(arch, query)),
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl
    diskann_wide::arch::Target1<
        diskann_wide::arch::aarch64::Neon,
        QueryComputer<half::f16>,
        MatRef<'_, Standard<half::f16>>,
    > for BuildComputer
{
    fn run(
        self,
        arch: diskann_wide::arch::aarch64::Neon,
        query: MatRef<'_, Standard<half::f16>>,
    ) -> QueryComputer<half::f16> {
        let arch = arch.retarget();
        QueryComputer {
            inner: Box::new(build_prepared::<half::f16, _, 8>(arch, query)),
        }
    }
}
