/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::Scalar;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::kernel::{MinMaxMaxSimKernel, validate};
use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::{
        self as mk,
        maxsim::packed_minmax_x_unpacked_minmax::{Driver, PackedA},
    },
    minmax::MinMaxMeta,
    multi_vector::{MatRef, MaxSimError, Overflow},
};

// Fix ISA and packing layout, leaving only the input bit widths generic.
macro_rules! prepared_kernel {
    ($name:ident, $arch:ty, $element:ty, $mr:literal, $nr:literal, $pack:literal) => {
        #[derive(Debug)]
        pub(super) struct $name<const A_BITS: usize, const B_BITS: usize>
        where
            Unsigned: Representation<A_BITS>,
        {
            arch: $arch,
            query: PackedA<$element, A_BITS, $mr, $pack>,
            rows: usize,
            dim: usize,
        }

        impl<const A_BITS: usize, const B_BITS: usize> $name<A_BITS, B_BITS>
        where
            Unsigned: Representation<A_BITS> + Representation<B_BITS>,
        {
            pub(super) fn new(
                arch: $arch,
                query: MatRef<'_, MinMaxMeta<A_BITS>>,
            ) -> Result<Self, Overflow> {
                Ok(Self {
                    arch,
                    query: PackedA::new(query)?,
                    rows: query.num_vectors(),
                    dim: query.repr().intrinsic_dim(),
                })
            }
        }

        impl<const A_BITS: usize, const B_BITS: usize> MinMaxMaxSimKernel<B_BITS>
            for $name<A_BITS, B_BITS>
        where
            Unsigned: Representation<A_BITS> + Representation<B_BITS>,
        {
            fn nrows(&self) -> usize {
                self.rows
            }

            fn compute_max_sim(
                &self,
                document: MatRef<'_, MinMaxMeta<B_BITS>>,
                scores: &mut [f32],
            ) -> Result<(), MaxSimError> {
                if !validate(self.rows, self.dim, document, scores)? {
                    return Ok(());
                }
                // SAFETY: Validation establishes nonempty A, positive K, matching dimensions
                // and exactly one output per query row. Packing and ISA selection are fixed.
                let mut driver = unsafe {
                    Driver::<$arch, B_BITS, $mr, $nr, $pack>::new(
                        self.arch,
                        &self.query,
                        document,
                        scores,
                        mk::Cache::detect(),
                    )
                };
                mk::Drive::drive(&mut driver);
                for score in scores {
                    *score = (-*score).min(f32::MAX);
                }
                Ok(())
            }
        }
    };
}
prepared_kernel!(ScalarBytes, Scalar, u8, 16, 2, 4);
prepared_kernel!(ScalarWords, Scalar, i16, 8, 2, 2);
#[cfg(target_arch = "x86_64")]
prepared_kernel!(V3Bytes, V3, u8, 16, 4, 4);
#[cfg(target_arch = "x86_64")]
prepared_kernel!(V3Words, V3, i16, 16, 4, 2);
#[cfg(target_arch = "x86_64")]
prepared_kernel!(V4Bytes, V4, u8, 16, 8, 4);
#[cfg(target_arch = "x86_64")]
prepared_kernel!(V4Words, V4, i16, 16, 8, 2);
#[cfg(target_arch = "aarch64")]
prepared_kernel!(NeonBytes, Neon, u8, 8, 4, 4);
