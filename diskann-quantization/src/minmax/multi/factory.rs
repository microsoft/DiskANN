// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Factory for tiled MinMax-quantized multi-vector kernels.

use std::num::NonZeroUsize;

use diskann_wide::Architecture;
use diskann_wide::arch::Scalar;

#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::MinMaxMeta;
use super::kernel::{MinMaxErase, MinMaxMaxSimKernel};
use crate::matrix_kernels as mk;
use crate::minmax::MinMaxCompensation;
use crate::multi_vector::{MatRef, MaxSimError, MaxSimIsa, NotSupported};

#[derive(Debug)]
struct PackedQuery<const MR: usize> {
    nrows: usize,
    dim: usize,
    values: Vec<u8>,
    compensation: Vec<MinMaxCompensation>,
}

impl<const MR: usize> PackedQuery<MR> {
    fn new(query: MatRef<'_, MinMaxMeta<8>>) -> Self {
        let nrows = query.num_vectors();
        let dim = query.repr().intrinsic_dim();
        let mut values = vec![0; nrows.div_ceil(MR) * dim * MR];
        let mut compensation = Vec::with_capacity(nrows);

        for (row_index, row) in query.rows().enumerate() {
            compensation.push(row.meta());
            let block = row_index / MR;
            let lane = row_index % MR;
            let vector = row.vector();
            for k in 0..dim {
                // SAFETY: `k` is bounded by the common intrinsic dimension.
                values[(block * dim + k) * MR + lane] = unsafe { vector.get_unchecked(k) } as u8;
            }
        }

        Self {
            nrows,
            dim,
            values,
            compensation,
        }
    }
}

#[derive(Debug)]
struct Prepared<A, const MR: usize, const NR: usize> {
    arch: A,
    query: PackedQuery<MR>,
}

impl<A, const MR: usize, const NR: usize> Prepared<A, MR, NR>
where
    A: Architecture,
    for<'a> mk::maxsim::packed_u8_x_unpacked_u4::Driver<'a, A, MR, NR>: mk::Drive,
{
    fn run(&self, doc: MatRef<'_, MinMaxMeta<4>>, scores: &mut [f32]) -> Result<(), MaxSimError> {
        if scores.len() != self.query.nrows {
            return Err(MaxSimError::InvalidBufferLength(
                scores.len(),
                self.query.nrows,
            ));
        }
        if doc.repr().intrinsic_dim() != self.query.dim {
            return Err(MaxSimError::UnequalDim(
                doc.repr().intrinsic_dim(),
                self.query.dim,
            ));
        }
        if doc.num_vectors() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }

        let Some(k) = NonZeroUsize::new(self.query.dim).map(mk::DimK::new) else {
            scores.fill(0.0);
            return Ok(());
        };
        let mut driver = mk::maxsim::packed_u8_x_unpacked_u4::Driver::<_, MR, NR>::new(
            self.arch,
            &self.query.values,
            &self.query.compensation,
            self.query.nrows,
            doc,
            scores,
            k,
        );
        mk::Drive::drive(&mut driver);
        Ok(())
    }
}

macro_rules! impl_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl MinMaxMaxSimKernel<8, 4> for Prepared<$arch, $mr, $nr> {
            fn nrows(&self) -> usize {
                self.query.nrows
            }

            fn compute_max_sim(
                &self,
                doc: MatRef<'_, MinMaxMeta<4>>,
                scores: &mut [f32],
            ) -> Result<(), MaxSimError> {
                self.run(doc, scores)
            }
        }
    };
}

impl_kernel!(Scalar, 8, 6);
#[cfg(target_arch = "aarch64")]
impl_kernel!(Neon, 8, 8);
#[cfg(target_arch = "x86_64")]
impl_kernel!(V3, 16, 6);
#[cfg(target_arch = "x86_64")]
impl_kernel!(V4, 16, 6);

struct BuildAndErase<E>(E);

macro_rules! impl_builder {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl<E> diskann_wide::arch::Target1<$arch, E::Output, MatRef<'_, MinMaxMeta<8>>>
            for BuildAndErase<E>
        where
            E: MinMaxErase<8, 4>,
        {
            fn run(self, arch: $arch, query: MatRef<'_, MinMaxMeta<8>>) -> E::Output {
                self.0.erase(Prepared::<_, $mr, $nr> {
                    arch,
                    query: PackedQuery::new(query),
                })
            }
        }
    };
}

impl_builder!(Scalar, 8, 6);
#[cfg(target_arch = "aarch64")]
impl_builder!(Neon, 8, 8);
#[cfg(target_arch = "x86_64")]
impl_builder!(V3, 16, 6);
#[cfg(target_arch = "x86_64")]
impl_builder!(V4, 16, 6);

/// Build a tiled MinMax8-query by MinMax4-document MaxSim kernel.
///
/// # Errors
///
/// Returns [`NotSupported`] if the requested architecture is unavailable.
pub fn build_minmax_max_sim<E>(
    isa: MaxSimIsa,
    query: MatRef<'_, MinMaxMeta<8>>,
    erase: E,
) -> Result<E::Output, NotSupported>
where
    E: MinMaxErase<8, 4>,
{
    match isa {
        MaxSimIsa::Auto => Ok(diskann_wide::arch::dispatch1_no_features(
            BuildAndErase(erase),
            query,
        )),
        MaxSimIsa::Scalar => Ok(Scalar::new().run1(BuildAndErase(erase), query)),
        #[cfg(target_arch = "x86_64")]
        MaxSimIsa::X86_64_V3 => {
            let arch = V3::new_checked().ok_or(NotSupported {
                isa,
                reason: "AVX2/FMA unavailable on this CPU",
            })?;
            Ok(arch.run1(BuildAndErase(erase), query))
        }
        #[cfg(target_arch = "x86_64")]
        MaxSimIsa::X86_64_V4 => {
            let arch = V4::new_checked().ok_or(NotSupported {
                isa,
                reason: "AVX-512 unavailable on this CPU",
            })?;
            Ok(arch.run1(BuildAndErase(erase), query))
        }
        #[cfg(not(target_arch = "x86_64"))]
        MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => Err(NotSupported {
            isa,
            reason: "x86_64 target only",
        }),
        #[cfg(target_arch = "aarch64")]
        MaxSimIsa::Neon => {
            let arch = Neon::new_checked().ok_or(NotSupported {
                isa,
                reason: "Neon unavailable on this CPU",
            })?;
            Ok(arch.run1(BuildAndErase(erase), query))
        }
        #[cfg(not(target_arch = "aarch64"))]
        MaxSimIsa::Neon => Err(NotSupported {
            isa,
            reason: "aarch64 target only",
        }),
        MaxSimIsa::Reference => Err(NotSupported {
            isa,
            reason: "reference kernel unavailable",
        }),
    }
}

#[cfg(test)]
mod tests {
    use diskann_utils::ReborrowMut;
    use diskann_vector::DistanceFunctionMut;

    use super::*;
    use crate::CompressInto;
    use crate::algorithms::{Transform, transforms::NullTransform};
    use crate::minmax::MinMaxQuantizer;
    use crate::multi_vector::{BoxErase, Defaulted, Mat, MaxSim, QueryMatRef, Standard};
    use crate::num::Positive;

    fn check(isa: MaxSimIsa) {
        for (query_rows, doc_rows, dim) in
            [(1, 1, 1), (7, 5, 3), (8, 6, 8), (9, 7, 17), (17, 13, 64)]
        {
            let quantizer = MinMaxQuantizer::new(
                Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
                Positive::new(1.0).unwrap(),
            );
            let query_values: Vec<f32> = (0..query_rows * dim)
                .map(|i| ((i * 17 + 3) % 101) as f32 / 13.0)
                .collect();
            let doc_values: Vec<f32> = (0..doc_rows * dim)
                .map(|i| ((i * 29 + 7) % 97) as f32 / 11.0)
                .collect();

            let query_input =
                MatRef::new(Standard::new(query_rows, dim).unwrap(), &query_values).unwrap();
            let doc_input =
                MatRef::new(Standard::new(doc_rows, dim).unwrap(), &doc_values).unwrap();
            let mut query = Mat::new(MinMaxMeta::<8>::new(query_rows, dim), Defaulted).unwrap();
            let mut docs = Mat::new(MinMaxMeta::<4>::new(doc_rows, dim), Defaulted).unwrap();
            quantizer
                .compress_into(query_input, query.reborrow_mut())
                .unwrap();
            quantizer
                .compress_into(doc_input, docs.reborrow_mut())
                .unwrap();

            let mut expected = vec![0.0; query_rows];
            MaxSim::new(&mut expected).evaluate(QueryMatRef::from(query.as_view()), docs.as_view());

            let kernel = build_minmax_max_sim(isa, query.as_view(), BoxErase).unwrap();
            let mut actual = vec![0.0; query_rows];
            kernel.compute_max_sim(docs.as_view(), &mut actual).unwrap();
            assert_eq!(
                actual, expected,
                "mismatch for ({query_rows}, {doc_rows}, {dim}) using {isa:?}"
            );
        }
    }

    #[test]
    fn scalar_matches_existing_max_sim() {
        check(MaxSimIsa::Scalar);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_existing_max_sim() {
        check(MaxSimIsa::Neon);
    }

    #[test]
    fn auto_matches_existing_max_sim() {
        check(MaxSimIsa::Auto);
    }
}
