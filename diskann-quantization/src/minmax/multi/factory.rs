/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::kernel::{MinMaxErase, MinMaxMaxSimKernel};
use super::query::PackedQuery;
use crate::{
    matrix_kernels::{
        self as mk,
        maxsim::minmax8_x_minmax4::{Driver, b::RawRows, layout::MAX_DIM},
    },
    minmax::MinMaxMeta,
    multi_vector::{MatRef, MaxSimError, MaxSimIsa, NotSupported, Overflow},
};
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::{Architecture, arch::Scalar};

#[derive(Debug)]
struct Prepared<A, const MR: usize, const NR: usize, const PACK: usize> {
    arch: A,
    query: PackedQuery<MR, PACK>,
}

impl<A, const MR: usize, const NR: usize, const PACK: usize> MinMaxMaxSimKernel<8, 4>
    for Prepared<A, MR, NR, PACK>
where
    A: Architecture,
    for<'a> Driver<'a, A, MR, NR, PACK>: mk::Drive,
{
    fn nrows(&self) -> usize {
        self.query.rows()
    }
    fn compute_max_sim(
        &self,
        doc: MatRef<'_, MinMaxMeta<4>>,
        scores: &mut [f32],
    ) -> Result<(), MaxSimError> {
        if scores.len() != self.nrows() {
            return Err(MaxSimError::InvalidBufferLength(scores.len(), self.nrows()));
        }
        if doc.repr().intrinsic_dim() != self.query.dim() {
            return Err(MaxSimError::UnequalDim(
                doc.repr().intrinsic_dim(),
                self.query.dim(),
            ));
        }
        if doc.num_vectors() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }
        if self.query.dim() == 0 {
            scores.fill(0.0);
            return Ok(());
        }
        let Some(a) = self.query.view() else {
            return Ok(());
        };
        let mut driver = Driver::<_, MR, NR, PACK>::new(
            self.arch,
            a,
            RawRows::new(doc),
            scores,
            mk::Cache::detect(),
        );
        mk::Drive::drive(&mut driver);
        Ok(())
    }
}

struct BuildAndErase<E>(E);

macro_rules! builder {
    ($arch:ty, $mr:literal, $nr:literal, $pack:literal) => {
        impl<E: MinMaxErase<8, 4>>
            diskann_wide::arch::Target1<
                $arch,
                Result<E::Output, Overflow>,
                MatRef<'_, MinMaxMeta<8>>,
            > for BuildAndErase<E>
        {
            fn run(
                self,
                arch: $arch,
                query: MatRef<'_, MinMaxMeta<8>>,
            ) -> Result<E::Output, Overflow> {
                Ok(self.0.erase(Prepared::<_, $mr, $nr, $pack> {
                    arch,
                    query: PackedQuery::new(query)?,
                }))
            }
        }
    };
}

builder!(Scalar, 8, 5, 4);
#[cfg(target_arch = "x86_64")]
builder!(V3, 16, 5, 4);
#[cfg(target_arch = "x86_64")]
builder!(V4, 16, 8, 8);
#[cfg(target_arch = "aarch64")]
builder!(Neon, 8, 8, 4);

/// Prepare a MinMax8-query x MinMax4-document MaxSim kernel.
///
/// Auto selects the available architecture once. Canonical documents remain borrowed.
/// B-first traversal prepares a document tile once and reuses it across query panels.
/// Values and compensation are always exposed through synchronized panel iterators.
///
/// # Errors
///
/// Returns NotSupported for unavailable ISA or Reference requests, dimensions beyond the
/// u32 accumulator limit (1,122,867), or unrepresentable prepared-query extents.
pub fn build_minmax_max_sim<E: MinMaxErase<8, 4>>(
    isa: MaxSimIsa,
    query: MatRef<'_, MinMaxMeta<8>>,
    erase: E,
) -> Result<E::Output, NotSupported> {
    if query.repr().intrinsic_dim() > MAX_DIM {
        return Err(NotSupported {
            isa,
            reason: "MinMax8 dimension exceeds the u32 accumulator limit",
        });
    }
    let result = match isa {
        MaxSimIsa::Auto => diskann_wide::arch::dispatch1_no_features(BuildAndErase(erase), query),
        MaxSimIsa::Scalar => Scalar::new().run1(BuildAndErase(erase), query),
        #[cfg(target_arch = "x86_64")]
        MaxSimIsa::X86_64_V3 => V3::new_checked()
            .ok_or(NotSupported {
                isa,
                reason: "AVX2/FMA unavailable",
            })?
            .run1(BuildAndErase(erase), query),
        #[cfg(target_arch = "x86_64")]
        MaxSimIsa::X86_64_V4 => V4::new_checked()
            .ok_or(NotSupported {
                isa,
                reason: "AVX-512 unavailable",
            })?
            .run1(BuildAndErase(erase), query),
        #[cfg(target_arch = "aarch64")]
        MaxSimIsa::Neon => Neon::new_checked()
            .ok_or(NotSupported {
                isa,
                reason: "NEON/dotprod unavailable",
            })?
            .run1(BuildAndErase(erase), query),
        #[cfg(not(target_arch = "x86_64"))]
        MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => {
            return Err(NotSupported {
                isa,
                reason: "x86-64 ISA unavailable on this target",
            });
        }
        #[cfg(not(target_arch = "aarch64"))]
        MaxSimIsa::Neon => {
            return Err(NotSupported {
                isa,
                reason: "NEON unavailable on this target",
            });
        }
        MaxSimIsa::Reference => {
            return Err(NotSupported {
                isa,
                reason: "reference kernel unsupported",
            });
        }
    };
    result.map_err(|_| NotSupported {
        isa,
        reason: "prepared MinMax query exceeds the allocation limit",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CompressInto,
        algorithms::{Transform, transforms::NullTransform},
        bits::{Representation, Unsigned},
        minmax::MinMaxQuantizer,
        multi_vector::{BoxErase, Defaulted, Mat, MaxSim, QueryMatRef, Standard},
        num::Positive,
    };
    use diskann_utils::ReborrowMut;
    use diskann_vector::DistanceFunctionMut;
    use std::num::NonZeroUsize;

    fn data<const BITS: usize>(rows: usize, dim: usize, seed: usize) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut result = Mat::new(MinMaxMeta::new(rows, dim), Defaulted).unwrap();
        if dim != 0 {
            let mut i = seed;
            let source = Mat::from_fn(Standard::new(rows, dim).unwrap(), || {
                i += 1;
                ((i * 17 % 257) as f32 - 128.0) / 37.0
            });
            let quantizer = MinMaxQuantizer::new(
                Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
                Positive::new(1.0).unwrap(),
            );
            quantizer
                .compress_into(source.as_view(), result.reborrow_mut())
                .unwrap();
        }
        result
    }

    #[test]
    fn tiled_scalar_matches_pairwise() {
        for dim in [0, 1, 3, 4, 5, 63, 64, 65, 127, 128, 129, 250, 256, 257] {
            for q in (0..=17).chain([24, 25, 32, 33]) {
                let query = data::<8>(q, dim, 3);
                for isa in [
                    MaxSimIsa::Scalar,
                    MaxSimIsa::Auto,
                    MaxSimIsa::X86_64_V3,
                    MaxSimIsa::X86_64_V4,
                    MaxSimIsa::Neon,
                ]
                .into_iter()
                .filter(|isa| isa.is_available())
                {
                    let kernel = build_minmax_max_sim(isa, query.as_view(), BoxErase).unwrap();
                    for d in [0, 1, 2, 3, 4, 5, 6, 7, 13, 129] {
                        let doc = data::<4>(d, dim, 11);
                        let mut expected = vec![0.0; q];
                        MaxSim::new(&mut expected)
                            .evaluate(QueryMatRef::from(query.as_view()), doc.as_view());
                        let mut scores = vec![123.0; q + 2];
                        for _ in 0..2 {
                            kernel
                                .compute_max_sim(doc.as_view(), &mut scores[1..q + 1])
                                .unwrap();
                            assert_eq!(&scores[1..q + 1], expected, "{q}x{d}x{dim}");
                            assert_eq!((scores[0], scores[q + 1]), (123.0, 123.0));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shape_errors_and_unsupported_requests() {
        let query = data::<8>(9, 65, 3);
        let kernel = build_minmax_max_sim(MaxSimIsa::Scalar, query.as_view(), BoxErase).unwrap();
        let doc = data::<4>(7, 65, 11);
        let mut short = [123.0; 8];
        assert!(kernel.compute_max_sim(doc.as_view(), &mut short).is_err());
        assert_eq!(short, [123.0; 8]);
        for rows in [0, 7] {
            let wrong = data::<4>(rows, 64, 11);
            let mut scores = [123.0; 9];
            assert!(
                kernel
                    .compute_max_sim(wrong.as_view(), &mut scores)
                    .is_err()
            );
            assert_eq!(scores, [123.0; 9]);
        }
        for isa in [
            MaxSimIsa::X86_64_V3,
            MaxSimIsa::X86_64_V4,
            MaxSimIsa::Neon,
            MaxSimIsa::Reference,
        ] {
            assert_eq!(
                build_minmax_max_sim(isa, query.as_view(), BoxErase).is_err(),
                isa == MaxSimIsa::Reference || !isa.is_available()
            );
        }
        let excessive = Mat::new(MinMaxMeta::<8>::new(0, MAX_DIM + 1), Defaulted).unwrap();
        assert!(build_minmax_max_sim(MaxSimIsa::Scalar, excessive.as_view(), BoxErase).is_err());
    }

    #[test]
    fn prepared_query_ownership_nan_and_concurrent_calls() {
        let query = data::<8>(9, 65, 3);
        let mut doc = data::<4>(13, 65, 11);
        let kernel = build_minmax_max_sim(MaxSimIsa::Scalar, query.as_view(), BoxErase).unwrap();
        for mut row in doc.reborrow_mut().rows_mut() {
            let mut meta = row.meta();
            meta.a = f32::NAN;
            row.set_meta(meta);
        }
        drop(query);
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let kernel = &kernel;
                let doc = &doc;
                scope.spawn(move || {
                    let mut scores = [0.0; 9];
                    kernel.compute_max_sim(doc.as_view(), &mut scores).unwrap();
                    assert_eq!(scores, [f32::MAX; 9]);
                });
            }
        });
    }
}
