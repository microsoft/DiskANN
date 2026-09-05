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
use crate::matrix_kernels::maxsim::minmax8_x_minmax4::{APacking, Driver, PackedMinMax8};
use crate::multi_vector::{MatRef, MaxSimError, MaxSimIsa, NotSupported};

#[derive(Debug)]
struct Prepared<A, const MR: usize, const NR: usize> {
    arch: A,
    prepared: PackedMinMax8<MR>,
}

impl<A, const MR: usize, const NR: usize> Prepared<A, MR, NR>
where
    A: Architecture,
    for<'a> Driver<'a, A, MR, NR>: mk::Drive,
{
    fn run(&self, doc: MatRef<'_, MinMaxMeta<4>>, scores: &mut [f32]) -> Result<(), MaxSimError> {
        if scores.len() != self.prepared.nrows() {
            return Err(MaxSimError::InvalidBufferLength(
                scores.len(),
                self.prepared.nrows(),
            ));
        }
        if doc.repr().intrinsic_dim() != self.prepared.dim() {
            return Err(MaxSimError::UnequalDim(
                doc.repr().intrinsic_dim(),
                self.prepared.dim(),
            ));
        }
        if doc.num_vectors() == 0 {
            scores.fill(f32::MAX);
            return Ok(());
        }

        let Some(k) = NonZeroUsize::new(self.prepared.dim()).map(mk::DimK::new) else {
            scores.fill(0.0);
            return Ok(());
        };
        // SAFETY: The dimension and output length checks establish the driver's invariants.
        let mut driver = unsafe { Driver::new(self.arch, &self.prepared, doc, scores, k) };
        mk::Drive::drive(&mut driver);
        Ok(())
    }
}

macro_rules! impl_kernel {
    ($arch:ty, $mr:literal, $nr:literal) => {
        impl MinMaxMaxSimKernel<8, 4> for Prepared<$arch, $mr, $nr> {
            fn nrows(&self) -> usize {
                self.prepared.nrows()
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
impl_kernel!(V3, 16, 8);
#[cfg(target_arch = "x86_64")]
impl_kernel!(V4, 16, 8);

struct BuildAndErase<E>(E);

macro_rules! impl_builder {
    ($arch:ty, $mr:literal, $nr:literal, $packing:expr) => {
        impl<E> diskann_wide::arch::Target1<$arch, E::Output, MatRef<'_, MinMaxMeta<8>>>
            for BuildAndErase<E>
        where
            E: MinMaxErase<8, 4>,
        {
            fn run(self, arch: $arch, query: MatRef<'_, MinMaxMeta<8>>) -> E::Output {
                self.0.erase(Prepared::<_, $mr, $nr> {
                    arch,
                    prepared: PackedMinMax8::new(query, $packing),
                })
            }
        }
    };
}

impl_builder!(Scalar, 8, 6, APacking::RowMajor);
#[cfg(target_arch = "aarch64")]
impl_builder!(Neon, 8, 8, APacking::Grouped4);
#[cfg(target_arch = "x86_64")]
impl_builder!(V3, 16, 8, APacking::Grouped4);
#[cfg(target_arch = "x86_64")]
impl_builder!(V4, 16, 8, APacking::Grouped8);

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
    use crate::bits::{Representation, Unsigned};
    use crate::minmax::MinMaxQuantizer;
    use crate::multi_vector::{BoxErase, Defaulted, Mat, MaxSim, QueryMatRef, Standard};
    use crate::num::Positive;

    const ISAS: [MaxSimIsa; 5] = [
        MaxSimIsa::Scalar,
        MaxSimIsa::X86_64_V3,
        MaxSimIsa::X86_64_V4,
        MaxSimIsa::Neon,
        MaxSimIsa::Auto,
    ];

    fn compress<const BITS: usize>(
        values: &[f32],
        nrows: usize,
        dim: usize,
    ) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
            Positive::new(1.0).unwrap(),
        );
        let input = MatRef::new(Standard::new(nrows, dim).unwrap(), values).unwrap();
        let mut output = Mat::new(MinMaxMeta::<BITS>::new(nrows, dim), Defaulted).unwrap();
        quantizer
            .compress_into(input, output.reborrow_mut())
            .unwrap();
        output
    }

    fn check_problem(
        isa: MaxSimIsa,
        query: MatRef<'_, MinMaxMeta<8>>,
        docs: MatRef<'_, MinMaxMeta<4>>,
    ) {
        let kernel = build_minmax_max_sim(isa, query, BoxErase);
        if !isa.is_available() {
            assert_eq!(kernel.unwrap_err().isa, isa);
            return;
        }
        let kernel = kernel.unwrap();
        let nrows = query.num_vectors();
        assert_eq!(kernel.nrows(), nrows);

        let mut expected = vec![0.0; nrows];
        MaxSim::new(&mut expected).evaluate(QueryMatRef::from(query), docs);

        let mut actual = vec![f32::NEG_INFINITY; nrows + 2];
        kernel
            .compute_max_sim(docs, &mut actual[1..nrows + 1])
            .unwrap();
        assert_eq!(actual[0], f32::NEG_INFINITY);
        assert_eq!(actual[nrows + 1], f32::NEG_INFINITY);
        assert_eq!(
            actual[1..nrows + 1],
            expected,
            "mismatch for ({nrows}, {}, {}) using {isa:?}",
            docs.num_vectors(),
            query.repr().intrinsic_dim(),
        );
    }

    fn check(isa: MaxSimIsa) {
        for (query_rows, doc_rows, dim) in [
            (1, 1, 1),
            (7, 5, 3),
            (8, 6, 8),
            (9, 7, 17),
            (15, 9, 18),
            (17, 13, 64),
        ]
        .into_iter()
        .chain((1..=16).flat_map(|doc_rows| (1..=17).map(move |dim| (17, doc_rows, dim))))
        {
            let query_values: Vec<f32> = (0..query_rows * dim)
                .map(|i| ((i * 17 + 3) % 101) as f32 / 13.0 - 4.0)
                .collect();
            let doc_values: Vec<f32> = (0..doc_rows * dim)
                .map(|i| ((i * 29 + 7) % 97) as f32 / 11.0 - 3.0)
                .collect();
            let query = compress(&query_values, query_rows, dim);
            let docs = compress(&doc_values, doc_rows, dim);
            check_problem(isa, query.as_view(), docs.as_view());
        }
    }

    #[test]
    fn scalar_matches_existing_max_sim() {
        check(MaxSimIsa::Scalar);
    }

    #[test]
    fn v3_matches_existing_max_sim() {
        check(MaxSimIsa::X86_64_V3);
    }

    #[test]
    fn v4_matches_existing_max_sim() {
        check(MaxSimIsa::X86_64_V4);
    }

    #[test]
    fn neon_matches_existing_max_sim() {
        check(MaxSimIsa::Neon);
    }

    #[test]
    fn auto_matches_existing_max_sim() {
        check(MaxSimIsa::Auto);
    }

    #[test]
    fn nan_compensation_preserves_best_score() {
        for nrows in [1, 9, 17] {
            let query = compress(&[1e20, -1e20].repeat(nrows), nrows, 2);
            // Finite inputs can still overflow compensation into NaN.
            for values in [
                [1e20, -1e20, 1.0, 0.0, 0.5, 0.0],
                [1.0, 0.0, 1e20, -1e20, 0.5, 0.0],
                [1.0, 0.0, 0.5, 0.0, 1e20, -1e20],
                [1e20, -1e20, 1e20, -1e20, 1e20, -1e20],
            ] {
                let docs = compress(&values, 3, 2);
                for isa in ISAS {
                    check_problem(isa, query.as_view(), docs.as_view());
                }
            }
        }
    }

    #[test]
    fn invalid_shapes_leave_scores_unchanged() {
        let query = Mat::new(MinMaxMeta::<8>::new(3, 4), Defaulted).unwrap();
        let docs = Mat::new(MinMaxMeta::<4>::new(2, 4), Defaulted).unwrap();
        for isa in ISAS.into_iter().filter(|isa| isa.is_available()) {
            let kernel = build_minmax_max_sim(isa, query.as_view(), BoxErase).unwrap();
            for len in [2, 4] {
                let mut scores = vec![123.0; len];
                assert!(matches!(
                    kernel.compute_max_sim(docs.as_view(), &mut scores),
                    Err(MaxSimError::InvalidBufferLength(actual, 3)) if actual == len
                ));
                assert_eq!(scores, vec![123.0; len]);
            }
            for nrows in [0, 2] {
                for dim in [0, 3, 5] {
                    let docs = Mat::new(MinMaxMeta::<4>::new(nrows, dim), Defaulted).unwrap();
                    let mut scores = [123.0; 3];
                    assert!(matches!(
                        kernel.compute_max_sim(docs.as_view(), &mut scores),
                        Err(MaxSimError::UnequalDim(actual, 4)) if actual == dim
                    ));
                    assert_eq!(scores, [123.0; 3]);
                }
            }
        }
    }

    #[test]
    fn empty_inputs_match_existing_max_sim() {
        for dim in [0, 3] {
            for query_rows in [0, 3] {
                for doc_rows in [0, 2] {
                    let query = Mat::new(MinMaxMeta::<8>::new(query_rows, dim), Defaulted).unwrap();
                    let docs = Mat::new(MinMaxMeta::<4>::new(doc_rows, dim), Defaulted).unwrap();
                    for isa in ISAS {
                        check_problem(isa, query.as_view(), docs.as_view());
                    }
                }
            }
        }
    }

    #[test]
    fn prepared_query_owns_data_and_resets_scores() {
        let docs = [[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]].map(|values| compress::<4>(&values, 1, 2));
        for isa in ISAS.into_iter().filter(|isa| isa.is_available()) {
            let (kernel, expected) = {
                let query = compress(&[1.0, -1.0].repeat(17), 17, 2);
                let expected = docs.each_ref().map(|doc| {
                    let mut scores = vec![0.0; 17];
                    MaxSim::new(&mut scores)
                        .evaluate(QueryMatRef::from(query.as_view()), doc.as_view());
                    scores
                });
                (
                    build_minmax_max_sim(isa, query.as_view(), BoxErase).unwrap(),
                    expected,
                )
            };
            assert_eq!(kernel.nrows(), 17);
            let mut scores = vec![f32::NEG_INFINITY; 17];
            for (doc, expected) in docs.iter().zip(expected) {
                kernel.compute_max_sim(doc.as_view(), &mut scores).unwrap();
                assert_eq!(scores, expected, "{isa:?}");
            }
            let docs = Mat::new(MinMaxMeta::<4>::new(0, 2), Defaulted).unwrap();
            kernel.compute_max_sim(docs.as_view(), &mut scores).unwrap();
            assert_eq!(scores, vec![f32::MAX; 17]);
        }
    }

    #[test]
    fn reference_kernel_is_not_supported() {
        let query = Mat::new(MinMaxMeta::<8>::new(1, 2), Defaulted).unwrap();
        let err =
            build_minmax_max_sim(MaxSimIsa::Reference, query.as_view(), BoxErase).unwrap_err();
        assert_eq!(err.isa, MaxSimIsa::Reference);
    }
}
