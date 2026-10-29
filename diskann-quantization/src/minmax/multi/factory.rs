/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::arch::Scalar;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};

use super::{
    kernel::{MinMaxMaxSim, MinMaxMaxSimKernel, canonical_view},
    prepared,
    reference::ReferenceKernel,
};
use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::maxsim::packed_minmax_x_unpacked_minmax::max_k,
    minmax::MinMaxMeta,
    multi_vector::{BuildMaxSimError, MatRef, MaxSimIsa, NotSupported, QueryMatRef},
};

/// Build a kernel that owns a copy of the MinMax query.
///
/// `Auto` selects a backend at construction. Explicit ISA requests never silently
/// fall back. `Reference` uses an independent scalar implementation.
///
/// Kernels return negative maximum inner products, matching the MaxSim distance convention.
///
/// # Errors
///
/// Returns [`BuildMaxSimError`] for an unavailable ISA, a query dimension exceeding
/// the bit-width pair's accumulator limit, or an overflowing storage size.
/// Dimension and canonical-size checks apply to every backend, including empty queries;
/// packed-size checks apply only to prepared kernels.
/// Allocation exhaustion follows the allocator's normal behavior.
///
/// # Example
///
/// ```
/// use diskann_quantization::{
///     minmax::{MinMaxMeta, build_minmax_max_sim},
///     multi_vector::{Defaulted, Mat, MaxSimIsa, QueryMatRef},
/// };
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let query = Mat::new(MinMaxMeta::<8>::new(2, 16), Defaulted)?;
/// let document = Mat::new(MinMaxMeta::<4>::new(3, 16), Defaulted)?;
/// let query = QueryMatRef::from(query.as_view());
/// let kernel = build_minmax_max_sim::<8, 4>(MaxSimIsa::Auto, query)?;
/// let mut scores = vec![0.0; kernel.nrows()];
/// kernel.compute_max_sim(document.as_view(), &mut scores)?;
/// assert_eq!(scores, vec![0.0; 2]);
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
pub fn build_minmax_max_sim<const A_BITS: usize, const B_BITS: usize>(
    isa: MaxSimIsa,
    query: QueryMatRef<'_, MinMaxMeta<A_BITS>>,
) -> Result<MinMaxMaxSim<B_BITS>, BuildMaxSimError>
where
    Unsigned: MinMaxKernelFactory<A_BITS, B_BITS>,
{
    <Unsigned as MinMaxKernelFactory<A_BITS, B_BITS>>::build(isa, query)
}

/// Precompiled query/document bit-width combinations.
#[doc(hidden)]
pub trait MinMaxKernelFactory<const A_BITS: usize, const B_BITS: usize>:
    Representation<A_BITS> + Representation<B_BITS>
{
    fn build(
        isa: MaxSimIsa,
        query: QueryMatRef<'_, MinMaxMeta<A_BITS>>,
    ) -> Result<MinMaxMaxSim<B_BITS>, BuildMaxSimError>
    where
        Unsigned: Representation<A_BITS> + Representation<B_BITS>;
}

// Concrete, non-generic implementations keep backend compilation in this crate.
macro_rules! factory {
    ($a:literal, $b:literal, $scalar:ident, $v3:ident, $v4:ident) => {
        impl MinMaxKernelFactory<$a, $b> for Unsigned {
            fn build(
                isa: MaxSimIsa,
                query: QueryMatRef<'_, MinMaxMeta<$a>>,
            ) -> Result<MinMaxMaxSim<$b>, BuildMaxSimError> {
                let query = query.0;
                validate_query::<$a, $b>(query)?;
                let inner: Box<dyn MinMaxMaxSimKernel<$b>> = match RuntimeArch::resolve(isa)? {
                    RuntimeArch::Scalar(a) => Box::new(prepared::$scalar::<$a, $b>::new(a, query)?),
                    #[cfg(target_arch = "x86_64")]
                    RuntimeArch::V3(a) => Box::new(prepared::$v3::<$a, $b>::new(a, query)?),
                    #[cfg(target_arch = "x86_64")]
                    RuntimeArch::V4(a) => Box::new(prepared::$v4::<$a, $b>::new(a, query)?),
                    #[cfg(target_arch = "aarch64")]
                    RuntimeArch::Neon(a) => Box::new(prepared::NeonBytes::<$a, $b>::new(a, query)?),
                    RuntimeArch::Reference => Box::new(ReferenceKernel::<$a, $b>::new(query)?),
                };
                Ok(MinMaxMaxSim { inner })
            }
        }
    };
    ($($a:literal),+ $(,)?) => {$(
        factory!($a, 1, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 2, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 3, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 4, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 5, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 6, ScalarBytes, V3Bytes, V4Bytes);
        factory!($a, 7, ScalarBytes, V3Words, V4Bytes);
        factory!($a, 8, ScalarWords, V3Words, V4Words);
    )+};
}
factory!(1, 2, 3, 4, 5, 6, 7, 8);

fn validate_query<const A_BITS: usize, const B_BITS: usize>(
    query: MatRef<'_, MinMaxMeta<A_BITS>>,
) -> Result<(), BuildMaxSimError>
where
    Unsigned: Representation<A_BITS> + Representation<B_BITS>,
{
    let dim = query.repr().intrinsic_dim();
    let limit = max_k::<A_BITS, B_BITS>();
    if dim > limit {
        return Err(BuildMaxSimError::DimTooLarge(dim, limit));
    }
    canonical_view(query)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
enum RuntimeArch {
    Scalar(Scalar),
    #[cfg(target_arch = "x86_64")]
    V3(V3),
    #[cfg(target_arch = "x86_64")]
    V4(V4),
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    Reference,
}

impl RuntimeArch {
    fn resolve(isa: MaxSimIsa) -> Result<Self, NotSupported> {
        let unavailable = || NotSupported {
            isa,
            reason: "requested ISA unavailable on this host",
        };
        Ok(match isa {
            MaxSimIsa::Auto => {
                #[cfg(target_arch = "x86_64")]
                {
                    if let Some(arch) = V4::new_checked() {
                        return Ok(Self::V4(arch));
                    }
                    if let Some(arch) = V3::new_checked() {
                        return Ok(Self::V3(arch));
                    }
                }
                #[cfg(target_arch = "aarch64")]
                if let Some(arch) = Neon::new_checked() {
                    return Ok(Self::Neon(arch));
                }
                Self::Scalar(Scalar::new())
            }
            MaxSimIsa::Scalar => Self::Scalar(Scalar::new()),
            MaxSimIsa::Reference => Self::Reference,
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V3 => Self::V3(V3::new_checked().ok_or_else(unavailable)?),
            #[cfg(target_arch = "x86_64")]
            MaxSimIsa::X86_64_V4 => Self::V4(V4::new_checked().ok_or_else(unavailable)?),
            #[cfg(not(target_arch = "x86_64"))]
            MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => return Err(unavailable()),
            #[cfg(target_arch = "aarch64")]
            MaxSimIsa::Neon => Self::Neon(Neon::new_checked().ok_or_else(unavailable)?),
            #[cfg(not(target_arch = "aarch64"))]
            MaxSimIsa::Neon => return Err(unavailable()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        minmax::MinMaxCompensation,
        multi_vector::{Defaulted, Mat, MaxSimError},
    };

    fn matrix<const BITS: usize>(rows: usize, dim: usize) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut mat = Mat::new(MinMaxMeta::new(rows, dim), Defaulted).unwrap();
        for (i, mut row) in mat.rows_mut().enumerate() {
            let mut sum = 0;
            for d in 0..dim {
                let code = ((i * 17 + d * 29 + 7) % (1 << BITS)) as i64;
                row.vector_mut().set(d, code).unwrap();
                sum += code;
            }
            let a = (i + 1) as f32 / 8.0;
            row.set_meta(MinMaxCompensation {
                dim: dim as u32,
                a,
                b: i as f32 / 4.0 - 1.0,
                n: a * sum as f32,
                norm_squared: 0.0,
            });
        }
        mat
    }

    fn constant_matrix<const BITS: usize>(
        rows: usize,
        dim: usize,
        code: i64,
    ) -> Mat<MinMaxMeta<BITS>>
    where
        Unsigned: Representation<BITS>,
    {
        let mut mat = Mat::new(MinMaxMeta::new(rows, dim), Defaulted).unwrap();
        for mut row in mat.rows_mut() {
            for d in 0..dim {
                row.vector_mut().set(d, code).unwrap();
            }
            row.set_meta(MinMaxCompensation {
                dim: dim as u32,
                a: 1.0,
                n: dim as f32 * code as f32,
                ..Default::default()
            });
        }
        mat
    }

    fn available_isas() -> impl Iterator<Item = MaxSimIsa> {
        [
            MaxSimIsa::Scalar,
            MaxSimIsa::Auto,
            MaxSimIsa::X86_64_V3,
            MaxSimIsa::X86_64_V4,
            MaxSimIsa::Neon,
            MaxSimIsa::Reference,
        ]
        .into_iter()
        .filter(|isa| isa.is_available())
    }

    fn assert_matches_reference<const A_BITS: usize, const B_BITS: usize>()
    where
        Unsigned: MinMaxKernelFactory<A_BITS, B_BITS>,
    {
        // Exercise partial query panels, document panels, and K packs on every backend.
        let (rows, docs, dim) = (17, 9, 65);
        let query = matrix::<A_BITS>(rows, dim);
        let document = matrix::<B_BITS>(docs, dim);
        let reference =
            build_minmax_max_sim::<A_BITS, B_BITS>(MaxSimIsa::Reference, query.as_view().into())
                .unwrap();
        let mut expected = vec![0.0; rows];
        reference
            .compute_max_sim(document.as_view(), &mut expected)
            .unwrap();
        for isa in available_isas().filter(|&isa| isa != MaxSimIsa::Reference) {
            let kernel =
                build_minmax_max_sim::<A_BITS, B_BITS>(isa, query.as_view().into()).unwrap();
            assert_eq!(kernel.nrows(), rows);
            let mut guarded = vec![123.0; rows + 2];
            kernel
                .compute_max_sim(document.as_view(), &mut guarded[1..rows + 1])
                .unwrap();
            for (actual, expected) in guarded[1..rows + 1].iter().zip(&expected) {
                assert!(
                    actual == expected
                        || (actual.is_finite()
                            && expected.is_finite()
                            && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0)),
                    "{A_BITS}x{B_BITS}, {isa}: {actual} != {expected}"
                );
            }
            assert_eq!((guarded[0], guarded[rows + 1]), (123.0, 123.0));
        }
    }

    #[test]
    fn matches_reference_across_bit_widths() {
        macro_rules! row {
            ($a:literal) => {
                assert_matches_reference::<$a, 1>();
                assert_matches_reference::<$a, 2>();
                assert_matches_reference::<$a, 3>();
                assert_matches_reference::<$a, 4>();
                assert_matches_reference::<$a, 5>();
                assert_matches_reference::<$a, 6>();
                assert_matches_reference::<$a, 7>();
                assert_matches_reference::<$a, 8>();
            };
        }
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
        row!(8);
    }

    #[test]
    fn handles_empty_inputs() {
        fn check<const B_BITS: usize>()
        where
            Unsigned: MinMaxKernelFactory<8, B_BITS>,
        {
            for (rows, docs, dim, expected) in [
                (0, 5, 7, 0.0),
                (3, 0, 7, f32::MAX),
                (3, 5, 0, 0.0),
                (3, 0, 0, f32::MAX),
            ] {
                let query = matrix::<8>(rows, dim);
                let document = matrix::<B_BITS>(docs, dim);
                for isa in available_isas() {
                    let kernel =
                        build_minmax_max_sim::<8, B_BITS>(isa, query.as_view().into()).unwrap();
                    let mut scores = vec![f32::NAN; rows];
                    kernel
                        .compute_max_sim(document.as_view(), &mut scores)
                        .unwrap();
                    assert_eq!(
                        scores,
                        vec![expected; rows],
                        "{isa}, B_BITS={B_BITS}, {rows}x{docs}x{dim}"
                    );
                }
            }
        }
        check::<4>();
        check::<8>();
    }

    #[test]
    fn preserves_scores_on_error() {
        let query = matrix::<8>(2, 5);
        for isa in available_isas() {
            let kernel = build_minmax_max_sim::<8, 8>(isa, query.as_view().into()).unwrap();
            let doc = matrix::<8>(1, 6);
            let mut scores = [123.0; 2];
            assert!(matches!(
                kernel.compute_max_sim(doc.as_view(), &mut scores),
                Err(MaxSimError::UnequalDim(6, 5))
            ));
            assert_eq!(scores, [123.0; 2]);
            let doc = matrix::<8>(1, 5);
            let mut scores = [123.0];
            assert!(matches!(
                kernel.compute_max_sim(doc.as_view(), &mut scores),
                Err(MaxSimError::InvalidBufferLength(1, 2))
            ));
            assert_eq!(scores, [123.0]);
        }
    }

    #[test]
    fn rejects_unavailable_isas() {
        let query = matrix::<8>(2, 5);
        for isa in [MaxSimIsa::X86_64_V3, MaxSimIsa::X86_64_V4, MaxSimIsa::Neon] {
            if !isa.is_available() {
                assert!(matches!(
                    build_minmax_max_sim::<8, 8>(isa, query.as_view().into()),
                    Err(BuildMaxSimError::NotSupported(error)) if error.isa == isa
                ));
            }
        }
    }

    #[test]
    fn rejects_excessive_dimensions() {
        let limit = max_k::<8, 8>();
        let dim = limit + 1;
        let query = MatRef::new(MinMaxMeta::<8>::new(0, dim), &[] as &[u8]).unwrap();
        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            let result = build_minmax_max_sim::<8, 8>(isa, query.into());
            assert!(
                matches!(result, Err(BuildMaxSimError::DimTooLarge(found, max))
                if found == dim && max == limit),
                "{isa}"
            );
        }
    }

    #[test]
    fn owns_query_data() {
        let document = constant_matrix::<4>(2, 3, 2);
        for isa in available_isas() {
            let kernel = {
                let query = constant_matrix::<8>(2, 3, 1);
                build_minmax_max_sim::<8, 4>(isa, query.as_view().into()).unwrap()
            };
            let mut scores = [f32::NAN; 2];
            kernel
                .compute_max_sim(document.as_view(), &mut scores)
                .unwrap();
            assert_eq!(scores, [-6.0; 2], "{isa}");
        }
    }

    #[test]
    fn selects_backend_and_layout() {
        fn assert_layout<const B_BITS: usize, K: 'static>(
            isa: MaxSimIsa,
            query: QueryMatRef<'_, MinMaxMeta<8>>,
        ) where
            Unsigned: MinMaxKernelFactory<8, B_BITS>,
        {
            assert_eq!(
                build_minmax_max_sim::<8, B_BITS>(isa, query)
                    .unwrap()
                    .inner
                    .type_id(),
                std::any::TypeId::of::<K>(),
                "{isa}, B_BITS={B_BITS}",
            );
        }

        let query = matrix::<8>(1, 3);
        let query = QueryMatRef::from(query.as_view());
        assert_layout::<7, prepared::ScalarBytes<8, 7>>(MaxSimIsa::Scalar, query);
        assert_layout::<8, prepared::ScalarWords<8, 8>>(MaxSimIsa::Scalar, query);
        #[cfg(target_arch = "x86_64")]
        {
            if V3::new_checked().is_some() {
                assert_layout::<6, prepared::V3Bytes<8, 6>>(MaxSimIsa::X86_64_V3, query);
                assert_layout::<7, prepared::V3Words<8, 7>>(MaxSimIsa::X86_64_V3, query);
            }
            if V4::new_checked().is_some() {
                assert_layout::<7, prepared::V4Bytes<8, 7>>(MaxSimIsa::X86_64_V4, query);
                assert_layout::<8, prepared::V4Words<8, 8>>(MaxSimIsa::X86_64_V4, query);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if Neon::new_checked().is_some() {
            assert_layout::<8, prepared::NeonBytes<8, 8>>(MaxSimIsa::Neon, query);
        }
    }

    #[test]
    fn handles_nonfinite_scores() {
        fn check<const B_BITS: usize>()
        where
            Unsigned: MinMaxKernelFactory<8, B_BITS>,
        {
            let max_code = (1i64 << B_BITS) - 1;
            let query = constant_matrix::<8>(1, 5, 255);
            let mut document = constant_matrix::<B_BITS>(2, 5, max_code);
            let finite_score = -(5 * 255 * max_code) as f32;
            for isa in available_isas() {
                let kernel =
                    build_minmax_max_sim::<8, B_BITS>(isa, query.as_view().into()).unwrap();
                let mut scores = [0.0];
                for (scale, expected) in [
                    (1.0, finite_score),
                    (f32::NEG_INFINITY, f32::MAX),
                    (f32::INFINITY, f32::NEG_INFINITY),
                    (f32::NAN, f32::MAX),
                    (1.0, finite_score),
                ] {
                    for mut row in document.rows_mut() {
                        let mut meta = row.meta();
                        meta.a = scale;
                        row.set_meta(meta);
                    }
                    kernel
                        .compute_max_sim(document.as_view(), &mut scores)
                        .unwrap();
                    assert_eq!(scores, [expected], "{isa}, B_BITS={B_BITS}");
                }
                let mut row = document.get_row_mut(0).unwrap();
                let mut meta = row.meta();
                meta.a = f32::NAN;
                row.set_meta(meta);
                kernel
                    .compute_max_sim(document.as_view(), &mut scores)
                    .unwrap();
                assert_eq!(
                    scores,
                    [finite_score],
                    "{isa}, B_BITS={B_BITS}, mixed NaN/finite"
                );
            }
        }
        check::<6>();
        check::<7>();
        check::<8>();
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn rejects_query_size_overflow() {
        // With release overflow behavior, rows * 20 wraps to zero for K=0.
        // Reject this view before either packing or cloning can access its claimed rows.
        let rows = 1usize << (usize::BITS - 2);
        let query = MatRef::new(MinMaxMeta::<8>::new(rows, 0), &[] as &[u8]).unwrap();
        for isa in [MaxSimIsa::Auto, MaxSimIsa::Reference] {
            assert!(matches!(
                build_minmax_max_sim::<8, 4>(isa, query.into()),
                Err(BuildMaxSimError::Overflow(_))
            ));
        }
    }
}
