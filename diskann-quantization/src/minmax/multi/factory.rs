/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_wide::Architecture;
#[cfg(target_arch = "aarch64")]
use diskann_wide::arch::aarch64::Neon;
#[cfg(target_arch = "x86_64")]
use diskann_wide::arch::x86_64::{V3, V4};
use diskann_wide::arch::{Scalar, Target1};

use super::{kernel::MinMaxErase, prepared::Prepared, reference::ReferenceKernel};
use crate::{
    bits::{Representation, Unsigned},
    matrix_kernels::maxsim::packed_minmax_x_unpacked_minmax::max_k,
    minmax::{MinMaxCompensation, MinMaxMeta},
    multi_vector::{BuildMaxSimError, MatRef, MaxSimIsa, NotSupported, Overflow, QueryMatRef},
};

/// Selects backend layouts for a document bit width.
///
/// The built-in implementations keep document widths literal so unsupported
/// byte-dot layouts are never instantiated.
pub trait MinMaxKernelFactory<const B_BITS: usize>: Representation<B_BITS> {
    #[doc(hidden)]
    fn build<const A_BITS: usize, E: MinMaxErase<B_BITS>>(
        isa: MaxSimIsa,
        query: QueryMatRef<'_, MinMaxMeta<A_BITS>>,
        erase: E,
    ) -> Result<E::Output, BuildMaxSimError>
    where
        Unsigned: Representation<A_BITS> + Representation<B_BITS>;
}

struct BuildAndErase<const A_BITS: usize, const B_BITS: usize, E>(E);

macro_rules! kernel_builder {
    ($b:literal, $arch:ty, $mr:literal, $nr:literal, $pack:literal) => {
        impl<const A_BITS: usize, E: MinMaxErase<$b>>
            Target1<$arch, Result<E::Output, Overflow>, MatRef<'_, MinMaxMeta<A_BITS>>>
            for BuildAndErase<A_BITS, $b, E>
        where
            Unsigned: Representation<A_BITS>,
        {
            fn run(
                self,
                arch: $arch,
                query: MatRef<'_, MinMaxMeta<A_BITS>>,
            ) -> Result<E::Output, Overflow> {
                let kernel = Prepared::<$arch, A_BITS, $mr, $nr, $pack>::new(arch, query)?;
                Ok(self.0.erase(kernel))
            }
        }
    };
}

macro_rules! kernel_factory {
    (
        b_bits: $b:literal,
        scalar: { mr: $scalar_mr:literal, pack: $scalar_pack:literal },
        v3_pack: $v3_pack:literal,
        v4_pack: $v4_pack:literal $(,)?
    ) => {
        kernel_builder!($b, Scalar, $scalar_mr, 2, $scalar_pack);
        #[cfg(target_arch = "x86_64")]
        kernel_builder!($b, V3, 16, 4, $v3_pack);
        #[cfg(target_arch = "x86_64")]
        kernel_builder!($b, V4, 16, 8, $v4_pack);
        #[cfg(target_arch = "aarch64")]
        kernel_builder!($b, Neon, 8, 4, 4);

        impl MinMaxKernelFactory<$b> for Unsigned {
            fn build<const A_BITS: usize, E: MinMaxErase<$b>>(
                isa: MaxSimIsa,
                query: QueryMatRef<'_, MinMaxMeta<A_BITS>>,
                erase: E,
            ) -> Result<E::Output, BuildMaxSimError>
            where
                Unsigned: Representation<A_BITS>,
            {
                let query = query.0;
                validate_query::<A_BITS, $b>(query)?;
                let builder = BuildAndErase::<A_BITS, $b, E>(erase);
                let unavailable = || NotSupported {
                    isa,
                    reason: "requested ISA unavailable on this host",
                };
                let built = match isa {
                    MaxSimIsa::Auto => diskann_wide::arch::dispatch1_no_features(builder, query),
                    MaxSimIsa::Scalar => Scalar::new().run1(builder, query),
                    #[cfg(target_arch = "x86_64")]
                    MaxSimIsa::X86_64_V3 => V3::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(target_arch = "x86_64")]
                    MaxSimIsa::X86_64_V4 => V4::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(not(target_arch = "x86_64"))]
                    MaxSimIsa::X86_64_V3 | MaxSimIsa::X86_64_V4 => return Err(unavailable().into()),
                    #[cfg(target_arch = "aarch64")]
                    MaxSimIsa::Neon => Neon::new_checked()
                        .ok_or_else(unavailable)?
                        .run1(builder, query),
                    #[cfg(not(target_arch = "aarch64"))]
                    MaxSimIsa::Neon => return Err(unavailable().into()),
                    MaxSimIsa::Reference => {
                        return Ok(builder.0.erase(ReferenceKernel::new(query)));
                    }
                };
                Ok(built?)
            }
        }
    };
}

// Byte-dot layouts must handle A codes up to 255: V3's saturating intermediate
// limits B to 6 bits, while Scalar/V4's signed B bytes limit it to 7.
kernel_factory!(
    b_bits: 1,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 2,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 3,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 4,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 5,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 6,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 4,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 7,
    scalar: { mr: 16, pack: 4 },
    v3_pack: 2,
    v4_pack: 4,
);
kernel_factory!(
    b_bits: 8,
    scalar: { mr: 8, pack: 2 },
    v3_pack: 2,
    v4_pack: 2,
);

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
///     multi_vector::{BoxErase, Defaulted, Mat, MaxSimIsa, QueryMatRef},
/// };
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let query = Mat::new(MinMaxMeta::<8>::new(2, 16), Defaulted)?;
/// let document = Mat::new(MinMaxMeta::<4>::new(3, 16), Defaulted)?;
/// let query = QueryMatRef::from(query.as_view());
/// let kernel = build_minmax_max_sim::<8, 4, _>(MaxSimIsa::Auto, query, BoxErase)?;
/// let mut scores = vec![0.0; kernel.nrows()];
/// kernel.compute_max_sim(document.as_view(), &mut scores)?;
/// assert_eq!(scores, vec![0.0; 2]);
/// # Ok(())
/// # }
/// # example().unwrap();
/// ```
pub fn build_minmax_max_sim<const A_BITS: usize, const B_BITS: usize, E>(
    isa: MaxSimIsa,
    query: QueryMatRef<'_, MinMaxMeta<A_BITS>>,
    erase: E,
) -> Result<E::Output, BuildMaxSimError>
where
    Unsigned: Representation<A_BITS> + MinMaxKernelFactory<B_BITS>,
    E: MinMaxErase<B_BITS>,
{
    <Unsigned as MinMaxKernelFactory<B_BITS>>::build(isa, query, erase)
}

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
    let rows = query.num_vectors();
    let overflow = || Overflow::for_type::<u8>(rows, dim);
    let code_bytes = dim.checked_mul(A_BITS).ok_or_else(overflow)?.div_ceil(8);
    let row_bytes = code_bytes
        .checked_add(size_of::<MinMaxCompensation>())
        .ok_or_else(overflow)?;
    Overflow::check::<u8>(rows, row_bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        minmax::{MinMaxCompensation, MinMaxMaxSimKernel},
        multi_vector::{BoxErase, Defaulted, Mat, MaxSimError},
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

    fn assert_matches_reference<const A: usize, const B: usize>()
    where
        Unsigned: Representation<A> + MinMaxKernelFactory<B>,
    {
        // Exercise partial query panels, document panels, and K packs on every backend.
        let (rows, docs, dim) = (17, 9, 65);
        let query = matrix::<A>(rows, dim);
        let document = matrix::<B>(docs, dim);
        let reference =
            build_minmax_max_sim::<A, B, _>(MaxSimIsa::Reference, query.as_view().into(), BoxErase)
                .unwrap();
        let mut expected = vec![0.0; rows];
        reference
            .compute_max_sim(document.as_view(), &mut expected)
            .unwrap();
        for isa in available_isas().filter(|&isa| isa != MaxSimIsa::Reference) {
            let kernel =
                build_minmax_max_sim::<A, B, _>(isa, query.as_view().into(), BoxErase).unwrap();
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
                    "{A}x{B}, {isa}: {actual} != {expected}"
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
        fn check<const B: usize>()
        where
            Unsigned: MinMaxKernelFactory<B>,
        {
            for (rows, docs, dim, expected) in [
                (0, 5, 7, 0.0),
                (3, 0, 7, f32::MAX),
                (3, 5, 0, 0.0),
                (3, 0, 0, f32::MAX),
            ] {
                let query = matrix::<8>(rows, dim);
                let document = matrix::<B>(docs, dim);
                for isa in available_isas() {
                    let kernel =
                        build_minmax_max_sim::<8, B, _>(isa, query.as_view().into(), BoxErase)
                            .unwrap();
                    let mut scores = vec![f32::NAN; rows];
                    kernel
                        .compute_max_sim(document.as_view(), &mut scores)
                        .unwrap();
                    assert_eq!(
                        scores,
                        vec![expected; rows],
                        "{isa}, B={B}, {rows}x{docs}x{dim}"
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
            let kernel =
                build_minmax_max_sim::<8, 8, _>(isa, query.as_view().into(), BoxErase).unwrap();
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
                    build_minmax_max_sim::<8, 8, _>(isa, query.as_view().into(), BoxErase),
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
            for result in [
                build_minmax_max_sim::<8, 8, _>(isa, query.into(), BoxErase),
                <Unsigned as MinMaxKernelFactory<8>>::build(isa, query.into(), BoxErase),
            ] {
                assert!(
                    matches!(result, Err(BuildMaxSimError::DimTooLarge(found, max))
                        if found == dim && max == limit),
                    "{isa}",
                );
            }
        }
    }

    #[test]
    fn owns_query_data() {
        let document = constant_matrix::<4>(2, 3, 2);
        for isa in available_isas() {
            let kernel = {
                let query = constant_matrix::<8>(2, 3, 1);
                build_minmax_max_sim::<8, 4, _>(isa, query.as_view().into(), BoxErase).unwrap()
            };
            let mut scores = [f32::NAN; 2];
            kernel
                .compute_max_sim(document.as_view(), &mut scores)
                .unwrap();
            assert_eq!(scores, [-6.0; 2], "{isa}");
        }
    }

    struct Identify;
    impl<const B: usize> MinMaxErase<B> for Identify
    where
        Unsigned: Representation<B>,
    {
        type Output = std::any::TypeId;
        fn erase<K: MinMaxMaxSimKernel<B> + 'static>(self, _: K) -> Self::Output {
            std::any::TypeId::of::<K>()
        }
    }

    #[test]
    fn selects_backend_and_layout() {
        fn assert_layout<const B: usize, K: 'static>(
            isa: MaxSimIsa,
            query: QueryMatRef<'_, MinMaxMeta<8>>,
        ) where
            Unsigned: MinMaxKernelFactory<B>,
        {
            assert_eq!(
                build_minmax_max_sim::<8, B, _>(isa, query, Identify).unwrap(),
                std::any::TypeId::of::<K>(),
                "{isa}, B={B}",
            );
        }

        let query = matrix::<8>(1, 3);
        let query = QueryMatRef::from(query.as_view());
        assert_layout::<7, Prepared<Scalar, 8, 16, 2, 4>>(MaxSimIsa::Scalar, query);
        assert_layout::<8, Prepared<Scalar, 8, 8, 2, 2>>(MaxSimIsa::Scalar, query);
        #[cfg(target_arch = "x86_64")]
        {
            if V3::new_checked().is_some() {
                assert_layout::<6, Prepared<V3, 8, 16, 4, 4>>(MaxSimIsa::X86_64_V3, query);
                assert_layout::<7, Prepared<V3, 8, 16, 4, 2>>(MaxSimIsa::X86_64_V3, query);
            }
            if V4::new_checked().is_some() {
                assert_layout::<7, Prepared<V4, 8, 16, 8, 4>>(MaxSimIsa::X86_64_V4, query);
                assert_layout::<8, Prepared<V4, 8, 16, 8, 2>>(MaxSimIsa::X86_64_V4, query);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if Neon::new_checked().is_some() {
            assert_layout::<8, Prepared<Neon, 8, 8, 4, 4>>(MaxSimIsa::Neon, query);
        }
    }

    #[test]
    fn handles_nonfinite_scores() {
        fn check<const B: usize>()
        where
            Unsigned: MinMaxKernelFactory<B>,
        {
            let max_code = (1i64 << B) - 1;
            let query = constant_matrix::<8>(1, 5, 255);
            let mut document = constant_matrix::<B>(2, 5, max_code);
            let finite_score = -(5 * 255 * max_code) as f32;
            for isa in available_isas() {
                let kernel =
                    build_minmax_max_sim::<8, B, _>(isa, query.as_view().into(), BoxErase).unwrap();
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
                    assert_eq!(scores, [expected], "{isa}, B={B}");
                }
                let mut row = document.get_row_mut(0).unwrap();
                let mut meta = row.meta();
                meta.a = f32::NAN;
                row.set_meta(meta);
                kernel
                    .compute_max_sim(document.as_view(), &mut scores)
                    .unwrap();
                assert_eq!(scores, [finite_score], "{isa}, B={B}, mixed NaN/finite");
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
                build_minmax_max_sim::<8, 4, _>(isa, query.into(), BoxErase),
                Err(BuildMaxSimError::Overflow(_))
            ));
        }
    }
}
