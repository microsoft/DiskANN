/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{DistanceFunction, PreprocessedDistanceFunction};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    InputHints, PreScale, SphericalQuantizer, SupportedMetric,
    iface::{self, Opaque, OpaqueMut, Quantizer, QueryLayout},
};
use crate::{
    algorithms::TransformKind,
    alloc::{AlignedAllocator, GlobalAllocator, Poly, ScopedAllocator},
};

fn train(metric: SupportedMetric, hints: InputHints) -> SphericalQuantizer {
    let data = rowmajor::Owned::from_fn(2, 4, |rc| if rc.row == rc.col { 1.0f32 } else { 0.0 });
    SphericalQuantizer::train_with_hints(
        data.as_view(),
        TransformKind::Null,
        metric,
        PreScale::ReciprocalMeanNorm,
        hints,
        &mut StdRng::seed_from_u64(7),
        GlobalAllocator,
    )
    .unwrap()
}

fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 2.0e-4 * (1.0 + b.abs()), "{a} != {b}");
}

fn check_layouts<const BITS: usize>()
where
    iface::Impl<BITS>: iface::Constructible + Quantizer,
{
    let normalized = InputHints::new().with_normalized(true);
    let quantizer = train(SupportedMetric::Cosine, normalized);
    let reference = train(SupportedMetric::Cosine, InputHints::new());
    assert_eq!(quantizer.shift(), reference.shift());
    assert_eq!(quantizer.mean_norm(), reference.mean_norm());
    let plan = quantizer.as_quantizer_impl::<BITS>().unwrap();
    let plan: &dyn Quantizer = &plan;
    let mut code = vec![0; plan.bytes()];
    let mut ordinary = vec![0; plan.bytes()];
    plan.compress_with_hints(
        &[1.0, 0.0, 0.0, 0.0],
        normalized,
        OpaqueMut::new(&mut code),
        ScopedAllocator::global(),
    )
    .unwrap();
    plan.compress(
        &[3.0, 0.0, 0.0, 0.0],
        OpaqueMut::new(&mut ordinary),
        ScopedAllocator::global(),
    )
    .unwrap();
    assert_eq!(code, ordinary);

    for layout in [
        QueryLayout::SameAsData,
        QueryLayout::FullPrecision,
        QueryLayout::FourBitTransposed,
        QueryLayout::ScalarQuantized,
    ] {
        if !plan.is_supported(layout) {
            continue;
        }
        let unit = [0.6, 0.8, 0.0, 0.0];
        let raw = [3.0, 4.0, 0.0, 0.0];
        let hinted = plan
            .fused_query_computer_with_hints(
                &unit,
                layout,
                normalized,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .unwrap();
        let ordinary = plan
            .fused_query_computer(
                &raw,
                layout,
                false,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .unwrap();
        let expected = ordinary.evaluate_similarity(Opaque::new(&code)).unwrap();
        close(
            hinted.evaluate_similarity(Opaque::new(&code)).unwrap(),
            expected,
        );
        if layout == QueryLayout::FullPrecision {
            close(expected, 0.4);
        }

        let desc = plan.query_buffer_description(layout).unwrap();
        let alloc = AlignedAllocator::new(desc.align());
        let mut query = Poly::broadcast(0u8, desc.bytes(), alloc).unwrap();
        plan.compress_query_with_hints(
            &unit,
            layout,
            normalized,
            OpaqueMut::new(&mut query),
            ScopedAllocator::global(),
        )
        .unwrap();
        let computer = plan.query_computer(layout, GlobalAllocator).unwrap();
        close(
            computer
                .evaluate_similarity(Opaque::new(&query), Opaque::new(&code))
                .unwrap(),
            expected,
        );

        // Zero inputs remain valid with the normalized hint, including the full-query layout.
        let zero = plan
            .fused_query_computer_with_hints(
                &[0.0; 4],
                layout,
                normalized,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .unwrap();
        let ordinary_zero = plan
            .fused_query_computer(
                &[0.0; 4],
                layout,
                false,
                GlobalAllocator,
                ScopedAllocator::global(),
            )
            .unwrap();
        close(
            zero.evaluate_similarity(Opaque::new(&code)).unwrap(),
            ordinary_zero
                .evaluate_similarity(Opaque::new(&code))
                .unwrap(),
        );
    }
}

#[test]
fn normalized_data_and_raw_queries_all_layouts() {
    check_layouts::<1>();
    check_layouts::<2>();
    check_layouts::<3>();
    check_layouts::<4>();
    check_layouts::<8>();
}

fn check_rescale<const BITS: usize>()
where
    iface::Impl<BITS>: iface::Constructible + Quantizer,
{
    let distribution = InputHints::new().with_maybe_out_of_distribution(true);
    for metric in [
        SupportedMetric::InnerProduct,
        SupportedMetric::SquaredL2,
        SupportedMetric::Cosine,
    ] {
        let quantizer = train(metric, InputHints::new());
        let mut rescaled = [30.0, 40.0, 0.0, 0.0];
        if metric == SupportedMetric::InnerProduct {
            quantizer.rescale(&mut rescaled);
        }
        let plan = quantizer.as_quantizer_impl::<BITS>().unwrap();
        let plan: &dyn Quantizer = &plan;
        let mut code = vec![0; plan.bytes()];
        let mut ordinary = vec![0; plan.bytes()];
        plan.compress_with_hints(
            &[3.0, 4.0, 0.0, 0.0],
            distribution,
            OpaqueMut::new(&mut code),
            ScopedAllocator::global(),
        )
        .unwrap();
        plan.compress(
            &[3.0, 4.0, 0.0, 0.0],
            OpaqueMut::new(&mut ordinary),
            ScopedAllocator::global(),
        )
        .unwrap();
        assert_eq!(code, ordinary, "stored data must not be rescaled");

        for layout in [
            QueryLayout::SameAsData,
            QueryLayout::FullPrecision,
            QueryLayout::FourBitTransposed,
            QueryLayout::ScalarQuantized,
        ] {
            if !plan.is_supported(layout) {
                continue;
            }
            let hinted = plan
                .fused_query_computer_with_hints(
                    &[30.0, 40.0, 0.0, 0.0],
                    layout,
                    distribution,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .unwrap();
            let manual = plan
                .fused_query_computer(
                    &rescaled,
                    layout,
                    false,
                    GlobalAllocator,
                    ScopedAllocator::global(),
                )
                .unwrap();
            close(
                hinted.evaluate_similarity(Opaque::new(&code)).unwrap(),
                manual.evaluate_similarity(Opaque::new(&code)).unwrap(),
            );
            if metric == SupportedMetric::InnerProduct {
                let unit = plan
                    .fused_query_computer_with_hints(
                        &[0.6, 0.8, 0.0, 0.0],
                        layout,
                        distribution.with_normalized(true),
                        GlobalAllocator,
                        ScopedAllocator::global(),
                    )
                    .unwrap();
                close(
                    unit.evaluate_similarity(Opaque::new(&code)).unwrap(),
                    manual.evaluate_similarity(Opaque::new(&code)).unwrap(),
                );
                let zero = plan
                    .fused_query_computer_with_hints(
                        &[0.0; 4],
                        layout,
                        distribution,
                        GlobalAllocator,
                        ScopedAllocator::global(),
                    )
                    .unwrap();
                let ordinary_zero = plan
                    .fused_query_computer(
                        &[0.0; 4],
                        layout,
                        false,
                        GlobalAllocator,
                        ScopedAllocator::global(),
                    )
                    .unwrap();
                close(
                    zero.evaluate_similarity(Opaque::new(&code)).unwrap(),
                    ordinary_zero
                        .evaluate_similarity(Opaque::new(&code))
                        .unwrap(),
                );
            }
        }
    }
}

#[test]
fn query_rescale_matches_manual_rescale() {
    check_rescale::<1>();
    check_rescale::<2>();
    check_rescale::<3>();
    check_rescale::<4>();
    check_rescale::<8>();
}

#[test]
fn converted_integer_inputs_keep_cosine_semantics() {
    use diskann_vector::distance::Metric;
    let integers = rowmajor::Owned::from_fn(2, 8, |rc| if rc.row == 0 { 0u8 } else { 254 });
    let quantizer = SphericalQuantizer::train(
        integers.as_view(),
        TransformKind::Null,
        Metric::CosineNormalized.into(),
        PreScale::ReciprocalMeanNorm,
        &mut StdRng::seed_from_u64(7),
        GlobalAllocator,
    )
    .unwrap()
    .as_quantizer::<4>()
    .unwrap();
    let mut code = vec![0; quantizer.bytes()];
    quantizer
        .compress(
            &[254.0; 8],
            OpaqueMut::new(&mut code),
            ScopedAllocator::global(),
        )
        .unwrap();
    let query = quantizer
        .fused_query_computer(
            &[127.0; 8],
            QueryLayout::FullPrecision,
            false,
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .unwrap();
    close(query.evaluate_similarity(Opaque::new(&code)).unwrap(), 0.0);
}

#[cfg(feature = "flatbuffers")]
#[test]
fn hints_are_not_serialized() {
    let normalized = InputHints::new().with_normalized(true);
    let hinted = train(SupportedMetric::Cosine, normalized);
    let ordinary = train(SupportedMetric::Cosine, InputHints::new());
    let plan = hinted.as_quantizer_impl::<4>().unwrap();
    let bytes = plan.serialize(GlobalAllocator).unwrap();
    let ordinary = ordinary.as_quantizer_impl::<4>().unwrap();
    assert_eq!(&*bytes, &*ordinary.serialize(GlobalAllocator).unwrap());
    let restored = iface::Impl::<4>::try_deserialize(&bytes, GlobalAllocator).unwrap();
    let restored: &dyn Quantizer = &restored;
    assert_eq!(restored.metric(), SupportedMetric::Cosine);
    let mut code = vec![0; restored.bytes()];
    restored
        .compress(
            &[3.0, 0.0, 0.0, 0.0],
            OpaqueMut::new(&mut code),
            ScopedAllocator::global(),
        )
        .unwrap();
    let query = restored
        .fused_query_computer(
            &[3.0, 4.0, 0.0, 0.0],
            QueryLayout::FullPrecision,
            false,
            GlobalAllocator,
            ScopedAllocator::global(),
        )
        .unwrap();
    close(query.evaluate_similarity(Opaque::new(&code)).unwrap(), 0.4);
}
