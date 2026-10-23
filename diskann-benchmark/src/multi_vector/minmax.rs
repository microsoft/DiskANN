/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    algorithms::transforms::{NullTransform, Transform},
    bits::{Representation, Unsigned},
    minmax::{build_minmax_max_sim, MinMaxKernelFactory, MinMaxMeta, MinMaxQuantizer},
    multi_vector::{BoxErase, Defaulted, Mat, MatRef, MaxSimIsa, Standard},
    num::Positive,
    CompressInto,
};

use super::driver::{run_with_kernel, Data, RunResult};
use crate::inputs::multi_vector::{BitWidth, Run};

// Keep runtime-to-const dispatch at the benchmark boundary.
pub(super) fn run(
    run: &Run,
    isa: MaxSimIsa,
    query_bits: BitWidth,
    doc_bits: BitWidth,
) -> anyhow::Result<RunResult> {
    match query_bits.get() {
        1 => run_query::<1>(run, isa, doc_bits),
        2 => run_query::<2>(run, isa, doc_bits),
        3 => run_query::<3>(run, isa, doc_bits),
        4 => run_query::<4>(run, isa, doc_bits),
        5 => run_query::<5>(run, isa, doc_bits),
        6 => run_query::<6>(run, isa, doc_bits),
        7 => run_query::<7>(run, isa, doc_bits),
        8 => run_query::<8>(run, isa, doc_bits),
        _ => unreachable!("BitWidth is validated on construction"),
    }
}

fn run_query<const A: usize>(
    run: &Run,
    isa: MaxSimIsa,
    doc_bits: BitWidth,
) -> anyhow::Result<RunResult>
where
    Unsigned: Representation<A>,
{
    match doc_bits.get() {
        1 => run_typed::<A, 1>(run, isa),
        2 => run_typed::<A, 2>(run, isa),
        3 => run_typed::<A, 3>(run, isa),
        4 => run_typed::<A, 4>(run, isa),
        5 => run_typed::<A, 5>(run, isa),
        6 => run_typed::<A, 6>(run, isa),
        7 => run_typed::<A, 7>(run, isa),
        8 => run_typed::<A, 8>(run, isa),
        _ => unreachable!("BitWidth is validated on construction"),
    }
}

fn compress<const BITS: usize>(
    quantizer: &MinMaxQuantizer,
    source: MatRef<'_, Standard<f32>>,
) -> anyhow::Result<Mat<MinMaxMeta<BITS>>>
where
    Unsigned: Representation<BITS>,
{
    let mut compressed = Mat::new(
        MinMaxMeta::new(source.num_vectors(), source.vector_dim()),
        Defaulted,
    )?;
    quantizer.compress_into(source, compressed.as_view_mut())?;
    Ok(compressed)
}

fn run_typed<const A: usize, const B: usize>(run: &Run, isa: MaxSimIsa) -> anyhow::Result<RunResult>
where
    Unsigned: Representation<A> + MinMaxKernelFactory<B>,
{
    let data = Data::<f32>::new(run)?;
    let quantizer = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(run.dim)),
        Positive::new(1.0)?,
    );
    let query = compress::<A>(&quantizer, data.queries.as_view())?;
    let document = compress::<B>(&quantizer, data.docs.as_view())?;
    let query = query.as_view().into();
    let document = document.as_view();
    let kernel = build_minmax_max_sim::<A, B, _>(isa, query, BoxErase)?;
    let mut scores = vec![0.0; kernel.nrows()];

    Ok(run_with_kernel(run, &mut scores, |scores| {
        kernel.compute_max_sim(document, scores)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    fn shape() -> Run {
        Run {
            num_query_vectors: NonZeroUsize::new(17).unwrap(),
            num_doc_vectors: NonZeroUsize::new(9).unwrap(),
            dim: NonZeroUsize::new(65).unwrap(),
            loops_per_measurement: NonZeroUsize::new(1).unwrap(),
            num_measurements: NonZeroUsize::new(1).unwrap(),
        }
    }

    #[test]
    fn dispatches_all_widths() {
        let shape = shape();
        for a in 1..=8 {
            for b in 1..=8 {
                for isa in [MaxSimIsa::Auto, MaxSimIsa::Scalar, MaxSimIsa::Reference] {
                    let result = run(
                        &shape,
                        isa,
                        BitWidth::try_from(a).unwrap(),
                        BitWidth::try_from(b).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(result.run, shape);
                    assert_eq!(result.latencies.len(), 1);
                }
            }
        }
    }

    fn check_quantized_scores<const A: usize, const B: usize>()
    where
        Unsigned: Representation<A> + MinMaxKernelFactory<B>,
    {
        let shape = shape();
        let data = Data::<f32>::new(&shape).unwrap();
        let quantizer = MinMaxQuantizer::new(
            Transform::Null(NullTransform::new(shape.dim)),
            Positive::new(1.0).unwrap(),
        );
        let query = compress::<A>(&quantizer, data.queries.as_view()).unwrap();
        let document = compress::<B>(&quantizer, data.docs.as_view()).unwrap();
        let query = query.as_view().into();
        let document = document.as_view();
        let reference =
            build_minmax_max_sim::<A, B, _>(MaxSimIsa::Reference, query, BoxErase).unwrap();
        let mut expected = vec![0.0; reference.nrows()];
        reference.compute_max_sim(document, &mut expected).unwrap();

        for isa in [
            MaxSimIsa::Auto,
            MaxSimIsa::Scalar,
            MaxSimIsa::X86_64_V3,
            MaxSimIsa::X86_64_V4,
            MaxSimIsa::Neon,
        ]
        .into_iter()
        .filter(|isa| isa.is_available())
        {
            let kernel = build_minmax_max_sim::<A, B, _>(isa, query, BoxErase).unwrap();
            let mut scores = vec![f32::NAN; expected.len()];
            kernel.compute_max_sim(document, &mut scores).unwrap();
            for (i, (&actual, &expected)) in scores.iter().zip(&expected).enumerate() {
                assert!(
                    actual.is_finite()
                        && expected.is_finite()
                        && (actual - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                    "MinMax{A} x MinMax{B} ({isa}) row {i}: {actual} != reference {expected}",
                );
            }
        }
    }

    #[test]
    fn quantized_scores_match_reference_across_widths() {
        fn check_query<const A: usize>()
        where
            Unsigned: Representation<A>,
        {
            check_quantized_scores::<A, 1>();
            check_quantized_scores::<A, 2>();
            check_quantized_scores::<A, 3>();
            check_quantized_scores::<A, 4>();
            check_quantized_scores::<A, 5>();
            check_quantized_scores::<A, 6>();
            check_quantized_scores::<A, 7>();
            check_quantized_scores::<A, 8>();
        }
        check_query::<1>();
        check_query::<2>();
        check_query::<3>();
        check_query::<4>();
        check_query::<5>();
        check_query::<6>();
        check_query::<7>();
        check_query::<8>();
    }
}
