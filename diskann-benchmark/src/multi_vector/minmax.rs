/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_quantization::{
    algorithms::transforms::{NullTransform, Transform},
    bits::{Representation, Unsigned},
    minmax::{build_minmax_max_sim, MinMaxKernelFactory, MinMaxMeta, MinMaxQuantizer},
    multi_vector::{Defaulted, Mat, MatRef, MaxSimIsa, Standard},
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
    match (query_bits.get(), doc_bits.get()) {
        (1, 1) => run_with_bits::<1, 1>(run, isa),
        (2, 2) => run_with_bits::<2, 2>(run, isa),
        (3, 3) => run_with_bits::<3, 3>(run, isa),
        (4, 4) => run_with_bits::<4, 4>(run, isa),
        (5, 5) => run_with_bits::<5, 5>(run, isa),
        (6, 6) => run_with_bits::<6, 6>(run, isa),
        (7, 7) => run_with_bits::<7, 7>(run, isa),
        (8, 8) => run_with_bits::<8, 8>(run, isa),
        (8, 1) => run_with_bits::<8, 1>(run, isa),
        (8, 2) => run_with_bits::<8, 2>(run, isa),
        (8, 4) => run_with_bits::<8, 4>(run, isa),
        (a, b) => anyhow::bail!("unsupported MinMax bit-width pair: query {a}, document {b}"),
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

fn quantized_data<const A: usize, const B: usize>(
    run: &Run,
) -> anyhow::Result<(Mat<MinMaxMeta<A>>, Mat<MinMaxMeta<B>>)>
where
    Unsigned: Representation<A> + Representation<B>,
{
    let data = Data::<f32>::new(run)?;
    let quantizer = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(run.dim)),
        Positive::new(1.0)?,
    );
    Ok((
        compress::<A>(&quantizer, data.queries.as_view())?,
        compress::<B>(&quantizer, data.docs.as_view())?,
    ))
}

fn run_with_bits<const A: usize, const B: usize>(
    run: &Run,
    isa: MaxSimIsa,
) -> anyhow::Result<RunResult>
where
    Unsigned: MinMaxKernelFactory<A, B>,
{
    let (query, document) = quantized_data::<A, B>(run)?;
    let query = query.as_view().into();
    let document = document.as_view();
    let kernel = build_minmax_max_sim::<A, B>(isa, query)?;
    let mut scores = vec![0.0; kernel.nrows()];

    Ok(run_with_kernel(run, &mut scores, |scores| {
        kernel.compute_max_sim(document, scores)
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inputs::multi_vector::BenchIsa;
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
    fn dispatches_supported_pairs_and_rejects_others() {
        let shape = shape();
        for a in 1..=8 {
            for b in 1..=8 {
                for isa in [MaxSimIsa::Auto, MaxSimIsa::Scalar, MaxSimIsa::Reference] {
                    let result = run(
                        &shape,
                        isa,
                        BitWidth::try_from(a).unwrap(),
                        BitWidth::try_from(b).unwrap(),
                    );
                    if a != b && !(a == 8 && matches!(b, 1 | 2 | 4)) {
                        assert!(
                            result
                                .unwrap_err()
                                .to_string()
                                .contains("unsupported MinMax"),
                            "{a} x {b} ({isa})",
                        );
                        continue;
                    }
                    let result = result.unwrap();
                    assert_eq!(result.run, shape);
                    assert_eq!(result.latencies.len(), 1);
                }
            }
        }
    }

    fn check_quantized_scores<const A: usize, const B: usize>()
    where
        Unsigned: MinMaxKernelFactory<A, B>,
    {
        let shape = shape();
        let (query, document) = quantized_data::<A, B>(&shape).unwrap();
        let query = query.as_view().into();
        let document = document.as_view();
        let reference = build_minmax_max_sim::<A, B>(MaxSimIsa::Reference, query).unwrap();
        let mut expected = vec![0.0; reference.nrows()];
        reference.compute_max_sim(document, &mut expected).unwrap();

        for isa in BenchIsa::all()
            .into_iter()
            .map(MaxSimIsa::from)
            .filter(|isa| isa.is_available())
        {
            let kernel = build_minmax_max_sim::<A, B>(isa, query).unwrap();
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
        check_quantized_scores::<1, 1>();
        check_quantized_scores::<2, 2>();
        check_quantized_scores::<3, 3>();
        check_quantized_scores::<4, 4>();
        check_quantized_scores::<5, 5>();
        check_quantized_scores::<6, 6>();
        check_quantized_scores::<7, 7>();
        check_quantized_scores::<8, 8>();
        check_quantized_scores::<8, 1>();
        check_quantized_scores::<8, 2>();
        check_quantized_scores::<8, 4>();
    }
}
