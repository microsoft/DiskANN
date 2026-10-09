/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector MaxSim distance benchmarks with regression detection.
//!
//! One `Benchmark` is registered per element type supported by
//! [`MaxSimElement`]; the JSON `isa` field picks the kernel at run time.
//! The shared `multi-vector-op` input selects dense or MinMax8/MinMax4 vectors
//! with the optional `format` field.
//!
//! # Why two ISA enums?
//!
//! [`MaxSimIsa`] (library) and [`BenchIsa`] (this crate) are intentionally
//! separate so the library doesn't pin its public API on a serde version
//! or JSON shape. The benchmark owns its kebab-case JSON layout; the
//! library stays serde-agnostic.
//!
//! [`MaxSimIsa`]: diskann_quantization::multi_vector::MaxSimIsa
//! [`MaxSimElement`]: diskann_quantization::multi_vector::MaxSimElement
//! [`BenchIsa`]: crate::inputs::multi_vector::BenchIsa

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "multi-vector")] {
        mod driver;
        mod kernels;

        pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            kernels::register(registry)
        }
    } else {
        pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            registry.register_partially_gated::<crate::inputs::multi_vector::MultiVectorOp>(
                "multi-vector-op",
                diskann_benchmark_runner::Features::new("multi-vector"),
                "Multi-vector distance function benchmarks",
            )?;

            Ok(())
        }
    }
}

#[cfg(all(test, feature = "multi-vector"))]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_benchmark_runner::{
        benchmark::{MatchContext, PassFail, Regression, TestScore},
        utils::{
            datatype::DataType, num::NonNegativeFinite, percentiles::compute_percentiles,
            MicroSeconds,
        },
    };

    use diskann_quantization::{
        minmax::build_minmax_max_sim,
        multi_vector::{BoxErase, MaxSim, MaxSimIsa, QueryMatRef},
    };
    use diskann_vector::DistanceFunctionMut;

    use super::driver::{
        run_with_compute, CheckResult, Comparison, MinMax8Data, MultiVectorTolerance, RunResult,
    };
    use super::kernels::{Kernel, MinMax8Kernel};
    use crate::inputs::multi_vector::{BenchIsa, MultiVectorFormat, MultiVectorOp, Run};

    fn tiny_run() -> Run {
        Run {
            num_query_vectors: NonZeroUsize::new(2).unwrap(),
            num_doc_vectors: NonZeroUsize::new(2).unwrap(),
            dim: NonZeroUsize::new(4).unwrap(),
            loops_per_measurement: NonZeroUsize::new(1).unwrap(),
            num_measurements: NonZeroUsize::new(1).unwrap(),
        }
    }

    fn tiny_op() -> MultiVectorOp {
        MultiVectorOp {
            element_type: DataType::Float32,
            format: MultiVectorFormat::Dense,
            isa: BenchIsa::Auto,
            runs: vec![tiny_run()],
        }
    }

    #[test]
    fn multi_vector_format_selects_only_matching_kernel() {
        for (format, element_type, selected) in [
            (MultiVectorFormat::Dense, DataType::Float32, 0),
            (MultiVectorFormat::Dense, DataType::Float16, 1),
            (MultiVectorFormat::Dense, DataType::Int8, 2),
            (MultiVectorFormat::MinMax8, DataType::Float32, 3),
        ] {
            let input = MultiVectorOp {
                element_type,
                format,
                isa: BenchIsa::Scalar,
                runs: vec![tiny_run()],
            };
            let scores = [
                MatchContext::test(&Kernel::<f32>::new(), &input),
                MatchContext::test(&Kernel::<half::f16>::new(), &input),
                MatchContext::test(&Kernel::<i8>::new(), &input),
                MatchContext::test(&MinMax8Kernel, &input),
            ];
            for (index, score) in scores.into_iter().enumerate() {
                assert_eq!(
                    matches!(score, TestScore::Success(_)),
                    index == selected,
                    "{input:?}: {score:?}"
                );
            }
        }
    }

    fn tiny_result(minimum: u64) -> RunResult {
        let mut latencies = vec![MicroSeconds::new(minimum)];
        let percentiles = compute_percentiles(&mut latencies).unwrap();
        RunResult {
            run: tiny_run(),
            latencies,
            percentiles,
        }
    }

    fn tolerance(limit: f64) -> MultiVectorTolerance {
        MultiVectorTolerance {
            min_time_regression: NonNegativeFinite::new(limit).unwrap(),
        }
    }

    #[test]
    fn check_rejects_mismatched_runs() {
        let kernel = Kernel::<f32>::new();

        // Build a result whose `run` diverges from `tiny_run()` so the
        // regression check's `b.run == a.run` invariant fires.
        let mut latencies = vec![MicroSeconds::new(100)];
        let percentiles = compute_percentiles(&mut latencies).unwrap();
        let mismatched_result = RunResult {
            run: Run {
                num_query_vectors: NonZeroUsize::new(4).unwrap(),
                ..tiny_run()
            },
            latencies,
            percentiles,
        };

        let err = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(100)],
                &vec![mismatched_result],
            )
            .unwrap_err();

        assert_eq!(err.to_string(), "run 0 mismatched");
    }

    #[test]
    fn check_allows_negative_relative_change() {
        let kernel = Kernel::<f32>::new();

        let result = kernel
            .check(
                &tolerance(0.0),
                &tiny_op(),
                &vec![tiny_result(100)],
                &vec![tiny_result(95)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_passes_on_tolerance_boundary() {
        let kernel = Kernel::<f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(100)],
                &vec![tiny_result(105)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Pass(_)));
    }

    #[test]
    fn check_fails_above_tolerance_boundary() {
        let kernel = Kernel::<f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(100)],
                &vec![tiny_result(106)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }

    #[test]
    fn check_result_display_includes_failure_details() {
        let check = CheckResult {
            checks: vec![Comparison {
                run: tiny_run(),
                tolerance: tolerance(0.05),
                before_min: 100.0,
                after_min: 106.0,
            }],
        };

        let rendered = check.to_string();
        assert!(rendered.contains("Q"), "rendered = {rendered}");
        assert!(rendered.contains("Dim"), "rendered = {rendered}");
        assert!(rendered.contains("100.000"), "rendered = {rendered}");
        assert!(rendered.contains("106.000"), "rendered = {rendered}");
        assert!(rendered.contains("6.000 %"), "rendered = {rendered}");
        assert!(rendered.contains("FAIL"), "rendered = {rendered}");
    }

    /// A "before" value of 0 means the measurement was too fast to obtain a
    /// reliable signal, so we *could* be letting a regression through. We
    /// require at least a non-zero value.
    #[test]
    fn zero_values_rejected() {
        let kernel = Kernel::<f32>::new();

        let result = kernel
            .check(
                &tolerance(0.05),
                &tiny_op(),
                &vec![tiny_result(0)],
                &vec![tiny_result(0)],
            )
            .unwrap();

        assert!(matches!(result, PassFail::Fail(_)));
    }

    #[test]
    fn minmax8_fixtures_match_reference() {
        for (queries, docs, dim) in [
            (16, 64, 256),
            (7, 5, 63),
            (16, 16, 128),
            (17, 13, 129),
            (32, 65, 768),
        ] {
            let run = Run {
                num_query_vectors: NonZeroUsize::new(queries).unwrap(),
                num_doc_vectors: NonZeroUsize::new(docs).unwrap(),
                dim: NonZeroUsize::new(dim).unwrap(),
                ..tiny_run()
            };
            let data = MinMax8Data::new(&run).unwrap();
            let mut expected = vec![0.0; queries];
            MaxSim::new(&mut expected).evaluate(
                QueryMatRef::from(data.queries.as_view()),
                data.docs.as_view(),
            );
            assert!(expected.iter().all(|score| score.is_finite()));
            for isa in [
                MaxSimIsa::Scalar,
                MaxSimIsa::X86_64_V3,
                MaxSimIsa::X86_64_V4,
                MaxSimIsa::Neon,
                MaxSimIsa::Auto,
            ] {
                if !isa.is_available() {
                    continue;
                }
                let kernel = build_minmax_max_sim(isa, data.queries.as_view(), BoxErase).unwrap();
                let mut actual = vec![f32::NAN; queries];
                kernel
                    .compute_max_sim(data.docs.as_view(), &mut actual)
                    .unwrap();
                assert_eq!(actual, expected, "{isa:?}: {run:?}");
            }
        }
    }

    #[test]
    fn minmax8_uses_shared_regression_check() {
        let input = MultiVectorOp {
            element_type: DataType::Float32,
            format: MultiVectorFormat::MinMax8,
            isa: BenchIsa::Scalar,
            runs: vec![tiny_run()],
        };
        for (after, should_pass) in [(105, true), (106, false), (0, true)] {
            let result = MinMax8Kernel
                .check(
                    &tolerance(0.05),
                    &input,
                    &vec![tiny_result(100)],
                    &vec![tiny_result(after)],
                )
                .unwrap();
            assert_eq!(matches!(result, PassFail::Pass(_)), should_pass);
        }
    }

    #[test]
    fn timing_counts_computations_and_propagates_errors() {
        let run = Run {
            loops_per_measurement: NonZeroUsize::new(3).unwrap(),
            num_measurements: NonZeroUsize::new(2).unwrap(),
            ..tiny_run()
        };
        let mut calls = 0;
        let result = run_with_compute(&run, || {
            calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 6);
        assert_eq!(result.latencies.len(), 2);
        assert_eq!(result.run, run);

        let error = run_with_compute(&run, || anyhow::bail!("compute failed")).unwrap_err();
        assert_eq!(error.to_string(), "compute failed");
    }
}
