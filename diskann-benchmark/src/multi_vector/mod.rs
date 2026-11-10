/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector MaxSim distance benchmarks with regression detection.
//!
//! The JSON `element_type` and `isa` fields select the representation and backend.
//!
//! Timing excludes data generation, quantization, query preparation and warmup.
//! Each run reuses one document and score buffer; `ns/IP` is normalized by the
//! number of query/document vector pairs and calls per measurement.
//!
//! # Why two ISA enums?
//!
//! [`MaxSimIsa`] (library) and [`BenchIsa`] (this crate) are intentionally
//! separate so the library doesn't pin its public API on a serde version
//! or JSON shape. The benchmark owns its kebab-case JSON layout; the
//! library stays serde-agnostic.
//!
//! [`MaxSimIsa`]: diskann_quantization::multi_vector::MaxSimIsa
//! [`BenchIsa`]: crate::inputs::multi_vector::BenchIsa

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "multi-vector")] {
        mod driver;
        mod kernels;
        mod minmax;

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
        output::Memory,
        utils::{num::NonNegativeFinite, percentiles::compute_percentiles, MicroSeconds},
        Benchmark,
    };
    use diskann_quantization::multi_vector::MaxSimIsa;

    use super::driver::{CheckResult, Comparison, MultiVectorTolerance, RunResult};
    use super::kernels::Kernel;
    use crate::inputs::multi_vector::{BenchIsa, BitWidth, ElementType, MultiVectorOp, Run};

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
            element_type: ElementType::Float32,
            isa: BenchIsa::Auto,
            runs: vec![tiny_run()],
        }
    }

    fn representations() -> [ElementType; 5] {
        [
            ElementType::Float32,
            ElementType::Float16,
            ElementType::Int8,
            ElementType::MinMax {
                query_bits: BitWidth::try_from(4).unwrap(),
                doc_bits: BitWidth::try_from(4).unwrap(),
            },
            ElementType::MinMax {
                query_bits: BitWidth::try_from(8).unwrap(),
                doc_bits: BitWidth::try_from(4).unwrap(),
            },
        ]
    }

    fn run_op(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
        let mut output = Memory::new();
        match input.element_type {
            ElementType::Float32 => Kernel::standard::<f32>().run(input, &mut output),
            ElementType::Float16 => Kernel::standard::<half::f16>().run(input, &mut output),
            ElementType::Int8 => Kernel::standard::<i8>().run(input, &mut output),
            ElementType::MinMax { .. } => Kernel::minmax().run(input, &mut output),
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
        let kernel = Kernel::standard::<f32>();

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
        let kernel = Kernel::standard::<f32>();

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
        let kernel = Kernel::standard::<f32>();

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
        let kernel = Kernel::standard::<f32>();

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
        let kernel = Kernel::standard::<f32>();

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
    fn timing_harness_warms_up_before_measurements() {
        let run = Run {
            loops_per_measurement: NonZeroUsize::new(2).unwrap(),
            num_measurements: NonZeroUsize::new(3).unwrap(),
            ..tiny_run()
        };
        let mut calls = 0;
        let result = super::driver::run_with_kernel(&run, &mut [0.0; 2], |scores| {
            calls += 1;
            scores.fill(calls as f32);
            Ok(())
        })
        .unwrap();
        assert_eq!(calls, 7);
        assert_eq!(result.run, run);
        assert_eq!(result.latencies.len(), 3);
    }

    #[test]
    fn timing_harness_propagates_errors() {
        use diskann_quantization::multi_vector::MaxSimError;

        let run = Run {
            loops_per_measurement: NonZeroUsize::new(2).unwrap(),
            num_measurements: NonZeroUsize::new(3).unwrap(),
            ..tiny_run()
        };
        for fail_on in [1, 2, 3, 4, 7] {
            let mut calls = 0;
            let error = super::driver::run_with_kernel(&run, &mut [0.0; 2], |_| {
                calls += 1;
                if calls == fail_on {
                    Err(MaxSimError::InvalidBufferLength(0, 2))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
            assert!(matches!(error, MaxSimError::InvalidBufferLength(0, 2)));
            assert_eq!(calls, fail_on);
        }
    }

    #[test]
    fn benchmarks_match_representation_and_available_isa() {
        fn check_match<B: Benchmark<Input = MultiVectorOp>>(
            benchmark: &B,
            input: &MultiVectorOp,
            expected: bool,
        ) {
            let matched = matches!(MatchContext::test(benchmark, input), TestScore::Success(_));
            assert_eq!(matched, expected, "{} / {}", input.element_type, input.isa);
        }

        for element_type in representations() {
            for isa in BenchIsa::all() {
                let input = MultiVectorOp {
                    element_type,
                    isa,
                    ..tiny_op()
                };
                let available = MaxSimIsa::from(isa).is_available();
                check_match(
                    &Kernel::standard::<f32>(),
                    &input,
                    available && element_type == ElementType::Float32,
                );
                check_match(
                    &Kernel::standard::<half::f16>(),
                    &input,
                    available && element_type == ElementType::Float16,
                );
                check_match(
                    &Kernel::standard::<i8>(),
                    &input,
                    available && element_type == ElementType::Int8,
                );
                check_match(
                    &Kernel::minmax(),
                    &input,
                    available && matches!(element_type, ElementType::MinMax { .. }),
                );
            }
        }
    }

    #[test]
    fn benchmark_results_match_runs_and_round_trip() {
        for element_type in representations() {
            let input = MultiVectorOp {
                element_type,
                isa: BenchIsa::Scalar,
                runs: vec![
                    tiny_run(),
                    Run {
                        dim: NonZeroUsize::new(5).unwrap(),
                        num_measurements: NonZeroUsize::new(3).unwrap(),
                        ..tiny_run()
                    },
                ],
            };
            let results = run_op(&input).unwrap();
            assert_eq!(results.len(), input.runs.len());
            for (result, run) in results.iter().zip(&input.runs) {
                assert_eq!(&result.run, run);
                assert_eq!(result.latencies.len(), run.num_measurements.get());
            }

            let serialized = serde_json::to_vec(&results).unwrap();
            let restored: Vec<RunResult> = serde_json::from_slice(&serialized).unwrap();
            assert_eq!(restored.len(), results.len());
            for (restored, original) in restored.iter().zip(&results) {
                assert_eq!(restored.run, original.run);
                assert_eq!(restored.latencies, original.latencies);
                assert_eq!(restored.percentiles.minimum, original.percentiles.minimum);
            }
        }
    }

    #[test]
    fn benchmark_runs_propagate_errors() {
        for element_type in representations() {
            let input = MultiVectorOp {
                element_type,
                runs: vec![Run {
                    dim: NonZeroUsize::MAX,
                    ..tiny_run()
                }],
                ..tiny_op()
            };
            assert!(run_op(&input).is_err(), "{element_type}");
        }
    }

    #[test]
    fn minmax_rejects_primitive_input() {
        let error = Kernel::minmax()
            .run(&tiny_op(), &mut Memory::new())
            .unwrap_err();
        assert!(error.to_string().contains("expected a MinMax element type"));
    }

    #[test]
    fn minmax_uses_shared_regression_checks() {
        let input = MultiVectorOp {
            element_type: ElementType::MinMax {
                query_bits: BitWidth::try_from(8).unwrap(),
                doc_bits: BitWidth::try_from(4).unwrap(),
            },
            ..tiny_op()
        };
        for (after, passes) in [(95, true), (105, true), (106, false)] {
            let result = Kernel::minmax()
                .check(
                    &tolerance(0.05),
                    &input,
                    &vec![tiny_result(100)],
                    &vec![tiny_result(after)],
                )
                .unwrap();
            assert_eq!(matches!(result, PassFail::Pass(_)), passes);
        }
        let error = Kernel::minmax()
            .check(&tolerance(0.05), &input, &vec![tiny_result(100)], &vec![])
            .unwrap_err();
        assert_eq!(error.to_string(), "before has 1 runs but after has 0");
    }
}
