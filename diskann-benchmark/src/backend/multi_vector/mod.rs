/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector MaxSim distance benchmarks with regression detection.
//!
//! Registers one `Benchmark` entry per element type accepted by the sealed
//! [`MaxSimElement`] trait. The JSON `isa` field selects the kernel at run
//! time via the library's [`build_max_sim`] factory.
//!
//! # Adding a new in-tree experimental kernel
//!
//! 1. Library (`diskann-quantization::multi_vector::distance`): add a
//!    variant to [`MaxSimIsa`], implement [`MaxSimKernel<T>`] for the new
//!    kernel struct, and add a matching arm to [`MaxSimElement::build`].
//! 2. Benchmark ([`crate::inputs::multi_vector::BenchIsa`]): mirror the
//!    variant and extend `From<BenchIsa> for MaxSimIsa`.
//! 3. Set `"isa": "your-variant"` in the JSON job. The existing
//!    `Kernel<T>` registrations cover the rest.
//!
//! # Why two enums?
//!
//! [`MaxSimIsa`] (library) and [`BenchIsa`] are kept separate so the library
//! doesn't pin its public API on a serde version or JSON shape. The
//! benchmark owns its kebab-case JSON layout; the library is serde-agnostic.
//!
//! The factory follows the BYOTE ("Bring your own type erasure") pattern
//! described in [RFC #1068]. Callers that want a different output shape
//! (chamfer-only closure, batched evaluator, etc.) implement their own
//! [`Erase<T>`] in place of [`BoxErase`].
//!
//! [`build_max_sim`]: diskann_quantization::multi_vector::build_max_sim
//! [`MaxSimIsa`]: diskann_quantization::multi_vector::MaxSimIsa
//! [`MaxSimElement`]: diskann_quantization::multi_vector::MaxSimElement
//! [`MaxSimElement::build`]: diskann_quantization::multi_vector::MaxSimElement::build
//! [`MaxSimKernel<T>`]: diskann_quantization::multi_vector::MaxSimKernel
//! [`Erase<T>`]: diskann_quantization::multi_vector::Erase
//! [`BoxErase`]: diskann_quantization::multi_vector::BoxErase
//! [`BenchIsa`]: crate::inputs::multi_vector::BenchIsa
//! [RFC #1068]: https://github.com/microsoft/DiskANN/pull/1068

use diskann_benchmark_runner::Registry;

cfg_if::cfg_if! {
    if #[cfg(feature = "multi-vector")] {
        mod driver;
        mod kernels;

        pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            kernels::register(registry)
        }
    } else {
        crate::utils::stub_impl!("multi-vector", inputs::multi_vector::MultiVectorOp);

        pub(super) fn register_benchmarks(registry: &mut Registry) -> anyhow::Result<()> {
            imp::register("multi-vector-op", registry)
        }
    }
}

#[cfg(all(test, feature = "multi-vector"))]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_benchmark_runner::{
        benchmark::{PassFail, Regression},
        utils::{
            datatype::DataType, num::NonNegativeFinite, percentiles::compute_percentiles,
            MicroSeconds,
        },
    };

    use super::driver::{CheckResult, Comparison, MultiVectorTolerance, RunResult};
    use super::kernels::Kernel;
    use crate::inputs::multi_vector::{BenchIsa, MultiVectorOp, Run};

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
            isa: BenchIsa::Auto,
            runs: vec![tiny_run()],
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
}
