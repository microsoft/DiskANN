/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector MaxSim distance benchmarks with regression detection.
//!
//! This module is a **kernel-research substrate**, not just a benchmark. It
//! supports two distinct use cases:
//!
//! 1. **Head-to-head ISA (instruction set architecture) comparison.** Library
//!    kernels are registered per arch (`scalar`, `x86-64-v3`, `x86-64-v4`,
//!    `aarch64-neon`) plus `auto` (CPU-detected) and `reference` (fallback).
//!    Pinning to a specific ISA lets you compare e.g. AVX2 vs AVX512 on the
//!    same AVX512 host.
//!
//! 2. **Experimental kernel authoring.** External crates and the
//!    `experimental/` submodule can author new SIMD micro-kernels by
//!    implementing the public `Kernel<A>` trait in
//!    `diskann-quantization::multi_vector::distance::kernels`, plug them
//!    into the existing cache-aware tile orchestrator (`tiled_reduce`),
//!    and slot them into the benchmark via
//!    `QueryComputer::from_dyn(Box::new(...))`.
//!
//! # Adding a new experimental kernel
//!
//! See `experimental/template.rs` for the full step-by-step workflow with
//! a worked example. Summary:
//!
//! 1. Add a variant to [`crate::inputs::multi_vector::Arch`].
//! 2. Implement `Kernel<A>` for your micro-kernel.
//! 3. Implement `DynQueryComputer<T>` for your adapter, calling
//!    `tiled_reduce` with your kernel.
//! 4. Add a marker type + `DispatchRule<Arch>` impl so the new variant
//!    routes to your kernel.
//! 5. Add a `RunBenchmark<Marker>` impl + `register_regression(...)` call
//!    in `experimental::register`.
//!
//! **Validate experimental kernels under Miri:**
//! - Construct arch tokens via `Scalar::new()` (Miri-safe) or
//!   `V4::new_checked_miri()` (Miri-safe AVX-512 emulation). `V3::new_checked()`
//!   and `Neon::new_checked()` don't have `_miri` variants today; if you need
//!   them under Miri, follow `V4::new_checked_miri()`'s pattern.
//! - Gate Miri-unsupported intrinsics with `#[cfg(not(miri))]`.
//! - Reduce test-sweep size under `cfg(miri)` to keep runtimes reasonable.

use diskann_benchmark_runner::registry::Benchmarks;

cfg_if::cfg_if! {
    if #[cfg(feature = "multi-vector")] {
        mod driver;
        mod experimental;
        mod library_kernels;

        pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
            library_kernels::register(benchmarks);
            experimental::register(benchmarks);
        }
    } else {
        crate::utils::stub_impl!("multi-vector", inputs::multi_vector::MultiVectorOp);

        pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
            imp::register("multi-vector-op", benchmarks);
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

    use super::driver::{CheckResult, Comparison, RunResult};
    use super::library_kernels::{Auto, Kernel};
    use crate::inputs::multi_vector::{Arch, MultiVectorOp, MultiVectorTolerance, Run};

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
            arch: Arch::Auto,
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
        let kernel = Kernel::<Auto, f32>::new();

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
        let kernel = Kernel::<Auto, f32>::new();

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
        let kernel = Kernel::<Auto, f32>::new();

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
        let kernel = Kernel::<Auto, f32>::new();

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
        let kernel = Kernel::<Auto, f32>::new();

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
