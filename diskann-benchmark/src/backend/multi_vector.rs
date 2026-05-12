/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Multi-vector distance benchmarks (Chamfer / MaxSim) with regression detection.

use diskann_benchmark_runner::registry::Benchmarks;

// Create a stub-module if the "multi-vector" feature is disabled.
crate::utils::stub_impl!("multi-vector", inputs::multi_vector::MultiVectorOp);

pub(super) fn register_benchmarks(benchmarks: &mut Benchmarks) {
    #[cfg(feature = "multi-vector")]
    {
        use half::f16;

        // Optimized (architecture-dispatched QueryComputer).
        benchmarks.register_regression(
            "multi-vector-op-f32-optimized",
            imp::Kernel::<imp::Optimized, f32>::new(),
        );
        benchmarks.register_regression(
            "multi-vector-op-f16-optimized",
            imp::Kernel::<imp::Optimized, f16>::new(),
        );

        // Reference (Chamfer / MaxSim fallback path).
        benchmarks.register_regression(
            "multi-vector-op-f32-reference",
            imp::Kernel::<imp::Reference, f32>::new(),
        );
        benchmarks.register_regression(
            "multi-vector-op-f16-reference",
            imp::Kernel::<imp::Reference, f16>::new(),
        );
    }

    // Stub implementation
    #[cfg(not(feature = "multi-vector"))]
    imp::register("multi-vector-op", benchmarks);
}

#[cfg(feature = "multi-vector")]
mod imp {
    use std::io::Write;

    use diskann_benchmark_runner::{
        benchmark::{PassFail, Regression},
        dispatcher::{DispatchRule, FailureScore, MatchScore},
        utils::{datatype, num::relative_change, percentiles, MicroSeconds},
        Benchmark,
    };
    use diskann_quantization::multi_vector::{
        Chamfer, Init, Mat, MatRef, MaxSim, QueryComputer, Standard,
    };
    use diskann_vector::distance::InnerProduct;
    use diskann_vector::{DistanceFunctionMut, PureDistanceFunction};
    use half::f16;
    use rand::{
        distr::{Distribution, StandardUniform},
        rngs::StdRng,
        SeedableRng,
    };
    use serde::{Deserialize, Serialize};

    use crate::inputs::multi_vector::{
        Implementation, MultiVectorOp, MultiVectorTolerance, Operation, Run,
    };

    ///////////
    // Utils //
    ///////////

    #[derive(Debug, Clone, Copy)]
    pub(super) struct DisplayWrapper<'a, T: ?Sized>(pub(super) &'a T);

    impl<T: ?Sized> std::ops::Deref for DisplayWrapper<'_, T> {
        type Target = T;
        fn deref(&self) -> &T {
            self.0
        }
    }

    //////////////
    // Dispatch //
    //////////////

    /// Dispatch marker for the [`QueryComputer`] implementation.
    #[derive(Debug)]
    pub(super) struct Optimized;

    /// Dispatch marker for the [`Chamfer`] / [`MaxSim`] fallback.
    #[derive(Debug)]
    pub(super) struct Reference;

    /// A multi-vector benchmark.
    pub(super) struct Kernel<I, T> {
        _type: std::marker::PhantomData<(I, T)>,
    }

    impl<I, T> Kernel<I, T> {
        pub(super) fn new() -> Self {
            Self {
                _type: std::marker::PhantomData,
            }
        }
    }

    /// Pairs the standard `TryFrom<Implementation>` conversion with the static
    /// description info needed for friendly diagnostics in `Benchmark::description`.
    pub(super) trait ImplementationMatcher:
        TryFrom<Implementation, Error = FailureScore> + 'static
    {
        /// Human-readable description of which implementation this marker handles.
        const DESCRIPTION: &'static str;
        /// The implementation variant this marker expects (for mismatch diagnostics).
        const EXPECTED: Implementation;
    }

    impl TryFrom<Implementation> for Optimized {
        type Error = FailureScore;
        fn try_from(i: Implementation) -> Result<Self, Self::Error> {
            match i {
                Implementation::Optimized => Ok(Self),
                _ => Err(FailureScore(1)),
            }
        }
    }

    impl ImplementationMatcher for Optimized {
        const DESCRIPTION: &'static str = "QueryComputer (architecture-dispatched)";
        const EXPECTED: Implementation = Implementation::Optimized;
    }

    impl TryFrom<Implementation> for Reference {
        type Error = FailureScore;
        fn try_from(i: Implementation) -> Result<Self, Self::Error> {
            match i {
                Implementation::Reference => Ok(Self),
                _ => Err(FailureScore(1)),
            }
        }
    }

    impl ImplementationMatcher for Reference {
        const DESCRIPTION: &'static str = "Chamfer / MaxSim fallback";
        const EXPECTED: Implementation = Implementation::Reference;
    }

    impl<I, T> Benchmark for Kernel<I, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
        I: ImplementationMatcher,
        Kernel<I, T>: RunBenchmark<I>,
        T: 'static,
    {
        type Input = MultiVectorOp;
        type Output = Vec<RunResult>;

        fn try_match(&self, from: &MultiVectorOp) -> Result<MatchScore, FailureScore> {
            let mut failscore: Option<u32> = None;
            if datatype::Type::<T>::try_match(&from.element_type).is_err() {
                *failscore.get_or_insert(0) += 10;
            }
            if let Err(FailureScore(score)) = I::try_from(from.implementation) {
                *failscore.get_or_insert(0) += 2 + score;
            }

            match failscore {
                None => Ok(MatchScore(0)),
                Some(score) => Err(FailureScore(score)),
            }
        }

        fn run(
            &self,
            input: &MultiVectorOp,
            _: diskann_benchmark_runner::Checkpoint<'_>,
            mut output: &mut dyn diskann_benchmark_runner::Output,
        ) -> anyhow::Result<Self::Output> {
            // The dispatcher only invokes `run` after `try_match` has already accepted
            // the input, so a failure here would indicate a dispatcher bug.
            I::try_from(input.implementation).expect("try_match accepted the input");
            writeln!(output, "{}", input)?;
            let results = self.run_benchmark(input)?;
            writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
            Ok(results)
        }

        fn description(
            &self,
            f: &mut std::fmt::Formatter<'_>,
            input: Option<&MultiVectorOp>,
        ) -> std::fmt::Result {
            match input {
                None => {
                    writeln!(
                        f,
                        "- Element Type: {}",
                        diskann_benchmark_runner::dispatcher::Description::<
                            datatype::DataType,
                            datatype::Type<T>,
                        >::new()
                    )?;
                    writeln!(f, "- Implementation: {}", I::DESCRIPTION)?;
                }
                Some(input) => {
                    if let Err(err) = datatype::Type::<T>::try_match_verbose(&input.element_type) {
                        writeln!(f, "\n    - Mismatched element type: {}", err)?;
                    }
                    if I::try_from(input.implementation).is_err() {
                        writeln!(
                            f,
                            "\n    - Mismatched implementation: expected {}, got {}",
                            I::EXPECTED,
                            input.implementation
                        )?;
                    }
                }
            }
            Ok(())
        }
    }

    impl<I, T> Regression for Kernel<I, T>
    where
        datatype::Type<T>: DispatchRule<datatype::DataType>,
        I: ImplementationMatcher,
        Kernel<I, T>: RunBenchmark<I>,
        T: 'static,
    {
        type Tolerances = MultiVectorTolerance;
        type Pass = CheckResult;
        type Fail = CheckResult;

        fn check(
            &self,
            tolerance: &MultiVectorTolerance,
            _input: &MultiVectorOp,
            before: &Vec<RunResult>,
            after: &Vec<RunResult>,
        ) -> anyhow::Result<PassFail<CheckResult, CheckResult>> {
            anyhow::ensure!(
                before.len() == after.len(),
                "before has {} runs but after has {}",
                before.len(),
                after.len(),
            );

            let mut passed = true;
            let checks: Vec<Comparison> = std::iter::zip(before.iter(), after.iter())
                .enumerate()
                .map(|(i, (b, a))| {
                    anyhow::ensure!(b.run == a.run, "run {i} mismatched");

                    let computations_per_latency = b.computations_per_latency() as f64;

                    let before_min =
                        b.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;
                    let after_min =
                        a.percentiles.minimum.as_f64() * 1000.0 / computations_per_latency;

                    let comparison = Comparison {
                        run: b.run.clone(),
                        tolerance: *tolerance,
                        before_min,
                        after_min,
                    };

                    match relative_change(before_min, after_min) {
                        Ok(change) => {
                            if change > tolerance.min_time_regression.get() {
                                passed = false;
                            }
                        }
                        Err(_) => passed = false,
                    };

                    Ok(comparison)
                })
                .collect::<anyhow::Result<Vec<Comparison>>>()?;

            let check = CheckResult { checks };

            if passed {
                Ok(PassFail::Pass(check))
            } else {
                Ok(PassFail::Fail(check))
            }
        }
    }

    //////////////////////
    // Regression Check //
    //////////////////////

    /// Per-run comparison result showing before/after percentile differences.
    #[derive(Debug, Serialize)]
    pub(super) struct Comparison {
        run: Run,
        tolerance: MultiVectorTolerance,
        before_min: f64,
        after_min: f64,
    }

    /// Aggregated result of the regression check across all runs.
    #[derive(Debug, Serialize)]
    pub(super) struct CheckResult {
        checks: Vec<Comparison>,
    }

    impl std::fmt::Display for CheckResult {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let header = [
                "Operation",
                "Q",
                "D",
                "Dim",
                "Min Before (ns/IP @ Dim)",
                "Min After (ns/IP @ Dim)",
                "Change (%)",
                "Remark",
            ];

            let mut table =
                diskann_benchmark_runner::utils::fmt::Table::new(header, self.checks.len());

            for (i, c) in self.checks.iter().enumerate() {
                let mut row = table.row(i);
                let change = relative_change(c.before_min, c.after_min);

                row.insert(c.run.operation, 0);
                row.insert(c.run.num_query_vectors, 1);
                row.insert(c.run.num_doc_vectors, 2);
                row.insert(c.run.dim, 3);
                row.insert(format!("{:.3}", c.before_min), 4);
                row.insert(format!("{:.3}", c.after_min), 5);
                match change {
                    Ok(change) => {
                        row.insert(format!("{:.3} %", change * 100.0), 6);
                        if change > c.tolerance.min_time_regression.get() {
                            row.insert("FAIL", 7);
                        }
                    }
                    Err(err) => {
                        row.insert("invalid", 6);
                        row.insert(err, 7);
                    }
                }
            }

            table.fmt(f)
        }
    }

    ///////////////
    // Benchmark //
    ///////////////

    pub(super) trait RunBenchmark<I> {
        fn run_benchmark(&self, input: &MultiVectorOp) -> Result<Vec<RunResult>, anyhow::Error>;
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub(super) struct RunResult {
        /// The configuration for this run.
        run: Run,
        /// Per-measurement latencies (over `loops_per_measurement` calls).
        latencies: Vec<MicroSeconds>,
        /// Latency percentiles.
        percentiles: percentiles::Percentiles<MicroSeconds>,
    }

    impl RunResult {
        fn computations_per_latency(&self) -> usize {
            self.run.num_query_vectors.get()
                * self.run.num_doc_vectors.get()
                * self.run.loops_per_measurement.get()
        }
    }

    impl std::fmt::Display for DisplayWrapper<'_, [RunResult]> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if self.is_empty() {
                return Ok(());
            }

            // ns/IP is normalized as `min_latency_us * 1000 / (Q * D * loops)` and is
            // approximately linear in `dim`. Compare across rows with the same `Dim`;
            // divide further by `Dim` to recover ns per scalar multiply.
            writeln!(
                f,
                "ns/IP = time per (query, doc) inner-product call (~ linear in Dim)"
            )?;

            let header = [
                "Operation",
                "Q",
                "D",
                "Dim",
                "Min Time (ns/IP @ Dim)",
                "Mean Time (ns/IP @ Dim)",
                "Loops",
                "Measurements",
            ];

            let mut table = diskann_benchmark_runner::utils::fmt::Table::new(header, self.len());

            self.iter().enumerate().for_each(|(row, r)| {
                let mut row = table.row(row);

                let min_latency = r
                    .latencies
                    .iter()
                    .min()
                    .copied()
                    .unwrap_or(MicroSeconds::new(u64::MAX));
                let mean_latency = r.percentiles.mean;

                let computations_per_latency = r.computations_per_latency() as f64;

                // Convert time from micro-seconds to nano-seconds per inner-product call
                // (one (query, doc) pair, ~ linear in dim).
                let min_time = min_latency.as_f64() / computations_per_latency * 1000.0;
                let mean_time = mean_latency / computations_per_latency * 1000.0;

                row.insert(r.run.operation, 0);
                row.insert(r.run.num_query_vectors, 1);
                row.insert(r.run.num_doc_vectors, 2);
                row.insert(r.run.dim, 3);
                row.insert(format!("{:.3}", min_time), 4);
                row.insert(format!("{:.3}", mean_time), 5);
                row.insert(r.run.loops_per_measurement, 6);
                row.insert(r.run.num_measurements, 7);
            });

            table.fmt(f)
        }
    }

    fn run_loops<F>(run: &Run, mut body: F) -> RunResult
    where
        F: FnMut(),
    {
        let mut latencies = Vec::with_capacity(run.num_measurements.get());

        for _ in 0..run.num_measurements.get() {
            let start = std::time::Instant::now();
            for _ in 0..run.loops_per_measurement.get() {
                body();
            }
            latencies.push(start.elapsed().into());
        }

        let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();
        RunResult {
            run: run.clone(),
            latencies,
            percentiles,
        }
    }

    ///////////////////
    // Data fixtures //
    ///////////////////

    const RNG_SEED: u64 = 0x12345;

    struct Data<T: Copy> {
        queries: Mat<Standard<T>>,
        docs: Mat<Standard<T>>,
    }

    impl<T: Copy> Data<T>
    where
        StandardUniform: Distribution<T>,
    {
        fn new(run: &Run) -> Self {
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let queries = Mat::new(
                Standard::new(run.num_query_vectors.get(), run.dim.get()).unwrap(),
                Init(|| StandardUniform.sample(&mut rng)),
            )
            .unwrap();
            let docs = Mat::new(
                Standard::new(run.num_doc_vectors.get(), run.dim.get()).unwrap(),
                Init(|| StandardUniform.sample(&mut rng)),
            )
            .unwrap();
            Self { queries, docs }
        }
    }

    //////////////////////
    // Distance kernels //
    //////////////////////

    /// Object-safe abstraction over a per-shape distance executor.
    ///
    /// The two implementations ([`OptimizedDistance`] and [`ReferenceDistance`]) share the
    /// same hot-loop nest in [`run_with_distance`]; dispatching through `&dyn Distance<T>`
    /// keeps `run_loops` from being monomorphised over the implementation axis.
    trait Distance<T: Copy> {
        fn chamfer(&self, doc: MatRef<'_, Standard<T>>) -> f32;
        fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]);
    }

    /// Distance executor that drives [`QueryComputer`] (architecture-dispatched SIMD).
    struct OptimizedDistance<T: Copy>(QueryComputer<T>);

    impl<T: Copy> Distance<T> for OptimizedDistance<T> {
        fn chamfer(&self, doc: MatRef<'_, Standard<T>>) -> f32 {
            self.0.chamfer(doc)
        }
        fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]) {
            self.0.max_sim(doc, scores);
        }
    }

    /// Distance executor that drives the [`Chamfer`] / [`MaxSim`] fallback path.
    struct ReferenceDistance<'a, T: Copy>(
        diskann_quantization::multi_vector::distance::QueryMatRef<'a, Standard<T>>,
    );

    impl<T: Copy> Distance<T> for ReferenceDistance<'_, T>
    where
        InnerProduct: for<'q, 'd> PureDistanceFunction<&'q [T], &'d [T], f32>,
    {
        fn chamfer(&self, doc: MatRef<'_, Standard<T>>) -> f32 {
            Chamfer::evaluate(self.0, doc)
        }
        fn max_sim(&self, doc: MatRef<'_, Standard<T>>, scores: &mut [f32]) {
            // `MaxSim::new` is a non-empty check + pointer wrap, so constructing it per
            // iteration is free — no need to hoist it out of the loop.
            let mut max_sim = MaxSim::new(scores).unwrap();
            let _ = max_sim.evaluate(self.0, doc);
        }
    }

    /////////////////////
    // Implementations //
    /////////////////////

    /// Shared loop nest. The trait-object dispatch happens once per outer iteration of
    /// `run_loops`; the work inside each `chamfer` / `max_sim` call is O(Q*D*dim), so the
    /// vtable hop is in the noise.
    fn run_with_distance<T: Copy>(
        run: &Run,
        doc: MatRef<'_, Standard<T>>,
        dist: &dyn Distance<T>,
    ) -> RunResult {
        match run.operation {
            Operation::Chamfer => run_loops(run, || {
                let v = dist.chamfer(doc);
                std::hint::black_box(v);
            }),
            Operation::MaxSim => {
                let mut scores = vec![0.0f32; run.num_query_vectors.get()];
                run_loops(run, || {
                    dist.max_sim(doc, &mut scores);
                    std::hint::black_box(&mut scores);
                })
            }
        }
    }

    fn run_optimized<T>(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>
    where
        T: Copy,
        StandardUniform: Distribution<T>,
        QueryComputer<T>: NewFromMatRef<T>,
        OptimizedDistance<T>: Distance<T>,
    {
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            let data = Data::<T>::new(run);
            // `QueryComputer` performs query-side precomputation that is intentionally
            // amortized across many `chamfer` / `max_sim` calls; construct it once per
            // shape, outside the timed loop.
            let dist = OptimizedDistance(<QueryComputer<T> as NewFromMatRef<T>>::new_from(
                data.queries.as_view(),
            ));
            results.push(run_with_distance(run, data.docs.as_view(), &dist));
        }
        Ok(results)
    }

    /// Drive the [`Chamfer`] / [`MaxSim`] fallback path.
    fn run_reference<T>(input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>>
    where
        T: Copy,
        StandardUniform: Distribution<T>,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
        for<'a> ReferenceDistance<'a, T>: Distance<T>,
    {
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            let data = Data::<T>::new(run);
            let dist = ReferenceDistance(data.queries.as_view().into());
            results.push(run_with_distance(run, data.docs.as_view(), &dist));
        }
        Ok(results)
    }

    /// Element-type-erasing constructor for [`QueryComputer`].
    ///
    /// `QueryComputer::<T>::new` is defined as an inherent method on the concrete
    /// `QueryComputer<f32>` / `QueryComputer<half::f16>` types (not a generic), so we need
    /// this shim trait to let generic code (e.g. `run_optimized<T>`) call it.
    trait NewFromMatRef<T: Copy> {
        fn new_from(query: MatRef<'_, Standard<T>>) -> QueryComputer<T>;
    }

    impl NewFromMatRef<f32> for QueryComputer<f32> {
        fn new_from(query: MatRef<'_, Standard<f32>>) -> QueryComputer<f32> {
            QueryComputer::<f32>::new(query)
        }
    }

    impl NewFromMatRef<f16> for QueryComputer<f16> {
        fn new_from(query: MatRef<'_, Standard<f16>>) -> QueryComputer<f16> {
            QueryComputer::<f16>::new(query)
        }
    }

    impl<T> RunBenchmark<Optimized> for Kernel<Optimized, T>
    where
        T: Copy + 'static,
        StandardUniform: Distribution<T>,
        QueryComputer<T>: NewFromMatRef<T>,
        OptimizedDistance<T>: Distance<T>,
    {
        fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
            run_optimized::<T>(input)
        }
    }

    impl<T> RunBenchmark<Reference> for Kernel<Reference, T>
    where
        T: Copy + 'static,
        StandardUniform: Distribution<T>,
        InnerProduct: for<'a, 'b> PureDistanceFunction<&'a [T], &'b [T], f32>,
        for<'a> ReferenceDistance<'a, T>: Distance<T>,
    {
        fn run_benchmark(&self, input: &MultiVectorOp) -> anyhow::Result<Vec<RunResult>> {
            run_reference::<T>(input)
        }
    }

    ///////////
    // Tests //
    ///////////

    #[cfg(test)]
    mod tests {
        use std::num::NonZeroUsize;

        use diskann_benchmark_runner::{
            benchmark::{PassFail, Regression},
            utils::{datatype::DataType, num::NonNegativeFinite, percentiles::compute_percentiles},
        };

        use super::*;

        fn tiny_run(operation: Operation) -> Run {
            Run {
                operation,
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
                implementation: Implementation::Optimized,
                runs: vec![tiny_run(Operation::Chamfer)],
            }
        }

        fn tiny_result(operation: Operation, minimum: u64) -> RunResult {
            let run = tiny_run(operation);
            let minimum = MicroSeconds::new(minimum);
            let mut latencies = vec![minimum];
            let percentiles = compute_percentiles(&mut latencies).unwrap();
            RunResult {
                run,
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
            let kernel = Kernel::<Optimized, f32>::new();

            let err = kernel
                .check(
                    &tolerance(0.0),
                    &tiny_op(),
                    &vec![tiny_result(Operation::Chamfer, 100)],
                    &vec![tiny_result(Operation::MaxSim, 100)],
                )
                .unwrap_err();

            assert_eq!(err.to_string(), "run 0 mismatched");
        }

        #[test]
        fn check_allows_negative_relative_change() {
            let kernel = Kernel::<Optimized, f32>::new();

            let result = kernel
                .check(
                    &tolerance(0.0),
                    &tiny_op(),
                    &vec![tiny_result(Operation::Chamfer, 100)],
                    &vec![tiny_result(Operation::Chamfer, 95)],
                )
                .unwrap();

            assert!(matches!(result, PassFail::Pass(_)));
        }

        #[test]
        fn check_passes_on_tolerance_boundary() {
            let kernel = Kernel::<Optimized, f32>::new();

            let result = kernel
                .check(
                    &tolerance(0.05),
                    &tiny_op(),
                    &vec![tiny_result(Operation::Chamfer, 100)],
                    &vec![tiny_result(Operation::Chamfer, 105)],
                )
                .unwrap();

            assert!(matches!(result, PassFail::Pass(_)));
        }

        #[test]
        fn check_fails_above_tolerance_boundary() {
            let kernel = Kernel::<Optimized, f32>::new();

            let result = kernel
                .check(
                    &tolerance(0.05),
                    &tiny_op(),
                    &vec![tiny_result(Operation::Chamfer, 100)],
                    &vec![tiny_result(Operation::Chamfer, 106)],
                )
                .unwrap();

            assert!(matches!(result, PassFail::Fail(_)));
        }

        #[test]
        fn check_result_display_includes_failure_details() {
            let check = CheckResult {
                checks: vec![Comparison {
                    run: tiny_run(Operation::Chamfer),
                    tolerance: tolerance(0.05),
                    before_min: 100.0,
                    after_min: 106.0,
                }],
            };

            let rendered = check.to_string();
            assert!(rendered.contains("Operation"), "rendered = {rendered}");
            assert!(rendered.contains("chamfer"), "rendered = {rendered}");
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
            let kernel = Kernel::<Optimized, f32>::new();

            let result = kernel
                .check(
                    &tolerance(0.05),
                    &tiny_op(),
                    &vec![tiny_result(Operation::Chamfer, 0)],
                    &vec![tiny_result(Operation::Chamfer, 0)],
                )
                .unwrap();

            assert!(matches!(result, PassFail::Fail(_)));
        }
    }
}
