/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A/B benchmark holding the **paneled structure fixed and varying the element
//! precision**, so the f32-vs-MinMax ratio is measured in one process:
//!
//! - **f32** — `PaneledF32Query` / `PaneledF32Docs`, exact f32 MaxSim.
//! - **MinMax** — `PaneledQuantQuery` / `PaneledQuantDocs`, MaxSim over 4-bit
//!   MinMax codes with the dequant fused into the `Drain`.
//! - **Fused V3** — the production block-transposed f32 kernel, as a calibration
//!   anchor: it appears in `multi-vector-paneled-f32.json` too, so its column
//!   cross-checks this job's numbers against that one, and it answers whether
//!   either paneled path beats production.
//!
//! Both paneled paths are built from the *same* random f32 matrices; the MinMax
//! path quantizes internally at build, which is excluded from the timing.
//!
//! # Reading the numbers
//!
//! **The two paneled paths do not compute the same thing.** f32 is exact; MinMax
//! is a 4-bit approximation. `f32/MinMax` is therefore the *cost of precision* at
//! a fixed kernel structure, not a like-for-like kernel delta — a value near
//! `1.00x` means dropping to 4 bits bought no speed, not that the two kernels are
//! equally good at the same task.
//!
//! Why this job exists: running the f32 and quantized benchmarks separately puts
//! the two numbers in different processes, and this box shifts clock state
//! mid-run (levels ~1.3x apart), which swamps the ratio. Timing them adjacently
//! within one shape protects it, exactly as the existing jobs do for their own
//! internal ratios.
//!
//! x86_64 (V3/AVX2) only.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::{fmt::Table, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::distance::{
    PaneledF32Docs, PaneledF32Query, PaneledQuantDocs, PaneledQuantQuery,
};
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimIsa};
use serde::{Deserialize, Serialize};

use super::driver::Data;
use crate::inputs::multi_vector::{MultiVectorPaneledPrecisionOp, Run};
use crate::utils::DisplayWrapper;

// ─────────────────────────────────────────────────────────────────────────
//  Kernel.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct PaneledPrecisionKernel;

impl PaneledPrecisionKernel {
    pub(super) const fn new() -> Self {
        Self
    }
}

impl Benchmark for PaneledPrecisionKernel {
    type Input = MultiVectorPaneledPrecisionOp;
    type Output = Vec<PrecisionRunResult>;

    fn try_match(&self, _from: &MultiVectorPaneledPrecisionOp, context: &MatchContext) -> Score {
        if PaneledF32Query::is_supported() && PaneledQuantQuery::is_supported() {
            context.success(0)
        } else {
            context.fail(1, &"AVX2 (V3) unavailable on this CPU")
        }
    }

    fn run(
        &self,
        input: &MultiVectorPaneledPrecisionOp,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;
        let mut results = Vec::with_capacity(input.runs.len());
        for run in input.runs.iter() {
            results.push(run_ab(run)?);
        }
        writeln!(output, "\n\n{}", DisplayWrapper(&*results))?;
        Ok(results)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "- Paneled MaxSim at f32 vs 4-bit MinMax, plus fused V3 anchor (V3/AVX2)"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  A/B timing.
// ─────────────────────────────────────────────────────────────────────────

/// Run `f` `loops_per_measurement` times per measurement, `num_measurements`
/// times, returning the per-measurement latencies and their percentiles.
fn measure(run: &Run, mut f: impl FnMut()) -> Series {
    let mut latencies = Vec::with_capacity(run.num_measurements.get());
    for _ in 0..run.num_measurements.get() {
        let start = std::time::Instant::now();
        for _ in 0..run.loops_per_measurement.get() {
            f();
        }
        latencies.push(start.elapsed().into());
    }
    let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();
    Series {
        latencies,
        percentiles,
    }
}

/// Build all three paths for one shape and time them (build / quantize excluded).
fn run_ab(run: &Run) -> anyhow::Result<PrecisionRunResult> {
    let data = Data::<f32>::new(run)?;

    // Path A — paneled at f32.
    let mut f32_query = PaneledF32Query::build(data.queries.as_view())
        .ok_or_else(|| anyhow::anyhow!("AVX2 (V3) unavailable for the paneled f32 kernel"))?;
    let f32_docs = PaneledF32Docs::build(data.docs.as_view());

    // Path B — paneled at 4-bit MinMax over the same source matrices (each side
    // quantizes internally at build).
    let mut minmax_query = PaneledQuantQuery::build(data.queries.as_view())
        .ok_or_else(|| anyhow::anyhow!("AVX2 (V3) unavailable for the paneled quantized kernel"))?;
    let minmax_docs = PaneledQuantDocs::build(data.docs.as_view());

    // Path C — the production fused V3 f32 kernel, as a calibration anchor.
    let fused_kernel =
        build_max_sim::<f32, _>(MaxSimIsa::X86_64_V3, data.queries.as_view(), BoxErase)?;

    let nq = run.num_query_vectors.get();
    let mut scores = vec![0.0f32; nq];
    let doc_view = data.docs.as_view();

    // Launder inputs *and* output through `black_box` each iteration: the paneled
    // paths are in-crate with loop-invariant inputs and could otherwise be hoisted
    // out of the measured loop, while the factory kernel is an opaque cross-crate
    // call that cannot be — which would make the comparison asymmetric.
    let paneled_f32 = measure(run, || {
        let docs = std::hint::black_box(&f32_docs);
        f32_query.compute_max_sim(docs, &mut scores);
        std::hint::black_box(&mut scores);
    });

    // Timed adjacent to `paneled_f32` — this adjacency is the whole point of the
    // job, and is what the separate f32 and quantized jobs cannot give.
    let paneled_minmax = measure(run, || {
        let docs = std::hint::black_box(&minmax_docs);
        minmax_query.compute_max_sim(docs, &mut scores);
        std::hint::black_box(&mut scores);
    });

    let fused = measure(run, || {
        let doc_view = std::hint::black_box(doc_view);
        fused_kernel
            .compute_max_sim(doc_view, &mut scores)
            .expect("scores.len() == kernel.nrows() by construction");
        std::hint::black_box(&mut scores);
    });

    Ok(PrecisionRunResult {
        run: run.clone(),
        paneled_f32,
        paneled_minmax,
        fused,
    })
}

// ─────────────────────────────────────────────────────────────────────────
//  Result types.
// ─────────────────────────────────────────────────────────────────────────

/// One timed series (per-measurement latencies + percentiles).
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct Series {
    latencies: Vec<MicroSeconds>,
    percentiles: percentiles::Percentiles<MicroSeconds>,
}

impl Series {
    /// Minimum latency, in microseconds.
    fn min_us(&self) -> f64 {
        self.latencies
            .iter()
            .min()
            .copied()
            .unwrap_or(MicroSeconds::new(u64::MAX))
            .as_f64()
    }
}

/// f32-vs-MinMax-vs-fused result for one shape.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct PrecisionRunResult {
    pub(super) run: Run,
    pub(super) paneled_f32: Series,
    pub(super) paneled_minmax: Series,
    pub(super) fused: Series,
}

impl PrecisionRunResult {
    fn computations(&self) -> f64 {
        (self.run.num_query_vectors.get()
            * self.run.num_doc_vectors.get()
            * self.run.loops_per_measurement.get()) as f64
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [PrecisionRunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            "ns/IP = min time per (query, doc) inner-product call. \
             f32/MinMax > 1 ⇒ dropping to 4-bit codes is faster — it is the cost of \
             precision at a fixed kernel structure, NOT a like-for-like kernel delta \
             (the f32 path is exact, the MinMax path is a 4-bit approximation). \
             Fused V3 is an f32 calibration anchor shared with the paneled-f32 job."
        )?;

        let header = [
            "Q",
            "D",
            "Dim",
            "Paneled f32",
            "Paneled MinMax",
            "f32/MinMax",
            "Fused V3",
        ];
        let mut table = Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let comps = r.computations();
            let as_f32 = r.paneled_f32.min_us() / comps * 1000.0;
            let minmax = r.paneled_minmax.min_us() / comps * 1000.0;
            let fused = r.fused.min_us() / comps * 1000.0;
            let ratio = if minmax > 0.0 { as_f32 / minmax } else { 0.0 };

            let mut row = table.row(row);
            row.insert(r.run.num_query_vectors, 0);
            row.insert(r.run.num_doc_vectors, 1);
            row.insert(r.run.dim, 2);
            row.insert(format!("{:.3}", as_f32), 3);
            row.insert(format!("{:.3}", minmax), 4);
            row.insert(format!("{:.2}x", ratio), 5);
            row.insert(format!("{:.3}", fused), 6);
        });

        table.fmt(f)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register(
        "multi-vector-paneled-precision-op",
        PaneledPrecisionKernel::new(),
    )?;
    Ok(())
}
