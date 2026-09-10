/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A/B benchmark for **f32** multi-vector MaxSim across three paths at identical
//! shapes over identical data:
//!
//! - **Paneled** — the paneled rebuild (views own their panel decomposition, one
//!   `Drain` seam), driven through `PaneledF32Query`.
//! - **Fused** — the production block-transposed V3 kernel via the public factory
//!   (`MaxSimIsa::X86_64_V3`).
//! - **Reference** — the `MaxSimIsa::Reference` baseline. Despite the name it is
//!   *not* scalar: it is a naive double loop over `(q, d)` pairs whose per-pair
//!   inner product is itself SIMD. What it lacks is fusion across queries,
//!   register-resident accumulators across the dim loop, and tile-level cache
//!   management.
//!
//! The coarse tiler is deliberately absent: it has no f32 instantiation (only 4-bit
//! MinMax and f16), so there is nothing to time.
//!
//! # Reading the numbers
//!
//! Not perfectly apples-to-apples. The paneled path pre-materializes its doc side
//! once in `build` (excluded from the timing), while the fused kernel is handed a
//! `MatRef` per call. Treat `Paneled/Fused` as a ceiling on the paneled structure's
//! win, not a pure abstraction delta.
//!
//! x86_64 (V3/AVX2) only.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::{fmt::Table, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::distance::{PaneledF32Docs, PaneledF32Query};
use diskann_quantization::multi_vector::{build_max_sim, BoxErase, MaxSimIsa};
use serde::{Deserialize, Serialize};

use super::driver::Data;
use crate::inputs::multi_vector::{MultiVectorPaneledF32Op, Run};
use crate::utils::DisplayWrapper;

// ─────────────────────────────────────────────────────────────────────────
//  Kernel.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct PaneledF32Kernel;

impl PaneledF32Kernel {
    pub(super) const fn new() -> Self {
        Self
    }
}

impl Benchmark for PaneledF32Kernel {
    type Input = MultiVectorPaneledF32Op;
    type Output = Vec<F32RunResult>;

    fn try_match(&self, _from: &MultiVectorPaneledF32Op, context: &MatchContext) -> Score {
        if PaneledF32Query::is_supported() {
            context.success(0)
        } else {
            context.fail(1, &"AVX2 (V3) unavailable on this CPU")
        }
    }

    fn run(
        &self,
        input: &MultiVectorPaneledF32Op,
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
        writeln!(f, "- f32 MaxSim, paneled / fused V3 / reference (V3/AVX2)")
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

/// Build all three paths for one shape and time them (build cost excluded).
fn run_ab(run: &Run) -> anyhow::Result<F32RunResult> {
    let data = Data::<f32>::new(run)?;

    // Path A — the paneled rebuild.
    let mut paneled_query = PaneledF32Query::build(data.queries.as_view())
        .ok_or_else(|| anyhow::anyhow!("AVX2 (V3) unavailable for the paneled f32 kernel"))?;
    let paneled_docs = PaneledF32Docs::build(data.docs.as_view());

    // Path B / C — the production factory kernels over the same query matrix.
    let fused_kernel =
        build_max_sim::<f32, _>(MaxSimIsa::X86_64_V3, data.queries.as_view(), BoxErase)?;
    let ref_kernel =
        build_max_sim::<f32, _>(MaxSimIsa::Reference, data.queries.as_view(), BoxErase)?;

    let nq = run.num_query_vectors.get();
    let mut scores = vec![0.0f32; nq];
    let doc_view = data.docs.as_view();

    // Launder inputs *and* output through `black_box` each iteration, matching the
    // quantized A/B: the factory kernels are opaque cross-crate calls, but the
    // paneled path is in-crate and could otherwise be hoisted out of the loop.
    let paneled = measure(run, || {
        let docs = std::hint::black_box(&paneled_docs);
        paneled_query.compute_max_sim(docs, &mut scores);
        std::hint::black_box(&mut scores);
    });

    // Timed adjacent to `paneled` so the ratio survives cross-run clock variance.
    let fused = measure(run, || {
        let doc_view = std::hint::black_box(doc_view);
        fused_kernel
            .compute_max_sim(doc_view, &mut scores)
            .expect("scores.len() == kernel.nrows() by construction");
        std::hint::black_box(&mut scores);
    });

    let reference = measure(run, || {
        let doc_view = std::hint::black_box(doc_view);
        ref_kernel
            .compute_max_sim(doc_view, &mut scores)
            .expect("scores.len() == kernel.nrows() by construction");
        std::hint::black_box(&mut scores);
    });

    Ok(F32RunResult {
        run: run.clone(),
        paneled,
        fused,
        reference,
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

/// Paneled-vs-fused-vs-reference result for one shape.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct F32RunResult {
    pub(super) run: Run,
    pub(super) paneled: Series,
    pub(super) fused: Series,
    pub(super) reference: Series,
}

impl F32RunResult {
    fn computations(&self) -> f64 {
        (self.run.num_query_vectors.get()
            * self.run.num_doc_vectors.get()
            * self.run.loops_per_measurement.get()) as f64
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [F32RunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            "ns/IP = min time per (query, doc) inner-product call. \
             Panel/Fused < 1 ⇒ paneled faster than the production V3 kernel. \
             Speedup = reference ÷ paneled."
        )?;

        let header = [
            "Q",
            "D",
            "Dim",
            "Paneled",
            "Fused V3",
            "Panel/Fused",
            "Reference",
            "Ref/Panel",
        ];
        let mut table = Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let comps = r.computations();
            let paneled = r.paneled.min_us() / comps * 1000.0;
            let fused = r.fused.min_us() / comps * 1000.0;
            let reference = r.reference.min_us() / comps * 1000.0;
            let vs_fused = if fused > 0.0 { paneled / fused } else { 0.0 };
            let speedup = if paneled > 0.0 {
                reference / paneled
            } else {
                0.0
            };

            let mut row = table.row(row);
            row.insert(r.run.num_query_vectors, 0);
            row.insert(r.run.num_doc_vectors, 1);
            row.insert(r.run.dim, 2);
            row.insert(format!("{:.3}", paneled), 3);
            row.insert(format!("{:.3}", fused), 4);
            row.insert(format!("{:.2}x", vs_fused), 5);
            row.insert(format!("{:.3}", reference), 6);
            row.insert(format!("{:.2}x", speedup), 7);
        });

        table.fmt(f)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("multi-vector-paneled-f32-op", PaneledF32Kernel::new())?;
    Ok(())
}
