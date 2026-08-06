/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A/B benchmark for **f16** multi-vector MaxSim: the coarse tiler's f16 path
//! (per-tile f16→f32 widen into a reused buffer, then an f32 store kernel + identity
//! postprocess reuse the tiled pipeline) vs the production `f16.rs` **preprocess**
//! path (per-tile f16→f32 `ConvertTo` + the fused f32 micro-kernel), reached via
//! [`build_max_sim`].
//!
//! Both convert per tile now, so the ratio mostly isolates one structural
//! difference: the tiler is *strip-based* (the kernel stores an A-major strip, then a
//! separate reduce pass), while the reference is *fused* (the kernel maxes straight
//! into state, no strip). Expect near-parity.
//!
//! x86_64 (V3/AVX2) only.

use std::io::Write;

use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    utils::{fmt::Table, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::multi_vector::distance::{QuantTiledF16Docs, QuantTiledF16Query};
use diskann_quantization::multi_vector::{
    build_max_sim, BoxErase, Mat, MatRef, MaxSimIsa, Standard,
};
use serde::{Deserialize, Serialize};

use super::driver::Data;
use crate::inputs::multi_vector::{MultiVectorTiledF16Op, Run};
use crate::utils::DisplayWrapper;

// ─────────────────────────────────────────────────────────────────────────
//  Kernel.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct TiledF16Kernel;

impl TiledF16Kernel {
    pub(super) const fn new() -> Self {
        Self
    }
}

impl Benchmark for TiledF16Kernel {
    type Input = MultiVectorTiledF16Op;
    type Output = Vec<TiledF16RunResult>;

    fn try_match(&self, _from: &MultiVectorTiledF16Op) -> Result<MatchScore, FailureScore> {
        if QuantTiledF16Query::is_supported() {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(0))
        }
    }

    fn run(
        &self,
        input: &MultiVectorTiledF16Op,
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

    fn description(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        input: Option<&MultiVectorTiledF16Op>,
    ) -> std::fmt::Result {
        match input {
            None => writeln!(f, "- f16 tiler vs f16.rs preprocess (V3/AVX2)")?,
            Some(_) => {
                if !QuantTiledF16Query::is_supported() {
                    writeln!(f, "\n    - AVX2 (V3) unavailable on this CPU")?;
                }
            }
        }
        Ok(())
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

/// Narrow an f32 fixture to f16 (row-major, same shape) so both f16 paths see
/// identical bits.
fn narrow(src: MatRef<'_, Standard<f32>>) -> Mat<Standard<half::f16>> {
    let (n, dim) = (src.num_vectors(), src.vector_dim());
    let s = src.as_slice();
    let mut i = 0;
    Mat::from_fn(Standard::new(n, dim).expect("n×dim"), || {
        let v = diskann_wide::cast_f32_to_f16(s[i]);
        i += 1;
        v
    })
}

/// Build both f16 kernels for one shape and time them (build / convert excluded).
fn run_ab(run: &Run) -> anyhow::Result<TiledF16RunResult> {
    // f16 fixtures, generated as f32 then narrowed so both paths see identical bits.
    let data = Data::<f32>::new(run)?;
    let q_f16 = narrow(data.queries.as_view());
    let d_f16 = narrow(data.docs.as_view());

    // Path A — the coarse tiler's f16 path (per-tile f16→f32 into a reused buffer).
    let mut tiled_query = QuantTiledF16Query::build(q_f16.as_view())
        .ok_or_else(|| anyhow::anyhow!("AVX2 (V3) unavailable for the tiled f16 kernel"))?;
    let tiled_docs = QuantTiledF16Docs::build(d_f16.as_view());

    // Path B — the production preprocess path (per-tile f16→f32 + fused f32 kernel).
    let preprocess =
        build_max_sim::<half::f16, _>(MaxSimIsa::X86_64_V3, q_f16.as_view(), BoxErase)?;

    let nq = run.num_query_vectors.get();
    let mut scores = vec![0.0f32; nq];

    let tiled = measure(run, || {
        let docs = std::hint::black_box(&tiled_docs);
        tiled_query.compute_max_sim(docs, &mut scores);
        std::hint::black_box(&mut scores);
    });

    let preprocess = measure(run, || {
        let doc = std::hint::black_box(d_f16.as_view());
        preprocess
            .compute_max_sim(doc, &mut scores)
            .expect("scores.len() == nrows by construction");
        std::hint::black_box(&mut scores);
    });

    Ok(TiledF16RunResult {
        run: run.clone(),
        tiled,
        preprocess,
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
    fn min_us(&self) -> f64 {
        self.latencies
            .iter()
            .min()
            .copied()
            .unwrap_or(MicroSeconds::new(u64::MAX))
            .as_f64()
    }
}

/// Tiled-vs-preprocess result for one shape.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct TiledF16RunResult {
    pub(super) run: Run,
    pub(super) tiled: Series,
    pub(super) preprocess: Series,
}

impl TiledF16RunResult {
    fn computations(&self) -> f64 {
        (self.run.num_query_vectors.get()
            * self.run.num_doc_vectors.get()
            * self.run.loops_per_measurement.get()) as f64
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [TiledF16RunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            "ns/IP = min time per (query, doc) inner-product call; \
             Tiled/Preprocess = tiled ÷ preprocess (>1 ⇒ tiler slower). Both convert \
             f16→f32 per tile; the residual is strip-based (tiler) vs fused (preprocess)."
        )?;

        let header = [
            "Q",
            "D",
            "Dim",
            "Tiled (ns/IP)",
            "Preprocess (ns/IP)",
            "Tiled/Preprocess",
        ];
        let mut table = Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let comps = r.computations();
            let tiled = r.tiled.min_us() / comps * 1000.0;
            let preprocess = r.preprocess.min_us() / comps * 1000.0;
            let ratio = if preprocess > 0.0 {
                tiled / preprocess
            } else {
                0.0
            };

            let mut row = table.row(row);
            row.insert(r.run.num_query_vectors, 0);
            row.insert(r.run.num_doc_vectors, 1);
            row.insert(r.run.dim, 2);
            row.insert(format!("{:.3}", tiled), 3);
            row.insert(format!("{:.3}", preprocess), 4);
            row.insert(format!("{:.2}x", ratio), 5);
        });

        table.fmt(f)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("multi-vector-tiled-f16-op", TiledF16Kernel::new())?;
    Ok(())
}
