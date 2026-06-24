/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A/B benchmark for **4-bit MinMax quantized** multi-vector MaxSim: the
//! experimental *staged integer* kernel (block-transposed `i16` query + `u8`
//! doc codes, `vpmaddwd` accumulation, metadata postprocess) vs the scalar
//! [`MinMaxKernel`] reference — at identical shapes and identical quantization.
//!
//! Both paths consume the *same* random f32 multi-vectors quantized to 4-bit
//! MinMax (Null transform, scale 1.0), so the comparison isolates the distance
//! kernel. The build / quantize cost is excluded from the timing.
//!
//! x86_64 (V3/AVX2) only — the quantized staged kernel has no other backend.

use std::io::Write;
use std::num::NonZeroUsize;

use diskann_benchmark_runner::{
    benchmark::{FailureScore, MatchScore},
    utils::{fmt::Table, percentiles, MicroSeconds},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_quantization::algorithms::transforms::NullTransform;
use diskann_quantization::algorithms::Transform;
use diskann_quantization::minmax::{MinMaxMeta, MinMaxQuantizer};
use diskann_quantization::multi_vector::distance::{QuantStagedDocs, QuantStagedQuery};
use diskann_quantization::multi_vector::{Defaulted, Mat, MatRef, MaxSim, QueryMatRef, Standard};
use diskann_quantization::num::Positive;
use diskann_quantization::CompressInto;
use diskann_utils::ReborrowMut;
use diskann_vector::DistanceFunctionMut;
use serde::{Deserialize, Serialize};

use super::driver::Data;
use crate::inputs::multi_vector::{MultiVectorQuantOp, Run};
use crate::utils::DisplayWrapper;

// ─────────────────────────────────────────────────────────────────────────
//  Kernel.
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub(super) struct QuantKernel;

impl QuantKernel {
    pub(super) const fn new() -> Self {
        Self
    }
}

impl Benchmark for QuantKernel {
    type Input = MultiVectorQuantOp;
    type Output = Vec<QuantRunResult>;

    fn try_match(&self, _from: &MultiVectorQuantOp) -> Result<MatchScore, FailureScore> {
        // The staged integer kernel requires AVX2 (V3).
        if QuantStagedQuery::is_supported() {
            Ok(MatchScore(0))
        } else {
            Err(FailureScore(0))
        }
    }

    fn run(
        &self,
        input: &MultiVectorQuantOp,
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
        input: Option<&MultiVectorQuantOp>,
    ) -> std::fmt::Result {
        match input {
            None => writeln!(f, "- 4-bit MinMax quantized staged MaxSim (V3/AVX2)")?,
            Some(_) => {
                if !QuantStagedQuery::is_supported() {
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

/// Quantize an f32 multi-vector to 4-bit MinMax (Null transform, scale 1.0) —
/// the same quantizer both paths share so the codes + metadata are identical.
fn quantize(input: MatRef<'_, Standard<f32>>) -> Mat<MinMaxMeta<4>> {
    let dim = input.vector_dim();
    let n = input.num_vectors();
    let q = MinMaxQuantizer::new(
        Transform::Null(NullTransform::new(NonZeroUsize::new(dim).unwrap())),
        Positive::new(1.0).unwrap(),
    );
    let mut out: Mat<MinMaxMeta<4>> = Mat::new(MinMaxMeta::new(n, dim), Defaulted).unwrap();
    q.compress_into(input, out.reborrow_mut()).unwrap();
    out
}

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

/// Build both kernels for one shape and time them (build / quantize excluded).
fn run_ab(run: &Run) -> anyhow::Result<QuantRunResult> {
    let data = Data::<f32>::new(run)?;

    // Path A — staged integer kernel (quantizes internally at build time).
    let mut query = QuantStagedQuery::build(data.queries.as_view())
        .ok_or_else(|| anyhow::anyhow!("AVX2 (V3) unavailable for the staged quantized kernel"))?;
    let docs = QuantStagedDocs::build(data.docs.as_view());

    // Path B — scalar MinMax reference over the same quantization.
    let q_ref = quantize(data.queries.as_view());
    let d_ref = quantize(data.docs.as_view());

    let nq = run.num_query_vectors.get();
    let mut scores = vec![0.0f32; nq];

    // Launder BOTH the inputs and the output through `black_box` each iteration.
    // Output-only `black_box` is not enough: the reference chain is `#[inline(always)]`
    // end-to-end with loop-invariant inputs, so the optimizer could hoist/elide it out
    // of the measured loop (the staged path is an opaque cross-crate call and cannot be),
    // making the A/B asymmetric. Laundering the inputs forces both paths to re-run the
    // full per-call work every iteration.
    let staged = measure(run, || {
        let docs = std::hint::black_box(&docs);
        query.compute_max_sim(docs, &mut scores);
        std::hint::black_box(&mut scores);
    });

    let reference = measure(run, || {
        let q_ref = std::hint::black_box(&q_ref);
        let d_ref = std::hint::black_box(&d_ref);
        let query_ref: QueryMatRef<_> = q_ref.as_view().into();
        MaxSim::new(&mut scores).evaluate(query_ref, d_ref.as_view());
        std::hint::black_box(&mut scores);
    });

    Ok(QuantRunResult {
        run: run.clone(),
        staged,
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

/// Staged-vs-reference result for one shape.
#[derive(Debug, Serialize, Deserialize)]
pub(super) struct QuantRunResult {
    pub(super) run: Run,
    pub(super) staged: Series,
    pub(super) reference: Series,
}

impl QuantRunResult {
    fn computations(&self) -> f64 {
        (self.run.num_query_vectors.get()
            * self.run.num_doc_vectors.get()
            * self.run.loops_per_measurement.get()) as f64
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [QuantRunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        writeln!(
            f,
            "ns/IP = min time per (query, doc) inner-product call; \
             Speedup = reference / staged (>1 ⇒ staged faster)"
        )?;

        let header = [
            "Q",
            "D",
            "Dim",
            "Staged (ns/IP)",
            "Reference (ns/IP)",
            "Speedup",
        ];
        let mut table = Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let comps = r.computations();
            let staged = r.staged.min_us() / comps * 1000.0;
            let reference = r.reference.min_us() / comps * 1000.0;
            let speedup = if staged > 0.0 {
                reference / staged
            } else {
                0.0
            };

            let mut row = table.row(row);
            row.insert(r.run.num_query_vectors, 0);
            row.insert(r.run.num_doc_vectors, 1);
            row.insert(r.run.dim, 2);
            row.insert(format!("{:.3}", staged), 3);
            row.insert(format!("{:.3}", reference), 4);
            row.insert(format!("{:.2}x", speedup), 5);
        });

        table.fmt(f)
    }
}

// ─────────────────────────────────────────────────────────────────────────
//  Registration.
// ─────────────────────────────────────────────────────────────────────────

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("multi-vector-quant-op", QuantKernel::new())?;
    Ok(())
}
