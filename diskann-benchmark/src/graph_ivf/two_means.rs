/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, io::Write};

use diskann::ivf::workingset::PointRef;
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::percentiles::{self, Percentiles},
    Benchmark, Checkpoint, Output,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::Serialize;

use crate::inputs::graph_ivf_two_means::{FitOrder, GraphIvfTwoMeans, Layout};

mod diagnostics;
mod measurement;
pub(super) mod replay;
use measurement::{Strategy, BASELINE};

pub(super) struct Strategies;

impl Benchmark for Strategies {
    type Input = GraphIvfTwoMeans;
    type Output = Results;
    fn try_match(&self, _: &Self::Input, context: &MatchContext) -> Score {
        context.success(0)
    }
    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Complete two-means: paired baseline, transposed L2/dot, controlled Lloyd reference"
        )
    }
    fn run(
        &self,
        input: &Self::Input,
        _: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Results> {
        let corpus = input
            .data
            .as_ref()
            .map(|file| super::data::load_vectors(file, input.data_type))
            .transpose()?;
        if let Some(data) = &corpus {
            super::data::validate_vectors(data.as_view(), "two-means corpus")?;
            anyhow::ensure!(
                input.dimensions.len() == 1 && input.dimensions[0].get() == data.ncols(),
                "file dimension must exactly match the requested dimension"
            );
            anyhow::ensure!(
                input
                    .bucket_sizes
                    .iter()
                    .all(|size| size.get() <= data.nrows()),
                "bucket is larger than corpus"
            );
        }
        let mut workloads = Vec::new();
        for &layout in &input.layouts {
            for size in &input.bucket_sizes {
                for dim in &input.dimensions {
                    let data =
                        Data::new(layout, size.get(), dim.get(), input.seed, corpus.as_ref())?;
                    let points = data.points();
                    for iterations in &input.iterations {
                        writeln!(
                            output,
                            "two-means: {}, {} x {}, {} Lloyd steps",
                            layout.name(),
                            size,
                            dim,
                            iterations
                        )?;
                        workloads.push(run_workload(
                            input,
                            layout,
                            &points,
                            dim.get(),
                            iterations.get(),
                            output,
                        )?);
                    }
                }
            }
        }
        Ok(Results {
            schema_version: 2,
            threads: 1,
            timing_scope: "validate + scratch + pack/norms + seed + distance/selection + ordered f64 sums + center updates + scratch/packing release; diagnostics run after all timed rounds",
            numerical_reference: "paired_point",
            source: if corpus.is_some() { "file_sample" } else { "synthetic_uniform" },
            workloads,
        })
    }
}

impl Layout {
    const fn name(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Scattered => "scattered",
        }
    }
}

#[derive(Serialize)]
pub(super) struct Results {
    schema_version: u32,
    threads: usize,
    timing_scope: &'static str,
    numerical_reference: &'static str,
    source: &'static str,
    workloads: Vec<WorkloadResults>,
}
#[derive(Serialize)]
struct WorkloadResults {
    workload: Workload,
    invocation_order: Vec<Invocation>,
    strategies: Vec<StrategyResult>,
}
#[derive(Serialize)]
struct Invocation {
    round: usize,
    position: usize,
    warmup: bool,
    strategy: &'static str,
}
#[derive(Serialize)]
struct Workload {
    layout: &'static str,
    bucket_size: usize,
    dimensions: usize,
    iterations: usize,
}
#[derive(Serialize)]
struct StrategyResult {
    strategy: &'static str,
    status: &'static str,
    samples: usize,
    total_ns: Option<Percentiles<u64>>,
    preparation_ns: Option<Percentiles<u64>>,
    /// Preserve measurement order for paired analysis. No slow samples are discarded.
    total_samples_ns: Vec<u64>,
    packed_bytes: Option<usize>,
    scratch_bytes: Option<usize>,
    diagnostics: Option<diagnostics::Comparison>,
    failures: Vec<String>,
}

struct Samples {
    totals: Vec<u64>,
    preparations: Vec<u64>,
    last: Option<measurement::Run>,
    failures: Vec<String>,
}

fn run_workload(
    input: &GraphIvfTwoMeans,
    layout: Layout,
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    mut output: &mut dyn Output,
) -> anyhow::Result<WorkloadResults> {
    let mut measurements: Vec<_> = input
        .strategies
        .iter()
        .map(|_| Samples {
            totals: Vec::new(),
            preparations: Vec::new(),
            last: None,
            failures: Vec::new(),
        })
        .collect();
    let mut invocation_order = Vec::new();
    // Rotate and reverse strategy order between complete rounds to reduce ordering bias.
    for round in 0..input.warmup_repetitions + input.repetitions.get() {
        let order = if input.order == FitOrder::Abba {
            vec![0, 1, 1, 0]
        } else {
            let mut order: Vec<_> = (0..input.strategies.len()).collect();
            order.rotate_left(round % input.strategies.len());
            if round % 2 != 0 {
                order.reverse();
            }
            order
        };
        for (position, index) in order.into_iter().enumerate() {
            invocation_order.push(Invocation {
                round,
                position,
                warmup: round < input.warmup_repetitions,
                strategy: input.strategies[index].name(),
            });
            let run = measurement::fit(
                black_box(points),
                dim,
                iterations,
                input.seed,
                input.strategies[index],
            );
            let samples = &mut measurements[index];
            let run = match run {
                Ok(run) => run,
                Err(error) if input.strategies[index] == Strategy::TransposedLloydReference => {
                    samples
                        .failures
                        .push(format!("round {round}, position {position}: {error:#}"));
                    continue;
                }
                Err(error) => return Err(error),
            };
            if round >= input.warmup_repetitions {
                samples.totals.push(duration_ns(run.total));
                samples.preparations.push(duration_ns(run.preparation));
                samples.last = Some(black_box(run));
            } else {
                black_box(run);
            }
        }
    }
    // Diagnostics use independent fits and identical seeding. They never interleave
    // with measured fits; both same-center and diverging-trajectory views are retained.
    let reference = diagnostics::trace(points, dim, iterations, input.seed, BASELINE)?;
    let mut strategies = Vec::new();
    for (strategy, mut samples) in input.strategies.iter().copied().zip(measurements) {
        let comparison = (|| -> anyhow::Result<_> {
            let trace = diagnostics::trace(points, dim, iterations, input.seed, strategy)?;
            if let Some(run) = &samples.last {
                anyhow::ensure!(
                    run.centers
                        .iter()
                        .zip(&trace.centers)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "diagnostic fit differs from timed fit"
                );
            }
            diagnostics::compare(points, dim, strategy, &reference, &trace)
        })();
        let comparison = match comparison {
            Ok(comparison) => Some(comparison),
            Err(error) if strategy == Strategy::TransposedLloydReference => {
                samples.failures.push(format!("diagnostic: {error:#}"));
                None
            }
            Err(error) => return Err(error),
        };
        let total_samples_ns = samples.totals.clone();
        let total_ns = if samples.totals.is_empty() {
            None
        } else {
            Some(percentiles::compute_percentiles(&mut samples.totals)?)
        };
        writeln!(
            output,
            "  {}: mean {:?} ns; failures {}",
            strategy.name(),
            total_ns.as_ref().map(|t| t.mean),
            samples.failures.len()
        )?;
        strategies.push(StrategyResult {
            strategy: strategy.name(),
            status: if samples.failures.is_empty() {
                "complete"
            } else {
                "failed_or_not_applicable"
            },
            samples: samples.totals.len(),
            total_ns,
            preparation_ns: if samples.preparations.is_empty() {
                None
            } else {
                Some(percentiles::compute_percentiles(&mut samples.preparations)?)
            },
            total_samples_ns,
            packed_bytes: samples.last.as_ref().map(|r| r.packed_bytes),
            scratch_bytes: samples.last.as_ref().map(|r| r.scratch_bytes),
            diagnostics: comparison,
            failures: samples.failures,
        });
    }
    Ok(WorkloadResults {
        invocation_order,
        workload: Workload {
            layout: layout.name(),
            bucket_size: points.len(),
            dimensions: dim,
            iterations,
        },
        strategies,
    })
}
fn duration_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

enum Data {
    Contiguous { values: Vec<f32>, dim: usize },
    Scattered { rows: Vec<Vec<f32>> },
}
impl Data {
    fn new(
        layout: Layout,
        size: usize,
        dim: usize,
        seed: u64,
        corpus: Option<&rowmajor::Owned<f32>>,
    ) -> anyhow::Result<Self> {
        let len = size
            .checked_mul(dim)
            .ok_or_else(|| anyhow::anyhow!("workload dimensions overflow"))?;
        let mut rng = StdRng::seed_from_u64(seed ^ ((size as u64) << 32) ^ dim as u64);
        let values: Vec<f32> = if let Some(data) = corpus {
            let selected = rand::seq::index::sample(&mut rng, data.nrows(), size);
            selected
                .into_iter()
                .flat_map(|i| data.row(i).iter().copied())
                .collect()
        } else {
            (0..len).map(|_| rng.random_range(-1.0..1.0)).collect()
        };
        Ok(Self::from_values(layout, size, dim, seed, values))
    }
    fn from_values(layout: Layout, size: usize, dim: usize, seed: u64, values: Vec<f32>) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ ((size as u64) << 32) ^ dim as u64);
        match layout {
            Layout::Contiguous => Self::Contiguous { values, dim },
            Layout::Scattered => {
                let mut order: Vec<_> = (0..size).collect();
                order.shuffle(&mut rng);
                let mut rows = vec![Vec::new(); size];
                // Allocate in a shuffled order, then restore logical point order via references.
                // Randomizing allocation does not change values, point order, seed, or f64 sums.
                for &logical in &order {
                    rows[logical] = values[logical * dim..(logical + 1) * dim].to_vec();
                }
                Self::Scattered { rows }
            }
        }
    }
    fn points(&self) -> Vec<PointRef<'_, usize>> {
        match self {
            Self::Contiguous { values, dim } => values
                .chunks_exact(*dim)
                .enumerate()
                .map(|(id, vector)| PointRef { id, vector })
                .collect(),
            Self::Scattered { rows } => rows
                .iter()
                .enumerate()
                .map(|(id, vector)| PointRef { id, vector })
                .collect(),
        }
    }
}
