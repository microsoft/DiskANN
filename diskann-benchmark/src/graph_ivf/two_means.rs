/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, io::Write};

use diskann::ivf::{
    two_means_benchmark::{self, Strategy},
    workingset::PointRef,
};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::percentiles::{self, Percentiles},
    Benchmark, Checkpoint, Output,
};
use diskann_utils::views::rowmajor::{self, Matrix};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::Serialize;

use crate::inputs::graph_ivf_two_means::{GraphIvfTwoMeans, Layout};

pub(super) struct Strategies;

impl Benchmark for Strategies {
    type Input = GraphIvfTwoMeans;
    type Output = Results;
    fn try_match(&self, _: &Self::Input, context: &MatchContext) -> Score {
        context.success(0)
    }
    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Complete two-means: existing L2, paired, packed paired, batched 4/8, SGEMM, center difference")
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
            threads: 1,
            timing_scope: "scratch + pack + seed + distance/selection + ordered f64 sums + center updates; diagnostics excluded",
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
    threads: usize,
    timing_scope: &'static str,
    numerical_reference: &'static str,
    source: &'static str,
    workloads: Vec<WorkloadResults>,
}
#[derive(Serialize)]
struct WorkloadResults {
    workload: Workload,
    strategies: Vec<StrategyResult>,
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
    samples: usize,
    total_ns: Percentiles<u64>,
    packing_ns: Percentiles<u64>,
    /// Preserve measurement order for paired analysis. No slow samples are discarded.
    total_samples_ns: Vec<u64>,
    packed_bytes: usize,
    assignment_disagreements: usize,
    center_bit_disagreements: usize,
    max_center_absolute_error: f64,
    min_reference_margin: Option<f64>,
}

struct Samples {
    totals: Vec<u64>,
    packings: Vec<u64>,
    last: Option<two_means_benchmark::Run>,
}

fn run_workload(
    input: &GraphIvfTwoMeans,
    layout: Layout,
    points: &[PointRef<'_, usize>],
    dim: usize,
    iterations: usize,
    mut output: &mut dyn Output,
) -> anyhow::Result<WorkloadResults> {
    let reference =
        two_means_benchmark::fit(points, dim, iterations, input.seed, Strategy::PairedPoint)?;
    let min_margin = reference
        .margins
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .reduce(f64::min);
    let mut measurements: Vec<_> = Strategy::ALL
        .iter()
        .map(|_| Samples {
            totals: Vec::new(),
            packings: Vec::new(),
            last: None,
        })
        .collect();
    // Rotate and reverse strategy order between complete rounds to reduce ordering bias.
    for round in 0..input.warmup_repetitions + input.repetitions.get() {
        let mut order: Vec<_> = (0..Strategy::ALL.len()).collect();
        order.rotate_left(round % Strategy::ALL.len());
        if round % 2 != 0 {
            order.reverse();
        }
        for index in order {
            let run = two_means_benchmark::fit(
                black_box(points),
                dim,
                iterations,
                input.seed,
                Strategy::ALL[index],
            )?;
            let samples = &mut measurements[index];
            if round >= input.warmup_repetitions {
                samples.totals.push(duration_ns(run.total));
                samples.packings.push(duration_ns(run.packing));
                samples.last = Some(black_box(run));
            } else {
                black_box(run);
            }
        }
    }
    let mut strategies = Vec::new();
    for (strategy, mut samples) in Strategy::ALL.into_iter().zip(measurements) {
        let run = samples
            .last
            .ok_or_else(|| anyhow::anyhow!("missing measured run"))?;
        let assignment_disagreements = run
            .assignments
            .iter()
            .zip(&reference.assignments)
            .filter(|(a, b)| a != b)
            .count();
        let center_bit_disagreements = run
            .centers
            .iter()
            .zip(&reference.centers)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_center_absolute_error = run
            .centers
            .iter()
            .zip(&reference.centers)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0, f64::max);
        if matches!(
            strategy,
            Strategy::Separate | Strategy::PairedPoint | Strategy::PackedPaired
        ) {
            anyhow::ensure!(
                assignment_disagreements == 0 && center_bit_disagreements == 0,
                "exact strategy changed fitted output"
            );
        }
        let total_samples_ns = samples.totals.clone();
        let total_ns = percentiles::compute_percentiles(&mut samples.totals)?;
        writeln!(
            output,
            "  {}: mean {:.0} ns; assignments {}, center bits {}",
            strategy.name(),
            total_ns.mean,
            assignment_disagreements,
            center_bit_disagreements
        )?;
        strategies.push(StrategyResult {
            strategy: strategy.name(),
            samples: samples.totals.len(),
            total_ns,
            packing_ns: percentiles::compute_percentiles(&mut samples.packings)?,
            total_samples_ns,
            packed_bytes: run.packed_bytes,
            assignment_disagreements,
            center_bit_disagreements,
            max_center_absolute_error,
            min_reference_margin: min_margin,
        });
    }
    Ok(WorkloadResults {
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
        Ok(match layout {
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
        })
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
