/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, io::Write};

use diskann::ivf::workingset::PointRef;
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::percentiles,
    Benchmark, Checkpoint, Output,
};
use diskann_utils::views::rowmajor::Matrix;

use super::{
    data::Data,
    diagnostics, measurement,
    result::{Invocation, Results, StrategyResult, Workload, WorkloadResults},
    BASELINE,
};
use crate::{
    graph_ivf::data::{load_vectors, validate_vectors},
    inputs::graph_ivf_two_means::{FitOrder, GraphIvfTwoMeans, Layout},
};

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
            "Complete two-means: paired baseline and exact two-point tiling"
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
            .map(|file| load_vectors(file, input.data_type))
            .transpose()?;
        if let Some(data) = &corpus {
            validate_vectors(data.as_view(), "two-means corpus")?;
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
            schema_version: 3,
            threads: 1,
            timing_scope: "validate + ISA selection + scratch + seed + distance/selection + ordered f64 sums + center updates + scratch release; diagnostics run after all timed rounds",
            numerical_reference: "paired_point",
            source: if corpus.is_some() { "file_sample" } else { "synthetic_uniform" },
            workloads,
        })
    }
}

#[derive(Default)]
struct Samples {
    totals: Vec<u64>,
    preparations: Vec<u64>,
    last: Option<measurement::Run>,
}

pub(in crate::graph_ivf) fn run_workload(
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
        .map(|_| Samples::default())
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
            )?;
            let samples = &mut measurements[index];
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
        let comparison = diagnostics::compare(points, dim, strategy, &reference, &trace)?;
        anyhow::ensure!(
            comparison.is_exact(),
            "{} violates baseline equivalence",
            strategy.name()
        );
        let total_samples_ns = samples.totals.clone();
        let total_ns = if samples.totals.is_empty() {
            None
        } else {
            Some(percentiles::compute_percentiles(&mut samples.totals)?)
        };
        writeln!(
            output,
            "  {}: mean {:?} ns; failures 0",
            strategy.name(),
            total_ns.as_ref().map(|t| t.mean)
        )?;
        strategies.push(StrategyResult {
            strategy: strategy.name(),
            status: "complete",
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
            diagnostics: Some(comparison),
            // Keep schema v3 compatible; execution and exactness errors return above.
            failures: Vec::new(),
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
