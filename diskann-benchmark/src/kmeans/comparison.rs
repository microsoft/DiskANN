/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
    time::Instant,
};

use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    utils::{percentiles, MicroSeconds},
    Benchmark, Checkpoint, Output, Registry,
};
use diskann_disk::utils::{k_meanspp_selecting_pivots, run_lloyds};
use diskann_providers::utils::{create_thread_pool, RayonThreadPool};
use diskann_quantization::algorithms::kmeans::{lloyds::lloyds, plusplus::kmeans_plusplus_into};
use diskann_utils::views::Matrix;
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

use crate::inputs::kmeans::{KmeansComparison, KmeansImplementation, KmeansPhase};

pub(super) fn register(registry: &mut Registry) -> anyhow::Result<()> {
    registry.register("kmeans-comparison", Comparison)?;
    Ok(())
}

struct Comparison;

impl Benchmark for Comparison {
    type Input = KmeansComparison;
    type Output = ComparisonOutput;

    fn try_match(&self, _input: &Self::Input, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Runs one K-means implementation with phase telemetry")
    }

    fn run(
        &self,
        input: &Self::Input,
        _checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> anyhow::Result<Self::Output> {
        writeln!(output, "{}", input)?;
        let mut workloads = Vec::new();

        for &threads in &input.thread_counts {
            let pool = create_thread_pool(threads.get())?;
            for &dim in &input.dimensions {
                for &num_centers in &input.center_counts {
                    let workload = Workload {
                        num_points: input.num_points.get(),
                        dim: dim.get(),
                        num_centers: num_centers.get(),
                        max_iterations: input.max_iterations.get(),
                        threads: threads.get(),
                    };
                    let result = run_workload(
                        input.implementation,
                        input.phase,
                        workload,
                        input.measurements.get(),
                        input.seed,
                        &pool,
                    )?;
                    writeln!(output, "{}", result)?;
                    workloads.push(result);
                }
            }
        }

        Ok(ComparisonOutput {
            implementation: input.implementation,
            phase: input.phase,
            workloads,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Workload {
    num_points: usize,
    dim: usize,
    num_centers: usize,
    max_iterations: usize,
    threads: usize,
}

#[derive(Debug, Serialize)]
struct PhaseMeasurements {
    samples: Vec<MicroSeconds>,
    percentiles: percentiles::Percentiles<MicroSeconds>,
}

#[derive(Debug, Serialize)]
struct WorkloadResult {
    workload: Workload,
    initialization: PhaseMeasurements,
    #[serde(skip_serializing_if = "Option::is_none")]
    lloyds: Option<PhaseMeasurements>,
    total: PhaseMeasurements,
    center_hash: u64,
    objective: f64,
}

#[derive(Debug, Serialize)]
struct ComparisonOutput {
    implementation: KmeansImplementation,
    phase: KmeansPhase,
    workloads: Vec<WorkloadResult>,
}

impl fmt::Display for WorkloadResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} points x {} dimensions x {} centers, {} iterations, {} threads: init median {:.3} ms",
            self.workload.num_points,
            self.workload.dim,
            self.workload.num_centers,
            self.workload.max_iterations,
            self.workload.threads,
            self.initialization.percentiles.median / 1_000.0,
        )?;
        if let Some(lloyds) = &self.lloyds {
            write!(
                f,
                ", Lloyd median {:.3} ms",
                lloyds.percentiles.median / 1_000.0
            )?;
        }
        writeln!(
            f,
            ", total median {:.3} ms",
            self.total.percentiles.median / 1_000.0
        )
    }
}

#[derive(Debug)]
struct Measurement {
    initialization: MicroSeconds,
    lloyds: Option<MicroSeconds>,
    total: MicroSeconds,
    centers: Vec<f32>,
}

fn run_workload(
    implementation: KmeansImplementation,
    phase: KmeansPhase,
    workload: Workload,
    num_measurements: usize,
    seed: u64,
    pool: &RayonThreadPool,
) -> anyhow::Result<WorkloadResult> {
    let data_seed = seed
        .wrapping_add(workload.dim as u64)
        .wrapping_add(workload.num_centers as u64);
    let mut rng = StdRng::seed_from_u64(data_seed);
    let data: Vec<f32> = (0..workload.num_points * workload.dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let data_matrix = Matrix::try_from(
        data.clone().into_boxed_slice(),
        workload.num_points,
        workload.dim,
    )
    .map_err(|_| anyhow::anyhow!("generated data matrix has an invalid shape"))?;

    let run = || {
        run_measurement(
            implementation,
            phase,
            &data,
            &data_matrix,
            workload,
            data_seed,
            pool,
        )
    };

    let quality = run()?;
    let objective = objective(&data, &quality.centers, workload);
    let center_hash = hash_centers(&quality.centers);

    let mut initialization = Vec::with_capacity(num_measurements);
    let mut lloyds = (phase == KmeansPhase::All).then(|| Vec::with_capacity(num_measurements));
    let mut total = Vec::with_capacity(num_measurements);
    for _ in 0..num_measurements {
        let measurement = run()?;
        initialization.push(measurement.initialization);
        if let (Some(samples), Some(sample)) = (&mut lloyds, measurement.lloyds) {
            samples.push(sample);
        }
        total.push(measurement.total);
    }

    Ok(WorkloadResult {
        workload,
        initialization: make_phase_measurements(initialization)?,
        lloyds: lloyds.map(make_phase_measurements).transpose()?,
        total: make_phase_measurements(total)?,
        center_hash,
        objective,
    })
}

fn run_measurement(
    implementation: KmeansImplementation,
    phase: KmeansPhase,
    data: &[f32],
    data_matrix: &Matrix<f32>,
    workload: Workload,
    seed: u64,
    pool: &RayonThreadPool,
) -> anyhow::Result<Measurement> {
    match implementation {
        KmeansImplementation::Disk => run_disk(phase, data, workload, seed, pool),
        KmeansImplementation::Quantization => {
            run_quantization(phase, data_matrix, workload, seed, pool)
        }
    }
}

fn run_disk(
    phase: KmeansPhase,
    data: &[f32],
    workload: Workload,
    seed: u64,
    pool: &RayonThreadPool,
) -> anyhow::Result<Measurement> {
    let mut centers = vec![0.0; workload.num_centers * workload.dim];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut canceled = false;
    let total_start = Instant::now();
    let initialization_start = Instant::now();
    k_meanspp_selecting_pivots(
        data,
        workload.num_points,
        workload.dim,
        &mut centers,
        workload.num_centers,
        &mut rng,
        &mut canceled,
        pool.as_ref(),
    )?;
    let initialization = initialization_start.elapsed().into();
    let lloyds = if phase == KmeansPhase::All {
        let start = Instant::now();
        run_lloyds(
            data,
            workload.num_points,
            workload.dim,
            &mut centers,
            workload.num_centers,
            workload.max_iterations,
            &mut canceled,
            pool.as_ref(),
        )?;
        Some(start.elapsed().into())
    } else {
        None
    };
    Ok(Measurement {
        initialization,
        lloyds,
        total: total_start.elapsed().into(),
        centers,
    })
}

fn run_quantization(
    phase: KmeansPhase,
    data: &Matrix<f32>,
    workload: Workload,
    seed: u64,
    pool: &RayonThreadPool,
) -> anyhow::Result<Measurement> {
    let mut centers = Matrix::new(0.0, workload.num_centers, workload.dim);
    let mut rng = StdRng::seed_from_u64(seed);
    let total_start = Instant::now();
    let initialization_start = Instant::now();
    pool.install(|| kmeans_plusplus_into(centers.as_mut_view(), data.as_view(), &mut rng))?;
    let initialization = initialization_start.elapsed().into();
    let lloyds = if phase == KmeansPhase::All {
        let start = Instant::now();
        pool.install(|| {
            lloyds(
                data.as_view(),
                centers.as_mut_view(),
                workload.max_iterations,
            )
        });
        Some(start.elapsed().into())
    } else {
        None
    };
    Ok(Measurement {
        initialization,
        lloyds,
        total: total_start.elapsed().into(),
        centers: centers.into_inner().into(),
    })
}

fn make_phase_measurements(mut samples: Vec<MicroSeconds>) -> anyhow::Result<PhaseMeasurements> {
    let percentiles = percentiles::compute_percentiles(&mut samples)?;
    Ok(PhaseMeasurements {
        samples,
        percentiles,
    })
}

fn hash_centers(centers: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    centers
        .iter()
        .for_each(|value| value.to_bits().hash(&mut hasher));
    hasher.finish()
}

fn objective(data: &[f32], centers: &[f32], workload: Workload) -> f64 {
    data.chunks_exact(workload.dim)
        .map(|point| {
            centers
                .chunks_exact(workload.dim)
                .map(|center| {
                    let distance: f32 = SquaredL2::evaluate(point, center);
                    f64::from(distance)
                })
                .min_by(f64::total_cmp)
                .expect("the benchmark requires at least one center")
        })
        .sum()
}
