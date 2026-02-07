// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//! Benchmark execution logic for multi-vector distance computations.

use std::io::Write;

use rand::{distr::StandardUniform, rngs::StdRng, SeedableRng};
use serde::Serialize;

use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use diskann_vector::DistanceFunction;

use crate::distance::{Chamfer, SgemmScratch};
use crate::{MultiVector, Standard, TransposedMultiVector};

use super::input::{MultiVectorOp, Run};

/// Result of a single benchmark run.
#[derive(Debug, Serialize)]
pub(super) struct RunResult {
    /// The run configuration.
    pub run: Run,
    /// Latency percentiles.
    pub percentiles: percentiles::Percentiles<MicroSeconds>,
    /// Checksum of the first 10 distance values for verification.
    /// Only populated when verify is enabled in the input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_distance_checksum: Option<f32>,
}

/// Display wrapper for formatting results.
pub(super) struct DisplayWrapper<'a, T: ?Sized>(pub &'a T);

impl<T: ?Sized> std::ops::Deref for DisplayWrapper<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

impl std::fmt::Display for DisplayWrapper<'_, [RunResult]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            return Ok(());
        }

        let header = [
            "Dim",
            "QueryTok",
            "DocTok",
            "Points",
            "Mean(µs)",
            "P90(µs)",
            "P99(µs)",
        ];

        let mut table = diskann_benchmark_runner::utils::fmt::Table::new(header, self.len());

        self.iter().enumerate().for_each(|(row, r)| {
            let mut row = table.row(row);

            row.insert(r.run.dim, 0);
            row.insert(r.run.num_query_token, 1);
            row.insert(r.run.num_doc_token, 2);
            row.insert(r.run.num_points, 3);
            row.insert(format!("{:.2}", r.percentiles.mean), 4);
            row.insert(format!("{:.2}", r.percentiles.p90.as_f64()), 5);
            row.insert(format!("{:.2}", r.percentiles.p99.as_f64()), 6);
        });

        table.fmt(f)
    }
}

/// Generate a random multi-vector with the given number of tokens and dimension.
fn generate_random_multivector(rng: &mut StdRng, num_tokens: usize, dim: usize) -> MultiVector {
    use rand::distr::Distribution;

    let mut mat = MultiVector::new(Standard::new(num_tokens, dim), 0.0f32).unwrap();
    for i in 0..num_tokens {
        if let Some(row) = mat.get_row_mut(i) {
            for val in row.iter_mut() {
                *val = StandardUniform.sample(rng);
            }
        }
    }
    mat
}

/// Run benchmark with the specified approach using the `DistanceFunction` trait.
pub(super) fn run_benchmark_with_approach<A: Default>(
    input: &MultiVectorOp,
    verify: bool,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<Vec<RunResult>, anyhow::Error>
where
    Chamfer<A>: for<'a> DistanceFunction<&'a MultiVector, &'a MultiVector>,
{
    // For MultiVector docs, we need to generate directly (no conversion)
    let mut results = Vec::new();

    for run in input.runs.iter() {
        let mut rng = StdRng::seed_from_u64(0x12345);

        // Generate query multi-vector (always row-major)
        let query = generate_random_multivector(&mut rng, run.num_query_token.get(), run.dim.get());

        // Generate document multi-vectors
        let docs: Vec<MultiVector> = (0..run.num_points.get())
            .map(|_| generate_random_multivector(&mut rng, run.num_doc_token.get(), run.dim.get()))
            .collect();

        let chamfer = Chamfer::<A>::new();

        // Collect latencies
        let mut latencies = Vec::with_capacity(run.num_measurements.get());
        let mut distances = vec![0.0f32; docs.len()];

        for _ in 0..run.num_measurements.get() {
            let start = std::time::Instant::now();
            for _ in 0..run.loops_per_measurement.get() {
                for (i, doc) in docs.iter().enumerate() {
                    distances[i] = chamfer.evaluate_similarity(&query, doc);
                }
                std::hint::black_box(&mut distances);
            }
            latencies.push(start.elapsed().into());
        }

        let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();

        // Compute checksum of first 10 distances for verification
        let sample_distance_checksum = if verify {
            Some(distances.iter().take(10).sum::<f32>())
        } else {
            None
        };

        results.push(RunResult {
            run: run.clone(),
            percentiles,
            sample_distance_checksum,
        });

        writeln!(
            output,
            "  Completed run: dim={}, points={}",
            run.dim, run.num_points
        )?;
    }

    Ok(results)
}

/// Run benchmark with the transposed approach using the `DistanceFunction` trait.
///
/// This variant uses `MultiVector` for the query and `TransposedMultiVector` for documents.
pub(super) fn run_benchmark_with_transposed_approach<A: Default>(
    input: &MultiVectorOp,
    verify: bool,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<Vec<RunResult>, anyhow::Error>
where
    Chamfer<A>: for<'a> DistanceFunction<&'a MultiVector, &'a TransposedMultiVector>,
{
    let mut results = Vec::new();

    for run in input.runs.iter() {
        let mut rng = StdRng::seed_from_u64(0x12345);

        // Generate query multi-vector (always row-major)
        let query = generate_random_multivector(&mut rng, run.num_query_token.get(), run.dim.get());

        // Generate document multi-vectors and transpose them
        let docs: Vec<TransposedMultiVector> = (0..run.num_points.get())
            .map(|_| {
                let mv =
                    generate_random_multivector(&mut rng, run.num_doc_token.get(), run.dim.get());
                TransposedMultiVector::from(&mv)
            })
            .collect();

        let chamfer = Chamfer::<A>::new();

        // Collect latencies
        let mut latencies = Vec::with_capacity(run.num_measurements.get());
        let mut distances = vec![0.0f32; docs.len()];

        for _ in 0..run.num_measurements.get() {
            let start = std::time::Instant::now();
            for _ in 0..run.loops_per_measurement.get() {
                for (i, doc) in docs.iter().enumerate() {
                    distances[i] = chamfer.evaluate_similarity(&query, doc);
                }
                std::hint::black_box(&mut distances);
            }
            latencies.push(start.elapsed().into());
        }

        let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();

        // Compute checksum of first 10 distances for verification
        let sample_distance_checksum = if verify {
            Some(distances.iter().take(10).sum::<f32>())
        } else {
            None
        };

        results.push(RunResult {
            run: run.clone(),
            percentiles,
            sample_distance_checksum,
        });

        writeln!(
            output,
            "  Completed run: dim={}, points={}",
            run.dim, run.num_points
        )?;
    }

    Ok(results)
}

/// Run benchmark with query-transposed approach using the `DistanceFunction` trait.
///
/// This variant uses `TransposedMultiVector` for the query and `MultiVector` for documents.
/// The scratch buffer is stored in the `Chamfer<QueryTransposedWithTilingApproach>` itself.
pub(super) fn run_benchmark_with_query_transposed_approach<A: Default>(
    input: &MultiVectorOp,
    verify: bool,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<Vec<RunResult>, anyhow::Error>
where
    Chamfer<A>: for<'a> DistanceFunction<&'a TransposedMultiVector, &'a MultiVector>,
{
    let mut results = Vec::new();

    for run in input.runs.iter() {
        let mut rng = StdRng::seed_from_u64(0x12345);

        // Generate query multi-vector and transpose it
        let query_mv =
            generate_random_multivector(&mut rng, run.num_query_token.get(), run.dim.get());
        let query = TransposedMultiVector::from(&query_mv);

        // Generate document multi-vectors (row-major)
        let docs: Vec<MultiVector> = (0..run.num_points.get())
            .map(|_| generate_random_multivector(&mut rng, run.num_doc_token.get(), run.dim.get()))
            .collect();

        // Chamfer instance holds the scratch buffer for reuse across documents
        let chamfer = Chamfer::<A>::new();

        // Collect latencies
        let mut latencies = Vec::with_capacity(run.num_measurements.get());
        let mut distances = vec![0.0f32; docs.len()];

        for _ in 0..run.num_measurements.get() {
            let start = std::time::Instant::now();
            for _ in 0..run.loops_per_measurement.get() {
                for (i, doc) in docs.iter().enumerate() {
                    distances[i] = chamfer.evaluate_similarity(&query, doc);
                }
                std::hint::black_box(&mut distances);
            }
            latencies.push(start.elapsed().into());
        }

        let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();

        // Compute checksum of first 10 distances for verification
        let sample_distance_checksum = if verify {
            Some(distances.iter().take(10).sum::<f32>())
        } else {
            None
        };

        results.push(RunResult {
            run: run.clone(),
            percentiles,
            sample_distance_checksum,
        });

        writeln!(
            output,
            "  Completed run: dim={}, points={}",
            run.dim, run.num_points
        )?;
    }

    Ok(results)
}

/// Run benchmark with SGEMM approach using pre-allocated scratch buffer.
///
/// This variant uses `Chamfer<SgemmApproach>::evaluate_similarity_with_scratch` to avoid
/// allocation on the hot path. The scratch buffer is pre-allocated before the timing
/// loop starts, ensuring fair comparison against custom SIMD approaches.
pub(super) fn run_benchmark_with_sgemm_approach<A: Default>(
    input: &MultiVectorOp,
    verify: bool,
    mut output: &mut dyn diskann_benchmark_runner::Output,
) -> Result<Vec<RunResult>, anyhow::Error>
where
    Chamfer<A>: SgemmEvaluator,
{
    let mut results = Vec::new();

    for run in input.runs.iter() {
        let mut rng = StdRng::seed_from_u64(0x12345);

        // Generate query multi-vector
        let query = generate_random_multivector(&mut rng, run.num_query_token.get(), run.dim.get());

        // Generate document multi-vectors (row-major, not transposed)
        let docs: Vec<MultiVector> = (0..run.num_points.get())
            .map(|_| generate_random_multivector(&mut rng, run.num_doc_token.get(), run.dim.get()))
            .collect();

        let chamfer = Chamfer::<A>::new();

        // Pre-allocate scratch buffer BEFORE timing loop
        // This ensures allocation time is excluded from measurements
        let mut scratch =
            SgemmScratch::with_capacity(run.num_query_token.get(), run.num_doc_token.get());

        // Collect latencies
        let mut latencies = Vec::with_capacity(run.num_measurements.get());
        let mut distances = vec![0.0f32; docs.len()];

        for _ in 0..run.num_measurements.get() {
            let start = std::time::Instant::now();
            for _ in 0..run.loops_per_measurement.get() {
                for (i, doc) in docs.iter().enumerate() {
                    distances[i] =
                        chamfer.evaluate_similarity_with_scratch(&query, doc, &mut scratch);
                }
                std::hint::black_box(&mut distances);
            }
            latencies.push(start.elapsed().into());
        }

        let percentiles = percentiles::compute_percentiles(&mut latencies).unwrap();

        // Compute checksum of first 10 distances for verification
        let sample_distance_checksum = if verify {
            Some(distances.iter().take(10).sum::<f32>())
        } else {
            None
        };

        results.push(RunResult {
            run: run.clone(),
            percentiles,
            sample_distance_checksum,
        });

        writeln!(
            output,
            "  Completed run: dim={}, points={}",
            run.dim, run.num_points
        )?;
    }

    Ok(results)
}

/// Trait for SGEMM-based distance computation with scratch buffer support.
///
/// This trait is implemented by `Chamfer<SgemmApproach>` to provide the
/// `evaluate_similarity_with_scratch` method for benchmarking.
pub(super) trait SgemmEvaluator {
    /// Evaluates similarity using a pre-allocated scratch buffer.
    fn evaluate_similarity_with_scratch(
        &self,
        query: &MultiVector,
        doc: &MultiVector,
        scratch: &mut SgemmScratch,
    ) -> f32;
}

impl SgemmEvaluator for Chamfer<crate::distance::SgemmApproach> {
    fn evaluate_similarity_with_scratch(
        &self,
        query: &MultiVector,
        doc: &MultiVector,
        scratch: &mut SgemmScratch,
    ) -> f32 {
        self.evaluate_similarity_with_scratch(query, doc, scratch)
    }
}
