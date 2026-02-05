/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Exhaustive (brute-force) multi-vector KNN search using Chamfer distance.
//!
//! This module computes exact K-nearest neighbors for multi-vector data by
//! evaluating Chamfer distance between each query and all data points.

use std::io::Write;

use diskann_benchmark_runner::{
    describeln,
    dispatcher::{self, DispatchRule, FailureScore, MatchScore},
    output::Output,
    utils::MicroSeconds,
    Any,
};
use diskann_quantization::multi_vector::{distance::Chamfer, Mat, Standard};
use diskann_vector::PureDistanceFunction;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Serialize;

use crate::{inputs, utils::datafiles};

const NAME: &str = "exhaustive-multi-vector";

pub(super) fn register_benchmarks(benchmarks: &mut diskann_benchmark_runner::registry::Benchmarks) {
    benchmarks.register::<MultiExhaustive<'static>>(NAME, |object, _checkpoint, output| {
        match object.run(output) {
            Ok(v) => Ok(serde_json::to_value(v)?),
            Err(err) => Err(err),
        }
    });
}

macro_rules! write_field {
    ($f:ident, $field:tt, $fmt:literal, $($expr:tt)*) => {
        writeln!($f, concat!("{:>19}: ", $fmt), $field, $($expr)*)
    }
}

/////////////////////
// MultiExhaustive //
/////////////////////

/// Dispatcher target for exhaustive multi-vector search.
pub(super) struct MultiExhaustive<'a> {
    input: &'a inputs::multi::ExhaustiveSearch,
}

impl<'a> MultiExhaustive<'a> {
    fn new(input: &'a inputs::multi::ExhaustiveSearch) -> Self {
        Self { input }
    }

    fn run(self, mut output: &mut dyn Output) -> anyhow::Result<Results> {
        let input = self.input;
        writeln!(output, "{}", input)?;

        // Load data and queries
        writeln!(output, "Loading multi-vector data...")?;
        let data: Vec<Mat<Standard<f32>>> = datafiles::load_multi_vectors(&input.data)?;
        let num_data = data.len();
        writeln!(output, "  Loaded {} data points", num_data)?;

        writeln!(output, "Loading multi-vector queries...")?;
        let queries: Vec<Mat<Standard<f32>>> = datafiles::load_multi_vectors(&input.queries)?;
        let num_queries = queries.len();
        writeln!(output, "  Loaded {} queries", num_queries)?;

        let k = input.num_nearest_neighbors;
        if k > num_data {
            anyhow::bail!(
                "K ({}) exceeds number of data points ({})",
                k,
                num_data
            );
        }

        // Set up thread pool
        let threadpool = rayon::ThreadPoolBuilder::new()
            .num_threads(input.num_threads.get())
            .build()?;

        // Progress bar
        let progress = ProgressBar::with_draw_target(
            Some(num_queries as u64),
            output.draw_target(),
        );
        progress.set_style(ProgressStyle::with_template(
            "Exhaustive search [{elapsed_precise}] {wide_bar} {percent}%",
        )?);

        // Run exhaustive search
        writeln!(output, "Running exhaustive search (K={})...", k)?;
        let start = std::time::Instant::now();

        let results: Vec<Vec<u32>> = threadpool.install(|| {
            (0..num_queries)
                .into_par_iter()
                .map(|query_idx| {
                    let query_mat = queries[query_idx].as_view();

                    // Compute distances to all data points
                    let mut distances: Vec<(f32, u32)> = data
                        .iter()
                        .enumerate()
                        .map(|(doc_idx, doc)| {
                            let dist = Chamfer::evaluate(query_mat, doc.as_view());
                            (dist, doc_idx as u32)
                        })
                        .collect();

                    // Sort by distance (ascending - lower is more similar)
                    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                    progress.inc(1);

                    // Take top-K
                    distances.into_iter().take(k).map(|(_, id)| id).collect()
                })
                .collect()
        });

        progress.finish();
        let search_time: MicroSeconds = start.elapsed().into();

        // Write results to binary groundtruth file
        writeln!(output, "Writing groundtruth to {}...", input.output)?;
        write_groundtruth(&input.output, &results, k)?;

        let result = Results {
            num_data,
            num_queries,
            k,
            num_threads: input.num_threads.get(),
            search_time,
            qps: (num_queries as f64) / search_time.as_seconds(),
        };

        writeln!(output, "\n{}", result)?;
        Ok(result)
    }
}

/// Write groundtruth results in binary format.
///
/// Format:
/// - num_queries (u32)
/// - k (u32)
/// - For each query: k neighbor IDs (u32)
fn write_groundtruth(path: &str, results: &[Vec<u32>], k: usize) -> anyhow::Result<()> {
    use std::io::BufWriter;

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(&(results.len() as u32).to_le_bytes())?;
    writer.write_all(&(k as u32).to_le_bytes())?;

    // Write results
    for neighbors in results {
        for &id in neighbors.iter().take(k) {
            writer.write_all(&id.to_le_bytes())?;
        }
        // Pad with zeros if needed
        for _ in neighbors.len()..k {
            writer.write_all(&0u32.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

//////////////
// Dispatch //
//////////////

impl dispatcher::Map for MultiExhaustive<'static> {
    type Type<'a> = MultiExhaustive<'a>;
}

impl<'a> DispatchRule<&'a inputs::multi::ExhaustiveSearch> for MultiExhaustive<'a> {
    type Error = std::convert::Infallible;

    fn try_match(_from: &&'a inputs::multi::ExhaustiveSearch) -> Result<MatchScore, FailureScore> {
        Ok(MatchScore(0))
    }

    fn convert(from: &'a inputs::multi::ExhaustiveSearch) -> Result<Self, Self::Error> {
        Ok(Self::new(from))
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a inputs::multi::ExhaustiveSearch>,
    ) -> std::fmt::Result {
        match from {
            None => {
                describeln!(f, "- Exhaustive KNN search for multi-vector data")?;
                describeln!(f, "- Uses Chamfer (asymmetric max-sim) distance")?;
                describeln!(f, "- Outputs binary groundtruth file")?;
            }
            Some(_) => {
                // No additional constraints to check
            }
        }
        Ok(())
    }
}

impl<'a> DispatchRule<&'a Any> for MultiExhaustive<'a> {
    type Error = anyhow::Error;

    fn try_match(from: &&'a Any) -> Result<MatchScore, FailureScore> {
        from.try_match::<inputs::multi::ExhaustiveSearch, Self>()
    }

    fn convert(from: &'a Any) -> Result<Self, Self::Error> {
        from.convert::<inputs::multi::ExhaustiveSearch, Self>()
    }

    fn description(
        f: &mut std::fmt::Formatter<'_>,
        from: Option<&&'a Any>,
    ) -> std::fmt::Result {
        Any::description::<inputs::multi::ExhaustiveSearch, Self>(
            f,
            from,
            inputs::multi::ExhaustiveSearch::tag(),
        )
    }
}

/////////////
// Results //
/////////////

#[derive(Debug, Serialize)]
struct Results {
    num_data: usize,
    num_queries: usize,
    k: usize,
    num_threads: usize,
    search_time: MicroSeconds,
    qps: f64,
}

impl std::fmt::Display for Results {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_field!(f, "Data Points", "{}", self.num_data)?;
        write_field!(f, "Queries", "{}", self.num_queries)?;
        write_field!(f, "K", "{}", self.k)?;
        write_field!(f, "Threads", "{}", self.num_threads)?;
        write_field!(f, "Search Time", "{:.3}s", self.search_time.as_seconds())?;
        write_field!(f, "QPS", "{:.2}", self.qps)?;
        Ok(())
    }
}
