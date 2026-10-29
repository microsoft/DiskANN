/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Incremental IVF-Flat build and search using a benchmark-private memory provider.

mod build;
mod data;
mod provider;
mod search;
#[cfg(test)]
mod tests;

use std::io::Write;

use anyhow::{ensure, Result};
use diskann_benchmark_runner::{
    benchmark::{MatchContext, Score},
    Benchmark, Checkpoint, Output, Registry,
};
use serde::Serialize;

use crate::inputs::ivf::IvfBuildSearch;

pub(crate) fn register_benchmarks(registry: &mut Registry) -> Result<()> {
    registry.register("ivf-index", Ivf)?;
    Ok(())
}

struct Ivf;

impl Benchmark for Ivf {
    type Input = IvfBuildSearch;
    type Output = IvfResult;

    fn try_match(&self, _input: &IvfBuildSearch, context: &MatchContext) -> Score {
        context.success(0)
    }

    fn description(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IVF-Flat build/search, f32 distances, in-memory provider"
        )
    }

    fn run(
        &self,
        input: &IvfBuildSearch,
        checkpoint: Checkpoint<'_>,
        mut output: &mut dyn Output,
    ) -> Result<IvfResult> {
        writeln!(output, "{input}")?;
        let (base, queries) = match &input.data {
            Some(path) => (
                data::read_vectors(path, input.format, input.data_type)?,
                input
                    .search
                    .as_ref()
                    .and_then(|s| s.queries.as_deref())
                    .map(|path| data::read_vectors(path, input.format, input.data_type))
                    .transpose()?
                    .unwrap_or_default(),
            ),
            None => data::synthetic(),
        };
        let query_limit = input
            .search
            .as_ref()
            .and_then(|s| s.query_limit)
            .map_or(usize::MAX, |n| n.get());
        ensure!(
            !base.is_empty() && base.len() <= u32::MAX as usize,
            "empty or oversized dataset"
        );
        let dim = base[0].len();
        ensure!(
            dim > 0
                && base
                    .iter()
                    .chain(queries.iter().take(query_limit))
                    .all(|r| r.len() == dim && r.iter().all(|x| x.is_finite())),
            "invalid vector data"
        );
        if let Some(config) = &input.search {
            ensure!(
                !queries.is_empty() && config.k.get() <= base.len(),
                "invalid query count or k"
            );
        }
        writeln!(
            output,
            "Building {} vectors, D={dim}, batch={}",
            base.len(),
            input.build.batch_size
        )?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let build_index = || {
            let mut completed_batches = 0;
            runtime.block_on(build::build(&base, &input.build, &mut || {
                completed_batches += 1;
                #[derive(Serialize)]
                struct Progress {
                    inserted: usize,
                }
                checkpoint.checkpoint(&Progress {
                    inserted: (completed_batches * input.build.batch_size.get()).min(base.len()),
                })
            }))
        };
        #[cfg(feature = "ivf-parallel")]
        let (index, build) = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(input.build.num_threads.get())
                .build()?;
            pool.install(build_index)?
        };
        #[cfg(not(feature = "ivf-parallel"))]
        let (index, build) = build_index()?;
        checkpoint.checkpoint(&build)?;
        let (search, ground_truth_ms) = match &input.search {
            Some(config) => search::run(index, &base, queries, config, &build, checkpoint, output)?,
            None => (Vec::new(), 0.0),
        };
        let result = IvfResult {
            architecture: format!("{:?}", diskann_wide::ARCH),
            build,
            ground_truth_ms,
            search,
        };
        writeln!(output, "{result}")?;
        Ok(result)
    }
}

#[derive(Debug, Serialize)]
struct IvfResult {
    architecture: String,
    build: build::BuildResult,
    ground_truth_ms: f64,
    search: Vec<search::SearchResults>,
}

impl std::fmt::Display for IvfResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Build: {:.3} ms; fit: {:.3} ms; assignment: {:.3} ms; lists: {}",
            self.build.build_ms, self.build.fit_ms, self.build.assignment_ms, self.build.lists
        )?;
        for result in &self.search {
            writeln!(
                f,
                "nprobe={}: recall@{}={:.6}, candidates/query={:.1}, QPS={:?}",
                result.nprobe,
                result.recall.recall_k,
                result.recall.average,
                result.avg_candidates,
                result.qps
            )?;
        }
        Ok(())
    }
}
