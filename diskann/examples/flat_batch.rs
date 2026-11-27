/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! End-to-end synthetic comparison, including packing, filtering, norms and merging.
//! Args: ROWS DIM QUERIES K WORKERS REPS TYPE METRIC KEEP_PERCENT [QUERY_BLOCK] [DB_BLOCK].
//! TYPE: f32, f16, i8, u8. METRIC uses diskann-vector's names.

use std::{hint::black_box, num::NonZeroUsize, str::FromStr, sync::Arc, time::Instant};

use anyhow::{Context, Result, ensure};
use diskann::{
    flat::gemm::{
        self, BatchOptions, BatchStats, Dataset, Engine, GemmElement, GemmOptions, MemorySource,
        OwnedBlock, SearchResult,
    },
    graph::glue::CopyIds,
};
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::distance::Metric;

struct Config {
    rows: usize,
    dim: usize,
    queries: usize,
    k: NonZeroUsize,
    workers: NonZeroUsize,
    reps: usize,
    metric: Metric,
    keep: usize,
    query_block: NonZeroUsize,
    database_block: NonZeroUsize,
}

fn vectors(rows: usize, dim: usize, seed: usize, normalized: bool) -> rowmajor::Owned<f32> {
    let mut values = rowmajor::Owned::from_fn(rows, dim, |rc| {
        // Deterministic inputs independent of the search implementations.
        let x = (rc.row.wrapping_mul(2654435761) ^ rc.col.wrapping_mul(2246822519) ^ seed) % 65521;
        (x as f32 * 0.013).sin() * 63.0
    });
    if normalized {
        for row in values.rows_mut() {
            let norm = row
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                .sqrt() as f32;
            if norm > 0.0 {
                for x in row {
                    *x /= norm;
                }
            }
        }
    }
    values
}

fn run<T: GemmElement>(config: Config, convert: impl Fn(f32) -> T) -> Result<()> {
    let normalized = T::effective_metric(config.metric) == Metric::CosineNormalized;
    let raw = vectors(config.rows, config.dim, 31, normalized);
    let data = Arc::new(rowmajor::Owned::from_fn(config.rows, config.dim, |rc| {
        convert(raw.row(rc.row)[rc.col])
    }));
    let raw = vectors(config.queries, config.dim, 173, normalized);
    let queries = Arc::new(rowmajor::Owned::from_fn(config.queries, config.dim, |rc| {
        convert(raw.row(rc.row)[rc.col])
    }));
    let ids: Arc<[u64]> = (0..config.rows as u64)
        .map(|i| (1u64 << 40) + i)
        .collect::<Vec<_>>()
        .into();
    let dataset = Dataset::new(data.as_view(), &ids)?;
    let keep = config.keep;
    let filter = Arc::new(move |q: usize, id: u64| ((id + q as u64 * 7) % 100) < keep as u64);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.workers.get())
            .build()?,
    );
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    let mut options = BatchOptions {
        gemm: GemmOptions {
            max_workers: config.workers,
            query_block_size: config.query_block,
            database_block_size: config.database_block,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut execute = |variant: usize| -> Result<Vec<SearchResult<u64>>> {
        options.engine = match variant {
            0 => Engine::Direct,
            1 => Engine::Gemm,
            _ => Engine::Auto,
        };
        if variant != 3 {
            Ok(gemm::knn_search(
                dataset,
                queries.as_view(),
                config.k,
                config.metric,
                filter.as_ref(),
                &options,
                &pool,
            )?)
        } else {
            let mut source = MemorySource::new(OwnedBlock::new(data.clone(), ids.clone())?);
            let mut outputs = (0..config.queries).map(|_| Vec::new()).collect::<Vec<_>>();
            let stats = runtime.block_on(gemm::knn_search_batch(
                &mut source,
                queries.clone(),
                config.k,
                config.metric,
                filter.clone(),
                &options,
                pool.clone(),
                CopyIds,
                &mut outputs,
            ))?;
            Ok(outputs
                .into_iter()
                .zip(stats)
                .map(|(neighbors, stats)| SearchResult { neighbors, stats })
                .collect())
        }
    };
    let reference = execute(0)?;
    for variant in 1..4 {
        black_box(execute(variant)?);
    }
    let mut timings: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    let mut results = Vec::new();
    for rep in 0..config.reps {
        // Rotate measurement order to reduce systematic cache/thermal bias.
        for step in 0..4 {
            let variant = (rep + step) % 4;
            let start = Instant::now();
            let result = black_box(execute(variant)?);
            timings[variant].push(start.elapsed().as_secs_f64() * 1000.0);
            if rep + 1 == config.reps {
                results.push((variant, result));
            }
        }
    }
    println!(
        "engine,median_ms_batch,amortized_ms_query,id_overlap,max_score_error,cmps,eligible,gemm_tiles,direct_tiles,recomputations"
    );
    results.sort_by_key(|r| r.0);
    for (variant, result) in results {
        timings[variant].sort_by(f64::total_cmp);
        let ms = timings[variant][config.reps / 2];
        let mut overlap = 0usize;
        let mut total = 0usize;
        let mut error = 0.0f32;
        let mut stats = BatchStats::default();
        for (q, (got, expected)) in result.iter().zip(&reference).enumerate() {
            ensure!(
                got.neighbors.len() == expected.neighbors.len(),
                "result count mismatch"
            );
            total += expected.neighbors.len();
            for n in &got.neighbors {
                ensure!(filter(q, *n.id()), "filtered candidate returned");
                if let Some(e) = expected.neighbors.iter().find(|e| e.id() == n.id()) {
                    overlap += 1;
                    error = error.max((*e.distance() - *n.distance()).abs());
                }
            }
            stats.cmps += got.stats.cmps;
            stats.eligible_candidates += got.stats.eligible_candidates;
            stats.gemm_tiles += got.stats.gemm_tiles;
            stats.direct_tiles += got.stats.direct_tiles;
            stats.recomputations += got.stats.recomputations;
        }
        let name = ["direct", "gemm", "auto", "stream-auto"][variant];
        println!(
            "{name},{ms:.6},{:.6},{:.8},{error:e},{},{},{},{},{}",
            ms / config.queries as f64,
            if total == 0 {
                1.0
            } else {
                overlap as f64 / total as f64
            },
            stats.cmps,
            stats.eligible_candidates,
            stats.gemm_tiles,
            stats.direct_tiles,
            stats.recomputations
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let number = |i: usize, default: usize| -> Result<usize> {
        args.get(i).map_or(Ok(default), |x| {
            x.parse().context("invalid integer argument")
        })
    };
    let nonzero = |i, default| -> Result<NonZeroUsize> {
        NonZeroUsize::new(number(i, default)?).context("argument must be nonzero")
    };
    let config = Config {
        rows: number(0, 16384)?,
        dim: number(1, 128)?,
        queries: nonzero(2, 32)?.get(),
        k: nonzero(3, 10)?,
        workers: nonzero(4, 1)?,
        reps: nonzero(5, 3)?.get(),
        metric: Metric::from_str(args.get(7).map_or("innerproduct", String::as_str))?,
        keep: number(8, 100)?,
        query_block: nonzero(9, 32)?,
        database_block: nonzero(10, 1024)?,
    };
    ensure!(config.keep <= 100, "KEEP_PERCENT must be in [0, 100]");
    match args.get(6).map_or("f32", String::as_str) {
        "f32" => run(config, |x| x),
        "f16" => run(config, half::f16::from_f32),
        "i8" => run(config, |x| x as i8),
        "u8" => run(config, |x| (x + 64.0) as u8),
        _ => anyhow::bail!("TYPE must be f32, f16, i8, or u8"),
    }
}
