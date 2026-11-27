/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Paired Wikipedia Cohere benchmark of scalar scoring and parallel, tiled GEMM.
//! Arguments: DATA_DIRECTORY OUTPUT_CSV QUERIES QUERY_BATCH THREADS K REPS [DATABASE_BLOCK].
//! Use a release build with the flat-gemm feature. Query rows are sampled evenly.

use std::{
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    num::NonZeroUsize,
    path::Path,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use diskann::{
    flat::{
        DistanceHandler, DistancesUnordered,
        gemm::{self, GemmOptions},
    },
    graph::glue::CopyIds,
    neighbor::Neighbor,
    provider::HasId,
    utils::{VectorRepr, vector_repr::BufferedDistance},
};
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{PreprocessedDistanceFunction, distance::Metric};
use futures_util::FutureExt;
use rayon::ThreadPool;

// Match the existing flat-index benchmark's scoring and batch submission paths.
struct Visitor<'a> {
    data: rowmajor::Ref<'a, f32>,
    computer: BufferedDistance<f32>,
}
impl HasId for Visitor<'_> {
    type Id = u32;
}
impl DistancesUnordered for Visitor<'_> {
    type Error = diskann::error::Infallible;

    async fn distances_unordered<F>(&mut self, mut f: F) -> Result<(), Self::Error>
    where
        F: Send + FnMut(u32, f32),
    {
        for (i, vector) in self.data.rows().enumerate() {
            f(i as u32, self.computer.evaluate_similarity(vector));
        }
        Ok(())
    }

    async fn distances_unordered_with<H>(&mut self, handler: &mut H) -> Result<(), Self::Error>
    where
        H: DistanceHandler<u32>,
    {
        let dim = self.data.ncols();
        if dim == 0 || !handler.prefers_buffered_batches() {
            return self.distances_unordered(|id, d| handler.add(id, d)).await;
        }
        const BATCH: usize = 32;
        let mut ids = [0; BATCH];
        let mut distances = [0.0; BATCH];
        for (block, vectors) in self.data.as_slice().chunks(BATCH * dim).enumerate() {
            let len = vectors.len() / dim;
            for (lane, vector) in vectors.chunks_exact(dim).enumerate() {
                ids[lane] = (block * BATCH + lane) as u32;
                distances[lane] = self.computer.evaluate_similarity(vector);
            }
            handler.add_batch(&ids[..len], &distances[..len]);
        }
        Ok(())
    }
}

fn scalar(
    data: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: NonZeroUsize,
    pool: &ThreadPool,
) -> Result<Vec<Vec<Neighbor<u32>>>> {
    let mut results: Vec<Result<Vec<Neighbor<u32>>>> =
        (0..queries.nrows()).map(|_| Ok(Vec::new())).collect();
    pool.scope(|scope| {
        for (query, result) in queries.rows().zip(results.iter_mut()) {
            scope.spawn(move |_| {
                *result = (|| {
                    let mut visitor = Visitor {
                        data,
                        computer: f32::query_distance(query, Metric::InnerProduct),
                    };
                    let mut output = Vec::with_capacity(k.get());
                    let stats =
                        diskann::flat::knn_search(&mut visitor, k, CopyIds, (), &mut output)
                            .now_or_never()
                            .context("unexpectedly suspended")??;
                    ensure!(stats.cmps as usize == data.nrows(), "incomplete scan");
                    ensure!(
                        stats.result_count as usize == output.len(),
                        "incorrect statistics"
                    );
                    Ok(output)
                })();
            });
        }
    });
    results.into_iter().collect()
}

#[derive(Default)]
struct Validation {
    overlap: usize,
    count: usize,
    old_hits: usize,
    new_hits: usize,
    gt_count: usize,
    max_abs_error: f64,
}

impl Validation {
    fn check(
        &mut self,
        data: rowmajor::Ref<'_, f32>,
        queries: rowmajor::Ref<'_, f32>,
        truth: &[Vec<u32>],
        before: &[Vec<Neighbor<u32>>],
        after: &[Vec<Neighbor<u32>>],
        k: usize,
    ) -> Result<()> {
        ensure!(before.len() == after.len() && after.len() == queries.nrows());
        for (qi, (a, b)) in before.iter().zip(after).enumerate() {
            ensure!(a.len() == k && b.len() == k, "result count differs");
            let old_ids: std::collections::HashSet<_> = a.iter().map(|n| *n.id()).collect();
            let new_ids: std::collections::HashSet<_> = b.iter().map(|n| *n.id()).collect();
            ensure!(new_ids.len() == k, "duplicate IDs");
            self.overlap += old_ids.intersection(&new_ids).count();
            self.count += k;
            for (left, right) in a.iter().zip(b) {
                let tolerance = 1e-4 * (1.0 + f64::from(left.distance().abs()));
                ensure!(
                    (f64::from(*left.distance()) - f64::from(*right.distance())).abs() <= tolerance,
                    "top-k score mismatch: {left:?} versus {right:?}"
                );
                let score = -queries
                    .row(qi)
                    .iter()
                    .zip(data.row(*right.id() as usize))
                    .map(|(&x, &y)| f64::from(x) * f64::from(y))
                    .sum::<f64>();
                let error = (f64::from(*right.distance()) - score).abs();
                self.max_abs_error = self.max_abs_error.max(error);
                ensure!(
                    error <= 1e-4 * (1.0 + score.abs()),
                    "GEMM score error {error}"
                );
            }
            ensure!(
                b.windows(2)
                    .all(|pair| pair[0].distance() <= pair[1].distance()),
                "unsorted results"
            );
            if k <= truth[qi].len() {
                let expected = &truth[qi][..k];
                self.old_hits += expected.iter().filter(|id| old_ids.contains(id)).count();
                self.new_hits += expected.iter().filter(|id| new_ids.contains(id)).count();
                self.gt_count += k;
            }
        }
        Ok(())
    }
}

fn positive(args: &[String], index: usize, name: &str) -> Result<NonZeroUsize> {
    let value = args
        .get(index)
        .with_context(|| format!("missing {name}"))?
        .parse()?;
    NonZeroUsize::new(value).with_context(|| format!("{name} must be positive"))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (7..=8).contains(&args.len()),
        "usage: flat_gemm DATA_DIRECTORY OUTPUT_CSV QUERIES QUERY_BATCH THREADS K REPS [DATABASE_BLOCK]"
    );
    let sample_count = positive(&args, 2, "QUERIES")?;
    let batch = positive(&args, 3, "QUERY_BATCH")?;
    let threads = positive(&args, 4, "THREADS")?;
    let k = positive(&args, 5, "K")?;
    let reps = positive(&args, 6, "REPS")?;
    let mut options = GemmOptions {
        max_workers: threads,
        query_block_size: batch,
        ..Default::default()
    };
    if args.len() == 8 {
        options.database_block_size = positive(&args, 7, "DATABASE_BLOCK")?;
    }
    let dir = Path::new(&args[0]);
    let data = diskann_utils::io::read_bin::<f32>(&mut File::open(
        dir.join("wikipedia_base.bin.crop_nb_1000000"),
    )?)?;
    let all_queries =
        diskann_utils::io::read_bin::<f32>(&mut File::open(dir.join("wikipedia_query.bin"))?)?;
    let all_truth = diskann_utils::io::read_bin::<u32>(&mut File::open(dir.join("wikipedia-1M"))?)?;
    ensure!(sample_count.get() <= all_queries.nrows() && k.get() <= data.nrows());
    ensure!(data.nrows() <= u32::MAX as usize && all_truth.nrows() == all_queries.nrows());
    ensure!(data.ncols() == all_queries.ncols());
    let queries = rowmajor::Owned::from_fn(sample_count.get(), all_queries.ncols(), |rc| {
        all_queries.row(rc.row * all_queries.nrows() / sample_count.get())[rc.col]
    });
    let truth: Vec<_> = (0..sample_count.get())
        .map(|i| {
            all_truth
                .row(i * all_queries.nrows() / sample_count.get())
                .to_vec()
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.get())
        .build()?;
    let mut csv = BufWriter::new(File::create(&args[1])?);
    writeln!(
        csv,
        "rep,first_query,nq,threads,k,query_block,database_block,first,scalar_ms,gemm_ms"
    )?;
    println!(
        "{}x{}, {} sampled queries, batch={batch}, threads={threads}, k={k}, reps={reps}, GEMM tiles={}x{}",
        data.nrows(),
        data.ncols(),
        queries.nrows(),
        options.query_block_size,
        options.database_block_size
    );
    let warm = queries
        .window_iter(batch)
        .next()
        .context("empty query set")?;
    black_box(scalar(data.as_view(), warm, k, &pool)?);
    black_box(gemm::knn_inner_product(
        data.as_view(),
        warm,
        k,
        &options,
        &pool,
    )?);
    let mut validation = Validation::default();
    let mut totals = [0.0; 2];
    for rep in 0..reps.get() {
        for (group, query_batch) in queries.window_iter(batch).enumerate() {
            let first = (rep + group) % 2;
            let mut results: [Vec<Vec<Neighbor<u32>>>; 2] = Default::default();
            let mut times = [0.0; 2];
            for version in [first, 1 - first] {
                let start = Instant::now();
                results[version] = if version == 0 {
                    scalar(data.as_view(), query_batch, k, &pool)?
                } else {
                    gemm::knn_inner_product(data.as_view(), query_batch, k, &options, &pool)?
                };
                times[version] = start.elapsed().as_secs_f64() * 1000.0;
                black_box(&results[version]);
                totals[version] += times[version];
            }
            let q0 = group * batch.get();
            validation.check(
                data.as_view(),
                query_batch,
                &truth[q0..q0 + query_batch.nrows()],
                &results[0],
                &results[1],
                k.get(),
            )?;
            writeln!(
                csv,
                "{rep},{q0},{},{threads},{k},{},{},{first},{:.6},{:.6}",
                query_batch.nrows(),
                options.query_block_size,
                options.database_block_size,
                times[0],
                times[1]
            )?;
            csv.flush()?;
        }
        println!("completed repetition {}/{reps}", rep + 1);
    }
    let queries_run = (queries.nrows() * reps.get()) as f64;
    println!(
        "scalar amortized ms/query={:.6}, GEMM={:.6}, speedup={:.3}x, QPS={:.2}->{:.2}",
        totals[0] / queries_run,
        totals[1] / queries_run,
        totals[0] / totals[1],
        queries_run * 1000.0 / totals[0],
        queries_run * 1000.0 / totals[1]
    );
    println!(
        "top-k ID overlap={}/{}, groundtruth hits scalar={}, GEMM={}, denominator={}, max absolute score error vs f64={:.9}",
        validation.overlap,
        validation.count,
        validation.old_hits,
        validation.new_hits,
        validation.gt_count,
        validation.max_abs_error
    );
    Ok(())
}
