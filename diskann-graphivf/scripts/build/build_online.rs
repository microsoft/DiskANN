/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Online (incremental) graph-IVF build.
//!
//! The corpus is decompressed to `f32` for the online clustering (routing,
//! 2-means splits, and neighborhood reassignment all run in full precision),
//! but the inverted lists on disk are written from the original corpus rows in
//! their on-disk element type, selected by `--format` (`minmax8` default, `f16`,
//! or `f32`) — identical on-disk format to the static builds, so the result
//! loads and searches through the same `sweep` tooling.
//!
//! The clusterer is seeded with a lightweight k-means over the first
//! `warmup_points` corpus points, then every point is streamed in insert order.
//! Per-split telemetry (centroid-count growth, reassignment counts, and split
//! latencies over the whole build) is written to a CSV for offline analysis.
//!
//! Run:
//! ```text
//! cargo run --release --example build_online -- \
//!     <corpus.bin> <out_prefix> --split-threshold <n> [options]
//!
//! # ~16384 clusters (th=106), s=5 reassignment neighbors, f16 lists
//! cargo run --release --example build_online -- \
//!     corpus_f16.bin out_prefix --split-threshold 106 \
//!     --reassign-neighbors 5 --max-clusters 16384 --format f16
//! ```
//!
//! The online index takes **no target cluster count**: it splits whenever a
//! cluster exceeds `split_threshold` and keeps doing so for every inserted
//! point, so the final cluster count emerges from the data and threshold. The
//! centroid graph is pre-allocated eagerly, so a hard id budget is still needed;
//! it is auto-sized to `capacity_mult * 2 * num_points / split_threshold` (the
//! natural equilibrium is `~ 2 * num_points / split_threshold` live clusters,
//! and each split retires one id), generous enough that it never binds.
//!
//! Writes `<out_prefix>_th<split_threshold>_<format>.graphivf_{lists,meta,centroids.fbin}`
//! and `<out_prefix>_th<split_threshold>_<format>.splits.csv`.

use std::{fs::File, path::Path, time::Instant};

use bytemuck::Pod;
use diskann::utils::VectorRepr;
use diskann_graphivf::{GraphParams, Half, Metric, OnlineClusterer, OnlineParams, SeedStrategy};
use diskann_providers::common::MinMaxElement;
use diskann_utils::{io::read_bin, views::Matrix};

const GRAPH_DEGREE: usize = 32;
const GRAPH_SLACK: f32 = 1.2;
const GRAPH_L_BUILD: usize = 64;
const GRAPH_ALPHA: f32 = 1.2;
const SEED: u64 = 0;

const USAGE: &str = "\
usage: build_online <corpus.bin> <out_prefix> --split-threshold <n> [options]

options:
  --split-threshold <n>     split a cluster once it exceeds this many points (required)
  --warmup-centroids <n>    initial centroids from a light k-means over a prefix  [default 100]
  --warmup-points <n>       leading corpus points used for the warmup             [default 10000]
  --threads <n>             build worker threads                                  [default 16]
  --assign-l <n>            centroid-graph search-list size for routing inserts   [default 64]
  --two-means-iters <n>     Lloyd iterations per split 2-means                    [default 12]
  --metric <l2|ip>          stored/search metric (cluster/assign always L2)       [default l2]
  --normalize               L2-normalize child centroids after a split            [default off]
  --capacity-mult <n>       centroid id-budget headroom                           [default 3]
  --reassign-neighbors <n>  s nearest clusters pooled for reassignment on split   [default 8]
  --reassign-l <n>          search-list size selecting those s neighbors    [default max(s,assign_l)]
  --max-clusters <n>        0 = uncapped growth; else hard cap on live clusters   [default 0]
  --format <fmt>            stored element type: minmax8 | f16 | f32              [default minmax8]";

/// On-disk element type of the corpus rows and the inverted lists.
#[derive(Clone, Copy)]
enum Format {
    /// 8-bit MinMax-quantized rows (`MinMaxElement<8>`).
    MinMax8,
    /// IEEE binary16 (`f16`) rows (`Half`).
    F16,
    /// IEEE binary32 (`f32`) rows.
    F32,
}

impl Format {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "minmax8" | "mm8" => Ok(Format::MinMax8),
            "f16" | "fp16" | "half" => Ok(Format::F16),
            "f32" | "fp32" | "float" => Ok(Format::F32),
            other => Err(format!(
                "unknown --format {other:?} (expected minmax8|f16|f32)"
            )),
        }
    }

    /// Suffix appended to the output prefix (e.g. `_minmax8`).
    fn suffix(&self) -> &'static str {
        match self {
            Format::MinMax8 => "minmax8",
            Format::F16 => "f16",
            Format::F32 => "f32",
        }
    }
}

/// Parsed knobs shared by every element type; corpus-size-derived values (such
/// as the centroid id budget) are computed inside [`run`].
struct Config {
    corpus_path: String,
    out_prefix: String,
    split_threshold: usize,
    warmup_centroids: usize,
    warmup_points: usize,
    num_threads: usize,
    assign_l: usize,
    two_means_iters: usize,
    metric: Metric,
    normalize: bool,
    capacity_mult: usize,
    reassign_neighbors: usize,
    reassign_l: usize,
    max_clusters: Option<usize>,
    fmt: &'static str,
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    match s {
        "l2" | "L2" => Ok(Metric::L2),
        "ip" | "IP" | "inner_product" => Ok(Metric::InnerProduct),
        other => Err(format!("unknown --metric {other:?} (expected l2 or ip)")),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut it = std::env::args().skip(1);
    let corpus_path = it.next().ok_or(USAGE)?;
    let out_prefix = it.next().ok_or(USAGE)?;

    let mut split_threshold: Option<usize> = None;
    let mut warmup_centroids: usize = 100;
    let mut warmup_points: usize = 10_000;
    let mut num_threads: usize = 16;
    let mut assign_l: usize = 64;
    let mut two_means_iters: usize = 12;
    let mut metric = Metric::L2;
    let mut normalize = false;
    let mut capacity_mult: usize = 3;
    let mut reassign_neighbors: usize = 8;
    let mut reassign_l: Option<usize> = None;
    let mut max_clusters_arg: usize = 0;
    let mut format = Format::MinMax8;

    let parse_num = |s: Option<String>, flag: &str| -> Result<usize, String> {
        s.ok_or_else(|| format!("{flag} requires a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--split-threshold" => {
                split_threshold = Some(parse_num(it.next(), "--split-threshold")?)
            }
            "--warmup-centroids" => warmup_centroids = parse_num(it.next(), "--warmup-centroids")?,
            "--warmup-points" => warmup_points = parse_num(it.next(), "--warmup-points")?,
            "--threads" => num_threads = parse_num(it.next(), "--threads")?,
            "--assign-l" => assign_l = parse_num(it.next(), "--assign-l")?,
            "--two-means-iters" => two_means_iters = parse_num(it.next(), "--two-means-iters")?,
            "--metric" => metric = parse_metric(&it.next().ok_or("--metric requires a value")?)?,
            "--normalize" => normalize = true,
            "--capacity-mult" => capacity_mult = parse_num(it.next(), "--capacity-mult")?,
            "--reassign-neighbors" => {
                reassign_neighbors = parse_num(it.next(), "--reassign-neighbors")?
            }
            "--reassign-l" => reassign_l = Some(parse_num(it.next(), "--reassign-l")?),
            "--max-clusters" => max_clusters_arg = parse_num(it.next(), "--max-clusters")?,
            "--format" => format = Format::parse(&it.next().ok_or("--format requires a value")?)?,
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}").into()),
        }
    }

    let split_threshold = split_threshold.ok_or("--split-threshold is required")?;
    let reassign_l = reassign_l.unwrap_or(reassign_neighbors.max(assign_l));
    // 0 = uncapped (data-driven growth); otherwise cap live clusters at this many.
    let max_clusters = (max_clusters_arg != 0).then_some(max_clusters_arg);

    let cfg = Config {
        corpus_path,
        out_prefix,
        split_threshold,
        warmup_centroids,
        warmup_points,
        num_threads,
        assign_l,
        two_means_iters,
        metric,
        normalize,
        capacity_mult,
        reassign_neighbors,
        reassign_l,
        max_clusters,
        fmt: format.suffix(),
    };

    match format {
        Format::MinMax8 => run::<MinMaxElement<8>>(cfg),
        Format::F16 => run::<Half>(cfg),
        Format::F32 => run::<f32>(cfg),
    }
}

/// Run the online build storing rows in element type `T` (selected by `--format`).
fn run<T: VectorRepr + Pod>(cfg: Config) -> Result<(), Box<dyn std::error::Error>> {
    let Config {
        corpus_path,
        out_prefix,
        split_threshold,
        warmup_centroids,
        warmup_points,
        num_threads,
        assign_l,
        two_means_iters,
        metric,
        normalize,
        capacity_mult,
        reassign_neighbors,
        reassign_l,
        max_clusters,
        fmt,
    } = cfg;

    let prefix = format!("{out_prefix}_th{split_threshold}_{fmt}");
    let csv_path = format!("{prefix}.splits.csv");

    // Load the corpus. These rows are stored verbatim in the inverted lists.
    let corpus: Matrix<T> = read_bin(&mut File::open(&corpus_path)?)?;
    let num_points = corpus.nrows();
    let stored_ncols = corpus.ncols();
    println!("corpus ({fmt}): {num_points} x {stored_ncols} rows  {corpus_path}");

    // Decompress to f32 for clustering (identity for f32; MinMax8/f16 decode).
    let decomp_start = Instant::now();
    let dim = T::full_dimension(corpus.row(0))?;
    let mut f32_buf = vec![0.0f32; num_points * dim];
    for (src, dst) in corpus
        .as_slice()
        .chunks(stored_ncols)
        .zip(f32_buf.chunks_mut(dim))
    {
        T::as_f32_into(src, dst)?;
    }
    let points: Matrix<f32> = Matrix::try_from(f32_buf.into_boxed_slice(), num_points, dim)?;
    println!(
        "decompressed to f32: {num_points} x {dim}  ({:.1} s)",
        decomp_start.elapsed().as_secs_f64()
    );

    // Online build id budget. The centroid graph is pre-allocated eagerly, so
    // size a generous id budget. For an uncapped build the equilibrium is ~2N/T
    // live clusters (each split retires one id, so ~4N/T ids are consumed);
    // `capacity_mult` adds headroom. A capped build never exceeds `max_clusters`
    // live clusters, so `~2*max_clusters` ids suffice — take the max of both so
    // the budget binds in neither case.
    let centroid_capacity = ((capacity_mult * 2 * num_points) / split_threshold.max(1))
        .max(2 * warmup_centroids)
        .max(warmup_centroids + 1)
        .max(max_clusters.map_or(0, |m| 2 * m + warmup_centroids));
    println!("centroid id budget: {centroid_capacity} (capacity_mult={capacity_mult})");

    let params = OnlineParams {
        max_clusters,
        centroid_capacity,
        split_threshold,
        assign_l,
        reassign_neighbors,
        reassign_l,
        two_means_iters,
        graph: GraphParams {
            degree: GRAPH_DEGREE,
            slack: GRAPH_SLACK,
            l_build: GRAPH_L_BUILD,
            alpha: GRAPH_ALPHA,
        },
        metric,
        normalize_centroids: normalize,
        num_threads,
        seed: SEED,
    };

    let seed = SeedStrategy::Warmup {
        num_centroids: warmup_centroids,
        warmup_points,
        iters: 15,
    };

    let cap_desc = match max_clusters {
        Some(m) => format!("capped at {m} clusters"),
        None => "uncapped".to_string(),
    };
    println!(
        "online build: {cap_desc} (split_threshold={split_threshold}), \
         warmup={warmup_centroids} centroids over {warmup_points} points, \
         assign_l={assign_l}, reassign_neighbors={reassign_neighbors}, reassign_l={reassign_l}, \
         two_means_iters={two_means_iters}, \
         {num_threads} threads, metric={metric:?}, normalize={normalize} \
         (cluster/assign always L2)..."
    );

    let build_start = Instant::now();
    let mut clusterer = OnlineClusterer::with_seed(points, seed, params)?;
    println!("seeded with {} centroids", clusterer.num_clusters());

    // Stream every point in corpus order.
    let insert_start = Instant::now();
    let report_every = (num_points / 20).max(1);
    for pid in 0..num_points as u32 {
        clusterer.insert(pid)?;
        if (pid as usize + 1).is_multiple_of(report_every) {
            let t = clusterer.telemetry();
            println!(
                "  inserted {:>9}/{num_points}  clusters={:>6}  splits={:>6}  \
                 reassigned={:>10}",
                pid + 1,
                clusterer.num_clusters(),
                t.total_splits,
                t.total_reassigned,
            );
        }
    }
    let insert_elapsed = insert_start.elapsed();

    // Write the index using the stored `T` rows verbatim for the inverted lists.
    let flush_start = Instant::now();
    clusterer.flush::<T>(Path::new(&prefix), corpus.as_view())?;
    let flush_elapsed = flush_start.elapsed();

    // Persist the per-split telemetry timeline.
    let t = clusterer.telemetry();
    t.write_csv(Path::new(&csv_path))?;

    let sizes = clusterer.cluster_sizes();
    let (min_sz, max_sz) = sizes
        .iter()
        .fold((usize::MAX, 0usize), |(lo, hi), &s| (lo.min(s), hi.max(s)));
    let mean_sz = num_points as f64 / sizes.len().max(1) as f64;

    println!("\n=== online build summary ===");
    println!(
        "total wall-clock:   {:.1} s",
        build_start.elapsed().as_secs_f64()
    );
    println!("  insertion:        {:.1} s", insert_elapsed.as_secs_f64());
    println!("  flush:            {:.1} s", flush_elapsed.as_secs_f64());
    println!("routing time:       {:.1} s", t.routing_us as f64 / 1e6);
    println!("split time:         {:.1} s", t.split_us as f64 / 1e6);
    println!("final clusters:     {}", clusterer.num_clusters());
    println!("total splits:       {}", t.total_splits);
    println!("total reassigned:   {}", t.total_reassigned);
    println!("cluster sizes:      min={min_sz} mean={mean_sz:.1} max={max_sz}");
    println!("residual:           {:.3e}", clusterer.residual());
    println!("wrote index to      {prefix}.graphivf_{{lists,meta,centroids.fbin}}");
    println!("wrote split log to  {csv_path}");

    Ok(())
}
