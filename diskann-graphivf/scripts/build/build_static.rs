/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Unified **static** graph-IVF build.
//!
//! One harness for every centroid-seeding strategy and stored element type; the
//! strategy, the on-disk `--format`, and all build parameters are supplied as
//! command-line flags. Clustering and point→centroid assignment are **always
//! squared-L2** in full precision (the corpus is decoded to `f32` for k-means);
//! `--metric ip` only sets the metric recorded in the index (used to navigate
//! the centroid graph and score results at search time).
//!
//! `--format` selects the element type stored in the inverted lists:
//! `minmax8` (8-bit MinMax quantized, default), `f16`, or `f32`. The corpus
//! `.bin` must already be in that element type.
//!
//! # Seeding strategies (`--seed`)
//!
//! | Strategy      | Centroids come from                                            | Requires        |
//! |---------------|----------------------------------------------------------------|-----------------|
//! | `sampled`     | Forgy k-means over a random corpus sample                      | `--sample-size` |
//! | `random`      | exactly `--clusters` corpus rows drawn uniformly at random     | —               |
//! | `precomputed` | an existing centroid `fbin`, reused verbatim (`--iters` forced 0)| `--centroids`   |
//! | `forgy-f32`   | Forgy init drawn from a *separate* f32 corpus (memory-frugal)  | `--init-corpus` |
//!
//! `forgy-f32` streams `--clusters` random rows out of the f32 corpus without
//! loading it whole, so a huge corpus (e.g. 10M x 768 f32 ≈ 30 GB) can seed a
//! build whose working set is just the MinMax8 corpus plus one f32 scratch copy.
//!
//! # Example
//!
//! ```text
//! # 16384-cluster sampled build, 10 Lloyd iters, inner-product search metric
//! cargo run --release --example build_static -- \
//!     corpus_minmax8.bin out_prefix --seed sampled --clusters 16384 \
//!     --sample-size 200000 --iters 10 --threads 16 --metric ip
//!
//! # Unit-sphere build (former caselaw recipe): random init, 4 iters, normalized
//! cargo run --release --example build_static -- \
//!     corpus_minmax8.bin out_prefix --seed random --clusters 16384 \
//!     --iters 4 --normalize --threads 16
//! ```
//!
//! Writes `<out_prefix>_<clusters>_<format>.graphivf_{lists,meta,centroids.fbin}`.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use bytemuck::Pod;
use diskann_graphivf::{
    AssignMethod, BuildParams, CentroidInit, EmptyClusterPolicy, GraphIvfIndex, GraphParams, Half,
    Metric, VectorRepr,
};
use diskann_providers::common::MinMaxElement;
use diskann_utils::{
    io::{read_bin, write_bin},
    views::{Matrix, MatrixView},
};
use rand::{rngs::StdRng, SeedableRng};

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

const ASSIGN_L: usize = 32;
const GRAPH_DEGREE: usize = 32;
const GRAPH_SLACK: f32 = 1.2;
const GRAPH_L_BUILD: usize = 64;
const GRAPH_ALPHA: f32 = 1.2;

/// `--assign auto`: exact scan below this cluster count, graph assigner at or above.
const AUTO_GRAPH_ASSIGN_THRESHOLD: usize = 16384;

const USAGE: &str = "\
usage: build_static <corpus.bin> <out_prefix> --seed <strategy> --clusters <k> [options]

seeding strategies (--seed):
  sampled       Forgy k-means over a random corpus sample   (requires --sample-size)
  random        RandomUniform: exactly <k> random corpus rows
  precomputed   reuse a centroid fbin verbatim              (requires --centroids; forces --iters 0)
  forgy-f32     Forgy init drawn from a separate f32 corpus (requires --init-corpus)

options:
  --clusters <k>        number of centroids (required)
  --format <fmt>        stored element type: minmax8 | f16 | f32        [default minmax8]
  --iters <n>           Lloyd refinement iterations                     [default 0]
  --sample-size <n>     sample rows for `sampled`                       [default: corpus size]
  --centroids <path>    centroid fbin for `precomputed`
  --init-corpus <path>  f32 corpus for `forgy-f32` seeding
  --threads <n>         build worker threads                            [default 16]
  --assign <mode>       auto | exact | graph                            [default auto]
  --rebuild-every <n>   graph-assigner centroid-graph rebuild cadence   [default 1]
  --rerank <n>          graph-assigner exact re-rank depth              [default 8]
  --metric <l2|ip>      stored/search metric (cluster/assign always L2) [default l2]
  --normalize           L2-normalize centroids after each Lloyd iter
  --rng-seed <n>        RNG seed                                        [default 0]";

/// Which centroid-initialization strategy to run.
enum Seeding {
    Sampled,
    Random,
    Precomputed,
    ForgyF32,
}

impl Seeding {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "sampled" => Ok(Seeding::Sampled),
            "random" => Ok(Seeding::Random),
            "precomputed" => Ok(Seeding::Precomputed),
            "forgy-f32" | "forgy_f32" => Ok(Seeding::ForgyF32),
            other => Err(format!(
                "unknown --seed {other:?} (expected sampled|random|precomputed|forgy-f32)"
            )),
        }
    }
}

/// How to assign points to centroids during and after k-means.
enum Assign {
    Auto,
    Exact,
    Graph,
}

impl Assign {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Assign::Auto),
            "exact" => Ok(Assign::Exact),
            "graph" => Ok(Assign::Graph),
            other => Err(format!(
                "unknown --assign {other:?} (expected auto|exact|graph)"
            )),
        }
    }

    /// Resolve to a concrete [`AssignMethod`] for `num_clusters`.
    fn resolve(&self, num_clusters: usize, rebuild_every: usize, rerank: usize) -> AssignMethod {
        let graph = AssignMethod::Graph {
            rebuild_every,
            rerank,
        };
        match self {
            Assign::Exact => AssignMethod::Exact,
            Assign::Graph => graph,
            Assign::Auto if num_clusters >= AUTO_GRAPH_ASSIGN_THRESHOLD => graph,
            Assign::Auto => AssignMethod::Exact,
        }
    }
}

/// Parsed command line.
struct Args {
    corpus_path: String,
    out_prefix: String,
    format: Format,
    seeding: Seeding,
    num_clusters: usize,
    kmeans_iters: usize,
    sample_size: Option<usize>,
    centroids_path: Option<String>,
    init_corpus_path: Option<String>,
    num_threads: usize,
    assign: Assign,
    rebuild_every: usize,
    rerank: usize,
    metric: Metric,
    normalize: bool,
    rng_seed: u64,
}

fn parse_metric(s: &str) -> Result<Metric, String> {
    match s {
        "l2" | "L2" => Ok(Metric::L2),
        "ip" | "IP" | "inner_product" => Ok(Metric::InnerProduct),
        other => Err(format!("unknown --metric {other:?} (expected l2 or ip)")),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let corpus_path = it.next().ok_or(USAGE)?;
    let out_prefix = it.next().ok_or(USAGE)?;

    let mut seeding: Option<Seeding> = None;
    let mut format = Format::MinMax8;
    let mut num_clusters: Option<usize> = None;
    let mut kmeans_iters: usize = 0;
    let mut sample_size: Option<usize> = None;
    let mut centroids_path: Option<String> = None;
    let mut init_corpus_path: Option<String> = None;
    let mut num_threads: usize = 16;
    let mut assign = Assign::Auto;
    let mut rebuild_every: usize = 1;
    let mut rerank: usize = 8;
    let mut metric = Metric::L2;
    let mut normalize = false;
    let mut rng_seed: u64 = 0;

    let parse_num = |s: Option<String>, flag: &str| -> Result<usize, String> {
        s.ok_or_else(|| format!("{flag} requires a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };

    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--seed" => {
                seeding = Some(Seeding::parse(
                    &it.next().ok_or("--seed requires a value")?,
                )?)
            }
            "--clusters" => num_clusters = Some(parse_num(it.next(), "--clusters")?),
            "--format" => format = Format::parse(&it.next().ok_or("--format requires a value")?)?,
            "--iters" => kmeans_iters = parse_num(it.next(), "--iters")?,
            "--sample-size" => sample_size = Some(parse_num(it.next(), "--sample-size")?),
            "--centroids" => {
                centroids_path = Some(it.next().ok_or("--centroids requires a value")?)
            }
            "--init-corpus" => {
                init_corpus_path = Some(it.next().ok_or("--init-corpus requires a value")?)
            }
            "--threads" => num_threads = parse_num(it.next(), "--threads")?,
            "--assign" => assign = Assign::parse(&it.next().ok_or("--assign requires a value")?)?,
            "--rebuild-every" => rebuild_every = parse_num(it.next(), "--rebuild-every")?,
            "--rerank" => rerank = parse_num(it.next(), "--rerank")?,
            "--metric" => metric = parse_metric(&it.next().ok_or("--metric requires a value")?)?,
            "--normalize" => normalize = true,
            "--rng-seed" => rng_seed = parse_num(it.next(), "--rng-seed")? as u64,
            other => return Err(format!("unknown flag {other:?}\n\n{USAGE}")),
        }
    }

    Ok(Args {
        corpus_path,
        out_prefix,
        format,
        seeding: seeding.ok_or("--seed is required")?,
        num_clusters: num_clusters.ok_or("--clusters is required")?,
        kmeans_iters,
        sample_size,
        centroids_path,
        init_corpus_path,
        num_threads,
        assign,
        rebuild_every,
        rerank,
        metric,
        normalize,
        rng_seed,
    })
}

/// Read the `(num_points, dim)` header of an fbin file (two little-endian u32s).
fn read_fbin_header(path: &str) -> std::io::Result<(usize, usize)> {
    let mut f = File::open(path)?;
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr)?;
    let n = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    Ok((n, d))
}

/// Read the given row indices from an f32 fbin file via seeks, returning a
/// row-major `rows.len() x dim` matrix without loading the whole corpus.
fn read_rows_f32(path: &str, dim: usize, rows: &[usize]) -> std::io::Result<Matrix<f32>> {
    let mut f = File::open(path)?;
    let mut buf = vec![0.0f32; rows.len() * dim];
    let mut bytes = vec![0u8; dim * 4];
    for (dst, &r) in buf.chunks_mut(dim).zip(rows.iter()) {
        f.seek(SeekFrom::Start(8 + (r * dim * 4) as u64))?;
        f.read_exact(&mut bytes)?;
        for (d, chunk) in dst.iter_mut().zip(bytes.chunks_exact(4)) {
            *d = f32::from_le_bytes(chunk.try_into().unwrap());
        }
    }
    Matrix::try_from(buf.into_boxed_slice(), rows.len(), dim)
        .map_err(|_| std::io::Error::other("init centroid shape"))
}

/// Draw `num_clusters` random rows from the f32 corpus at `init_corpus_path` and
/// write them to `centroids_path` (the build then reuses them via `Precomputed`).
fn seed_from_f32_corpus(
    init_corpus_path: &str,
    centroids_path: &str,
    num_clusters: usize,
    rng_seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let (n_f32, dim) = read_fbin_header(init_corpus_path)?;
    println!("init corpus (f32 header): {n_f32} x {dim}  {init_corpus_path}");
    if num_clusters > n_f32 {
        return Err(format!("--clusters {num_clusters} > init-corpus rows {n_f32}").into());
    }
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut idx = rand::seq::index::sample(&mut rng, n_f32, num_clusters).into_vec();
    idx.sort_unstable(); // sequential seeks are far faster than random ones
    println!("drawing {num_clusters} Forgy init centroids from the f32 corpus...");
    let init = read_rows_f32(init_corpus_path, dim, &idx)?;
    let mut f = File::create(centroids_path)?;
    write_bin(init.as_view(), &mut f)?;
    println!("wrote init centroids to {centroids_path}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    match args.format {
        Format::MinMax8 => run::<MinMaxElement<8>>(args),
        Format::F16 => run::<Half>(args),
        Format::F32 => run::<f32>(args),
    }
}

/// Build the index storing rows in element type `T` (selected by `--format`).
fn run<T: VectorRepr + Pod>(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let fmt = args.format.suffix();
    let prefix = format!("{}_{}_{fmt}", args.out_prefix, args.num_clusters);
    let centroids_out = format!("{prefix}.graphivf_centroids.fbin");

    let corpus: Matrix<T> = read_bin(&mut File::open(&args.corpus_path)?)?;
    println!(
        "corpus ({fmt}): {} x {} rows  {}",
        corpus.nrows(),
        corpus.ncols(),
        args.corpus_path
    );

    // Resolve the seeding strategy to a `CentroidInit` and effective iterations.
    // `precomputed`/`forgy-f32` both drive the build via `CentroidInit::Precomputed`
    // (the latter after materializing init centroids into `centroids_out`).
    let mut kmeans_iters = args.kmeans_iters;
    let init: CentroidInit<'_> = match args.seeding {
        Seeding::Sampled => {
            let samples = args.sample_size.unwrap_or(corpus.nrows());
            CentroidInit::Forgy { samples }
        }
        Seeding::Random => CentroidInit::RandomUniform,
        Seeding::Precomputed => {
            kmeans_iters = 0; // reuse supplied centroids verbatim
            let path = args
                .centroids_path
                .as_deref()
                .ok_or("--seed precomputed requires --centroids <path>")?;
            CentroidInit::Precomputed {
                path: Path::new(path),
            }
        }
        Seeding::ForgyF32 => {
            let init_corpus = args
                .init_corpus_path
                .as_deref()
                .ok_or("--seed forgy-f32 requires --init-corpus <f32.bin>")?;
            seed_from_f32_corpus(
                init_corpus,
                &centroids_out,
                args.num_clusters,
                args.rng_seed,
            )?;
            CentroidInit::Precomputed {
                path: Path::new(&centroids_out),
            }
        }
    };

    // Forgy samples a subset; every other strategy clusters the full corpus.
    let sample_size = match args.seeding {
        Seeding::Sampled => args.sample_size.unwrap_or(corpus.nrows()),
        _ => corpus.nrows(),
    };
    let assign_method = args
        .assign
        .resolve(args.num_clusters, args.rebuild_every, args.rerank);

    let params = BuildParams {
        num_clusters: args.num_clusters,
        metric: args.metric,
        sample_size,
        kmeans_iters,
        assign_l: ASSIGN_L,
        graph: GraphParams {
            degree: GRAPH_DEGREE,
            slack: GRAPH_SLACK,
            l_build: GRAPH_L_BUILD,
            alpha: GRAPH_ALPHA,
        },
        num_threads: args.num_threads,
        seed: args.rng_seed,
        assign_method,
        empty_clusters: EmptyClusterPolicy::PreserveOld,
        normalize_centroids: args.normalize,
    };

    println!(
        "building {fmt} index: {} clusters, {kmeans_iters} Lloyd iters, assign={assign_method:?}, \
         normalize={}, {} threads, metric={:?} (cluster/assign always L2)...",
        args.num_clusters, args.normalize, args.num_threads, args.metric
    );
    let view: MatrixView<'_, T> = corpus.as_view();
    let (_index, profile) =
        GraphIvfIndex::<T>::build_compressed_profiled(view, init, &params, Path::new(&prefix))?;
    println!("{profile}");
    println!("wrote {fmt} index to {prefix}.graphivf_{{lists,meta,centroids.fbin}}");
    Ok(())
}
