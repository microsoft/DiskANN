/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Refine an existing set of centroids with one (or more) Lloyd iterations over
//! the **entire** corpus, then build a fresh graph + inverted lists from the
//! refined centroids.
//!
//! This seeds Lloyd's with the centroids written by a previous build (which used
//! a *sample* for k-means) and runs `--iters` full-corpus iterations, so the new
//! centroids are the exact corpus means of the seed partition's neighborhoods.
//! Everything is written to a new prefix so the original index is untouched.
//!
//! Run (defaults refine the 40960 index's centroids over the enron corpus):
//! ```text
//! cargo run --release --example refine_centroids_full_corpus -- \
//!     [corpus_fp16.bin] [seed_centroids.fbin] [out_prefix] [iters]
//! ```

use std::{fs::File, path::Path};

use diskann_graphivf::{BuildParams, GraphIvfIndex, GraphParams, Half, Metric};
use diskann_utils::{
    io::read_bin,
    views::{Matrix, MatrixView},
};

const DEFAULT_CORPUS: &str =
    "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4/normalized_dim_384_vector_fp16_1087932_vectors.bin";
const DEFAULT_SEED_CENTROIDS: &str =
    "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4/graphivf_index_40960.graphivf_centroids.fbin";
const DEFAULT_OUT_PREFIX: &str =
    "C:/Users/adkrishnan/Projects/data/enron-email-1M-fbv4/graphivf_index_40960_full";

const NUM_CLUSTERS: usize = 40960;
const ASSIGN_L: usize = 32;
const GRAPH_DEGREE: usize = 32;
const GRAPH_SLACK: f32 = 1.2;
const GRAPH_L_BUILD: usize = 64;
const GRAPH_ALPHA: f32 = 1.2;
const NUM_THREADS: usize = 8;
const SEED: u64 = 0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let corpus_path = args.next().unwrap_or_else(|| DEFAULT_CORPUS.to_string());
    let seed_path = args
        .next()
        .unwrap_or_else(|| DEFAULT_SEED_CENTROIDS.to_string());
    let out_prefix = args
        .next()
        .unwrap_or_else(|| DEFAULT_OUT_PREFIX.to_string());
    let iters: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(1);

    // --- Load corpus (fp16 -> f32) -------------------------------------------
    let corpus_u16: Matrix<u16> = read_bin(&mut File::open(&corpus_path)?)?;
    let num_points = corpus_u16.nrows();
    let dim = corpus_u16.ncols();
    let corpus_f32: Vec<f32> = corpus_u16
        .as_slice()
        .iter()
        .map(|&bits| Half::from_bits(bits).to_f32())
        .collect();
    let corpus = Matrix::try_from(corpus_f32.into_boxed_slice(), num_points, dim)
        .map_err(|_| "corpus matrix shape mismatch")?;
    println!("corpus:    {num_points} x {dim}  ({corpus_path})");

    // --- Load seed centroids (f32) -------------------------------------------
    let seed: Matrix<f32> = read_bin(&mut File::open(&seed_path)?)?;
    println!(
        "seeds:     {} x {}  ({seed_path})",
        seed.nrows(),
        seed.ncols()
    );

    let params = BuildParams {
        num_clusters: NUM_CLUSTERS,
        metric: Metric::L2,
        // Ignored by the seeded build path; set to the corpus size to satisfy
        // `sample_size >= num_clusters` validation.
        sample_size: num_points,
        kmeans_iters: iters,
        assign_l: ASSIGN_L,
        graph: GraphParams {
            degree: GRAPH_DEGREE,
            slack: GRAPH_SLACK,
            l_build: GRAPH_L_BUILD,
            alpha: GRAPH_ALPHA,
        },
        num_threads: NUM_THREADS,
        seed: SEED,
    };

    println!(
        "refining {NUM_CLUSTERS} centroids with {iters} Lloyd iter(s) over the full corpus...\n"
    );
    let corpus_view: MatrixView<'_, f32> = corpus.as_view();
    let seed_view: MatrixView<'_, f32> = seed.as_view();
    let (_index, profile) = GraphIvfIndex::<Half>::build_from_seed_centroids_profiled(
        corpus_view,
        seed_view,
        &params,
        Path::new(&out_prefix),
    )?;

    println!("{profile}");
    println!("wrote index to {out_prefix}.graphivf_{{lists,meta,centroids.fbin}}");
    Ok(())
}
