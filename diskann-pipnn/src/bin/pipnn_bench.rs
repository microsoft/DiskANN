/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! PiPNN Benchmark Binary
//!
//! Loads a dataset in .bin/.fbin format, builds an index using PiPNN,
//! evaluates recall, and reports build times.
//!
//! Usage:
//!   pipnn-bench --data <path.fbin> [--queries <path.fbin>] [--groundtruth <path.bin>]
//!               [--k <knn>] [--max-degree <R>] [--c-max <leaf_size>]
//!               [--replicas <num>] [--search-l <list_size>]

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use rand::SeedableRng;

use diskann_pipnn::builder;
use diskann_pipnn::leaf_build::brute_force_knn;
use diskann_pipnn::PiPNNConfig;

/// PiPNN Benchmark: build and evaluate ANN index using PiPNN algorithm.
#[derive(Parser, Debug)]
#[command(name = "pipnn-bench")]
#[command(about = "Build and evaluate PiPNN ANN index")]
struct Args {
    /// Path to the data file (.fbin format: [npoints:u32][ndims:u32][data:f32...]).
    /// Not required when using --synthetic.
    #[arg(long)]
    data: Option<PathBuf>,

    /// Path to the query file (.fbin format). If not provided, random queries are generated.
    #[arg(long)]
    queries: Option<PathBuf>,

    /// Path to the groundtruth file (.bin format: [nqueries:u32][k:u32][ids:u32...]).
    /// If not provided, brute-force groundtruth is computed.
    #[arg(long)]
    groundtruth: Option<PathBuf>,

    /// Number of nearest neighbors to find.
    #[arg(long, default_value = "10")]
    k: usize,

    /// Maximum graph degree (R).
    #[arg(long, default_value = "64")]
    max_degree: usize,

    /// Maximum leaf size (C_max).
    #[arg(long, default_value = "1024")]
    c_max: usize,

    /// Minimum cluster size (C_min). Defaults to c_max / 4.
    #[arg(long)]
    c_min: Option<usize>,

    /// k-NN within each leaf.
    #[arg(long, default_value = "3")]
    leaf_k: usize,

    /// Number of partitioning replicas.
    #[arg(long, default_value = "2")]
    replicas: usize,

    /// Number of LSH hyperplanes for HashPrune.
    #[arg(long, default_value = "12")]
    num_hash_planes: usize,

    /// Maximum reservoir size per node in HashPrune.
    #[arg(long, default_value = "128")]
    l_max: usize,

    /// Search list size (L).
    #[arg(long, default_value = "100")]
    search_l: usize,

    /// Number of random queries if no query file is provided.
    #[arg(long, default_value = "100")]
    num_queries: usize,

    /// Apply final RobustPrune pass.
    #[arg(long, default_value = "false")]
    final_prune: bool,

    /// Use synthetic data with this many points (ignores --data).
    #[arg(long)]
    synthetic: Option<usize>,

    /// Dimensions for synthetic data.
    #[arg(long, default_value = "128")]
    synthetic_dims: usize,

    /// Fanout sequence (comma-separated, e.g. "10,3").
    #[arg(long, default_value = "10,3")]
    fanout: String,

    /// Sampling fraction for RBC leaders.
    #[arg(long, default_value = "0.05")]
    p_samp: f64,

    /// Force fp16 interpretation of input files.
    #[arg(long)]
    fp16: bool,

    /// Use cosine distance (dot product on normalized vectors) instead of L2.
    #[arg(long)]
    cosine: bool,

    /// Save the built index in DiskANN format at this path prefix.
    /// Creates <prefix> (graph) and <prefix>.data (vectors).
    /// Can then be loaded by diskann-benchmark with index-source=Load.
    #[arg(long)]
    save_path: Option<PathBuf>,
}

/// Read a binary matrix file as f32.
/// Supports both f32 (.fbin) and fp16 (.bin) formats.
/// For fp16, auto-detects by checking if file size matches fp16 layout.
fn read_bin_matrix(path: &PathBuf, force_fp16: bool) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    let npoints = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let ndims = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let num_elements = npoints * ndims;

    let file_size = std::fs::metadata(path)?.len() as usize;
    let is_fp16 = force_fp16 || file_size == 8 + num_elements * 2;

    let data = if is_fp16 {
        // Read as fp16 and convert to f32
        let mut raw = vec![0u8; num_elements * 2];
        file.read_exact(&mut raw)?;
        let fp16_data: &[u16] = bytemuck::cast_slice(&raw);
        fp16_data.iter().map(|&bits| half::f16::from_bits(bits).to_f32()).collect()
    } else {
        let mut data = vec![0.0f32; num_elements];
        let byte_slice = bytemuck::cast_slice_mut::<f32, u8>(&mut data);
        file.read_exact(byte_slice)?;
        data
    };

    println!("Loaded {}: {} points x {} dims ({})", path.display(), npoints, ndims,
        if is_fp16 { "fp16->f32" } else { "f32" });
    Ok((data, npoints, ndims))
}

/// Read a groundtruth file: [nqueries:u32 LE][k:u32 LE][ids: nqueries*k u32 LE].
fn read_groundtruth(path: &PathBuf) -> Result<(Vec<Vec<u32>>, usize), Box<dyn std::error::Error>> {
    let mut file = BufReader::new(File::open(path)?);

    let mut header = [0u8; 8];
    file.read_exact(&mut header)?;

    let nqueries = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut ids = vec![0u32; nqueries * k];
    let byte_slice = bytemuck::cast_slice_mut::<u32, u8>(&mut ids);
    file.read_exact(byte_slice)?;

    let groundtruth: Vec<Vec<u32>> = (0..nqueries)
        .map(|i| ids[i * k..(i + 1) * k].to_vec())
        .collect();

    println!("Loaded groundtruth: {} queries x {} neighbors", nqueries, k);
    Ok((groundtruth, k))
}

/// Generate random data for synthetic benchmarks.
fn generate_synthetic(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..npoints * ndims)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

/// Compute recall@k.
fn compute_recall(
    approx_results: &[(usize, f32)],
    groundtruth: &[usize],
    k: usize,
) -> f64 {
    let gt_set: std::collections::HashSet<usize> =
        groundtruth.iter().take(k).copied().collect();
    let found = approx_results
        .iter()
        .take(k)
        .filter(|&&(id, _)| gt_set.contains(&id))
        .count();
    found as f64 / k as f64
}

/// Save PiPNN graph in DiskANN canonical graph format.
///
/// Graph file layout (matches diskann-providers/src/storage/bin.rs save_graph):
///   Header (24 bytes):
///     - u64 LE: total file size (header + data)
///     - u32 LE: max degree (observed)
///     - u32 LE: start point ID (medoid)
///     - u64 LE: number of additional/frozen points (0)
///   Per node (in order 0..npoints):
///     - u32 LE: number of neighbors L
///     - L x u32 LE: neighbor IDs
///
/// No data file is written — the original data file on disk is used directly.
fn save_diskann_graph(
    graph: &builder::PiPNNGraph,
    prefix: &PathBuf,
    start_point: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{Write, Seek, SeekFrom};

    let mut f = std::io::BufWriter::new(File::create(prefix)?);

    // Write placeholder header (will update file_size and max_degree at the end).
    let mut index_size: u64 = 24;
    let mut observed_max_degree: u32 = 0;

    f.write_all(&index_size.to_le_bytes())?;       // placeholder file_size
    f.write_all(&observed_max_degree.to_le_bytes())?; // placeholder max_degree
    f.write_all(&start_point.to_le_bytes())?;
    let num_additional: u64 = 1; // 1 frozen/start point (the medoid)
    f.write_all(&num_additional.to_le_bytes())?;

    // Write per-node adjacency lists.
    for adj in &graph.adjacency {
        let num_neighbors = adj.len() as u32;
        f.write_all(&num_neighbors.to_le_bytes())?;
        for &neighbor in adj {
            f.write_all(&neighbor.to_le_bytes())?;
        }
        observed_max_degree = observed_max_degree.max(num_neighbors);
        index_size += (4 + adj.len() * 4) as u64;
    }

    // Seek back and write correct file_size and max_degree.
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&index_size.to_le_bytes())?;
    f.write_all(&observed_max_degree.to_le_bytes())?;
    f.flush()?;

    println!("  Saved graph: {} ({} nodes, max_degree={}, start={})",
        prefix.display(), graph.npoints, observed_max_degree, start_point);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Force OpenBLAS to single-threaded mode since rayon handles outer parallelism.
    std::env::set_var("OPENBLAS_NUM_THREADS", "1");

    let args = Args::parse();

    // Parse fanout.
    let fanout: Vec<usize> = args
        .fanout
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<Vec<_>, _>>()?;

    // Load or generate data.
    let (data, npoints, ndims) = if let Some(n) = args.synthetic {
        println!("Generating synthetic data: {} points x {} dims", n, args.synthetic_dims);
        let data = generate_synthetic(n, args.synthetic_dims, 42);
        (data, n, args.synthetic_dims)
    } else if let Some(ref data_path) = args.data {
        read_bin_matrix(data_path, args.fp16)?
    } else {
        return Err("Either --data or --synthetic must be specified".into());
    };

    // Build PiPNN index.
    let c_min = args.c_min.unwrap_or(args.c_max / 4);

    let metric = if args.cosine {
        diskann_vector::distance::Metric::CosineNormalized
    } else {
        diskann_vector::distance::Metric::L2
    };

    let config = PiPNNConfig {
        num_hash_planes: args.num_hash_planes,
        c_max: args.c_max,
        c_min,
        p_samp: args.p_samp,
        fanout,
        k: args.leaf_k,
        max_degree: args.max_degree,
        replicas: args.replicas,
        l_max: args.l_max,
        final_prune: args.final_prune,
        metric,
    };

    println!("\n=== PiPNN Build ===");
    let build_start = Instant::now();
    let graph = builder::build(&data, npoints, ndims, &config);
    let build_time = build_start.elapsed();
    println!("Build time: {:.3}s", build_time.as_secs_f64());
    println!(
        "Graph stats: avg_degree={:.1}, max_degree={}, isolated={}",
        graph.avg_degree(),
        graph.max_degree(),
        graph.num_isolated()
    );

    // Save graph in DiskANN format if requested.
    if let Some(ref save_path) = args.save_path {
        println!("\nSaving graph to DiskANN format at {:?}...", save_path);
        let save_start = Instant::now();
        save_diskann_graph(&graph, save_path, graph.medoid as u32)?;
        println!("Saved in {:.3}s", save_start.elapsed().as_secs_f64());
    }

    // Load or generate queries.
    let (queries, num_queries, _query_dims) = if let Some(ref qpath) = args.queries {
        let (q, nq, qd) = read_bin_matrix(qpath, args.fp16)?;
        assert_eq!(qd, ndims, "query dims {} != data dims {}", qd, ndims);
        (q, nq, qd)
    } else {
        let nq = args.num_queries;
        println!("\nGenerating {} random queries...", nq);
        let q = generate_synthetic(nq, ndims, 999);
        (q, nq, ndims)
    };

    // Load or compute groundtruth.
    let groundtruth: Vec<Vec<usize>> = if let Some(ref gtpath) = args.groundtruth {
        let (gt, _gt_k) = read_groundtruth(gtpath)?;
        gt.into_iter()
            .map(|ids| ids.into_iter().map(|id| id as usize).collect())
            .collect()
    } else {
        println!("Computing brute-force groundtruth...");
        let gt_start = Instant::now();
        let gt: Vec<Vec<usize>> = (0..num_queries)
            .map(|qi| {
                let query = &queries[qi * ndims..(qi + 1) * ndims];
                brute_force_knn(&data, ndims, npoints, query, args.k)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();
        println!("Groundtruth computed in {:.3}s", gt_start.elapsed().as_secs_f64());
        gt
    };

    // Evaluate recall at multiple search_l values.
    println!("\n=== Search Evaluation ===");
    let search_ls = [50, 100, 200, 500];

    for &search_l in &search_ls {
        let search_start = Instant::now();
        let mut total_recall = 0.0;

        for qi in 0..num_queries {
            let query = &queries[qi * ndims..(qi + 1) * ndims];
            let results = graph.search(&data, query, args.k, search_l);
            let recall = compute_recall(&results, &groundtruth[qi], args.k);
            total_recall += recall;
        }

        let search_time = search_start.elapsed();
        let avg_recall = total_recall / num_queries as f64;
        let qps = num_queries as f64 / search_time.as_secs_f64();

        println!(
            "  L={:<4}  recall@{}={:.4}  QPS={:.0}  time={:.3}s",
            search_l, args.k, avg_recall, qps, search_time.as_secs_f64()
        );
    }

    println!("\n=== Summary ===");
    println!("Points: {}", npoints);
    println!("Dimensions: {}", ndims);
    println!("Build time: {:.3}s", build_time.as_secs_f64());
    println!("Avg degree: {:.1}", graph.avg_degree());
    println!("Max degree: {}", graph.max_degree());
    println!("Isolated nodes: {}", graph.num_isolated());

    Ok(())
}
