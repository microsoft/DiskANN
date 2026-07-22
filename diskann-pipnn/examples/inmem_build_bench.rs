/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! In-memory PiPNN build benchmark for BigANN 10M.
//!
//! Skips disk index / PQ / search — times the pure in-memory PiPNN build call,
//! using the fixed PiPNN A/B configuration.
//!
//! Usage:
//!   cargo run --release -p diskann-pipnn --example inmem_build_bench -- \
//!     <path-to-fp16.bin> [npoints] [skip-hash-prune]

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::{num::NonZeroUsize, time::Instant};

use diskann_pipnn::{builder, PiPNNBuildContext, PiPNNConfig};
use diskann_vector::distance::Metric;
use half::f16;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn read_fp16_dataset(path: &str, limit_points: Option<usize>) -> (Vec<f16>, usize, usize) {
    let mut file = File::open(path).expect("open dataset");
    let mut hdr = [0u8; 8];
    file.read_exact(&mut hdr).expect("read header");
    let npoints_file = u32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let ndims = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let npoints = limit_points.unwrap_or(npoints_file).min(npoints_file);
    let nelems = npoints * ndims;
    let bytes = nelems * 2;

    let mut data = vec![f16::from_f32(0.0); nelems];
    file.seek(SeekFrom::Start(8)).unwrap();
    let buf: &mut [u8] = bytemuck::must_cast_slice_mut(&mut data);
    file.read_exact(&mut buf[..bytes]).expect("read body");

    println!(
        "loaded {}/{} points × {} dims ({} MB)",
        npoints,
        npoints_file,
        ndims,
        bytes >> 20
    );
    (data, npoints, ndims)
}

fn read_rss_kb() -> u64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            return rest.split_whitespace().next().unwrap().parse().unwrap_or(0);
        }
    }
    0
}

fn main() {
    let _ = tracing_subscriber::fmt().try_init();

    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "datasets/bigann_10m_fp16.bin".to_string());
    let limit_points = args.get(2).and_then(|s| s.parse::<usize>().ok());
    let skip_hash_prune = args.get(3).is_some_and(|arg| arg == "skip-hash-prune");

    let (data, npoints, ndims) = read_fp16_dataset(&path, limit_points);

    // Fixed A/B configuration.
    let config = PiPNNConfig {
        num_hash_planes: 14,
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
        l_max: 72,
        final_prune: true,
        skip_hash_prune,
    };
    let ctx = PiPNNBuildContext::new(
        config.clone(),
        NonZeroUsize::new(64).unwrap(),
        1.2,
        Metric::L2,
        16,
    )
    .expect("config");
    println!("config: {:?}", config);
    let rss_pre = read_rss_kb();
    println!("RSS before build: {} MB", rss_pre / 1024);

    let t0 = Instant::now();
    let graph = builder::build_typed(&data, npoints, ndims, &ctx).expect("build");
    let dt = t0.elapsed();

    let rss_post = read_rss_kb();
    let avg_degree = graph.iter().map(Vec::len).sum::<usize>() as f64 / graph.len() as f64;
    println!(
        "BUILD_WALL={:.3}s  RSS_post={} MB  npoints={}  avg_degree={:.2}",
        dt.as_secs_f64(),
        rss_post / 1024,
        graph.len(),
        avg_degree
    );
}
