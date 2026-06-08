//! In-memory PiPNN build benchmark for BigANN 10M.
//!
//! Skips disk index / PQ / search — times the pure in-memory PiPNN build call,
//! matching the config from `examples/bigann_10m_pipnn.json`.
//!
//! Usage:
//!   cargo run --release -p diskann-pipnn --example inmem_build_bench -- \
//!     <path-to-fp16.bin> [npoints]

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use diskann_pipnn::PiPNNConfig;
use diskann_pipnn::builder::PiPNNBuilder;
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
            return rest.trim().split_whitespace().next().unwrap().parse().unwrap_or(0);
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

    let (data, npoints, ndims) = read_fp16_dataset(&path, limit_points);

    // Exact config from diskann-pipnn/examples/bigann_10m_pipnn.json:
    let config = PiPNNConfig {
        num_hash_planes: 12,
        c_max: 256,
        c_min: 16,
        p_samp: 0.005,
        fanout: vec![10, 3],
        k: 3,
        max_degree: 64,
        replicas: 1,
        l_max: 64,
        metric: Metric::L2,
        final_prune: false,
        alpha: 1.2,
        num_threads: 16,
        leader_cap: 1000,
        saturate_after_prune: false,
    };
    println!("config: {:?}", config);
    let rss_pre = read_rss_kb();
    println!("RSS before build: {} MB", rss_pre / 1024);

    let t0 = Instant::now();
    let graph = PiPNNBuilder::new(config)
        .build_typed(&data, npoints, ndims)
        .expect("build");
    let dt = t0.elapsed();

    let rss_post = read_rss_kb();
    println!(
        "BUILD_WALL={:.3}s  RSS_peak~{} MB  npoints={}  avg_degree={:.2}",
        dt.as_secs_f64(),
        rss_post / 1024,
        graph.npoints,
        graph.avg_degree()
    );
    println!("stats: {:?}", graph.build_stats);
}
