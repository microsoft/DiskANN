//! Isolated leaf-build + HashPrune-insert benchmark.
//!
//! Runs partition + LSH-init ONCE (untimed), then loops the leaf+HP phase N
//! times under `cargo run --release --example bench_leaf_hp_isolated`.
//! Reports per-iteration wall + total edges. No search, no save, no PQ.
//!
//! Use with perf:
//!   perf record -F 199 -g -D 100 -e cycles -- \
//!     ./target/release/examples/bench_leaf_hp_isolated <fp16.bin> [npoints] [iters]
//!
//! The `-D 100` skips the dataset load and untimed partition phases since
//! the bench prints "BEGIN_LEAF_HP_LOOP" marker just before the timed loop.

use std::env;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use diskann_pipnn::builder::bench_leaf_hp_phase;
use diskann_pipnn::hash_prune::HashPrune;
use diskann_pipnn::partition::{partition, PartitionConfig};
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

fn main() {
    let _ = tracing_subscriber::fmt().try_init();
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        "/home/weiyaoluo/datasets/bigann/bigann_10m_fp16.bin".to_string()
    });
    let limit_points = args.get(2).and_then(|s| s.parse::<usize>().ok());
    let iters = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(1);
    let num_threads = std::env::var("PIPNN_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16usize);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("rayon pool");

    let (data, npoints, ndims) = read_fp16_dataset(&path, limit_points);
    let metric = Metric::L2;

    // Config from configs/scaling/t16.json (BigANN 10M PiPNN R=64)
    let c_max: usize = std::env::var("PIPNN_C_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let c_min: usize = std::env::var("PIPNN_C_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let p_samp: f64 = std::env::var("PIPNN_P_SAMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.005);
    let fanout: Vec<usize> = vec![10, 3];
    let k: usize = std::env::var("PIPNN_K").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let num_hash_planes: usize = std::env::var("PIPNN_HP").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let l_max: usize = std::env::var("PIPNN_L_MAX").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_degree: usize = std::env::var("PIPNN_R").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let leader_cap: usize = 1000; // static; no env override

    println!(
        "config: c_max={} c_min={} p_samp={} fanout={:?} k={} hp={} l_max={} R={} T={}",
        c_max, c_min, p_samp, fanout, k, num_hash_planes, l_max, max_degree, num_threads
    );

    pool.install(|| {
        // ----- SETUP (untimed-for-leaf-HP) -----
        println!("--- SETUP: HashPrune init (LSH sketches) ---");
        let t = Instant::now();
        let hash_prune = HashPrune::new(
            &data, npoints, ndims, num_hash_planes, l_max, max_degree, 42,
        );
        println!("  hp_init wall: {:.3}s", t.elapsed().as_secs_f64());

        println!("--- SETUP: partition ---");
        let t = Instant::now();
        let partition_config = PartitionConfig::new(
            c_max, c_min, p_samp, fanout, metric, leader_cap,
        ).expect("partition config");
        let leaves = partition(&data, ndims, npoints, &partition_config, 1000u64);
        let part_wall = t.elapsed().as_secs_f64();
        let total_pts: usize = leaves.iter().map(|l| l.indices.len()).sum();
        println!(
            "  partition wall: {:.3}s  num_leaves={}  avg_leaf={:.1}",
            part_wall,
            leaves.len(),
            total_pts as f64 / leaves.len() as f64
        );

        // Marker line — used by `perf record -D ...` to align timestamp if desired,
        // though here we still measure from main start; the loop output is what's timed.
        println!("BEGIN_LEAF_HP_LOOP iters={}", iters);

        for it in 0..iters {
            let (wall, edges) =
                bench_leaf_hp_phase(&data, ndims, &leaves, &hash_prune, k, metric);
            println!(
                "  iter {}/{}: leaf_hp_wall={:.3}s  total_edges={}",
                it + 1,
                iters,
                wall,
                edges
            );
        }
        println!("END_LEAF_HP_LOOP");
    });
}
