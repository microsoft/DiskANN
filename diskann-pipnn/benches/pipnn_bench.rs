/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Criterion benchmarks for PiPNN hot-path components.
//!
//! Run with: cargo bench -p diskann-pipnn

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};

use diskann_pipnn::gemm;
use diskann_pipnn::hash_prune::HashPrune;
use diskann_pipnn::leaf_build;
use diskann_pipnn::partition::{self, PartitionConfig};
use diskann_pipnn::quantize;
use diskann_vector::distance::Metric;

/// Generate random f32 data for benchmarking.
fn random_data(npoints: usize, ndims: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..npoints * ndims)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}

// ==================
// GEMM benchmarks
// ==================

fn bench_sgemm_aat(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/sgemm_aat");

    for &(m, k) in &[(256, 128), (512, 128), (1024, 128), (512, 384)] {
        let a = random_data(m, k, 42);
        let mut result = vec![0.0f32; m * m];

        group.throughput(Throughput::Elements((m * m) as u64));
        group.bench_with_input(
            BenchmarkId::new("m_x_k", format!("{}x{}", m, k)),
            &(m, k),
            |b, &(m, k)| {
                b.iter(|| {
                    gemm::sgemm_aat(&a, m, k, &mut result);
                });
            },
        );
    }
    group.finish();
}

fn bench_sgemm_abt(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm/sgemm_abt");

    for &(m, n, k) in &[(1000, 100, 128), (1000, 100, 384), (10000, 100, 128)] {
        let a = random_data(m, k, 42);
        let b = random_data(n, k, 99);
        let mut result = vec![0.0f32; m * n];

        group.throughput(Throughput::Elements((m * n) as u64));
        group.bench_with_input(
            BenchmarkId::new("m_n_k", format!("{}x{}x{}", m, n, k)),
            &(m, n, k),
            |b_iter, &(m, n, k)| {
                b_iter.iter(|| {
                    gemm::sgemm_abt(&a, m, k, &b, n, &mut result);
                });
            },
        );
    }
    group.finish();
}

// ========================
// Quantization benchmarks
// ========================

/// Train SQ parameters and quantize. Benchmark helper.
fn train_and_quantize(data: &[f32], npoints: usize, ndims: usize) -> quantize::QuantizedData {
    use diskann_quantization::scalar::train::ScalarQuantizationParameters;
    use diskann_utils::views::MatrixView;

    let data_matrix = MatrixView::try_from(data, npoints, ndims)
        .expect("data length must equal npoints * ndims");
    let quantizer = ScalarQuantizationParameters::default().train(data_matrix);
    let shift = quantizer.shift().to_vec();
    let scale = quantizer.scale();
    let inverse_scale = if scale == 0.0 { 1.0 } else { 1.0 / scale };
    quantize::quantize_1bit(data, npoints, ndims, &shift, inverse_scale)
}

fn bench_hamming_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize/hamming_matrix");

    for &n in &[64, 256, 512, 1024] {
        let ndims = 128;
        let data = random_data(n, ndims, 42);
        let qd = train_and_quantize(&data, n, ndims);
        let indices: Vec<usize> = (0..n).collect();

        group.throughput(Throughput::Elements((n * n) as u64));
        group.bench_with_input(
            BenchmarkId::new("n_points", n),
            &n,
            |b, _| {
                b.iter(|| {
                    qd.compute_distance_matrix(&indices);
                });
            },
        );
    }
    group.finish();
}

// ========================
// Leaf build benchmarks
// ========================

fn bench_build_leaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("leaf_build/build_leaf");

    for &(n, ndims, k) in &[(128, 128, 3), (512, 128, 4), (1024, 128, 4), (512, 384, 5)] {
        let data = random_data(n, ndims, 42);
        let indices: Vec<usize> = (0..n).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("n_d_k", format!("{}x{}x{}", n, ndims, k)),
            &(),
            |b, _| {
                b.iter(|| {
                    leaf_build::build_leaf(&data, ndims, &indices, k, Metric::L2);
                });
            },
        );
    }
    group.finish();
}

fn bench_build_leaf_quantized(c: &mut Criterion) {
    let mut group = c.benchmark_group("leaf_build/build_leaf_quantized");

    for &(n, ndims, k) in &[(128, 128, 3), (512, 128, 4), (1024, 128, 4)] {
        let data = random_data(n, ndims, 42);
        let qd = train_and_quantize(&data, n, ndims);
        let indices: Vec<usize> = (0..n).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("n_d_k", format!("{}x{}x{}", n, ndims, k)),
            &(),
            |b, _| {
                b.iter(|| {
                    leaf_build::build_leaf_quantized(&qd, &indices, k);
                });
            },
        );
    }
    group.finish();
}

// ========================
// HashPrune benchmarks
// ========================

fn bench_hash_prune_add_edges(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_prune/add_edges_batched");

    for &npoints in &[10_000, 100_000] {
        let ndims = 128;
        let data = random_data(npoints, ndims, 42);
        let hp = HashPrune::new(&data, npoints, ndims, 12, 128, 64, 42);

        // Simulate edges from a single leaf
        let leaf_size = 512;
        let k = 4;
        let leaf_data = random_data(leaf_size, ndims, 99);
        let leaf_indices: Vec<usize> = (0..leaf_size).collect();
        let edges = leaf_build::build_leaf(&leaf_data, ndims, &leaf_indices, k, Metric::L2);

        group.throughput(Throughput::Elements(edges.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("npoints", npoints),
            &(),
            |b, _| {
                b.iter(|| {
                    hp.add_edges_batched(&edges);
                });
            },
        );
    }
    group.finish();
}

// ==========================
// Partition benchmarks
// ==========================

fn bench_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition/parallel_partition");
    group.sample_size(10);

    for &(npoints, ndims) in &[(10_000, 128), (50_000, 128), (10_000, 384)] {
        let data = random_data(npoints, ndims, 42);
        let indices: Vec<usize> = (0..npoints).collect();
        let config = PartitionConfig {
            c_max: 1024,
            c_min: 256,
            p_samp: 0.05,
            fanout: vec![8],
            metric: Metric::L2,
        };

        group.throughput(Throughput::Elements(npoints as u64));
        group.bench_with_input(
            BenchmarkId::new("n_d", format!("{}x{}", npoints, ndims)),
            &(),
            |b, _| {
                b.iter(|| {
                    partition::parallel_partition(&data, ndims, &indices, &config, 42);
                });
            },
        );
    }
    group.finish();
}

// ==========================
// End-to-end build benchmark
// ==========================

fn bench_full_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build/full");
    group.sample_size(10);

    for &(npoints, ndims) in &[(1_000, 128), (10_000, 128), (10_000, 384)] {
        let data = random_data(npoints, ndims, 42);
        let config = diskann_pipnn::PiPNNConfig {
            c_max: 512,
            c_min: 128,
            k: 3,
            max_degree: 32,
            replicas: 1,
            l_max: 64,
            p_samp: 0.05,
            fanout: vec![8],
            ..Default::default()
        };

        group.throughput(Throughput::Elements(npoints as u64));
        group.bench_with_input(
            BenchmarkId::new("n_d", format!("{}x{}", npoints, ndims)),
            &(),
            |b, _| {
                b.iter(|| {
                    diskann_pipnn::builder::build(&data, npoints, ndims, &config).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    gemm_benches,
    bench_sgemm_aat,
    bench_sgemm_abt,
);

criterion_group!(
    quantize_benches,
    bench_hamming_distance_matrix,
);

criterion_group!(
    leaf_benches,
    bench_build_leaf,
    bench_build_leaf_quantized,
);

criterion_group!(
    hash_prune_benches,
    bench_hash_prune_add_edges,
);

criterion_group!(
    partition_benches,
    bench_partition,
);

criterion_group!(
    build_benches,
    bench_full_build,
);

criterion_main!(
    gemm_benches,
    quantize_benches,
    leaf_benches,
    hash_prune_benches,
    partition_benches,
    build_benches,
);
