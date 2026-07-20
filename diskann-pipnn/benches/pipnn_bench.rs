/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Criterion benchmarks for PiPNN hot-path components.
//!
//! Run with: cargo bench -p diskann-pipnn

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};

use diskann_linalg as gemm;

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
            replicas: 1,
            l_max: 64,
            p_samp: 0.05,
            fanout: vec![8],
            ..Default::default()
        };
        let ctx = diskann_pipnn::PiPNNBuildContext::new(
            config,
            std::num::NonZeroUsize::new(32).unwrap(),
            1.2,
            diskann_vector::distance::Metric::L2,
            0,
        )
        .unwrap();

        group.throughput(Throughput::Elements(npoints as u64));
        group.bench_with_input(
            BenchmarkId::new("n_d", format!("{}x{}", npoints, ndims)),
            &(),
            |b, _| {
                b.iter(|| {
                    diskann_pipnn::builder::build_typed::<f32>(&data, npoints, ndims, &ctx)
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(gemm_benches, bench_sgemm_aat, bench_sgemm_abt,);

criterion_group!(build_benches, bench_full_build,);

criterion_main!(gemm_benches, build_benches,);
