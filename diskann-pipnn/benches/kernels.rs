/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use diskann_linalg::{sgemm, sgemm_aat_lower, Transpose};
use diskann_pipnn::{
    leaf_kernel::{nearest_leaf_neighbors, LeafNeighbor, LeafTopK, LeafTopKWorkspace},
    partition_kernel::{nearest_leaders, PartitionTopK},
};
use diskann_vector::distance::Metric;

const BIGANN_DIMENSIONS: usize = 128;
const PARTITION_FANOUT: usize = 10;
const LEAF_K: usize = 2;
const LEAF_SIZES: [usize; 3] = [64, 256, 512];
const METRICS: [Metric; 4] = [
    Metric::L2,
    Metric::Cosine,
    Metric::CosineNormalized,
    Metric::InnerProduct,
];

fn fixed_data(rows: usize, columns: usize, sequence: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| {
            let value = index
                .wrapping_mul(1_664_525)
                .wrapping_add(sequence.wrapping_mul(1_013_904_223))
                % 2_003;
            (value as f32 - 1_001.0) / 1_001.0
        })
        .collect()
}

fn normalize_rows(data: &mut [f32], columns: usize) {
    for row in data.chunks_exact_mut(columns) {
        let inverse_norm = row
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .recip();
        row.iter_mut().for_each(|value| *value *= inverse_norm);
    }
}

fn lower_dots(points: usize, metric: Metric) -> Vec<f32> {
    let mut data = fixed_data(points, BIGANN_DIMENSIONS, points);
    if metric == Metric::CosineNormalized {
        normalize_rows(&mut data, BIGANN_DIMENSIONS);
    }
    let mut dots = vec![0.0; points * points];
    sgemm_aat_lower(&data, points, BIGANN_DIMENSIONS, &mut dots).unwrap();
    dots
}

fn benchmark_partition_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipnn/partition-topk");
    for (rows, leaders) in [(1_024, 64), (512, 256), (128, 1_000)] {
        let points = fixed_data(rows, BIGANN_DIMENSIONS, rows);
        let leader_data = fixed_data(leaders, BIGANN_DIMENSIONS, leaders);
        let mut dots = vec![0.0; rows * leaders];
        sgemm(
            Transpose::None,
            Transpose::Ordinary,
            rows,
            leaders,
            BIGANN_DIMENSIONS,
            1.0,
            &points,
            &leader_data,
            None,
            &mut dots,
        )
        .unwrap();
        let leader_scales = leader_data
            .chunks_exact(BIGANN_DIMENSIONS)
            .map(|row| row.iter().map(|value| value * value).sum())
            .collect::<Vec<_>>();
        let input = PartitionTopK {
            dots: &dots,
            rows,
            leaders,
            row_scales: &[],
            leader_scales: &leader_scales,
            metric: Metric::L2,
        };
        let mut output = vec![0; rows * PARTITION_FANOUT];

        group.throughput(Throughput::Elements(rows as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "l2",
                format!("{BIGANN_DIMENSIONS}d/{rows}x{leaders}/k{PARTITION_FANOUT}"),
            ),
            &input,
            |bencher, input| {
                bencher.iter(|| {
                    nearest_leaders(*input, PARTITION_FANOUT, &mut output).unwrap();
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_lower_aat(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipnn/lower-aat");
    for points in LEAF_SIZES {
        let data = fixed_data(points, BIGANN_DIMENSIONS, points);
        let mut dots = vec![0.0; points * points];

        group.throughput(Throughput::Elements((points * (points + 1) / 2) as u64));
        group.bench_function(
            BenchmarkId::new("f32", format!("{points}x{BIGANN_DIMENSIONS}")),
            |bencher| {
                bencher.iter(|| {
                    sgemm_aat_lower(&data, points, BIGANN_DIMENSIONS, &mut dots).unwrap();
                    black_box(&dots);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_leaf_topk(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipnn/leaf-topk");
    for points in LEAF_SIZES {
        for metric in METRICS {
            let dots = lower_dots(points, metric);
            let input = LeafTopK {
                dots: &dots,
                points,
                metric,
            };
            let mut output = vec![LeafNeighbor::default(); points * LEAF_K];
            let mut workspace = LeafTopKWorkspace::new();
            nearest_leaf_neighbors(input, LEAF_K, &mut output, &mut workspace).unwrap();

            group.throughput(Throughput::Elements((points * (points - 1) / 2) as u64));
            group.bench_with_input(
                BenchmarkId::new(metric.as_str(), format!("{points}/k{LEAF_K}")),
                &input,
                |bencher, input| {
                    bencher.iter(|| {
                        nearest_leaf_neighbors(*input, LEAF_K, &mut output, &mut workspace)
                            .unwrap();
                        black_box(&output);
                    });
                },
            );
        }
    }
    group.finish();
}

fn benchmark_full_leaf(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipnn/full-leaf-numerical");
    for points in LEAF_SIZES {
        let data = fixed_data(points, BIGANN_DIMENSIONS, points);
        let mut dots = vec![0.0; points * points];
        let mut output = vec![LeafNeighbor::default(); points * LEAF_K];
        let mut workspace = LeafTopKWorkspace::new();
        sgemm_aat_lower(&data, points, BIGANN_DIMENSIONS, &mut dots).unwrap();
        nearest_leaf_neighbors(
            LeafTopK {
                dots: &dots,
                points,
                metric: Metric::L2,
            },
            LEAF_K,
            &mut output,
            &mut workspace,
        )
        .unwrap();

        group.throughput(Throughput::Elements(points as u64));
        group.bench_function(
            BenchmarkId::new("l2", format!("{points}x{BIGANN_DIMENSIONS}/k{LEAF_K}")),
            |bencher| {
                bencher.iter(|| {
                    sgemm_aat_lower(&data, points, BIGANN_DIMENSIONS, &mut dots).unwrap();
                    nearest_leaf_neighbors(
                        LeafTopK {
                            dots: &dots,
                            points,
                            metric: Metric::L2,
                        },
                        LEAF_K,
                        &mut output,
                        &mut workspace,
                    )
                    .unwrap();
                    black_box(&output);
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets =
        benchmark_partition_topk,
        benchmark_lower_aat,
        benchmark_leaf_topk,
        benchmark_full_leaf
}
criterion_main!(benches);
