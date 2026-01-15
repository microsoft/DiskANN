/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use std::time::Duration;

use criterion::{BenchmarkGroup, BenchmarkId, Criterion, black_box, measurement::Measurement};
use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

fn bench_npq_for_size<T: Measurement>(group: &mut BenchmarkGroup<T>, size: usize) {
    group.bench_with_input(BenchmarkId::new("insert", size), &size, |f, &i| {
        let data = generate_neighbors(100_000);
        let mut queue = NeighborPriorityQueue::new(i);
        f.iter(|| {
            queue.clear();
            for n in black_box(data.iter()) {
                queue.insert(*n);
            }
            black_box(&1);
        });
    });
}

pub fn benchmark_priority_queue_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighborqueue");
    group
        .measurement_time(Duration::from_secs(10))
        .sample_size(500);

    bench_npq_for_size(&mut group, 10);
    bench_npq_for_size(&mut group, 20);
    bench_npq_for_size(&mut group, 50);
    bench_npq_for_size(&mut group, 100);
    bench_npq_for_size(&mut group, 200);
    bench_npq_for_size(&mut group, 500);
    bench_npq_for_size(&mut group, 1000)
}

pub fn benchmark_priority_queue_has_notvisited_node(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighborqueue");
    group
        .measurement_time(Duration::from_secs(3))
        .sample_size(10000);

    let queue: NeighborPriorityQueue<u32> = NeighborPriorityQueue::new(64_usize);
    group.bench_function("Neighbor Priority Queue has_notvisited_node", |f| {
        f.iter(|| {
            for _ in black_box(0..100) {
                queue.has_notvisited_node();
            }
        });
    });
}

fn generate_neighbors(count: usize) -> Vec<Neighbor<u32>> {
    let seed: [u8; 32] = [73; 32];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Uniform::new(0.0, 1.0).unwrap();
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let distance = range.sample(&mut rng) as f32;
        let n = Neighbor::new(i as u32, distance);
        result.push(n);
    }

    result
}
