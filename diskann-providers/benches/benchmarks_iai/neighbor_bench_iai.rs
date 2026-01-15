/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use iai_callgrind::black_box;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

fn bench_npq_for_size_iai(size: usize, data: Vec<Neighbor<u32>>) {
    let mut queue = NeighborPriorityQueue::new(size);
    queue.clear();
    for n in black_box(data.iter()) {
        queue.insert(*n);
    }
    black_box(&1);
}

iai_callgrind::library_benchmark_group!(
    name = priority_queue_insert_bench_iai;
    benchmarks = benchmark_priority_queue_insert_iai,benchmark_priority_queue_has_notvisited_node_iai
);

#[iai_callgrind::library_benchmark]
#[bench::with_setup_0(generate_neighbors(100_000), 10)]
#[bench::with_setup_1(generate_neighbors(100_000), 20)]
#[bench::with_setup_2(generate_neighbors(100_000), 50)]
#[bench::with_setup_3(generate_neighbors(100_000), 100)]
#[bench::with_setup_4(generate_neighbors(100_000), 200)]
#[bench::with_setup_5(generate_neighbors(100_000), 500)]
#[bench::with_setup_6(generate_neighbors(100_000), 1000)]
pub fn benchmark_priority_queue_insert_iai(data: Vec<Neighbor<u32>>, i: usize) {
    bench_npq_for_size_iai(i, data);
}

#[iai_callgrind::library_benchmark]
pub fn benchmark_priority_queue_has_notvisited_node_iai() {
    let queue: NeighborPriorityQueue<u32> = NeighborPriorityQueue::new(64_usize);

    for _ in black_box(0..100) {
        queue.has_notvisited_node();
    }
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
