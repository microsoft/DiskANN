/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use diskann_vector::contains::ContainsSimd;
use iai_callgrind::black_box;
use rand::{distr::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

iai_callgrind::library_benchmark_group!(
    name = benchmark_contains_bench_iai;
    benchmarks = benchmark_contains,
);

type DataTuple = (Box<[u32]>, Box<[u32]>, Box<[u32]>);

#[iai_callgrind::library_benchmark(setup = setup_data)]
pub(crate) fn benchmark_contains(data: DataTuple) {
    let (data1, data2, data3) = data;

    black_box(u32::contains_simd(black_box(&data1), 101));
    black_box(u32::contains_simd(black_box(&data2), 101));
    black_box(u32::contains_simd(black_box(&data3), 101));
}

fn setup_data() -> DataTuple {
    let data1: Box<[u32]> = get_random_content(73);
    let data2 = get_random_content(31);
    let data3 = get_random_content(97);

    (data1, data2, data3)
}

fn get_random_content(seed: u8) -> Box<[u32]> {
    let seed: [u8; 32] = [seed; 32]; // Constant seed of a random number.
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let range = Uniform::new(0, 100).unwrap();
    let len = rng.random_range(0..100);

    (0..len)
        .map(|_| range.sample(&mut rng))
        .collect::<Vec<u32>>()
        .into_boxed_slice()
}
