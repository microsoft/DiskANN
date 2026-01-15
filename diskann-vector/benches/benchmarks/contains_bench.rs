/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */
use criterion::{black_box, Criterion};
use diskann_vector::contains::ContainsSimd;
use rand::{distr::Uniform, prelude::Distribution, rngs::StdRng, Rng, SeedableRng};

pub(crate) fn benchmark_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains");
    let data1 = get_random_content(73);
    let data2 = get_random_content(31);
    let data3 = get_random_content(97);

    group.bench_function("Contains [Vectorized]", |f| {
        f.iter(|| {
            black_box(u32::contains_simd(black_box(&data1), 101));
            black_box(u32::contains_simd(black_box(&data2), 101));
            black_box(u32::contains_simd(black_box(&data3), 101))
        })
    });
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
