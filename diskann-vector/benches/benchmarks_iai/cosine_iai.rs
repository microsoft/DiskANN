/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_vector::{distance::Cosine, DistanceFunction};
use half::f16;
use iai_callgrind::black_box;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardUniform};

use crate::utils::{AlignedDataset, AlignedVector, DatasetArgs, InlineBarrier, Static};
iai_callgrind::library_benchmark_group!(
    name = cosine_i8_iai;
    benchmarks = cosine_i8_v2,
);

iai_callgrind::library_benchmark_group!(
    name = cosine_u8_iai;
    benchmarks = cosine_u8_v2,
);

iai_callgrind::library_benchmark_group!(
    name = cosine_f32_iai;
    benchmarks = cosine_f32_v2,
);

iai_callgrind::library_benchmark_group!(
    name = cosine_f16_iai;
    benchmarks = cosine_f16_v2,
);
fn cosine_i8_common() -> (AlignedDataset<i8>, AlignedVector<i8>, Vec<f32>) {
    const DIM: usize = 128;
    let dim = Static::<DIM>;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<i8> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::<i8>::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = cosine_i8_common)]
fn cosine_i8_v2(data: (AlignedDataset<i8>, AlignedVector<i8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let query_sized: &[i8; 128] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(Cosine {});
    dataset
        .iter_sized(Static::<128>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] = func.evaluate_similarity(query_sized, v);
        });

    black_box(output);
}

fn cosine_u8_common() -> (AlignedDataset<u8>, AlignedVector<u8>, Vec<f32>) {
    const DIM: usize = 128;
    let dim = Static::<DIM>;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<u8> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::<u8>::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = cosine_u8_common)]
fn cosine_u8_v2(data: (AlignedDataset<u8>, AlignedVector<u8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let query_sized: &[u8; 128] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(Cosine {});
    dataset
        .iter_sized(Static::<128>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] = func.evaluate_similarity(query_sized, v);
        });

    black_box(output);
}

fn cosine_f32_common() -> (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>) {
    const DIM: usize = 104;
    let dim = Static::<DIM>;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = Normal::<f32>::new(-1.0, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<f32> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = cosine_f32_common)]
fn cosine_f32_v2(data: (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let query_sized: &[f32; 104] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(Cosine {});
    dataset
        .iter_sized(Static::<104>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] = func.evaluate_similarity(query_sized, v);
        });

    black_box(output);
}

fn cosine_f16_common() -> (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>) {
    const DIM: usize = 384;
    let dim = Static::<DIM>;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(384).unwrap(),
        NonZeroUsize::new(40).unwrap(),
    );

    let dist = Normal::<f16>::new(f16::from_f32(-1.0), f16::from_f32(1.0)).unwrap();
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<f16> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = cosine_f16_common)]
fn cosine_f16_v2(data: (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let func = InlineBarrier::new(Cosine {});
    let query_sized: &[f16; 384] = query.as_slice().try_into().unwrap();
    dataset
        .iter_sized(Static::<384>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] = func.evaluate_similarity(query_sized, v);
        });

    black_box(output);
}
