/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use criterion::black_box;
use diskann_vector::distance::SquaredL2;
use half::f16;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardUniform};

use crate::utils::{AlignedDataset, AlignedVector, DatasetArgs, InlineBarrier, Static};

iai_callgrind::library_benchmark_group!(
    name = l2_f32_iai;
    benchmarks = benchmark_f32_v2, benchmark_f32_v2_sized,
);

iai_callgrind::library_benchmark_group!(
    name = l2_f16_iai;
    benchmarks = benchmark_f16_v2, benchmark_f16_v2_sized,
);

iai_callgrind::library_benchmark_group!(
    name = l2_i8_iai;
    benchmarks = benchmark_i8_v2, benchmark_i8_v2_sized,
);

iai_callgrind::library_benchmark_group!(
    name = l2_u8_iai;
    benchmarks = benchmark_u8_v2, benchmark_u8_v2_sized,
);

const DIM_104: usize = 104;
const DIM_128: usize = 128;

fn prepare_benchmark_f32() -> (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>) {
    let dim = DIM_104;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim).unwrap(),
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

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_f32)]
pub fn benchmark_f32_v2(data: (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let func = InlineBarrier::new(SquaredL2 {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] =
            diskann_vector::DistanceFunction::evaluate_similarity(&func, query.as_slice(), v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_f32)]
pub fn benchmark_f32_v2_sized(data: (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let query_sized: &[f32; DIM_104] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(SquaredL2 {});
    dataset
        .iter_sized(Static::<DIM_104>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func, query_sized, v);
        });

    black_box(output);
}

fn prepare_benchmark_f16() -> (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>) {
    let dim = DIM_104;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim).unwrap(),
        NonZeroUsize::new(128).unwrap(),
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

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_f16)]
pub fn benchmark_f16_v2(data: (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let func = InlineBarrier::new(SquaredL2 {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] = diskann_vector::DistanceFunction::evaluate_similarity(&func, &*query, v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_f16)]
pub fn benchmark_f16_v2_sized(data: (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let query_sized: &[f16; DIM_104] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(SquaredL2 {});
    dataset
        .iter_sized(Static::<DIM_104>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func, query_sized, v);
        });

    black_box(output);
}

fn prepare_benchmark_i8() -> (AlignedDataset<i8>, AlignedVector<i8>, Vec<f32>) {
    let dim = DIM_128;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<i8> = AlignedDataset::new(args, &mut rng, dist);
    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_i8)]
pub fn benchmark_i8_v2(data: (AlignedDataset<i8>, AlignedVector<i8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let func = InlineBarrier::new(SquaredL2 {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] = diskann_vector::DistanceFunction::evaluate_similarity(&func, &*query, v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_i8)]
pub fn benchmark_i8_v2_sized(data: (AlignedDataset<i8>, AlignedVector<i8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let query_sized: &[i8; DIM_128] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(SquaredL2 {});
    dataset
        .iter_sized(Static::<DIM_128>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func, query_sized, v);
        });

    black_box(output);
}

fn prepare_benchmark_u8() -> (AlignedDataset<u8>, AlignedVector<u8>, Vec<f32>) {
    let dim = Static::<128>;

    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<u8> = AlignedDataset::new(args, &mut rng, dist);
    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_u8)]
pub fn benchmark_u8_v2(data: (AlignedDataset<u8>, AlignedVector<u8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let func = InlineBarrier::new(SquaredL2 {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] = diskann_vector::DistanceFunction::evaluate_similarity(&func, &*query, v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = prepare_benchmark_u8)]
pub fn benchmark_u8_v2_sized(data: (AlignedDataset<u8>, AlignedVector<u8>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let query_sized: &[u8; DIM_128] = query.as_slice().try_into().unwrap();
    let func = InlineBarrier::new(SquaredL2 {});
    dataset
        .iter_sized(Static::<DIM_128>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func, query_sized, v);
        });

    black_box(output);
}
