/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_vector::distance::CosineNormalized;
use half::f16;
use iai_callgrind::black_box;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::utils::{AlignedDataset, AlignedVector, DatasetArgs, InlineBarrier, Static};

iai_callgrind::library_benchmark_group!(
    name = cosine_normalized_f32_iai;
    benchmarks = benchmark_f32_v2, benchmark_f32_v2_sized,
);

iai_callgrind::library_benchmark_group!(
    name = cosine_normalized_f16_iai;
    benchmarks = benchmark_f16_v2, benchmark_f16_v2_sized,
);

const DIM_104: usize = 104;
const DIM_384: usize = 384;

fn setup_benchmark_f32() -> (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>) {
    let seed = 0xc0ff33;
    let dim = DIM_104;
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );
    let dist = Normal::<f32>::new(-1.0, 1.0).unwrap();

    let mut rng = StdRng::seed_from_u64(seed);
    let dataset: AlignedDataset<f32> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = setup_benchmark_f32)]
pub(crate) fn benchmark_f32_v2(data: (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    // Version 2
    let func_v2 = InlineBarrier::new(CosineNormalized {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] = diskann_vector::DistanceFunction::evaluate_similarity(&func_v2, &*query, v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = setup_benchmark_f32)]
pub(crate) fn benchmark_f32_v2_sized(data: (AlignedDataset<f32>, AlignedVector<f32>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let query_sized: &[f32; DIM_104] = query.as_slice().try_into().unwrap();

    let func_v2 = InlineBarrier::new(CosineNormalized {});
    dataset
        .iter_sized(Static::<DIM_104>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func_v2, query_sized, v);
        });

    black_box(output);
}

fn setup_benchmark_f16() -> (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>) {
    let dim = DIM_384;
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim).unwrap(),
        NonZeroUsize::new(dim).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );
    let seed = 0xc0ff33;
    let dist = Normal::<f16>::new(f16::from_f32(-1.0), f16::from_f32(1.0)).unwrap();
    let mut rng = StdRng::seed_from_u64(seed);

    let dataset: AlignedDataset<f16> = AlignedDataset::new(args, &mut rng, dist);

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));

    let output: Vec<f32> = vec![0.0; dataset.len()];

    (dataset, query, output)
}

#[iai_callgrind::library_benchmark(setup = setup_benchmark_f16)]
pub(crate) fn benchmark_f16_v2(data: (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>)) {
    let (dataset, query, mut output) = data;

    let func = InlineBarrier::new(CosineNormalized {});
    dataset.iter().enumerate().for_each(|(i, v)| {
        output[i] = diskann_vector::DistanceFunction::evaluate_similarity(&func, &*query, v);
    });

    black_box(output);
}

#[iai_callgrind::library_benchmark(setup = setup_benchmark_f16)]
pub(crate) fn benchmark_f16_v2_sized(data: (AlignedDataset<f16>, AlignedVector<f16>, Vec<f32>)) {
    let (dataset, query, mut output) = data;
    let query_sized: &[f16; DIM_384] = query.as_slice().try_into().unwrap();

    let func = InlineBarrier::new(CosineNormalized {});
    dataset
        .iter_sized(Static::<DIM_384>)
        .enumerate()
        .for_each(|(i, v)| {
            output[i] =
                diskann_vector::DistanceFunction::evaluate_similarity(&func, query_sized, v);
        });

    black_box(output);
}
