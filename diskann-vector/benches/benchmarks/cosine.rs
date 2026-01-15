/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use criterion::Criterion;
use diskann_vector::{distance::Cosine, DistanceFunction};
use half::f16;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal, StandardUniform};

use crate::utils::{
    record::format_benchmark, AlignedDataset, AlignedVector, DatasetArgs, InlineBarrier, Static,
};

pub(crate) fn benchmark_f32(c: &mut Criterion) {
    const DIM: usize = 104;
    let dim = Static::<DIM>;

    let mut group = c.benchmark_group("bulk-cosine-f32");
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = Normal::<f32>::new(-1.0, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<f32> = AlignedDataset::new(args, &mut rng, dist);
    let mut output: Vec<f32> = vec![0.0; dataset.len()];

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));
    let query_sized: &[f32; DIM] = query.as_slice().try_into().unwrap();

    /////////////
    // Aligned //
    /////////////

    group.bench_function(format_benchmark(dim.dynamic(), true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter().enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(&*query, v);
            })
        });
    });

    group.bench_function(format_benchmark(dim, true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter_sized(dim).enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(query_sized, v);
            })
        });
    });
}

pub(crate) fn benchmark_f16(c: &mut Criterion) {
    const DIM: usize = 384;
    let dim = Static::<DIM>;

    let mut group = c.benchmark_group("bulk-cosine-f16");
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(384).unwrap(),
        NonZeroUsize::new(40).unwrap(),
    );

    let dist = Normal::<f16>::new(f16::from_f32(-1.0), f16::from_f32(1.0)).unwrap();
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<f16> = AlignedDataset::new(args, &mut rng, dist);
    let mut output: Vec<f32> = vec![0.0; dataset.len()];

    let mut query = AlignedVector::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));
    let query_sized: &[f16; DIM] = query.as_slice().try_into().unwrap();

    /////////////
    // Aligned //
    /////////////

    group.bench_function(format_benchmark(dim.dynamic(), true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter().enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(&*query, v);
            })
        });
    });

    group.bench_function(format_benchmark(dim, true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter_sized(dim).enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(query_sized, v);
            })
        });
    });
}

pub(crate) fn benchmark_i8(c: &mut Criterion) {
    const DIM: usize = 128;
    let dim = Static::<DIM>;

    let mut group = c.benchmark_group("bulk-cosine-i8");
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<i8> = AlignedDataset::new(args, &mut rng, dist);
    let mut output: Vec<f32> = vec![0.0; dataset.len()];

    let mut query = AlignedVector::<i8>::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));
    let query_sized: &[i8; DIM] = query.as_slice().try_into().unwrap();

    /////////////
    // Aligned //
    /////////////

    group.bench_function(format_benchmark(dim.dynamic(), true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter().enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(&*query, v);
            })
        });
    });

    group.bench_function(format_benchmark(dim, true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter_sized(dim).enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(query_sized, v);
            })
        });
    });
}

pub(crate) fn benchmark_u8(c: &mut Criterion) {
    const DIM: usize = 128;
    let dim = Static::<DIM>;

    let mut group = c.benchmark_group("bulk-cosine-u8");
    let args = DatasetArgs::new(
        0,
        NonZeroUsize::new(dim.into()).unwrap(),
        NonZeroUsize::new(128).unwrap(),
        NonZeroUsize::new(48).unwrap(),
    );

    let dist = StandardUniform;
    let mut rng = StdRng::seed_from_u64(0xc0ff33);

    let dataset: AlignedDataset<u8> = AlignedDataset::new(args, &mut rng, dist);
    let mut output: Vec<f32> = vec![0.0; dataset.len()];

    let mut query = AlignedVector::<u8>::new(args.dim(), args.alignment());
    query.iter_mut().for_each(|i| *i = dist.sample(&mut rng));
    let query_sized: &[u8; DIM] = query.as_slice().try_into().unwrap();

    /////////////
    // Aligned //
    /////////////

    group.bench_function(format_benchmark(dim.dynamic(), true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter().enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(&*query, v);
            })
        });
    });

    group.bench_function(format_benchmark(dim, true), |f| {
        let func = InlineBarrier::new(Cosine {});
        f.iter(|| {
            dataset.iter_sized(dim).enumerate().for_each(|(i, v)| {
                output[i] = func.evaluate_similarity(query_sized, v);
            })
        });
    });
}
