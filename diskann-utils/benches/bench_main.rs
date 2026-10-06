/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use diskann_utils::views::rowmajor::{Matrix, MatrixMut, Owned};
use rayon::prelude::ParallelIterator;

const NCOLS: usize = 100;

fn update_rows<'a, I>(rows: I)
where
    I: ParallelIterator<Item = &'a mut [u32]>,
{
    rows.for_each(|row| {
        let (first, rest) = row.split_first_mut().unwrap();
        *first = first.wrapping_add(1);

        let mut previous = *first;
        for value in rest {
            *value = value.wrapping_add(previous).wrapping_add(1);
            previous = *value;
        }
    });
}

fn benchmark_par_rows_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_rows_mut");

    for nrows in [5, 1_000_000] {
        group.throughput(Throughput::Elements((nrows * NCOLS) as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{nrows}x{NCOLS}")),
            |b| {
                let mut matrix = Owned::from_element(nrows, NCOLS, 0u32);
                b.iter(|| {
                    update_rows(black_box(&mut matrix).par_rows_mut());
                    black_box(matrix.as_slice());
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    targets = benchmark_par_rows_mut
}
criterion_main!(benches);
