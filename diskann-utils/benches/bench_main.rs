/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use diskann_utils::views::rowmajor::{Matrix, MatrixMut, Owned};
use rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};

const NCOLS: usize = 100;

fn par_rows_mut_baseline<M>(
    matrix: &mut M,
) -> impl IndexedParallelIterator<Item = &mut [M::Element]>
where
    M: MatrixMut,
    M::Element: Send,
{
    let ncols = matrix.ncols();
    assert!(
        ncols != 0 || matrix.nrows() == 0,
        "`MatrixMut::par_rows_mut` does not support matrices with rows and zero columns"
    );
    matrix.as_mut_slice().par_chunks_exact_mut(ncols.max(1))
}

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

fn benchmark_shape(c: &mut Criterion, nrows: usize) {
    let mut group = c.benchmark_group(format!("par_rows_mut/{nrows}x{NCOLS}"));
    group.throughput(Throughput::Elements((nrows * NCOLS) as u64));

    group.bench_function(BenchmarkId::new("baseline", nrows), |b| {
        let mut matrix = Owned::from_element(nrows, NCOLS, 0u32);
        b.iter(|| {
            update_rows(par_rows_mut_baseline(black_box(&mut matrix)));
            black_box(matrix.as_slice());
        });
    });

    group.bench_function(BenchmarkId::new("current", nrows), |b| {
        let mut matrix = Owned::from_element(nrows, NCOLS, 0u32);
        b.iter(|| {
            update_rows(black_box(&mut matrix).par_rows_mut());
            black_box(matrix.as_slice());
        });
    });

    group.finish();
}

fn benchmark_par_rows_mut(c: &mut Criterion) {
    benchmark_shape(c, 5);
    benchmark_shape(c, 1_000_000);
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
