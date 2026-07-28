/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{convert::Infallible, hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use diskann::{
    graph::{
        config::PruneKind,
        prune::{self, Policy},
    },
    neighbor::Neighbor,
};

const DEGREE: usize = 64;
const CANDIDATE_COUNTS: [usize; 3] = [64, 128, 750];

fn candidates(count: usize) -> Vec<Neighbor<u32>> {
    (1..=count as u32)
        .map(|id| Neighbor::new(id, id as f32))
        .collect()
}

fn benchmark_robust_prune(c: &mut Criterion) {
    let mut group = c.benchmark_group("vamana/robust-prune");

    for count in CANDIDATE_COUNTS {
        for (kind_name, kind) in [
            ("triangle", PruneKind::TriangleInequality),
            ("occluding", PruneKind::Occluding),
        ] {
            for saturate in [false, true] {
                let input = candidates(count);
                let policy = Policy::new(DEGREE, 1.2, kind, saturate);
                let name = format!(
                    "{kind_name}/{count}-to-{DEGREE}/{}",
                    if saturate { "saturated" } else { "pruned" }
                );

                group.throughput(Throughput::Elements(count as u64));
                group.bench_function(BenchmarkId::from_parameter(name), |bencher| {
                    bencher.iter_batched(
                        || {
                            let mut scratch = prune::Scratch::new();
                            scratch.candidates_mut().extend_from_slice(&input);
                            scratch
                        },
                        |mut scratch| {
                            // Vamana constructs its provider-element cache for every
                            // occlusion call while reusing the surrounding scratch.
                            let mut cache = Vec::new();
                            let mut context = scratch.as_context(count);
                            let result = prune::robust_prune(
                                &mut context,
                                policy,
                                &mut cache,
                                Some,
                                |left, right| Ok::<_, Infallible>(left.abs_diff(*right) as f32),
                                |_| false,
                            );
                            debug_assert!(result.is_ok());
                            black_box(scratch.neighbors());
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = benchmark_robust_prune
}
criterion_main!(benches);
