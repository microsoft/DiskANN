/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{hint::black_box, iter, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use diskann::{
    graph::{
        self, AdjacencyList, DiskANNIndex,
        config::{MaxDegree, PruneKind},
        test::provider::{self as test_provider, Provider, StartPoint},
    },
    provider::NeighborAccessor,
};
use diskann_vector::distance::Metric;

const DEGREE: usize = 64;
const CANDIDATE_COUNTS: [usize; 3] = [64, 128, 750];

struct PruneCase {
    index: DiskANNIndex<Provider>,
    strategy: test_provider::Strategy,
}

impl PruneCase {
    #[allow(clippy::unwrap_used)] // Deterministic benchmark fixture construction.
    fn new(count: usize, kind: PruneKind, saturate: bool) -> Self {
        let source = 0_u32;
        let start_id = count as u32 + 1;
        let mut source_neighbors = AdjacencyList::new();
        for candidate in 1..=count as u32 {
            source_neighbors.push(candidate);
        }

        let provider_config = test_provider::Config::new(
            Metric::L2,
            count.max(DEGREE),
            StartPoint::new(start_id, vec![0.0]),
        )
        .unwrap();
        let points = (0..=count as u32).map(|id| {
            let neighbors = if id == source {
                source_neighbors.clone()
            } else {
                AdjacencyList::new()
            };
            (id, vec![id as f32], neighbors)
        });
        let provider = Provider::new_from(
            provider_config,
            iter::once((start_id, AdjacencyList::new())),
            points,
        )
        .unwrap();
        let config = graph::config::Builder::new_with(
            DEGREE,
            MaxDegree::new(count.max(DEGREE)),
            count.max(DEGREE),
            kind,
            |builder| {
                builder
                    .alpha(1.2)
                    .saturate_after_prune(saturate)
                    .max_occlusion_size(count);
            },
        )
        .build()
        .unwrap();

        Self {
            index: DiskANNIndex::new(config, provider, None),
            strategy: test_provider::Strategy::new(),
        }
    }

    #[allow(clippy::unwrap_used)] // A benchmark fixture failure must abort the sample.
    async fn run(self) -> AdjacencyList<u32> {
        self.index
            .prune_range(
                &self.strategy,
                &test_provider::Context::default(),
                iter::once(0),
            )
            .await
            .unwrap();

        let mut neighbors = AdjacencyList::new();
        self.index
            .provider()
            .neighbors()
            .get_neighbors(0, &mut neighbors)
            .await
            .unwrap();
        neighbors
    }
}

#[allow(clippy::unwrap_used)] // Runtime construction is benchmark harness setup.
fn benchmark_robust_prune(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("vamana/robust-prune");

    for count in CANDIDATE_COUNTS {
        for (kind_name, kind) in [
            ("triangle", PruneKind::TriangleInequality),
            ("occluding", PruneKind::Occluding),
        ] {
            for saturate in [false, true] {
                let name = format!(
                    "{kind_name}/{count}-to-{DEGREE}/{}",
                    if saturate { "saturated" } else { "pruned" }
                );
                group.throughput(Throughput::Elements(count as u64));
                group.bench_function(BenchmarkId::from_parameter(name), |bencher| {
                    bencher.iter_batched(
                        || PruneCase::new(count, kind, saturate),
                        |case| black_box(runtime.block_on(case.run())),
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
