/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::neighbor::{Neighbor, NeighborPriorityQueue};
use diskann_benchmark_runner::utils::MicroSeconds;
use diskann_vector::PreprocessedDistanceFunction;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

pub(super) trait QuantStore {
    type Item<'a>
    where
        Self: 'a;
    fn iter(&self) -> impl Iterator<Item = Self::Item<'_>>;
}

pub(super) trait CreateQuantComputer<Q>
where
    Q: QuantStore,
{
    type Computer<'a>: for<'b> PreprocessedDistanceFunction<Q::Item<'b>>
    where
        Self: 'a,
        Q: 'a;

    fn create_quant_computer<'a>(
        &self,
        store: &'a Q,
        query: &[f32],
    ) -> anyhow::Result<Self::Computer<'a>>;
}

#[derive(Debug, Clone)]
pub(super) struct LinearSearch {
    pub(super) ids: diskann_utils::views::Matrix<u32>,
    pub(super) preprocess: Vec<MicroSeconds>,
    pub(super) search: Vec<MicroSeconds>,
    pub(super) total: MicroSeconds,
}

pub(super) fn linear_search<Q, C>(
    store: &Q,
    queries: diskann_utils::views::MatrixView<f32>,
    builder: &C,
    results_per_query: usize,
    progress: &indicatif::ProgressBar,
) -> anyhow::Result<LinearSearch>
where
    Q: QuantStore + Sync,
    C: CreateQuantComputer<Q> + Sync,
{
    let mut output =
        diskann_utils::views::Matrix::<u32>::new(u32::MAX, queries.nrows(), results_per_query);

    struct Times {
        preprocess: MicroSeconds,
        search: MicroSeconds,
    }

    let total = std::time::Instant::now();

    // Lints: Using `ParallelIterator::collect`. It's the caller's responsibility to invoke
    // this in a properly sized Rayon environment.
    #[allow(clippy::disallowed_methods)]
    let times: Vec<Times> = output
        .par_row_iter_mut()
        .zip(queries.par_row_iter())
        .map(|(o, q)| -> anyhow::Result<Times> {
            let mut queue = NeighborPriorityQueue::<u32>::new(results_per_query);

            let start = std::time::Instant::now();
            let f = builder.create_quant_computer(store, q)?;
            let preprocess = start.elapsed().into();

            store.iter().enumerate().for_each(|(j, d)| {
                let distance = f.evaluate_similarity(d);
                queue.insert(Neighbor::new(j as u32, distance));
            });

            let search = start.elapsed().into();

            std::iter::zip(o.iter_mut(), queue.iter()).for_each(|(o, neighbor)| *o = neighbor.id);
            progress.inc(1);
            Ok(Times { preprocess, search })
        })
        .collect::<anyhow::Result<Vec<Times>>>()?;
    let total: MicroSeconds = total.elapsed().into();

    let result = LinearSearch {
        ids: output,
        preprocess: times.iter().map(|i| i.preprocess).collect(),
        search: times.iter().map(|i| i.search).collect(),
        total,
    };

    Ok(result)
}
