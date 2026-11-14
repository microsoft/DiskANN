/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{num::NonZeroU32, sync::Arc};

use diskann::utils::IntoUsize;
use diskann_benchmark_core::recall::Rows;
use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};

use super::{
    object::{Index, PartialIndex},
    parse,
    stats::{Counters, KnnSearch, KnnStats},
};
use crate::support::{
    datatype::{DataType, Dataset, DatasetView},
    filters::InvertedIndex,
    io::{load_and_convert, open},
};

#[derive(Debug)]
pub(super) struct Bundle {
    pub(super) data: Dataset,
    pub(super) labels: InvertedIndex,
    pub(super) queries: Dataset,
}

pub(super) fn load_data(input: &parse::Data, data_type: DataType) -> anyhow::Result<Bundle> {
    let data = {
        let mut io = open(&*input.data)?;
        load_and_convert(&mut io, input.data_type, data_type, &input.preprocess)?
    };

    let labels = InvertedIndex::parse(std::io::BufReader::new(open(&*input.labels)?))?;
    let queries = {
        let mut io = open(&*input.queries)?;
        load_and_convert(&mut io, input.data_type, data_type, &input.preprocess)?
    };

    Ok(Bundle {
        data,
        labels,
        queries,
    })
}

pub(super) fn insert(
    index: &dyn PartialIndex,
    dataset: DatasetView<'_>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Counters> {
    let before = index.counters();
    for (i, r) in dataset.iter().enumerate() {
        rt.block_on(index.insert(r, i as u64))?;
    }
    before.delta(&index.counters())
}

pub(super) fn multi_insert(
    index: &Arc<dyn Index>,
    dataset: DatasetView<'_>,
    batch_size: NonZeroU32,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Counters> {
    let before = index.counters();
    let batch_size = batch_size.get().into_usize();

    let mut start = 0;
    while start != dataset.nrows() {
        let end = (start + batch_size).min(dataset.nrows());

        #[expect(clippy::expect_used)]
        let batch = dataset
            .subview(start..end)
            .expect("this should be in-bounds by construction");

        let ids = (start as u64)..(end as u64);
        rt.block_on(index.clone().multi_insert(batch, ids))?;
        start = end;
    }
    before.delta(&index.counters())
}

pub(super) fn knn(
    index: &dyn PartialIndex,
    searches: &parse::Knn,
    queries: DatasetView<'_>,
    rt: &tokio::runtime::Handle,
) -> anyhow::Result<Vec<KnnStats>> {
    let parse::Knn {
        searches,
        groundtruth,
    } = searches;

    let groundtruth = {
        let mut io = open(&**groundtruth)?;
        let groundtruth = diskann_utils::io::read_bin::<u32>(&mut io)?;
        groundtruth.map(|v| u64::from(*v))
    };

    anyhow::ensure!(
        queries.nrows() == Matrix::nrows(&groundtruth),
        "number of queries ({}) must match number of groundtruth entries ({})",
        queries.nrows(),
        Matrix::nrows(&groundtruth),
    );

    let mut results = Vec::new();

    for (k, params) in searches.iter() {
        let mut ids = rowmajor::Owned::from_element(queries.nrows(), *k, u64::MAX);

        let before = index.counters();
        let mut misc = KnnSearch::new();
        let mut neighbors = Vec::new();
        for (out, query) in std::iter::zip(ids.rows_mut(), queries.iter()) {
            neighbors.clear();

            let stats = rt.block_on(index.search(query, *params, &mut neighbors))?;
            misc += stats;

            std::iter::zip(out.iter_mut(), neighbors.iter()).for_each(|(d, s)| *d = *s.id());
        }
        let counters = before.delta(&index.counters())?;

        let recall = diskann_benchmark_core::recall::knn(
            &groundtruth,
            None,
            &ids.as_view(),
            *k,
            *k,
            diskann_benchmark_core::recall::GroundTruthMode::Fixed,
        )?;

        let stats = KnnStats {
            counters,
            recall: recall.into(),
            misc,
        };

        results.push(stats);
    }

    Ok(results)
}

pub(super) fn filter_search(
    index: &dyn Index,
    labels: &InvertedIndex,
    searches: parse::CommonFilter<'_>,
    queries: DatasetView<'_>,
) -> anyhow::Result<Vec<KnnStats>> {
    let parse::CommonFilter {
        searches,
        predicates,
        groundtruth,
    } = searches;

    let predicates = labels.queries(std::io::BufReader::new(open(predicates)?))?;

    anyhow::ensure!(
        queries.nrows() == Matrix::nrows(&predicates),
        "number of queries ({}) must match number of predicate maps ({})",
        queries.nrows(),
        Matrix::nrows(&predicates),
    );

    let groundtruth: Vec<Vec<u64>> = {
        let mut io = open(groundtruth)?;
        let raw = diskann_utils::io::read_staggered_bin::<u32>(&mut io)?;
        raw.into_iter()
            .map(|v| v.into_iter().map(u64::from).collect())
            .collect()
    };

    anyhow::ensure!(
        queries.nrows() == groundtruth.nrows(),
        "number of queries ({}) must match number of groundtruth entries ({})",
        queries.nrows(),
        groundtruth.nrows(),
    );

    let mut results = Vec::new();

    for (k, params) in searches {
        let mut ids = rowmajor::Owned::from_element(queries.nrows(), k, u64::MAX);
        let before = index.counters();
        let mut misc = KnnSearch::new();
        let mut neighbors = Vec::new();

        for (out, (query, predicate)) in std::iter::zip(
            ids.rows_mut(),
            std::iter::zip(queries.iter(), predicates.rows()),
        ) {
            neighbors.clear();

            let stats = index.filter_search(query, predicate, params.clone(), &mut neighbors)?;
            misc += stats;

            std::iter::zip(out.iter_mut(), neighbors.iter()).for_each(|(d, s)| *d = *s.id());
        }
        let counters = before.delta(&index.counters())?;

        let recall = diskann_benchmark_core::recall::knn(
            &groundtruth,
            None,
            &ids.as_view(),
            k,
            k,
            diskann_benchmark_core::recall::GroundTruthMode::Flexible,
        )?;

        let stats = KnnStats {
            counters,
            recall: recall.into(),
            misc,
        };

        results.push(stats);
    }

    Ok(results)
}
