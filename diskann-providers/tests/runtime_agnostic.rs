/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use diskann::{
    ANNResult,
    graph::{IdDistance, InplaceDeleteMethod, config, search::Knn},
    provider::{DefaultContext, Delete},
    task::{Task, TaskSpawner},
};
use diskann_providers::{
    index::diskann_async::new_index_with_spawner,
    model::graph::provider::async_::{
        common::{FullPrecision, TableBasedDeletes},
        inmem::{DefaultProviderParameters, SetStartPoints},
    },
};
use diskann_utils::views::Matrix;
use diskann_vector::distance::Metric;

#[derive(Debug, Default)]
struct SmolSpawner {
    scheduled: AtomicUsize,
}

impl TaskSpawner for SmolSpawner {
    fn spawn(&self, task: Task) -> ANNResult<()> {
        self.scheduled.fetch_add(1, Ordering::Relaxed);
        smol::spawn(task).detach();
        Ok(())
    }
}

#[test]
fn parallel_index_operations_run_without_tokio() {
    smol::block_on(async {
        let config = config::Builder::new_with(
            6,
            config::MaxDegree::default_slack(),
            20,
            Metric::L2.into(),
            |builder| {
                builder.max_minibatch_par(4);
            },
        )
        .build()
        .unwrap();
        let params =
            DefaultProviderParameters::simple(8, 2, Metric::L2, config.max_degree_u32().get());
        let spawner = Arc::new(SmolSpawner::default());
        let index =
            new_index_with_spawner::<f32, _>(config, params, TableBasedDeletes, spawner.clone())
                .unwrap();
        index
            .provider()
            .set_start_points(std::iter::once([0.0f32, 0.0].as_slice()))
            .unwrap();

        let mut vectors = Matrix::new(0.0f32, 8, 2);
        for (i, row) in vectors.row_iter_mut().enumerate() {
            row.copy_from_slice(&[(i % 4) as f32, (i / 4) as f32]);
        }
        let context = DefaultContext;
        index
            .multi_insert::<FullPrecision, Matrix<f32>>(
                FullPrecision,
                &context,
                Arc::new(vectors),
                Arc::from((0..8u32).collect::<Vec<_>>()),
            )
            .await
            .unwrap();

        let mut ids = [u32::MAX; 3];
        let mut distances = [f32::INFINITY; 3];
        let mut output = IdDistance::new(&mut ids, &mut distances);
        index
            .search(
                Knn::new_default(8).unwrap(),
                &FullPrecision,
                &context,
                &[0.0f32, 0.0],
                &mut output,
            )
            .await
            .unwrap();
        assert_eq!(ids[0], 0);

        index
            .multi_inplace_delete(
                FullPrecision,
                &context,
                Arc::from([6u32, 7u32]),
                3,
                InplaceDeleteMethod::OneHop,
            )
            .await
            .unwrap();
        for id in [6, 7] {
            assert!(
                index
                    .provider()
                    .status_by_internal_id(&context, id)
                    .await
                    .unwrap()
                    .is_deleted()
            );
        }
        assert!(spawner.scheduled.load(Ordering::Relaxed) > 4);
    });
}
