/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::time::Duration;

use super::{DefaultContext, Store, Strategy, index, matrix, seeded};
use crate::ivf::BuildMetrics;
use diskann_utils::views::rowmajor::Matrix;

fn assert_empty(metrics: &BuildMetrics) {
    assert_eq!(metrics.insert_time, Duration::ZERO);
    assert_eq!(metrics.split_time, Duration::ZERO);
    assert_eq!(metrics.fit_time, Duration::ZERO);
    assert_eq!(metrics.assignment_time, Duration::ZERO);
    assert_eq!(
        (
            metrics.split_calls,
            metrics.split_parents,
            metrics.split_points
        ),
        (0, 0, 0)
    );
}

#[tokio::test]
async fn initialization_empty_batches_and_failed_publication_do_not_count() {
    let mut index = index(
        Store {
            dim: 2,
            next_list: 1,
            ..Store::default()
        },
        2,
    );
    index
        .initialize(&Strategy, &DefaultContext, matrix(&[[0.0, 0.0]]).as_view())
        .await
        .unwrap();
    let empty: [(u32, &[f32]); 0] = [];
    index
        .insert_batch(&Strategy, &DefaultContext, &empty)
        .await
        .unwrap();
    assert_empty(index.build_metrics());
    for fail_fill in [true, false] {
        let mut store = seeded();
        store.fail_fill = fail_fill;
        store.fail_finish = !fail_fill;
        let mut failing = super::index(store, 2);
        assert!(
            failing
                .insert_batch(&Strategy, &DefaultContext, &[(5, &[1.0, 1.0][..])])
                .await
                .is_err()
        );
        assert_empty(failing.build_metrics());
    }
}

#[tokio::test]
async fn successful_splits_accumulate_work_and_nested_phase_times() {
    let mut index = index(seeded(), 2);
    let batch = [
        (5, &[1.0, 1.0][..]),
        (6, &[11.0, 1.0][..]),
        (7, &[101.0, 0.0][..]),
    ];
    index
        .insert_batch(&Strategy, &DefaultContext, &batch)
        .await
        .unwrap();
    let before = *index.build_metrics();
    assert_eq!(
        (
            before.split_calls,
            before.split_parents,
            before.split_points
        ),
        (1, 2, 6)
    );
    assert!(before.insert_time >= before.split_time);
    assert!(before.split_time >= before.fit_time + before.assignment_time);
    assert!(
        index
            .insert_batch(&Strategy, &DefaultContext, &batch)
            .await
            .is_err()
    );
    assert_eq!(index.build_metrics().insert_time, before.insert_time);
    assert_eq!(index.build_metrics().split_calls, before.split_calls);
    index
        .insert_batch(&Strategy, &DefaultContext, &[(8, &[102.0, 0.0][..])])
        .await
        .unwrap();
    let after = index.build_metrics();
    assert_eq!(
        (after.split_calls, after.split_parents, after.split_points),
        (2, 3, 9)
    );
    assert!(after.insert_time >= before.insert_time);
    assert!(after.split_time >= before.split_time);
    assert!(after.insert_time >= after.split_time);
    assert!(after.split_time >= after.fit_time + after.assignment_time);
}
