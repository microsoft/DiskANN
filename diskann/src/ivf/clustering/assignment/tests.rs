/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

fn refs(rows: &[Vec<f32>]) -> Vec<PointRef<'_, usize>> {
    rows.iter()
        .enumerate()
        .map(|(id, vector)| PointRef { id, vector })
        .collect()
}

#[test]
fn finite_cancellation_is_observable_not_hidden_by_a_tolerance() {
    // True distances are 0.0625 and 0.5625. Expanding squared norms loses
    // the deciding low bits in f32. Paired L2 retains them.
    let point = [10000.25f32];
    let points = [PointRef {
        id: 0,
        vector: point.as_slice(),
    }];
    let centers = [10000.0, 10001.0];
    assert_eq!(paired(&point, &centers, 1), 0);
    let mut runner = Assignment::new(Strategy::CenterDifference, 1, 1, &[]);
    let mut sums = [0.0; 2];
    let mut counts = [0; 2];
    runner
        .assign_and_sum(&points, &centers, 1, &mut sums, &mut counts)
        .unwrap();
    assert_eq!(
        counts,
        [0, 1],
        "the experimental formula must expose its loss of precision"
    );
}

#[test]
fn dot_product_overflow_falls_back_to_paired_for_finite_inputs() {
    let rows = vec![vec![f32::MAX; 9], vec![-f32::MAX; 9], vec![0.0; 9]];
    let points = refs(&rows);
    let centers = [rows[0].clone(), rows[1].clone()].concat();
    let mut expected_sums = vec![0.0; 18];
    let mut expected_counts = [0; 2];
    for point in &points {
        sum_assignment(
            point.vector,
            usize::from(paired(point.vector, &centers, 9)),
            9,
            &mut expected_sums,
            &mut expected_counts,
        );
    }
    assert_eq!(expected_counts, [2, 1]);
    let packed: Vec<_> = rows.iter().flatten().copied().collect();
    for strategy in [Strategy::Sgemm, Strategy::CenterDifference] {
        let mut runner = Assignment::new(strategy, rows.len(), 9, &packed);
        let mut sums = vec![0.0; 18];
        let mut counts = [0; 2];
        runner
            .assign_and_sum(&points, &centers, 9, &mut sums, &mut counts)
            .unwrap();
        assert_eq!(counts, expected_counts);
        assert_eq!(sums, expected_sums);
    }
}
