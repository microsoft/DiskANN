/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use rand::{Rng, RngCore, SeedableRng};

use super::*;

#[test]
fn assignment_hook_preserves_empty_center_and_rejects_unwritten_ids() {
    let rows = [vec![-10.0], vec![10.0]];
    let points: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(id, vector)| PointRef {
            id,
            vector: vector.as_slice(),
        })
        .collect();
    let mut seed = StdRng::seed_from_u64(7);
    let first = seed.random_range(0..2);
    let second = (first + seed.random_range(1..2)) % 2;
    let mut centers = [42.0; 2];
    let mut rng = StdRng::seed_from_u64(7);
    let mut calls = 0;
    fit_with_assignment(
        &points,
        &mut centers,
        5,
        &mut rng,
        &mut [0.0; 2],
        |_, out| {
            calls += 1;
            out.fill(0);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(calls, 5);
    assert_eq!(centers, [0.0, rows[second][0]]);
    assert_eq!(seed.next_u64(), rng.next_u64());
    for value in [2, u32::MAX] {
        assert!(
            fit_with_assignment(
                &points,
                &mut centers,
                1,
                &mut rng,
                &mut [0.0; 2],
                |_, out| {
                    out.fill(value);
                    Ok(())
                }
            )
            .is_err()
        );
    }
    assert!(
        fit_with_assignment(
            &points,
            &mut centers,
            1,
            &mut rng,
            &mut [0.0; 2],
            |_, _| Ok(())
        )
        .is_err()
    );
    assert!(
        fit_with_assignment(&points, &mut centers, 1, &mut rng, &mut [0.0; 2], |_, _| {
            Err(ANNError::message("assignment failed"))
        })
        .is_err()
    );
    assert!(
        fit_with_assignment(&points, &mut [0.0], 1, &mut rng, &mut [0.0; 2], |_, _| Ok(
            ()
        ))
        .is_err()
    );
    assert!(
        fit_with_assignment(
            &points[..1],
            &mut centers,
            1,
            &mut rng,
            &mut [0.0; 2],
            |_, _| Ok(())
        )
        .is_err()
    );
}
