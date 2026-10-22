/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[test]
fn reports_packing_and_diagnostics_for_complete_fit() {
    let points = [
        PointRef {
            id: 0,
            vector: &[-10.0; 3],
        },
        PointRef {
            id: 1,
            vector: &[10.0; 3],
        },
    ];
    for strategy in TwoMeansStrategy::ALL {
        let run = fit(&points, 3, 3, 7, strategy).unwrap();
        assert!(run.preparation <= run.total);
        assert!(run.margins.iter().all(|&value| value == 1200.0));
        assert_ne!(run.assignments[0], run.assignments[1]);
        assert_eq!(
            run.packed_bytes,
            if matches!(
                strategy,
                TwoMeansStrategy::PackedPaired | TwoMeansStrategy::Sgemm
            ) {
                24
            } else {
                0
            }
        );
    }
}
