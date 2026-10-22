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
    for strategy in Strategy::ALL {
        let run = fit(&points, 3, 3, 7, strategy).unwrap();
        assert!(run.preparation <= run.total);
        assert_ne!(run.centers[0], run.centers[3]);
        assert_eq!(run.packed_bytes, if strategy == BASELINE { 0 } else { 192 });
        assert_eq!(
            run.scratch_bytes,
            6 * 8
                + if strategy == BASELINE {
                    0
                } else if strategy == Strategy::TransposedLloydReference {
                    16
                } else {
                    8
                }
        );
    }
}
