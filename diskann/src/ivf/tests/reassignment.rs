/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::{
    Store,
    replay::{data, replay},
};
use crate::ivf::{IVFIndex, ReassignmentMode};

#[test]
fn reassignment_replay_preserves_layout() {
    #[cfg(not(feature = "ivf-experimental-compute"))]
    let modes = [ReassignmentMode::Baseline];
    #[cfg(feature = "ivf-experimental-compute")]
    let modes = [ReassignmentMode::Baseline, ReassignmentMode::Gemm];
    for dim in [33, 128, 769] {
        let data = data(dim);
        for mode in modes {
            // Fitting stays at its baseline so this tests reassignment independently.
            let configure =
                |index: &mut IVFIndex<Store>| index.set_reassignment_mode(mode).unwrap();
            let expected = replay(&data, 30, configure);
            assert_eq!(replay(&data, 30, configure), expected, "{mode:?}, D={dim}");
            #[cfg(feature = "ivf-parallel")]
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                assert_eq!(
                    pool.install(|| replay(&data, 30, configure)),
                    expected,
                    "{mode:?}, D={dim}, threads={threads}"
                );
            }
        }
    }
}

#[test]
fn reassignment_mode_requires_experimental_feature() {
    let mut index = super::index(super::seeded(), 2);
    index
        .set_reassignment_mode(ReassignmentMode::Baseline)
        .unwrap();
    assert_eq!(
        index.set_reassignment_mode(ReassignmentMode::Gemm).is_ok(),
        cfg!(feature = "ivf-experimental-compute")
    );
}

#[cfg(feature = "ivf-experimental-compute")]
#[test]
fn fitting_and_reassignment_modes_are_independent() {
    use crate::ivf::{
        TwoMeansMode, clustering,
        workingset::{ListView, PointRef},
    };
    use diskann_utils::views::rowmajor::Matrix;
    use rand::{SeedableRng, rngs::StdRng};

    let rows: Vec<Vec<Vec<f32>>> = (0..4)
        .map(|parent| {
            (0..9)
                .map(|row| vec![parent as f32 * 10.0 + row as f32 * 0.125, row as f32])
                .collect()
        })
        .collect();
    let lists: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(parent, points)| ListView {
            id: parent,
            points: points
                .iter()
                .enumerate()
                .map(|(id, vector)| PointRef {
                    id: 9 * parent + id,
                    vector,
                })
                .collect(),
        })
        .collect();
    for fit in [TwoMeansMode::Baseline, TwoMeansMode::Projection] {
        let mut previous_centers = None;
        for assignment in [ReassignmentMode::Baseline, ReassignmentMode::Gemm] {
            let children =
                clustering::split(&lists, 2, 5, &mut StdRng::seed_from_u64(7), fit, assignment)
                    .unwrap();
            let centers: Vec<_> = children
                .centroids
                .as_slice()
                .iter()
                .map(|x| x.to_bits())
                .collect();
            if let Some(expected) = &previous_centers {
                assert_eq!(&centers, expected);
            }
            previous_centers = Some(centers);
            let stats = children.compute;
            assert_eq!(stats.projection_points > 0, fit == TwoMeansMode::Projection);
            assert_eq!(stats.gemm_calls > 0, assignment == ReassignmentMode::Gemm);
            assert_eq!(
                stats.gemm_points,
                if assignment == ReassignmentMode::Gemm {
                    36
                } else {
                    0
                }
            );
            assert_eq!(stats.numeric_fallbacks, 0);
            let mut members: Vec<_> = children.members.into_iter().flatten().collect();
            members.sort_unstable();
            assert_eq!(members, (0..36).collect::<Vec<_>>());
        }
    }
}
