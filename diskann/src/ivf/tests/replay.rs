/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Compare index layouts across repeated builds and caller thread pools.

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{Config, IVFIndex, Store, Strategy, TwoMeansMode};
use crate::provider::DefaultContext;

const PARENTS: usize = 8;
const BATCH_SIZE: usize = 512;
const SPLIT_THRESHOLD: usize = 128;
const SEED: u64 = 7;

#[derive(Debug, PartialEq, Eq)]
struct ListSnapshot {
    id: u32,
    centroid_bits: Vec<u32>,
    members: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Snapshot {
    inserted: usize,
    splits: usize,
    lists: Vec<ListSnapshot>,
}

pub(super) fn replay(
    data: &rowmajor::Owned<f32>,
    iterations: usize,
    configure: impl Fn(&mut IVFIndex<Store>),
) -> Vec<Snapshot> {
    assert!(data.nrows() >= PARENTS && data.ncols() > 0);
    let mut initial = rowmajor::Owned::from_element(PARENTS, data.ncols(), 0.0);
    for (center, row) in initial.rows_mut().zip(data.rows()) {
        center.copy_from_slice(row);
    }
    // Descending IDs expose accidental sorting of the ordered memberships.
    let points: Vec<_> = data
        .rows()
        .enumerate()
        .map(|(i, row)| (u32::try_from(data.nrows() - i - 1).unwrap(), row))
        .collect();
    let mut index = IVFIndex::new(
        Store {
            dim: data.ncols(),
            next_list: 1,
            ..Store::default()
        },
        Config {
            split_threshold: SPLIT_THRESHOLD,
            two_means_iterations: iterations,
            seed: SEED,
        },
    )
    .unwrap();
    configure(&mut index);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime.block_on(async {
        index
            .initialize(&Strategy, &DefaultContext, initial.as_view())
            .await
            .unwrap();
        let mut snapshots = Vec::new();
        for batch in points.chunks(BATCH_SIZE) {
            let stats = index
                .insert_batch(&Strategy, &DefaultContext, batch)
                .await
                .unwrap();
            snapshots.push(Snapshot {
                inserted: stats.inserted,
                splits: stats.splits,
                lists: index
                    .provider()
                    .lists
                    .iter()
                    .map(|(&id, list)| ListSnapshot {
                        id,
                        centroid_bits: list.centroid.iter().map(|x| x.to_bits()).collect(),
                        members: list.members.clone(),
                    })
                    .collect(),
            });
        }
        let mut members: Vec<_> = index
            .provider()
            .lists
            .values()
            .flat_map(|list| list.members.iter().copied())
            .collect();
        members.sort_unstable();
        assert_eq!(
            members,
            (10_000..10_000 + points.len() as u32).collect::<Vec<_>>()
        );
        assert!(snapshots.iter().any(|step| step.splits > 1));
        snapshots
    })
}

#[test]
fn two_means_replay_preserves_layout() {
    #[cfg(not(feature = "ivf-experimental-compute"))]
    let modes = [TwoMeansMode::Baseline];
    #[cfg(feature = "ivf-experimental-compute")]
    let modes = [TwoMeansMode::Baseline, TwoMeansMode::Projection];
    for dim in [33, 128, 769] {
        let data = data(dim);
        for mode in modes {
            let expected = replay(&data, 30, |index| index.set_two_means_mode(mode).unwrap());
            assert_eq!(
                replay(&data, 30, |index| index.set_two_means_mode(mode).unwrap()),
                expected,
                "{mode:?}, D={dim}"
            );
            #[cfg(feature = "ivf-parallel")]
            for threads in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                assert_eq!(
                    pool.install(|| replay(&data, 30, |index| index
                        .set_two_means_mode(mode)
                        .unwrap())),
                    expected,
                    "{mode:?}, D={dim}, threads={threads}",
                );
            }
        }
    }
}

pub(super) fn data(dim: usize) -> rowmajor::Owned<f32> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut data = rowmajor::Owned::from_element(2048, dim, 0.0);
    for (i, row) in data.rows_mut().enumerate() {
        for value in row {
            *value = (i % PARENTS) as f32 * 10.0 + rng.random_range(-1.0..1.0);
        }
    }
    data
}

#[test]
fn two_means_mode_requires_experimental_feature() {
    let mut index = super::index(super::seeded(), 2);
    index.set_two_means_mode(TwoMeansMode::Baseline).unwrap();
    assert_eq!(
        index.set_two_means_mode(TwoMeansMode::Projection).is_ok(),
        cfg!(feature = "ivf-experimental-compute")
    );
}
