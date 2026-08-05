/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#![allow(
    clippy::unwrap_used,
    reason = "deterministic benchmark fixture construction must abort on invalid setup"
)]

use diskann::graph::pipnn::{
    leaf_kernel::{LeafInput, LeafKernel, LeafKernelWorkspace, LeafNeighbor, leaf_neighbor_count},
    partition_kernel::{PartitionInput, PartitionKernel, PartitionScales},
};
use diskann_utils::views::{MatrixView, MutMatrixView};
use diskann_vector::distance::Metric;
use iai_callgrind::black_box;

const PARTITION_POINTS: usize = 256;
const LEADERS: usize = 32;
const FANOUT: usize = 4;
const LEAF_POINTS: usize = 128;
const LEAF_K: usize = 3;

type PartitionFixture = (PartitionKernel, Vec<f32>, Vec<f32>, Vec<u32>);
type LeafFixture = (LeafKernel, LeafKernelWorkspace, Vec<f32>, Vec<LeafNeighbor>);

fn setup_partition() -> PartitionFixture {
    let dots = (0..PARTITION_POINTS * LEADERS)
        .map(|index| ((index * 17 + 11) % 257) as f32 / 257.0)
        .collect();
    let leader_squared_norms = (0..LEADERS)
        .map(|leader| 1.0 + leader as f32 / LEADERS as f32)
        .collect();
    (
        PartitionKernel::new(Metric::L2),
        dots,
        leader_squared_norms,
        vec![u32::MAX; PARTITION_POINTS * FANOUT],
    )
}

#[iai_callgrind::library_benchmark(setup = setup_partition)]
fn assign_points_to_leaders(fixture: PartitionFixture) {
    let (kernel, dots, leader_squared_norms, mut output) = fixture;
    kernel
        .nearest_leaders(
            PartitionInput {
                dots: MatrixView::try_from(dots.as_slice(), PARTITION_POINTS, LEADERS).unwrap(),
                scales: PartitionScales::L2 {
                    leader_squared_norms: &leader_squared_norms,
                },
            },
            MutMatrixView::try_from(output.as_mut_slice(), PARTITION_POINTS, FANOUT).unwrap(),
        )
        .unwrap();
    black_box(output);
}

fn setup_leaf() -> LeafFixture {
    let mut dots = vec![f32::NAN; LEAF_POINTS * LEAF_POINTS];
    for source in 0..LEAF_POINTS {
        dots[source * LEAF_POINTS + source] = 1.0 + (source % 7) as f32;
        for target in 0..source {
            dots[source * LEAF_POINTS + target] =
                ((source * 17 + target * 11) % 257) as f32 / 257.0;
        }
    }
    let neighbors = leaf_neighbor_count(LEAF_POINTS, LEAF_K).unwrap();
    (
        LeafKernel::new(Metric::L2),
        LeafKernelWorkspace::new(),
        dots,
        vec![LeafNeighbor::default(); LEAF_POINTS * neighbors],
    )
}

#[iai_callgrind::library_benchmark(setup = setup_leaf)]
fn select_leaf_neighbors(fixture: LeafFixture) {
    let (kernel, mut workspace, dots, mut output) = fixture;
    kernel
        .nearest_neighbors(
            LeafInput {
                dots: MatrixView::try_from(dots.as_slice(), LEAF_POINTS, LEAF_POINTS).unwrap(),
            },
            MutMatrixView::try_from(output.as_mut_slice(), LEAF_POINTS, LEAF_K).unwrap(),
            &mut workspace,
        )
        .unwrap();
    black_box(output);
}

iai_callgrind::library_benchmark_group!(
    name = pipnn_kernels;
    benchmarks = assign_points_to_leaders, select_leaf_neighbors,
);
