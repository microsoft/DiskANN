/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Algorithmic core of SPFresh's Lightweight Incremental RE-balancing (LIRE).
//!
//! This module implements the paper's two necessary conditions for split
//! reassignment and a capacity-balanced binary centroid fit. Storage versioning,
//! asynchronous rebuild workers, and append-only SSD blocks belong to SPFresh's
//! system layer and are intentionally outside the synchronous in-memory
//! OnlineClusterer.

use diskann_utils::views::Matrix;
use rand::{rngs::StdRng, Rng};

use crate::{cluster, GraphIvfError, Result};

pub(super) struct BalancedSplit {
    pub(super) children: [Box<[f32]>; 2],
    #[cfg(test)]
    pub(super) assignments: Vec<u8>,
}

pub(super) fn balanced_two_means(
    points: &Matrix<f32>,
    members: &[u32],
    rng: &mut StdRng,
    max_iters: usize,
    split_threshold: usize,
    normalize: bool,
) -> Result<BalancedSplit> {
    if members.len() < 2 {
        return Err(GraphIvfError::invalid(
            "balanced split requires at least two members",
        ));
    }
    let dim = points.ncols();
    let a = rng.random_range(0..members.len());
    let b = members
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != a)
        .max_by(|(_, left), (_, right)| {
            cluster::sq_l2(points.row(members[a] as usize), points.row(**left as usize)).total_cmp(
                &cluster::sq_l2(
                    points.row(members[a] as usize),
                    points.row(**right as usize),
                ),
            )
        })
        .map(|(index, _)| index)
        .expect("at least two members");
    let mut children = [
        points.row(members[a] as usize).to_vec().into_boxed_slice(),
        points.row(members[b] as usize).to_vec().into_boxed_slice(),
    ];
    let mut assignments = vec![u8::MAX; members.len()];
    let mut previous = vec![u8::MAX; members.len()];
    let mut margins = Vec::with_capacity(members.len());
    // Each child must fit the posting capacity and receive at least one quarter
    // of the parent. SPFresh calls for an even, capacity-bounded split, while
    // its referenced multi-constraint SPANN implementation is not public; this
    // explicit 1:3 bound preserves geometry better than strict 1:1 and rules
    // out degenerate tiny children.
    let child_capacity = split_threshold.max(members.len().div_ceil(2));
    let child_min = members.len().div_ceil(4);

    for _ in 0..max_iters.max(1) {
        margins.clear();
        for (index, &pid) in members.iter().enumerate() {
            let point = points.row(pid as usize);
            margins.push((
                index,
                cluster::sq_l2(point, &children[0]) - cluster::sq_l2(point, &children[1]),
            ));
        }
        margins.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for &(index, margin) in &margins {
            assignments[index] = u8::from(margin > 0.0);
        }
        let mut child0_len = assignments.iter().filter(|&&child| child == 0).count();
        if child0_len > child_capacity {
            // Move the child-0 points with the weakest preference for child 0.
            for &(index, _) in margins[..child0_len]
                .iter()
                .rev()
                .take(child0_len - child_capacity)
            {
                assignments[index] = 1;
            }
            child0_len = child_capacity;
        }
        let child1_len = members.len() - child0_len;
        if child1_len > child_capacity {
            // Move the child-1 points with the weakest preference for child 1.
            for &(index, _) in margins[child0_len..]
                .iter()
                .take(child1_len - child_capacity)
            {
                assignments[index] = 0;
            }
        }
        let child0_len = assignments.iter().filter(|&&child| child == 0).count();
        if child0_len < child_min {
            for &(index, _) in margins[child0_len..].iter().take(child_min - child0_len) {
                assignments[index] = 0;
            }
        }
        let child1_len = assignments.iter().filter(|&&child| child == 1).count();
        if child1_len < child_min {
            for &(index, _) in margins[..members.len() - child1_len]
                .iter()
                .rev()
                .take(child_min - child1_len)
            {
                assignments[index] = 1;
            }
        }
        if assignments == previous {
            break;
        }
        previous.copy_from_slice(&assignments);

        let mut sums = [vec![0.0f64; dim], vec![0.0f64; dim]];
        let mut counts = [0usize; 2];
        for (&pid, &assigned) in members.iter().zip(&assignments) {
            let child = assigned as usize;
            counts[child] += 1;
            for (sum, &value) in sums[child].iter_mut().zip(points.row(pid as usize)) {
                *sum += value as f64;
            }
        }
        for child in 0..2 {
            debug_assert!(counts[child] > 0);
            let inv = 1.0 / counts[child] as f64;
            for (value, &sum) in children[child].iter_mut().zip(&sums[child]) {
                *value = (sum * inv) as f32;
            }
            if normalize {
                cluster::normalize(&mut children[child]);
            }
        }
    }

    Ok(BalancedSplit {
        children,
        #[cfg(test)]
        assignments,
    })
}

pub(super) fn old_posting_may_move_elsewhere(
    point: &[f32],
    old: &[f32],
    children: &[Box<[f32]>; 2],
) -> bool {
    let old_distance = cluster::sq_l2(point, old);
    children
        .iter()
        .all(|child| old_distance <= cluster::sq_l2(point, child))
}

pub(super) fn neighbor_may_move_to_child(
    point: &[f32],
    old: &[f32],
    children: &[Box<[f32]>; 2],
) -> bool {
    let old_distance = cluster::sq_l2(point, old);
    children
        .iter()
        .any(|child| cluster::sq_l2(point, child) <= old_distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn matrix(values: Vec<f32>, rows: usize, cols: usize) -> Matrix<f32> {
        Matrix::try_from(values.into_boxed_slice(), rows, cols).unwrap()
    }

    #[test]
    fn balanced_fit_keeps_training_buckets_within_one() {
        let points = matrix(vec![0.0, 0.1, 0.2, 9.8, 9.9, 10.0, 10.1], 7, 1);
        let members: Vec<u32> = (0..7).collect();
        let mut rng = StdRng::seed_from_u64(4);
        let split = balanced_two_means(&points, &members, &mut rng, 8, 4, false).unwrap();
        let left = split
            .assignments
            .iter()
            .filter(|&&child| child == 0)
            .count();
        let right = split.assignments.len() - left;
        assert!(left.abs_diff(right) <= 1, "{left} vs {right}");
        let mut centers = [split.children[0][0], split.children[1][0]];
        centers.sort_by(f32::total_cmp);
        assert!(centers[0] < 1.0 && centers[1] > 9.0);
    }

    #[test]
    fn lire_conditions_select_only_possible_npa_violations() {
        let children = [vec![-1.0].into_boxed_slice(), vec![1.0].into_boxed_slice()];
        assert!(old_posting_may_move_elsewhere(&[0.0], &[0.0], &children));
        assert!(!old_posting_may_move_elsewhere(&[0.9], &[0.0], &children));
        assert!(neighbor_may_move_to_child(&[1.2], &[0.0], &children));
        assert!(!neighbor_may_move_to_child(&[0.1], &[0.0], &children));
    }

    #[test]
    fn balanced_fit_prevents_a_degenerate_tiny_child() {
        let points = matrix(vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10.0], 8, 1);
        let members: Vec<u32> = (0..8).collect();
        let mut rng = StdRng::seed_from_u64(8);
        let split = balanced_two_means(&points, &members, &mut rng, 8, 8, false).unwrap();
        let left = split
            .assignments
            .iter()
            .filter(|&&child| child == 0)
            .count();
        let right = split.assignments.len() - left;
        assert!(left >= 2 && right >= 2, "{left} vs {right}");
    }
}
