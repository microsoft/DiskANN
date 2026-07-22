/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use super::*;

#[test]
fn test_partition_basic() {
    let npoints = 1000;
    let ndims = 8;
    let data: Vec<f32> = {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..npoints * ndims).map(|_| rng.random::<f32>()).collect()
    };
    let config = PiPNNConfig {
        c_max: 64,
        c_min: 16,
        p_samp: 0.1,
        fanout: vec![4, 2],
        ..PiPNNConfig::default()
    };
    let leaves = partition(&data, ndims, npoints, &config, Metric::L2, 123).unwrap();

    // All points should appear at least once (overlapping partitions).
    let mut seen = vec![false; npoints];
    for leaf in &leaves {
        assert!(leaf.indices.len() <= config.c_max, "leaf too large");
        for &idx in &leaf.indices {
            seen[idx as usize] = true;
        }
    }
    assert!(seen.iter().all(|&s| s), "some points missing");
}

#[test]
fn test_partition_small_dataset() {
    let npoints = 50;
    let ndims = 4;
    let data: Vec<f32> = vec![1.0; npoints * ndims];
    let config = PiPNNConfig {
        c_max: 64,
        c_min: 8,
        p_samp: 0.1,
        fanout: vec![3],
        ..PiPNNConfig::default()
    };
    let leaves = partition(&data, ndims, npoints, &config, Metric::L2, 0).unwrap();
    assert_eq!(leaves.len(), 1);
    assert_eq!(leaves[0].indices.len(), npoints);
}
