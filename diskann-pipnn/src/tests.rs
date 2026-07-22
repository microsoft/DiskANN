/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::num::NonZeroUsize;

use diskann_vector::distance::Metric;

use super::*;

#[test]
fn config_rejects_invalid_values() {
    let invalid = [
        PiPNNConfig {
            c_max: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            c_min: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            c_min: 65,
            c_max: 64,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            p_samp: 0.0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            p_samp: f64::NAN,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            p_samp: 1.1,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            fanout: Vec::new(),
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            fanout: vec![5, 0],
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            fanout: vec![partition::MAX_FANOUT + 1],
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            k: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            replicas: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            l_max: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            l_max: hash_prune::MAX_RESERVOIR_LEN + 1,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            num_hash_planes: 0,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            num_hash_planes: 17,
            ..PiPNNConfig::default()
        },
    ];

    for config in invalid {
        assert!(config.validate().is_err(), "accepted {config:?}");
    }
}

#[test]
fn config_accepts_supported_boundaries() {
    for config in [
        PiPNNConfig::default(),
        PiPNNConfig {
            p_samp: 1.0,
            num_hash_planes: 1,
            l_max: 1,
            ..PiPNNConfig::default()
        },
        PiPNNConfig {
            num_hash_planes: 16,
            l_max: hash_prune::MAX_RESERVOIR_LEN,
            fanout: vec![partition::MAX_FANOUT],
            ..PiPNNConfig::default()
        },
    ] {
        config.validate().unwrap();
    }
}

#[test]
fn direct_candidates_require_final_prune_but_not_hashprune_settings() {
    let invalid = PiPNNConfig {
        skip_hash_prune: true,
        final_prune: false,
        ..PiPNNConfig::default()
    };
    assert!(invalid.validate().is_err());

    let valid = PiPNNConfig {
        skip_hash_prune: true,
        final_prune: true,
        l_max: 0,
        num_hash_planes: 0,
        ..PiPNNConfig::default()
    };
    valid.validate().unwrap();
}

#[test]
fn build_context_rejects_invalid_alpha() {
    let degree = NonZeroUsize::new(16).unwrap();
    for alpha in [0.9, f32::INFINITY, f32::NAN] {
        assert!(
            PiPNNBuildContext::new(PiPNNConfig::default(), degree, alpha, Metric::L2, 0).is_err(),
            "accepted alpha {alpha}"
        );
    }
}

#[test]
fn config_round_trips_through_json() {
    let config = PiPNNConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let decoded: PiPNNConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, config);
}

#[test]
fn bigann10m_peak_memory_estimate_covers_profiled_peak() {
    let config = PiPNNConfig {
        num_hash_planes: 14,
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
        l_max: 72,
        final_prune: true,
        skip_hash_prune: false,
    };
    let estimated = config
        .estimated_peak_memory_bytes(10_000_000, 128, 2, 16)
        .unwrap();
    let measured = 12_789_440usize * 1024;

    assert!(estimated >= measured);
    assert!(estimated < measured * 102 / 100);
}

#[test]
fn peak_memory_estimate_covers_smaller_profiled_builds() {
    let config = PiPNNConfig {
        num_hash_planes: 14,
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
        l_max: 72,
        final_prune: true,
        skip_hash_prune: false,
    };

    for (npoints, measured_kib) in [(100_000, 453_744usize), (1_000_000, 1_679_928)] {
        let estimated = config
            .estimated_peak_memory_bytes(npoints, 128, 2, 16)
            .unwrap();
        assert!(estimated >= measured_kib * 1024);
    }
}

#[test]
fn peak_memory_estimate_covers_single_thread_profile() {
    let config = PiPNNConfig {
        num_hash_planes: 14,
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
        l_max: 72,
        final_prune: true,
        skip_hash_prune: false,
    };

    let estimated = config
        .estimated_peak_memory_bytes(100_000, 128, 2, 1)
        .unwrap();
    assert!(estimated >= 162_596usize * 1024);
}

#[test]
fn peak_memory_estimate_scales_with_replicas() {
    let one = PiPNNConfig {
        fanout: vec![10, 3],
        replicas: 1,
        ..PiPNNConfig::default()
    };
    let two = PiPNNConfig {
        replicas: 2,
        ..one.clone()
    };

    assert!(
        two.estimated_peak_memory_bytes(1_000, 128, 2, 16)
            > one.estimated_peak_memory_bytes(1_000, 128, 2, 16)
    );
}

#[test]
fn direct_candidate_peak_memory_estimate_covers_profiled_peak() {
    let config = PiPNNConfig {
        num_hash_planes: 0,
        c_max: 512,
        c_min: 64,
        p_samp: 0.01,
        fanout: vec![10, 3],
        k: 2,
        replicas: 1,
        l_max: 0,
        final_prune: true,
        skip_hash_prune: true,
    };
    let estimated = config
        .estimated_peak_memory_bytes(10_000_000, 128, 2, 16)
        .unwrap();
    let measured = 8_656_004usize * 1024;

    assert!(estimated >= measured);
    assert!(estimated < measured * 115 / 100);
}
