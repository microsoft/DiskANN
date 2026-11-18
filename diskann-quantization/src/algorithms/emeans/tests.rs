/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_utils::{
    strided::Strided,
    views::rowmajor::{self, Matrix, MatrixMut},
};
use rand::{Rng, seq::SliceRandom};

use std::cell::Cell;

use rand::{SeedableRng, rngs::StdRng};

use crate::cancel::{Cancelation, DontCancel};

use super::*;

#[derive(Default)]
struct ReferenceAssignment {
    batch_sizes: Vec<usize>,
    observed_centers: Vec<Vec<f32>>,
}

impl Assignment for ReferenceAssignment {
    type Error = std::convert::Infallible;

    fn assign(
        &mut self,
        batch: rowmajor::Ref<'_, f32>,
        centers: rowmajor::Ref<'_, f32>,
        labels: &mut [u32],
    ) -> Result<(), Self::Error> {
        self.batch_sizes.push(batch.nrows());
        self.observed_centers.push(centers.as_slice().to_vec());
        for (row, label) in batch.rows().zip(labels) {
            let mut best = (f64::INFINITY, 0);
            for (j, center) in centers.rows().enumerate() {
                let distance: f64 = row
                    .iter()
                    .zip(center)
                    .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                    .sum();
                if distance < best.0 {
                    best = (distance, j);
                }
            }
            *label = best.1 as u32;
        }
        Ok(())
    }
}

#[test]
fn strided_input_and_trait_objects_match_contiguous_simd_training() -> TestResult {
    let (n, d, stride) = (67, 3, 7);
    let mut storage = rowmajor::Owned::from_element(n, stride, f32::NAN);
    let packed = rowmajor::Owned::from_fn(n, d, |rc| {
        (rc.row % 3) as f32 * 20.0 + (rc.row % 7 + rc.col) as f32 * 0.1
    });
    for (dst, src) in storage.rows_mut().zip(packed.rows()) {
        dst[2..2 + d].copy_from_slice(src);
    }
    let strided = Strided::try_from_data(&storage.as_slice()[2..], n, d, stride)?;
    let mut initial = rowmajor::Owned::from_element(3, d, 0.0);
    crate::algorithms::kmeans::plusplus::kmeans_plusplus_into(
        initial.as_view_mut(),
        packed.as_view(),
        &mut StdRng::seed_from_u64(8),
    )?;
    let config = EmeansConfig {
        epochs: 3,
        batch_size: Some(16),
        ..Default::default()
    };
    let cancel: &(dyn crate::cancel::Cancelation + Sync) = &DontCancel;
    let mut baseline = initial.clone();
    let expected = emeans_from_centers(
        packed.as_view(),
        baseline.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(13),
        cancel,
    )?;
    let mut native = initial.clone();
    let mut native_rng = StdRng::seed_from_u64(13);
    let native_rng: &mut dyn rand::RngCore = &mut native_rng;
    let native_result =
        emeans_from_centers(strided, native.as_view_mut(), &config, native_rng, cancel)?;
    assert_eq!(native.as_slice(), baseline.as_slice());
    assert_eq!(native_result.assignments, expected.assignments);

    let mut backend = ReferenceAssignment::default();
    let mut custom = initial;
    let mut rng = StdRng::seed_from_u64(13);
    let rng: &mut dyn rand::RngCore = &mut rng;
    let result = emeans_from_centers_with_assignment(
        strided,
        custom.as_view_mut(),
        &config,
        &mut backend as &mut dyn Assignment<Error = std::convert::Infallible>,
        rng,
        cancel,
    )?;
    assert_eq!(result.assignments, expected.assignments);
    assert_eq!(custom.as_slice(), baseline.as_slice());
    assert_eq!(backend.batch_sizes.len(), 15);
    assert_eq!(
        backend
            .batch_sizes
            .iter()
            .filter(|&&size| size == 3)
            .count(),
        3
    );
    assert_ne!(backend.observed_centers[0], backend.observed_centers[1]);
    Ok(())
}

#[test]
fn sampled_custom_backend_matches_default_and_supports_reuse() -> TestResult {
    let mut data_rng = StdRng::seed_from_u64(3);
    let mut backend = ReferenceAssignment::default();
    for (n, d, k, batch_size) in [(79, 6, 7, 11), (35, 3, 4, 16)] {
        let data = rowmajor::Owned::from_fn(n, d, |_| data_rng.random_range(-2.0_f32..2.0));
        let mut native = rowmajor::Owned::from_element(k, d, f32::NAN);
        let mut custom = rowmajor::Owned::from_element(k, d, f32::NAN);
        let config = EmeansConfig {
            epochs: 3,
            batch_size: Some(batch_size),
            ..Default::default()
        };
        let a = emeans(
            data.as_view(),
            native.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(9),
            &DontCancel,
        )?;
        let b = emeans_with_assignment(
            data.as_view(),
            custom.as_view_mut(),
            &config,
            &mut backend,
            &mut StdRng::seed_from_u64(9),
            &DontCancel,
        )?;
        assert_eq!(a.assignments, b.assignments);
        assert_eq!(a.revived_clusters, b.revived_clusters);
        assert_eq!(native.as_slice(), custom.as_slice());
    }
    Ok(())
}

struct FailingAssignment {
    calls: usize,
}

impl Assignment for FailingAssignment {
    type Error = std::io::Error;
    fn assign(
        &mut self,
        _: rowmajor::Ref<'_, f32>,
        _: rowmajor::Ref<'_, f32>,
        _: &mut [u32],
    ) -> Result<(), Self::Error> {
        self.calls += 1;
        Err(std::io::Error::other("assignment unavailable"))
    }
}

#[test]
fn assignment_failure_retains_source() {
    let data = rowmajor::Owned::from_element(4, 3, 1.0);
    let mut centers = rowmajor::Owned::from_element(2, 3, 0.0);
    let mut backend = FailingAssignment { calls: 0 };
    let error = emeans_from_centers_with_assignment(
        data.as_view(),
        centers.as_view_mut(),
        &EmeansConfig::default(),
        &mut backend,
        &mut StdRng::seed_from_u64(0),
        &DontCancel,
    )
    .unwrap_err();
    assert!(matches!(error, EmeansError::Assignment(_)));
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "assignment unavailable"
    );
    assert_eq!(backend.calls, 1);
    assert!(centers.as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn invalid_and_unwritten_assignments_are_errors() {
    struct InvalidAssignment {
        write: bool,
        calls: usize,
    }
    impl Assignment for InvalidAssignment {
        type Error = std::convert::Infallible;
        fn assign(
            &mut self,
            _: rowmajor::Ref<'_, f32>,
            centers: rowmajor::Ref<'_, f32>,
            labels: &mut [u32],
        ) -> Result<(), Self::Error> {
            if self.write {
                labels.fill(centers.nrows() as u32);
            } else if self.calls == 0 {
                labels.fill(0);
            }
            self.calls += 1;
            Ok(())
        }
    }
    let data = rowmajor::Owned::from_element(4, 3, 1.0);
    for write in [false, true] {
        let mut centers = rowmajor::Owned::from_element(2, 3, 0.0);
        let mut backend = InvalidAssignment { write, calls: 0 };
        let error = emeans_from_centers_with_assignment(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig {
                batch_size: Some(1),
                ..Default::default()
            },
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            &DontCancel,
        )
        .unwrap_err();
        let EmeansError::InvalidAssignment {
            row,
            label,
            clusters,
        } = error
        else {
            panic!("expected invalid assignment, got {error}");
        };
        assert!(row < data.nrows());
        assert_eq!(clusters, 2);
        assert_eq!(label, if write { 2 } else { u32::MAX });
        assert_eq!(backend.calls, if write { 1 } else { 2 });
    }
}

#[test]
fn custom_backend_validation_and_cancellation_precede_assignment() {
    let data = rowmajor::Owned::from_element(4, 3, 1.0);
    let mut centers = rowmajor::Owned::from_element(2, 3, 0.0);
    let mut backend = FailingAssignment { calls: 0 };
    let mut empty_centers = rowmajor::Owned::from_element(0, 3, 0.0);
    assert!(matches!(
        emeans_with_assignment(
            data.as_view(),
            empty_centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            &DontCancel
        ),
        Err(EmeansError::InvalidShape)
    ));
    let overflow = EmeansConfig {
        epochs: usize::MAX,
        batch_size: Some(1),
        ..Default::default()
    };
    assert!(matches!(
        emeans_with_assignment(
            data.as_view(),
            centers.as_view_mut(),
            &overflow,
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            &DontCancel
        ),
        Err(EmeansError::SizeOverflow)
    ));
    let config = EmeansConfig {
        epochs: 0,
        ..Default::default()
    };
    assert!(matches!(
        emeans_with_assignment(
            data.as_view(),
            centers.as_view_mut(),
            &config,
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            &DontCancel
        ),
        Err(EmeansError::InvalidConfig(_))
    ));
    let canceled = CancelAfter {
        polls: Cell::new(0),
        allowed: 0,
    };
    let cancel: &dyn crate::cancel::Cancelation = &canceled;
    assert!(matches!(
        emeans_with_assignment(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            cancel
        ),
        Err(EmeansError::Canceled)
    ));
    let invalid = rowmajor::Owned::from_element(4, 3, f32::NAN);
    assert!(matches!(
        emeans_with_assignment(
            invalid.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            &DontCancel
        ),
        Err(EmeansError::NumericalRange)
    ));
    assert_eq!(backend.calls, 0);
    assert!(centers.as_slice().iter().all(|&x| x == 0.0));

    let mut backend = ReferenceAssignment::default();
    let canceled = CancelAfter {
        polls: Cell::new(0),
        allowed: 4,
    };
    let cancel: &dyn Cancelation = &canceled;
    assert!(matches!(
        emeans_from_centers_with_assignment(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut backend,
            &mut StdRng::seed_from_u64(0),
            cancel
        ),
        Err(EmeansError::Canceled)
    ));
    assert_eq!(backend.batch_sizes.len(), 1);
    assert!(centers.as_slice().iter().any(|&x| x != 0.0));
}

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn assert_near(got: f64, expected: f64, tolerance: f64) {
    assert!(
        (got - expected).abs() <= tolerance,
        "expected {expected}, got {got}, tolerance {tolerance}"
    );
}

// Reference uses direct squared differences and the sum/count EMA equations, independent
// of the SIMD assignment kernel and production centroid interpolation formula.
fn scalar_reference(
    data: rowmajor::Ref<'_, f32>,
    mut centers: rowmajor::Owned<f32>,
    epochs: usize,
    batch_size: usize,
    alpha0: f64,
    seed: u64,
) -> (rowmajor::Owned<f32>, Vec<u32>) {
    let (n, k, d) = (data.nrows(), centers.nrows(), data.ncols());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut order: Vec<_> = (0..n).collect();
    let mut mass = vec![2.0_f64.powi(-16); k];
    let mut labels = vec![u32::MAX; n];
    let mut step = 0;
    for _ in 0..epochs {
        order.shuffle(&mut rng);
        for indices in order.chunks(batch_size) {
            let mut counts = vec![0; k];
            let mut sums = vec![0.0; k * d];
            for &i in indices {
                let mut best = (f64::INFINITY, 0);
                for (j, center) in centers.rows().enumerate() {
                    let distance: f64 = data
                        .row(i)
                        .iter()
                        .zip(center)
                        .map(|(&x, &c)| (f64::from(x) - f64::from(c)).powi(2))
                        .sum();
                    if distance < best.0 {
                        best = (distance, j);
                    }
                }
                labels[i] = best.1 as u32;
                counts[best.1] += 1;
                for (coordinate, &value) in data.row(i).iter().enumerate() {
                    sums[best.1 * d + coordinate] += f64::from(value);
                }
            }
            let progress = step as f64 / (epochs * n.div_ceil(batch_size)) as f64;
            let alpha = alpha0 * (0.55 + 0.45 * (std::f64::consts::PI * progress).cos());
            for j in 0..k {
                let old_mass = mass[j];
                mass[j] = (1.0 - alpha) * old_mass + alpha * counts[j] as f64;
                if counts[j] > 0 {
                    for (coordinate, value) in centers.row_mut(j).iter_mut().enumerate() {
                        let old_sum = old_mass * f64::from(*value);
                        let new_sum = (1.0 - alpha) * old_sum + alpha * sums[j * d + coordinate];
                        *value = (new_sum / mass[j]) as f32;
                    }
                }
            }
            step += 1;
        }
    }
    // Finalization is independently reconstructed by original row index.
    for j in 0..k {
        let members: Vec<_> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label as usize == j).then_some(i))
            .collect();
        if !members.is_empty() {
            for coordinate in 0..d {
                let sum: f64 = members
                    .iter()
                    .map(|&i| f64::from(data.row(i)[coordinate]))
                    .sum();
                centers.row_mut(j)[coordinate] = (sum / members.len() as f64) as f32;
            }
        }
    }
    (centers, labels)
}

#[test]
fn matches_scalar_ema_with_tail_batches() -> TestResult {
    let mut data_rng = StdRng::seed_from_u64(0xf53ee);
    for dim in [1, 3, 7, 17, 32] {
        let data = rowmajor::Owned::from_fn(67, dim, |_| data_rng.random_range(-2.0_f32..2.0));
        for batch_size in [1, 7, 16, 24, 67] {
            let mut centers = rowmajor::Owned::from_fn(5, dim, |rc| data.row(rc.row * 9)[rc.col]);
            let config = EmeansConfig {
                epochs: 3,
                batch_size: Some(batch_size),
                step_size: Some(0.35),
                revive: false,
                ..Default::default()
            };
            let (expected, labels) =
                scalar_reference(data.as_view(), centers.clone(), 3, batch_size, 0.35, 29);
            let result = emeans_from_centers(
                data.as_view(),
                centers.as_view_mut(),
                &config,
                &mut StdRng::seed_from_u64(29),
                &DontCancel,
            )?;
            assert_eq!(result.assignments, labels, "d={dim}, batch={batch_size}");
            assert_eq!(result.updates, 3 * 67_usize.div_ceil(batch_size));
            for (&got, &expected) in centers.as_slice().iter().zip(expected.as_slice()) {
                assert_near(f64::from(got), f64::from(expected), 2e-6);
            }
        }
    }
    Ok(())
}

#[test]
fn final_means_use_original_order_including_last_short_batch() -> TestResult {
    let data = rowmajor::Owned::from_fn(35, 3, |rc| {
        let group = (rc.row * 11) % 3;
        (20 * group + rc.row % 7 + rc.col) as f32
    });
    let mut centers = rowmajor::Owned::from_fn(3, 3, |rc| (20 * rc.row) as f32);
    let config = EmeansConfig {
        epochs: 4,
        batch_size: Some(16),
        ..Default::default()
    };
    let result = emeans_from_centers(
        data.as_view(),
        centers.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(72),
        &DontCancel,
    )?;
    assert_eq!(result.assignments.len(), 35);
    assert_eq!(result.updates, 12);
    for j in 0..3 {
        let members: Vec<_> = result
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label as usize == j).then_some(i))
            .collect();
        assert!(!members.is_empty());
        for coordinate in 0..3 {
            let sum: f64 = members
                .iter()
                .map(|&i| f64::from(data.row(i)[coordinate]))
                .sum();
            assert_eq!(
                centers.row(j)[coordinate],
                (sum / members.len() as f64) as f32
            );
        }
    }
    Ok(())
}

#[test]
fn pq_sized_revival_preserves_final_means_with_a_tail_batch() -> TestResult {
    let (n, dim, k) = (521, 8, 256);
    let data = rowmajor::Owned::from_fn(n, dim, |rc| {
        let group = (rc.row * 11) % 16;
        (20 * group + rc.row % 7 + rc.col) as f32
    });
    let mut centers = rowmajor::Owned::from_fn(k, dim, |rc| {
        if rc.row < 16 {
            (20 * rc.row + rc.col) as f32
        } else {
            (10_000 + rc.row + rc.col) as f32
        }
    });
    // A single epoch makes every revival overlap the closing-mean accumulation.
    let config = EmeansConfig {
        epochs: 1,
        ..Default::default()
    };
    let result = emeans_from_centers(
        data.as_view(),
        centers.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(72),
        &DontCancel,
    )?;
    assert_eq!(result.batch_size, 2);
    assert_eq!(n % result.batch_size, 1);
    assert_eq!(result.updates, n.div_ceil(result.batch_size));
    assert!(result.revived_clusters > 0);
    assert_eq!(result.assignments.len(), n);
    assert!(result.assignments.iter().all(|&label| (label as usize) < k));
    assert!(centers.as_slice().iter().all(|value| value.is_finite()));
    for j in 0..k {
        let members: Vec<_> = result
            .assignments
            .iter()
            .enumerate()
            .filter_map(|(i, &label)| (label as usize == j).then_some(i))
            .collect();
        if members.is_empty() {
            continue;
        }
        for coordinate in 0..dim {
            let sum: f64 = members
                .iter()
                .map(|&i| f64::from(data.row(i)[coordinate]))
                .sum();
            let expected = (sum / members.len() as f64) as f32;
            assert_near(
                f64::from(centers.row(j)[coordinate]),
                f64::from(expected),
                0.0,
            );
        }
    }
    Ok(())
}

#[test]
fn full_batch_one_epoch_equals_lloyd_update() -> TestResult {
    let data =
        rowmajor::Owned::from_fn(31, 5, |rc| (20 * (rc.row % 3) + rc.row % 5 + rc.col) as f32);
    let mut centers = rowmajor::Owned::from_fn(3, 5, |rc| (20 * rc.row) as f32);
    let mut baseline = centers.clone();
    let config = EmeansConfig {
        epochs: 1,
        batch_size: Some(31),
        step_size: Some(1.0),
        revive: false,
        ..Default::default()
    };
    let result = emeans_from_centers(
        data.as_view(),
        centers.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(15),
        &DontCancel,
    )?;
    let (labels, _) =
        crate::algorithms::kmeans::lloyds::lloyds(data.as_view(), baseline.as_view_mut(), 1);
    assert_eq!(result.assignments, labels);
    assert_eq!(centers.as_slice(), baseline.as_slice());
    Ok(())
}

#[test]
fn native_initialization_is_reproducible_and_ignores_output_contents() -> TestResult {
    let data = rowmajor::Owned::from_fn(79, 6, |rc| {
        ((rc.row * 17 + rc.col * 31) % 101) as f32 / 10.0
    });
    let mut first = rowmajor::Owned::from_element(7, 6, f32::NAN);
    let mut second = rowmajor::Owned::from_element(7, 6, 123.0);
    let config = EmeansConfig::default();
    let a = emeans(
        data.as_view(),
        first.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(9),
        &DontCancel,
    )?;
    let b = emeans(
        data.as_view(),
        second.as_view_mut(),
        &config,
        &mut StdRng::seed_from_u64(9),
        &DontCancel,
    )?;
    assert_eq!(a.assignments, b.assignments);
    assert_eq!(first.as_slice(), second.as_slice());
    assert!(first.as_slice().iter().all(|x| x.is_finite()));
    Ok(())
}

#[test]
fn single_point_all_centers_and_duplicate_rows() -> TestResult {
    for (n, k, value) in [(1, 1, 2.0), (19, 7, 0.0), (19, 19, -3.0)] {
        let data = rowmajor::Owned::from_element(n, 3, value);
        let mut centers = rowmajor::Owned::from_element(k, 3, 0.0);
        let result = emeans(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut StdRng::seed_from_u64(8),
            &DontCancel,
        )?;
        assert_eq!(result.assignments.len(), n);
        assert!(result.assignments.iter().all(|&j| (j as usize) < k));
        assert!(centers.as_slice().iter().all(|x| x.is_finite()));
        for &label in &result.assignments {
            assert!(centers.row(label as usize).iter().all(|&x| x == value));
        }
    }
    let data = rowmajor::Owned::from_fn(17, 2, |rc| (rc.row * 10 + rc.col) as f32);
    let mut centers = rowmajor::Owned::from_element(17, 2, 0.0);
    let result = emeans(
        data.as_view(),
        centers.as_view_mut(),
        &EmeansConfig::default(),
        &mut StdRng::seed_from_u64(8),
        &DontCancel,
    )?;
    for (row, &label) in data.rows().zip(&result.assignments) {
        assert_eq!(row, centers.row(label as usize));
    }
    Ok(())
}

#[test]
fn rejects_invalid_shapes_and_configs() {
    for (n, d, k, cd) in [
        (0, 3, 1, 3),
        (3, 0, 1, 0),
        (3, 3, 0, 3),
        (3, 3, 4, 3),
        (3, 3, 1, 2),
    ] {
        let data = rowmajor::Owned::from_element(n, d, 0.0);
        let mut centers = rowmajor::Owned::from_element(k, cd, 0.0);
        let result = emeans(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut StdRng::seed_from_u64(0),
            &DontCancel,
        );
        assert!(matches!(result, Err(EmeansError::InvalidShape)));
    }
    let mut invalid = vec![
        EmeansConfig {
            epochs: 0,
            ..Default::default()
        },
        EmeansConfig {
            batch_size: Some(0),
            ..Default::default()
        },
    ];
    for alpha in [0.0, -0.5, 1.01, f64::NAN, f64::INFINITY] {
        invalid.push(EmeansConfig {
            step_size: Some(alpha),
            ..Default::default()
        });
    }
    for config in invalid {
        assert!(matches!(
            config.resolve_parameters(10, 2, 3),
            Err(EmeansError::InvalidConfig(_))
        ));
    }
    let overflow = EmeansConfig {
        epochs: usize::MAX,
        batch_size: Some(1),
        ..Default::default()
    };
    assert!(matches!(
        overflow.resolve_parameters(2, 1, 1),
        Err(EmeansError::SizeOverflow)
    ));
}

#[test]
fn rejects_nonfinite_and_overflowing_norms_before_training() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let data = rowmajor::Owned::from_element(5, 3, value);
        let mut centers = rowmajor::Owned::from_element(2, 3, 1.0);
        let result = emeans(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut StdRng::seed_from_u64(0),
            &DontCancel,
        );
        assert!(matches!(result, Err(EmeansError::NumericalRange)));
        assert!(centers.as_slice().iter().all(|&x| x == 1.0));
        let data = rowmajor::Owned::from_element(5, 3, 1.0);
        centers.row_mut(1)[0] = value;
        let result = emeans_from_centers(
            data.as_view(),
            centers.as_view_mut(),
            &EmeansConfig::default(),
            &mut StdRng::seed_from_u64(0),
            &DontCancel,
        );
        assert!(matches!(result, Err(EmeansError::NumericalRange)));
    }
}

struct CancelAfter {
    polls: Cell<usize>,
    allowed: usize,
}

impl Cancelation for CancelAfter {
    fn should_cancel(&self) -> bool {
        let polls = self.polls.get();
        self.polls.set(polls + 1);
        polls >= self.allowed
    }
}

#[test]
fn cancellation_is_observed_before_and_during_training() {
    let data = rowmajor::Owned::from_fn(17, 3, |rc| (rc.row + rc.col) as f32);
    let config = EmeansConfig {
        batch_size: Some(4),
        ..Default::default()
    };
    for allowed in [0, 4] {
        let mut centers = rowmajor::Owned::from_element(2, 3, 100.0);
        let cancel = CancelAfter {
            polls: Cell::new(0),
            allowed,
        };
        let result = emeans_from_centers(
            data.as_view(),
            centers.as_view_mut(),
            &config,
            &mut StdRng::seed_from_u64(3),
            &cancel,
        );
        assert!(matches!(result, Err(EmeansError::Canceled)));
        if allowed == 0 {
            assert!(centers.as_slice().iter().all(|&x| x == 100.0));
        } else {
            assert!(centers.as_slice().iter().any(|&x| x != 100.0));
        }
    }
}
