/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Error propagation through the IVF index API.
//!
//! A failure in any accessor or strategy must surface as an [`ANNError`] out of
//! `knn_search` / `insert` rather than being swallowed or panicking.

use std::num::NonZeroUsize;

use crate::{
    ivf::test::{
        cases::{DIM, METRIC, N_LISTS, grid_index},
        provider::{Context, ScanMode, Strategy},
    },
    neighbor::{BackInserter, Neighbor},
    test::tokio::current_thread_runtime,
};

/// An injected fine-scan failure must escalate out of `knn_search` (sequential path).
#[test]
fn scan_failure_escalates_sequential() {
    let index = grid_index();
    // The center query probes every quadrant; failing on list 0 is guaranteed to be hit.
    let strategy = Strategy::new(DIM, METRIC)
        .with_mode(ScanMode::Sequential)
        .failing_on_list(0);

    let context = Context::new();
    let mut buf = vec![Neighbor::<u32>::default(); 4];
    let result = current_thread_runtime().block_on(index.knn_search(
        NonZeroUsize::new(4).unwrap(),
        N_LISTS,
        &strategy,
        &context,
        [3.5f32, 3.5].as_slice(),
        &mut BackInserter::new(buf.as_mut_slice()),
    ));

    assert!(result.is_err(), "scan failure must propagate");
}

/// The same failure must escalate from the concurrent path, which has to shut down its
/// in-flight tasks before returning the error.
#[test]
fn scan_failure_escalates_concurrent() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC)
        .with_mode(ScanMode::Concurrent)
        .failing_on_list(0);

    let context = Context::new();
    let mut buf = vec![Neighbor::<u32>::default(); 4];
    let result = current_thread_runtime().block_on(index.knn_search(
        NonZeroUsize::new(4).unwrap(),
        N_LISTS,
        &strategy,
        &context,
        [3.5f32, 3.5].as_slice(),
        &mut BackInserter::new(buf.as_mut_slice()),
    ));

    assert!(
        result.is_err(),
        "scan failure must propagate from concurrent path"
    );
}

/// A query whose dimension disagrees with the strategy must fail at accessor construction.
#[test]
fn dimension_mismatch_errors() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC);

    let context = Context::new();
    let mut buf = vec![Neighbor::<u32>::default(); 1];
    let bad_query = vec![1.0f32, 2.0, 3.0]; // DIM is 2.
    let result = current_thread_runtime().block_on(index.knn_search(
        NonZeroUsize::new(1).unwrap(),
        1,
        &strategy,
        &context,
        bad_query.as_slice(),
        &mut BackInserter::new(buf.as_mut_slice()),
    ));

    assert!(result.is_err(), "dimension mismatch must be rejected");
}

/// A dimension mismatch on insert is likewise rejected.
#[test]
fn insert_dimension_mismatch_errors() {
    let index = grid_index();
    let strategy = Strategy::new(DIM, METRIC);
    let new_id = index.provider().len() as u32;
    let bad_vector = vec![1.0f32]; // DIM is 2.

    let result = current_thread_runtime().block_on(index.insert(
        &strategy,
        &Context::new(),
        &new_id,
        bad_vector.as_slice(),
    ));

    assert!(
        result.is_err(),
        "insert dimension mismatch must be rejected"
    );
}
