/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

use diskann_utils::views::rowmajor::{self, Matrix, MatrixMut};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};

use crate::{
    inputs::graph_ivf::{FileType, GraphIvfBuild},
    utils::datafiles::{self, BinFile},
};

pub(super) fn load_data(
    input: &GraphIvfBuild,
) -> anyhow::Result<(rowmajor::Owned<f32>, Option<rowmajor::Owned<f32>>)> {
    let data = load_vectors(&input.dataset.data, input.dataset.data_type)?;
    let queries = if input.search.is_some() {
        Some(load_vectors(
            input
                .dataset
                .queries
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing queries file"))?,
            input.dataset.data_type,
        )?)
    } else {
        None
    };
    Ok((data, queries))
}

// Exact FP16 promotion happens outside all build/search timers, without renormalization.
pub(super) fn load_vectors(
    path: &std::path::Path,
    data_type: FileType,
) -> anyhow::Result<rowmajor::Owned<f32>> {
    match data_type {
        FileType::Float32 => datafiles::load_dataset(BinFile(path)),
        FileType::Float16 => {
            let source = datafiles::load_dataset::<half::f16>(BinFile(path))?;
            let mut data = rowmajor::Owned::try_from_element(source.nrows(), source.ncols(), 0.0)?;
            for (output, &value) in data.as_mut_slice().iter_mut().zip(source.as_slice()) {
                *output = value.to_f32();
            }
            Ok(data)
        }
    }
}

pub(super) fn validate_vectors(data: rowmajor::Ref<'_, f32>, name: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        data.nrows() > 0 && data.ncols() > 0,
        "{name} must not be empty"
    );
    // Also reject finite coordinates whose squared L2 distances could overflow.
    let limit = (f32::MAX as f64 / (8.0 * data.ncols() as f64)).sqrt();
    anyhow::ensure!(
        data.as_slice()
            .iter()
            .all(|x| x.is_finite() && (f64::from(*x)).abs() <= limit),
        "{name} contains non-finite or excessively large coordinates"
    );
    Ok(())
}

pub(super) fn load_groundtruth(
    input: &GraphIvfBuild,
    data: rowmajor::Ref<'_, f32>,
    queries: rowmajor::Ref<'_, f32>,
    k: usize,
) -> anyhow::Result<rowmajor::Owned<u32>> {
    if let Some(path) = &input.dataset.groundtruth {
        return datafiles::load_groundtruth(BinFile(path), Some(k));
    }
    let mut truth = rowmajor::Owned::try_from_element(queries.nrows(), k, 0)?;
    for (query, out) in queries.rows().zip(truth.rows_mut()) {
        // Independent exhaustive oracle; never consult the constructed index.
        let mut distances: Vec<(u32, f32)> = data
            .rows()
            .enumerate()
            .map(|(id, vector)| (id as u32, SquaredL2::evaluate(query, vector)))
            .collect();
        let compare =
            |a: &(u32, f32), b: &(u32, f32)| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0));
        if k < distances.len() {
            distances.select_nth_unstable_by(k, compare);
            distances.truncate(k);
        }
        distances.sort_unstable_by(compare);
        for (out, (id, _)) in out.iter_mut().zip(distances) {
            *out = id;
        }
    }
    Ok(truth)
}

pub(super) fn validate_groundtruth(
    truth: rowmajor::Ref<'_, u32>,
    queries: usize,
    vectors: usize,
    k: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        truth.nrows() == queries && truth.ncols() >= k,
        "groundtruth shape does not match queries/k"
    );
    for row in truth.rows() {
        let ids = &row[..k];
        anyhow::ensure!(
            ids.iter().all(|&id| (id as usize) < vectors),
            "groundtruth ID is outside dataset"
        );
        anyhow::ensure!(
            ids.iter().copied().collect::<HashSet<_>>().len() == k,
            "groundtruth contains duplicate IDs"
        );
    }
    Ok(())
}
