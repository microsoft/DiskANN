/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::Path;

use anyhow::{bail, ensure, Result};
use diskann_benchmark_runner::utils::datatype::DataType;
use diskann_utils::views::rowmajor::{self, Matrix};
use diskann_vector::{distance::SquaredL2, PureDistanceFunction};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    inputs::ivf::DataFormat,
    utils::datafiles::{self, BinFile},
};

pub(super) type Rows = Vec<Vec<f32>>;

pub(super) fn read_vectors(path: &Path, format: DataFormat, data_type: DataType) -> Result<Rows> {
    match (format, data_type) {
        (DataFormat::Fvecs, DataType::Float32) => read_fvecs(path),
        (DataFormat::Bin, DataType::Float32) => read_matrix::<f32>(path),
        (DataFormat::Bin, DataType::Float16) => read_matrix::<half::f16>(path),
        _ => bail!("unsupported IVF vector format/type"),
    }
}

fn read_matrix<T: bytemuck::Pod + Into<f32>>(path: &Path) -> Result<Rows> {
    let data = datafiles::load_dataset::<T>(BinFile(path))?;
    ensure!(data.nrows() > 0 && data.ncols() > 0, "empty vector matrix");
    ensure!(
        std::fs::metadata(path)?.len() == 8 + std::mem::size_of_val(data.as_slice()) as u64,
        "vector payload does not match the selected element type"
    );
    Ok(data
        .rows()
        .map(|row| row.iter().map(|&value| value.into()).collect())
        .collect())
}

pub(super) fn read_fvecs(path: &Path) -> Result<Rows> {
    let bytes = std::fs::read(path)?;
    let mut rows = Vec::new();
    let mut offset = 0;
    let mut dimension = None;
    while offset < bytes.len() {
        ensure!(bytes.len() - offset >= 4, "truncated fvecs header");
        let header: [u8; 4] = bytes[offset..offset + 4].try_into()?;
        let d = usize::try_from(i32::from_le_bytes(header))?;
        ensure!(
            d > 0 && dimension.is_none_or(|old| old == d),
            "invalid/mixed fvecs dimensions"
        );
        dimension = Some(d);
        offset += 4;
        let count = d
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("dimension overflow"))?;
        ensure!(count <= bytes.len() - offset, "truncated fvecs row");
        rows.push(
            bytes[offset..offset + count]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        );
        offset += count;
    }
    Ok(rows)
}

// Truth may contain only IDs, or IDs followed by a same-sized f32 distance matrix.
pub(super) fn read_truth(
    path: &Path,
    queries: usize,
    base: usize,
    k: usize,
) -> Result<rowmajor::Owned<u32>> {
    let data = datafiles::load_groundtruth(BinFile(path), Some(k))?;
    ensure!(data.nrows() == queries, "ground-truth shape mismatch");
    let payload = std::mem::size_of_val(data.as_slice()) as u64;
    let bytes = std::fs::metadata(path)?.len();
    ensure!(
        bytes == 8 + payload || bytes == 8 + 2 * payload,
        "invalid ground-truth payload"
    );
    let mut selected = Vec::with_capacity(queries * k);
    for row in data.rows() {
        let ids = &row[..k];
        ensure!(
            ids.iter()
                .enumerate()
                .all(|(i, &id)| (id as usize) < base && !ids[..i].contains(&id)),
            "ground truth contains invalid or duplicate IDs"
        );
        selected.extend_from_slice(ids);
    }
    Ok(rowmajor::Owned::try_from_data(
        selected.into_boxed_slice(),
        queries,
        k,
    )?)
}

pub(super) fn flat_top_k(base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    // Independent full sort checks the search API's bounded top-k implementation.
    let mut all: Vec<(f32, u32)> = base
        .iter()
        .enumerate()
        .map(|(id, vector)| (SquaredL2::evaluate(query, vector.as_slice()), id as u32))
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    all.into_iter().take(k).map(|(_, id)| id).collect()
}

pub(super) fn synthetic() -> (Rows, Rows) {
    // Fixed smoke data, independent of the clustering seed.
    let mut rng = StdRng::seed_from_u64(101);
    let centers: Rows = (0..16)
        .map(|_| (0..32).map(|_| rng.random_range(-1.0f32..1.0)).collect())
        .collect();
    let mut sample = |n: usize| -> Rows {
        (0..n)
            .map(|i| {
                centers[i % centers.len()]
                    .iter()
                    .map(|&c| c + rng.random_range(-0.4f32..0.4))
                    .collect()
            })
            .collect()
    };
    (sample(4096), sample(128))
}
